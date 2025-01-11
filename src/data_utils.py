# src/data_utils.py
"""Unified data processing and evaluation utilities."""

import os
import glob
import logging
from typing import List, Tuple, Optional, Dict, Union
import tensorflow as tf
import numpy as np
from scipy import stats
from sklearn import metrics
from tqdm.auto import tqdm
from .features import FeatureExtractor

logger = logging.getLogger(__name__)

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def get_file_paths(directory: str, pattern: str = "*.wav") -> List[str]:
    """Get file paths matching pattern."""
    search_pattern = os.path.join(directory, pattern)
    logger.info(f"Searching for files in: {search_pattern}")
    
    files = sorted(glob.glob(search_pattern))
    if not files:
        raise ValueError(f"No files found in {search_pattern}")
    
    return files

def split_files(files: List[str], validation_split: float, seed: int = 42) -> Tuple[List[str], List[str]]:
    """Split files into training and validation sets."""
    np.random.seed(seed)
    shuffled_files = np.random.permutation(files)
    split_idx = int(len(files) * (1 - validation_split))
    train_files = shuffled_files[:split_idx].tolist()
    val_files = shuffled_files[split_idx:].tolist()
    logger.info(f"Split data into {len(train_files)} training and {len(val_files)} validation files")
    return train_files, val_files

def extract_features_batch(feature_extractor: FeatureExtractor, 
                         file_paths: List[str]) -> np.ndarray:
    """Extract features from a batch of files."""
    features_list = []
    for file_path in tqdm(file_paths, desc="Extracting features"):
        try:
            features = feature_extractor.extract_features(file_path)
            features_list.append(features)
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            continue
    
    if not features_list:
        raise ValueError("No features could be extracted from the files")
    
    return np.vstack(features_list)

def create_dataset(features: Union[np.ndarray, List[str]], 
                  batch_size: int,
                  feature_extractor: Optional[FeatureExtractor] = None,
                  shuffle: bool = True,
                  cache: bool = True) -> tf.data.Dataset:
    """Create TensorFlow dataset from features or file paths.
    
    Args:
        features: Either preprocessed features array or list of file paths
        batch_size: Batch size for dataset
        feature_extractor: Required if features is a list of file paths
        shuffle: Whether to shuffle the dataset
        cache: Whether to cache the dataset
    """
    if isinstance(features, list):
        if feature_extractor is None:
            raise ValueError("feature_extractor required when providing file paths")
        
        def load_and_preprocess(file_path: str) -> Tuple[tf.Tensor, tf.Tensor]:
            features = feature_extractor.extract_features(file_path.numpy().decode())
            features = tf.cast(features, tf.float32)
            return features, features
        
        # Create dataset from file paths
        dataset = tf.data.Dataset.from_tensor_slices(features)
        
        # Add preprocessing
        dataset = dataset.map(
            lambda x: tf.py_function(
                load_and_preprocess,
                [x],
                [tf.float32, tf.float32]
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Set shapes
        feature_dim = feature_extractor.get_feature_dim()[-1]
        dataset = dataset.map(
            lambda x, y: (
                tf.ensure_shape(x, [None, feature_dim]),
                tf.ensure_shape(y, [None, feature_dim])
            )
        )
        
    else:
        # Create dataset from preprocessed features
        dataset = tf.data.Dataset.from_tensor_slices((features, features))
    
    # Apply common transformations
    if cache:
        dataset = dataset.cache()
    
    if shuffle:
        buffer_size = 10000 if isinstance(features, np.ndarray) else min(len(features), 1000)
        dataset = dataset.shuffle(buffer_size=buffer_size)
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

def calculate_anomaly_scores(model: tf.keras.Model,
                           features: np.ndarray,
                           batch_size: int = 32) -> np.ndarray:
    """Calculate reconstruction error-based anomaly scores."""
    reconstructed = model.predict(features, batch_size=batch_size, verbose=1)
    return np.mean(np.square(features - reconstructed), axis=1)

def fit_threshold(scores: np.ndarray,
                 percentile: float = 0.95) -> Tuple[float, Tuple[float, float, float]]:
    """Fit score distribution and calculate threshold."""
    # Fit gamma distribution
    shape, loc, scale = stats.gamma.fit(scores)
    
    # Calculate threshold
    threshold = stats.gamma.ppf(percentile, shape, loc=loc, scale=scale)
    
    return threshold, (shape, loc, scale)

def evaluate_model(predictions: np.ndarray,
                  scores: np.ndarray,
                  labels: Optional[np.ndarray] = None) -> Dict:
    """Evaluate model performance."""
    results = {
        'anomaly_scores': scores.tolist(),
        'predictions': predictions.tolist()
    }
    
    if labels is not None:
        results.update({
            'accuracy': metrics.accuracy_score(labels, predictions),
            'precision': metrics.precision_score(labels, predictions),
            'recall': metrics.recall_score(labels, predictions),
            'f1': metrics.f1_score(labels, predictions),
            'auc': metrics.roc_auc_score(labels, scores)
        })
    
    return results

def save_results(results: dict, file_path: str):
    """Save evaluation results to file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'w') as f:
        for metric, value in results.items():
            f.write(f"{metric}: {value}\n")