# src/utils.py
"""Utility functions for the anomaly detection system."""

import os
import glob
import logging
from typing import List, Tuple, Optional
import tensorflow as tf
import numpy as np
from scipy import stats
from sklearn import metrics
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def get_file_paths(directory: str, pattern: str = "*.wav") -> List[str]:
    """Get file paths for training or testing.
    
    Args:
        directory: Base directory to search in
        pattern: File pattern to match (default: "*.wav")
    
    Returns:
        List of file paths
    """
    search_pattern = os.path.join(directory, pattern)
    logger.info(f"Searching for files in: {search_pattern}")
    
    files = sorted(glob.glob(search_pattern))
    
    if not files:
        raise ValueError(f"No files found in {search_pattern}")
    
    return files

def create_dataset(file_paths: List[str], feature_extractor, batch_size: int) -> tf.data.Dataset:
    """Create an optimized TensorFlow dataset."""
    
    def load_and_preprocess(file_path: str) -> Tuple[tf.Tensor, tf.Tensor]:
        """Optimized load and preprocess function."""
        try:
            # Extract features
            features = feature_extractor.extract_features(file_path.numpy().decode())
            # Convert to tensor with explicit type
            features = tf.cast(features, tf.float16)  # Use float16 for mixed precision
            return features, features
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            raise
    
    # Create dataset from file paths
    dataset = tf.data.Dataset.from_tensor_slices(file_paths)
    
    # Optimize loading and preprocessing
    dataset = dataset.map(
        lambda x: tf.py_function(
            load_and_preprocess,
            [x],
            [tf.float16, tf.float16]
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Set shapes explicitly
    seq_length, feature_dim = feature_extractor.get_feature_dim()
    dataset = dataset.map(
        lambda x, y: (
            tf.ensure_shape(x, [seq_length, feature_dim]),
            tf.ensure_shape(y, [seq_length, feature_dim])
        )
    )
    
    # Optimize dataset performance
    dataset = dataset.cache()  # Cache the processed data
    dataset = dataset.shuffle(buffer_size=min(len(file_paths), 1000))
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

def calculate_anomaly_scores(model: tf.keras.Model, 
                           features: np.ndarray) -> np.ndarray:
    """Calculate anomaly scores for features."""
    reconstructed = model.predict(features)
    return np.mean(np.square(features - reconstructed), axis=1)

def fit_score_distribution(scores: np.ndarray) -> Tuple[float, float, float]:
    """Fit a gamma distribution to the scores."""
    return stats.gamma.fit(scores)

def calculate_threshold(score_distribution: Tuple[float, float, float], 
                    percentile: float = 0.95) -> float:
    """Calculate anomaly threshold based on score distribution.
    
    Args:
        score_distribution: Tuple of (alpha, loc, beta) parameters from gamma distribution fit
        percentile: The percentile to use for threshold (default: 0.95)
        
    Returns:
        float: The calculated threshold value
    """
    alpha, loc, beta = score_distribution
    return stats.gamma.ppf(percentile, alpha, loc=loc, scale=beta)

def evaluate_predictions(y_true: np.ndarray,
                       y_score: np.ndarray,
                       threshold: float) -> dict:
    """Calculate various evaluation metrics."""
    y_pred = y_score > threshold
    
    return {
        'auc': metrics.roc_auc_score(y_true, y_score),
        'accuracy': metrics.accuracy_score(y_true, y_pred),
        'precision': metrics.precision_score(y_true, y_pred),
        'recall': metrics.recall_score(y_true, y_pred),
        'f1': metrics.f1_score(y_true, y_pred)
    }

def save_results(results: dict, file_path: str):
    """Save evaluation results to file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'w') as f:
        for metric, value in results.items():
            f.write(f"{metric}: {value}\n")