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
    """Create a TensorFlow dataset from audio files."""
    def load_and_preprocess(file_path: str) -> Tuple[tf.Tensor, tf.Tensor]:
        # Extract features with fixed length
        features = feature_extractor.extract_features(file_path.numpy().decode())
        # Convert to tensor
        features = tf.convert_to_tensor(features, dtype=tf.float32)
        return features, features  # Input = Target for autoencoder
    
    seq_length, feature_dim = feature_extractor.get_feature_dim()
    
    dataset = tf.data.Dataset.from_tensor_slices(file_paths)
    dataset = dataset.map(
        lambda x: tf.py_function(
            load_and_preprocess, 
            [x], 
            [tf.float32, tf.float32]
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Set shapes explicitly
    dataset = dataset.map(
        lambda x, y: (
            tf.ensure_shape(x, [seq_length, feature_dim]),
            tf.ensure_shape(y, [seq_length, feature_dim])
        )
    )
    
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def calculate_anomaly_scores(model: tf.keras.Model, 
                           features: np.ndarray) -> np.ndarray:
    """Calculate anomaly scores for features."""
    reconstructed = model.predict(features)
    return np.mean(np.square(features - reconstructed), axis=1)

def fit_score_distribution(scores: np.ndarray) -> Tuple[float, float, float]:
    """Fit a gamma distribution to the scores."""
    return stats.gamma.fit(scores)

def calculate_threshold(distribution_params: Tuple[float, float, float],
                       percentile: float = 90) -> float:
    """Calculate threshold based on score distribution."""
    shape, loc, scale = distribution_params
    return stats.gamma.ppf(q=percentile/100, a=shape, loc=loc, scale=scale)

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