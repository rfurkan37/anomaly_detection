# src/train.py
"""Training module for the anomaly detection system."""

import os
import logging
from typing import List, Optional, Dict, Tuple
import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision
from tqdm import tqdm
import scipy.stats  # Add this import
from scipy import stats  # This is also needed
from sklearn import metrics  # Add this as well since it's used in evaluate()

from .config import ModelConfig, TrainingConfig
from .features import FeatureExtractor
from .model import AnomalyDetector

logger = logging.getLogger(__name__)

class Trainer:
    """Trainer class for anomaly detection system."""
    
    def __init__(self, model_config: ModelConfig, training_config: TrainingConfig):
        self.model_config = model_config
        self.training_config = training_config
        self.feature_extractor = FeatureExtractor(model_config)
        self.model = None
        self.threshold = None
        self.score_distribution = None
    
    def _split_files(self, files: List[str], validation_split: float) -> Tuple[List[str], List[str]]:
        """Split files into training and validation sets."""
        np.random.seed(42)  # For reproducibility
        shuffled_files = np.random.permutation(files)
        split_idx = int(len(files) * (1 - validation_split))
        train_files = shuffled_files[:split_idx].tolist()
        val_files = shuffled_files[split_idx:].tolist()
        logger.info(f"Split data into {len(train_files)} training and {len(val_files)} validation files")
        return train_files, val_files

    def _extract_features_batch(self, file_paths: List[str]) -> np.ndarray:
        """Extract features from a batch of files."""
        features_list = []
        for file_path in tqdm(file_paths, desc="Extracting features"):
            try:
                features = self.feature_extractor.extract_features(file_path)
                features_list.append(features)
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {str(e)}")
                continue
        
        if not features_list:
            raise ValueError("No features could be extracted from the files")
        
        return np.vstack(features_list)

    def _create_tf_dataset(self, features: np.ndarray, batch_size: int, shuffle: bool = True) -> tf.data.Dataset:
        """Create a TensorFlow dataset from features."""
        dataset = tf.data.Dataset.from_tensor_slices((features, features))
        if shuffle:
            dataset = dataset.shuffle(buffer_size=10000)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    def train(self, train_files: List[str]) -> Dict:
        """Train the anomaly detection model."""
        logger.info("Starting model training...")
        
        # Split files
        train_split, val_split = self._split_files(
            train_files, 
            self.model_config.validation_split
        )
        
        # Extract features
        logger.info("Processing training data...")
        train_features = self._extract_features_batch(train_split)
        logger.info("Processing validation data...")
        val_features = self._extract_features_batch(val_split)
        
        # Create datasets
        train_dataset = self._create_tf_dataset(
            train_features, 
            self.model_config.batch_size, 
            shuffle=True
        )
        val_dataset = self._create_tf_dataset(
            val_features, 
            self.model_config.batch_size, 
            shuffle=False
        )
        
        # Create and compile model
        input_dim = train_features.shape[-1]
        self.model = AnomalyDetector(input_dim, self.model_config)
        self.model.compile_model(self.model_config.lr)
        
        # Setup callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                min_delta=1e-4
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_delta=1e-4,
                cooldown=1
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir='./logs',
                histogram_freq=1,
                update_freq='epoch'
            )
        ]
        
        # Train model
        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=self.model_config.epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        # Calculate threshold
        logger.info("Calculating anomaly threshold...")
        self.calculate_threshold(train_split)
        
        return history.history
    
    def calculate_threshold(self, train_files: List[str]) -> np.ndarray:
        """Calculate anomaly threshold efficiently."""
        logger.info("Calculating anomaly threshold...")
        
        features = self._extract_features_batch(train_files)
        reconstructed = self.model.predict(features, batch_size=32, verbose=1)
        scores = np.mean(np.square(features - reconstructed), axis=1)
        
        # Fit score distribution using gamma distribution
        shape_hat, loc_hat, scale_hat = scipy.stats.gamma.fit(scores)
        self.score_distribution = (shape_hat, loc_hat, scale_hat)
        
        # Calculate threshold as the 95th percentile
        self.threshold = scipy.stats.gamma.ppf(0.95, shape_hat, loc=loc_hat, scale=scale_hat)
        
        return scores
    
    def evaluate(self, test_files: List[str], labels: Optional[List[int]] = None) -> Dict:
        """Evaluate model performance."""
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
        
        # Extract features from test files
        test_features = self._extract_features_batch(test_files)
        
        # Calculate reconstruction error
        reconstructed = self.model.predict(test_features, batch_size=32, verbose=1)
        scores = np.mean(np.square(test_features - reconstructed), axis=1)
        
        # Calculate results
        results = {
            'anomaly_scores': scores.tolist(),
            'predictions': (scores > self.threshold).tolist()
        }
        
        if labels is not None:
            results.update({
                'accuracy': metrics.accuracy_score(labels, results['predictions']),
                'precision': metrics.precision_score(labels, results['predictions']),
                'recall': metrics.recall_score(labels, results['predictions']),
                'f1': metrics.f1_score(labels, results['predictions']),
                'auc': metrics.roc_auc_score(labels, scores)
            })
        
        return results
    
    def save(self, path: str):
        """Save the trained model and configuration."""
        if self.model is None:
            raise ValueError("No trained model to save")
        
        os.makedirs(path, exist_ok=True)
        
        # Save model
        model_path = os.path.join(path, 'model.keras')  # Using .keras extension as recommended
        self.model.save(model_path)
        
        # Save threshold and distribution
        np.savez(
            os.path.join(path, 'threshold.npz'),
            threshold=self.threshold,
            distribution=self.score_distribution
        )

    @classmethod
    def load(cls, path: str, model_config: ModelConfig, training_config: TrainingConfig) -> 'Trainer':
        """Load a saved model and configuration."""
        trainer = cls(model_config, training_config)
        
        # Load model
        model_path = os.path.join(path, 'model.keras')
        trainer.model = AnomalyDetector.load(model_path)
        
        # Load threshold and distribution
        data = np.load(os.path.join(path, 'threshold.npz'))
        trainer.threshold = data['threshold']
        trainer.score_distribution = data['distribution']
        
        return trainer