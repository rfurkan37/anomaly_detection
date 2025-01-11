# src/train.py
"""Training module for the anomaly detection system."""

import os
import logging
from typing import List, Optional, Dict, Tuple
import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision

from .config import ModelConfig, TrainingConfig
from .features import FeatureExtractor
from .model import AnomalyDetector
from .utils import (
    create_dataset,
    calculate_anomaly_scores,
    fit_score_distribution,
    calculate_threshold,
    evaluate_predictions
)

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
        # Shuffle files
        np.random.seed(42)  # For reproducibility
        shuffled_files = np.random.permutation(files)
        
        # Calculate split index
        split_idx = int(len(files) * (1 - validation_split))
        
        # Split files
        train_files = shuffled_files[:split_idx].tolist()
        val_files = shuffled_files[split_idx:].tolist()
        
        logger.info(f"Split data into {len(train_files)} training and {len(val_files)} validation files")
        
        return train_files, val_files
        
    def train(self, train_files: List[str]) -> Dict:
        """Train the anomaly detection model."""
        logger.info("Starting model training...")
        
        # Get input dimension from a sample file
        sample_features = self.feature_extractor.extract_features(train_files[0])
        input_dim = sample_features.shape[-1]
        
        # Create and compile model
        self.model = AnomalyDetector(input_dim, self.model_config)
        self.model.compile_model(self.model_config.lr)
        
        # Split data
        train_split, val_split = self._split_files(
            train_files, 
            self.model_config.validation_split
        )
        
        # Create datasets with optimized settings
        train_dataset = create_dataset(
            train_split,
            self.feature_extractor,
            self.model_config.batch_size
        )
        val_dataset = create_dataset(
            val_split,
            self.feature_extractor,
            self.model_config.batch_size
        )
        
        # Enable mixed precision
        mixed_precision.set_global_policy('mixed_float16')
        
        # Optimize callback settings
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
        
        # Train with updated settings
        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=self.model_config.epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        # Calculate threshold efficiently
        logger.info("Calculating anomaly threshold...")
        self.calculate_threshold(train_files)
        
        return history.history
    
    def evaluate(self, test_files: List[str], labels: Optional[List[int]] = None) -> Dict:
        """Evaluate model performance."""
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
            
        # Extract features and calculate scores
        features = np.vstack([
            self.feature_extractor.extract_features(f) for f in test_files
        ])
        scores = calculate_anomaly_scores(self.model, features)
        
        # Average scores per file
        file_scores = []
        start_idx = 0
        for file_path in test_files:
            file_features = self.feature_extractor.extract_features(file_path)
            end_idx = start_idx + len(file_features)
            file_scores.append(np.mean(scores[start_idx:end_idx]))
            start_idx = end_idx
        
        results = {
            'anomaly_scores': file_scores,
            'predictions': [score > self.threshold for score in file_scores]
        }
        
        # Calculate metrics if labels provided
        if labels is not None:
            results.update(evaluate_predictions(
                np.array(labels),
                np.array(file_scores),
                self.threshold
            ))
            
        return results
    
    def save(self, path: str):
        """Save the trained model and configuration."""
        if self.model is None:
            raise ValueError("No trained model to save")
            
        os.makedirs(path, exist_ok=True)
        
        # Save model
        model_path = os.path.join(path, 'model')
        self.model.save(model_path)
        
        # Save threshold and distribution
        np.savez(
            os.path.join(path, 'threshold.npz'),
            threshold=self.threshold,
            distribution=self.score_distribution
        )
    
    def calculate_threshold(self, train_files):
        """Calculate anomaly threshold efficiently."""
        logger.info("Calculating anomaly threshold...")
        
        # Process files in batches
        batch_size = 32
        all_scores = []
        
        for i in range(0, len(train_files), batch_size):
            batch_files = train_files[i:i + batch_size]
            batch_features = []
            
            # Extract features in parallel using tf.data
            dataset = tf.data.Dataset.from_tensor_slices(batch_files)
            dataset = dataset.map(
                lambda x: tf.py_function(
                    self.feature_extractor.extract_features,
                    [x],
                    tf.float32
                ),
                num_parallel_calls=tf.data.AUTOTUNE
            )
            dataset = dataset.prefetch(tf.data.AUTOTUNE)
            
            # Convert to numpy and calculate scores
            batch_features = list(dataset.as_numpy_iterator())
            batch_features = np.vstack(batch_features)
            
            # Calculate scores for batch
            reconstructed = self.model.predict(
                batch_features, 
                batch_size=32,
                verbose=0
            )
            scores = np.mean(np.square(batch_features - reconstructed), axis=1)
            all_scores.extend(scores)
        
        all_scores = np.array(all_scores)
        self.score_distribution = fit_score_distribution(all_scores)
        self.threshold = calculate_threshold(self.score_distribution)
        
        return all_scores
    
    @classmethod
    def load(cls, path: str, model_config: ModelConfig, training_config: TrainingConfig) -> 'Trainer':
        """Load a saved model and configuration."""
        trainer = cls(model_config, training_config)
        
        # Load model
        model_path = os.path.join(path, 'model')
        trainer.model = AnomalyDetector.load(model_path)
        
        # Load threshold and distribution
        data = np.load(os.path.join(path, 'threshold.npz'))
        trainer.threshold = data['threshold']
        trainer.score_distribution = data['distribution']
        
        return trainer