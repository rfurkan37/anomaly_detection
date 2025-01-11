# src/model.py
"""Autoencoder model for anomaly detection."""

import tensorflow as tf
from typing import List, Dict, Any, Optional
from tensorflow import keras
from keras import Sequential
from .config import ModelConfig

class AnomalyDetector(tf.keras.Model):
    """Autoencoder-based anomaly detector."""
    
    def __init__(self, input_dim: int, config: ModelConfig):
        super().__init__()
        self.config = config
        self.input_dim = input_dim
        
        # Add flatten layer to handle 3D input
        self.flatten = tf.keras.layers.Flatten()
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()
    
    def _build_encoder(self) -> tf.keras.Sequential:
        """Build encoder part of the model."""
        return tf.keras.Sequential([
            # Input layer
            tf.keras.layers.Dense(
                self.config.hidden_dims[0],
                activation='relu',
                kernel_initializer='he_normal'
            ),
            
            # Additional dense layers
            *[tf.keras.layers.Dense(
                dim, 
                activation='relu',
                kernel_initializer='he_normal'
            ) for dim in self.config.hidden_dims[1:]],
            
            # Bottleneck
            tf.keras.layers.Dense(
                self.config.bottleneck_dim,
                activation='relu',
                kernel_initializer='he_normal'
            )
        ])
    
    def _build_decoder(self) -> tf.keras.Sequential:
        """Build decoder part of the model."""
        return tf.keras.Sequential([
            # Dense layers
            *[tf.keras.layers.Dense(
                dim,
                activation='relu', 
                kernel_initializer='he_normal'
            ) for dim in reversed(self.config.hidden_dims)],
            
            # Output layer
            tf.keras.layers.Dense(self.input_dim)
        ])
    
    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Forward pass through the model."""
        # Flatten input first
        x = self.flatten(inputs)
        encoded = self.encoder(x, training=training)
        decoded = self.decoder(encoded, training=training)
        # Reshape output to match input shape
        batch_size = tf.shape(inputs)[0]
        seq_length = tf.shape(inputs)[1]
        feature_dim = self.input_dim // seq_length
        return tf.reshape(decoded, [batch_size, seq_length, feature_dim])
    
    def compile_model(self, learning_rate: float = 1e-3):
        """Compile the model."""
        self.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
    @classmethod
    def load(cls, path: str):
        """Load model from path."""
        return tf.keras.models.load_model(path)