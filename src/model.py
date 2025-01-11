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
        self.total_input_dim = self.config.sequence_length * self.input_dim
        
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()
    
    def _build_encoder(self) -> tf.keras.Sequential:
        """Build encoder part of the model."""
        return tf.keras.Sequential([
            # Input and flatten
            tf.keras.layers.Input(shape=(self.config.sequence_length, self.input_dim)),
            tf.keras.layers.Flatten(),
            
            # Dense layers with batch norm
            *[tf.keras.layers.Dense(
                dim, 
                activation='relu',
                kernel_initializer='he_normal'
            ) for dim in self.config.hidden_dims],
            
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
            tf.keras.layers.Dense(self.total_input_dim),
            tf.keras.layers.Reshape((self.config.sequence_length, self.input_dim))
        ])
    
    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Forward pass through the model."""
        encoded = self.encoder(inputs, training=training)
        decoded = self.decoder(encoded, training=training)
        return decoded
    
    def compile_model(self, learning_rate: float = 1e-3):
        """Compile the model."""
        self.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate),
            loss='mse',
            metrics=['mae']
        )