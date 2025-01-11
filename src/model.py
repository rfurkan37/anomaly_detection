# src/model.py
"""Autoencoder model for anomaly detection."""

import tensorflow as tf
from typing import List
from tensorflow.keras import Sequential
from .config import ModelConfig

class AnomalyDetector(tf.keras.Model):
    """Autoencoder-based anomaly detector."""
    
    def __init__(self, input_dim: int, config: ModelConfig):
        super(AnomalyDetector, self).__init__()
        
        # Ensure input_dim is flattened
        self.input_dim = input_dim
        self.encoder = self._build_encoder(input_dim, config)
        self.decoder = self._build_decoder(input_dim, config)
    def _build_encoder(self, input_dim: int, config: ModelConfig) -> Sequential:
        """Build encoder part of the model."""
        layers = []
        
        # Input layer with explicit shape
        layers.append(tf.keras.layers.Input(shape=(input_dim,)))
        
        # Hidden layers
        for dim in config.hidden_dims:
            layers.extend([
                tf.keras.layers.Dense(dim),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation('relu')
            ])
        
        # Bottleneck
        layers.extend([
            tf.keras.layers.Dense(config.bottleneck_dim),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu')
        ])
        
        return tf.keras.Sequential(layers)
    def _build_decoder(self, input_dim: int, config: ModelConfig) -> Sequential:
        """Build decoder part of the model."""
        layers = []
        
        # Hidden layers
        for dim in reversed(config.hidden_dims):
            layers.extend([
                tf.keras.layers.Dense(dim),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation('relu')
            ])
        
        # Output layer
        layers.append(tf.keras.layers.Dense(input_dim))
        
        return tf.keras.Sequential(layers)
    
    def call(self, x):
        # Ensure input is properly shaped
        if len(x.shape) > 2:
            x = tf.reshape(x, [-1, self.input_dim])
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def compile_model(self, learning_rate: float):
        """Compile the model with specified settings."""
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    @classmethod
    def load(cls, path: str) -> 'AnomalyDetector':
        """Load a saved model."""
        return tf.keras.models.load_model(path)