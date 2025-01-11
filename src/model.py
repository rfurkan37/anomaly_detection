# src/model.py
"""Autoencoder model for anomaly detection."""

import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from .config import ModelConfig

class AnomalyDetector(tf.keras.Model):
    """Autoencoder-based anomaly detector matching DCASE baseline."""
    
    def __init__(self, input_dim: int, config: ModelConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.input_dim = input_dim
        
        # Remove Conv2D layer - DCASE doesn't use it
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()
    
    def _build_encoder(self) -> tf.keras.Sequential:
        """Build encoder exactly matching DCASE baseline."""
        return tf.keras.Sequential([
            # First dense block
            tf.keras.layers.Dense(128, kernel_initializer='he_normal'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            
            # Second dense block
            tf.keras.layers.Dense(128, kernel_initializer='he_normal'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            
            # Third dense block
            tf.keras.layers.Dense(128, kernel_initializer='he_normal'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            
            # Fourth dense block
            tf.keras.layers.Dense(128, kernel_initializer='he_normal'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            
            # Bottleneck
            tf.keras.layers.Dense(8, kernel_initializer='he_normal'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu')
        ])
    
    def _build_decoder(self) -> tf.keras.Sequential:
        """Build decoder exactly matching DCASE baseline."""
        return tf.keras.Sequential([
            # First dense block
            tf.keras.layers.Dense(128, kernel_initializer='he_normal'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            
            # Second dense block
            tf.keras.layers.Dense(128, kernel_initializer='he_normal'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            
            # Third dense block
            tf.keras.layers.Dense(128, kernel_initializer='he_normal'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            
            # Fourth dense block
            tf.keras.layers.Dense(128, kernel_initializer='he_normal'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            
            # Output layer
            tf.keras.layers.Dense(self.input_dim)
        ])
    
    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Forward pass without spatial structure preservation."""
        encoded = self.encoder(inputs, training=training)
        decoded = self.decoder(encoded, training=training)
        return decoded
    
    def compile_model(self, learning_rate: float = 1e-3):
        """Compile model with MSE loss matching DCASE."""
        self.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate),
            loss='mse'  # DCASE uses mean_squared_error
        )
        
    def get_config(self):
        """Provide configuration for model serialization."""
        config = super().get_config()
        config.update({
            'input_dim': self.input_dim,
            'config': self.config
        })
        return config
    
    @classmethod
    def from_config(cls, config, custom_objects=None):
        """Reconstruct the model from its configuration."""
        # Remove any serialization-specific keys
        config.pop('trainable', None)
        config.pop('dtype', None)
        
        # Recreate the model
        input_dim = config.pop('input_dim')
        model_config = config.pop('config')
        return cls(input_dim, model_config, **config)
    
    def save(self, filepath, **kwargs):
        """Override save method to ensure custom objects are handled."""
        custom_objects = {
            'AnomalyDetector': AnomalyDetector,
            'ModelConfig': type(self.config)
        }
        super().save(filepath, custom_objects=custom_objects, **kwargs)