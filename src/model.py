# src/model.py
"""Autoencoder model for anomaly detection."""

import tensorflow as tf
from tensorflow import keras
import json
import os
from .config import ModelConfig

class AnomalyDetector(tf.keras.Model):
    """Autoencoder-based anomaly detector matching DCASE baseline."""
    
    def __init__(
        self,
        input_dim: int,
        config: ModelConfig = None,
        name: str = 'anomaly_detector',
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.input_dim = input_dim
        self.config = config if config is not None else ModelConfig()
        
        # Initialize but don't build yet
        self._encoder = None
        self._decoder = None
    
    @property
    def encoder(self):
        """Lazy initialization of encoder."""
        if self._encoder is None:
            self._encoder = self._build_encoder()
        return self._encoder
    
    @property
    def decoder(self):
        """Lazy initialization of decoder."""
        if self._decoder is None:
            self._decoder = self._build_decoder()
        return self._decoder
    
    def _build_encoder(self) -> tf.keras.Sequential:
        """Build encoder exactly matching DCASE baseline."""
        return tf.keras.Sequential([
            # Input layer with explicit shape
            tf.keras.layers.InputLayer(input_shape=(self.input_dim,)),
            
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
            loss='mse'
        )
    
    def get_config(self):
        """Get model configuration."""
        config = super().get_config()
        
        # Add model-specific configuration
        config.update({
            'input_dim': self.input_dim,
            'config': self.config.to_dict() if self.config else None
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        """Create model from configuration."""
        # Make a copy of the config to avoid modifying the original
        config = config.copy()
        
        # Extract the essential arguments
        input_dim = config.pop('input_dim', None)
        model_config_dict = config.pop('config', None)
        
        # Handle missing input_dim
        if input_dim is None:
            if 'build_config' in config and 'input_shape' in config['build_config']:
                input_dim = config['build_config']['input_shape'][-1]
            else:
                raise ValueError("Could not determine input_dim from config")
        
        # Create ModelConfig instance from dict if available
        if model_config_dict is not None:
            model_config = ModelConfig(**model_config_dict)
        else:
            model_config = ModelConfig()
        
        # Remove TensorFlow-specific config items
        for key in ['trainable', 'dtype', 'name', 'build_config', 'compile_config']:
            config.pop(key, None)
        
        # Create new instance
        return cls(
            input_dim=input_dim,
            config=model_config,
            **config
        )
    
    def save(self, filepath: str, save_format='keras', **kwargs):
        """Save model with configuration."""
        # Ensure the directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save the model
        super().save(filepath, save_format=save_format, **kwargs)
        
        # Save configuration separately for easier access
        config_path = os.path.join(os.path.dirname(filepath), 'model_config.json')
        with open(config_path, 'w') as f:
            json.dump({
                'input_dim': self.input_dim,
                'config': self.config.to_dict()
            }, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str):
        """Load model with configuration."""
        try:
            # Load the model
            model = tf.keras.models.load_model(
                filepath,
                custom_objects={'AnomalyDetector': cls}
            )
            return model
            
        except Exception as e:
            raise ValueError(f"Error loading model: {str(e)}")