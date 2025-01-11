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
        name: str = 'anomaly_detector',  # Add name parameter
        trainable: bool = True,          # Add trainable parameter
        dtype = None,                    # Add dtype parameter
        **kwargs
    ):
        super().__init__(name=name, trainable=trainable, dtype=dtype, **kwargs)
        self.input_dim = input_dim
        self.config = config if config is not None else ModelConfig()
        
        # Build encoder and decoder immediately
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()
        
        # Build the model once to initialize all variables
        dummy_input = tf.keras.layers.Input(shape=(input_dim,))
        self.call(dummy_input)
    
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
            loss='mse'
        )
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'input_dim': self.input_dim,
            'config': self.config.to_dict() if self.config else None
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        """Create model from configuration."""
        # Extract base config for parent class
        base_config = {
            'name': config.get('name', 'anomaly_detector'),
            'trainable': config.get('trainable', True),
            'dtype': config.get('dtype', None)
        }
        
        # Extract AnomalyDetector specific config
        input_dim = config['input_dim']
        model_config_dict = config.get('config')
        
        if model_config_dict is not None:
            model_config = ModelConfig(**model_config_dict)
        else:
            model_config = ModelConfig()
            
        # Create new instance
        return cls(
            input_dim=input_dim,
            config=model_config,
            **base_config
        )
    
    def save(self, filepath: str):
        """Save model with configuration."""
        # Ensure the directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save the Keras model
        super().save(filepath)
        
        # Save configuration separately
        config_path = os.path.join(os.path.dirname(filepath), 'model_config.json')
        with open(config_path, 'w') as f:
            json.dump({
                'input_dim': self.input_dim,
                'config': self.config.to_dict()
            }, f, indent=2)
    
    @classmethod
    def load_model(cls, filepath: str, custom_objects=None):
        """Load the model from a saved file."""
        try:
            # Load the Keras model
            model = tf.keras.models.load_model(
                filepath,
                custom_objects={'AnomalyDetector': cls}
            )
            
            # Load the configuration if it exists
            config_path = os.path.join(os.path.dirname(filepath), 'model_config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                model.input_dim = config['input_dim']
                model.config = ModelConfig(**config['config'])
            
            return model
            
        except Exception as e:
            raise ValueError(f"Error loading model: {str(e)}")