# src/persistence.py
"""Model persistence module for saving and loading models and their configurations."""

import os
import json
import logging
import numpy as np
import tensorflow as tf
from typing import Optional, Tuple, Dict, Any

from .config import ModelConfig
from .model import AnomalyDetector

logger = logging.getLogger(__name__)

class ModelPersistence:
    """Handles saving and loading of models and their associated data."""
    
    MODEL_FILENAME = 'model.keras'
    CONFIG_FILENAME = 'config.json'
    THRESHOLD_FILENAME = 'threshold.npz'
    
    @classmethod
    def save_model(cls, 
                  model: AnomalyDetector,
                  save_path: str,
                  threshold: Optional[float] = None,
                  score_distribution: Optional[Tuple] = None,
                  metadata: Optional[Dict[str, Any]] = None) -> None:
        """Save model with all associated data.
        
        Args:
            model: The AnomalyDetector model to save
            save_path: Directory path to save the model
            threshold: Optional anomaly detection threshold
            score_distribution: Optional tuple of distribution parameters
            metadata: Optional additional metadata to save
        """
        os.makedirs(save_path, exist_ok=True)
        
        # Save the model
        model_path = os.path.join(save_path, cls.MODEL_FILENAME)
        model.save(model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save config
        config_data = {
            'input_dim': model.input_dim,
            'model_config': model.config.to_dict() if model.config else None,
            'metadata': metadata or {}
        }
        
        config_path = os.path.join(save_path, cls.CONFIG_FILENAME)
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        logger.info(f"Configuration saved to {config_path}")
        
        # Save threshold and distribution if provided
        if threshold is not None or score_distribution is not None:
            threshold_path = os.path.join(save_path, cls.THRESHOLD_FILENAME)
            np.savez(
                threshold_path,
                threshold=threshold,
                distribution=score_distribution
            )
            logger.info(f"Threshold data saved to {threshold_path}")
    
    @classmethod
    def load_model(cls, 
                  load_path: str,
                  return_threshold: bool = False) -> Tuple[AnomalyDetector, Optional[Dict]]:
        """Load model and associated data.
        
        Args:
            load_path: Directory path containing the saved model
            return_threshold: Whether to return threshold information
            
        Returns:
            Tuple containing:
                - Loaded AnomalyDetector model
                - Dictionary with threshold, distribution and metadata (if return_threshold=True)
                
        Raises:
            ValueError: If model files are missing or corrupt
        """
        try:
            # Load configuration first
            config_path = os.path.join(load_path, cls.CONFIG_FILENAME)
            if not os.path.exists(config_path):
                raise ValueError(f"Configuration file not found at {config_path}")
                
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            # Load the model
            model_path = os.path.join(load_path, cls.MODEL_FILENAME)
            if not os.path.exists(model_path):
                raise ValueError(f"Model file not found at {model_path}")
                
            model = tf.keras.models.load_model(
                model_path,
                custom_objects={'AnomalyDetector': AnomalyDetector}
            )
            
            # Restore model configuration
            model.input_dim = config_data['input_dim']
            if config_data.get('model_config'):
                model.config = ModelConfig(**config_data['model_config'])
            
            if not return_threshold:
                return model, None
                
            # Load threshold data if requested
            threshold_data = {}
            threshold_path = os.path.join(load_path, cls.THRESHOLD_FILENAME)
            if os.path.exists(threshold_path):
                with np.load(threshold_path) as data:
                    threshold_data.update({
                        'threshold': float(data['threshold']),
                        'distribution': tuple(data['distribution'])
                    })
            
            # Include metadata
            threshold_data['metadata'] = config_data.get('metadata', {})
            
            return model, threshold_data
            
        except Exception as e:
            raise ValueError(f"Error loading model: {str(e)}")