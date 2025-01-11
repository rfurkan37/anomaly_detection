# src/optimize.py
"""Model optimization module for deployment."""

import os
import logging
from typing import List, Optional
import tensorflow as tf
import numpy as np

from .config import ModelConfig, TrainingConfig
from .features import FeatureExtractor
from .model import AnomalyDetector

logger = logging.getLogger(__name__)

class ModelOptimizer:
    """Model optimization for deployment."""
    
    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self.feature_extractor = FeatureExtractor(model_config)
        
    def create_representative_dataset(self, files: List[str]):
        """Create representative dataset for quantization."""
        def representative_dataset():
            for file_path in files:
                features = self.feature_extractor.extract_features(file_path)
                for i in range(0, min(100, len(features))):
                    sample = features[i:i+1]
                    yield [sample.astype(np.float32)]
        return representative_dataset
    
    def optimize_model(self, model_path: str, representative_files: List[str]) -> bytes:
        """Optimize model for deployment."""
        logger.info(f"Loading model from {model_path}")
        model = tf.keras.models.load_model(model_path)
        
        logger.info("Creating TFLite converter...")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Enable optimizations
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = self.create_representative_dataset(representative_files)
        
        # Force INT8 quantization
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        
        logger.info("Converting model...")
        return converter.convert()
    
    def save_optimized_model(self, tflite_model: bytes, output_path: str):
        """Save the optimized TFLite model."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        logger.info(f"Optimized model saved to: {output_path}")
        logger.info(f"Model size: {len(tflite_model) / 1024:.2f} KB")
    
    def verify_optimized_model(self, tflite_path: str, test_file: str):
        """Verify the optimized model with a test file."""
        logger.info("Verifying optimized model...")
        
        # Load and prepare test data
        test_features = self.feature_extractor.extract_features(test_file)
        test_features = test_features.astype(np.float32)
        
        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Verify model
        interpreter.set_tensor(input_details[0]['index'], test_features)
        interpreter.invoke()
        tflite_output = interpreter.get_tensor(output_details[0]['index'])
        
        logger.info("Model verification complete")
        return tflite_output

def optimize_models(model_dir: str, 
                   output_dir: str,
                   representative_files: List[str],
                   model_config: ModelConfig):
    """Optimize all models in a directory."""
    optimizer = ModelOptimizer(model_config)
    
    # Find all model directories
    model_dirs = [d for d in os.listdir(model_dir) 
                 if os.path.isdir(os.path.join(model_dir, d))]
    
    for model_name in model_dirs:
        try:
            logger.info(f"Optimizing model: {model_name}")
            
            # Paths
            model_path = os.path.join(model_dir, model_name, 'model')
            output_path = os.path.join(output_dir, f"{model_name}_quantized.tflite")
            
            # Optimize model
            tflite_model = optimizer.optimize_model(model_path, representative_files)
            
            # Save optimized model
            optimizer.save_optimized_model(tflite_model, output_path)
            
            # Verify if test file provided
            if representative_files:
                optimizer.verify_optimized_model(output_path, representative_files[0])
                
        except Exception as e:
            logger.error(f"Error optimizing {model_name}: {str(e)}")
            continue
            
    logger.info("Optimization complete")