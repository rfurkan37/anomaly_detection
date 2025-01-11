#!/usr/bin/env python3
"""Main script for training anomaly detection models with pre-training validation."""

import os
import sys
import argparse
import logging
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass

from src.config import ModelConfig, TrainingConfig
from src.train import Trainer
from src.utils import setup_logging, get_file_paths, save_results

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train anomaly detection model with system validation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--mode',
        choices=['dev', 'eval'],
        required=True,
        help='Development or evaluation mode'
    )
    
    parser.add_argument(
        '--machine-type',
        type=str,
        required=True,
        help='Type of machine to process (e.g., gearbox, valve, etc.)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to custom configuration file (optional)'
    )
    
    parser.add_argument(
        '--skip-checks',
        action='store_true',
        help='Skip system validation checks'
    )
    
    parser.add_argument(
        '--gpu-memory-limit',
        type=float,
        default=None,
        help='Limit GPU memory usage (in GB)'
    )
    
    return parser.parse_args()

class SystemChecker:
    """System and environment checker for model training."""
    
    @staticmethod
    def check_gpu():
        """Check GPU availability and configuration."""
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if not gpus:
                logging.warning("No GPU found. Training will be slow on CPU.")
                return False
            
            # Configure GPU memory growth
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logging.info(f"Enabled memory growth for GPU: {gpu.name}")
            
            return True
            
        except Exception as e:
            logging.error(f"Error checking GPU: {str(e)}")
            return False

    @staticmethod
    def configure_gpu(memory_limit: Optional[float] = None):
        """Configure GPU settings."""
        if memory_limit:
            try:
                gpus = tf.config.list_physical_devices('GPU')
                for gpu in gpus:
                    tf.config.set_logical_device_configuration(
                        gpu,
                        [tf.config.LogicalDeviceConfiguration(
                            memory_limit=memory_limit * 1024  # Convert GB to MB
                        )]
                    )
                logging.info(f"Set GPU memory limit to {memory_limit}GB")
            except Exception as e:
                logging.error(f"Error setting GPU memory limit: {str(e)}")

    @staticmethod
    def check_tensorflow():
        """Verify TensorFlow installation and configuration."""
        logging.info(f"TensorFlow version: {tf.__version__}")
        
        # Check if TF can access GPU
        if tf.test.is_built_with_cuda():
            logging.info("TensorFlow is built with CUDA")
        else:
            logging.warning("TensorFlow is not built with CUDA")

    @staticmethod
    def check_data_files(train_dir: str, test_dir: str) -> Tuple[bool, Optional[str]]:
        """Verify data files exist and are valid."""
        try:
            train_files = get_file_paths(train_dir, pattern="*.wav")
            test_files = get_file_paths(test_dir, pattern="*.wav")
            
            if not train_files:
                return False, "No training files found"
            if not test_files:
                return False, "No test files found"
                
            # Check for balanced data in training set
            normal_count = sum(1 for f in train_files if 'normal' in f.lower())
            anomaly_count = sum(1 for f in train_files if 'anomaly' in f.lower())
            
            logging.info(f"Training data distribution:")
            logging.info(f"  - Normal samples: {normal_count}")
            logging.info(f"  - Anomaly samples: {anomaly_count}")
            
            return True, None
            
        except Exception as e:
            return False, str(e)

def run_system_checks(config: TrainingConfig, model_config: ModelConfig, 
                     machine_type: str, mode: str, gpu_memory_limit: Optional[float] = None) -> bool:
    """Run all system checks before training."""
    checker = SystemChecker()
    
    # Configure GPU first
    checker.configure_gpu(gpu_memory_limit)
    
    # Check system and TensorFlow
    checker.check_tensorflow()
    has_gpu = checker.check_gpu()
    
    if not has_gpu:
        response = input("No GPU found. Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return False
    
    # Check data files
    base_dir = config.dev_directory if mode == 'dev' else config.eval_directory
    train_dir = os.path.join(base_dir, machine_type, 'train')
    test_dir = os.path.join(base_dir, machine_type, 'test')
    
    data_valid, error_msg = checker.check_data_files(train_dir, test_dir)
    if not data_valid:
        logging.error(f"Data validation failed: {error_msg}")
        return False
    
    logging.info("All system checks passed successfully!")
    return True

def main():
    """Main execution function with pre-training validation."""
    setup_logging()
    logger = logging.getLogger(__name__)
    args = parse_args()
    
    logger.info(f"Starting training for {args.machine_type} in {args.mode} mode")
    
    # Load configurations
    model_config = ModelConfig()
    training_config = TrainingConfig()
    
    # Run system checks if not skipped
    if not args.skip_checks:
        if not run_system_checks(training_config, model_config, 
                               args.machine_type, args.mode, 
                               args.gpu_memory_limit):
            logger.error("System checks failed. Aborting training.")
            sys.exit(1)
    
    try:
        # Initialize training manager
        manager = TrainingManager(
            model_config=model_config,
            training_config=training_config,
            machine_type=args.machine_type,
            mode=args.mode
        )
        
        # Setup paths
        paths = manager.setup_paths()
        
        # Get files
        train_files = get_file_paths(paths.train_dir, pattern='*.wav')
        test_files = get_file_paths(paths.test_dir, pattern='*.wav')
        logger.info(f"Found {len(train_files)} training files and {len(test_files)} test files")
        
        # Train model
        manager.train_model(train_files)
        
        logger.info("Training completed successfully")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    main()