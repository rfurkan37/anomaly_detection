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
            
            # Check GPU memory
            gpu_memory = tf.config.experimental.get_memory_info('GPU:0')
            free_memory_mb = gpu_memory['free'] / (1024 * 1024)
            if free_memory_mb < 2000:  # Less than 2GB free
                logging.warning(f"Low GPU memory: {free_memory_mb:.2f}MB free")
            
            logging.info(f"Found {len(gpus)} GPU(s)")
            for gpu in gpus:
                logging.info(f"GPU Device: {gpu.device_type} - {gpu.name}")
            return True
            
        except Exception as e:
            logging.error(f"Error checking GPU: {str(e)}")
            return False

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
    def check_directory_structure(config: TrainingConfig, machine_type: str, mode: str) -> bool:
        """Verify required directory structure exists."""
        base_dir = config.dev_directory if mode == 'dev' else config.eval_directory
        required_dirs = [
            Path(base_dir) / machine_type / 'train',
            Path(base_dir) / machine_type / 'test',
            Path(config.model_directory),
            Path(config.result_directory)
        ]
        
        missing_dirs = []
        for dir_path in required_dirs:
            if not dir_path.exists():
                missing_dirs.append(dir_path)
                
        if missing_dirs:
            logging.error("Missing required directories:")
            for dir_path in missing_dirs:
                logging.error(f"  - {dir_path}")
            return False
            
        return True

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
                
            # Check file sizes
            small_files = []
            for file_path in train_files + test_files:
                if os.path.getsize(file_path) < 1024:  # Less than 1KB
                    small_files.append(file_path)
            
            if small_files:
                logging.warning("Found suspiciously small files:")
                for file_path in small_files[:5]:  # Show first 5
                    logging.warning(f"  - {file_path}")
                    
            # Check for balanced data in training set
            normal_count = sum(1 for f in train_files if 'normal' in f.lower())
            anomaly_count = sum(1 for f in train_files if 'anomaly' in f.lower())
            
            logging.info(f"Training data distribution:")
            logging.info(f"  - Normal samples: {normal_count}")
            logging.info(f"  - Anomaly samples: {anomaly_count}")
            
            return True, None
            
        except Exception as e:
            return False, str(e)

    @staticmethod
    def check_model_config(config: ModelConfig) -> bool:
        """Validate model configuration."""
        try:
            # Check basic parameter ranges
            if config.n_mels <= 0 or config.n_frames <= 0:
                logging.error("Invalid feature extraction parameters")
                return False
                
            # Check memory requirements
            approx_memory_mb = (
                config.batch_size * 
                config.n_mels * 
                config.n_frames * 
                4  # 4 bytes per float32
            ) / (1024 * 1024)  # Convert to MB
            
            if approx_memory_mb > 1024:  # More than 1GB per batch
                logging.warning(f"High memory usage per batch: {approx_memory_mb:.2f}MB")
                
            return True
            
        except Exception as e:
            logging.error(f"Error validating config: {str(e)}")
            return False

def run_system_checks(config: TrainingConfig, model_config: ModelConfig, 
                     machine_type: str, mode: str) -> bool:
    """Run all system checks before training."""
    checker = SystemChecker()
    
    # Check system and TensorFlow
    checker.check_tensorflow()
    has_gpu = checker.check_gpu()
    
    if not has_gpu:
        response = input("No GPU found. Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return False
    
    # Check directories and data
    if not checker.check_directory_structure(config, machine_type, mode):
        return False
        
    base_dir = config.dev_directory if mode == 'dev' else config.eval_directory
    train_dir = os.path.join(base_dir, machine_type, 'train')
    test_dir = os.path.join(base_dir, machine_type, 'test')
    
    data_valid, error_msg = checker.check_data_files(train_dir, test_dir)
    if not data_valid:
        logging.error(f"Data validation failed: {error_msg}")
        return False
    
    # Check configuration
    if not checker.check_model_config(model_config):
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
    
    # Run system checks
    if not run_system_checks(training_config, model_config, args.machine_type, args.mode):
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