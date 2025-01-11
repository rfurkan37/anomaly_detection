#!/usr/bin/env python3
"""Main script for training anomaly detection models."""

import os
import argparse
import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass

from src.config import ModelConfig, TrainingConfig
from src.train import Trainer
from src.utils import setup_logging, get_file_paths, save_results

@dataclass
class TrainingPaths:
    """Container for training-related paths."""
    train_dir: str
    test_dir: str
    model_save_dir: str
    results_dir: str

    def create_directories(self):
        """Create necessary directories if they don't exist."""
        os.makedirs(self.model_save_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

class TrainingManager:
    """Manager class for training process."""
    
    def __init__(self, 
                 model_config: ModelConfig,
                 training_config: TrainingConfig,
                 machine_type: str,
                 mode: str):
        self.model_config = model_config
        self.training_config = training_config
        self.machine_type = machine_type
        self.mode = mode
        self.trainer = None
        self.logger = logging.getLogger(__name__)
        
    def setup_paths(self) -> TrainingPaths:
        """Setup necessary paths for training."""
        # Determine base directory based on mode
        base_dir = (self.training_config.dev_directory 
                   if self.mode == 'dev' 
                   else self.training_config.eval_directory)
        
        # Create path structure
        paths = TrainingPaths(
            train_dir=os.path.join(base_dir, self.machine_type, 'train'),
            test_dir=os.path.join(base_dir, self.machine_type, 'test'),
            model_save_dir=os.path.join(self.training_config.model_directory, 
                                      self.machine_type),
            results_dir=self.training_config.result_directory
        )
        
        # Ensure directories exist
        paths.create_directories()
        
        # Log paths for debugging
        self.logger.info(f"Train directory: {paths.train_dir}")
        self.logger.info(f"Test directory: {paths.test_dir}")
        self.logger.info(f"Model save directory: {paths.model_save_dir}")
        self.logger.info(f"Results directory: {paths.results_dir}")
        
        return paths
    
    def get_training_files(self, paths: TrainingPaths) -> Tuple[List[str], List[str]]:
        """Get training and test files."""
        try:
            # Get training files (normal only)
            all_train_files = get_file_paths(paths.train_dir)
            train_files = [f for f in all_train_files if 'normal' in f]
            
            if not train_files:
                raise ValueError(f"No normal training files found in {paths.train_dir}")
            
            # Get test files
            test_files = get_file_paths(paths.test_dir)
            
            self.logger.info(f"Found {len(train_files)} training files")
            self.logger.info(f"Found {len(test_files)} test files")
            
            return train_files, test_files
            
        except Exception as e:
            self.logger.error(f"Error getting files: {str(e)}")
            raise
    
    def get_test_labels(self, test_files: List[str]) -> Optional[List[int]]:
        """Generate labels for test files."""
        if self.mode != 'dev':
            return None
            
        return [1 if 'anomaly' in f else 0 for f in test_files]
    
    def train_model(self, train_files: List[str]) -> None:
        """Train the anomaly detection model."""
        self.logger.info("Initializing trainer...")
        self.trainer = Trainer(self.model_config, self.training_config)
        
        self.logger.info("Starting training...")
        history = self.trainer.train(train_files)
        
        # Log training summary
        final_loss = history['loss'][-1]
        final_val_loss = history.get('val_loss', [0])[-1]
        self.logger.info(f"Training completed - Final loss: {final_loss:.4f}, "
                        f"Val loss: {final_val_loss:.4f}")
    
    def evaluate_model(self, 
                      test_files: List[str], 
                      test_labels: Optional[List[int]]) -> Optional[dict]:
        """Evaluate the trained model."""
        if not test_labels:
            return None
            
        self.logger.info("Evaluating model...")
        results = self.trainer.evaluate(test_files, test_labels)
        
        # Log metrics
        metrics = {k: v for k, v in results.items() 
                  if k not in ['anomaly_scores', 'predictions']}
        for metric, value in metrics.items():
            self.logger.info(f"{metric}: {value:.4f}")
            
        return results
    
    def save_model(self, paths: TrainingPaths) -> None:
        """Save the trained model and results."""
        self.logger.info(f"Saving model to {paths.model_save_dir}")
        self.trainer.save(paths.model_save_dir)
    
    def save_results(self, 
                    results: Optional[dict], 
                    paths: TrainingPaths) -> None:
        """Save evaluation results."""
        if not results:
            return
            
        results_path = os.path.join(
            paths.results_dir,
            f"{self.machine_type}_results.txt"
        )
        save_results(results, results_path)
        self.logger.info(f"Results saved to {results_path}")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train anomaly detection model'
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
        help='Type of machine to process'
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Path to custom configuration file'
    )
    return parser.parse_args()

def main():
    """Main execution function."""

    # Setup
    setup_logging()
    logger = logging.getLogger(__name__)
    args = parse_args()
    
    logger.info(f"Starting training for {args.machine_type} in {args.mode} mode")
    
    # Load configurations
    model_config = ModelConfig()
    training_config = TrainingConfig()
    
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
        train_files, test_files = manager.get_training_files(paths)
        test_labels = manager.get_test_labels(test_files)
        
        # Train model
        manager.train_model(train_files)
        
        # Evaluate if in dev mode
        results = manager.evaluate_model(test_files, test_labels)
        
        # Save model and results
        manager.save_model(paths)
        manager.save_results(results, paths)
        
        logger.info("Training completed successfully")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    main()