# src/config.py
"""Configuration for the anomaly detection system."""

from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    """Model configuration parameters."""
    # Feature extraction
    n_mels: int = 128
    n_frames: int = 5
    n_fft: int = 1024
    hop_length: int = 512
    power: float = 2.0
    sequence_length: int = 300  # Fixed sequence length for all samples
    
    # Training
    lr: float = 0.001
    epochs: int = 100
    batch_size: int = 32  # Reduced batch size to help with memory
    validation_split: float = 0.1
    
    # Model architecture
    hidden_dims: list = None
    bottleneck_dim: int = 8
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [128, 128, 128, 128]

@dataclass
class TrainingConfig:
    """Training configuration parameters."""
    dev_directory: str = "data/dev_data"
    eval_directory: str = "data/eval_data"
    model_directory: str = "models"
    result_directory: str = "results"
    result_file: str = "result.csv"
    max_fpr: float = 0.1
    decision_threshold: float = 0.9