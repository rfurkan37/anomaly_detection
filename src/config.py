# src/config.py
"""Configuration for the anomaly detection system."""

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
import json

@dataclass
class ModelConfig:
    """Model configuration parameters."""
    # Feature extraction
    n_mels: int = 128
    n_frames: int = 5
    n_fft: int = 1024
    hop_length: int = 512
    power: float = 2.0
    n_hop_frames: int = 1  # Number of frames to skip for next frame
    
    # Training
    lr: float = 0.001
    epochs: int = 100
    batch_size: int = 512  # Reduced batch size to help with memory
    validation_split: float = 0.1
    shuffle: bool = True
    verbose: int = 1
    
    # Model architecture
    hidden_dims: list = None
    bottleneck_dim: int = 8
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [128, 128, 128, 128]

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert config to JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        """Create config from dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'ModelConfig':
        """Create config from JSON string."""
        config_dict = json.loads(json_str)
        return cls.from_dict(config_dict)

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
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert config to JSON string."""
        return json.dumps(self.to_dict())