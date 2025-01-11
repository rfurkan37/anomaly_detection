# src/features.py
"""Feature extraction module for audio processing."""

import logging
import numpy as np
import librosa
import tensorflow as tf
from typing import Tuple, Union
from .config import ModelConfig

logger = logging.getLogger(__name__)

class FeatureExtractor:
    """Audio feature extraction pipeline matching DCASE baseline."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        
    def extract_features(self, audio_path: Union[str, tf.Tensor]) -> np.ndarray:
        """Extract mel-spectrogram features exactly as DCASE baseline.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            np.ndarray: Features with shape (n_vectors, feature_dim)
        """
        try:
            # Convert tensor to string if needed
            if isinstance(audio_path, tf.Tensor):
                audio_path = audio_path.numpy().decode('utf-8')
            elif isinstance(audio_path, bytes):
                audio_path = audio_path.decode('utf-8')
            
            # Load audio
            y, sr = librosa.load(audio_path, sr=None, mono=True)
            
            # Generate mel spectrogram exactly as DCASE
            mel_spec = librosa.feature.melspectrogram(
                y=y,
                sr=sr,
                n_fft=self.config.n_fft,
                hop_length=self.config.hop_length,
                n_mels=self.config.n_mels,
                power=self.config.power
            )
            
            # Convert to log scale with same epsilon handling as DCASE
            log_mel_spec = 20.0 / self.config.power * np.log10(
                np.maximum(mel_spec, np.finfo(float).eps)
            )
            
            # Frame features exactly as DCASE
            features = self._frame_features(log_mel_spec)
            
            return features
            
        except Exception as e:
            logger.error(f"Error processing {audio_path}: {str(e)}")
            raise
    
    def _frame_features(self, features: np.ndarray) -> np.ndarray:
        """Create frames exactly as DCASE baseline."""
        n_frames = self.config.n_frames
        n_mels = self.config.n_mels
        
        # Calculate dimensions exactly as DCASE
        n_samples = features.shape[1]
        n_vectors = n_samples - n_frames + 1
        
        if n_vectors < 1:
            return np.empty((0, n_mels * n_frames))
        
        # Create vectors exactly as DCASE
        vectors = np.zeros((n_vectors, n_mels * n_frames))
        for t in range(n_frames):
            vectors[:, n_mels * t : n_mels * (t + 1)] = \
                features[:, t : t + n_vectors].T
                
        # Apply hop frames - this is crucial
        vectors = vectors[::self.config.n_hop_frames, :]
        
        return vectors

    def get_feature_dim(self) -> Tuple[int, int]:
        """Get feature dimensions."""
        return self.config.sequence_length, self.config.n_mels * self.config.n_frames