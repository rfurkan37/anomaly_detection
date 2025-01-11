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
    """Audio feature extraction pipeline."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        
    def extract_features(self, audio_path: Union[str, tf.Tensor]) -> np.ndarray:
        """Extract mel-spectrogram features from audio file.
        
        Args:
            audio_path: Path to audio file, can be string or TensorFlow tensor
            
        Returns:
            np.ndarray: Extracted features
        """
        try:
            # Convert tensor to string if needed
            if isinstance(audio_path, tf.Tensor):
                audio_path = audio_path.numpy().decode('utf-8')
            elif isinstance(audio_path, bytes):
                audio_path = audio_path.decode('utf-8')
            
            # Load audio
            y, sr = librosa.load(audio_path, sr=None, mono=True)
            
            # Generate mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=y,
                sr=sr,
                n_fft=self.config.n_fft,
                hop_length=self.config.hop_length,
                n_mels=self.config.n_mels,
                power=self.config.power
            )
            
            # Convert to log scale
            log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Get frame features
            features = self._frame_features(log_mel_spec)
            
            # Flatten the features to 2D
            features = features.reshape(-1, features.shape[-1])
            
            return features
            
        except Exception as e:
            logger.error(f"Error processing {audio_path}: {str(e)}")
            raise
    
    def _frame_features(self, features: np.ndarray) -> np.ndarray:
        """Create overlapping frames from features."""
        n_samples = features.shape[1]
        n_dims = features.shape[0] * self.config.n_frames
        n_vectors = n_samples - self.config.n_frames + 1
        
        vectors = np.zeros((n_vectors, n_dims))
        for t in range(self.config.n_frames):
            vectors[:, features.shape[0] * t : features.shape[0] * (t + 1)] = \
                features[:, t : t + n_vectors].T
        
        return vectors

    def _pad_or_truncate(self, features: np.ndarray) -> np.ndarray:
        """Pad or truncate features to fixed length."""
        if features.shape[0] > self.config.sequence_length:
            # Truncate to fixed length
            return features[:self.config.sequence_length, :]
        elif features.shape[0] < self.config.sequence_length:
            # Pad with zeros to fixed length
            padding = np.zeros((self.config.sequence_length - features.shape[0], features.shape[1]))
            return np.vstack([features, padding])
        else:
            return features

    def get_feature_dim(self) -> Tuple[int, int]:
        """Get the feature dimensions (sequence_length, feature_dim)."""
        return self.config.sequence_length, self.config.n_mels * self.config.n_frames