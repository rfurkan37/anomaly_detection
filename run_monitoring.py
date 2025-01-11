#!/usr/bin/env python3
"""Real-time audio monitoring application."""

import os
import argparse
import logging
import queue
import threading
import time
from typing import Optional, Tuple

import numpy as np
import sounddevice as sd
import librosa
import tensorflow as tf
from scipy import stats
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from src.config import ModelConfig
from src.features import FeatureExtractor

class AudioMonitor:
    """Real-time audio monitoring system."""
    
    def __init__(self, 
                 model_path: str,
                 model_config: ModelConfig,
                 sample_rate: int = 22050,
                 block_duration: float = 1.0,
                 window_duration: float = 5.0):
        
        self.model_config = model_config
        self.sample_rate = sample_rate
        self.block_duration = block_duration
        self.window_duration = window_duration
        
        # Calculate buffer sizes
        self.block_size = int(self.sample_rate * self.block_duration)
        self.window_size = int(self.sample_rate * self.window_duration)
        
        # Initialize components
        self.feature_extractor = FeatureExtractor(model_config)
        self.model = tf.keras.models.load_model(model_path)
        
        # Load threshold
        threshold_path = os.path.join(os.path.dirname(model_path), 'threshold.npz')
        threshold_data = np.load(threshold_path)
        self.threshold = threshold_data['threshold']
        
        # Audio buffer
        self.audio_buffer = np.zeros(self.window_size)
        self.buffer_lock = threading.Lock()
        
        # Processing queue
        self.audio_queue = queue.Queue()
        self.scores_queue = queue.Queue()
        
        # State
        self.running = False
        self.stream: Optional[sd.InputStream] = None
        
    def audio_callback(self, indata: np.ndarray, frames: int, 
                      time_info: dict, status: sd.CallbackFlags) -> None:
        """Callback for audio stream."""
        if status:
            logging.warning(f"Audio callback status: {status}")
        
        # Put audio data in queue
        self.audio_queue.put(indata.copy())
    
    def process_audio(self) -> None:
        """Process audio data from queue."""
        while self.running:
            try:
                # Get audio data from queue
                audio_data = self.audio_queue.get(timeout=1.0)
                
                # Update buffer
                with self.buffer_lock:
                    self.audio_buffer = np.roll(self.audio_buffer, -len(audio_data))
                    self.audio_buffer[-len(audio_data):] = audio_data.flatten()
                
                # Extract features
                features = self._extract_features(self.audio_buffer)
                
                if features is not None:
                    # Get prediction
                    reconstructed = self.model.predict(features, verbose=0)
                    score = np.mean(np.square(features - reconstructed))
                    
                    # Put score in queue
                    self.scores_queue.put(score)
            
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Error processing audio: {e}")
    
    def _extract_features(self, audio: np.ndarray) -> Optional[np.ndarray]:
        """Extract features from audio data."""
        try:
            # Generate mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=self.sample_rate,
                n_mels=self.model_config.n_mels,
                n_fft=self.model_config.n_fft,
                hop_length=self.model_config.hop_length,
                power=self.model_config.power
            )
            
            # Convert to log scale
            log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Frame features
            features = self.feature_extractor._frame_features(log_mel_spec)
            
            return features
            
        except Exception as e:
            logging.error(f"Error extracting features: {e}")
            return None
    
    def start(self) -> None:
        """Start monitoring."""
        self.running = True
        
        # Start audio stream
        self.stream = sd.InputStream(
            channels=1,
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            callback=self.audio_callback
        )
        self.stream.start()
        
        # Start processing thread
        self.process_thread = threading.Thread(target=self.process_audio)
        self.process_thread.start()
    
    def stop(self) -> None:
        """Stop monitoring."""
        self.running = False
        
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
        
        if self.process_thread is not None:
            self.process_thread.join()

class MonitoringGUI:
    """GUI for audio monitoring."""
    
    def __init__(self, monitor: AudioMonitor):
        self.monitor = monitor
        
        # Create main window
        self.root = tk.Tk()
        self.root.title("Audio Anomaly Monitor")
        self.root.geometry("800x600")
        
        # Create graph
        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initialize plot
        self.scores = []
        self.line, = self.ax.plot(self.scores)
        self.threshold_line = self.ax.axhline(
            y=self.monitor.threshold,
            color='r',
            linestyle='--'
        )
        self.ax.set_ylim(0, self.monitor.threshold * 2)
        self.ax.set_title("Anomaly Scores")
        
        # Create status label
        self.status_var = tk.StringVar(value="Status: Normal")
        self.status_label = ttk.Label(
            self.root,
            textvariable=self.status_var,
            font=("Arial", 16)
        )
        self.status_label.pack(pady=10)
        
        # Create controls
        self.control_frame = ttk.Frame(self.root)
        self.control_frame.pack(pady=10)
        
        self.start_button = ttk.Button(
            self.control_frame,
            text="Start",
            command=self.start_monitoring
        )
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(
            self.control_frame,
            text="Stop",
            command=self.stop_monitoring,
            state=tk.DISABLED
        )
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        # Update timer
        self.update_interval = 100  # ms
        self.update_id = None
        
    def update_plot(self) -> None:
        """Update plot with new scores."""
        try:
            # Get new scores
            while not self.monitor.scores_queue.empty():
                score = self.monitor.scores_queue.get_nowait()
                self.scores.append(score)
                
                # Keep only recent scores
                if len(self.scores) > 100:
                    self.scores.pop(0)
                
                # Update status
                if score > self.monitor.threshold:
                    self.status_var.set("Status: ANOMALY DETECTED!")
                    self.status_label.configure(foreground="red")
                else:
                    self.status_var.set("Status: Normal")
                    self.status_label.configure(foreground="green")
            
            # Update plot
            self.line.set_data(range(len(self.scores)), self.scores)
            self.ax.relim()
            self.ax.autoscale_view()
            self.canvas.draw()
            
        except Exception as e:
            logging.error(f"Error updating plot: {e}")
        
        # Schedule next update
        self.update_id = self.root.after(self.update_interval, self.update_plot)
    
    def start_monitoring(self) -> None:
        """Start monitoring."""
        self.monitor.start()
        self.start_button.configure(state=tk.DISABLED)
        self.stop_button.configure(state=tk.NORMAL)
        self.update_plot()
    
    def stop_monitoring(self) -> None:
        """Stop monitoring."""
        self.monitor.stop()
        self.start_button.configure(state=tk.NORMAL)
        self.stop_button.configure(state=tk.DISABLED)
        if self.update_id is not None:
            self.root.after_cancel(self.update_id)
    
    def run(self) -> None:
        """Run the GUI."""
        self.root.mainloop()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Real-time audio anomaly monitoring'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model'
    )
    parser.add_argument(
        '--sample-rate',
        type=int,
        default=22050,
        help='Audio sample rate'
    )
    return parser.parse_args()

def main():
    """Main execution function."""
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Parse arguments
    args = parse_args()
    
    try:
        # Create monitor
        monitor = AudioMonitor(
            model_path=args.model,
            model_config=ModelConfig(),
            sample_rate=args.sample_rate
        )
        
        # Create and run GUI
        gui = MonitoringGUI(monitor)
        gui.run()
        
    except Exception as e:
        logging.error(f"Error in monitoring application: {e}")
        raise

if __name__ == "__main__":
    main()