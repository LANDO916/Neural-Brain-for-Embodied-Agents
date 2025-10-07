#!/usr/bin/env python3
"""
Real-Time Audio Processing Module
Provides live audio analysis with streaming capabilities and real-time visualization
"""

import os
import sys
import json
import time
import threading
import queue
import numpy as np
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from collections import deque
import warnings

# Audio processing
import librosa
import soundfile as sf

# Real-time audio capture
try:
    import pyaudio
    HAS_PYAUDIO = True
except ImportError:
    HAS_PYAUDIO = False

# Real-time plotting
try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.collections import LineCollection
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# WebSocket for real-time data streaming
try:
    import websockets
    import asyncio
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False

# Advanced signal processing
try:
    from scipy import signal
    from scipy.fft import fft, fftfreq
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

warnings.filterwarnings('ignore')

class RealTimeAudioBuffer:
    """Circular buffer for real-time audio processing"""
    
    def __init__(self, max_duration: float = 30.0, sample_rate: int = 22050):
        self.max_duration = max_duration
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration * sample_rate)
        
        # Circular buffer
        self.buffer = np.zeros(self.max_samples, dtype=np.float32)
        self.write_pos = 0
        self.samples_written = 0
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Metadata
        self.timestamps = deque(maxlen=self.max_samples)
    
    def write(self, audio_chunk: np.ndarray) -> None:
        """Write audio chunk to buffer"""
        with self.lock:
            chunk_size = len(audio_chunk)
            current_time = time.time()
            
            # Handle buffer wrap-around
            if self.write_pos + chunk_size <= self.max_samples:
                self.buffer[self.write_pos:self.write_pos + chunk_size] = audio_chunk
                # Add timestamps
                for i in range(chunk_size):
                    self.timestamps.append(current_time + i / self.sample_rate)
            else:
                # Split chunk across buffer boundary
                first_part_size = self.max_samples - self.write_pos
                second_part_size = chunk_size - first_part_size
                
                self.buffer[self.write_pos:] = audio_chunk[:first_part_size]
                self.buffer[:second_part_size] = audio_chunk[first_part_size:]
                
                # Add timestamps with wrap-around
                for i in range(first_part_size):
                    self.timestamps.append(current_time + i / self.sample_rate)
                for i in range(second_part_size):
                    self.timestamps.append(current_time + (first_part_size + i) / self.sample_rate)
            
            self.write_pos = (self.write_pos + chunk_size) % self.max_samples
            self.samples_written += chunk_size
    
    def read_latest(self, duration: float) -> tuple[np.ndarray, List[float]]:
        """Read latest audio data from buffer"""
        with self.lock:
            samples_requested = int(duration * self.sample_rate)
            samples_available = min(samples_requested, self.samples_written, self.max_samples)
            
            if samples_available == 0:
                return np.array([]), []
            
            # Calculate read position
            read_pos = (self.write_pos - samples_available) % self.max_samples
            
            # Read data
            if read_pos + samples_available <= self.max_samples:
                audio_data = self.buffer[read_pos:read_pos + samples_available].copy()
            else:
                # Handle wrap-around
                first_part_size = self.max_samples - read_pos
                second_part_size = samples_available - first_part_size
                
                audio_data = np.concatenate([
                    self.buffer[read_pos:],
                    self.buffer[:second_part_size]
                ])
            
            # Get corresponding timestamps
            timestamps = list(self.timestamps)[-samples_available:] if self.timestamps else []
            
            return audio_data, timestamps
    
    def get_buffer_info(self) -> Dict[str, Any]:
        """Get buffer status information"""
        with self.lock:
            return {
                "max_duration": self.max_duration,
                "current_fill": min(self.samples_written / self.max_samples, 1.0),
                "samples_written": self.samples_written,
                "write_position": self.write_pos,
                "buffer_size": self.max_samples
            }


class RealTimeFeatureExtractor:
    """Extract audio features in real-time with minimal latency"""
    
    def __init__(self, sample_rate: int = 22050, hop_length: int = 512):
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_fft = 2048
        
        # Feature history for smoothing
        self.feature_history = {
            'spectral_centroid': deque(maxlen=50),
            'rms_energy': deque(maxlen=50),
            'zero_crossing_rate': deque(maxlen=50),
            'tempo': deque(maxlen=10),
            'pitch': deque(maxlen=30)
        }
        
        # Pre-compute mel filterbank for efficiency
        self.mel_filters = librosa.filters.mel(
            sr=sample_rate, 
            n_fft=self.n_fft, 
            n_mels=13
        )
    
    def extract_features(self, audio_chunk: np.ndarray) -> Dict[str, Any]:
        """Extract real-time features from audio chunk"""
        if len(audio_chunk) < self.hop_length:
            return self._get_empty_features()
        
        features = {}
        
        try:
            # Basic spectral features
            features.update(self._extract_spectral_features(audio_chunk))
            
            # Temporal features
            features.update(self._extract_temporal_features(audio_chunk))
            
            # Advanced features (if enough data)
            if len(audio_chunk) >= self.sample_rate:  # At least 1 second
                features.update(self._extract_advanced_features(audio_chunk))
            
            # Smooth features using history
            features = self._smooth_features(features)
            
        except Exception as e:
            print(f"Feature extraction error: {e}")
            features = self._get_empty_features()
        
        return features
    
    def _extract_spectral_features(self, audio: np.ndarray) -> Dict[str, float]:
        """Extract spectral features optimized for real-time"""
        # Compute STFT
        stft = librosa.stft(audio, hop_length=self.hop_length, n_fft=self.n_fft)
        magnitude = np.abs(stft)
        
        # Spectral centroid
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(S=magnitude, sr=self.sample_rate))
        
        # Spectral rolloff
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(S=magnitude, sr=self.sample_rate))
        
        # Spectral bandwidth
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(S=magnitude, sr=self.sample_rate))
        
        # Spectral flatness
        spectral_flatness = np.mean(librosa.feature.spectral_flatness(S=magnitude))
        
        return {
            'spectral_centroid': float(spectral_centroid),
            'spectral_rolloff': float(spectral_rolloff),
            'spectral_bandwidth': float(spectral_bandwidth),
            'spectral_flatness': float(spectral_flatness)
        }
    
    def _extract_temporal_features(self, audio: np.ndarray) -> Dict[str, float]:
        """Extract temporal features"""
        # RMS energy
        rms_energy = np.mean(librosa.feature.rms(y=audio, hop_length=self.hop_length))
        
        # Zero crossing rate
        zcr = np.mean(librosa.feature.zero_crossing_rate(audio, hop_length=self.hop_length))
        
        # Dynamic range
        dynamic_range = float(np.max(np.abs(audio)) - np.min(np.abs(audio)))
        
        return {
            'rms_energy': float(rms_energy),
            'zero_crossing_rate': float(zcr),
            'dynamic_range': dynamic_range
        }
    
    def _extract_advanced_features(self, audio: np.ndarray) -> Dict[str, float]:
        """Extract advanced features requiring longer audio segments"""
        advanced = {}
        
        try:
            # Tempo estimation
            tempo, _ = librosa.beat.beat_track(y=audio, sr=self.sample_rate)
            advanced['tempo'] = float(tempo)
            
            # Pitch estimation (fundamental frequency)
            pitches, magnitudes = librosa.piptrack(y=audio, sr=self.sample_rate, threshold=0.1)
            
            # Extract dominant pitch
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)
            
            if pitch_values:
                dominant_pitch = float(np.median(pitch_values))
                advanced['dominant_pitch'] = dominant_pitch
            else:
                advanced['dominant_pitch'] = 0.0
            
            # Harmonic-percussive separation
            y_harmonic, y_percussive = librosa.effects.hpss(audio)
            harmonic_energy = float(np.sum(y_harmonic**2))
            percussive_energy = float(np.sum(y_percussive**2))
            total_energy = harmonic_energy + percussive_energy
            
            advanced['harmonic_ratio'] = harmonic_energy / total_energy if total_energy > 0 else 0
            advanced['percussive_ratio'] = percussive_energy / total_energy if total_energy > 0 else 0
            
        except Exception as e:
            print(f"Advanced feature extraction error: {e}")
            advanced.update({
                'tempo': 0.0,
                'dominant_pitch': 0.0,
                'harmonic_ratio': 0.0,
                'percussive_ratio': 0.0
            })
        
        return advanced
    
    def _smooth_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """Apply temporal smoothing to features"""
        smoothed = features.copy()
        
        for feature_name, value in features.items():
            if feature_name in self.feature_history:
                # Add to history
                self.feature_history[feature_name].append(value)
                
                # Apply exponential smoothing
                if len(self.feature_history[feature_name]) > 1:
                    alpha = 0.3  # Smoothing factor
                    history = list(self.feature_history[feature_name])
                    smoothed_value = history[-1]
                    
                    for i in range(len(history) - 2, -1, -1):
                        smoothed_value = alpha * history[i] + (1 - alpha) * smoothed_value
                    
                    smoothed[feature_name] = float(smoothed_value)
        
        return smoothed
    
    def _get_empty_features(self) -> Dict[str, float]:
        """Return empty feature set for error cases"""
        return {
            'spectral_centroid': 0.0,
            'spectral_rolloff': 0.0,
            'spectral_bandwidth': 0.0,
            'spectral_flatness': 0.0,
            'rms_energy': 0.0,
            'zero_crossing_rate': 0.0,
            'dynamic_range': 0.0,
            'tempo': 0.0,
            'dominant_pitch': 0.0,
            'harmonic_ratio': 0.0,
            'percussive_ratio': 0.0
        }


class RealTimeAudioProcessor:
    """Main real-time audio processing system"""
    
    def __init__(self, 
                 sample_rate: int = 22050,
                 chunk_size: int = 1024,
                 buffer_duration: float = 30.0,
                 analysis_interval: float = 0.1):
        
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.buffer_duration = buffer_duration
        self.analysis_interval = analysis_interval
        
        # Components
        self.audio_buffer = RealTimeAudioBuffer(buffer_duration, sample_rate)
        self.feature_extractor = RealTimeFeatureExtractor(sample_rate)
        
        # PyAudio setup
        self.pyaudio_instance = None
        self.audio_stream = None
        
        # Processing state
        self.is_processing = False
        self.processing_thread = None
        self.analysis_thread = None
        
        # Results
        self.current_features = {}
        self.feature_history = deque(maxlen=1000)  # Store last 1000 feature sets
        self.callbacks = []
        
        # Performance monitoring
        self.processing_stats = {
            'chunks_processed': 0,
            'total_processing_time': 0.0,
            'average_processing_time': 0.0,
            'buffer_overruns': 0
        }
        
        print("🎙️ Real-Time Audio Processor Initialized")
        print(f"   Sample Rate: {sample_rate} Hz")
        print(f"   Chunk Size: {chunk_size} samples")
        print(f"   Buffer Duration: {buffer_duration} seconds")
        print(f"   Analysis Interval: {analysis_interval} seconds")
        print(f"   PyAudio Available: {'✅' if HAS_PYAUDIO else '❌'}")
    
    def add_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Add callback function for real-time feature updates"""
        self.callbacks.append(callback)
    
    def start_processing(self, input_device_index: Optional[int] = None) -> bool:
        """Start real-time audio processing"""
        if not HAS_PYAUDIO:
            print("❌ PyAudio not available for real-time processing")
            return False
        
        if self.is_processing:
            print("⚠️ Already processing")
            return True
        
        try:
            # Initialize PyAudio
            self.pyaudio_instance = pyaudio.PyAudio()
            
            # Open audio stream
            self.audio_stream = self.pyaudio_instance.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.sample_rate,
                input=True,
                input_device_index=input_device_index,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback
            )
            
            # Start processing threads
            self.is_processing = True
            self.analysis_thread = threading.Thread(target=self._analysis_loop, daemon=True)
            self.analysis_thread.start()
            
            # Start audio stream
            self.audio_stream.start_stream()
            
            print("🎙️ Real-time processing started")
            return True
            
        except Exception as e:
            print(f"❌ Failed to start processing: {e}")
            self.stop_processing()
            return False
    
    def stop_processing(self):
        """Stop real-time audio processing"""
        print("🛑 Stopping real-time processing...")
        
        self.is_processing = False
        
        if self.audio_stream:
            self.audio_stream.stop_stream()
            self.audio_stream.close()
            self.audio_stream = None
        
        if self.pyaudio_instance:
            self.pyaudio_instance.terminate()
            self.pyaudio_instance = None
        
        if self.analysis_thread and self.analysis_thread.is_alive():
            self.analysis_thread.join(timeout=2.0)
        
        print("✅ Real-time processing stopped")
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback for audio data"""
        if status:
            print(f"⚠️ Audio callback status: {status}")
            self.processing_stats['buffer_overruns'] += 1
        
        # Convert audio data
        audio_chunk = np.frombuffer(in_data, dtype=np.float32)
        
        # Write to buffer
        try:
            self.audio_buffer.write(audio_chunk)
        except Exception as e:
            print(f"❌ Buffer write error: {e}")
        
        return (None, pyaudio.paContinue)
    
    def _analysis_loop(self):
        """Main analysis loop running in separate thread"""
        print("🔄 Analysis loop started")
        
        while self.is_processing:
            try:
                start_time = time.time()
                
                # Get recent audio data
                audio_data, timestamps = self.audio_buffer.read_latest(2.0)  # 2 seconds
                
                if len(audio_data) > 0:
                    # Extract features
                    features = self.feature_extractor.extract_features(audio_data)
                    
                    # Add metadata
                    analysis_result = {
                        'timestamp': time.time(),
                        'features': features,
                        'buffer_info': self.audio_buffer.get_buffer_info(),
                        'audio_length': len(audio_data) / self.sample_rate
                    }
                    
                    # Update current features
                    self.current_features = analysis_result
                    self.feature_history.append(analysis_result)
                    
                    # Call registered callbacks
                    for callback in self.callbacks:
                        try:
                            callback(analysis_result)
                        except Exception as e:
                            print(f"⚠️ Callback error: {e}")
                    
                    # Update performance stats
                    processing_time = time.time() - start_time
                    self._update_processing_stats(processing_time)
                
                # Sleep until next analysis interval
                time.sleep(self.analysis_interval)
                
            except Exception as e:
                print(f"❌ Analysis loop error: {e}")
                time.sleep(0.1)
        
        print("🔄 Analysis loop stopped")
    
    def _update_processing_stats(self, processing_time: float):
        """Update processing performance statistics"""
        self.processing_stats['chunks_processed'] += 1
        self.processing_stats['total_processing_time'] += processing_time
        self.processing_stats['average_processing_time'] = (
            self.processing_stats['total_processing_time'] / 
            self.processing_stats['chunks_processed']
        )
    
    def get_current_features(self) -> Dict[str, Any]:
        """Get most recent feature analysis"""
        return self.current_features.copy()
    
    def get_feature_history(self, duration: float) -> List[Dict[str, Any]]:
        """Get feature history for specified duration"""
        if not self.feature_history:
            return []
        
        current_time = time.time()
        cutoff_time = current_time - duration
        
        return [
            result for result in self.feature_history 
            if result['timestamp'] >= cutoff_time
        ]
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing performance statistics"""
        return self.processing_stats.copy()
    
    def save_session(self, filepath: str, duration: float = 300.0):
        """Save recent analysis session to file"""
        session_data = {
            'session_info': {
                'timestamp': datetime.now().isoformat(),
                'duration_requested': duration,
                'sample_rate': self.sample_rate,
                'processing_stats': self.get_processing_stats()
            },
            'feature_history': self.get_feature_history(duration),
            'current_features': self.get_current_features()
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(session_data, f, indent=2, default=str)
            print(f"💾 Session saved: {filepath}")
        except Exception as e:
            print(f"❌ Failed to save session: {e}")


class RealTimeVisualizer:
    """Real-time visualization of audio features"""
    
    def __init__(self, processor: RealTimeAudioProcessor):
        self.processor = processor
        self.fig = None
        self.axes = None
        self.lines = {}
        self.is_running = False
        
        if not HAS_MATPLOTLIB:
            print("❌ Matplotlib not available for visualization")
            return
        
        # Setup matplotlib for real-time plotting
        plt.ion()
        self.setup_plots()
        
        # Register as callback
        self.processor.add_callback(self.update_plots)
        
        print("📊 Real-time visualizer initialized")
    
    def setup_plots(self):
        """Setup matplotlib plots for real-time visualization"""
        if not HAS_MATPLOTLIB:
            return
        
        # Create figure with subplots
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.suptitle('Real-Time Audio Analysis', fontsize=14)
        
        # Spectral features plot
        ax1 = self.axes[0, 0]
        ax1.set_title('Spectral Features')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Frequency (Hz)')
        ax1.grid(True)
        
        # Energy and dynamics plot
        ax2 = self.axes[0, 1]
        ax2.set_title('Energy & Dynamics')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Amplitude')
        ax2.grid(True)
        
        # Tempo and rhythm plot
        ax3 = self.axes[1, 0]
        ax3.set_title('Tempo & Rhythm')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('BPM / Ratio')
        ax3.grid(True)
        
        # Real-time waveform
        ax4 = self.axes[1, 1]
        ax4.set_title('Live Audio Waveform')
        ax4.set_xlabel('Samples')
        ax4.set_ylabel('Amplitude')
        ax4.grid(True)
        
        # Initialize empty lines
        self.lines['spectral_centroid'], = ax1.plot([], [], 'b-', label='Spectral Centroid')
        self.lines['spectral_rolloff'], = ax1.plot([], [], 'r-', label='Spectral Rolloff')
        ax1.legend()
        
        self.lines['rms_energy'], = ax2.plot([], [], 'g-', label='RMS Energy')
        self.lines['zcr'], = ax2.plot([], [], 'orange', label='Zero Crossing Rate')
        ax2.legend()
        
        self.lines['tempo'], = ax3.plot([], [], 'm-', label='Tempo')
        self.lines['harmonic_ratio'], = ax3.plot([], [], 'c-', label='Harmonic Ratio')
        ax3.legend()
        
        self.lines['waveform'], = ax4.plot([], [], 'k-', alpha=0.7)
        
        plt.tight_layout()
        plt.show(block=False)
    
    def update_plots(self, analysis_result: Dict[str, Any]):
        """Update plots with new analysis data"""
        if not HAS_MATPLOTLIB or not self.fig:
            return
        
        try:
            # Get recent history for plotting
            history = self.processor.get_feature_history(30.0)  # 30 seconds
            
            if len(history) < 2:
                return
            
            # Extract time series data
            timestamps = [r['timestamp'] for r in history]
            # Normalize timestamps to relative time
            base_time = timestamps[0]
            rel_times = [(t - base_time) for t in timestamps]
            
            features = [r['features'] for r in history]
            
            # Update spectral features
            spectral_centroids = [f.get('spectral_centroid', 0) for f in features]
            spectral_rolloffs = [f.get('spectral_rolloff', 0) for f in features]
            
            self.lines['spectral_centroid'].set_data(rel_times, spectral_centroids)
            self.lines['spectral_rolloff'].set_data(rel_times, spectral_rolloffs)
            
            # Update energy features
            rms_energies = [f.get('rms_energy', 0) * 1000 for f in features]  # Scale for visibility
            zcrs = [f.get('zero_crossing_rate', 0) * 1000 for f in features]
            
            self.lines['rms_energy'].set_data(rel_times, rms_energies)
            self.lines['zcr'].set_data(rel_times, zcrs)
            
            # Update tempo and rhythm
            tempos = [f.get('tempo', 0) for f in features]
            harmonic_ratios = [f.get('harmonic_ratio', 0) * 200 for f in features]  # Scale for visibility
            
            self.lines['tempo'].set_data(rel_times, tempos)
            self.lines['harmonic_ratio'].set_data(rel_times, harmonic_ratios)
            
            # Update waveform
            audio_data, _ = self.processor.audio_buffer.read_latest(0.1)  # 100ms
            if len(audio_data) > 0:
                # Downsample for display
                downsample_factor = max(1, len(audio_data) // 1000)
                display_audio = audio_data[::downsample_factor]
                self.lines['waveform'].set_data(range(len(display_audio)), display_audio)
            
            # Auto-scale axes
            for ax in self.axes.flat:
                ax.relim()
                ax.autoscale_view()
            
            # Update display
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            
        except Exception as e:
            print(f"⚠️ Visualization update error: {e}")
    
    def close(self):
        """Close visualization"""
        if self.fig:
            plt.close(self.fig)
            self.fig = None


def main():
    """Main function for real-time audio processing demo"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Real-Time Audio Analysis System")
    parser.add_argument("--sample-rate", type=int, default=22050, help="Audio sample rate")
    parser.add_argument("--chunk-size", type=int, default=1024, help="Audio chunk size")
    parser.add_argument("--buffer-duration", type=float, default=30.0, help="Buffer duration in seconds")
    parser.add_argument("--analysis-interval", type=float, default=0.1, help="Analysis interval in seconds")
    parser.add_argument("--device", type=int, help="Audio input device index")
    parser.add_argument("--no-visualization", action="store_true", help="Disable real-time visualization")
    parser.add_argument("--duration", type=float, default=60.0, help="Recording duration in seconds")
    parser.add_argument("--save-session", help="Save session data to file")
    
    args = parser.parse_args()
    
    print("🎙️ Starting Real-Time Audio Analysis System")
    print("=" * 50)
    
    # Initialize processor
    processor = RealTimeAudioProcessor(
        sample_rate=args.sample_rate,
        chunk_size=args.chunk_size,
        buffer_duration=args.buffer_duration,
        analysis_interval=args.analysis_interval
    )
    
    # Initialize visualizer if requested
    visualizer = None
    if not args.no_visualization and HAS_MATPLOTLIB:
        visualizer = RealTimeVisualizer(processor)
    
    # Add console output callback
    def console_callback(result):
        features = result['features']
        timestamp = result['timestamp']
        print(f"\r[{timestamp:.1f}] "
              f"Energy: {features.get('rms_energy', 0):.4f} | "
              f"Centroid: {features.get('spectral_centroid', 0):.0f}Hz | "
              f"Tempo: {features.get('tempo', 0):.1f}BPM", end="")
    
    processor.add_callback(console_callback)
    
    try:
        # Start processing
        if processor.start_processing(input_device_index=args.device):
            print(f"\n🎵 Recording for {args.duration} seconds...")
            print("Press Ctrl+C to stop early")
            
            # Run for specified duration
            time.sleep(args.duration)
            
        else:
            print("❌ Failed to start audio processing")
            return
    
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    
    finally:
        # Stop processing
        processor.stop_processing()
        
        if visualizer:
            visualizer.close()
        
        # Save session if requested
        if args.save_session:
            processor.save_session(args.save_session, args.duration)
        
        # Print final statistics
        stats = processor.get_processing_stats()
        print(f"\n📊 Final Statistics:")
        print(f"   Chunks processed: {stats['chunks_processed']}")
        print(f"   Average processing time: {stats['average_processing_time']:.4f}s")
        print(f"   Buffer overruns: {stats['buffer_overruns']}")
        
        print("✅ Real-time analysis complete")


if __name__ == "__main__":
    main()