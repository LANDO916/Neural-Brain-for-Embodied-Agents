#!/usr/bin/env python3
"""
Neural Network-Based Audio Analysis Module
Advanced deep learning models for music transcription, genre classification, and emotion recognition
"""

import os
import sys
import json
import warnings
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import pickle
from datetime import datetime

# Audio processing
import librosa
import soundfile as sf

# Deep learning
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, optimizers
    from tensorflow.keras.utils import to_categorical
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False

# Scientific computing
try:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import classification_report, confusion_matrix
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# Additional ML tools
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

warnings.filterwarnings('ignore')

class AudioFeatureExtractor:
    """Advanced feature extraction for neural networks"""
    
    def __init__(self, sample_rate: int = 22050, n_mels: int = 128, n_fft: int = 2048, hop_length: int = 512):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        # Pre-compute mel filterbank
        self.mel_filters = librosa.filters.mel(
            sr=sample_rate, 
            n_fft=n_fft, 
            n_mels=n_mels
        )
        
        print(f"🧠 Neural Audio Feature Extractor Initialized")
        print(f"   Sample Rate: {sample_rate} Hz")
        print(f"   Mel Bands: {n_mels}")
        print(f"   FFT Size: {n_fft}")
        print(f"   Hop Length: {hop_length}")
    
    def extract_melspectrogram(self, audio: np.ndarray, max_length: int = 1000) -> np.ndarray:
        """Extract mel-spectrogram for neural network input"""
        # Compute mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        # Convert to log scale
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize to [0, 1]
        log_mel_spec = (log_mel_spec - log_mel_spec.min()) / (log_mel_spec.max() - log_mel_spec.min() + 1e-8)
        
        # Pad or truncate to fixed length
        if log_mel_spec.shape[1] < max_length:
            # Pad with zeros
            pad_width = max_length - log_mel_spec.shape[1]
            log_mel_spec = np.pad(log_mel_spec, ((0, 0), (0, pad_width)), mode='constant')
        else:
            # Truncate
            log_mel_spec = log_mel_spec[:, :max_length]
        
        return log_mel_spec.T  # Shape: (time, frequency)
    
    def extract_chromagram(self, audio: np.ndarray, max_length: int = 1000) -> np.ndarray:
        """Extract chromagram features"""
        chroma = librosa.feature.chroma_stft(
            y=audio,
            sr=self.sample_rate,
            hop_length=self.hop_length
        )
        
        # Pad or truncate
        if chroma.shape[1] < max_length:
            pad_width = max_length - chroma.shape[1]
            chroma = np.pad(chroma, ((0, 0), (0, pad_width)), mode='constant')
        else:
            chroma = chroma[:, :max_length]
        
        return chroma.T  # Shape: (time, 12)
    
    def extract_mfcc(self, audio: np.ndarray, n_mfcc: int = 13, max_length: int = 1000) -> np.ndarray:
        """Extract MFCC features"""
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        # Pad or truncate
        if mfcc.shape[1] < max_length:
            pad_width = max_length - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_length]
        
        return mfcc.T  # Shape: (time, n_mfcc)
    
    def extract_spectral_features(self, audio: np.ndarray, max_length: int = 1000) -> np.ndarray:
        """Extract traditional spectral features"""
        # Spectral centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate, hop_length=self.hop_length)[0]
        
        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate, hop_length=self.hop_length)[0]
        
        # Spectral bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.sample_rate, hop_length=self.hop_length)[0]
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio, hop_length=self.hop_length)[0]
        
        # RMS energy
        rms = librosa.feature.rms(y=audio, hop_length=self.hop_length)[0]
        
        # Combine features
        features = np.array([spectral_centroid, spectral_rolloff, spectral_bandwidth, zcr, rms])
        
        # Pad or truncate
        if features.shape[1] < max_length:
            pad_width = max_length - features.shape[1]
            features = np.pad(features, ((0, 0), (0, pad_width)), mode='constant')
        else:
            features = features[:, :max_length]
        
        return features.T  # Shape: (time, 5)
    
    def extract_combined_features(self, audio: np.ndarray, max_length: int = 1000) -> Dict[str, np.ndarray]:
        """Extract all features for multi-input neural networks"""
        return {
            'mel_spectrogram': self.extract_melspectrogram(audio, max_length),
            'chromagram': self.extract_chromagram(audio, max_length),
            'mfcc': self.extract_mfcc(audio, max_length),
            'spectral': self.extract_spectral_features(audio, max_length)
        }


class GenreClassificationCNN:
    """Convolutional Neural Network for music genre classification"""
    
    def __init__(self, n_classes: int = 10, input_shape: Tuple[int, int] = (1000, 128)):
        self.n_classes = n_classes
        self.input_shape = input_shape
        self.model = None
        self.scaler = StandardScaler() if HAS_SKLEARN else None
        self.label_encoder = LabelEncoder() if HAS_SKLEARN else None
        self.is_trained = False
        
        self.genre_labels = [
            'blues', 'classical', 'country', 'disco', 'hiphop', 
            'jazz', 'metal', 'pop', 'reggae', 'rock'
        ]
        
        print(f"🎵 Genre Classification CNN Initialized")
        print(f"   Classes: {n_classes}")
        print(f"   Input Shape: {input_shape}")
    
    def build_model(self):
        """Build CNN architecture for genre classification"""
        if not HAS_TENSORFLOW:
            print("❌ TensorFlow not available")
            return None
        
        model = models.Sequential([
            # First convolutional block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*self.input_shape, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),
            layers.Dropout(0.25),
            
            # Second convolutional block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),
            layers.Dropout(0.25),
            
            # Third convolutional block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),
            layers.Dropout(0.25),
            
            # Fourth convolutional block
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),
            layers.Dropout(0.25),
            
            # Global average pooling
            layers.GlobalAveragePooling2D(),
            
            # Dense layers
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            
            # Output layer
            layers.Dense(self.n_classes, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        print("✅ CNN model built successfully")
        return model
    
    def generate_synthetic_data(self, n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic training data for demonstration"""
        print(f"🔄 Generating {n_samples} synthetic samples...")
        
        X = []
        y = []
        
        np.random.seed(42)  # For reproducibility
        
        for genre_idx in range(min(self.n_classes, len(self.genre_labels))):
            samples_per_genre = n_samples // self.n_classes
            
            for _ in range(samples_per_genre):
                # Generate synthetic mel-spectrogram
                if self.genre_labels[genre_idx] == 'classical':
                    # Classical: Lower frequencies, more harmonic structure
                    freq_center = 40 + np.random.normal(0, 10)
                    mel_spec = self._generate_harmonic_spectrogram(freq_center, harmonics=5)
                elif self.genre_labels[genre_idx] == 'metal':
                    # Metal: Higher frequencies, more noise
                    freq_center = 80 + np.random.normal(0, 15)
                    mel_spec = self._generate_noisy_spectrogram(freq_center, noise_level=0.3)
                elif self.genre_labels[genre_idx] == 'jazz':
                    # Jazz: Complex harmonic structure
                    freq_center = 60 + np.random.normal(0, 20)
                    mel_spec = self._generate_complex_spectrogram(freq_center)
                elif self.genre_labels[genre_idx] == 'pop':
                    # Pop: Balanced frequencies
                    freq_center = 64 + np.random.normal(0, 12)
                    mel_spec = self._generate_balanced_spectrogram(freq_center)
                else:
                    # Default pattern
                    freq_center = 50 + np.random.normal(0, 15)
                    mel_spec = self._generate_generic_spectrogram(freq_center)
                
                X.append(mel_spec)
                y.append(genre_idx)
        
        X = np.array(X)
        y = to_categorical(np.array(y), num_classes=self.n_classes)
        
        print(f"✅ Generated data shape: X={X.shape}, y={y.shape}")
        return X, y
    
    def _generate_harmonic_spectrogram(self, freq_center: float, harmonics: int = 3) -> np.ndarray:
        """Generate harmonic spectrogram pattern"""
        spec = np.random.normal(0, 0.1, self.input_shape)
        
        # Add harmonic peaks
        for h in range(1, harmonics + 1):
            freq_idx = int(freq_center * h) % self.input_shape[1]
            time_pattern = np.sin(np.linspace(0, 4*np.pi, self.input_shape[0])) * 0.5 + 0.5
            spec[:, freq_idx] += time_pattern * (1.0 / h)
        
        return np.clip(spec, 0, 1)
    
    def _generate_noisy_spectrogram(self, freq_center: float, noise_level: float = 0.2) -> np.ndarray:
        """Generate noisy spectrogram pattern"""
        spec = np.random.normal(0, noise_level, self.input_shape)
        
        # Add broad frequency content
        for i in range(5):
            freq_idx = int(freq_center + i * 10) % self.input_shape[1]
            time_pattern = np.random.random(self.input_shape[0]) * 0.8
            spec[:, freq_idx] += time_pattern
        
        return np.clip(spec, 0, 1)
    
    def _generate_complex_spectrogram(self, freq_center: float) -> np.ndarray:
        """Generate complex spectrogram pattern"""
        spec = np.random.normal(0, 0.05, self.input_shape)
        
        # Add multiple frequency components with time variation
        for i in range(7):
            freq_idx = int(freq_center + i * 8 + np.random.normal(0, 3)) % self.input_shape[1]
            phase = np.random.random() * 2 * np.pi
            time_pattern = np.sin(np.linspace(phase, phase + 6*np.pi, self.input_shape[0])) * 0.3 + 0.4
            spec[:, freq_idx] += time_pattern
        
        return np.clip(spec, 0, 1)
    
    def _generate_balanced_spectrogram(self, freq_center: float) -> np.ndarray:
        """Generate balanced spectrogram pattern"""
        spec = np.random.normal(0, 0.08, self.input_shape)
        
        # Add balanced frequency content
        for i in range(4):
            freq_idx = int(freq_center + i * 15) % self.input_shape[1]
            time_pattern = np.ones(self.input_shape[0]) * 0.6 + np.random.normal(0, 0.1, self.input_shape[0])
            spec[:, freq_idx] += time_pattern
        
        return np.clip(spec, 0, 1)
    
    def _generate_generic_spectrogram(self, freq_center: float) -> np.ndarray:
        """Generate generic spectrogram pattern"""
        spec = np.random.normal(0, 0.1, self.input_shape)
        
        # Add some structure
        freq_idx = int(freq_center) % self.input_shape[1]
        time_pattern = np.random.random(self.input_shape[0]) * 0.7
        spec[:, freq_idx] += time_pattern
        
        return np.clip(spec, 0, 1)
    
    def train(self, X: np.ndarray, y: np.ndarray, validation_split: float = 0.2, epochs: int = 50):
        """Train the genre classification model"""
        if not HAS_TENSORFLOW or self.model is None:
            print("❌ Model not available or not built")
            return False
        
        print(f"🚀 Training genre classification model...")
        print(f"   Training samples: {X.shape[0]}")
        print(f"   Epochs: {epochs}")
        
        # Reshape for CNN (add channel dimension)
        if len(X.shape) == 3:
            X = X[..., np.newaxis]
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y.argmax(axis=1)
        )
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
        ]
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        self.is_trained = True
        
        # Evaluate
        val_loss, val_accuracy = self.model.evaluate(X_val, y_val, verbose=0)
        print(f"✅ Training complete!")
        print(f"   Validation accuracy: {val_accuracy:.3f}")
        print(f"   Validation loss: {val_loss:.3f}")
        
        return True
    
    def predict(self, mel_spectrogram: np.ndarray) -> Dict[str, Any]:
        """Predict genre from mel-spectrogram"""
        if not self.is_trained or self.model is None:
            return {"error": "Model not trained"}
        
        # Reshape for prediction
        if len(mel_spectrogram.shape) == 2:
            mel_spectrogram = mel_spectrogram[np.newaxis, ..., np.newaxis]
        elif len(mel_spectrogram.shape) == 3:
            mel_spectrogram = mel_spectrogram[..., np.newaxis]
        
        # Predict
        predictions = self.model.predict(mel_spectrogram, verbose=0)
        
        # Get top predictions
        top_indices = np.argsort(predictions[0])[-3:][::-1]  # Top 3
        
        results = {
            "predicted_genre": self.genre_labels[top_indices[0]],
            "confidence": float(predictions[0][top_indices[0]]),
            "top_predictions": [
                {
                    "genre": self.genre_labels[idx],
                    "confidence": float(predictions[0][idx])
                }
                for idx in top_indices
            ],
            "all_probabilities": {
                self.genre_labels[i]: float(predictions[0][i])
                for i in range(len(self.genre_labels))
            }
        }
        
        return results
    
    def save_model(self, filepath: str):
        """Save trained model"""
        if self.model is None:
            print("❌ No model to save")
            return False
        
        try:
            self.model.save(filepath)
            print(f"💾 Model saved: {filepath}")
            return True
        except Exception as e:
            print(f"❌ Failed to save model: {e}")
            return False
    
    def load_model(self, filepath: str):
        """Load trained model"""
        if not HAS_TENSORFLOW:
            print("❌ TensorFlow not available")
            return False
        
        try:
            self.model = keras.models.load_model(filepath)
            self.is_trained = True
            print(f"📁 Model loaded: {filepath}")
            return True
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False


class EmotionRecognitionRNN:
    """Recurrent Neural Network for emotion recognition in music"""
    
    def __init__(self, n_emotions: int = 4, sequence_length: int = 100, feature_dim: int = 13):
        self.n_emotions = n_emotions
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.model = None
        self.is_trained = False
        
        self.emotion_labels = ['happy', 'sad', 'angry', 'calm'][:n_emotions]
        
        print(f"🎭 Emotion Recognition RNN Initialized")
        print(f"   Emotions: {self.emotion_labels}")
        print(f"   Sequence Length: {sequence_length}")
        print(f"   Feature Dimension: {feature_dim}")
    
    def build_model(self):
        """Build RNN architecture for emotion recognition"""
        if not HAS_TENSORFLOW:
            print("❌ TensorFlow not available")
            return None
        
        model = models.Sequential([
            # LSTM layers
            layers.LSTM(128, return_sequences=True, input_shape=(self.sequence_length, self.feature_dim)),
            layers.Dropout(0.3),
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.3),
            layers.LSTM(32),
            layers.Dropout(0.3),
            
            # Dense layers
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.3),
            
            # Output layer
            layers.Dense(self.n_emotions, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        print("✅ RNN model built successfully")
        return model
    
    def generate_synthetic_data(self, n_samples: int = 800) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic emotional music data"""
        print(f"🔄 Generating {n_samples} synthetic emotion samples...")
        
        X = []
        y = []
        
        np.random.seed(42)
        
        for emotion_idx, emotion in enumerate(self.emotion_labels):
            samples_per_emotion = n_samples // self.n_emotions
            
            for _ in range(samples_per_emotion):
                # Generate synthetic MFCC sequence based on emotion
                if emotion == 'happy':
                    # Happy: Higher energy, more variation
                    sequence = self._generate_happy_sequence()
                elif emotion == 'sad':
                    # Sad: Lower energy, minor characteristics
                    sequence = self._generate_sad_sequence()
                elif emotion == 'angry':
                    # Angry: High energy, sharp changes
                    sequence = self._generate_angry_sequence()
                elif emotion == 'calm':
                    # Calm: Stable, low variation
                    sequence = self._generate_calm_sequence()
                else:
                    sequence = self._generate_neutral_sequence()
                
                X.append(sequence)
                y.append(emotion_idx)
        
        X = np.array(X)
        y = to_categorical(np.array(y), num_classes=self.n_emotions)
        
        print(f"✅ Generated emotion data shape: X={X.shape}, y={y.shape}")
        return X, y
    
    def _generate_happy_sequence(self) -> np.ndarray:
        """Generate happy emotion sequence"""
        # Higher frequency content, more variation
        base_pattern = np.random.normal(2, 1, (self.sequence_length, self.feature_dim))
        
        # Add upward trends
        for i in range(self.feature_dim):
            trend = np.linspace(0, 1, self.sequence_length) * np.random.uniform(0.5, 1.5)
            base_pattern[:, i] += trend
        
        # Add periodic variations
        time_axis = np.linspace(0, 4*np.pi, self.sequence_length)
        for i in range(0, self.feature_dim, 2):
            base_pattern[:, i] += np.sin(time_axis + i) * 0.5
        
        return base_pattern
    
    def _generate_sad_sequence(self) -> np.ndarray:
        """Generate sad emotion sequence"""
        # Lower frequency content, less variation
        base_pattern = np.random.normal(-1, 0.5, (self.sequence_length, self.feature_dim))
        
        # Add downward trends
        for i in range(self.feature_dim):
            trend = np.linspace(0, -0.5, self.sequence_length) * np.random.uniform(0.5, 1.0)
            base_pattern[:, i] += trend
        
        # Add slow variations
        time_axis = np.linspace(0, 2*np.pi, self.sequence_length)
        for i in range(1, self.feature_dim, 3):
            base_pattern[:, i] += np.cos(time_axis + i) * 0.3
        
        return base_pattern
    
    def _generate_angry_sequence(self) -> np.ndarray:
        """Generate angry emotion sequence"""
        # High energy, sharp variations
        base_pattern = np.random.normal(1, 2, (self.sequence_length, self.feature_dim))
        
        # Add sharp spikes
        spike_indices = np.random.choice(self.sequence_length, size=self.sequence_length//10, replace=False)
        for idx in spike_indices:
            feature_idx = np.random.randint(0, self.feature_dim)
            base_pattern[idx, feature_idx] += np.random.uniform(2, 4)
        
        # Add aggressive high-frequency content
        time_axis = np.linspace(0, 8*np.pi, self.sequence_length)
        for i in range(self.feature_dim):
            base_pattern[:, i] += np.sin(time_axis * (i+1)) * 0.8
        
        return base_pattern
    
    def _generate_calm_sequence(self) -> np.ndarray:
        """Generate calm emotion sequence"""
        # Stable, low variation
        base_pattern = np.random.normal(0, 0.3, (self.sequence_length, self.feature_dim))
        
        # Add gentle, slow variations
        time_axis = np.linspace(0, np.pi, self.sequence_length)
        for i in range(self.feature_dim):
            base_pattern[:, i] += np.sin(time_axis + i*0.1) * 0.2
        
        # Smooth the sequence
        for i in range(self.feature_dim):
            # Simple moving average
            kernel = np.ones(5) / 5
            if len(base_pattern[:, i]) >= len(kernel):
                base_pattern[:, i] = np.convolve(base_pattern[:, i], kernel, mode='same')
        
        return base_pattern
    
    def _generate_neutral_sequence(self) -> np.ndarray:
        """Generate neutral emotion sequence"""
        return np.random.normal(0, 1, (self.sequence_length, self.feature_dim))
    
    def train(self, X: np.ndarray, y: np.ndarray, validation_split: float = 0.2, epochs: int = 30):
        """Train the emotion recognition model"""
        if not HAS_TENSORFLOW or self.model is None:
            print("❌ Model not available or not built")
            return False
        
        print(f"🚀 Training emotion recognition model...")
        print(f"   Training samples: {X.shape[0]}")
        print(f"   Epochs: {epochs}")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y.argmax(axis=1)
        )
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.7, patience=4)
        ]
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=16,
            callbacks=callbacks,
            verbose=1
        )
        
        self.is_trained = True
        
        # Evaluate
        val_loss, val_accuracy = self.model.evaluate(X_val, y_val, verbose=0)
        print(f"✅ Training complete!")
        print(f"   Validation accuracy: {val_accuracy:.3f}")
        print(f"   Validation loss: {val_loss:.3f}")
        
        return True
    
    def predict(self, mfcc_sequence: np.ndarray) -> Dict[str, Any]:
        """Predict emotion from MFCC sequence"""
        if not self.is_trained or self.model is None:
            return {"error": "Model not trained"}
        
        # Reshape for prediction if needed
        if len(mfcc_sequence.shape) == 2:
            mfcc_sequence = mfcc_sequence[np.newaxis, ...]
        
        # Predict
        predictions = self.model.predict(mfcc_sequence, verbose=0)
        
        # Get results
        predicted_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_idx])
        
        results = {
            "predicted_emotion": self.emotion_labels[predicted_idx],
            "confidence": confidence,
            "all_probabilities": {
                self.emotion_labels[i]: float(predictions[0][i])
                for i in range(len(self.emotion_labels))
            }
        }
        
        return results


class NeuralAudioAnalyzer:
    """Main neural network-based audio analyzer"""
    
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
        self.feature_extractor = AudioFeatureExtractor(sample_rate)
        
        # Models
        self.genre_model = None
        self.emotion_model = None
        
        print(f"🧠 Neural Audio Analyzer Initialized")
        print(f"   TensorFlow Available: {'✅' if HAS_TENSORFLOW else '❌'}")
        print(f"   Scikit-learn Available: {'✅' if HAS_SKLEARN else '❌'}")
    
    def initialize_models(self, train_models: bool = True):
        """Initialize and optionally train all neural models"""
        if not HAS_TENSORFLOW:
            print("❌ TensorFlow not available for neural models")
            return False
        
        print("🔄 Initializing neural models...")
        
        # Genre classification model
        self.genre_model = GenreClassificationCNN()
        self.genre_model.build_model()
        
        # Emotion recognition model
        self.emotion_model = EmotionRecognitionRNN()
        self.emotion_model.build_model()
        
        if train_models:
            # Train genre model
            print("\n📚 Training genre classification model...")
            X_genre, y_genre = self.genre_model.generate_synthetic_data(n_samples=1000)
            self.genre_model.train(X_genre, y_genre, epochs=20)
            
            # Train emotion model
            print("\n📚 Training emotion recognition model...")
            X_emotion, y_emotion = self.emotion_model.generate_synthetic_data(n_samples=800)
            self.emotion_model.train(X_emotion, y_emotion, epochs=15)
        
        print("✅ Neural models initialized")
        return True
    
    def analyze_audio(self, audio_path: str) -> Dict[str, Any]:
        """Perform neural network-based audio analysis"""
        print(f"🧠 Neural analysis: {Path(audio_path).name}")
        
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            duration = len(audio) / sr
            
            results = {
                "metadata": {
                    "file": audio_path,
                    "duration": duration,
                    "sample_rate": sr,
                    "analysis_time": datetime.now().isoformat(),
                    "analyzer": "neural"
                },
                "neural_analysis": {}
            }
            
            # Extract features
            features = self.feature_extractor.extract_combined_features(audio)
            
            # Genre classification
            if self.genre_model and self.genre_model.is_trained:
                print("   → Genre classification...")
                genre_result = self.genre_model.predict(features['mel_spectrogram'])
                results["neural_analysis"]["genre_classification"] = genre_result
            
            # Emotion recognition
            if self.emotion_model and self.emotion_model.is_trained:
                print("   → Emotion recognition...")
                # Use MFCC for emotion recognition
                emotion_result = self.emotion_model.predict(features['mfcc'])
                results["neural_analysis"]["emotion_recognition"] = emotion_result
            
            # Feature analysis
            results["neural_analysis"]["feature_shapes"] = {
                key: list(value.shape) for key, value in features.items()
            }
            
            return results
            
        except Exception as e:
            return {
                "metadata": {"file": audio_path, "error": str(e)},
                "neural_analysis": {"error": str(e)}
            }
    
    def save_models(self, directory: str = "neural_models"):
        """Save all trained models"""
        os.makedirs(directory, exist_ok=True)
        
        if self.genre_model:
            genre_path = os.path.join(directory, "genre_model.h5")
            self.genre_model.save_model(genre_path)
        
        if self.emotion_model:
            emotion_path = os.path.join(directory, "emotion_model.h5")
            self.emotion_model.save_model(emotion_path)
    
    def load_models(self, directory: str = "neural_models"):
        """Load pre-trained models"""
        # Initialize models first
        if self.genre_model is None:
            self.genre_model = GenreClassificationCNN()
            self.genre_model.build_model()
        
        if self.emotion_model is None:
            self.emotion_model = EmotionRecognitionRNN()
            self.emotion_model.build_model()
        
        # Load weights
        genre_path = os.path.join(directory, "genre_model.h5")
        if os.path.exists(genre_path):
            self.genre_model.load_model(genre_path)
        
        emotion_path = os.path.join(directory, "emotion_model.h5")
        if os.path.exists(emotion_path):
            self.emotion_model.load_model(emotion_path)


def main():
    """Main function for neural audio analysis demo"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Neural Network Audio Analysis")
    parser.add_argument("--train", action="store_true", help="Train models with synthetic data")
    parser.add_argument("--analyze", help="Audio file to analyze")
    parser.add_argument("--save-models", help="Directory to save trained models")
    parser.add_argument("--load-models", help="Directory to load models from")
    
    args = parser.parse_args()
    
    print("🧠 Neural Network Audio Analysis System")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = NeuralAudioAnalyzer()
    
    if args.load_models:
        analyzer.load_models(args.load_models)
    elif args.train:
        analyzer.initialize_models(train_models=True)
        
        if args.save_models:
            analyzer.save_models(args.save_models)
    else:
        analyzer.initialize_models(train_models=False)
    
    if args.analyze:
        if not os.path.exists(args.analyze):
            print(f"❌ File not found: {args.analyze}")
            return
        
        # Analyze audio file
        results = analyzer.analyze_audio(args.analyze)
        
        # Display results
        print(f"\n📊 Neural Analysis Results:")
        neural_results = results.get("neural_analysis", {})
        
        if "genre_classification" in neural_results:
            genre = neural_results["genre_classification"]
            print(f"   Genre: {genre.get('predicted_genre', 'unknown')} "
                  f"(confidence: {genre.get('confidence', 0):.3f})")
        
        if "emotion_recognition" in neural_results:
            emotion = neural_results["emotion_recognition"]
            print(f"   Emotion: {emotion.get('predicted_emotion', 'unknown')} "
                  f"(confidence: {emotion.get('confidence', 0):.3f})")
        
        # Save results
        output_file = f"neural_analysis_{Path(args.analyze).stem}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"💾 Results saved: {output_file}")


if __name__ == "__main__":
    main()