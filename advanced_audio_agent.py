#!/usr/bin/env python3
"""
Advanced Multi-Dimensional Audio Analysis Agent
Enhanced version with additional features and capabilities

This extends the basic agent with:
- Genre classification
- Mood and emotion detection
- Advanced rhythm analysis
- Instrument detection
- Audio quality analysis
- Real-time streaming capability
"""

import os
import sys
import json
import warnings
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import concurrent.futures
from datetime import datetime
import threading
import queue
import time

# Extended imports for advanced features
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

# Text processing
try:
    import speech_recognition as sr
    from textblob import TextBlob
    import nltk
    HAS_SPEECH = True
except ImportError:
    HAS_SPEECH = False

# Audio processing
try:
    from pydub import AudioSegment
    HAS_PYDUB = True
except ImportError:
    HAS_PYDUB = False

warnings.filterwarnings('ignore')

class AdvancedAudioAnalysisAgent:
    """Advanced multi-dimensional audio analysis agent with extended capabilities"""
    
    def __init__(self, sample_rate: int = 22050, enable_streaming: bool = False):
        self.sample_rate = sample_rate
        self.enable_streaming = enable_streaming
        self.results = {}
        self.analysis_history = []
        
        # Initialize components
        if HAS_SPEECH:
            self.recognizer = sr.Recognizer()
        
        # Initialize machine learning models
        self.genre_classifier = None
        self.mood_classifier = None
        self._initialize_ml_models()
        
        # Streaming components
        if enable_streaming:
            self.audio_queue = queue.Queue()
            self.streaming_thread = None
            self.is_streaming = False
        
        print("🎵 Advanced Audio Analysis Agent Initialized")
        print(f"   Sample Rate: {sample_rate} Hz")
        print(f"   Speech Recognition: {'✅' if HAS_SPEECH else '❌'}")
        print(f"   Machine Learning: {'✅' if HAS_SKLEARN else '❌'}")
        print(f"   Streaming Mode: {'✅' if enable_streaming else '❌'}")
        print(f"   Advanced Features: {'✅' if HAS_PANDAS else '❌'}")
    
    def _initialize_ml_models(self):
        """Initialize machine learning models for classification"""
        if not HAS_SKLEARN:
            return
        
        # Simple genre classification model (would be trained on real data in production)
        self.genre_classifier = RandomForestClassifier(n_estimators=50, random_state=42)
        self.genre_scaler = StandardScaler()
        
        # Mood classification model
        self.mood_classifier = RandomForestClassifier(n_estimators=30, random_state=42)
        self.mood_scaler = StandardScaler()
        
        # Pre-train with synthetic data for demonstration
        self._pretrain_models()
    
    def _pretrain_models(self):
        """Pre-train models with synthetic data for demonstration"""
        if not HAS_SKLEARN:
            return
        
        try:
            # Generate synthetic training data for genre classification
            np.random.seed(42)
            n_samples = 1000
            
            # Features: [tempo, spectral_centroid, spectral_rolloff, zcr, mfcc_mean, energy]
            # Genres: rock, pop, classical, jazz, electronic
            genre_features = []
            genre_labels = []
            
            genres = ['rock', 'pop', 'classical', 'jazz', 'electronic']
            
            for i, genre in enumerate(genres):
                for _ in range(n_samples // len(genres)):
                    if genre == 'rock':
                        features = [
                            np.random.normal(120, 20),  # tempo
                            np.random.normal(2000, 500),  # spectral_centroid
                            np.random.normal(4000, 1000),  # spectral_rolloff
                            np.random.normal(0.15, 0.05),  # zcr
                            np.random.normal(-10, 5),  # mfcc_mean
                            np.random.normal(0.3, 0.1)  # energy
                        ]
                    elif genre == 'pop':
                        features = [
                            np.random.normal(110, 15),
                            np.random.normal(1800, 400),
                            np.random.normal(3500, 800),
                            np.random.normal(0.12, 0.04),
                            np.random.normal(-8, 4),
                            np.random.normal(0.25, 0.08)
                        ]
                    elif genre == 'classical':
                        features = [
                            np.random.normal(80, 25),
                            np.random.normal(1500, 600),
                            np.random.normal(3000, 1200),
                            np.random.normal(0.08, 0.03),
                            np.random.normal(-12, 6),
                            np.random.normal(0.2, 0.1)
                        ]
                    elif genre == 'jazz':
                        features = [
                            np.random.normal(140, 30),
                            np.random.normal(2200, 700),
                            np.random.normal(4500, 1500),
                            np.random.normal(0.18, 0.06),
                            np.random.normal(-9, 5),
                            np.random.normal(0.28, 0.12)
                        ]
                    else:  # electronic
                        features = [
                            np.random.normal(128, 10),
                            np.random.normal(2500, 800),
                            np.random.normal(5000, 2000),
                            np.random.normal(0.20, 0.08),
                            np.random.normal(-6, 3),
                            np.random.normal(0.35, 0.15)
                        ]
                    
                    genre_features.append(features)
                    genre_labels.append(i)
            
            # Train genre classifier
            X_genre = np.array(genre_features)
            y_genre = np.array(genre_labels)
            
            X_genre_scaled = self.genre_scaler.fit_transform(X_genre)
            self.genre_classifier.fit(X_genre_scaled, y_genre)
            
            # Generate synthetic mood data
            # Moods: happy, sad, energetic, calm, aggressive
            mood_features = []
            mood_labels = []
            
            moods = ['happy', 'sad', 'energetic', 'calm', 'aggressive']
            
            for i, mood in enumerate(moods):
                for _ in range(n_samples // len(moods)):
                    if mood == 'happy':
                        features = [
                            np.random.normal(120, 15),  # tempo
                            np.random.normal(2000, 400),  # brightness
                            np.random.normal(0.25, 0.1),  # energy
                            np.random.normal(0.6, 0.2),  # valence
                            np.random.normal(0.8, 0.1)  # major_key_likelihood
                        ]
                    elif mood == 'sad':
                        features = [
                            np.random.normal(70, 20),
                            np.random.normal(1200, 300),
                            np.random.normal(0.15, 0.05),
                            np.random.normal(0.2, 0.1),
                            np.random.normal(0.3, 0.2)
                        ]
                    elif mood == 'energetic':
                        features = [
                            np.random.normal(140, 20),
                            np.random.normal(2500, 500),
                            np.random.normal(0.4, 0.1),
                            np.random.normal(0.7, 0.2),
                            np.random.normal(0.7, 0.2)
                        ]
                    elif mood == 'calm':
                        features = [
                            np.random.normal(80, 15),
                            np.random.normal(1500, 400),
                            np.random.normal(0.2, 0.08),
                            np.random.normal(0.5, 0.2),
                            np.random.normal(0.6, 0.3)
                        ]
                    else:  # aggressive
                        features = [
                            np.random.normal(160, 25),
                            np.random.normal(3000, 800),
                            np.random.normal(0.5, 0.15),
                            np.random.normal(0.3, 0.15),
                            np.random.normal(0.4, 0.3)
                        ]
                    
                    mood_features.append(features)
                    mood_labels.append(i)
            
            # Train mood classifier
            X_mood = np.array(mood_features)
            y_mood = np.array(mood_labels)
            
            X_mood_scaled = self.mood_scaler.fit_transform(X_mood)
            self.mood_classifier.fit(X_mood_scaled, y_mood)
            
            print("   ML Models: Pre-trained with synthetic data")
            
        except Exception as e:
            print(f"   ML Models: Training failed - {e}")
    
    def analyze_audio(self, audio_path: str, include_advanced: bool = True) -> Dict[str, Any]:
        """
        Perform comprehensive audio analysis with advanced features
        
        Args:
            audio_path: Path to the audio file
            include_advanced: Whether to include advanced ML-based analysis
            
        Returns:
            Dictionary containing all analysis results
        """
        print(f"\n🔍 Advanced Analysis: {Path(audio_path).name}")
        
        # Load audio
        try:
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            duration = len(y) / sr
            print(f"   Duration: {duration:.2f} seconds")
            print(f"   Sample Rate: {sr} Hz")
        except Exception as e:
            return {"error": f"Failed to load audio: {e}"}
        
        # Initialize results structure
        results = {
            "metadata": {
                "file": audio_path,
                "duration": duration,
                "sample_rate": sr,
                "samples": len(y),
                "analysis_time": datetime.now().isoformat(),
                "agent_version": "advanced"
            },
            "musical_structural": {},
            "lyrical_vocal": {},
            "sonic_spectral": {},
            "advanced_features": {}
        }
        
        # Core analysis (parallel processing)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            # Submit core analysis tasks
            musical_future = executor.submit(self._analyze_musical_structural_advanced, y, sr)
            vocal_future = executor.submit(self._analyze_lyrical_vocal_advanced, y, sr, audio_path)
            spectral_future = executor.submit(self._analyze_sonic_spectral_advanced, y, sr)
            
            # Submit advanced analysis if enabled
            if include_advanced:
                advanced_future = executor.submit(self._analyze_advanced_features, y, sr)
            
            # Collect results
            results["musical_structural"] = musical_future.result()
            results["lyrical_vocal"] = vocal_future.result()
            results["sonic_spectral"] = spectral_future.result()
            
            if include_advanced:
                results["advanced_features"] = advanced_future.result()
        
        # Store results
        self.results = results
        self.analysis_history.append({
            "timestamp": datetime.now().isoformat(),
            "file": audio_path,
            "duration": duration,
            "summary": self._generate_summary(results)
        })
        
        return results
    
    def _analyze_musical_structural_advanced(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Enhanced musical & structural analysis"""
        print("🎼 Advanced Musical & Structural Analysis...")
        
        analysis = {}
        
        try:
            # Enhanced Rhythm Analysis
            print("   → Advanced Rhythm Analysis")
            
            # Multiple tempo estimation methods
            onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
            onset_times = librosa.frames_to_time(onset_frames, sr=sr)
            
            # Tempo estimation
            if len(onset_times) > 1:
                intervals = np.diff(onset_times)
                tempo_from_onsets = 60.0 / np.mean(intervals) if len(intervals) > 0 else 0
                
                # Alternative tempo estimation using beat tracking
                try:
                    tempo_alt, beats = librosa.beat.beat_track(y=y, sr=sr)
                    tempo_confidence = len(beats) / (len(y) / sr * 2)  # beats per second / expected
                except:
                    tempo_alt = tempo_from_onsets
                    tempo_confidence = 0.5
                
                # Combine estimates
                final_tempo = (tempo_from_onsets + float(tempo_alt)) / 2
            else:
                final_tempo = 0
                tempo_confidence = 0
            
            # Rhythm complexity analysis
            if len(onset_times) > 2:
                rhythm_variability = np.std(intervals) / np.mean(intervals) if len(intervals) > 0 else 0
                rhythm_regularity = 1.0 / (1.0 + rhythm_variability)
            else:
                rhythm_variability = 0
                rhythm_regularity = 0
            
            analysis["rhythm"] = {
                "estimated_tempo": round(final_tempo, 1),
                "tempo_confidence": float(tempo_confidence),
                "onsets_detected": len(onset_times),
                "rhythm_regularity": float(rhythm_regularity),
                "rhythm_complexity": float(rhythm_variability),
                "beat_strength": float(np.mean(np.diff(onset_times)) if len(onset_times) > 1 else 0)
            }
            
            # Enhanced Harmony Analysis
            print("   → Advanced Harmony Analysis")
            
            # Chromagram analysis
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            chroma_mean = np.mean(chroma, axis=1)
            chroma_std = np.std(chroma, axis=1)
            
            # Key detection with confidence
            key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            key_strengths = chroma_mean
            estimated_key_idx = np.argmax(key_strengths)
            estimated_key = key_names[estimated_key_idx]
            key_confidence = key_strengths[estimated_key_idx] / np.sum(key_strengths)
            
            # Major/minor detection
            major_profile = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1])  # Major scale pattern
            minor_profile = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0])  # Minor scale pattern
            
            major_correlation = np.corrcoef(chroma_mean, major_profile)[0, 1]
            minor_correlation = np.corrcoef(chroma_mean, minor_profile)[0, 1]
            
            mode = "major" if major_correlation > minor_correlation else "minor"
            mode_confidence = max(major_correlation, minor_correlation)
            
            # Chord progression analysis
            chord_changes = np.sum(np.abs(np.diff(chroma, axis=1)) > 0.1)
            harmonic_complexity = np.mean(chroma_std)
            
            analysis["harmony"] = {
                "estimated_key": estimated_key,
                "key_confidence": float(key_confidence),
                "mode": mode,
                "mode_confidence": float(mode_confidence) if not np.isnan(mode_confidence) else 0.0,
                "tonal_clarity": float(np.max(chroma_mean) / np.mean(chroma_mean)),
                "harmonic_complexity": float(harmonic_complexity),
                "chord_changes": int(chord_changes),
                "chromaticism": float(np.std(chroma_mean))
            }
            
            # Advanced Structure Analysis
            print("   → Advanced Structure Analysis")
            
            # MFCC-based segmentation
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            
            # Novelty detection for structural boundaries
            try:
                boundaries = librosa.segment.agglomerative(mfcc, k=min(8, int(len(y) / sr / 10)))
                boundary_times = librosa.frames_to_time(boundaries, sr=sr)
            except:
                boundary_times = np.linspace(0, len(y)/sr, min(4, int(len(y)/sr/15)))
            
            # Segment analysis
            segments = []
            for i in range(len(boundary_times)-1):
                start_time = boundary_times[i]
                end_time = boundary_times[i+1]
                
                start_frame = int(start_time * sr)
                end_frame = int(end_time * sr)
                
                if end_frame > start_frame:
                    segment_audio = y[start_frame:end_frame]
                    segment_energy = np.mean(segment_audio**2)
                    segment_spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=segment_audio, sr=sr))
                    
                    segments.append({
                        "start_time": float(start_time),
                        "end_time": float(end_time),
                        "duration": float(end_time - start_time),
                        "energy": float(segment_energy),
                        "spectral_centroid": float(segment_spectral_centroid)
                    })
            
            analysis["structure"] = {
                "total_segments": len(segments),
                "segments": segments[:10],  # Limit output
                "structural_complexity": float(len(boundary_times) / (len(y) / sr)),
                "average_segment_duration": float(np.mean([s["duration"] for s in segments]) if segments else 0)
            }
            
        except Exception as e:
            analysis["error"] = str(e)
        
        return analysis
    
    def _analyze_lyrical_vocal_advanced(self, y: np.ndarray, sr: int, audio_path: str) -> Dict[str, Any]:
        """Enhanced lyrical & vocal analysis"""
        print("🎤 Advanced Lyrical & Vocal Analysis...")
        
        analysis = {}
        
        try:
            # Enhanced Voice Activity Detection
            print("   → Enhanced Voice Activity Detection")
            
            # Multiple VAD methods
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            
            # Voice activity indicators
            voice_indicators = []
            confidence_scores = []
            
            for i in range(len(spectral_centroids)):
                # Multiple criteria for voice detection
                spectral_criterion = 1000 < spectral_centroids[i] < 4000
                rolloff_criterion = 2000 < spectral_rolloff[i] < 8000
                zcr_criterion = 0.05 < zcr[i] < 0.35
                
                # MFCC-based criterion (voice has specific MFCC patterns)
                if i < mfcc.shape[1]:
                    mfcc_frame = mfcc[:, i]
                    mfcc_criterion = -50 < mfcc_frame[1] < 20  # Second MFCC coefficient
                else:
                    mfcc_criterion = False
                
                # Combine criteria
                criteria_met = sum([spectral_criterion, rolloff_criterion, zcr_criterion, mfcc_criterion])
                voice_probability = criteria_met / 4.0
                
                voice_indicators.append(1 if voice_probability > 0.5 else 0)
                confidence_scores.append(voice_probability)
            
            voice_ratio = np.mean(voice_indicators)
            voice_confidence = np.mean(confidence_scores)
            
            # Voice quality metrics
            if voice_ratio > 0.1:
                voice_frames = [i for i, v in enumerate(voice_indicators) if v == 1]
                if voice_frames:
                    voice_spectral_centroid = np.mean([spectral_centroids[i] for i in voice_frames])
                    voice_stability = 1.0 - np.std([spectral_centroids[i] for i in voice_frames]) / voice_spectral_centroid
                else:
                    voice_spectral_centroid = 0
                    voice_stability = 0
            else:
                voice_spectral_centroid = 0
                voice_stability = 0
            
            analysis["voice_activity"] = {
                "voice_presence_ratio": float(voice_ratio),
                "voice_confidence": float(voice_confidence),
                "likely_contains_vocals": voice_ratio > 0.3,
                "voice_quality": {
                    "spectral_centroid": float(voice_spectral_centroid),
                    "stability": float(voice_stability),
                    "clarity": float(voice_confidence)
                }
            }
            
            # Enhanced Speech Recognition
            if HAS_SPEECH and voice_ratio > 0.2:
                print("   → Enhanced Speech Recognition")
                try:
                    # Multiple recognition attempts with different settings
                    recognition_results = []
                    
                    if HAS_PYDUB:
                        # Try with different audio processing
                        audio_segment = AudioSegment.from_file(audio_path)
                        
                        # Normalize audio for better recognition
                        normalized_audio = audio_segment.normalize()
                        
                        # Try different chunk sizes
                        chunk_duration = min(30, len(y) / sr)  # Max 30 seconds
                        
                        temp_files = []
                        for chunk_start in range(0, int(len(y) / sr), 15):  # 15-second chunks
                            chunk_end = min(chunk_start + chunk_duration, len(y) / sr)
                            
                            chunk_audio = normalized_audio[chunk_start*1000:chunk_end*1000]
                            temp_file = f"temp_chunk_{chunk_start}.wav"
                            chunk_audio.export(temp_file, format="wav")
                            temp_files.append(temp_file)
                            
                            try:
                                with sr.AudioFile(temp_file) as source:
                                    # Adjust for ambient noise
                                    self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                                    audio_data = self.recognizer.record(source)
                                
                                # Try multiple recognition services
                                try:
                                    text = self.recognizer.recognize_google(audio_data, language='en-US')
                                    if text.strip():
                                        recognition_results.append({
                                            "text": text,
                                            "confidence": "medium",
                                            "chunk": f"{chunk_start}-{chunk_end}s",
                                            "service": "google"
                                        })
                                except sr.UnknownValueError:
                                    pass
                                except sr.RequestError:
                                    pass
                                
                            except Exception:
                                pass
                        
                        # Clean up temp files
                        for temp_file in temp_files:
                            if os.path.exists(temp_file):
                                os.remove(temp_file)
                    
                    # Combine recognition results
                    if recognition_results:
                        all_text = " ".join([r["text"] for r in recognition_results])
                        analysis["speech_recognition"] = {
                            "transcribed_text": all_text,
                            "chunks": recognition_results,
                            "total_chunks": len(recognition_results),
                            "confidence": "medium"
                        }
                        
                        # Enhanced text analysis
                        if all_text:
                            print("   → Enhanced Text Analysis")
                            blob = TextBlob(all_text)
                            
                            # Advanced text metrics
                            words = all_text.split()
                            sentences = all_text.split('.')
                            
                            # Language complexity
                            avg_word_length = np.mean([len(word) for word in words]) if words else 0
                            vocabulary_richness = len(set(words)) / len(words) if words else 0
                            
                            # Sentiment analysis
                            sentiment = blob.sentiment
                            
                            # Emotional content analysis (simple keyword-based)
                            emotion_keywords = {
                                'joy': ['happy', 'joy', 'excited', 'wonderful', 'amazing', 'love'],
                                'sadness': ['sad', 'cry', 'tears', 'sorrow', 'grief', 'depressed'],
                                'anger': ['angry', 'mad', 'furious', 'rage', 'hate', 'annoyed'],
                                'fear': ['afraid', 'scared', 'terrified', 'anxiety', 'worry', 'nervous'],
                                'surprise': ['surprised', 'shocked', 'amazed', 'unexpected', 'wow']
                            }
                            
                            emotion_scores = {}
                            text_lower = all_text.lower()
                            for emotion, keywords in emotion_keywords.items():
                                score = sum(1 for keyword in keywords if keyword in text_lower)
                                emotion_scores[emotion] = score / len(words) if words else 0
                            
                            analysis["text_analysis"] = {
                                "word_count": len(words),
                                "sentence_count": len([s for s in sentences if s.strip()]),
                                "average_word_length": float(avg_word_length),
                                "vocabulary_richness": float(vocabulary_richness),
                                "sentiment": {
                                    "polarity": float(sentiment.polarity),
                                    "subjectivity": float(sentiment.subjectivity)
                                },
                                "emotion_analysis": emotion_scores,
                                "dominant_emotion": max(emotion_scores.items(), key=lambda x: x[1])[0] if emotion_scores else "neutral"
                            }
                    else:
                        analysis["speech_recognition"] = {"note": "No clear speech detected in audio"}
                
                except Exception as e:
                    analysis["speech_recognition"] = {"error": str(e)}
            else:
                analysis["speech_recognition"] = {"note": "Voice activity too low for speech recognition"}
            
            # Enhanced Vocal Characteristics
            print("   → Enhanced Vocal Characteristics")
            if voice_ratio > 0.1:
                try:
                    # Pitch analysis
                    pitches, magnitudes = librosa.piptrack(y=y, sr=sr, threshold=0.1)
                    
                    vocal_pitches = []
                    pitch_confidences = []
                    
                    for t in range(pitches.shape[1]):
                        # Get multiple pitch candidates
                        frame_pitches = pitches[:, t]
                        frame_magnitudes = magnitudes[:, t]
                        
                        # Find peaks
                        valid_indices = frame_magnitudes > 0.1
                        if np.any(valid_indices):
                            valid_pitches = frame_pitches[valid_indices]
                            valid_magnitudes = frame_magnitudes[valid_indices]
                            
                            # Focus on vocal range
                            vocal_range_mask = (valid_pitches > 80) & (valid_pitches < 1000)
                            if np.any(vocal_range_mask):
                                vocal_range_pitches = valid_pitches[vocal_range_mask]
                                vocal_range_magnitudes = valid_magnitudes[vocal_range_mask]
                                
                                # Select strongest pitch in vocal range
                                strongest_idx = np.argmax(vocal_range_magnitudes)
                                pitch = vocal_range_pitches[strongest_idx]
                                confidence = vocal_range_magnitudes[strongest_idx]
                                
                                vocal_pitches.append(pitch)
                                pitch_confidences.append(confidence)
                    
                    if vocal_pitches:
                        # Vocal range analysis
                        min_pitch = float(np.min(vocal_pitches))
                        max_pitch = float(np.max(vocal_pitches))
                        mean_pitch = float(np.mean(vocal_pitches))
                        median_pitch = float(np.median(vocal_pitches))
                        
                        # Pitch stability
                        pitch_variance = float(np.var(vocal_pitches))
                        pitch_stability = 1.0 / (1.0 + pitch_variance / (mean_pitch ** 2))
                        
                        # Vibrato detection (pitch oscillation)
                        if len(vocal_pitches) > 10:
                            pitch_diff = np.diff(vocal_pitches)
                            vibrato_rate = np.sum(np.abs(pitch_diff) > 5) / len(pitch_diff)
                        else:
                            vibrato_rate = 0
                        
                        # Voice classification
                        if mean_pitch < 165:
                            voice_type = "bass/baritone"
                        elif mean_pitch < 220:
                            voice_type = "tenor/alto"
                        elif mean_pitch < 330:
                            voice_type = "mezzo-soprano"
                        else:
                            voice_type = "soprano"
                        
                        analysis["vocal_characteristics"] = {
                            "pitch_analysis": {
                                "min_pitch": min_pitch,
                                "max_pitch": max_pitch,
                                "mean_pitch": mean_pitch,
                                "median_pitch": median_pitch,
                                "range_semitones": float(12 * np.log2(max_pitch / min_pitch)) if min_pitch > 0 else 0
                            },
                            "voice_quality": {
                                "stability": float(pitch_stability),
                                "vibrato_rate": float(vibrato_rate),
                                "pitch_variance": pitch_variance
                            },
                            "voice_classification": {
                                "estimated_type": voice_type,
                                "confidence": float(np.mean(pitch_confidences))
                            },
                            "frames_analyzed": len(vocal_pitches)
                        }
                    else:
                        analysis["vocal_characteristics"] = {"note": "No clear vocal pitch detected"}
                
                except Exception as e:
                    analysis["vocal_characteristics"] = {"error": str(e)}
            
        except Exception as e:
            analysis["error"] = str(e)
        
        return analysis
    
    def _analyze_sonic_spectral_advanced(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Enhanced sonic & spectral analysis"""
        print("🔊 Advanced Sonic & Spectral Analysis...")
        
        analysis = {}
        
        try:
            # Enhanced Spectral Features
            print("   → Enhanced Spectral Features")
            
            # Comprehensive spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
            
            # Temporal features
            rms = librosa.feature.rms(y=y)[0]
            
            analysis["spectral_features"] = {
                "spectral_centroid": {
                    "mean": float(np.mean(spectral_centroids)),
                    "std": float(np.std(spectral_centroids)),
                    "max": float(np.max(spectral_centroids)),
                    "min": float(np.min(spectral_centroids)),
                    "range": float(np.max(spectral_centroids) - np.min(spectral_centroids))
                },
                "spectral_rolloff": {
                    "mean": float(np.mean(spectral_rolloff)),
                    "std": float(np.std(spectral_rolloff)),
                    "max": float(np.max(spectral_rolloff)),
                    "min": float(np.min(spectral_rolloff))
                },
                "spectral_bandwidth": {
                    "mean": float(np.mean(spectral_bandwidth)),
                    "std": float(np.std(spectral_bandwidth))
                },
                "spectral_contrast": {
                    "mean": float(np.mean(spectral_contrast)),
                    "std": float(np.std(spectral_contrast))
                },
                "spectral_flatness": {
                    "mean": float(np.mean(spectral_flatness)),
                    "std": float(np.std(spectral_flatness))
                },
                "zero_crossing_rate": {
                    "mean": float(np.mean(zero_crossing_rate)),
                    "std": float(np.std(zero_crossing_rate))
                }
            }
            
            # Advanced Frequency Analysis
            print("   → Advanced Frequency Analysis")
            
            # Multi-resolution spectral analysis
            stft = librosa.stft(y)
            magnitude = np.abs(stft)
            
            # Frequency band analysis
            freqs = librosa.fft_frequencies(sr=sr)
            
            # Define frequency bands
            freq_bands = {
                "sub_bass": (20, 60),
                "bass": (60, 250),
                "low_mid": (250, 500),
                "mid": (500, 2000),
                "high_mid": (2000, 4000),
                "presence": (4000, 6000),
                "brilliance": (6000, 20000)
            }
            
            band_energies = {}
            for band_name, (low_freq, high_freq) in freq_bands.items():
                band_mask = (freqs >= low_freq) & (freqs <= high_freq)
                if np.any(band_mask):
                    band_energy = np.mean(magnitude[band_mask, :])
                    band_energies[band_name] = float(band_energy)
                else:
                    band_energies[band_name] = 0.0
            
            # Spectral peaks
            spectrum = np.mean(magnitude, axis=1)
            peak_indices = []
            for i in range(1, len(spectrum)-1):
                if (spectrum[i] > spectrum[i-1] and 
                    spectrum[i] > spectrum[i+1] and
                    spectrum[i] > np.max(spectrum) * 0.1):
                    peak_indices.append(i)
            
            peak_frequencies = [freqs[i] for i in peak_indices[:10]]
            
            analysis["frequency_analysis"] = {
                "dominant_frequency": float(freqs[np.argmax(spectrum)]),
                "peak_frequencies": [float(f) for f in peak_frequencies],
                "frequency_bands": band_energies,
                "spectral_entropy": float(-np.sum(spectrum * np.log2(spectrum + 1e-10)) / len(spectrum)),
                "spectral_slope": float(np.polyfit(freqs, spectrum, 1)[0])
            }
            
            # Enhanced Dynamic Analysis
            print("   → Enhanced Dynamic Analysis")
            
            # Multiple dynamic measures
            rms_db = 20 * np.log10(rms + 1e-10)
            
            # Peak detection
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(rms, height=np.mean(rms))
            
            # Dynamic range in different time scales
            dynamic_range_db = float(20 * np.log10(np.max(rms) / (np.min(rms) + 1e-10)))
            crest_factor = float(np.max(np.abs(y)) / (np.mean(rms) + 1e-10))
            
            # Loudness variation over time
            loudness_variation = float(np.std(rms_db))
            
            # Attack and decay analysis
            if len(peaks) > 0:
                attack_times = []
                decay_times = []
                
                for peak_idx in peaks[:10]:  # Analyze first 10 peaks
                    # Attack time (rise to peak)
                    if peak_idx > 10:
                        pre_peak = rms[max(0, peak_idx-10):peak_idx]
                        if len(pre_peak) > 0:
                            attack_start = np.argmin(pre_peak)
                            attack_time = (len(pre_peak) - attack_start) * 512 / sr  # frames to seconds
                            attack_times.append(attack_time)
                    
                    # Decay time (fall from peak)
                    if peak_idx < len(rms) - 10:
                        post_peak = rms[peak_idx:min(len(rms), peak_idx+10)]
                        if len(post_peak) > 1:
                            decay_time = 10 * 512 / sr  # Simplified decay time
                            decay_times.append(decay_time)
                
                avg_attack_time = float(np.mean(attack_times)) if attack_times else 0
                avg_decay_time = float(np.mean(decay_times)) if decay_times else 0
            else:
                avg_attack_time = 0
                avg_decay_time = 0
            
            analysis["dynamics"] = {
                "rms_energy": {
                    "mean": float(np.mean(rms)),
                    "std": float(np.std(rms)),
                    "max": float(np.max(rms)),
                    "min": float(np.min(rms))
                },
                "dynamic_range_db": dynamic_range_db,
                "crest_factor": crest_factor,
                "loudness_variation": loudness_variation,
                "peaks_detected": len(peaks),
                "attack_decay": {
                    "average_attack_time": avg_attack_time,
                    "average_decay_time": avg_decay_time
                }
            }
            
            # Enhanced Harmonic-Percussive Analysis
            print("   → Enhanced Harmonic/Percussive Analysis")
            
            try:
                # Separate harmonic and percussive components
                y_harmonic, y_percussive = librosa.effects.hpss(y, margin=(1.0, 5.0))
                
                # Energy analysis
                harmonic_energy = float(np.sum(y_harmonic**2))
                percussive_energy = float(np.sum(y_percussive**2))
                total_energy = harmonic_energy + percussive_energy
                
                # Spectral characteristics of each component
                harmonic_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y_harmonic, sr=sr)))
                percussive_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y_percussive, sr=sr)))
                
                # Temporal characteristics
                harmonic_zcr = float(np.mean(librosa.feature.zero_crossing_rate(y_harmonic)))
                percussive_zcr = float(np.mean(librosa.feature.zero_crossing_rate(y_percussive)))
                
                analysis["harmonic_percussive"] = {
                    "energy_ratios": {
                        "harmonic_ratio": float(harmonic_energy / total_energy) if total_energy > 0 else 0,
                        "percussive_ratio": float(percussive_energy / total_energy) if total_energy > 0 else 0
                    },
                    "spectral_characteristics": {
                        "harmonic_centroid": harmonic_centroid,
                        "percussive_centroid": percussive_centroid,
                        "centroid_ratio": float(percussive_centroid / harmonic_centroid) if harmonic_centroid > 0 else 0
                    },
                    "temporal_characteristics": {
                        "harmonic_zcr": harmonic_zcr,
                        "percussive_zcr": percussive_zcr,
                        "zcr_ratio": float(percussive_zcr / harmonic_zcr) if harmonic_zcr > 0 else 0
                    }
                }
                
            except Exception as e:
                analysis["harmonic_percussive"] = {"error": str(e)}
            
            # Audio Quality Analysis
            print("   → Audio Quality Analysis")
            
            # SNR estimation
            signal_power = np.mean(y**2)
            noise_estimate = np.min(rms)  # Simplified noise estimation
            snr_db = float(10 * np.log10(signal_power / (noise_estimate**2 + 1e-10)))
            
            # Clipping detection
            clipping_threshold = 0.95
            clipped_samples = np.sum(np.abs(y) > clipping_threshold)
            clipping_percentage = float(clipped_samples / len(y) * 100)
            
            # Silence detection
            silence_threshold = np.max(rms) * 0.01
            silence_frames = np.sum(rms < silence_threshold)
            silence_percentage = float(silence_frames / len(rms) * 100)
            
            analysis["audio_quality"] = {
                "estimated_snr_db": snr_db,
                "clipping": {
                    "clipped_samples": int(clipped_samples),
                    "clipping_percentage": clipping_percentage
                },
                "silence": {
                    "silence_frames": int(silence_frames),
                    "silence_percentage": silence_percentage
                },
                "overall_quality": "high" if snr_db > 40 and clipping_percentage < 0.1 else 
                                 "medium" if snr_db > 20 and clipping_percentage < 1.0 else "low"
            }
            
        except Exception as e:
            analysis["error"] = str(e)
        
        return analysis
    
    def _analyze_advanced_features(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Advanced ML-based feature analysis"""
        print("🤖 Advanced ML Features Analysis...")
        
        analysis = {}
        
        try:
            # Genre Classification
            if self.genre_classifier is not None and HAS_SKLEARN:
                print("   → Genre Classification")
                
                # Extract features for genre classification
                features = self._extract_genre_features(y, sr)
                if features is not None:
                    features_scaled = self.genre_scaler.transform([features])
                    genre_probabilities = self.genre_classifier.predict_proba(features_scaled)[0]
                    predicted_genre_idx = np.argmax(genre_probabilities)
                    
                    genres = ['rock', 'pop', 'classical', 'jazz', 'electronic']
                    predicted_genre = genres[predicted_genre_idx]
                    confidence = float(genre_probabilities[predicted_genre_idx])
                    
                    genre_scores = {genre: float(prob) for genre, prob in zip(genres, genre_probabilities)}
                    
                    analysis["genre_classification"] = {
                        "predicted_genre": predicted_genre,
                        "confidence": confidence,
                        "all_scores": genre_scores,
                        "features_used": len(features)
                    }
            
            # Mood Detection
            if self.mood_classifier is not None and HAS_SKLEARN:
                print("   → Mood Detection")
                
                # Extract features for mood classification
                mood_features = self._extract_mood_features(y, sr)
                if mood_features is not None:
                    mood_features_scaled = self.mood_scaler.transform([mood_features])
                    mood_probabilities = self.mood_classifier.predict_proba(mood_features_scaled)[0]
                    predicted_mood_idx = np.argmax(mood_probabilities)
                    
                    moods = ['happy', 'sad', 'energetic', 'calm', 'aggressive']
                    predicted_mood = moods[predicted_mood_idx]
                    mood_confidence = float(mood_probabilities[predicted_mood_idx])
                    
                    mood_scores = {mood: float(prob) for mood, prob in zip(moods, mood_probabilities)}
                    
                    analysis["mood_detection"] = {
                        "predicted_mood": predicted_mood,
                        "confidence": mood_confidence,
                        "all_scores": mood_scores,
                        "emotional_valence": self._calculate_valence(mood_scores),
                        "energy_level": self._calculate_energy_level(mood_scores)
                    }
            
            # Instrument Detection
            print("   → Instrument Detection")
            instrument_analysis = self._detect_instruments(y, sr)
            analysis["instrument_detection"] = instrument_analysis
            
            # Audio Complexity Analysis
            print("   → Audio Complexity Analysis")
            complexity_analysis = self._analyze_audio_complexity(y, sr)
            analysis["complexity_analysis"] = complexity_analysis
            
            # Texture and Timbre Analysis
            print("   → Texture and Timbre Analysis")
            texture_analysis = self._analyze_texture_timbre(y, sr)
            analysis["texture_timbre"] = texture_analysis
            
            # Rhythmic Pattern Analysis
            print("   → Rhythmic Pattern Analysis")
            rhythm_patterns = self._analyze_rhythm_patterns(y, sr)
            analysis["rhythm_patterns"] = rhythm_patterns
            
        except Exception as e:
            analysis["error"] = str(e)
        
        return analysis
    
    def _extract_genre_features(self, y: np.ndarray, sr: int) -> Optional[List[float]]:
        """Extract features for genre classification"""
        try:
            # Tempo
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            
            # Spectral features
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
            zcr = np.mean(librosa.feature.zero_crossing_rate(y))
            
            # MFCC
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_mean = np.mean(mfcc[0])  # First MFCC coefficient
            
            # Energy
            rms = librosa.feature.rms(y=y)
            energy = np.mean(rms)
            
            return [float(tempo), float(spectral_centroid), float(spectral_rolloff), 
                   float(zcr), float(mfcc_mean), float(energy)]
        except:
            return None
    
    def _extract_mood_features(self, y: np.ndarray, sr: int) -> Optional[List[float]]:
        """Extract features for mood classification"""
        try:
            # Tempo
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            
            # Brightness (spectral centroid)
            brightness = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            
            # Energy
            energy = np.mean(librosa.feature.rms(y=y))
            
            # Valence (estimated from harmony)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            chroma_mean = np.mean(chroma, axis=1)
            
            # Major scale profile for valence estimation
            major_profile = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1])
            valence = np.corrcoef(chroma_mean, major_profile)[0, 1]
            valence = valence if not np.isnan(valence) else 0.5
            
            # Major key likelihood
            major_correlation = valence
            
            return [float(tempo), float(brightness), float(energy), 
                   float(valence), float(major_correlation)]
        except:
            return None
    
    def _calculate_valence(self, mood_scores: Dict[str, float]) -> float:
        """Calculate emotional valence from mood scores"""
        positive_moods = ['happy', 'energetic', 'calm']
        negative_moods = ['sad', 'aggressive']
        
        positive_score = sum(mood_scores.get(mood, 0) for mood in positive_moods)
        negative_score = sum(mood_scores.get(mood, 0) for mood in negative_moods)
        
        # Normalize to [-1, 1] range
        total = positive_score + negative_score
        if total > 0:
            return (positive_score - negative_score) / total
        return 0.0
    
    def _calculate_energy_level(self, mood_scores: Dict[str, float]) -> float:
        """Calculate energy level from mood scores"""
        high_energy_moods = ['energetic', 'aggressive', 'happy']
        low_energy_moods = ['calm', 'sad']
        
        high_energy = sum(mood_scores.get(mood, 0) for mood in high_energy_moods)
        low_energy = sum(mood_scores.get(mood, 0) for mood in low_energy_moods)
        
        total = high_energy + low_energy
        if total > 0:
            return high_energy / total
        return 0.5
    
    def _detect_instruments(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Detect likely instruments in the audio"""
        try:
            # Separate harmonic and percussive components
            y_harmonic, y_percussive = librosa.effects.hpss(y)
            
            # Instrument detection based on spectral characteristics
            instruments = {}
            
            # Drum detection (high percussive energy)
            perc_energy = np.sum(y_percussive**2)
            harm_energy = np.sum(y_harmonic**2)
            total_energy = perc_energy + harm_energy
            
            drums_likelihood = perc_energy / total_energy if total_energy > 0 else 0
            instruments["drums"] = {
                "likelihood": float(drums_likelihood),
                "confidence": "high" if drums_likelihood > 0.3 else "medium" if drums_likelihood > 0.1 else "low"
            }
            
            # Piano detection (harmonic content with specific spectral profile)
            if harm_energy > 0:
                spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y_harmonic, sr=sr))
                spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y_harmonic, sr=sr))
                
                # Piano typically has moderate spectral centroid and good harmonic structure
                piano_score = 0
                if 800 < spectral_centroid < 3000:
                    piano_score += 0.3
                if 2000 < spectral_rolloff < 8000:
                    piano_score += 0.3
                
                # Check for harmonic structure
                chroma = librosa.feature.chroma_stft(y=y_harmonic, sr=sr)
                harmonic_clarity = np.max(np.mean(chroma, axis=1)) / np.mean(chroma)
                if harmonic_clarity > 2:
                    piano_score += 0.4
                
                instruments["piano"] = {
                    "likelihood": float(piano_score),
                    "confidence": "high" if piano_score > 0.7 else "medium" if piano_score > 0.4 else "low"
                }
            
            # Guitar detection (moderate harmonic content with specific frequency characteristics)
            if harm_energy > 0:
                # Guitar typically has energy in specific frequency bands
                stft = librosa.stft(y_harmonic)
                magnitude = np.abs(stft)
                freqs = librosa.fft_frequencies(sr=sr)
                
                # Guitar fundamental frequencies range
                guitar_freq_mask = (freqs >= 80) & (freqs <= 1000)
                guitar_energy = np.mean(magnitude[guitar_freq_mask, :]) if np.any(guitar_freq_mask) else 0
                
                total_spectral_energy = np.mean(magnitude)
                guitar_ratio = guitar_energy / total_spectral_energy if total_spectral_energy > 0 else 0
                
                instruments["guitar"] = {
                    "likelihood": float(min(guitar_ratio * 2, 1.0)),  # Scale and cap at 1.0
                    "confidence": "medium"
                }
            
            # Synthesizer detection (electronic characteristics)
            spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=y))
            synth_likelihood = spectral_flatness  # High flatness suggests electronic/synthetic origin
            
            instruments["synthesizer"] = {
                "likelihood": float(synth_likelihood),
                "confidence": "high" if synth_likelihood > 0.8 else "medium" if synth_likelihood > 0.5 else "low"
            }
            
            return {
                "detected_instruments": instruments,
                "dominant_instrument": max(instruments.items(), key=lambda x: x[1]["likelihood"])[0],
                "harmonic_percussive_ratio": float(harm_energy / perc_energy) if perc_energy > 0 else float('inf')
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _analyze_audio_complexity(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Analyze overall audio complexity"""
        try:
            # Spectral complexity
            stft = librosa.stft(y)
            magnitude = np.abs(stft)
            
            # Spectral entropy (measure of spectral complexity)
            spectrum = np.mean(magnitude, axis=1)
            normalized_spectrum = spectrum / np.sum(spectrum)
            spectral_entropy = -np.sum(normalized_spectrum * np.log2(normalized_spectrum + 1e-10))
            
            # Temporal complexity (rate of change)
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            temporal_variation = np.std(spectral_centroid) / np.mean(spectral_centroid)
            
            # Harmonic complexity
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            harmonic_variance = np.mean(np.var(chroma, axis=1))
            
            # Rhythmic complexity (onset density and regularity)
            onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
            onset_density = len(onset_frames) / (len(y) / sr)  # onsets per second
            
            if len(onset_frames) > 2:
                onset_times = librosa.frames_to_time(onset_frames, sr=sr)
                onset_intervals = np.diff(onset_times)
                rhythmic_regularity = 1.0 / (1.0 + np.std(onset_intervals) / np.mean(onset_intervals))
            else:
                rhythmic_regularity = 0
            
            # Overall complexity score
            complexity_factors = [
                ("spectral", float(spectral_entropy / 10)),  # Normalize
                ("temporal", float(min(temporal_variation, 1.0))),
                ("harmonic", float(min(harmonic_variance * 10, 1.0))),
                ("rhythmic", float(1.0 - rhythmic_regularity))
            ]
            
            overall_complexity = np.mean([score for _, score in complexity_factors])
            
            return {
                "overall_complexity": float(overall_complexity),
                "complexity_breakdown": dict(complexity_factors),
                "spectral_entropy": float(spectral_entropy),
                "temporal_variation": float(temporal_variation),
                "harmonic_variance": float(harmonic_variance),
                "onset_density": float(onset_density),
                "rhythmic_regularity": float(rhythmic_regularity),
                "complexity_level": "high" if overall_complexity > 0.7 else 
                                 "medium" if overall_complexity > 0.4 else "low"
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _analyze_texture_timbre(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Analyze audio texture and timbre characteristics"""
        try:
            # MFCC analysis for timbre
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            
            # Timbre features
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_std = np.std(mfcc, axis=1)
            
            # Brightness (related to spectral centroid)
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            brightness = float(np.mean(spectral_centroid))
            
            # Roughness (spectral irregularity)
            spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            roughness = float(np.mean(np.std(spectral_contrast, axis=1)))
            
            # Warmth (low-frequency content)
            stft = librosa.stft(y)
            magnitude = np.abs(stft)
            freqs = librosa.fft_frequencies(sr=sr)
            
            low_freq_mask = freqs <= 500
            mid_freq_mask = (freqs > 500) & (freqs <= 4000)
            
            low_energy = np.mean(magnitude[low_freq_mask, :]) if np.any(low_freq_mask) else 0
            mid_energy = np.mean(magnitude[mid_freq_mask, :]) if np.any(mid_freq_mask) else 0
            
            warmth = float(low_energy / (mid_energy + 1e-10))
            
            # Texture density (number of simultaneous events)
            onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
            onset_density = len(onset_frames) / (len(y) / sr)
            
            # Spectral flux (rate of spectral change)
            spectral_flux = np.mean(np.diff(magnitude, axis=1)**2)
            
            return {
                "timbre_features": {
                    "mfcc_coefficients": [float(x) for x in mfcc_mean[:5]],  # First 5 MFCCs
                    "mfcc_variation": [float(x) for x in mfcc_std[:5]],
                    "brightness": brightness,
                    "roughness": roughness,
                    "warmth": warmth
                },
                "texture_characteristics": {
                    "density": float(onset_density),
                    "spectral_flux": float(spectral_flux),
                    "temporal_stability": float(1.0 / (1.0 + np.std(spectral_centroid) / np.mean(spectral_centroid)))
                },
                "perceptual_qualities": {
                    "perceived_brightness": "bright" if brightness > 2000 else "moderate" if brightness > 1000 else "dark",
                    "perceived_texture": "dense" if onset_density > 5 else "moderate" if onset_density > 2 else "sparse",
                    "perceived_warmth": "warm" if warmth > 1.5 else "neutral" if warmth > 0.5 else "cool"
                }
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _analyze_rhythm_patterns(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Analyze rhythmic patterns and groove characteristics"""
        try:
            # Beat tracking
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            beat_times = librosa.frames_to_time(beats, sr=sr)
            
            analysis = {
                "tempo_analysis": {
                    "estimated_tempo": float(tempo),
                    "beats_detected": len(beats),
                    "beat_consistency": 0
                },
                "groove_analysis": {},
                "rhythmic_features": {}
            }
            
            if len(beat_times) > 3:
                # Beat consistency
                beat_intervals = np.diff(beat_times)
                tempo_stability = 1.0 / (1.0 + np.std(beat_intervals) / np.mean(beat_intervals))
                analysis["tempo_analysis"]["beat_consistency"] = float(tempo_stability)
                
                # Groove analysis
                # Swing detection (timing deviations)
                expected_interval = 60.0 / tempo
                timing_deviations = beat_intervals - expected_interval
                swing_factor = float(np.std(timing_deviations))
                
                # Syncopation detection (off-beat emphasis)
                # Analyze energy at beat positions vs off-beat positions
                rms = librosa.feature.rms(y=y)[0]
                
                on_beat_energy = []
                off_beat_energy = []
                
                for i, beat_time in enumerate(beat_times[:-1]):
                    # On-beat energy
                    beat_frame = int(beat_time * sr / 512)  # Convert to frame index
                    if beat_frame < len(rms):
                        on_beat_energy.append(rms[beat_frame])
                    
                    # Off-beat energy (halfway between beats)
                    if i < len(beat_times) - 1:
                        off_beat_time = beat_time + (beat_times[i+1] - beat_time) / 2
                        off_beat_frame = int(off_beat_time * sr / 512)
                        if off_beat_frame < len(rms):
                            off_beat_energy.append(rms[off_beat_frame])
                
                if on_beat_energy and off_beat_energy:
                    syncopation = float(np.mean(off_beat_energy) / np.mean(on_beat_energy))
                else:
                    syncopation = 0
                
                analysis["groove_analysis"] = {
                    "swing_factor": swing_factor,
                    "syncopation": syncopation,
                    "groove_type": "swing" if swing_factor > 0.1 else "straight",
                    "rhythmic_emphasis": "syncopated" if syncopation > 0.8 else "on-beat"
                }
                
                # Advanced rhythmic features
                onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
                if len(onset_frames) > 0:
                    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
                    
                    # Microtiming analysis
                    if len(beat_times) > 1 and len(onset_times) > 1:
                        # Find onsets that are close to beats
                        beat_aligned_onsets = []
                        for beat_time in beat_times:
                            closest_onset_idx = np.argmin(np.abs(onset_times - beat_time))
                            closest_onset_time = onset_times[closest_onset_idx]
                            if abs(closest_onset_time - beat_time) < 0.1:  # Within 100ms
                                beat_aligned_onsets.append(closest_onset_time - beat_time)
                        
                        if beat_aligned_onsets:
                            microtiming_precision = float(1.0 / (1.0 + np.std(beat_aligned_onsets)))
                        else:
                            microtiming_precision = 0
                    else:
                        microtiming_precision = 0
                    
                    # Onset density variation
                    onset_density_over_time = []
                    window_size = 4.0  # 4-second windows
                    for start_time in np.arange(0, len(y)/sr - window_size, window_size/2):
                        end_time = start_time + window_size
                        window_onsets = onset_times[(onset_times >= start_time) & (onset_times < end_time)]
                        density = len(window_onsets) / window_size
                        onset_density_over_time.append(density)
                    
                    density_variation = float(np.std(onset_density_over_time)) if onset_density_over_time else 0
                    
                    analysis["rhythmic_features"] = {
                        "total_onsets": len(onset_times),
                        "onset_density": float(len(onset_times) / (len(y) / sr)),
                        "microtiming_precision": microtiming_precision,
                        "density_variation": density_variation,
                        "rhythmic_complexity": float(density_variation + (1.0 - microtiming_precision))
                    }
            
            return analysis
            
        except Exception as e:
            return {"error": str(e)}
    
    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a concise summary of the analysis results"""
        summary = {
            "file_info": {
                "duration": results["metadata"]["duration"],
                "sample_rate": results["metadata"]["sample_rate"]
            }
        }
        
        # Musical summary
        if "musical_structural" in results and "rhythm" in results["musical_structural"]:
            rhythm = results["musical_structural"]["rhythm"]
            summary["musical"] = {
                "tempo": rhythm.get("estimated_tempo", 0),
                "rhythm_regularity": rhythm.get("rhythm_regularity", 0)
            }
            
            if "harmony" in results["musical_structural"]:
                harmony = results["musical_structural"]["harmony"]
                summary["musical"].update({
                    "key": harmony.get("estimated_key", "unknown"),
                    "mode": harmony.get("mode", "unknown")
                })
        
        # Vocal summary
        if "lyrical_vocal" in results and "voice_activity" in results["lyrical_vocal"]:
            voice = results["lyrical_vocal"]["voice_activity"]
            summary["vocal"] = {
                "has_vocals": voice.get("likely_contains_vocals", False),
                "voice_confidence": voice.get("voice_confidence", 0)
            }
        
        # Spectral summary
        if "sonic_spectral" in results and "spectral_features" in results["sonic_spectral"]:
            spectral = results["sonic_spectral"]["spectral_features"]
            if "spectral_centroid" in spectral:
                summary["spectral"] = {
                    "brightness": spectral["spectral_centroid"].get("mean", 0),
                    "energy": spectral.get("spectral_contrast", {}).get("mean", 0)
                }
        
        # Advanced features summary
        if "advanced_features" in results:
            advanced = results["advanced_features"]
            if "genre_classification" in advanced:
                summary["genre"] = {
                    "predicted": advanced["genre_classification"].get("predicted_genre", "unknown"),
                    "confidence": advanced["genre_classification"].get("confidence", 0)
                }
            
            if "mood_detection" in advanced:
                summary["mood"] = {
                    "predicted": advanced["mood_detection"].get("predicted_mood", "unknown"),
                    "confidence": advanced["mood_detection"].get("confidence", 0),
                    "valence": advanced["mood_detection"].get("emotional_valence", 0),
                    "energy": advanced["mood_detection"].get("energy_level", 0)
                }
        
        return summary
    
    def save_results(self, output_path: str, include_raw_data: bool = False) -> bool:
        """Save analysis results to JSON file with enhanced format"""
        try:
            def convert_numpy_types(obj):
                """Convert numpy types to native Python types for JSON serialization"""
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {key: convert_numpy_types(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                else:
                    return obj
            
            # Prepare output data
            output_data = {
                "analysis_metadata": {
                    "agent_version": "advanced",
                    "analysis_timestamp": datetime.now().isoformat(),
                    "features_included": list(self.results.keys()) if self.results else [],
                    "ml_models_used": {
                        "genre_classification": self.genre_classifier is not None,
                        "mood_detection": self.mood_classifier is not None
                    }
                },
                "analysis_results": convert_numpy_types(self.results),
                "analysis_history": convert_numpy_types(self.analysis_history[-10:])  # Last 10 analyses
            }
            
            # Add raw data if requested (for debugging/research)
            if include_raw_data:
                output_data["raw_features"] = {
                    "note": "Raw feature data included for research purposes",
                    "features": {}  # Would include raw spectral data, etc.
                }
            
            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Results saved to: {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving results: {e}")
            return False
    
    def print_summary(self, detailed: bool = False):
        """Print a formatted summary of the analysis results"""
        if not self.results:
            print("❌ No analysis results available")
            return
        
        print("\n" + "="*80)
        print("🎵 ADVANCED AUDIO ANALYSIS SUMMARY")
        print("="*80)
        
        # File information
        metadata = self.results.get("metadata", {})
        print(f"\n📁 File: {Path(metadata.get('file', 'unknown')).name}")
        print(f"⏱️  Duration: {metadata.get('duration', 0):.2f} seconds")
        print(f"🔊 Sample Rate: {metadata.get('sample_rate', 0)} Hz")
        print(f"📊 Analysis Time: {metadata.get('analysis_time', 'unknown')}")
        
        # Musical Analysis
        musical = self.results.get("musical_structural", {})
        if musical:
            print(f"\n🎼 MUSICAL ANALYSIS")
            
            rhythm = musical.get("rhythm", {})
            if rhythm:
                print(f"   Tempo: {rhythm.get('estimated_tempo', 0):.1f} BPM")
                print(f"   Rhythm Regularity: {rhythm.get('rhythm_regularity', 0):.2f}")
                print(f"   Beat Strength: {rhythm.get('beat_strength', 0):.3f}")
            
            harmony = musical.get("harmony", {})
            if harmony:
                print(f"   Key: {harmony.get('estimated_key', 'unknown')} {harmony.get('mode', '')}")
                print(f"   Key Confidence: {harmony.get('key_confidence', 0):.2f}")
                print(f"   Harmonic Complexity: {harmony.get('harmonic_complexity', 0):.3f}")
        
        # Vocal Analysis
        vocal = self.results.get("lyrical_vocal", {})
        if vocal:
            print(f"\n🎤 VOCAL ANALYSIS")
            
            voice_activity = vocal.get("voice_activity", {})
            if voice_activity:
                has_vocals = voice_activity.get("likely_contains_vocals", False)
                confidence = voice_activity.get("voice_confidence", 0)
                print(f"   Contains Vocals: {'Yes' if has_vocals else 'No'} (confidence: {confidence:.2f})")
            
            if vocal.get("speech_recognition") and "transcribed_text" in vocal["speech_recognition"]:
                text = vocal["speech_recognition"]["transcribed_text"]
                print(f"   Transcribed Text: \"{text[:100]}...\" " if len(text) > 100 else f"   Transcribed Text: \"{text}\"")
            
            vocal_chars = vocal.get("vocal_characteristics", {})
            if vocal_chars and "pitch_analysis" in vocal_chars:
                pitch = vocal_chars["pitch_analysis"]
                print(f"   Vocal Range: {pitch.get('min_pitch', 0):.1f} - {pitch.get('max_pitch', 0):.1f} Hz")
                voice_type = vocal_chars.get("voice_classification", {}).get("estimated_type", "unknown")
                print(f"   Voice Type: {voice_type}")
        
        # Spectral Analysis
        spectral = self.results.get("sonic_spectral", {})
        if spectral:
            print(f"\n🔊 SPECTRAL ANALYSIS")
            
            features = spectral.get("spectral_features", {})
            if features and "spectral_centroid" in features:
                brightness = features["spectral_centroid"].get("mean", 0)
                print(f"   Brightness: {brightness:.1f} Hz")
            
            quality = spectral.get("audio_quality", {})
            if quality:
                print(f"   Audio Quality: {quality.get('overall_quality', 'unknown')}")
                snr = quality.get("estimated_snr_db", 0)
                print(f"   Signal-to-Noise Ratio: {snr:.1f} dB")
        
        # Advanced Features
        advanced = self.results.get("advanced_features", {})
        if advanced:
            print(f"\n🤖 ADVANCED ANALYSIS")
            
            genre = advanced.get("genre_classification", {})
            if genre:
                predicted_genre = genre.get("predicted_genre", "unknown")
                confidence = genre.get("confidence", 0)
                print(f"   Predicted Genre: {predicted_genre} (confidence: {confidence:.2f})")
            
            mood = advanced.get("mood_detection", {})
            if mood:
                predicted_mood = mood.get("predicted_mood", "unknown")
                confidence = mood.get("confidence", 0)
                valence = mood.get("emotional_valence", 0)
                energy = mood.get("energy_level", 0)
                print(f"   Predicted Mood: {predicted_mood} (confidence: {confidence:.2f})")
                print(f"   Emotional Valence: {valence:.2f} (negative ← 0 → positive)")
                print(f"   Energy Level: {energy:.2f} (low ← 0.5 → high)")
            
            instruments = advanced.get("instrument_detection", {})
            if instruments and "detected_instruments" in instruments:
                dominant = instruments.get("dominant_instrument", "unknown")
                print(f"   Dominant Instrument: {dominant}")
            
            complexity = advanced.get("complexity_analysis", {})
            if complexity:
                level = complexity.get("complexity_level", "unknown")
                score = complexity.get("overall_complexity", 0)
                print(f"   Complexity: {level} (score: {score:.2f})")
        
        # Detailed analysis
        if detailed:
            print(f"\n📊 DETAILED METRICS")
            
            # Show more detailed breakdown
            if musical and "structure" in musical:
                structure = musical["structure"]
                segments = structure.get("total_segments", 0)
                complexity = structure.get("structural_complexity", 0)
                print(f"   Structural Segments: {segments}")
                print(f"   Structural Complexity: {complexity:.3f}")
            
            if advanced and "texture_timbre" in advanced:
                texture = advanced["texture_timbre"]
                if "perceptual_qualities" in texture:
                    qualities = texture["perceptual_qualities"]
                    print(f"   Perceived Brightness: {qualities.get('perceived_brightness', 'unknown')}")
                    print(f"   Perceived Texture: {qualities.get('perceived_texture', 'unknown')}")
                    print(f"   Perceived Warmth: {qualities.get('perceived_warmth', 'unknown')}")
        
        print("\n" + "="*80)
    
    def start_streaming(self, chunk_duration: float = 1.0):
        """Start real-time audio streaming analysis (placeholder)"""
        if not self.enable_streaming:
            print("❌ Streaming not enabled. Initialize with enable_streaming=True")
            return
        
        print("🎙️ Starting real-time audio analysis...")
        print("⚠️  Note: Real-time streaming requires additional audio input setup")
        # This would integrate with PyAudio or similar for real-time analysis
        # Implementation would depend on specific use case
    
    def stop_streaming(self):
        """Stop real-time audio streaming"""
        if self.is_streaming:
            self.is_streaming = False
            print("🛑 Streaming stopped")
    
    def batch_analyze(self, audio_files: List[str], output_dir: str = "batch_results") -> Dict[str, Any]:
        """Analyze multiple audio files in batch"""
        print(f"\n🔄 Batch Analysis: {len(audio_files)} files")
        
        results = {}
        os.makedirs(output_dir, exist_ok=True)
        
        for i, audio_path in enumerate(audio_files):
            print(f"\n[{i+1}/{len(audio_files)}] Processing: {Path(audio_path).name}")
            
            try:
                file_results = self.analyze_audio(audio_path, include_advanced=True)
                results[audio_path] = file_results
                
                # Save individual results
                output_filename = f"{Path(audio_path).stem}_analysis.json"
                output_path = os.path.join(output_dir, output_filename)
                
                # Temporarily store current results
                temp_results = self.results
                self.results = file_results
                self.save_results(output_path)
                self.results = temp_results
                
            except Exception as e:
                print(f"❌ Error processing {audio_path}: {e}")
                results[audio_path] = {"error": str(e)}
        
        # Save batch summary
        batch_summary = {
            "batch_info": {
                "total_files": len(audio_files),
                "successful": len([r for r in results.values() if "error" not in r]),
                "failed": len([r for r in results.values() if "error" in r]),
                "analysis_time": datetime.now().isoformat()
            },
            "individual_results": results
        }
        
        summary_path = os.path.join(output_dir, "batch_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(batch_summary, f, indent=2, default=str)
        
        print(f"\n✅ Batch analysis complete. Results saved to: {output_dir}")
        return batch_summary


def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Advanced Multi-Dimensional Audio Analysis Agent")
    parser.add_argument("audio_file", help="Path to audio file for analysis")
    parser.add_argument("--output", "-o", help="Output JSON file path", default="analysis_results.json")
    parser.add_argument("--detailed", "-d", action="store_true", help="Show detailed analysis output")
    parser.add_argument("--no-advanced", action="store_true", help="Skip advanced ML-based analysis")
    parser.add_argument("--sample-rate", type=int, default=22050, help="Audio sample rate")
    parser.add_argument("--streaming", action="store_true", help="Enable streaming mode")
    
    args = parser.parse_args()
    
    # Initialize agent
    agent = AdvancedAudioAnalysisAgent(
        sample_rate=args.sample_rate,
        enable_streaming=args.streaming
    )
    
    # Analyze audio
    results = agent.analyze_audio(
        args.audio_file,
        include_advanced=not args.no_advanced
    )
    
    # Print summary
    agent.print_summary(detailed=args.detailed)
    
    # Save results
    agent.save_results(args.output)


if __name__ == "__main__":
    main()