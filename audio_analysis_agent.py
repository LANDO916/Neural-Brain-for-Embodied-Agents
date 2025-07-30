#!/usr/bin/env python3
"""
Audio Analysis Agent - Layer 1: Parallel Deconstruction
A comprehensive audio analysis system that analyzes audio across three distinct dimensions:
1. Musical & Structural Analysis
2. Lyrical & Vocal Analysis  
3. Sonic & Spectral Analysis
"""

import os
import sys
import json
import numpy as np
import warnings
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

@dataclass
class AnalysisResult:
    """Container for analysis results"""
    musical_structural: Dict
    lyrical_vocal: Dict
    sonic_spectral: Dict
    metadata: Dict

class AudioAnalyzer:
    """Main audio analysis class"""
    
    def __init__(self):
        self.sample_rate = 22050
        self.hop_length = 512
        self.frame_length = 2048
        
    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Load audio file with librosa"""
        try:
            import librosa
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            return y, sr
        except ImportError:
            print("Warning: librosa not available, using basic audio loading")
            return self._basic_audio_load(audio_path)
        except Exception as e:
            print(f"Error loading audio: {e}")
            return None, None
    
    def _basic_audio_load(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Basic audio loading fallback"""
        try:
            import wave
            with wave.open(audio_path, 'rb') as wav_file:
                frames = wav_file.readframes(wav_file.getnframes())
                audio_data = np.frombuffer(frames, dtype=np.int16)
                sample_rate = wav_file.getframerate()
                # Normalize to float
                audio_data = audio_data.astype(np.float32) / 32768.0
                return audio_data, sample_rate
        except Exception as e:
            print(f"Error in basic audio loading: {e}")
            return None, None

    # ===== MUSICAL & STRUCTURAL ANALYSIS =====
    
    def bpm_and_tempo_mapper(self, audio_path: str) -> Dict:
        """Analyze BPM and tempo variations"""
        try:
            import librosa
            y, sr = self.load_audio(audio_path)
            if y is None:
                return {"error": "Could not load audio"}
            
            # Detect tempo
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            
            # Detect onset strength
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            
            # Detect dynamic tempo
            tempo_dynamic = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
            
            return {
                "bpm": float(tempo),
                "dynamic_tempo": float(tempo_dynamic),
                "onset_strength_mean": float(np.mean(onset_env)),
                "onset_strength_std": float(np.std(onset_env))
            }
        except Exception as e:
            return {"error": f"BPM analysis failed: {str(e)}"}
    
    def groove_and_swing_analyzer(self, audio_path: str) -> Dict:
        """Analyze groove and swing patterns"""
        try:
            import librosa
            y, sr = self.load_audio(audio_path)
            if y is None:
                return {"error": "Could not load audio"}
            
            # Detect beats
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            
            # Calculate beat intervals
            beat_intervals = np.diff(beats)
            beat_intervals_seconds = beat_intervals / sr
            
            # Analyze swing (ratio of consecutive beat intervals)
            if len(beat_intervals) >= 2:
                swing_ratios = beat_intervals[1::2] / beat_intervals[::2]
                swing_score = np.mean(swing_ratios)
            else:
                swing_score = 1.0
            
            # Calculate syncopation (deviation from regular timing)
            expected_interval = 60.0 / tempo
            syncopation = np.std(beat_intervals_seconds - expected_interval)
            
            return {
                "swing_score": float(swing_score),
                "syncopation": float(syncopation),
                "beat_regularity": float(1.0 / (1.0 + syncopation)),
                "num_beats": len(beats)
            }
        except Exception as e:
            return {"error": f"Groove analysis failed: {str(e)}"}
    
    def tonal_center_identifier(self, audio_path: str) -> Dict:
        """Identify key and tonal center"""
        try:
            import librosa
            y, sr = self.load_audio(audio_path)
            if y is None:
                return {"error": "Could not load audio"}
            
            # Extract chromagram
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
            
            # Find key using correlation with key profiles
            key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            major_profile = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
            minor_profile = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
            
            # Normalize profiles
            major_profile = np.array(major_profile) / np.sum(major_profile)
            minor_profile = np.array(minor_profile) / np.sum(minor_profile)
            
            # Calculate correlations
            chroma_mean = np.mean(chroma, axis=1)
            chroma_mean = chroma_mean / np.sum(chroma_mean)
            
            major_correlations = [np.corrcoef(chroma_mean, np.roll(major_profile, i))[0,1] for i in range(12)]
            minor_correlations = [np.corrcoef(chroma_mean, np.roll(minor_profile, i))[0,1] for i in range(12)]
            
            # Find best key
            major_max_idx = np.argmax(major_correlations)
            minor_max_idx = np.argmax(minor_correlations)
            
            if major_correlations[major_max_idx] > minor_correlations[minor_max_idx]:
                key = key_names[major_max_idx]
                mode = "major"
                confidence = major_correlations[major_max_idx]
            else:
                key = key_names[minor_max_idx]
                mode = "minor"
                confidence = minor_correlations[minor_max_idx]
            
            return {
                "key": key,
                "mode": mode,
                "confidence": float(confidence),
                "chroma_profile": chroma_mean.tolist()
            }
        except Exception as e:
            return {"error": f"Tonal analysis failed: {str(e)}"}
    
    def harmonic_journey_mapper(self, audio_path: str) -> Dict:
        """Map harmonic progressions and key changes"""
        try:
            import librosa
            y, sr = self.load_audio(audio_path)
            if y is None:
                return {"error": "Could not load audio"}
            
            # Extract chromagram
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=self.hop_length)
            
            # Detect chord changes
            chord_changes = []
            for i in range(1, chroma.shape[1]):
                # Calculate difference between consecutive frames
                diff = np.sum(np.abs(chroma[:, i] - chroma[:, i-1]))
                if diff > np.mean(np.diff(chroma, axis=1)) + np.std(np.diff(chroma, axis=1)):
                    chord_changes.append(i * self.hop_length / sr)
            
            # Calculate harmonic complexity
            harmonic_complexity = np.mean(np.std(chroma, axis=0))
            
            return {
                "chord_changes": chord_changes,
                "num_chord_changes": len(chord_changes),
                "harmonic_complexity": float(harmonic_complexity),
                "chroma_variation": float(np.std(chroma))
            }
        except Exception as e:
            return {"error": f"Harmonic analysis failed: {str(e)}"}
    
    def genre_and_style_agent(self, audio_path: str) -> Dict:
        """Classify genre and style based on audio features"""
        try:
            import librosa
            y, sr = self.load_audio(audio_path)
            if y is None:
                return {"error": "Could not load audio"}
            
            # Extract various features
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
            
            # Calculate feature statistics
            features = {
                "mfcc_mean": np.mean(mfcc, axis=1).tolist(),
                "mfcc_std": np.std(mfcc, axis=1).tolist(),
                "spectral_centroid_mean": float(np.mean(spectral_centroid)),
                "spectral_rolloff_mean": float(np.mean(spectral_rolloff)),
                "zero_crossing_rate_mean": float(np.mean(zero_crossing_rate))
            }
            
            # Simple genre classification based on features
            # This is a basic heuristic - in practice you'd use a trained model
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            
            if tempo > 140:
                style_hint = "fast/electronic"
            elif tempo > 100:
                style_hint = "medium/rock"
            else:
                style_hint = "slow/ballad"
            
            return {
                "style_hint": style_hint,
                "features": features,
                "note": "Basic heuristic classification - consider using trained models for better accuracy"
            }
        except Exception as e:
            return {"error": f"Genre analysis failed: {str(e)}"}

    # ===== LYRICAL & VOCAL ANALYSIS =====
    
    def speech_to_text_engine(self, audio_path: str) -> Dict:
        """Convert speech to text using available methods"""
        try:
            # Try whisper first
            try:
                import whisper
                model = whisper.load_model("base")
                result = model.transcribe(audio_path)
                return {
                    "transcript": result["text"],
                    "language": result.get("language", "unknown"),
                    "method": "whisper"
                }
            except ImportError:
                # Fallback to basic transcription attempt
                return {
                    "transcript": "[Audio transcription not available - install whisper for full functionality]",
                    "language": "unknown",
                    "method": "fallback"
                }
        except Exception as e:
            return {"error": f"Transcription failed: {str(e)}"}
    
    def poetic_structure_analyzer(self, transcript: str) -> Dict:
        """Analyze poetic structure and literary devices"""
        try:
            if not transcript or transcript.startswith("[Audio"):
                return {"error": "No transcript available"}
            
            # Basic analysis
            words = transcript.lower().split()
            sentences = transcript.split('.')
            
            # Count syllables (basic approximation)
            vowels = 'aeiou'
            syllable_count = sum(1 for word in words for char in word if char in vowels)
            
            # Analyze sentence structure
            avg_sentence_length = len(words) / max(len(sentences), 1)
            
            # Look for repetition patterns
            word_freq = {}
            for word in words:
                if len(word) > 2:  # Skip short words
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            repeated_words = {word: count for word, count in word_freq.items() if count > 1}
            
            return {
                "word_count": len(words),
                "syllable_count": syllable_count,
                "avg_sentence_length": avg_sentence_length,
                "repeated_words": repeated_words,
                "vocabulary_diversity": len(set(words)) / max(len(words), 1)
            }
        except Exception as e:
            return {"error": f"Poetic analysis failed: {str(e)}"}
    
    def lexical_sentiment_scorer(self, transcript: str) -> Dict:
        """Score sentiment of lyrics"""
        try:
            if not transcript or transcript.startswith("[Audio"):
                return {"error": "No transcript available"}
            
            # Basic sentiment analysis using word lists
            positive_words = ['love', 'happy', 'joy', 'beautiful', 'wonderful', 'amazing', 'great', 'good', 'nice']
            negative_words = ['hate', 'sad', 'pain', 'terrible', 'awful', 'bad', 'horrible', 'angry', 'fear']
            
            words = transcript.lower().split()
            
            positive_count = sum(1 for word in words if word in positive_words)
            negative_count = sum(1 for word in words if word in negative_words)
            
            total_sentiment_words = positive_count + negative_count
            if total_sentiment_words > 0:
                sentiment_score = (positive_count - negative_count) / total_sentiment_words
            else:
                sentiment_score = 0
            
            return {
                "sentiment_score": sentiment_score,
                "positive_words": positive_count,
                "negative_words": negative_count,
                "sentiment_intensity": abs(sentiment_score),
                "overall_tone": "positive" if sentiment_score > 0 else "negative" if sentiment_score < 0 else "neutral"
            }
        except Exception as e:
            return {"error": f"Sentiment analysis failed: {str(e)}"}
    
    def thematic_extractor(self, transcript: str) -> Dict:
        """Extract themes and topics from lyrics"""
        try:
            if not transcript or transcript.startswith("[Audio"):
                return {"error": "No transcript available"}
            
            # Basic theme detection using keyword matching
            themes = {
                "love": ['love', 'heart', 'romance', 'relationship', 'kiss', 'hug'],
                "nature": ['nature', 'tree', 'flower', 'mountain', 'ocean', 'sky'],
                "emotion": ['feel', 'emotion', 'sad', 'happy', 'angry', 'fear'],
                "time": ['time', 'moment', 'forever', 'yesterday', 'tomorrow'],
                "journey": ['road', 'path', 'journey', 'travel', 'walk', 'run']
            }
            
            words = transcript.lower().split()
            theme_counts = {}
            
            for theme, keywords in themes.items():
                count = sum(1 for word in words if word in keywords)
                if count > 0:
                    theme_counts[theme] = count
            
            # Find dominant themes
            dominant_themes = sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)
            
            return {
                "detected_themes": theme_counts,
                "dominant_themes": dominant_themes[:3],
                "theme_diversity": len(theme_counts)
            }
        except Exception as e:
            return {"error": f"Theme extraction failed: {str(e)}"}
    
    def pitch_and_melody_contour_analyzer(self, audio_path: str) -> Dict:
        """Analyze pitch and melody contours"""
        try:
            import librosa
            y, sr = self.load_audio(audio_path)
            if y is None:
                return {"error": "Could not load audio"}
            
            # Extract pitch
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            
            # Get the pitch with highest magnitude at each time
            pitch_track = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_track.append(pitch)
            
            if not pitch_track:
                return {"error": "No pitch detected"}
            
            pitch_track = np.array(pitch_track)
            
            # Convert to MIDI note numbers
            midi_notes = librosa.hz_to_midi(pitch_track)
            
            # Analyze melody
            pitch_range = np.max(midi_notes) - np.min(midi_notes)
            pitch_variance = np.var(midi_notes)
            
            # Detect melodic intervals
            intervals = np.diff(midi_notes)
            interval_variance = np.var(intervals)
            
            return {
                "pitch_range_semitones": float(pitch_range),
                "pitch_variance": float(pitch_variance),
                "interval_variance": float(interval_variance),
                "melodic_complexity": float(interval_variance / max(pitch_variance, 1)),
                "avg_pitch": float(np.mean(midi_notes)),
                "pitch_contour": midi_notes.tolist()[:100]  # First 100 notes for visualization
            }
        except Exception as e:
            return {"error": f"Pitch analysis failed: {str(e)}"}
    
    def vocal_timbre_identifier(self, audio_path: str) -> Dict:
        """Identify vocal timbre characteristics"""
        try:
            import librosa
            y, sr = self.load_audio(audio_path)
            if y is None:
                return {"error": "Could not load audio"}
            
            # Extract spectral features
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            
            # Calculate timbre characteristics
            brightness = np.mean(spectral_centroid)
            warmth = np.mean(spectral_bandwidth)
            presence = np.mean(spectral_rolloff)
            
            # Classify timbre based on spectral characteristics
            if brightness > 2000:
                timbre_type = "bright"
            elif brightness < 1000:
                timbre_type = "dark"
            else:
                timbre_type = "neutral"
            
            return {
                "timbre_type": timbre_type,
                "brightness": float(brightness),
                "warmth": float(warmth),
                "presence": float(presence),
                "spectral_balance": float(brightness / max(warmth, 1))
            }
        except Exception as e:
            return {"error": f"Timbre analysis failed: {str(e)}"}

    # ===== SONIC & SPECTRAL ANALYSIS =====
    
    def source_separator(self, audio_path: str) -> Dict:
        """Separate audio sources (basic implementation)"""
        try:
            import librosa
            y, sr = self.load_audio(audio_path)
            if y is None:
                return {"error": "Could not load audio"}
            
            # Basic source separation using spectral subtraction
            # This is a simplified approach - full spleeter would be better
            
            # Separate vocals using HPSS (Harmonic/Percussive Source Separation)
            y_harmonic, y_percussive = librosa.effects.hpss(y)
            
            # Calculate energy ratios
            harmonic_energy = np.sum(y_harmonic**2)
            percussive_energy = np.sum(y_percussive**2)
            total_energy = harmonic_energy + percussive_energy
            
            if total_energy > 0:
                harmonic_ratio = harmonic_energy / total_energy
                percussive_ratio = percussive_energy / total_energy
            else:
                harmonic_ratio = percussive_ratio = 0.5
            
            return {
                "harmonic_ratio": float(harmonic_ratio),
                "percussive_ratio": float(percussive_ratio),
                "separation_quality": "basic",
                "note": "Full source separation requires spleeter or similar tools"
            }
        except Exception as e:
            return {"error": f"Source separation failed: {str(e)}"}
    
    def timbre_fingerprinter(self, audio_path: str) -> Dict:
        """Create timbre fingerprint of instruments"""
        try:
            import librosa
            y, sr = self.load_audio(audio_path)
            if y is None:
                return {"error": "Could not load audio"}
            
            # Extract MFCCs for timbre analysis
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
            
            # Calculate spectral features
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            
            # Create timbre fingerprint
            fingerprint = {
                "mfcc_mean": np.mean(mfcc, axis=1).tolist(),
                "mfcc_std": np.std(mfcc, axis=1).tolist(),
                "spectral_centroid_mean": float(np.mean(spectral_centroid)),
                "spectral_bandwidth_mean": float(np.mean(spectral_bandwidth)),
                "spectral_rolloff_mean": float(np.mean(spectral_rolloff))
            }
            
            return {
                "timbre_fingerprint": fingerprint,
                "fingerprint_dimensions": len(fingerprint["mfcc_mean"]) + 3
            }
        except Exception as e:
            return {"error": f"Timbre fingerprinting failed: {str(e)}"}
    
    def loudness_meter(self, audio_path: str) -> Dict:
        """Measure loudness in LUFS"""
        try:
            import librosa
            y, sr = self.load_audio(audio_path)
            if y is None:
                return {"error": "Could not load audio"}
            
            # Calculate RMS energy
            rms = librosa.feature.rms(y=y)
            
            # Calculate loudness (approximation of LUFS)
            # Note: True LUFS requires pyloudnorm, but we can approximate
            loudness = 20 * np.log10(np.mean(rms) + 1e-10)
            
            # Calculate dynamic range
            dynamic_range = 20 * np.log10(np.max(rms) / (np.min(rms) + 1e-10))
            
            # Calculate crest factor
            crest_factor = np.max(np.abs(y)) / (np.mean(rms) + 1e-10)
            
            return {
                "loudness_db": float(loudness),
                "dynamic_range_db": float(dynamic_range),
                "crest_factor": float(crest_factor),
                "rms_mean": float(np.mean(rms)),
                "rms_std": float(np.std(rms))
            }
        except Exception as e:
            return {"error": f"Loudness analysis failed: {str(e)}"}
    
    def spectral_power_monitor(self, audio_path: str) -> Dict:
        """Monitor spectral power across frequency bands"""
        try:
            import librosa
            y, sr = self.load_audio(audio_path)
            if y is None:
                return {"error": "Could not load audio"}
            
            # Calculate spectrogram
            D = librosa.stft(y)
            S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
            
            # Define frequency bands (bass, mid, treble)
            freqs = librosa.fft_frequencies(sr=sr)
            
            # Bass: 20-250 Hz
            bass_mask = (freqs >= 20) & (freqs <= 250)
            bass_power = np.mean(S_db[bass_mask, :])
            
            # Mid: 250-4000 Hz
            mid_mask = (freqs > 250) & (freqs <= 4000)
            mid_power = np.mean(S_db[mid_mask, :])
            
            # Treble: 4000-20000 Hz
            treble_mask = (freqs > 4000) & (freqs <= 20000)
            treble_power = np.mean(S_db[treble_mask, :])
            
            # Calculate spectral balance
            total_power = bass_power + mid_power + treble_power
            if total_power != 0:
                bass_ratio = bass_power / total_power
                mid_ratio = mid_power / total_power
                treble_ratio = treble_power / total_power
            else:
                bass_ratio = mid_ratio = treble_ratio = 1/3
            
            return {
                "bass_power_db": float(bass_power),
                "mid_power_db": float(mid_power),
                "treble_power_db": float(treble_power),
                "bass_ratio": float(bass_ratio),
                "mid_ratio": float(mid_ratio),
                "treble_ratio": float(treble_ratio),
                "spectral_balance": "balanced" if abs(bass_ratio - mid_ratio) < 0.2 else "unbalanced"
            }
        except Exception as e:
            return {"error": f"Spectral analysis failed: {str(e)}"}
    
    def stereo_image_analyzer(self, audio_path: str) -> Dict:
        """Analyze stereo image and phase correlation"""
        try:
            import librosa
            y, sr = self.load_audio(audio_path)
            if y is None:
                return {"error": "Could not load audio"}
            
            # Check if stereo
            if len(y.shape) == 1:
                # Mono audio
                return {
                    "stereo_width": 0.0,
                    "phase_correlation": 1.0,
                    "channel_count": 1,
                    "stereo_type": "mono"
                }
            
            # Stereo audio
            left_channel = y[0]
            right_channel = y[1]
            
            # Calculate stereo width
            mid = (left_channel + right_channel) / 2
            side = (left_channel - right_channel) / 2
            
            mid_energy = np.sum(mid**2)
            side_energy = np.sum(side**2)
            
            if mid_energy > 0:
                stereo_width = side_energy / mid_energy
            else:
                stereo_width = 0
            
            # Calculate phase correlation
            correlation = np.corrcoef(left_channel, right_channel)[0, 1]
            if np.isnan(correlation):
                correlation = 0
            
            return {
                "stereo_width": float(stereo_width),
                "phase_correlation": float(correlation),
                "channel_count": 2,
                "stereo_type": "stereo",
                "mid_side_ratio": float(side_energy / max(mid_energy, 1e-10))
            }
        except Exception as e:
            return {"error": f"Stereo analysis failed: {str(e)}"}
    
    def reverb_impulse_modeler(self, audio_path: str) -> Dict:
        """Model reverb characteristics"""
        try:
            import librosa
            y, sr = self.load_audio(audio_path)
            if y is None:
                return {"error": "Could not load audio"}
            
            # Calculate energy decay
            energy = librosa.feature.rms(y=y)
            
            # Find decay time (time for energy to drop by 60dB)
            # This is a simplified approach
            max_energy = np.max(energy)
            threshold = max_energy * 0.001  # -60dB
            
            decay_frames = 0
            for i in range(len(energy[0])):
                if energy[0, i] < threshold:
                    decay_frames = i
                    break
            
            decay_time = decay_frames * self.hop_length / sr
            
            # Estimate room size based on decay time
            if decay_time < 0.5:
                room_size = "small"
            elif decay_time < 1.5:
                room_size = "medium"
            else:
                room_size = "large"
            
            return {
                "decay_time_seconds": float(decay_time),
                "estimated_room_size": room_size,
                "reverb_density": float(np.std(energy)),
                "early_reflections": "detected" if decay_time > 0.1 else "minimal"
            }
        except Exception as e:
            return {"error": f"Reverb analysis failed: {str(e)}"}

    def analyze_audio(self, audio_path: str) -> AnalysisResult:
        """Run complete audio analysis across all three dimensions"""
        
        print(f"Analyzing audio file: {audio_path}")
        print("=" * 50)
        
        # Musical & Structural Analysis
        print("1. Musical & Structural Analysis...")
        musical_structural = {
            "bpm_and_tempo": self.bpm_and_tempo_mapper(audio_path),
            "groove_and_swing": self.groove_and_swing_analyzer(audio_path),
            "tonal_center": self.tonal_center_identifier(audio_path),
            "harmonic_journey": self.harmonic_journey_mapper(audio_path),
            "genre_and_style": self.genre_and_style_agent(audio_path)
        }
        
        # Lyrical & Vocal Analysis
        print("2. Lyrical & Vocal Analysis...")
        transcript_result = self.speech_to_text_engine(audio_path)
        transcript = transcript_result.get("transcript", "")
        
        lyrical_vocal = {
            "transcription": transcript_result,
            "poetic_structure": self.poetic_structure_analyzer(transcript),
            "sentiment": self.lexical_sentiment_scorer(transcript),
            "themes": self.thematic_extractor(transcript),
            "pitch_and_melody": self.pitch_and_melody_contour_analyzer(audio_path),
            "vocal_timbre": self.vocal_timbre_identifier(audio_path)
        }
        
        # Sonic & Spectral Analysis
        print("3. Sonic & Spectral Analysis...")
        sonic_spectral = {
            "source_separation": self.source_separator(audio_path),
            "timbre_fingerprint": self.timbre_fingerprinter(audio_path),
            "loudness": self.loudness_meter(audio_path),
            "spectral_power": self.spectral_power_monitor(audio_path),
            "stereo_image": self.stereo_image_analyzer(audio_path),
            "reverb_model": self.reverb_impulse_modeler(audio_path)
        }
        
        # Metadata
        metadata = {
            "audio_file": audio_path,
            "analysis_timestamp": str(np.datetime64('now')),
            "analysis_version": "1.0",
            "sample_rate": self.sample_rate
        }
        
        return AnalysisResult(
            musical_structural=musical_structural,
            lyrical_vocal=lyrical_vocal,
            sonic_spectral=sonic_spectral,
            metadata=metadata
        )

def main(audio_path: str):
    """Main function to run audio analysis"""
    
    # Check if file exists
    if not os.path.exists(audio_path):
        print(f"Error: Audio file '{audio_path}' not found.")
        return
    
    # Create analyzer and run analysis
    analyzer = AudioAnalyzer()
    result = analyzer.analyze_audio(audio_path)
    
    # Convert to dictionary and print results
    results_dict = asdict(result)
    
    print("\n" + "=" * 50)
    print("ANALYSIS COMPLETE")
    print("=" * 50)
    
    # Print summary
    print("\nSUMMARY:")
    print("-" * 20)
    
    # Musical summary
    bpm_info = results_dict["musical_structural"]["bpm_and_tempo"]
    if "bpm" in bpm_info:
        print(f"BPM: {bpm_info['bpm']:.1f}")
    
    key_info = results_dict["musical_structural"]["tonal_center"]
    if "key" in key_info:
        print(f"Key: {key_info['key']} {key_info['mode']}")
    
    # Loudness summary
    loudness_info = results_dict["sonic_spectral"]["loudness"]
    if "loudness_db" in loudness_info:
        print(f"Loudness: {loudness_info['loudness_db']:.1f} dB")
    
    # Save results to JSON file
    output_file = f"{Path(audio_path).stem}_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(results_dict, f, indent=2, default=str)
    
    print(f"\nDetailed results saved to: {output_file}")
    
    # Print full results
    print("\nDETAILED RESULTS:")
    print(json.dumps(results_dict, indent=2, default=str))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python audio_analysis_agent.py <audio_file>")
        print("Example: python audio_analysis_agent.py song.wav")
        sys.exit(1)
    
    main(sys.argv[1])