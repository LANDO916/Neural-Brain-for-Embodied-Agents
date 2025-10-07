#!/usr/bin/env python3
"""
Multi-Dimensional Audio Analysis Agent
Based on the Layer 1: Parallel Deconstruction architecture

This agent analyzes audio across three distinct dimensions simultaneously:
1. Musical & Structural Analysis
2. Lyrical & Vocal Analysis 
3. Sonic & Spectral Analysis

All components use free and open-source libraries.
"""

import os
import sys
import logging
import json
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import concurrent.futures
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Audio processing libraries
try:
    import madmom
except ImportError:
    madmom = None
    print("Warning: madmom not available - some beat tracking features disabled")

# Speech recognition
try:
    import speech_recognition as sr
    from faster_whisper import WhisperModel
except ImportError:
    print("Warning: Speech recognition libraries not available")

# Text processing
try:
    import nltk
    from textblob import TextBlob
except ImportError:
    print("Warning: Text processing libraries not available")

# Scientific computing
import pandas as pd
from scipy import signal, stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

class AudioAnalysisAgent:
    """
    Multi-dimensional audio analysis agent implementing parallel deconstruction
    """
    
    def __init__(self, 
                 whisper_model_size: str = "base",
                 sample_rate: int = 22050,
                 hop_length: int = 512,
                 n_fft: int = 2048):
        """
        Initialize the audio analysis agent
        
        Args:
            whisper_model_size: Size of Whisper model for transcription
            sample_rate: Target sample rate for analysis
            hop_length: Hop length for STFT analysis
            n_fft: FFT window size
        """
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_fft = n_fft
        
        # Initialize speech recognition
        self.recognizer = sr.Recognizer()
        try:
            self.whisper_model = WhisperModel(whisper_model_size, device="cpu")
        except:
            self.whisper_model = None
            print("Warning: Whisper model not available")
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("Audio Analysis Agent initialized successfully")

    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Load audio file with error handling"""
        try:
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            self.logger.info(f"Loaded audio: {audio_path} ({len(y)/sr:.2f}s)")
            return y, sr
        except Exception as e:
            self.logger.error(f"Error loading audio {audio_path}: {e}")
            raise

    def musical_structural_analysis(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Musical & Structural Analysis Agent
        Analyzes BPM, time signature, swing, harmony, key, chords, modulations, genre, and style
        """
        results = {}
        
        try:
            # Rhythm & Groove Analysis
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=self.hop_length)
            results['bpm'] = float(tempo)
            results['beat_times'] = beats.tolist()
            
            # BPM & Tempo Mapping
            onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=self.hop_length)
            onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=self.hop_length)
            results['onset_times'] = onset_times.tolist()
            
            # Groove & Swing Analysis (tempo variations)
            if len(beats) > 1:
                beat_intervals = np.diff(librosa.frames_to_time(beats, sr=sr))
                results['tempo_variation_std'] = float(np.std(beat_intervals))
                results['swing_factor'] = float(np.mean(beat_intervals[1::2] - beat_intervals[::2]) if len(beat_intervals) > 2 else 0)
            
            # Harmony & Key Analysis
            # Chromagram for key detection
            chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=self.hop_length)
            results['chroma_mean'] = chroma.mean(axis=1).tolist()
            
            # Key detection using template matching
            key_templates = self._get_key_templates()
            chroma_mean = chroma.mean(axis=1)
            correlations = []
            for template in key_templates:
                correlation = np.corrcoef(chroma_mean, template)[0, 1]
                correlations.append(correlation if not np.isnan(correlation) else 0)
            
            key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'] * 2  # Major and minor
            best_key_idx = np.argmax(correlations)
            results['estimated_key'] = key_names[best_key_idx]
            results['key_confidence'] = float(max(correlations))
            
            # Tonal Center Identifier
            tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
            results['tonal_centroid'] = tonnetz.mean(axis=1).tolist()
            
            # Harmonic Journey Mapper (chord progressions)
            # Simplified chord detection using chromagram
            chroma_frames = chroma.T
            n_chords = min(8, len(chroma_frames) // 10)  # Divide into segments
            if n_chords > 1:
                kmeans = KMeans(n_clusters=n_chords, random_state=42, n_init=10)
                chord_labels = kmeans.fit_predict(chroma_frames)
                results['chord_progression'] = chord_labels.tolist()
                results['unique_chords'] = int(n_chords)
            
            # Genre & Style Analysis
            # Spectral features for genre classification
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
            
            results['spectral_centroid_mean'] = float(np.mean(spectral_centroids))
            results['spectral_rolloff_mean'] = float(np.mean(spectral_rolloff))
            results['spectral_bandwidth_mean'] = float(np.mean(spectral_bandwidth))
            results['zero_crossing_rate_mean'] = float(np.mean(zero_crossing_rate))
            
            # Taxonomy Classifier (genre prediction based on features)
            genre_features = np.array([
                results['spectral_centroid_mean'],
                results['spectral_rolloff_mean'],
                results['spectral_bandwidth_mean'],
                results['zero_crossing_rate_mean'],
                results['bpm']
            ])
            results['genre_features'] = genre_features.tolist()
            results['estimated_genre'] = self._classify_genre(genre_features)
            
            # Stylistic Feature Extractor
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            results['mfccs_mean'] = mfccs.mean(axis=1).tolist()
            results['mfccs_std'] = mfccs.std(axis=1).tolist()
            
        except Exception as e:
            self.logger.error(f"Error in musical analysis: {e}")
            results['error'] = str(e)
        
        return results

    def lyrical_vocal_analysis(self, y: np.ndarray, sr: int, audio_path: str = None) -> Dict[str, Any]:
        """
        Lyrical & Vocal Analysis Agent
        Transcription, poetics, sentiment analysis, and vocal performance analysis
        """
        results = {}
        
        try:
            # Speech-to-Text Engine
            if self.whisper_model and audio_path:
                try:
                    segments, info = self.whisper_model.transcribe(audio_path, beam_size=5)
                    transcription = ""
                    segment_data = []
                    for segment in segments:
                        transcription += segment.text + " "
                        segment_data.append({
                            'start': segment.start,
                            'end': segment.end,
                            'text': segment.text,
                            'confidence': getattr(segment, 'avg_logprob', 0)
                        })
                    results['transcription'] = transcription.strip()
                    results['segments'] = segment_data
                    results['detected_language'] = info.language
                    results['language_probability'] = info.language_probability
                except Exception as e:
                    self.logger.warning(f"Whisper transcription failed: {e}")
                    results['transcription'] = ""
            else:
                results['transcription'] = ""
            
            # Poetic Structure Analyzer
            if results['transcription']:
                text = results['transcription']
                words = text.split()
                results['word_count'] = len(words)
                results['unique_words'] = len(set(word.lower().strip('.,!?;:"()[]') for word in words))
                results['lexical_diversity'] = results['unique_words'] / max(results['word_count'], 1)
                
                # Simple rhyme scheme detection
                lines = text.split('.')
                results['estimated_lines'] = len(lines)
                
                # Syllable counting (approximate)
                syllable_count = sum(self._count_syllables(word) for word in words)
                results['total_syllables'] = syllable_count
                results['avg_syllables_per_word'] = syllable_count / max(len(words), 1)
            
            # Lyric & Sentiment Luminator
            if results['transcription']:
                try:
                    blob = TextBlob(results['transcription'])
                    results['sentiment_polarity'] = blob.sentiment.polarity
                    results['sentiment_subjectivity'] = blob.sentiment.subjectivity
                    
                    # Lexical Sentiment Scorer
                    positive_words = sum(1 for word in blob.words if TextBlob(word).sentiment.polarity > 0.1)
                    negative_words = sum(1 for word in blob.words if TextBlob(word).sentiment.polarity < -0.1)
                    results['positive_word_ratio'] = positive_words / max(len(blob.words), 1)
                    results['negative_word_ratio'] = negative_words / max(len(blob.words), 1)
                    
                except Exception as e:
                    self.logger.warning(f"Sentiment analysis failed: {e}")
                    results['sentiment_polarity'] = 0
                    results['sentiment_subjectivity'] = 0
            
            # Vocal Performance Agent
            # Separate vocal using HPSS
            y_harmonic, y_percussive = librosa.effects.hpss(y)
            
            # Pitch & Melody Contour Analyzer
            pitches, magnitudes = librosa.piptrack(y=y_harmonic, sr=sr, threshold=0.1)
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)
            
            if pitch_values:
                results['fundamental_frequency_mean'] = float(np.mean(pitch_values))
                results['fundamental_frequency_std'] = float(np.std(pitch_values))
                results['pitch_range'] = float(max(pitch_values) - min(pitch_values))
            else:
                results['fundamental_frequency_mean'] = 0
                results['fundamental_frequency_std'] = 0
                results['pitch_range'] = 0
            
            # Vocal Timbre Identifier
            # Spectral features for vocal quality
            spectral_centroid = librosa.feature.spectral_centroid(y=y_harmonic, sr=sr)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y_harmonic, sr=sr)[0]
            spectral_contrast = librosa.feature.spectral_contrast(y=y_harmonic, sr=sr)
            
            results['vocal_brightness'] = float(np.mean(spectral_centroid))
            results['vocal_bandwidth'] = float(np.mean(spectral_bandwidth))
            results['vocal_contrast'] = spectral_contrast.mean(axis=1).tolist()
            
            # Vocal quality assessment (breathy, raspy, etc.)
            harmonicity = self._compute_harmonicity(y_harmonic, sr)
            results['vocal_harmonicity'] = float(harmonicity)
            
        except Exception as e:
            self.logger.error(f"Error in lyrical/vocal analysis: {e}")
            results['error'] = str(e)
        
        return results

    def sonic_spectral_analysis(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Sonic & Spectral Analysis Agent
        Instrumentation, timbre, dynamics, energy, and spatial analysis
        """
        results = {}
        
        try:
            # Instrumentation & Timbre Agent
            # Source Separator (Harmonic-Percussive Separation)
            y_harmonic, y_percussive = librosa.effects.hpss(y, margin=3.0)
            
            harmonic_energy = float(np.mean(y_harmonic**2))
            percussive_energy = float(np.mean(y_percussive**2))
            total_energy = harmonic_energy + percussive_energy
            
            results['harmonic_ratio'] = harmonic_energy / max(total_energy, 1e-10)
            results['percussive_ratio'] = percussive_energy / max(total_energy, 1e-10)
            
            # Timbre Fingerprinter
            # Spectral features for instrument identification
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
            
            results['timbre_mfccs'] = mfccs.mean(axis=1).tolist()
            results['timbre_contrast'] = spectral_contrast.mean(axis=1).tolist()
            results['timbre_tonnetz'] = tonnetz.mean(axis=1).tolist()
            
            # Instrument detection based on spectral characteristics
            results['estimated_instruments'] = self._detect_instruments(y, sr)
            
            # Dynamics & Energy Agent
            # Loudness Meter (LUFS approximation)
            rms = librosa.feature.rms(y=y, hop_length=self.hop_length)[0]
            lufs_approx = 20 * np.log10(np.mean(rms) + 1e-10)
            results['integrated_loudness_lufs'] = float(lufs_approx)
            results['loudness_range'] = float(20 * np.log10(np.max(rms) + 1e-10) - 20 * np.log10(np.min(rms) + 1e-10))
            
            # Spectral Power Monitor
            stft = librosa.stft(y, hop_length=self.hop_length, n_fft=self.n_fft)
            magnitude = np.abs(stft)
            
            # Energy in different frequency bands
            freqs = librosa.fft_frequencies(sr=sr, n_fft=self.n_fft)
            bass_mask = freqs <= 250
            mid_mask = (freqs > 250) & (freqs <= 4000)
            treble_mask = freqs > 4000
            
            bass_energy = float(np.mean(magnitude[bass_mask, :]))
            mid_energy = float(np.mean(magnitude[mid_mask, :]))
            treble_energy = float(np.mean(magnitude[treble_mask, :]))
            
            total_spectral_energy = bass_energy + mid_energy + treble_energy
            results['bass_energy_ratio'] = bass_energy / max(total_spectral_energy, 1e-10)
            results['mid_energy_ratio'] = mid_energy / max(total_spectral_energy, 1e-10)
            results['treble_energy_ratio'] = treble_energy / max(total_spectral_energy, 1e-10)
            
            # Dynamic range analysis
            db_values = 20 * np.log10(rms + 1e-10)
            results['dynamic_range_db'] = float(np.max(db_values) - np.min(db_values))
            results['crest_factor'] = float(np.max(np.abs(y)) / (np.sqrt(np.mean(y**2)) + 1e-10))
            
            # Spatial & Acoustic Agent
            # Stereo Image Analyzer (if stereo)
            if len(y.shape) > 1:
                results['stereo_width'] = float(np.std(y[0] - y[1]))
                results['stereo_correlation'] = float(np.corrcoef(y[0], y[1])[0, 1])
            else:
                results['stereo_width'] = 0.0
                results['stereo_correlation'] = 1.0
            
            # Reverb Impulse Modeler (estimate room characteristics)
            # Autocorrelation for reverb estimation
            autocorr = np.correlate(y, y, mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            autocorr = autocorr / autocorr[0]  # Normalize
            
            # Find decay characteristics
            decay_threshold = 0.1
            decay_idx = np.where(autocorr < decay_threshold)[0]
            if len(decay_idx) > 0:
                rt60_estimate = decay_idx[0] / sr * 6  # Rough RT60 estimate
                results['estimated_rt60'] = float(rt60_estimate)
            else:
                results['estimated_rt60'] = 0.0
            
            # Frequency response analysis
            frequencies, response = signal.welch(y, sr, nperseg=2048)
            results['frequency_response'] = {
                'frequencies': frequencies.tolist(),
                'magnitude': 10 * np.log10(response + 1e-10).tolist()
            }
            
            # Spectral statistics
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]
            
            results['spectral_centroid_mean'] = float(np.mean(spectral_centroid))
            results['spectral_bandwidth_mean'] = float(np.mean(spectral_bandwidth))
            results['spectral_rolloff_mean'] = float(np.mean(spectral_rolloff))
            
        except Exception as e:
            self.logger.error(f"Error in sonic/spectral analysis: {e}")
            results['error'] = str(e)
        
        return results

    def analyze_audio(self, audio_path: str, save_results: bool = True) -> Dict[str, Any]:
        """
        Main analysis function - processes audio through all three analysis dimensions
        """
        self.logger.info(f"Starting analysis of: {audio_path}")
        
        # Load audio
        y, sr = self.load_audio(audio_path)
        
        # Parallel analysis of all three dimensions
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            # Submit all analysis tasks
            musical_future = executor.submit(self.musical_structural_analysis, y, sr)
            lyrical_future = executor.submit(self.lyrical_vocal_analysis, y, sr, audio_path)
            sonic_future = executor.submit(self.sonic_spectral_analysis, y, sr)
            
            # Collect results
            try:
                musical_results = musical_future.result(timeout=300)  # 5 min timeout
                lyrical_results = lyrical_future.result(timeout=300)
                sonic_results = sonic_future.result(timeout=300)
            except concurrent.futures.TimeoutError:
                self.logger.error("Analysis timed out")
                raise
        
        # Combine all results
        analysis_results = {
            'file_info': {
                'path': audio_path,
                'duration_seconds': len(y) / sr,
                'sample_rate': sr,
                'channels': 1 if len(y.shape) == 1 else y.shape[0]
            },
            'musical_structural': musical_results,
            'lyrical_vocal': lyrical_results,
            'sonic_spectral': sonic_results,
            'analysis_metadata': {
                'agent_version': '1.0',
                'libraries_used': [
                    'librosa', 'faster-whisper', 'scikit-learn', 
                    'textblob', 'numpy', 'scipy'
                ]
            }
        }
        
        # Save results if requested
        if save_results:
            output_path = Path(audio_path).with_suffix('.analysis.json')
            with open(output_path, 'w') as f:
                json.dump(analysis_results, f, indent=2)
            self.logger.info(f"Analysis saved to: {output_path}")
        
        self.logger.info("Analysis completed successfully")
        return analysis_results

    def _get_key_templates(self) -> List[np.ndarray]:
        """Generate major and minor key templates for key detection"""
        # Krumhansl-Schmuckler key profiles
        major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
        
        templates = []
        # Generate all 12 major and 12 minor keys
        for shift in range(12):
            templates.append(np.roll(major_profile, shift))
            templates.append(np.roll(minor_profile, shift))
        
        return templates

    def _classify_genre(self, features: np.ndarray) -> str:
        """Simple genre classification based on audio features"""
        # Simplified genre classification using heuristics
        spectral_centroid, spectral_rolloff, spectral_bandwidth, zcr, bpm = features
        
        if bpm > 140 and zcr > 0.1:
            return "Electronic/Dance"
        elif bpm < 80 and spectral_centroid < 2000:
            return "Ballad/Slow"
        elif spectral_bandwidth > 2000 and zcr > 0.15:
            return "Rock/Metal"
        elif 80 <= bpm <= 120 and spectral_centroid < 3000:
            return "Pop"
        elif bpm > 120 and spectral_rolloff > 4000:
            return "Hip-Hop/Rap"
        else:
            return "Other/Unknown"

    def _count_syllables(self, word: str) -> int:
        """Approximate syllable counting"""
        word = word.lower().strip('.,!?;:"()[]')
        vowels = "aeiouy"
        syllable_count = 0
        prev_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_was_vowel:
                syllable_count += 1
            prev_was_vowel = is_vowel
        
        if word.endswith('e'):
            syllable_count -= 1
        
        return max(1, syllable_count) if word else 0

    def _compute_harmonicity(self, y: np.ndarray, sr: int) -> float:
        """Compute harmonicity measure for vocal quality assessment"""
        try:
            # Autocorrelation-based harmonicity
            autocorr = np.correlate(y, y, mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            autocorr = autocorr / autocorr[0] if autocorr[0] != 0 else autocorr
            
            # Find peaks in autocorrelation (harmonic structure)
            peaks, _ = signal.find_peaks(autocorr[:sr//50], height=0.1)  # Look for fundamental freq
            
            if len(peaks) > 0:
                return float(np.mean(autocorr[peaks[:5]]))  # Average of first few peaks
            else:
                return 0.0
        except:
            return 0.0

    def _detect_instruments(self, y: np.ndarray, sr: int) -> List[str]:
        """Simple instrument detection based on spectral characteristics"""
        instruments = []
        
        # Separate harmonic and percussive components
        y_harmonic, y_percussive = librosa.effects.hpss(y, margin=3.0)
        
        # Energy ratio analysis
        harmonic_energy = np.mean(y_harmonic**2)
        percussive_energy = np.mean(y_percussive**2)
        total_energy = harmonic_energy + percussive_energy
        
        if percussive_energy / total_energy > 0.3:
            instruments.append("Drums/Percussion")
        
        if harmonic_energy / total_energy > 0.4:
            # Analyze harmonic content for instrument detection
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y_harmonic, sr=sr))
            
            if spectral_centroid > 3000:
                instruments.append("Lead Guitar/High Strings")
            elif 1000 < spectral_centroid < 3000:
                instruments.append("Vocals/Mid-range")
            elif spectral_centroid < 1000:
                instruments.append("Bass/Low Strings")
        
        # Check for presence of strong low frequencies
        stft = librosa.stft(y, n_fft=2048)
        magnitude = np.abs(stft)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
        
        bass_energy = np.mean(magnitude[freqs <= 250])
        total_spectral_energy = np.mean(magnitude)
        
        if bass_energy / total_spectral_energy > 0.3:
            instruments.append("Bass")
        
        return instruments if instruments else ["Unknown"]

    def create_visualization(self, results: Dict[str, Any], save_path: str = None) -> None:
        """Create comprehensive visualization of analysis results"""
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('Audio Analysis Results - Parallel Deconstruction', fontsize=16)
        
        # Musical & Structural Analysis
        ax1 = axes[0, 0]
        musical = results['musical_structural']
        if 'chroma_mean' in musical:
            ax1.bar(range(12), musical['chroma_mean'])
            ax1.set_title('Chroma Features (Key Analysis)')
            ax1.set_xlabel('Pitch Class')
            ax1.set_ylabel('Strength')
            ax1.set_xticks(range(12))
            ax1.set_xticklabels(['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'])
        
        ax2 = axes[0, 1]
        if 'mfccs_mean' in musical:
            ax2.plot(musical['mfccs_mean'])
            ax2.set_title('MFCC Features (Timbre)')
            ax2.set_xlabel('MFCC Coefficient')
            ax2.set_ylabel('Value')
        
        # Lyrical & Vocal Analysis
        ax3 = axes[1, 0]
        lyrical = results['lyrical_vocal']
        if 'sentiment_polarity' in lyrical and 'sentiment_subjectivity' in lyrical:
            ax3.scatter([lyrical['sentiment_polarity']], [lyrical['sentiment_subjectivity']], s=100)
            ax3.set_xlim(-1, 1)
            ax3.set_ylim(0, 1)
            ax3.set_xlabel('Sentiment Polarity')
            ax3.set_ylabel('Sentiment Subjectivity')
            ax3.set_title('Sentiment Analysis')
            ax3.grid(True)
        
        ax4 = axes[1, 1]
        if 'vocal_contrast' in lyrical:
            ax4.bar(range(len(lyrical['vocal_contrast'])), lyrical['vocal_contrast'])
            ax4.set_title('Vocal Spectral Contrast')
            ax4.set_xlabel('Frequency Band')
            ax4.set_ylabel('Contrast')
        
        # Sonic & Spectral Analysis
        ax5 = axes[2, 0]
        sonic = results['sonic_spectral']
        if all(k in sonic for k in ['bass_energy_ratio', 'mid_energy_ratio', 'treble_energy_ratio']):
            energies = [sonic['bass_energy_ratio'], sonic['mid_energy_ratio'], sonic['treble_energy_ratio']]
            labels = ['Bass', 'Mid', 'Treble']
            ax5.pie(energies, labels=labels, autopct='%1.1f%%')
            ax5.set_title('Frequency Band Energy Distribution')
        
        ax6 = axes[2, 1]
        if 'frequency_response' in sonic:
            freqs = sonic['frequency_response']['frequencies']
            response = sonic['frequency_response']['magnitude']
            ax6.semilogx(freqs, response)
            ax6.set_title('Frequency Response')
            ax6.set_xlabel('Frequency (Hz)')
            ax6.set_ylabel('Magnitude (dB)')
            ax6.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Visualization saved to: {save_path}")
        
        plt.show()

def main():
    """Main execution function with CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Multi-Dimensional Audio Analysis Agent')
    parser.add_argument('audio_file', help='Path to audio file to analyze')
    parser.add_argument('--whisper-model', default='base', choices=['tiny', 'base', 'small', 'medium', 'large'],
                       help='Whisper model size for transcription')
    parser.add_argument('--no-save', action='store_true', help='Don\'t save analysis results')
    parser.add_argument('--visualize', action='store_true', help='Create and show visualization')
    parser.add_argument('--output-dir', help='Output directory for results')
    
    args = parser.parse_args()
    
    # Check if audio file exists
    if not os.path.exists(args.audio_file):
        print(f"Error: Audio file '{args.audio_file}' not found.")
        sys.exit(1)
    
    # Initialize agent
    agent = AudioAnalysisAgent(whisper_model_size=args.whisper_model)
    
    try:
        # Perform analysis
        results = agent.analyze_audio(args.audio_file, save_results=not args.no_save)
        
        # Print summary
        print("\n" + "="*60)
        print("AUDIO ANALYSIS SUMMARY")
        print("="*60)
        
        # File info
        file_info = results['file_info']
        print(f"File: {file_info['path']}")
        print(f"Duration: {file_info['duration_seconds']:.2f} seconds")
        print(f"Sample Rate: {file_info['sample_rate']} Hz")
        
        # Musical analysis
        musical = results['musical_structural']
        print(f"\n📊 MUSICAL & STRUCTURAL ANALYSIS:")
        print(f"  BPM: {musical.get('bpm', 'N/A'):.1f}" if 'bpm' in musical else "  BPM: N/A")
        print(f"  Estimated Key: {musical.get('estimated_key', 'N/A')}")
        print(f"  Estimated Genre: {musical.get('estimated_genre', 'N/A')}")
        
        # Lyrical analysis
        lyrical = results['lyrical_vocal']
        print(f"\n🎤 LYRICAL & VOCAL ANALYSIS:")
        if lyrical.get('transcription'):
            print(f"  Transcription: \"{lyrical['transcription'][:100]}...\"")
            print(f"  Language: {lyrical.get('detected_language', 'N/A')}")
            print(f"  Sentiment: {lyrical.get('sentiment_polarity', 0):.2f}")
        else:
            print("  No vocals/lyrics detected")
        
        # Sonic analysis
        sonic = results['sonic_spectral']
        print(f"\n🔊 SONIC & SPECTRAL ANALYSIS:")
        print(f"  Loudness (LUFS): {sonic.get('integrated_loudness_lufs', 'N/A'):.1f}" if 'integrated_loudness_lufs' in sonic else "  Loudness: N/A")
        print(f"  Dynamic Range: {sonic.get('dynamic_range_db', 'N/A'):.1f} dB" if 'dynamic_range_db' in sonic else "  Dynamic Range: N/A")
        print(f"  Estimated Instruments: {', '.join(sonic.get('estimated_instruments', ['N/A']))}")
        
        print("\n" + "="*60)
        
        # Create visualization if requested
        if args.visualize:
            save_path = None
            if args.output_dir:
                os.makedirs(args.output_dir, exist_ok=True)
                save_path = os.path.join(args.output_dir, Path(args.audio_file).stem + '_analysis.png')
            
            agent.create_visualization(results, save_path)
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()