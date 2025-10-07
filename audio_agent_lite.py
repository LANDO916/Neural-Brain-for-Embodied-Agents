#!/usr/bin/env python3
"""
Multi-Dimensional Audio Analysis Agent - Lite Version
Free and open-source implementation

This agent analyzes audio across three dimensions:
1. Musical & Structural Analysis
2. Lyrical & Vocal Analysis 
3. Sonic & Spectral Analysis
"""

import os
import sys
import json
import warnings
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Dict, List, Optional, Any
import concurrent.futures
from datetime import datetime

# Text processing
try:
    import speech_recognition as sr
    from textblob import TextBlob
    import nltk
    HAS_SPEECH = True
except ImportError:
    HAS_SPEECH = False
    print("Warning: Speech processing libraries not available")

# Audio processing
try:
    from pydub import AudioSegment
    HAS_PYDUB = True
except ImportError:
    HAS_PYDUB = False

warnings.filterwarnings('ignore')

class AudioAnalysisAgent:
    """Multi-dimensional audio analysis agent using free libraries"""
    
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
        self.results = {}
        
        # Initialize speech recognition if available
        if HAS_SPEECH:
            self.recognizer = sr.Recognizer()
            
        print("🎵 Audio Analysis Agent Initialized")
        print(f"   Sample Rate: {sample_rate} Hz")
        print(f"   Speech Recognition: {'✅' if HAS_SPEECH else '❌'}")
        print(f"   Audio Format Support: {'✅' if HAS_PYDUB else '❌'}")
    
    def analyze_audio(self, audio_path: str) -> Dict[str, Any]:
        """
        Perform comprehensive audio analysis across all three dimensions
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Dictionary containing all analysis results
        """
        print(f"\n🔍 Analyzing: {Path(audio_path).name}")
        
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
                "analysis_time": datetime.now().isoformat()
            },
            "musical_structural": {},
            "lyrical_vocal": {},
            "sonic_spectral": {}
        }
        
        # Parallel analysis across three dimensions
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            # Submit all three analysis tasks
            musical_future = executor.submit(self._analyze_musical_structural, y, sr)
            vocal_future = executor.submit(self._analyze_lyrical_vocal, y, sr, audio_path)
            spectral_future = executor.submit(self._analyze_sonic_spectral, y, sr)
            
            # Collect results
            results["musical_structural"] = musical_future.result()
            results["lyrical_vocal"] = vocal_future.result()
            results["sonic_spectral"] = spectral_future.result()
        
        self.results = results
        return results
    
    def _analyze_musical_structural(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Musical & Structural Analysis"""
        print("🎼 Musical & Structural Analysis...")
        
        analysis = {}
        
        try:
            # Rhythm & Groove Analysis
            print("   → Rhythm & Groove")
            
            # Tempo and beat tracking (with error handling)
            try:
                onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
                onset_times = librosa.frames_to_time(onset_frames, sr=sr)
                
                # Simple tempo estimation from onsets
                if len(onset_times) > 1:
                    intervals = np.diff(onset_times)
                    avg_interval = np.mean(intervals)
                    estimated_tempo = 60.0 / avg_interval if avg_interval > 0 else 0
                else:
                    estimated_tempo = 0
                
                analysis["rhythm"] = {
                    "estimated_tempo": round(estimated_tempo, 1),
                    "onsets_detected": len(onset_times),
                    "rhythmic_regularity": float(np.std(intervals)) if len(intervals) > 0 else 0.0
                }
            except Exception as e:
                analysis["rhythm"] = {"error": str(e), "estimated_tempo": 0}
            
            # Harmony & Key Analysis
            print("   → Harmony & Key")
            try:
                # Chromagram for harmonic analysis
                chroma = librosa.feature.chroma_stft(y=y, sr=sr)
                chroma_mean = np.mean(chroma, axis=1)
                
                # Simple key estimation (most prominent chroma bin)
                key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
                estimated_key = key_names[np.argmax(chroma_mean)]
                
                analysis["harmony"] = {
                    "estimated_key": estimated_key,
                    "tonal_clarity": float(np.max(chroma_mean) / np.mean(chroma_mean)),
                    "harmonic_complexity": float(np.std(chroma_mean))
                }
            except Exception as e:
                analysis["harmony"] = {"error": str(e)}
            
            # Structure Analysis
            print("   → Song Structure")
            try:
                # Spectral clustering for structural segmentation
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                
                # Simple segmentation based on MFCC similarity
                segment_length = int(sr * 4)  # 4-second segments
                segments = []
                
                for i in range(0, len(y), segment_length):
                    end_idx = min(i + segment_length, len(y))
                    if end_idx - i > sr:  # Only process segments longer than 1 second
                        segment_mfcc = librosa.feature.mfcc(y=y[i:end_idx], sr=sr, n_mfcc=13)
                        segments.append({
                            "start_time": i / sr,
                            "end_time": end_idx / sr,
                            "mfcc_mean": np.mean(segment_mfcc).item()
                        })
                
                analysis["structure"] = {
                    "total_segments": len(segments),
                    "segments": segments[:10]  # Limit output size
                }
            except Exception as e:
                analysis["structure"] = {"error": str(e)}
                
        except Exception as e:
            analysis["error"] = str(e)
        
        return analysis
    
    def _analyze_lyrical_vocal(self, y: np.ndarray, sr: int, audio_path: str) -> Dict[str, Any]:
        """Lyrical & Vocal Analysis"""
        print("🎤 Lyrical & Vocal Analysis...")
        
        analysis = {}
        
        try:
            # Voice Activity Detection
            print("   → Voice Activity Detection")
            
            # Simple voice activity detection using spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            
            # Heuristic for voice presence (spectral characteristics typical of speech)
            voice_indicators = []
            for i in range(len(spectral_centroids)):
                # Voice typically has specific spectral characteristics
                if (1000 < spectral_centroids[i] < 4000 and 
                    2000 < spectral_rolloff[i] < 8000 and
                    0.1 < zcr[i] < 0.3):
                    voice_indicators.append(1)
                else:
                    voice_indicators.append(0)
            
            voice_ratio = np.mean(voice_indicators)
            
            analysis["voice_activity"] = {
                "voice_presence_ratio": float(voice_ratio),
                "likely_contains_vocals": voice_ratio > 0.3,
                "mean_spectral_centroid": float(np.mean(spectral_centroids)),
                "mean_spectral_rolloff": float(np.mean(spectral_rolloff))
            }
            
            # Speech-to-Text (if available and likely contains speech)
            if HAS_SPEECH and voice_ratio > 0.2:
                print("   → Speech Recognition")
                try:
                    # Convert to format suitable for speech recognition
                    if HAS_PYDUB:
                        # Use pydub for better format handling
                        audio_segment = AudioSegment.from_file(audio_path)
                        wav_path = "temp_speech.wav"
                        audio_segment.export(wav_path, format="wav")
                        
                        with sr.AudioFile(wav_path) as source:
                            audio_data = self.recognizer.record(source, duration=min(60, len(y)/sr))
                            
                        # Try speech recognition
                        try:
                            text = self.recognizer.recognize_google(audio_data, language='en-US')
                            analysis["speech_recognition"] = {
                                "transcribed_text": text,
                                "confidence": "medium",
                                "language": "en-US"
                            }
                            
                            # Text analysis if we got transcription
                            if text:
                                print("   → Text Analysis")
                                blob = TextBlob(text)
                                analysis["text_analysis"] = {
                                    "word_count": len(text.split()),
                                    "sentiment_polarity": blob.sentiment.polarity,
                                    "sentiment_subjectivity": blob.sentiment.subjectivity,
                                    "language_detected": str(blob.detect_language()) if hasattr(blob, 'detect_language') else 'en'
                                }
                        except sr.UnknownValueError:
                            analysis["speech_recognition"] = {"error": "Could not understand audio"}
                        except sr.RequestError as e:
                            analysis["speech_recognition"] = {"error": f"Service error: {e}"}
                        
                        # Clean up temp file
                        if os.path.exists(wav_path):
                            os.remove(wav_path)
                    else:
                        analysis["speech_recognition"] = {"error": "Audio format conversion not available"}
                        
                except Exception as e:
                    analysis["speech_recognition"] = {"error": str(e)}
            else:
                analysis["speech_recognition"] = {"note": "Voice activity too low or speech libraries not available"}
            
            # Vocal Characteristics Analysis
            print("   → Vocal Characteristics")
            if voice_ratio > 0.1:
                try:
                    # Pitch analysis for vocal sections
                    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
                    
                    # Extract vocal pitch estimates
                    vocal_pitches = []
                    for t in range(pitches.shape[1]):
                        index = magnitudes[:, t].argmax()
                        pitch = pitches[index, t]
                        if pitch > 80 and pitch < 400:  # Typical vocal range
                            vocal_pitches.append(pitch)
                    
                    if vocal_pitches:
                        analysis["vocal_characteristics"] = {
                            "estimated_vocal_range": {
                                "min_pitch": float(np.min(vocal_pitches)),
                                "max_pitch": float(np.max(vocal_pitches)),
                                "mean_pitch": float(np.mean(vocal_pitches))
                            },
                            "pitch_variability": float(np.std(vocal_pitches)),
                            "vocal_frames_detected": len(vocal_pitches)
                        }
                    else:
                        analysis["vocal_characteristics"] = {"note": "No clear vocal pitch detected"}
                        
                except Exception as e:
                    analysis["vocal_characteristics"] = {"error": str(e)}
            
        except Exception as e:
            analysis["error"] = str(e)
        
        return analysis
    
    def _analyze_sonic_spectral(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Sonic & Spectral Analysis"""
        print("🔊 Sonic & Spectral Analysis...")
        
        analysis = {}
        
        try:
            # Spectral Features
            print("   → Spectral Features")
            
            # Basic spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
            
            analysis["spectral_features"] = {
                "spectral_centroid": {
                    "mean": float(np.mean(spectral_centroids)),
                    "std": float(np.std(spectral_centroids)),
                    "max": float(np.max(spectral_centroids)),
                    "min": float(np.min(spectral_centroids))
                },
                "spectral_rolloff": {
                    "mean": float(np.mean(spectral_rolloff)),
                    "std": float(np.std(spectral_rolloff))
                },
                "spectral_bandwidth": {
                    "mean": float(np.mean(spectral_bandwidth)),
                    "std": float(np.std(spectral_bandwidth))
                },
                "zero_crossing_rate": {
                    "mean": float(np.mean(zero_crossing_rate)),
                    "std": float(np.std(zero_crossing_rate))
                }
            }
            
            # Frequency Analysis
            print("   → Frequency Analysis")
            
            # Compute FFT for frequency analysis
            fft = np.fft.fft(y)
            magnitude = np.abs(fft)
            freqs = np.fft.fftfreq(len(magnitude), 1/sr)
            
            # Find prominent frequencies
            positive_freqs = freqs[:len(freqs)//2]
            positive_magnitude = magnitude[:len(magnitude)//2]
            
            # Find peaks
            peak_indices = []
            for i in range(1, len(positive_magnitude)-1):
                if (positive_magnitude[i] > positive_magnitude[i-1] and 
                    positive_magnitude[i] > positive_magnitude[i+1] and
                    positive_magnitude[i] > np.max(positive_magnitude) * 0.1):
                    peak_indices.append(i)
            
            prominent_frequencies = [positive_freqs[i] for i in peak_indices[:10]]
            
            analysis["frequency_analysis"] = {
                "dominant_frequency": float(positive_freqs[np.argmax(positive_magnitude)]),
                "prominent_frequencies": [float(f) for f in prominent_frequencies],
                "frequency_range": {
                    "min": 0.0,
                    "max": float(sr / 2)
                }
            }
            
            # Dynamic Range Analysis
            print("   → Dynamic Range")
            
            # RMS energy over time
            rms = librosa.feature.rms(y=y)[0]
            
            analysis["dynamics"] = {
                "rms_energy": {
                    "mean": float(np.mean(rms)),
                    "std": float(np.std(rms)),
                    "max": float(np.max(rms)),
                    "min": float(np.min(rms))
                },
                "dynamic_range_db": float(20 * np.log10(np.max(rms) / (np.min(rms) + 1e-10))),
                "loudness_variability": float(np.std(rms))
            }
            
            # Harmonic-Percussive Separation
            print("   → Harmonic/Percussive Analysis")
            try:
                y_harmonic, y_percussive = librosa.effects.hpss(y)
                
                harmonic_energy = np.sum(y_harmonic**2)
                percussive_energy = np.sum(y_percussive**2)
                total_energy = harmonic_energy + percussive_energy
                
                analysis["harmonic_percussive"] = {
                    "harmonic_ratio": float(harmonic_energy / total_energy),
                    "percussive_ratio": float(percussive_energy / total_energy),
                    "harmonic_energy": float(harmonic_energy),
                    "percussive_energy": float(percussive_energy)
                }
            except Exception as e:
                analysis["harmonic_percussive"] = {"error": str(e)}
            
        except Exception as e:
            analysis["error"] = str(e)
        
        return analysis
    
    def save_results(self, output_path: str = None) -> str:
        """Save analysis results to JSON file"""
        if not self.results:
            raise ValueError("No analysis results to save. Run analyze_audio() first.")
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"audio_analysis_{timestamp}.json"
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        serializable_results = convert_numpy_types(self.results)
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"💾 Results saved to: {output_path}")
        return output_path
    
    def print_summary(self):
        """Print a human-readable summary of the analysis"""
        if not self.results:
            print("No analysis results available.")
            return
        
        print("\n" + "="*60)
        print("🎵 AUDIO ANALYSIS SUMMARY")
        print("="*60)
        
        # Metadata
        meta = self.results.get("metadata", {})
        print(f"📁 File: {Path(meta.get('file', 'Unknown')).name}")
        print(f"⏱️  Duration: {meta.get('duration', 0):.2f} seconds")
        print(f"🎚️  Sample Rate: {meta.get('sample_rate', 0)} Hz")
        
        # Musical Analysis
        print("\n🎼 MUSICAL & STRUCTURAL ANALYSIS")
        print("-" * 40)
        musical = self.results.get("musical_structural", {})
        
        if "rhythm" in musical:
            rhythm = musical["rhythm"]
            if "estimated_tempo" in rhythm:
                print(f"   Tempo: {rhythm['estimated_tempo']} BPM")
                print(f"   Onsets: {rhythm.get('onsets_detected', 'N/A')}")
        
        if "harmony" in musical:
            harmony = musical["harmony"]
            if "estimated_key" in harmony:
                print(f"   Key: {harmony['estimated_key']}")
                print(f"   Tonal Clarity: {harmony.get('tonal_clarity', 0):.2f}")
        
        # Vocal Analysis
        print("\n🎤 LYRICAL & VOCAL ANALYSIS")
        print("-" * 40)
        vocal = self.results.get("lyrical_vocal", {})
        
        if "voice_activity" in vocal:
            voice = vocal["voice_activity"]
            print(f"   Voice Presence: {voice.get('voice_presence_ratio', 0)*100:.1f}%")
            print(f"   Contains Vocals: {'Yes' if voice.get('likely_contains_vocals', False) else 'No'}")
        
        if "speech_recognition" in vocal and "transcribed_text" in vocal["speech_recognition"]:
            text = vocal["speech_recognition"]["transcribed_text"]
            print(f"   Transcribed: \"{text[:100]}{'...' if len(text) > 100 else ''}\"")
        
        if "text_analysis" in vocal:
            text_analysis = vocal["text_analysis"]
            sentiment = text_analysis.get("sentiment_polarity", 0)
            sentiment_label = "Positive" if sentiment > 0.1 else "Negative" if sentiment < -0.1 else "Neutral"
            print(f"   Sentiment: {sentiment_label} ({sentiment:.2f})")
        
        # Spectral Analysis
        print("\n🔊 SONIC & SPECTRAL ANALYSIS")
        print("-" * 40)
        spectral = self.results.get("sonic_spectral", {})
        
        if "spectral_features" in spectral:
            features = spectral["spectral_features"]
            if "spectral_centroid" in features:
                centroid = features["spectral_centroid"]["mean"]
                print(f"   Spectral Centroid: {centroid:.0f} Hz")
        
        if "frequency_analysis" in spectral:
            freq = spectral["frequency_analysis"]
            print(f"   Dominant Frequency: {freq.get('dominant_frequency', 0):.0f} Hz")
        
        if "dynamics" in spectral:
            dynamics = spectral["dynamics"]
            print(f"   Dynamic Range: {dynamics.get('dynamic_range_db', 0):.1f} dB")
        
        if "harmonic_percussive" in spectral:
            hp = spectral["harmonic_percussive"]
            harmonic_pct = hp.get("harmonic_ratio", 0) * 100
            percussive_pct = hp.get("percussive_ratio", 0) * 100
            print(f"   Harmonic: {harmonic_pct:.1f}% | Percussive: {percussive_pct:.1f}%")
        
        print("\n" + "="*60)

# Demo function
def main():
    """Demo the audio analysis agent"""
    print("🎵 Multi-Dimensional Audio Analysis Agent")
    print("==========================================")
    
    # Initialize agent
    agent = AudioAnalysisAgent()
    
    # Check if we have test audio
    test_files = ["test_audio.wav", "demo_song.wav"]
    audio_file = None
    
    for test_file in test_files:
        if os.path.exists(test_file):
            audio_file = test_file
            break
    
    if not audio_file:
        print("\n⚠️  No test audio found. Creating demo audio...")
        # Create simple test audio
        sr = 22050
        duration = 10.0
        t = np.linspace(0, duration, int(sr * duration))
        
        # Create a simple melody with rhythm
        frequencies = [440, 523, 659, 784]  # C major chord frequencies
        signal = np.zeros_like(t)
        
        for i, freq in enumerate(frequencies):
            # Add each frequency with different timing
            start_time = i * 2.0
            end_time = start_time + 3.0
            mask = (t >= start_time) & (t <= end_time)
            signal[mask] += 0.25 * np.sin(2 * np.pi * freq * t[mask])
        
        # Add rhythm
        beat_freq = 2.0  # 120 BPM
        envelope = 0.5 * (1 + np.sin(2 * np.pi * beat_freq * t))
        signal *= envelope
        
        # Normalize
        signal = signal / np.max(np.abs(signal))
        
        audio_file = "demo_audio.wav"
        sf.write(audio_file, signal, sr)
        print(f"✅ Created demo audio: {audio_file}")
    
    # Analyze the audio
    print(f"\n🔍 Analyzing: {audio_file}")
    results = agent.analyze_audio(audio_file)
    
    # Display results
    agent.print_summary()
    
    # Save results
    output_file = agent.save_results()
    
    print(f"\n✨ Analysis complete!")
    print(f"📄 Detailed results saved to: {output_file}")
    
    return results

if __name__ == "__main__":
    main()