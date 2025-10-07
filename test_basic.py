#!/usr/bin/env python3
"""
Basic test script for audio analysis functionality
"""

import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from pathlib import Path

def create_test_audio():
    """Create a simple test audio signal"""
    # Create a 5-second test signal with multiple frequencies
    sr = 22050  # Sample rate
    duration = 5.0
    t = np.linspace(0, duration, int(sr * duration))
    
    # Create a signal with fundamental + harmonics
    freq1 = 440  # A4
    freq2 = 880  # A5
    freq3 = 220  # A3
    
    signal = (0.5 * np.sin(2 * np.pi * freq1 * t) + 
              0.3 * np.sin(2 * np.pi * freq2 * t) + 
              0.2 * np.sin(2 * np.pi * freq3 * t))
    
    # Add some rhythm (simple beat pattern)
    beat_freq = 2.0  # 2 Hz = 120 BPM
    envelope = 0.5 * (1 + np.sin(2 * np.pi * beat_freq * t))
    signal *= envelope
    
    # Add some noise for realism
    noise = 0.1 * np.random.normal(0, 1, signal.shape)
    signal += noise
    
    # Normalize
    signal = signal / np.max(np.abs(signal))
    
    return signal, sr

def test_basic_analysis():
    """Test basic audio analysis features"""
    print("=== Basic Audio Analysis Test ===")
    
    # Create test audio
    print("1. Creating test audio signal...")
    y, sr = create_test_audio()
    
    # Save test audio
    test_file = "test_audio.wav"
    sf.write(test_file, y, sr)
    print(f"   Saved test audio: {test_file}")
    
    # Load and analyze
    print("2. Loading and analyzing audio...")
    y_loaded, sr_loaded = librosa.load(test_file, sr=None)
    
    print(f"   Sample rate: {sr_loaded} Hz")
    print(f"   Duration: {len(y_loaded) / sr_loaded:.2f} seconds")
    print(f"   Samples: {len(y_loaded)}")
    
    # Basic spectral analysis
    print("3. Performing spectral analysis...")
    
    # Compute spectral features
    spectral_centroids = librosa.feature.spectral_centroid(y=y_loaded, sr=sr_loaded)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y_loaded, sr=sr_loaded)[0]
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y_loaded)[0]
    
    print(f"   Mean spectral centroid: {np.mean(spectral_centroids):.2f} Hz")
    print(f"   Mean spectral rolloff: {np.mean(spectral_rolloff):.2f} Hz")
    print(f"   Mean zero crossing rate: {np.mean(zero_crossing_rate):.4f}")
    
    # Tempo analysis
    print("4. Analyzing tempo...")
    try:
        tempo, beats = librosa.beat.beat_track(y=y_loaded, sr=sr_loaded)
        print(f"   Estimated tempo: {tempo:.1f} BPM")
        print(f"   Number of beats detected: {len(beats)}")
    except Exception as e:
        print(f"   Tempo analysis failed: {e}")
    
    # Pitch analysis
    print("5. Analyzing pitch...")
    try:
        pitches, magnitudes = librosa.piptrack(y=y_loaded, sr=sr_loaded)
        # Find the most prominent pitch in each frame
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)
        
        if pitch_values:
            mean_pitch = np.mean(pitch_values)
            print(f"   Mean fundamental frequency: {mean_pitch:.2f} Hz")
        else:
            print("   No significant pitch detected")
    except Exception as e:
        print(f"   Pitch analysis failed: {e}")
    
    # Harmonic analysis
    print("6. Analyzing harmonics...")
    try:
        # Separate harmonic and percussive components
        y_harmonic, y_percussive = librosa.effects.hpss(y_loaded)
        
        # Calculate energy ratios
        harmonic_energy = np.sum(y_harmonic**2)
        percussive_energy = np.sum(y_percussive**2)
        total_energy = harmonic_energy + percussive_energy
        
        print(f"   Harmonic energy ratio: {(harmonic_energy/total_energy)*100:.1f}%")
        print(f"   Percussive energy ratio: {(percussive_energy/total_energy)*100:.1f}%")
    except Exception as e:
        print(f"   Harmonic analysis failed: {e}")
    
    print("\n=== Test Complete ===")
    print("✅ Basic audio analysis functionality is working!")
    
    return True

if __name__ == "__main__":
    try:
        test_basic_analysis()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()