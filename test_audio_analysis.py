#!/usr/bin/env python3
"""
Test script for the Audio Analysis Agent
Generates a simple test audio file and runs analysis on it
"""

import numpy as np
import soundfile as sf
import os
from audio_analysis_agent import AudioAnalyzer

def generate_test_audio(filename="test_audio.wav", duration=5.0, sample_rate=22050):
    """Generate a simple test audio file with multiple components"""
    
    print(f"Generating test audio file: {filename}")
    print(f"Duration: {duration} seconds, Sample Rate: {sample_rate} Hz")
    
    # Time array
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # Generate a simple melody (440 Hz A note with some variation)
    frequency = 440.0  # A4 note
    melody = 0.3 * np.sin(2 * np.pi * frequency * t)
    
    # Add some harmonic content
    melody += 0.1 * np.sin(2 * np.pi * frequency * 2 * t)  # Second harmonic
    melody += 0.05 * np.sin(2 * np.pi * frequency * 3 * t)  # Third harmonic
    
    # Add some rhythm (simple beat)
    beat_frequency = 2.0  # 2 Hz = 120 BPM
    rhythm = 0.1 * np.sin(2 * np.pi * beat_frequency * t)
    
    # Combine melody and rhythm
    audio = melody + rhythm
    
    # Add some noise for realism
    noise = 0.01 * np.random.randn(len(audio))
    audio += noise
    
    # Normalize audio
    audio = audio / np.max(np.abs(audio))
    
    # Save as WAV file
    sf.write(filename, audio, sample_rate)
    
    print(f"Test audio saved as: {filename}")
    return filename

def test_analysis():
    """Test the audio analysis agent"""
    
    # Generate test audio
    test_file = generate_test_audio()
    
    print("\n" + "="*60)
    print("TESTING AUDIO ANALYSIS AGENT")
    print("="*60)
    
    # Create analyzer
    analyzer = AudioAnalyzer()
    
    # Run analysis
    try:
        result = analyzer.analyze_audio(test_file)
        
        print("\n" + "="*60)
        print("ANALYSIS RESULTS SUMMARY")
        print("="*60)
        
        # Print key results
        musical = result.musical_structural
        sonic = result.sonic_spectral
        
        # BPM
        if "bpm" in musical["bpm_and_tempo"]:
            print(f"🎵 BPM: {musical['bpm_and_tempo']['bpm']:.1f}")
        
        # Key
        if "key" in musical["tonal_center"]:
            print(f"🎼 Key: {musical['tonal_center']['key']} {musical['tonal_center']['mode']}")
        
        # Loudness
        if "loudness_db" in sonic["loudness"]:
            print(f"🔊 Loudness: {sonic['loudness']['loudness_db']:.1f} dB")
        
        # Spectral balance
        if "spectral_balance" in sonic["spectral_power"]:
            print(f"📊 Spectral Balance: {sonic['spectral_power']['spectral_balance']}")
        
        # Stereo info
        if "stereo_type" in sonic["stereo_image"]:
            print(f"🎧 Audio Type: {sonic['stereo_image']['stereo_type']}")
        
        print("\n✅ Analysis completed successfully!")
        print(f"📄 Detailed results saved to: {test_file.replace('.wav', '_analysis.json')}")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        print("This might be due to missing dependencies.")
        print("Try installing the required packages:")
        print("pip install -r requirements.txt")

if __name__ == "__main__":
    test_analysis()