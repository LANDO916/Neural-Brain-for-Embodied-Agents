#!/usr/bin/env python3
"""
Example usage of the Multi-Dimensional Audio Analysis Agent

This script demonstrates different ways to use the agent
for various types of audio analysis.
"""

import os
import numpy as np
import soundfile as sf
from audio_agent_lite import AudioAnalysisAgent

def create_sample_music():
    """Create a sample music file for testing"""
    print("🎵 Creating sample music file...")
    
    sr = 22050
    duration = 8.0
    t = np.linspace(0, duration, int(sr * duration))
    
    # Create a simple chord progression (C-Am-F-G)
    chords = [
        [262, 330, 392],  # C major (C-E-G)
        [220, 262, 330],  # A minor (A-C-E) 
        [175, 220, 262],  # F major (F-A-C)
        [196, 247, 294]   # G major (G-B-D)
    ]
    
    signal = np.zeros_like(t)
    chord_duration = duration / len(chords)
    
    for i, chord in enumerate(chords):
        start_time = i * chord_duration
        end_time = (i + 1) * chord_duration
        mask = (t >= start_time) & (t < end_time)
        
        for freq in chord:
            signal[mask] += 0.2 * np.sin(2 * np.pi * freq * t[mask])
    
    # Add rhythm
    beat_freq = 2.0  # 120 BPM
    envelope = 0.3 + 0.7 * np.abs(np.sin(2 * np.pi * beat_freq * t))
    signal *= envelope
    
    # Add some harmonics
    signal += 0.1 * np.sin(4 * np.pi * 440 * t)  # Second harmonic of A
    
    # Normalize
    signal = signal / np.max(np.abs(signal))
    
    filename = "sample_music.wav"
    sf.write(filename, signal, sr)
    print(f"✅ Created: {filename}")
    return filename

def create_sample_speech():
    """Create a sample 'speech-like' audio for testing voice detection"""
    print("🎤 Creating sample speech-like audio...")
    
    sr = 22050
    duration = 5.0
    t = np.linspace(0, duration, int(sr * duration))
    
    # Create speech-like formants (approximating vowel sounds)
    formants = [
        (800, 1200, 2500),   # /a/ sound
        (400, 2000, 2550),   # /i/ sound  
        (500, 1000, 2300),   # /o/ sound
    ]
    
    signal = np.zeros_like(t)
    segment_duration = duration / len(formants)
    
    for i, (f1, f2, f3) in enumerate(formants):
        start_time = i * segment_duration
        end_time = (i + 1) * segment_duration
        mask = (t >= start_time) & (t < end_time)
        
        # Fundamental frequency (pitch)
        f0 = 150  # Typical male voice
        segment_signal = 0.3 * np.sin(2 * np.pi * f0 * t[mask])
        
        # Add formants
        segment_signal += 0.2 * np.sin(2 * np.pi * f1 * t[mask])
        segment_signal += 0.15 * np.sin(2 * np.pi * f2 * t[mask])
        segment_signal += 0.1 * np.sin(2 * np.pi * f3 * t[mask])
        
        signal[mask] = segment_signal
    
    # Add speech-like modulation
    modulation = np.sin(2 * np.pi * 5 * t)  # 5 Hz modulation
    signal *= (0.8 + 0.2 * modulation)
    
    # Add noise for realism
    noise = 0.05 * np.random.normal(0, 1, signal.shape)
    signal += noise
    
    # Normalize
    signal = signal / np.max(np.abs(signal))
    
    filename = "sample_speech.wav"
    sf.write(filename, signal, sr)
    print(f"✅ Created: {filename}")
    return filename

def create_sample_percussion():
    """Create a sample percussion pattern"""
    print("🥁 Creating sample percussion audio...")
    
    sr = 22050
    duration = 4.0
    t = np.linspace(0, duration, int(sr * duration))
    
    signal = np.zeros_like(t)
    
    # Create kick drum pattern (every beat)
    kick_times = [0, 1, 2, 3]
    for kick_time in kick_times:
        start_idx = int(kick_time * sr)
        end_idx = min(start_idx + int(0.1 * sr), len(signal))
        
        # Low frequency sine for kick
        kick_t = np.linspace(0, 0.1, end_idx - start_idx)
        kick = np.sin(2 * np.pi * 60 * kick_t) * np.exp(-kick_t * 30)
        signal[start_idx:end_idx] += 0.6 * kick
    
    # Create hi-hat pattern (off-beats)
    hihat_times = [0.5, 1.5, 2.5, 3.5]
    for hihat_time in hihat_times:
        start_idx = int(hihat_time * sr)
        end_idx = min(start_idx + int(0.05 * sr), len(signal))
        
        # High frequency noise for hi-hat
        hihat_t = np.linspace(0, 0.05, end_idx - start_idx)
        hihat = np.random.normal(0, 1, len(hihat_t)) * np.exp(-hihat_t * 50)
        # Filter to high frequencies
        hihat = hihat * np.sin(2 * np.pi * 8000 * hihat_t)
        signal[start_idx:end_idx] += 0.3 * hihat
    
    # Normalize
    signal = signal / np.max(np.abs(signal))
    
    filename = "sample_percussion.wav"
    sf.write(filename, signal, sr)
    print(f"✅ Created: {filename}")
    return filename

def analyze_file(agent, filepath, description):
    """Analyze a single audio file and display results"""
    print(f"\n{'='*60}")
    print(f"🔍 ANALYZING: {description}")
    print(f"📁 File: {filepath}")
    print('='*60)
    
    try:
        # Perform analysis
        results = agent.analyze_audio(filepath)
        
        # Display summary
        agent.print_summary()
        
        # Save detailed results
        output_filename = f"analysis_{os.path.splitext(os.path.basename(filepath))[0]}.json"
        agent.save_results(output_filename)
        
        print(f"📊 Detailed analysis saved to: {output_filename}")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")

def main():
    """Main demonstration function"""
    print("🎵 Multi-Dimensional Audio Analysis Agent - Examples")
    print("=" * 55)
    
    # Initialize the agent
    agent = AudioAnalysisAgent()
    
    # Create sample audio files
    print("\n📁 Creating sample audio files for demonstration...")
    music_file = create_sample_music()
    speech_file = create_sample_speech()
    percussion_file = create_sample_percussion()
    
    # Analyze each type of audio
    analyze_file(agent, music_file, "Harmonic Music (Chord Progression)")
    analyze_file(agent, speech_file, "Speech-like Audio (Formant Patterns)")
    analyze_file(agent, percussion_file, "Percussion Pattern (Rhythmic)")
    
    # Summary
    print(f"\n{'='*60}")
    print("✨ DEMONSTRATION COMPLETE")
    print('='*60)
    print("\n📊 Generated Analysis Files:")
    for filename in ["analysis_sample_music.json", "analysis_sample_speech.json", "analysis_sample_percussion.json"]:
        if os.path.exists(filename):
            print(f"   📄 {filename}")
    
    print(f"\n🎵 Generated Audio Files:")
    for filename in [music_file, speech_file, percussion_file]:
        if os.path.exists(filename):
            print(f"   🔊 {filename}")
    
    print(f"\n💡 Tips:")
    print(f"   • Compare how different audio types show different patterns")
    print(f"   • Music shows strong harmonic content and key detection")
    print(f"   • Speech-like audio shows different spectral characteristics")
    print(f"   • Percussion shows strong percussive vs harmonic ratios")
    print(f"   • All analysis uses FREE, open-source libraries!")

if __name__ == "__main__":
    main()