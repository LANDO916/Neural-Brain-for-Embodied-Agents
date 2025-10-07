#!/usr/bin/env python3
"""
Demo script for the Multi-Dimensional Audio Analysis Agent
This script demonstrates the key capabilities of the agent.
"""

import os
import sys
import tempfile
import numpy as np
from pathlib import Path

# Add the current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from audio_analysis_agent import AudioAnalysisAgent

def create_demo_audio(filename: str = "demo_song.wav", duration: float = 10.0) -> str:
    """
    Create a simple demo audio file for testing purposes
    
    Args:
        filename: Name of the output file
        duration: Duration in seconds
        
    Returns:
        Path to the created audio file
    """
    try:
        import librosa
        import soundfile as sf
        
        # Create a simple synthetic audio with multiple components
        sr = 22050
        t = np.linspace(0, duration, int(duration * sr))
        
        # Base melody (C major scale)
        melody_freqs = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25]  # C4 to C5
        melody = np.zeros_like(t)
        
        # Add melody notes
        note_duration = duration / len(melody_freqs)
        for i, freq in enumerate(melody_freqs):
            start_idx = int(i * note_duration * sr)
            end_idx = int((i + 1) * note_duration * sr)
            if end_idx > len(t):
                end_idx = len(t)
            melody[start_idx:end_idx] += 0.3 * np.sin(2 * np.pi * freq * t[start_idx:end_idx])
        
        # Add harmonics
        harmony = 0.15 * np.sin(2 * np.pi * 130.81 * t)  # C3 bass line
        
        # Add percussion (kick drum pattern)
        kick_times = np.arange(0, duration, 0.5)  # Every half second
        percussion = np.zeros_like(t)
        for kick_time in kick_times:
            if kick_time < duration:
                kick_idx = int(kick_time * sr)
                # Simple kick drum envelope
                envelope_len = int(0.1 * sr)
                if kick_idx + envelope_len < len(t):
                    envelope = np.exp(-5 * np.linspace(0, 1, envelope_len))
                    percussion[kick_idx:kick_idx + envelope_len] += 0.4 * envelope * np.sin(2 * np.pi * 60 * t[kick_idx:kick_idx + envelope_len])
        
        # Combine all components
        audio = melody + harmony + percussion
        
        # Add some reverb-like effect
        audio = audio + 0.1 * np.concatenate([np.zeros(int(0.05 * sr)), audio[:-int(0.05 * sr)]])
        
        # Normalize
        audio = audio / np.max(np.abs(audio)) * 0.8
        
        # Save the audio file
        output_path = Path(filename)
        sf.write(output_path, audio, sr)
        print(f"✅ Created demo audio file: {output_path} ({duration:.1f}s)")
        return str(output_path)
        
    except Exception as e:
        print(f"⚠️  Could not create demo audio: {e}")
        print("Please provide your own audio file for analysis.")
        return None

def print_analysis_summary(results: dict) -> None:
    """Print a formatted summary of the analysis results"""
    print("\n" + "="*80)
    print("🎵 MULTI-DIMENSIONAL AUDIO ANALYSIS RESULTS")
    print("="*80)
    
    # File information
    file_info = results.get('file_info', {})
    print(f"\n📁 FILE INFORMATION:")
    print(f"   📄 File: {file_info.get('path', 'Unknown')}")
    print(f"   ⏱️  Duration: {file_info.get('duration_seconds', 0):.2f} seconds")
    print(f"   🎚️  Sample Rate: {file_info.get('sample_rate', 0)} Hz")
    
    # Musical & Structural Analysis
    musical = results.get('musical_structural', {})
    print(f"\n🎼 MUSICAL & STRUCTURAL ANALYSIS:")
    print(f"   🥁 BPM: {musical.get('bpm', 0):.1f}")
    print(f"   🎹 Estimated Key: {musical.get('estimated_key', 'Unknown')}")
    print(f"   🎵 Key Confidence: {musical.get('key_confidence', 0):.3f}")
    print(f"   🎭 Estimated Genre: {musical.get('estimated_genre', 'Unknown')}")
    print(f"   🎪 Tempo Variation (std): {musical.get('tempo_variation_std', 0):.4f}")
    print(f"   🎯 Spectral Centroid: {musical.get('spectral_centroid_mean', 0):.1f} Hz")
    
    # Lyrical & Vocal Analysis
    lyrical = results.get('lyrical_vocal', {})
    print(f"\n🎤 LYRICAL & VOCAL ANALYSIS:")
    if lyrical.get('transcription'):
        print(f"   📝 Transcription: \"{lyrical['transcription'][:100]}{'...' if len(lyrical['transcription']) > 100 else ''}\"")
        print(f"   🌍 Language: {lyrical.get('detected_language', 'Unknown')}")
        print(f"   😊 Sentiment Polarity: {lyrical.get('sentiment_polarity', 0):.3f} (-1=negative, +1=positive)")
        print(f"   🧠 Sentiment Subjectivity: {lyrical.get('sentiment_subjectivity', 0):.3f} (0=objective, 1=subjective)")
        print(f"   📊 Word Count: {lyrical.get('word_count', 0)}")
        print(f"   🔤 Lexical Diversity: {lyrical.get('lexical_diversity', 0):.3f}")
    else:
        print("   🔇 No vocals/lyrics detected")
    
    print(f"   🎵 Fundamental Frequency: {lyrical.get('fundamental_frequency_mean', 0):.1f} Hz")
    print(f"   🌟 Vocal Brightness: {lyrical.get('vocal_brightness', 0):.1f} Hz")
    print(f"   🎶 Vocal Harmonicity: {lyrical.get('vocal_harmonicity', 0):.3f}")
    
    # Sonic & Spectral Analysis
    sonic = results.get('sonic_spectral', {})
    print(f"\n🔊 SONIC & SPECTRAL ANALYSIS:")
    print(f"   📢 Loudness (LUFS): {sonic.get('integrated_loudness_lufs', 0):.1f}")
    print(f"   📊 Dynamic Range: {sonic.get('dynamic_range_db', 0):.1f} dB")
    print(f"   🎭 Crest Factor: {sonic.get('crest_factor', 0):.2f}")
    print(f"   🎸 Harmonic Ratio: {sonic.get('harmonic_ratio', 0):.3f}")
    print(f"   🥁 Percussive Ratio: {sonic.get('percussive_ratio', 0):.3f}")
    
    # Energy distribution
    bass_ratio = sonic.get('bass_energy_ratio', 0)
    mid_ratio = sonic.get('mid_energy_ratio', 0)
    treble_ratio = sonic.get('treble_energy_ratio', 0)
    print(f"   🔊 Energy Distribution:")
    print(f"      🔈 Bass (≤250Hz): {bass_ratio:.1%}")
    print(f"      🔉 Mid (250Hz-4kHz): {mid_ratio:.1%}")
    print(f"      🔊 Treble (>4kHz): {treble_ratio:.1%}")
    
    # Estimated instruments
    instruments = sonic.get('estimated_instruments', [])
    print(f"   🎺 Estimated Instruments: {', '.join(instruments)}")
    
    # Reverb characteristics
    rt60 = sonic.get('estimated_rt60', 0)
    print(f"   🏢 Estimated RT60 (reverb): {rt60:.2f} seconds")
    
    print("\n" + "="*80)
    print("✨ Analysis completed using 100% free and open-source libraries!")
    print("📊 Libraries used: librosa, faster-whisper, scikit-learn, textblob, scipy")
    print("="*80)

def run_demo():
    """Run the complete demo"""
    print("🎵 Multi-Dimensional Audio Analysis Agent Demo")
    print("=" * 50)
    print("This demo showcases the parallel deconstruction of audio across three dimensions:")
    print("1. 🎼 Musical & Structural Analysis")
    print("2. 🎤 Lyrical & Vocal Analysis")
    print("3. 🔊 Sonic & Spectral Analysis")
    print("=" * 50)
    
    # Check if user provided an audio file
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        if not os.path.exists(audio_file):
            print(f"❌ Error: Audio file '{audio_file}' not found.")
            return 1
    else:
        # Create demo audio
        print("\n🎵 No audio file provided, creating demo audio...")
        audio_file = create_demo_audio()
        if not audio_file:
            print("❌ Could not create demo audio. Please provide an audio file as argument.")
            print("Usage: python demo.py [audio_file.wav]")
            return 1
    
    print(f"\n🎯 Analyzing audio file: {audio_file}")
    
    try:
        # Initialize the agent
        print("\n🤖 Initializing Audio Analysis Agent...")
        agent = AudioAnalysisAgent(whisper_model_size="tiny")  # Use small model for demo
        
        # Perform analysis
        print("🔍 Starting parallel analysis across all dimensions...")
        print("   ⏳ This may take a moment for the first run (downloading models)...")
        
        results = agent.analyze_audio(audio_file, save_results=True)
        
        # Display results
        print_analysis_summary(results)
        
        # Create visualization
        try:
            print("\n📊 Generating visualization...")
            agent.create_visualization(results, f"{Path(audio_file).stem}_analysis.png")
            print("✅ Visualization saved!")
        except Exception as e:
            print(f"⚠️  Visualization failed: {e}")
        
        # Show JSON results location
        json_file = Path(audio_file).with_suffix('.analysis.json')
        if json_file.exists():
            print(f"\n💾 Detailed results saved to: {json_file}")
        
        print("\n🎉 Demo completed successfully!")
        
        # Clean up demo file if we created it
        if len(sys.argv) <= 1 and audio_file and os.path.exists(audio_file):
            try:
                os.remove(audio_file)
                print(f"🧹 Cleaned up demo file: {audio_file}")
            except:
                pass
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        print("\n🔧 Troubleshooting tips:")
        print("   • Make sure all dependencies are installed: pip install -r requirements.txt")
        print("   • Check that the audio file is in a supported format (WAV, MP3, FLAC, etc.)")
        print("   • Ensure you have sufficient disk space for model downloads")
        return 1

if __name__ == "__main__":
    sys.exit(run_demo())