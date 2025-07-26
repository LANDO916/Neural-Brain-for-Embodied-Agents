#!/usr/bin/env python3
"""
Demo script for the Audio Analysis Agent
Shows how to use the agent with different audio files and demonstrates various features
"""

import os
import sys
from audio_analysis_agent import AudioAnalyzer

def demo_basic_usage():
    """Demonstrate basic usage of the audio analysis agent"""
    print("🎵 AUDIO ANALYSIS AGENT DEMO")
    print("=" * 50)
    
    # Create analyzer
    analyzer = AudioAnalyzer()
    
    # Check if test audio exists
    if os.path.exists("test_audio.wav"):
        print("📁 Found test audio file: test_audio.wav")
        print("🔍 Running analysis...")
        
        # Run complete analysis
        result = analyzer.analyze_audio("test_audio.wav")
        
        # Print summary
        print("\n📊 ANALYSIS SUMMARY:")
        print("-" * 30)
        
        # Musical analysis
        musical = result.musical_structural
        if "bpm" in musical["bpm_and_tempo"]:
            print(f"🎵 BPM: {musical['bpm_and_tempo']['bpm']:.1f}")
        if "key" in musical["tonal_center"]:
            print(f"🎼 Key: {musical['tonal_center']['key']} {musical['tonal_center']['mode']}")
        if "style_hint" in musical["genre_and_style"]:
            print(f"🎸 Style: {musical['genre_and_style']['style_hint']}")
        
        # Sonic analysis
        sonic = result.sonic_spectral
        if "loudness_db" in sonic["loudness"]:
            print(f"🔊 Loudness: {sonic['loudness']['loudness_db']:.1f} dB")
        if "spectral_balance" in sonic["spectral_power"]:
            print(f"📊 Spectral Balance: {sonic['spectral_power']['spectral_balance']}")
        if "stereo_type" in sonic["stereo_image"]:
            print(f"🎧 Audio Type: {sonic['stereo_image']['stereo_type']}")
        
        print("\n✅ Analysis completed successfully!")
        print("📄 Detailed results saved to: test_audio_analysis.json")
        
    else:
        print("❌ Test audio file not found. Run test_audio_analysis.py first.")

def demo_individual_analysis():
    """Demonstrate individual analysis components"""
    print("\n🔧 INDIVIDUAL ANALYSIS COMPONENTS")
    print("=" * 50)
    
    if not os.path.exists("test_audio.wav"):
        print("❌ Test audio file not found.")
        return
    
    analyzer = AudioAnalyzer()
    
    # BPM and tempo analysis
    print("\n1. 🎵 BPM & Tempo Analysis:")
    bpm_result = analyzer.bpm_and_tempo_mapper("test_audio.wav")
    if "bpm" in bpm_result:
        print(f"   BPM: {bpm_result['bpm']:.1f}")
        print(f"   Dynamic Tempo: {bpm_result['dynamic_tempo']:.1f}")
        print(f"   Onset Strength: {bpm_result['onset_strength_mean']:.3f}")
    
    # Key detection
    print("\n2. 🎼 Key Detection:")
    key_result = analyzer.tonal_center_identifier("test_audio.wav")
    if "key" in key_result:
        print(f"   Key: {key_result['key']} {key_result['mode']}")
        print(f"   Confidence: {key_result['confidence']:.3f}")
    
    # Loudness analysis
    print("\n3. 🔊 Loudness Analysis:")
    loudness_result = analyzer.loudness_meter("test_audio.wav")
    if "loudness_db" in loudness_result:
        print(f"   Loudness: {loudness_result['loudness_db']:.1f} dB")
        print(f"   Dynamic Range: {loudness_result['dynamic_range_db']:.1f} dB")
        print(f"   Crest Factor: {loudness_result['crest_factor']:.2f}")
    
    # Spectral analysis
    print("\n4. 📊 Spectral Analysis:")
    spectral_result = analyzer.spectral_power_monitor("test_audio.wav")
    if "spectral_balance" in spectral_result:
        print(f"   Spectral Balance: {spectral_result['spectral_balance']}")
        print(f"   Bass Ratio: {spectral_result['bass_ratio']:.3f}")
        print(f"   Mid Ratio: {spectral_result['mid_ratio']:.3f}")
        print(f"   Treble Ratio: {spectral_result['treble_ratio']:.3f}")

def demo_batch_processing():
    """Demonstrate batch processing capabilities"""
    print("\n📦 BATCH PROCESSING DEMO")
    print("=" * 50)
    
    # Find all audio files in current directory
    audio_files = []
    for file in os.listdir('.'):
        if file.lower().endswith(('.wav', '.mp3', '.flac', '.ogg')):
            audio_files.append(file)
    
    if not audio_files:
        print("❌ No audio files found in current directory.")
        print("   Supported formats: .wav, .mp3, .flac, .ogg")
        return
    
    print(f"🎵 Found {len(audio_files)} audio file(s):")
    for file in audio_files:
        print(f"   - {file}")
    
    analyzer = AudioAnalyzer()
    
    print(f"\n🔍 Processing {len(audio_files)} file(s)...")
    for i, audio_file in enumerate(audio_files, 1):
        print(f"\n📁 [{i}/{len(audio_files)}] Analyzing: {audio_file}")
        
        try:
            # Run quick analysis (just BPM and key)
            bpm_result = analyzer.bpm_and_tempo_mapper(audio_file)
            key_result = analyzer.tonal_center_identifier(audio_file)
            
            if "bpm" in bpm_result:
                print(f"   🎵 BPM: {bpm_result['bpm']:.1f}")
            if "key" in key_result:
                print(f"   🎼 Key: {key_result['key']} {key_result['mode']}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")

def show_usage_examples():
    """Show usage examples"""
    print("\n📖 USAGE EXAMPLES")
    print("=" * 50)
    
    print("1. Basic usage:")
    print("   python3 audio_analysis_agent.py your_audio_file.wav")
    
    print("\n2. Programmatic usage:")
    print("   from audio_analysis_agent import AudioAnalyzer")
    print("   analyzer = AudioAnalyzer()")
    print("   result = analyzer.analyze_audio('your_audio_file.wav')")
    
    print("\n3. Individual analysis:")
    print("   bpm_result = analyzer.bpm_and_tempo_mapper('audio.wav')")
    print("   key_result = analyzer.tonal_center_identifier('audio.wav')")
    print("   loudness_result = analyzer.loudness_meter('audio.wav')")
    
    print("\n4. Supported audio formats:")
    print("   - WAV (recommended)")
    print("   - MP3")
    print("   - FLAC")
    print("   - OGG")
    print("   - Most formats supported by librosa")

def main():
    """Main demo function"""
    print("🎵 Audio Analysis Agent - Layer 1: Parallel Deconstruction")
    print("=" * 60)
    
    # Check if dependencies are available
    try:
        import librosa
        import numpy
        print("✅ All dependencies are available")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("   Run: pip install --break-system-packages -r requirements.txt")
        return
    
    # Run demos
    demo_basic_usage()
    demo_individual_analysis()
    demo_batch_processing()
    show_usage_examples()
    
    print("\n🎉 Demo completed!")
    print("\nFor more information, see README.md")

if __name__ == "__main__":
    main()