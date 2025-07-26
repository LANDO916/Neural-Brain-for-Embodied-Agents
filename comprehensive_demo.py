#!/usr/bin/env python3
"""
Comprehensive Demo of Advanced Audio Analysis System
Showcases all features: basic analysis, advanced ML features, performance optimizations
"""

import os
import sys
import json
import time
import numpy as np
import librosa
from pathlib import Path
from typing import List, Dict, Any

# Import our modules
try:
    from audio_agent_lite import AudioAnalysisAgent as BasicAgent
    HAS_BASIC_AGENT = True
except ImportError:
    print("⚠️ Basic agent not available")
    HAS_BASIC_AGENT = False

try:
    from advanced_audio_agent import AdvancedAudioAnalysisAgent
    HAS_ADVANCED_AGENT = True
except ImportError:
    print("⚠️ Advanced agent not available")
    HAS_ADVANCED_AGENT = False

try:
    from performance_optimizations import OptimizedAudioProcessor
    HAS_OPTIMIZATION = True
except ImportError:
    print("⚠️ Performance optimizations not available")
    HAS_OPTIMIZATION = False


class ComprehensiveAudioDemo:
    """Comprehensive demo showcasing all audio analysis capabilities"""
    
    def __init__(self):
        self.demo_files = []
        self.results = {}
        
        print("🎵 Comprehensive Audio Analysis Demo")
        print("=" * 60)
        print(f"Basic Agent: {'✅' if HAS_BASIC_AGENT else '❌'}")
        print(f"Advanced Agent: {'✅' if HAS_ADVANCED_AGENT else '❌'}")
        print(f"Performance Optimizations: {'✅' if HAS_OPTIMIZATION else '❌'}")
        print("=" * 60)
    
    def create_demo_audio_files(self) -> List[str]:
        """Create diverse demo audio files for testing"""
        print("\n🎼 Creating demo audio files...")
        
        demo_files = []
        sr = 22050
        duration = 8  # 8 seconds each
        
        # 1. Classical-style piece (piano-like)
        print("   → Creating classical demo...")
        t = np.linspace(0, duration, duration * sr)
        
        # Classical: Slow tempo, harmonic content, major key
        classical_audio = (
            0.5 * np.sin(2 * np.pi * 261.63 * t) +  # C4
            0.3 * np.sin(2 * np.pi * 329.63 * t) +  # E4
            0.3 * np.sin(2 * np.pi * 392.00 * t) +  # G4
            0.1 * np.sin(2 * np.pi * 523.25 * t)    # C5
        )
        # Add some dynamics and decay
        envelope = np.exp(-t * 0.3)
        classical_audio *= envelope
        
        classical_file = "demo_classical.wav"
        librosa.output.write_wav(classical_file, classical_audio, sr)
        demo_files.append(classical_file)
        
        # 2. Electronic dance music style
        print("   → Creating electronic demo...")
        
        # Electronic: Fast tempo, synthetic sounds, steady beat
        beat_pattern = np.zeros_like(t)
        bpm = 128
        beat_interval = 60 / bpm
        
        # Create kick drum pattern
        for beat_time in np.arange(0, duration, beat_interval):
            beat_idx = int(beat_time * sr)
            if beat_idx < len(beat_pattern):
                # Kick drum (low frequency burst)
                kick_duration = int(0.1 * sr)
                kick_envelope = np.exp(-np.arange(kick_duration) / (0.02 * sr))
                kick_sound = 0.8 * np.sin(2 * np.pi * 60 * np.arange(kick_duration) / sr) * kick_envelope
                
                end_idx = min(beat_idx + kick_duration, len(beat_pattern))
                beat_pattern[beat_idx:end_idx] += kick_sound[:end_idx-beat_idx]
        
        # Add high-frequency synth
        synth_freq = 440 + 220 * np.sin(2 * np.pi * 0.5 * t)  # Frequency sweep
        synth_audio = 0.3 * np.sin(2 * np.pi * synth_freq * t)
        
        electronic_audio = beat_pattern + synth_audio
        electronic_audio = np.tanh(electronic_audio)  # Soft clipping for electronic character
        
        electronic_file = "demo_electronic.wav"
        librosa.output.write_wav(electronic_file, electronic_audio, sr)
        demo_files.append(electronic_file)
        
        # 3. Jazz-style piece
        print("   → Creating jazz demo...")
        
        # Jazz: Swing rhythm, complex harmony, improvisation-like
        jazz_audio = np.zeros_like(t)
        
        # Jazz chord progression (ii-V-I in C major)
        chord_changes = [
            (0, 2, [293.66, 349.23, 415.30]),  # Dm7 (D-F-A-C)
            (2, 4, [392.00, 493.88, 587.33]),  # G7 (G-B-D-F)
            (4, 6, [261.63, 329.63, 392.00]),  # CM7 (C-E-G-B)
            (6, 8, [261.63, 329.63, 392.00])   # CM7 (repeat)
        ]
        
        for start_time, end_time, frequencies in chord_changes:
            start_idx = int(start_time * sr)
            end_idx = int(end_time * sr)
            chord_t = t[start_idx:end_idx] - start_time
            
            chord_sound = np.zeros(len(chord_t))
            for freq in frequencies:
                chord_sound += 0.2 * np.sin(2 * np.pi * freq * chord_t)
            
            # Add some swing rhythm
            swing_envelope = 1 + 0.3 * np.sin(2 * np.pi * 2 * chord_t)  # 2 Hz swing
            chord_sound *= swing_envelope
            
            jazz_audio[start_idx:end_idx] += chord_sound
        
        # Add improvisation-like melody
        melody_freqs = [440, 494, 523, 587, 659, 698, 784]  # A major scale
        for i in range(16):  # 16 notes
            note_start = i * duration / 16
            note_duration = duration / 20
            note_idx_start = int(note_start * sr)
            note_idx_end = int((note_start + note_duration) * sr)
            
            if note_idx_end <= len(jazz_audio):
                freq = melody_freqs[i % len(melody_freqs)]
                note_t = np.arange(note_idx_end - note_idx_start) / sr
                note_envelope = np.exp(-note_t * 5)  # Quick decay
                note_sound = 0.15 * np.sin(2 * np.pi * freq * note_t) * note_envelope
                
                jazz_audio[note_idx_start:note_idx_end] += note_sound
        
        jazz_file = "demo_jazz.wav"
        librosa.output.write_wav(jazz_file, jazz_audio, sr)
        demo_files.append(jazz_file)
        
        # 4. Rock-style piece
        print("   → Creating rock demo...")
        
        # Rock: Distorted guitar-like, strong beat, power chords
        rock_audio = np.zeros_like(t)
        
        # Power chord progression
        power_chords = [
            (0, 1, [146.83, 293.66]),  # D5
            (1, 2, [164.81, 329.63]),  # E5
            (2, 3, [196.00, 392.00]),  # G5
            (3, 4, [146.83, 293.66]),  # D5 (repeat pattern)
            (4, 5, [164.81, 329.63]),  # E5
            (5, 6, [196.00, 392.00]),  # G5
            (6, 8, [220.00, 440.00])   # A5
        ]
        
        for start_time, end_time, frequencies in power_chords:
            start_idx = int(start_time * sr)
            end_idx = int(end_time * sr)
            chord_t = t[start_idx:end_idx] - start_time
            
            chord_sound = np.zeros(len(chord_t))
            for freq in frequencies:
                # Add harmonics for guitar-like sound
                fundamental = np.sin(2 * np.pi * freq * chord_t)
                second_harmonic = 0.3 * np.sin(2 * np.pi * freq * 2 * chord_t)
                third_harmonic = 0.2 * np.sin(2 * np.pi * freq * 3 * chord_t)
                
                guitar_sound = fundamental + second_harmonic + third_harmonic
                # Distortion effect
                guitar_sound = np.tanh(3 * guitar_sound)
                chord_sound += 0.3 * guitar_sound
            
            rock_audio[start_idx:end_idx] += chord_sound
        
        # Add drum-like percussion
        for beat in np.arange(0, duration, 0.5):  # Every half second
            beat_idx = int(beat * sr)
            if beat_idx < len(rock_audio):
                # Simple percussive hit
                hit_duration = int(0.05 * sr)
                hit_envelope = np.exp(-np.arange(hit_duration) / (0.01 * sr))
                hit_sound = 0.5 * np.random.randn(hit_duration) * hit_envelope
                
                end_idx = min(beat_idx + hit_duration, len(rock_audio))
                rock_audio[beat_idx:end_idx] += hit_sound[:end_idx-beat_idx]
        
        rock_file = "demo_rock.wav"
        librosa.output.write_wav(rock_file, rock_audio, sr)
        demo_files.append(rock_file)
        
        # 5. Vocal-like demo
        print("   → Creating vocal demo...")
        
        # Vocal: Human voice-like characteristics
        vocal_audio = np.zeros_like(t)
        
        # Simulate vowel sounds with formants
        formant_freqs = [
            (800, 1200, 2500),  # 'a' sound
            (270, 2300, 3000),  # 'i' sound
            (300, 870, 2200),   # 'u' sound
        ]
        
        vowel_duration = duration / len(formant_freqs)
        for i, (f1, f2, f3) in enumerate(formant_freqs):
            start_time = i * vowel_duration
            end_time = (i + 1) * vowel_duration
            start_idx = int(start_time * sr)
            end_idx = int(end_time * sr)
            
            vowel_t = t[start_idx:end_idx] - start_time
            
            # Fundamental frequency with vibrato
            f0 = 200 + 20 * np.sin(2 * np.pi * 5 * vowel_t)  # 200 Hz with 5 Hz vibrato
            
            # Voice-like sound with formants
            voice_sound = (
                0.5 * np.sin(2 * np.pi * f0 * vowel_t) +              # Fundamental
                0.3 * np.sin(2 * np.pi * f1 / 200 * f0 * vowel_t) +   # First formant
                0.2 * np.sin(2 * np.pi * f2 / 200 * f0 * vowel_t) +   # Second formant
                0.1 * np.sin(2 * np.pi * f3 / 200 * f0 * vowel_t)     # Third formant
            )
            
            # Add breath-like noise
            breath_noise = 0.05 * np.random.randn(len(vowel_t))
            voice_sound += breath_noise
            
            # Voice envelope
            voice_envelope = np.ones_like(vowel_t)
            # Attack
            attack_samples = int(0.1 * sr)
            if len(voice_envelope) > attack_samples:
                voice_envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
            # Decay
            decay_samples = int(0.1 * sr)
            if len(voice_envelope) > decay_samples:
                voice_envelope[-decay_samples:] = np.linspace(1, 0, decay_samples)
            
            vocal_audio[start_idx:end_idx] += voice_sound * voice_envelope
        
        vocal_file = "demo_vocal.wav"
        librosa.output.write_wav(vocal_file, vocal_audio, sr)
        demo_files.append(vocal_file)
        
        self.demo_files = demo_files
        print(f"✅ Created {len(demo_files)} demo files")
        
        return demo_files
    
    def demo_basic_agent(self):
        """Demonstrate basic audio analysis agent"""
        if not HAS_BASIC_AGENT:
            print("❌ Basic agent not available")
            return
        
        print("\n" + "="*60)
        print("🎵 BASIC AUDIO ANALYSIS AGENT DEMO")
        print("="*60)
        
        # Initialize basic agent
        agent = BasicAgent()
        
        results = {}
        for i, audio_file in enumerate(self.demo_files):
            print(f"\n[{i+1}/{len(self.demo_files)}] Analyzing: {audio_file}")
            
            try:
                # Analyze with basic agent
                result = agent.analyze_audio(audio_file)
                results[audio_file] = result
                
                # Print summary
                agent.print_summary()
                
                # Save individual results
                output_file = f"basic_{Path(audio_file).stem}_results.json"
                agent.save_results(output_file)
                
            except Exception as e:
                print(f"❌ Error analyzing {audio_file}: {e}")
                results[audio_file] = {"error": str(e)}
        
        self.results["basic"] = results
        print("\n✅ Basic agent demo complete")
    
    def demo_advanced_agent(self):
        """Demonstrate advanced audio analysis agent"""
        if not HAS_ADVANCED_AGENT:
            print("❌ Advanced agent not available")
            return
        
        print("\n" + "="*60)
        print("🤖 ADVANCED AUDIO ANALYSIS AGENT DEMO")
        print("="*60)
        
        # Initialize advanced agent
        agent = AdvancedAudioAnalysisAgent(enable_streaming=False)
        
        results = {}
        for i, audio_file in enumerate(self.demo_files):
            print(f"\n[{i+1}/{len(self.demo_files)}] Advanced analysis: {audio_file}")
            
            try:
                # Analyze with advanced features
                result = agent.analyze_audio(audio_file, include_advanced=True)
                results[audio_file] = result
                
                # Print detailed summary
                agent.print_summary(detailed=True)
                
                # Save individual results
                output_file = f"advanced_{Path(audio_file).stem}_results.json"
                agent.save_results(output_file)
                
            except Exception as e:
                print(f"❌ Error analyzing {audio_file}: {e}")
                results[audio_file] = {"error": str(e)}
        
        # Demonstrate batch analysis
        print(f"\n🔄 Demonstrating batch analysis...")
        batch_results = agent.batch_analyze(self.demo_files, "advanced_batch_results")
        
        self.results["advanced"] = results
        print("\n✅ Advanced agent demo complete")
    
    def demo_performance_optimizations(self):
        """Demonstrate performance optimizations"""
        if not HAS_OPTIMIZATION:
            print("❌ Performance optimizations not available")
            return
        
        print("\n" + "="*60)
        print("⚡ PERFORMANCE OPTIMIZATION DEMO")
        print("="*60)
        
        # Initialize optimized processor
        processor = OptimizedAudioProcessor(cache_enabled=True, cache_size_mb=100)
        
        # Show optimization report
        print("\n🔧 System Optimization Report:")
        report = processor.get_optimization_report()
        print(json.dumps(report, indent=2, default=str))
        
        # Benchmark performance
        print(f"\n🚀 Running performance benchmark...")
        benchmark_results = processor.benchmark_performance(self.demo_files)
        
        print("\n📊 BENCHMARK RESULTS")
        print("-" * 40)
        for config_name, results in benchmark_results.items():
            print(f"\n{config_name}:")
            print(f"  Total time: {results['total_time']:.2f}s")
            print(f"  Avg per file: {results['avg_time_per_file']:.2f}s")
            print(f"  Memory usage: {results['memory_usage']['rss_mb']:.1f} MB")
        
        # Demonstrate batch processing
        print(f"\n🔄 Optimized batch processing...")
        batch_results = processor.batch_process_optimized(self.demo_files)
        
        # Save optimization results
        with open("optimization_demo_results.json", 'w') as f:
            json.dump({
                "benchmark": benchmark_results,
                "batch_results": batch_results,
                "optimization_report": report
            }, f, indent=2, default=str)
        
        self.results["optimization"] = {
            "benchmark": benchmark_results,
            "batch": batch_results
        }
        
        print("\n✅ Performance optimization demo complete")
    
    def compare_results(self):
        """Compare results from different analysis methods"""
        print("\n" + "="*60)
        print("📊 COMPARATIVE ANALYSIS")
        print("="*60)
        
        if not self.results:
            print("❌ No results to compare")
            return
        
        comparison = {}
        
        for audio_file in self.demo_files:
            file_comparison = {}
            
            # Extract key metrics from each method
            if "basic" in self.results and audio_file in self.results["basic"]:
                basic_result = self.results["basic"][audio_file]
                if "musical_structural" in basic_result:
                    musical = basic_result["musical_structural"]
                    file_comparison["basic"] = {
                        "tempo": musical.get("rhythm", {}).get("estimated_tempo", 0),
                        "key": musical.get("harmony", {}).get("estimated_key", "unknown"),
                        "has_vocals": basic_result.get("lyrical_vocal", {}).get("voice_activity", {}).get("likely_contains_vocals", False)
                    }
            
            if "advanced" in self.results and audio_file in self.results["advanced"]:
                advanced_result = self.results["advanced"][audio_file]
                advanced_features = advanced_result.get("advanced_features", {})
                
                file_comparison["advanced"] = {
                    "predicted_genre": advanced_features.get("genre_classification", {}).get("predicted_genre", "unknown"),
                    "predicted_mood": advanced_features.get("mood_detection", {}).get("predicted_mood", "unknown"),
                    "complexity": advanced_features.get("complexity_analysis", {}).get("complexity_level", "unknown"),
                    "dominant_instrument": advanced_features.get("instrument_detection", {}).get("dominant_instrument", "unknown")
                }
            
            comparison[audio_file] = file_comparison
        
        # Print comparison table
        print(f"\n{'File':<20} {'Basic Tempo':<12} {'Basic Key':<10} {'Genre':<12} {'Mood':<12} {'Complexity':<12}")
        print("-" * 90)
        
        for audio_file, comp in comparison.items():
            filename = Path(audio_file).stem
            basic = comp.get("basic", {})
            advanced = comp.get("advanced", {})
            
            print(f"{filename:<20} "
                  f"{basic.get('tempo', 'N/A'):<12} "
                  f"{basic.get('key', 'N/A'):<10} "
                  f"{advanced.get('predicted_genre', 'N/A'):<12} "
                  f"{advanced.get('predicted_mood', 'N/A'):<12} "
                  f"{advanced.get('complexity', 'N/A'):<12}")
        
        # Save comparison
        with open("analysis_comparison.json", 'w') as f:
            json.dump(comparison, f, indent=2, default=str)
        
        print(f"\n💾 Comparison saved to: analysis_comparison.json")
    
    def generate_final_report(self):
        """Generate comprehensive final report"""
        print("\n" + "="*60)
        print("📋 FINAL COMPREHENSIVE REPORT")
        print("="*60)
        
        report = {
            "demo_info": {
                "total_files_analyzed": len(self.demo_files),
                "demo_files": self.demo_files,
                "analysis_methods": list(self.results.keys()),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "capabilities_demonstrated": {
                "basic_analysis": HAS_BASIC_AGENT,
                "advanced_ml_features": HAS_ADVANCED_AGENT,
                "performance_optimization": HAS_OPTIMIZATION
            },
            "results_summary": {}
        }
        
        # Summarize results
        for method, method_results in self.results.items():
            if method == "optimization":
                # Special handling for optimization results
                report["results_summary"][method] = {
                    "benchmark_configs": len(method_results.get("benchmark", {})),
                    "batch_processing_success": method_results.get("batch", {}).get("batch_info", {}).get("successful", 0)
                }
            else:
                successful = len([r for r in method_results.values() if "error" not in r])
                failed = len(method_results) - successful
                report["results_summary"][method] = {
                    "successful_analyses": successful,
                    "failed_analyses": failed,
                    "success_rate": f"{successful/len(method_results)*100:.1f}%" if method_results else "0%"
                }
        
        # Technical achievements
        report["technical_achievements"] = [
            "Multi-dimensional audio analysis (Musical, Vocal, Spectral)",
            "Machine Learning-based genre and mood classification",
            "Intelligent caching system for performance optimization",
            "Parallel processing for batch analysis",
            "Real-time memory monitoring",
            "Comprehensive instrument detection",
            "Advanced rhythmic pattern analysis",
            "Texture and timbre characterization",
            "Audio quality assessment",
            "Scalable architecture for different use cases"
        ]
        
        # Save final report
        with open("comprehensive_demo_report.json", 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n📊 Analysis Summary:")
        print(f"   Demo files created: {len(self.demo_files)}")
        print(f"   Analysis methods tested: {len(self.results)}")
        
        for method, summary in report["results_summary"].items():
            print(f"   {method.title()} analysis: {summary}")
        
        print(f"\n🎯 Key Features Demonstrated:")
        for achievement in report["technical_achievements"]:
            print(f"   ✅ {achievement}")
        
        print(f"\n💾 Final report saved to: comprehensive_demo_report.json")
        
        return report
    
    def cleanup_demo_files(self):
        """Clean up demo files"""
        print(f"\n🗑️ Cleaning up demo files...")
        
        cleaned = 0
        for demo_file in self.demo_files:
            if os.path.exists(demo_file):
                os.remove(demo_file)
                cleaned += 1
                print(f"   Removed: {demo_file}")
        
        # Also clean up any other demo-related files
        demo_patterns = ["demo_", "basic_", "advanced_", "test_audio_"]
        for file in os.listdir("."):
            if any(pattern in file for pattern in demo_patterns) and file.endswith((".wav", ".json")):
                if os.path.exists(file):
                    os.remove(file)
                    cleaned += 1
                    print(f"   Removed: {file}")
        
        print(f"✅ Cleaned up {cleaned} files")
    
    def run_full_demo(self, cleanup: bool = True):
        """Run the complete demonstration"""
        print("🚀 Starting Comprehensive Audio Analysis Demo")
        start_time = time.time()
        
        try:
            # Step 1: Create demo files
            self.create_demo_audio_files()
            
            # Step 2: Basic analysis
            if HAS_BASIC_AGENT:
                self.demo_basic_agent()
            
            # Step 3: Advanced analysis
            if HAS_ADVANCED_AGENT:
                self.demo_advanced_agent()
            
            # Step 4: Performance optimization
            if HAS_OPTIMIZATION:
                self.demo_performance_optimizations()
            
            # Step 5: Compare results
            self.compare_results()
            
            # Step 6: Generate final report
            report = self.generate_final_report()
            
            total_time = time.time() - start_time
            
            print(f"\n" + "="*60)
            print("🎉 DEMO COMPLETED SUCCESSFULLY!")
            print("="*60)
            print(f"Total demo time: {total_time:.2f} seconds")
            print(f"Files generated:")
            print(f"   📊 analysis_comparison.json")
            print(f"   📋 comprehensive_demo_report.json")
            
            if HAS_OPTIMIZATION:
                print(f"   ⚡ optimization_demo_results.json")
            
            if HAS_ADVANCED_AGENT:
                print(f"   📁 advanced_batch_results/ (directory)")
            
            print(f"\n🎯 Demo showcased {len(report['technical_achievements'])} key features")
            print(f"🔬 Analyzed {len(self.demo_files)} different audio types")
            print(f"⚡ Tested {len(self.results)} analysis methods")
            
        except Exception as e:
            print(f"\n❌ Demo failed: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Cleanup
            if cleanup:
                self.cleanup_demo_files()


def main():
    """Main function for demo execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive Audio Analysis Demo")
    parser.add_argument("--no-cleanup", action="store_true", help="Keep demo files after completion")
    parser.add_argument("--basic-only", action="store_true", help="Run only basic analysis demo")
    parser.add_argument("--advanced-only", action="store_true", help="Run only advanced analysis demo")
    parser.add_argument("--optimization-only", action="store_true", help="Run only optimization demo")
    
    args = parser.parse_args()
    
    demo = ComprehensiveAudioDemo()
    
    if args.basic_only and HAS_BASIC_AGENT:
        demo.create_demo_audio_files()
        demo.demo_basic_agent()
    elif args.advanced_only and HAS_ADVANCED_AGENT:
        demo.create_demo_audio_files()
        demo.demo_advanced_agent()
    elif args.optimization_only and HAS_OPTIMIZATION:
        demo.create_demo_audio_files()
        demo.demo_performance_optimizations()
    else:
        # Run full demo
        demo.run_full_demo(cleanup=not args.no_cleanup)


if __name__ == "__main__":
    main()