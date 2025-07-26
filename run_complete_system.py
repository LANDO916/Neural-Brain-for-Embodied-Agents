#!/usr/bin/env python3
"""
Complete Audio Analysis System - Master Integration Script
Combines all components: basic agent, advanced agent, performance optimizations
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

# Configure paths
SCRIPT_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(SCRIPT_DIR))

print("🎵 Loading Complete Audio Analysis System...")
print("=" * 60)

# Import components with error handling
components_status = {}

try:
    from audio_agent_lite import AudioAnalysisAgent as BasicAgent
    components_status["basic_agent"] = "✅ Available"
except ImportError as e:
    components_status["basic_agent"] = f"❌ Failed: {e}"
    BasicAgent = None

try:
    from advanced_audio_agent import AdvancedAudioAnalysisAgent
    components_status["advanced_agent"] = "✅ Available"
except ImportError as e:
    components_status["advanced_agent"] = f"❌ Failed: {e}"
    AdvancedAudioAnalysisAgent = None

try:
    from performance_optimizations import OptimizedAudioProcessor
    components_status["performance_optimization"] = "✅ Available"
except ImportError as e:
    components_status["performance_optimization"] = f"❌ Failed: {e}"
    OptimizedAudioProcessor = None

try:
    from comprehensive_demo import ComprehensiveAudioDemo
    components_status["comprehensive_demo"] = "✅ Available"
except ImportError as e:
    components_status["comprehensive_demo"] = f"❌ Failed: {e}"
    ComprehensiveAudioDemo = None

# Display component status
for component, status in components_status.items():
    print(f"{component.replace('_', ' ').title()}: {status}")

print("=" * 60)


class CompleteAudioAnalysisSystem:
    """Master class integrating all audio analysis components"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        self.results = {}
        
        # Initialize available components
        self.basic_agent = None
        self.advanced_agent = None
        self.optimized_processor = None
        
        self._initialize_components()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "sample_rate": 22050,
            "enable_caching": True,
            "cache_size_mb": 500,
            "max_workers": 4,
            "enable_streaming": False,
            "include_advanced_analysis": True,
            "save_individual_results": True,
            "output_directory": "audio_analysis_results"
        }
    
    def _initialize_components(self):
        """Initialize available components"""
        print("\n🔧 Initializing Audio Analysis Components...")
        
        # Basic Agent
        if BasicAgent:
            try:
                self.basic_agent = BasicAgent(sample_rate=self.config["sample_rate"])
                print("   ✅ Basic Agent initialized")
            except Exception as e:
                print(f"   ❌ Basic Agent failed: {e}")
        
        # Advanced Agent
        if AdvancedAudioAnalysisAgent:
            try:
                self.advanced_agent = AdvancedAudioAnalysisAgent(
                    sample_rate=self.config["sample_rate"],
                    enable_streaming=self.config["enable_streaming"]
                )
                print("   ✅ Advanced Agent initialized")
            except Exception as e:
                print(f"   ❌ Advanced Agent failed: {e}")
        
        # Optimized Processor
        if OptimizedAudioProcessor:
            try:
                self.optimized_processor = OptimizedAudioProcessor(
                    cache_enabled=self.config["enable_caching"],
                    cache_size_mb=self.config["cache_size_mb"]
                )
                print("   ✅ Optimized Processor initialized")
            except Exception as e:
                print(f"   ❌ Optimized Processor failed: {e}")
        
        print("🔧 Component initialization complete")
    
    def analyze_single_file(self, audio_path: str, 
                           analysis_type: str = "comprehensive") -> Dict[str, Any]:
        """Analyze a single audio file with specified analysis type"""
        print(f"\n🎵 Analyzing: {Path(audio_path).name}")
        print(f"   Analysis type: {analysis_type}")
        
        if not os.path.exists(audio_path):
            return {"error": f"File not found: {audio_path}"}
        
        results = {
            "file_path": audio_path,
            "analysis_type": analysis_type,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "results": {}
        }
        
        # Basic Analysis
        if analysis_type in ["basic", "comprehensive"] and self.basic_agent:
            print("   → Running basic analysis...")
            try:
                basic_result = self.basic_agent.analyze_audio(audio_path)
                results["results"]["basic"] = basic_result
                print("   ✅ Basic analysis complete")
            except Exception as e:
                print(f"   ❌ Basic analysis failed: {e}")
                results["results"]["basic"] = {"error": str(e)}
        
        # Advanced Analysis
        if analysis_type in ["advanced", "comprehensive"] and self.advanced_agent:
            print("   → Running advanced analysis...")
            try:
                advanced_result = self.advanced_agent.analyze_audio(
                    audio_path, 
                    include_advanced=self.config["include_advanced_analysis"]
                )
                results["results"]["advanced"] = advanced_result
                print("   ✅ Advanced analysis complete")
            except Exception as e:
                print(f"   ❌ Advanced analysis failed: {e}")
                results["results"]["advanced"] = {"error": str(e)}
        
        # Optimized Analysis
        if analysis_type in ["optimized", "comprehensive"] and self.optimized_processor:
            print("   → Running optimized analysis...")
            try:
                optimized_result = self.optimized_processor.process_audio_file(
                    audio_path,
                    {"sample_rate": self.config["sample_rate"]}
                )
                results["results"]["optimized"] = optimized_result
                print("   ✅ Optimized analysis complete")
            except Exception as e:
                print(f"   ❌ Optimized analysis failed: {e}")
                results["results"]["optimized"] = {"error": str(e)}
        
        # Save individual results if requested
        if self.config["save_individual_results"]:
            self._save_individual_results(audio_path, results)
        
        return results
    
    def analyze_batch(self, audio_files: List[str], 
                     analysis_type: str = "comprehensive") -> Dict[str, Any]:
        """Analyze multiple audio files"""
        print(f"\n🔄 Batch Analysis: {len(audio_files)} files")
        print(f"   Analysis type: {analysis_type}")
        
        start_time = time.time()
        batch_results = {
            "batch_info": {
                "total_files": len(audio_files),
                "analysis_type": analysis_type,
                "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "config": self.config
            },
            "individual_results": {},
            "summary": {}
        }
        
        # Process files
        successful = 0
        failed = 0
        
        for i, audio_path in enumerate(audio_files):
            print(f"\n[{i+1}/{len(audio_files)}] Processing: {Path(audio_path).name}")
            
            try:
                result = self.analyze_single_file(audio_path, analysis_type)
                batch_results["individual_results"][audio_path] = result
                
                # Check if analysis was successful
                if any("error" not in res for res in result.get("results", {}).values()):
                    successful += 1
                else:
                    failed += 1
                    
            except Exception as e:
                print(f"   ❌ Processing failed: {e}")
                batch_results["individual_results"][audio_path] = {"error": str(e)}
                failed += 1
            
            # Progress update
            progress = ((i + 1) / len(audio_files)) * 100
            print(f"   📊 Progress: {progress:.1f}%")
        
        # Calculate summary
        total_time = time.time() - start_time
        batch_results["summary"] = {
            "successful": successful,
            "failed": failed,
            "success_rate": f"{(successful/len(audio_files)*100):.1f}%",
            "total_time": total_time,
            "avg_time_per_file": total_time / len(audio_files)
        }
        
        # Save batch results
        self._save_batch_results(batch_results, analysis_type)
        
        print(f"\n✅ Batch analysis complete!")
        print(f"   Successful: {successful}/{len(audio_files)}")
        print(f"   Total time: {total_time:.2f} seconds")
        print(f"   Average per file: {total_time/len(audio_files):.2f} seconds")
        
        return batch_results
    
    def run_benchmark(self) -> Dict[str, Any]:
        """Run performance benchmark across all available components"""
        print("\n🚀 Running System-Wide Performance Benchmark...")
        
        if not self.optimized_processor:
            print("❌ Optimized processor not available for benchmarking")
            return {"error": "Optimized processor not available"}
        
        try:
            benchmark_results = self.optimized_processor.benchmark_performance()
            
            # Save benchmark results
            with open("system_benchmark_results.json", 'w') as f:
                json.dump(benchmark_results, f, indent=2, default=str)
            
            print("✅ Benchmark complete! Results saved to system_benchmark_results.json")
            return benchmark_results
            
        except Exception as e:
            print(f"❌ Benchmark failed: {e}")
            return {"error": str(e)}
    
    def run_comprehensive_demo(self, cleanup: bool = True):
        """Run the comprehensive demo showcasing all features"""
        print("\n🎭 Running Comprehensive Demo...")
        
        if not ComprehensiveAudioDemo:
            print("❌ Comprehensive demo not available")
            return
        
        try:
            demo = ComprehensiveAudioDemo()
            demo.run_full_demo(cleanup=cleanup)
            print("✅ Comprehensive demo complete!")
        except Exception as e:
            print(f"❌ Demo failed: {e}")
            import traceback
            traceback.print_exc()
    
    def _save_individual_results(self, audio_path: str, results: Dict[str, Any]):
        """Save individual analysis results"""
        output_dir = Path(self.config["output_directory"])
        output_dir.mkdir(exist_ok=True)
        
        filename = Path(audio_path).stem
        output_file = output_dir / f"{filename}_analysis.json"
        
        try:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"   💾 Results saved: {output_file}")
        except Exception as e:
            print(f"   ⚠️ Failed to save results: {e}")
    
    def _save_batch_results(self, batch_results: Dict[str, Any], analysis_type: str):
        """Save batch analysis results"""
        output_dir = Path(self.config["output_directory"])
        output_dir.mkdir(exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"batch_{analysis_type}_{timestamp}.json"
        
        try:
            with open(output_file, 'w') as f:
                json.dump(batch_results, f, indent=2, default=str)
            print(f"💾 Batch results saved: {output_file}")
        except Exception as e:
            print(f"⚠️ Failed to save batch results: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        status = {
            "components": {
                "basic_agent": self.basic_agent is not None,
                "advanced_agent": self.advanced_agent is not None,
                "optimized_processor": self.optimized_processor is not None
            },
            "configuration": self.config,
            "capabilities": []
        }
        
        # Determine capabilities
        if self.basic_agent:
            status["capabilities"].extend([
                "Basic audio analysis",
                "Musical structure analysis",
                "Vocal activity detection",
                "Spectral feature extraction"
            ])
        
        if self.advanced_agent:
            status["capabilities"].extend([
                "Advanced ML-based analysis",
                "Genre classification",
                "Mood detection",
                "Instrument identification",
                "Audio complexity analysis",
                "Texture and timbre analysis"
            ])
        
        if self.optimized_processor:
            status["capabilities"].extend([
                "Performance optimization",
                "Intelligent caching",
                "Parallel processing",
                "Memory management",
                "Batch processing optimization"
            ])
        
        return status
    
    def print_system_info(self):
        """Print comprehensive system information"""
        print("\n" + "="*60)
        print("🎵 COMPLETE AUDIO ANALYSIS SYSTEM STATUS")
        print("="*60)
        
        status = self.get_system_status()
        
        print(f"\n🔧 Components Status:")
        for component, available in status["components"].items():
            status_icon = "✅" if available else "❌"
            print(f"   {component.replace('_', ' ').title()}: {status_icon}")
        
        print(f"\n⚙️ Configuration:")
        for key, value in status["configuration"].items():
            print(f"   {key}: {value}")
        
        print(f"\n🎯 Available Capabilities:")
        for capability in status["capabilities"]:
            print(f"   ✅ {capability}")
        
        print(f"\n📊 Performance Info:")
        if self.optimized_processor:
            try:
                perf_report = self.optimized_processor.get_optimization_report()
                system_info = perf_report.get("system_info", {})
                print(f"   CPU Cores: {system_info.get('cpu_count', 'Unknown')}")
                memory = system_info.get('memory_usage', {})
                print(f"   Memory Usage: {memory.get('rss_mb', 0):.1f} MB")
                
                optimizations = system_info.get('optimizations_enabled', {})
                for opt, enabled in optimizations.items():
                    icon = "✅" if enabled else "❌"
                    print(f"   {opt.title()}: {icon}")
            except:
                print("   Performance info unavailable")
        else:
            print("   Performance monitoring not available")
        
        print("="*60)


def main():
    """Main function with comprehensive CLI interface"""
    parser = argparse.ArgumentParser(
        description="Complete Audio Analysis System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_complete_system.py --status
  python run_complete_system.py --demo
  python run_complete_system.py --benchmark
  python run_complete_system.py audio.wav --analysis comprehensive
  python run_complete_system.py *.wav --batch --analysis advanced
        """
    )
    
    # System operations
    parser.add_argument("--status", action="store_true", 
                       help="Show system status and capabilities")
    parser.add_argument("--demo", action="store_true",
                       help="Run comprehensive demo")
    parser.add_argument("--benchmark", action="store_true",
                       help="Run performance benchmark")
    
    # File analysis
    parser.add_argument("files", nargs="*", help="Audio files to analyze")
    parser.add_argument("--batch", action="store_true",
                       help="Process files in batch mode")
    parser.add_argument("--analysis", choices=["basic", "advanced", "optimized", "comprehensive"],
                       default="comprehensive", help="Type of analysis to perform")
    
    # Configuration
    parser.add_argument("--sample-rate", type=int, default=22050,
                       help="Audio sample rate")
    parser.add_argument("--no-cache", action="store_true",
                       help="Disable caching")
    parser.add_argument("--cache-size", type=int, default=500,
                       help="Cache size in MB")
    parser.add_argument("--max-workers", type=int, default=4,
                       help="Maximum worker processes")
    parser.add_argument("--output-dir", default="audio_analysis_results",
                       help="Output directory for results")
    parser.add_argument("--no-save", action="store_true",
                       help="Don't save individual results")
    
    args = parser.parse_args()
    
    # Build configuration
    config = {
        "sample_rate": args.sample_rate,
        "enable_caching": not args.no_cache,
        "cache_size_mb": args.cache_size,
        "max_workers": args.max_workers,
        "enable_streaming": False,
        "include_advanced_analysis": True,
        "save_individual_results": not args.no_save,
        "output_directory": args.output_dir
    }
    
    # Initialize system
    system = CompleteAudioAnalysisSystem(config)
    
    # Execute requested operation
    if args.status:
        system.print_system_info()
    
    elif args.demo:
        system.run_comprehensive_demo(cleanup=True)
    
    elif args.benchmark:
        system.run_benchmark()
    
    elif args.files:
        if args.batch:
            # Batch processing
            system.analyze_batch(args.files, args.analysis)
        else:
            # Single file processing
            for audio_file in args.files:
                system.analyze_single_file(audio_file, args.analysis)
    
    else:
        # No specific operation requested - show status
        system.print_system_info()
        print(f"\n💡 Use --help to see available options")
        print(f"💡 Use --demo to run the comprehensive demonstration")
        print(f"💡 Use --status to see detailed system information")


if __name__ == "__main__":
    main()