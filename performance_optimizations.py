#!/usr/bin/env python3
"""
Performance Optimizations for Audio Analysis Agent
Features: Caching, parallel processing, memory management, batch processing optimizations
"""

import os
import sys
import json
import pickle
import hashlib
import warnings
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Callable
import concurrent.futures
from datetime import datetime, timedelta
import threading
import queue
import time
import gc
from functools import wraps, lru_cache
import multiprocessing as mp

# Memory profiling (optional)
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# Fast computation libraries
try:
    import numba
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

warnings.filterwarnings('ignore')

class AudioAnalysisCache:
    """Intelligent caching system for audio analysis results"""
    
    def __init__(self, cache_dir: str = ".audio_cache", max_size_mb: int = 500):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_size_mb = max_size_mb
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        """Load cache metadata"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except:
                return {"files": {}, "total_size": 0}
        return {"files": {}, "total_size": 0}
    
    def _save_metadata(self):
        """Save cache metadata"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def _get_file_hash(self, filepath: str, analysis_params: Dict) -> str:
        """Generate unique hash for file and analysis parameters"""
        # File content hash
        with open(filepath, 'rb') as f:
            file_content = f.read()
        file_hash = hashlib.md5(file_content).hexdigest()
        
        # Parameters hash
        params_str = json.dumps(analysis_params, sort_keys=True)
        params_hash = hashlib.md5(params_str.encode()).hexdigest()
        
        return f"{file_hash}_{params_hash}"
    
    def get(self, filepath: str, analysis_params: Dict) -> Optional[Dict]:
        """Retrieve cached analysis results"""
        cache_key = self._get_file_hash(filepath, analysis_params)
        
        if cache_key in self.metadata["files"]:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            if cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        results = pickle.load(f)
                    
                    # Update access time
                    self.metadata["files"][cache_key]["last_accessed"] = datetime.now().isoformat()
                    self._save_metadata()
                    
                    print(f"📦 Cache hit: {Path(filepath).name}")
                    return results
                except:
                    # Remove corrupted cache entry
                    self._remove_cache_entry(cache_key)
        
        return None
    
    def put(self, filepath: str, analysis_params: Dict, results: Dict):
        """Store analysis results in cache"""
        cache_key = self._get_file_hash(filepath, analysis_params)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        try:
            # Store results
            with open(cache_file, 'wb') as f:
                pickle.dump(results, f)
            
            # Update metadata
            file_size = cache_file.stat().st_size / (1024 * 1024)  # MB
            
            self.metadata["files"][cache_key] = {
                "filepath": filepath,
                "size_mb": file_size,
                "created": datetime.now().isoformat(),
                "last_accessed": datetime.now().isoformat(),
                "analysis_params": analysis_params
            }
            
            self.metadata["total_size"] += file_size
            
            # Clean up if cache is too large
            self._cleanup_cache()
            
            self._save_metadata()
            print(f"💾 Cached: {Path(filepath).name} ({file_size:.2f} MB)")
            
        except Exception as e:
            print(f"⚠️ Cache write failed: {e}")
    
    def _remove_cache_entry(self, cache_key: str):
        """Remove a cache entry"""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            cache_file.unlink()
        
        if cache_key in self.metadata["files"]:
            size_mb = self.metadata["files"][cache_key]["size_mb"]
            self.metadata["total_size"] -= size_mb
            del self.metadata["files"][cache_key]
    
    def _cleanup_cache(self):
        """Remove old entries if cache is too large"""
        while self.metadata["total_size"] > self.max_size_mb:
            # Remove oldest accessed file
            oldest_key = min(
                self.metadata["files"].keys(),
                key=lambda k: self.metadata["files"][k]["last_accessed"]
            )
            
            print(f"🗑️ Removing old cache: {self.metadata['files'][oldest_key]['filepath']}")
            self._remove_cache_entry(oldest_key)
    
    def clear(self):
        """Clear entire cache"""
        for cache_key in list(self.metadata["files"].keys()):
            self._remove_cache_entry(cache_key)
        self.metadata = {"files": {}, "total_size": 0}
        self._save_metadata()
        print("🗑️ Cache cleared")
    
    def stats(self) -> Dict:
        """Get cache statistics"""
        return {
            "total_entries": len(self.metadata["files"]),
            "total_size_mb": self.metadata["total_size"],
            "cache_dir": str(self.cache_dir),
            "entries": self.metadata["files"]
        }


class MemoryManager:
    """Memory management utilities"""
    
    @staticmethod
    def get_memory_usage() -> Dict[str, float]:
        """Get current memory usage"""
        if HAS_PSUTIL:
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            return {
                "rss_mb": memory_info.rss / 1024 / 1024,
                "vms_mb": memory_info.vms / 1024 / 1024,
                "percent": process.memory_percent()
            }
        return {"rss_mb": 0, "vms_mb": 0, "percent": 0}
    
    @staticmethod
    def memory_monitor(func: Callable) -> Callable:
        """Decorator to monitor memory usage of functions"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            if HAS_PSUTIL:
                start_memory = MemoryManager.get_memory_usage()
                print(f"🧠 Memory before {func.__name__}: {start_memory['rss_mb']:.1f} MB")
            
            result = func(*args, **kwargs)
            
            if HAS_PSUTIL:
                end_memory = MemoryManager.get_memory_usage()
                memory_diff = end_memory['rss_mb'] - start_memory['rss_mb']
                print(f"🧠 Memory after {func.__name__}: {end_memory['rss_mb']:.1f} MB (+{memory_diff:.1f} MB)")
            
            return result
        return wrapper
    
    @staticmethod
    def cleanup():
        """Force garbage collection"""
        collected = gc.collect()
        print(f"🗑️ Garbage collected {collected} objects")


# Optimized audio processing functions
if HAS_NUMBA:
    @numba.jit(nopython=True)
    def fast_spectral_centroid(magnitude, freqs):
        """Fast spectral centroid calculation using Numba"""
        weighted_freq_sum = np.sum(magnitude * freqs)
        magnitude_sum = np.sum(magnitude)
        return weighted_freq_sum / magnitude_sum if magnitude_sum > 0 else 0
    
    @numba.jit(nopython=True)
    def fast_zero_crossing_rate(audio):
        """Fast zero crossing rate calculation"""
        crossings = 0
        for i in range(1, len(audio)):
            if (audio[i] >= 0) != (audio[i-1] >= 0):
                crossings += 1
        return crossings / len(audio)
    
    @numba.jit(nopython=True)
    def fast_rms_energy(audio):
        """Fast RMS energy calculation"""
        return np.sqrt(np.mean(audio**2))
else:
    def fast_spectral_centroid(magnitude, freqs):
        """Fallback spectral centroid calculation"""
        weighted_freq_sum = np.sum(magnitude * freqs)
        magnitude_sum = np.sum(magnitude)
        return weighted_freq_sum / magnitude_sum if magnitude_sum > 0 else 0
    
    def fast_zero_crossing_rate(audio):
        """Fallback zero crossing rate calculation"""
        return np.mean(librosa.feature.zero_crossing_rate(audio))
    
    def fast_rms_energy(audio):
        """Fallback RMS energy calculation"""
        return np.sqrt(np.mean(audio**2))


class OptimizedAudioProcessor:
    """Optimized audio processing with performance improvements"""
    
    def __init__(self, cache_enabled: bool = True, cache_size_mb: int = 500):
        self.cache_enabled = cache_enabled
        self.cache = AudioAnalysisCache(max_size_mb=cache_size_mb) if cache_enabled else None
        self.memory_manager = MemoryManager()
        
        # Precomputed constants
        self.mel_filters = {}
        self.mfcc_filters = {}
        
        print(f"⚡ Optimized Audio Processor initialized")
        print(f"   Caching: {'✅' if cache_enabled else '❌'}")
        print(f"   Numba acceleration: {'✅' if HAS_NUMBA else '❌'}")
        print(f"   Memory monitoring: {'✅' if HAS_PSUTIL else '❌'}")
    
    @MemoryManager.memory_monitor
    def load_audio_optimized(self, filepath: str, target_sr: int = 22050) -> Tuple[np.ndarray, int]:
        """Optimized audio loading with memory management"""
        try:
            # Try to load with librosa (handles most formats)
            y, sr = librosa.load(filepath, sr=target_sr, mono=True)
            
            # Ensure float32 for better performance
            if y.dtype != np.float32:
                y = y.astype(np.float32)
            
            return y, sr
            
        except Exception as e:
            print(f"❌ Error loading {filepath}: {e}")
            raise
    
    def extract_features_parallel(self, y: np.ndarray, sr: int, 
                                 feature_groups: List[str] = None) -> Dict[str, Any]:
        """Extract features in parallel for better performance"""
        if feature_groups is None:
            feature_groups = ['spectral', 'temporal', 'harmonic', 'rhythmic']
        
        features = {}
        
        # Use ThreadPoolExecutor for I/O bound operations
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(feature_groups))) as executor:
            futures = {}
            
            if 'spectral' in feature_groups:
                futures['spectral'] = executor.submit(self._extract_spectral_features, y, sr)
            
            if 'temporal' in feature_groups:
                futures['temporal'] = executor.submit(self._extract_temporal_features, y, sr)
            
            if 'harmonic' in feature_groups:
                futures['harmonic'] = executor.submit(self._extract_harmonic_features, y, sr)
            
            if 'rhythmic' in feature_groups:
                futures['rhythmic'] = executor.submit(self._extract_rhythmic_features, y, sr)
            
            # Collect results
            for feature_type, future in futures.items():
                try:
                    features[feature_type] = future.result(timeout=30)
                except Exception as e:
                    print(f"⚠️ Error extracting {feature_type} features: {e}")
                    features[feature_type] = {"error": str(e)}
        
        return features
    
    def _extract_spectral_features(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Extract spectral features optimized"""
        features = {}
        
        try:
            # Use librosa for initial spectral analysis
            stft = librosa.stft(y, hop_length=512, n_fft=2048)
            magnitude = np.abs(stft)
            freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
            
            # Fast spectral centroid
            if HAS_NUMBA:
                centroids = []
                for frame in range(magnitude.shape[1]):
                    centroid = fast_spectral_centroid(magnitude[:, frame], freqs)
                    centroids.append(centroid)
                spectral_centroid = np.array(centroids)
            else:
                spectral_centroid = librosa.feature.spectral_centroid(S=magnitude, sr=sr)[0]
            
            features['spectral_centroid'] = {
                'mean': float(np.mean(spectral_centroid)),
                'std': float(np.std(spectral_centroid)),
                'max': float(np.max(spectral_centroid)),
                'min': float(np.min(spectral_centroid))
            }
            
            # Other spectral features
            spectral_rolloff = librosa.feature.spectral_rolloff(S=magnitude, sr=sr)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(S=magnitude, sr=sr)[0]
            spectral_flatness = librosa.feature.spectral_flatness(S=magnitude)[0]
            
            features.update({
                'spectral_rolloff': {
                    'mean': float(np.mean(spectral_rolloff)),
                    'std': float(np.std(spectral_rolloff))
                },
                'spectral_bandwidth': {
                    'mean': float(np.mean(spectral_bandwidth)),
                    'std': float(np.std(spectral_bandwidth))
                },
                'spectral_flatness': {
                    'mean': float(np.mean(spectral_flatness)),
                    'std': float(np.std(spectral_flatness))
                }
            })
            
            # Frequency band analysis
            freq_bands = {
                'sub_bass': (20, 60),
                'bass': (60, 250),
                'low_mid': (250, 500),
                'mid': (500, 2000),
                'high_mid': (2000, 4000),
                'presence': (4000, 6000),
                'brilliance': (6000, sr//2)
            }
            
            band_energies = {}
            for band_name, (low_freq, high_freq) in freq_bands.items():
                band_mask = (freqs >= low_freq) & (freqs <= high_freq)
                if np.any(band_mask):
                    band_energy = np.mean(magnitude[band_mask, :])
                    band_energies[band_name] = float(band_energy)
                else:
                    band_energies[band_name] = 0.0
            
            features['frequency_bands'] = band_energies
            
        except Exception as e:
            features['error'] = str(e)
        
        return features
    
    def _extract_temporal_features(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Extract temporal features optimized"""
        features = {}
        
        try:
            # Fast zero crossing rate
            if HAS_NUMBA:
                zcr = fast_zero_crossing_rate(y)
            else:
                zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
            
            # Fast RMS energy
            if HAS_NUMBA:
                rms_energy = fast_rms_energy(y)
            else:
                rms_energy = float(np.mean(librosa.feature.rms(y=y)))
            
            features.update({
                'zero_crossing_rate': zcr,
                'rms_energy': rms_energy,
                'duration': float(len(y) / sr)
            })
            
            # Attack/decay analysis (simplified)
            # Find peaks in RMS energy
            rms_frames = librosa.feature.rms(y=y, hop_length=512)[0]
            if len(rms_frames) > 10:
                # Simple attack time estimation
                max_energy_idx = np.argmax(rms_frames)
                if max_energy_idx > 0:
                    pre_peak = rms_frames[:max_energy_idx]
                    if len(pre_peak) > 0:
                        attack_frames = len(pre_peak)
                        attack_time = attack_frames * 512 / sr
                        features['attack_time'] = float(attack_time)
            
        except Exception as e:
            features['error'] = str(e)
        
        return features
    
    def _extract_harmonic_features(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Extract harmonic features optimized"""
        features = {}
        
        try:
            # Harmonic-percussive separation
            y_harmonic, y_percussive = librosa.effects.hpss(y)
            
            harmonic_energy = float(np.sum(y_harmonic**2))
            percussive_energy = float(np.sum(y_percussive**2))
            total_energy = harmonic_energy + percussive_energy
            
            features['harmonic_percussive'] = {
                'harmonic_ratio': float(harmonic_energy / total_energy) if total_energy > 0 else 0,
                'percussive_ratio': float(percussive_energy / total_energy) if total_energy > 0 else 0
            }
            
            # Chroma analysis for harmony
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            chroma_mean = np.mean(chroma, axis=1)
            
            # Key estimation
            key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            key_strengths = chroma_mean
            estimated_key_idx = np.argmax(key_strengths)
            
            features['harmony'] = {
                'estimated_key': key_names[estimated_key_idx],
                'key_confidence': float(key_strengths[estimated_key_idx] / np.sum(key_strengths)),
                'tonal_clarity': float(np.max(chroma_mean) / np.mean(chroma_mean))
            }
            
        except Exception as e:
            features['error'] = str(e)
        
        return features
    
    def _extract_rhythmic_features(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Extract rhythmic features optimized"""
        features = {}
        
        try:
            # Tempo and beat tracking
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            beat_times = librosa.frames_to_time(beats, sr=sr)
            
            features['tempo'] = {
                'estimated_tempo': float(tempo),
                'beats_detected': len(beats)
            }
            
            if len(beat_times) > 3:
                beat_intervals = np.diff(beat_times)
                tempo_stability = 1.0 / (1.0 + np.std(beat_intervals) / np.mean(beat_intervals))
                
                features['tempo']['stability'] = float(tempo_stability)
                features['tempo']['regularity'] = float(1.0 - np.std(beat_intervals) / np.mean(beat_intervals))
            
            # Onset detection
            onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
            onset_times = librosa.frames_to_time(onset_frames, sr=sr)
            
            features['onsets'] = {
                'count': len(onset_times),
                'density': float(len(onset_times) / (len(y) / sr))
            }
            
        except Exception as e:
            features['error'] = str(e)
        
        return features
    
    def process_audio_file(self, filepath: str, analysis_params: Dict = None) -> Dict[str, Any]:
        """Process single audio file with caching and optimization"""
        if analysis_params is None:
            analysis_params = {
                'sample_rate': 22050,
                'feature_groups': ['spectral', 'temporal', 'harmonic', 'rhythmic']
            }
        
        # Check cache first
        if self.cache_enabled:
            cached_result = self.cache.get(filepath, analysis_params)
            if cached_result:
                return cached_result
        
        print(f"🎵 Processing: {Path(filepath).name}")
        start_time = time.time()
        
        try:
            # Load audio
            y, sr = self.load_audio_optimized(filepath, analysis_params['sample_rate'])
            
            # Extract features
            features = self.extract_features_parallel(
                y, sr, 
                analysis_params.get('feature_groups', ['spectral', 'temporal', 'harmonic', 'rhythmic'])
            )
            
            # Prepare results
            results = {
                'metadata': {
                    'file': filepath,
                    'duration': len(y) / sr,
                    'sample_rate': sr,
                    'processing_time': time.time() - start_time,
                    'analysis_params': analysis_params
                },
                'features': features
            }
            
            # Cache results
            if self.cache_enabled:
                self.cache.put(filepath, analysis_params, results)
            
            # Clean up memory
            del y
            gc.collect()
            
            print(f"✅ Processed in {time.time() - start_time:.2f} seconds")
            return results
            
        except Exception as e:
            error_result = {
                'metadata': {
                    'file': filepath,
                    'error': str(e),
                    'processing_time': time.time() - start_time
                }
            }
            return error_result
    
    def batch_process_optimized(self, file_list: List[str], 
                               max_workers: int = None,
                               analysis_params: Dict = None) -> Dict[str, Any]:
        """Optimized batch processing with intelligent parallel execution"""
        if max_workers is None:
            max_workers = min(mp.cpu_count(), len(file_list), 4)  # Reasonable default
        
        print(f"🔄 Batch processing {len(file_list)} files with {max_workers} workers")
        
        results = {}
        start_time = time.time()
        
        # Process files in parallel
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all jobs
            future_to_file = {
                executor.submit(self.process_audio_file, filepath, analysis_params): filepath
                for filepath in file_list
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_file):
                filepath = future_to_file[future]
                try:
                    result = future.result(timeout=60)  # 1 minute timeout per file
                    results[filepath] = result
                    
                    progress = len(results) / len(file_list) * 100
                    print(f"📊 Progress: {progress:.1f}% ({len(results)}/{len(file_list)})")
                    
                except concurrent.futures.TimeoutError:
                    print(f"⏰ Timeout processing: {filepath}")
                    results[filepath] = {"error": "Processing timeout"}
                except Exception as e:
                    print(f"❌ Error processing {filepath}: {e}")
                    results[filepath] = {"error": str(e)}
        
        total_time = time.time() - start_time
        
        # Summary
        successful = len([r for r in results.values() if "error" not in r.get("metadata", {})])
        failed = len(file_list) - successful
        
        batch_summary = {
            'batch_info': {
                'total_files': len(file_list),
                'successful': successful,
                'failed': failed,
                'total_time': total_time,
                'avg_time_per_file': total_time / len(file_list),
                'max_workers': max_workers
            },
            'individual_results': results
        }
        
        print(f"\n✅ Batch processing complete!")
        print(f"   Total time: {total_time:.2f} seconds")
        print(f"   Successful: {successful}/{len(file_list)}")
        print(f"   Average time per file: {total_time/len(file_list):.2f} seconds")
        
        return batch_summary
    
    def benchmark_performance(self, test_files: List[str] = None) -> Dict[str, Any]:
        """Benchmark processing performance"""
        if test_files is None:
            # Create synthetic test data if no files provided
            print("🧪 Creating synthetic test data for benchmarking...")
            test_files = []
            for i in range(3):
                # Generate test audio
                duration = 10  # 10 seconds
                sr = 22050
                t = np.linspace(0, duration, duration * sr)
                
                # Generate synthetic audio with different characteristics
                if i == 0:  # Simple sine wave
                    y = np.sin(2 * np.pi * 440 * t)  # A4
                elif i == 1:  # Complex harmonic content
                    y = (np.sin(2 * np.pi * 440 * t) + 
                         0.5 * np.sin(2 * np.pi * 880 * t) + 
                         0.25 * np.sin(2 * np.pi * 1320 * t))
                else:  # Noise + sine
                    y = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.random.randn(len(t))
                
                # Save test file
                test_file = f"test_audio_{i}.wav"
                sf.write(test_file, y, sr)
                test_files.append(test_file)
        
        benchmark_results = {}
        
        # Test different configurations
        configs = [
            {'name': 'Single Thread', 'max_workers': 1},
            {'name': 'Multi Thread', 'max_workers': min(mp.cpu_count(), 4)},
            {'name': 'Cache Enabled', 'cache': True},
            {'name': 'Cache Disabled', 'cache': False}
        ]
        
        for config in configs:
            print(f"\n🏃 Testing: {config['name']}")
            
            # Temporarily adjust settings
            original_cache_enabled = self.cache_enabled
            if 'cache' in config:
                self.cache_enabled = config['cache']
            
            start_time = time.time()
            
            if 'max_workers' in config:
                results = self.batch_process_optimized(
                    test_files, 
                    max_workers=config['max_workers']
                )
            else:
                results = {}
                for filepath in test_files:
                    results[filepath] = self.process_audio_file(filepath)
            
            processing_time = time.time() - start_time
            
            benchmark_results[config['name']] = {
                'total_time': processing_time,
                'files_processed': len(test_files),
                'avg_time_per_file': processing_time / len(test_files),
                'memory_usage': self.memory_manager.get_memory_usage()
            }
            
            # Restore original settings
            self.cache_enabled = original_cache_enabled
        
        # Clean up test files
        for test_file in test_files:
            if os.path.exists(test_file):
                os.remove(test_file)
        
        return benchmark_results
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Generate optimization report"""
        report = {
            'system_info': {
                'cpu_count': mp.cpu_count(),
                'memory_usage': self.memory_manager.get_memory_usage(),
                'optimizations_enabled': {
                    'numba': HAS_NUMBA,
                    'psutil': HAS_PSUTIL,
                    'caching': self.cache_enabled
                }
            }
        }
        
        if self.cache_enabled:
            report['cache_stats'] = self.cache.stats()
        
        return report


def main():
    """Main function for performance testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Audio Analysis Performance Optimizer")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmark")
    parser.add_argument("--cache-size", type=int, default=500, help="Cache size in MB")
    parser.add_argument("--max-workers", type=int, help="Maximum worker processes")
    parser.add_argument("files", nargs="*", help="Audio files to process")
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = OptimizedAudioProcessor(cache_size_mb=args.cache_size)
    
    if args.benchmark:
        print("🚀 Running performance benchmark...")
        benchmark_results = processor.benchmark_performance()
        
        print("\n📊 BENCHMARK RESULTS")
        print("=" * 50)
        for config_name, results in benchmark_results.items():
            print(f"\n{config_name}:")
            print(f"  Total time: {results['total_time']:.2f}s")
            print(f"  Avg per file: {results['avg_time_per_file']:.2f}s")
            print(f"  Memory: {results['memory_usage']['rss_mb']:.1f} MB")
    
    elif args.files:
        print(f"🎵 Processing {len(args.files)} files...")
        results = processor.batch_process_optimized(
            args.files, 
            max_workers=args.max_workers
        )
        
        # Save results
        output_file = "optimized_batch_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"💾 Results saved to: {output_file}")
    
    else:
        print("🔧 Optimization report:")
        report = processor.get_optimization_report()
        print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()