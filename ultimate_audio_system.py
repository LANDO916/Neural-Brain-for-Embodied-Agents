#!/usr/bin/env python3
"""
Ultimate Audio Analysis System
Complete integration of all audio analysis capabilities with self-improving AI
"""

import os
import sys
import json
import asyncio
import threading
import time
import warnings
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from pathlib import Path
from datetime import datetime
import hashlib
import uuid

# Core audio analysis components
try:
    from audio_agent_lite import AudioAnalysisAgent
    HAS_BASIC = True
except ImportError:
    HAS_BASIC = False

try:
    from advanced_audio_agent import AdvancedAudioAnalysisAgent
    HAS_ADVANCED = True
except ImportError:
    HAS_ADVANCED = False

try:
    from performance_optimizations import OptimizedAudioProcessor
    HAS_OPTIMIZATION = True
except ImportError:
    HAS_OPTIMIZATION = False

try:
    from neural_audio_analysis import NeuralAudioAnalyzer
    HAS_NEURAL = True
except ImportError:
    HAS_NEURAL = False

try:
    from real_time_audio_processor import RealTimeAudioProcessor, RealTimeVisualizer
    HAS_REALTIME = True
except ImportError:
    HAS_REALTIME = False

try:
    from adaptive_meta_learning import MetaLearningSystem
    HAS_META_LEARNING = True
except ImportError:
    HAS_META_LEARNING = False

# External dependencies
try:
    import librosa
    import numpy as np
    HAS_AUDIO_LIBS = True
except ImportError:
    HAS_AUDIO_LIBS = False

warnings.filterwarnings('ignore')

class AnalysisMode:
    """Analysis mode constants"""
    BASIC = "basic"
    ADVANCED = "advanced"
    NEURAL = "neural"
    REALTIME = "realtime"
    OPTIMIZED = "optimized"
    COMPREHENSIVE = "comprehensive"
    ADAPTIVE = "adaptive"


class UltimateAudioSystem:
    """Ultimate audio analysis system with all capabilities"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = self._load_default_config()
        if config:
            self.config.update(config)
        
        # Component initialization status
        self.components = {
            'basic_agent': None,
            'advanced_agent': None,
            'optimization_processor': None,
            'neural_analyzer': None,
            'realtime_processor': None,
            'meta_learning_system': None
        }
        
        # System state
        self.is_initialized = False
        self.available_modes = []
        self.active_realtime_sessions = {}
        self.analysis_history = []
        self.performance_metrics = {}
        
        # Initialize components
        self._initialize_components()
        
        print("🎵 Ultimate Audio Analysis System Initialized")
        print("=" * 60)
        self._print_system_status()
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default system configuration"""
        return {
            # Audio processing
            'sample_rate': 22050,
            'hop_length': 512,
            'n_fft': 2048,
            'n_mels': 128,
            
            # Analysis settings
            'enable_caching': True,
            'cache_size_mb': 1000,
            'max_workers': 4,
            'analysis_timeout': 300,
            
            # Real-time processing
            'realtime_buffer_duration': 30.0,
            'realtime_analysis_interval': 0.1,
            'realtime_chunk_size': 1024,
            
            # Neural network settings
            'neural_model_training': False,
            'neural_model_path': 'neural_models',
            
            # Meta-learning settings
            'enable_meta_learning': True,
            'meta_learning_data_dir': 'meta_learning_data',
            'auto_optimization': True,
            
            # Output settings
            'output_directory': 'ultimate_analysis_results',
            'save_intermediate_results': True,
            'export_visualizations': False,
            
            # Performance
            'enable_gpu': False,
            'memory_limit_mb': 4096,
            'optimize_for_speed': True
        }
    
    def _initialize_components(self):
        """Initialize all available components"""
        print("🔄 Initializing system components...")
        
        # Basic audio agent
        if HAS_BASIC and HAS_AUDIO_LIBS:
            try:
                self.components['basic_agent'] = AudioAnalysisAgent(
                    sample_rate=self.config['sample_rate']
                )
                self.available_modes.append(AnalysisMode.BASIC)
                print("✅ Basic Audio Agent initialized")
            except Exception as e:
                print(f"⚠️ Basic Agent initialization failed: {e}")
        
        # Advanced audio agent
        if HAS_ADVANCED:
            try:
                self.components['advanced_agent'] = AdvancedAudioAnalysisAgent(
                    sample_rate=self.config['sample_rate']
                )
                self.available_modes.extend([AnalysisMode.ADVANCED, AnalysisMode.COMPREHENSIVE])
                print("✅ Advanced Audio Agent initialized")
            except Exception as e:
                print(f"⚠️ Advanced Agent initialization failed: {e}")
        
        # Optimization processor
        if HAS_OPTIMIZATION:
            try:
                self.components['optimization_processor'] = OptimizedAudioProcessor(
                    cache_size_mb=self.config['cache_size_mb']
                )
                self.available_modes.append(AnalysisMode.OPTIMIZED)
                print("✅ Optimization Processor initialized")
            except Exception as e:
                print(f"⚠️ Optimization Processor initialization failed: {e}")
        
        # Neural analyzer
        if HAS_NEURAL:
            try:
                self.components['neural_analyzer'] = NeuralAudioAnalyzer(
                    sample_rate=self.config['sample_rate']
                )
                
                # Initialize neural models if requested
                if self.config['neural_model_training']:
                    self.components['neural_analyzer'].initialize_models(train_models=True)
                else:
                    # Try to load pre-trained models
                    model_path = self.config['neural_model_path']
                    if os.path.exists(model_path):
                        self.components['neural_analyzer'].load_models(model_path)
                
                self.available_modes.append(AnalysisMode.NEURAL)
                print("✅ Neural Audio Analyzer initialized")
            except Exception as e:
                print(f"⚠️ Neural Analyzer initialization failed: {e}")
        
        # Real-time processor
        if HAS_REALTIME:
            try:
                self.components['realtime_processor'] = RealTimeAudioProcessor(
                    sample_rate=self.config['sample_rate'],
                    chunk_size=self.config['realtime_chunk_size'],
                    buffer_duration=self.config['realtime_buffer_duration'],
                    analysis_interval=self.config['realtime_analysis_interval']
                )
                self.available_modes.append(AnalysisMode.REALTIME)
                print("✅ Real-Time Audio Processor initialized")
            except Exception as e:
                print(f"⚠️ Real-Time Processor initialization failed: {e}")
        
        # Meta-learning system
        if HAS_META_LEARNING and self.config['enable_meta_learning']:
            try:
                self.components['meta_learning_system'] = MetaLearningSystem(
                    save_dir=self.config['meta_learning_data_dir']
                )
                self.components['meta_learning_system'].register_optimizable_parameters()
                self.available_modes.append(AnalysisMode.ADAPTIVE)
                print("✅ Meta-Learning System initialized")
            except Exception as e:
                print(f"⚠️ Meta-Learning System initialization failed: {e}")
        
        # Create output directory
        os.makedirs(self.config['output_directory'], exist_ok=True)
        
        self.is_initialized = True
        print("🎯 System initialization complete")
    
    def _print_system_status(self):
        """Print current system status"""
        print(f"📊 Available Analysis Modes: {', '.join(self.available_modes)}")
        print(f"🔧 Active Components: {sum(1 for c in self.components.values() if c is not None)}/{len(self.components)}")
        
        component_status = {
            'Basic Agent': '✅' if self.components['basic_agent'] else '❌',
            'Advanced Agent': '✅' if self.components['advanced_agent'] else '❌',
            'Optimization': '✅' if self.components['optimization_processor'] else '❌',
            'Neural Networks': '✅' if self.components['neural_analyzer'] else '❌',
            'Real-Time': '✅' if self.components['realtime_processor'] else '❌',
            'Meta-Learning': '✅' if self.components['meta_learning_system'] else '❌'
        }
        
        for component, status in component_status.items():
            print(f"   {component}: {status}")
        
        print(f"⚙️ Configuration: {self.config['sample_rate']}Hz, {self.config['n_mels']} mel bands")
    
    def analyze_audio(self, 
                     audio_path: str, 
                     mode: str = AnalysisMode.COMPREHENSIVE,
                     options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Perform audio analysis using specified mode"""
        if not self.is_initialized:
            return {"error": "System not initialized"}
        
        if mode not in self.available_modes:
            return {"error": f"Mode '{mode}' not available. Available modes: {self.available_modes}"}
        
        if not os.path.exists(audio_path):
            return {"error": f"Audio file not found: {audio_path}"}
        
        # Generate analysis ID
        analysis_id = str(uuid.uuid4())
        start_time = time.time()
        
        print(f"🎵 Starting {mode} analysis: {Path(audio_path).name}")
        print(f"   Analysis ID: {analysis_id}")
        
        try:
            # Get optimized parameters if meta-learning is available
            analysis_params = self._get_optimized_parameters(mode, options or {})
            
            # Perform analysis based on mode
            if mode == AnalysisMode.BASIC:
                results = self._analyze_basic(audio_path, analysis_params)
            elif mode == AnalysisMode.ADVANCED:
                results = self._analyze_advanced(audio_path, analysis_params)
            elif mode == AnalysisMode.NEURAL:
                results = self._analyze_neural(audio_path, analysis_params)
            elif mode == AnalysisMode.OPTIMIZED:
                results = self._analyze_optimized(audio_path, analysis_params)
            elif mode == AnalysisMode.COMPREHENSIVE:
                results = self._analyze_comprehensive(audio_path, analysis_params)
            elif mode == AnalysisMode.ADAPTIVE:
                results = self._analyze_adaptive(audio_path, analysis_params)
            else:
                results = {"error": f"Unknown analysis mode: {mode}"}
            
            # Calculate timing and performance metrics
            processing_time = time.time() - start_time
            timing_info = {
                'total_time': processing_time,
                'start_time': start_time,
                'end_time': time.time()
            }
            
            # Add metadata
            results['analysis_metadata'] = {
                'analysis_id': analysis_id,
                'mode': mode,
                'file_path': audio_path,
                'parameters': analysis_params,
                'timing': timing_info,
                'timestamp': datetime.now().isoformat(),
                'system_version': '2.0.0'
            }
            
            # Record performance for meta-learning
            if self.components['meta_learning_system']:
                self.components['meta_learning_system'].record_analysis_performance(
                    analysis_id, analysis_params, results, timing_info
                )
            
            # Save results if configured
            if self.config['save_intermediate_results']:
                self._save_analysis_results(results)
            
            # Update analysis history
            self.analysis_history.append({
                'analysis_id': analysis_id,
                'mode': mode,
                'file': audio_path,
                'duration': processing_time,
                'timestamp': datetime.now().isoformat()
            })
            
            print(f"✅ Analysis complete in {processing_time:.2f} seconds")
            return results
            
        except Exception as e:
            error_result = {
                'error': str(e),
                'analysis_metadata': {
                    'analysis_id': analysis_id,
                    'mode': mode,
                    'file_path': audio_path,
                    'error_time': datetime.now().isoformat()
                }
            }
            print(f"❌ Analysis failed: {e}")
            return error_result
    
    def _get_optimized_parameters(self, mode: str, base_options: Dict[str, Any]) -> Dict[str, Any]:
        """Get optimized parameters for analysis"""
        params = {
            'sample_rate': self.config['sample_rate'],
            'hop_length': self.config['hop_length'],
            'n_fft': self.config['n_fft'],
            'n_mels': self.config['n_mels']
        }
        
        # Add base options
        params.update(base_options)
        
        # Get meta-learning optimized parameters if available
        if self.components['meta_learning_system'] and self.config['auto_optimization']:
            try:
                optimization = self.components['meta_learning_system'].get_optimized_parameters('user_satisfaction')
                if 'parameters' in optimization:
                    # Only update if confidence is high enough
                    if optimization.get('confidence', 0) > 0.3:
                        params.update(optimization['parameters'])
                        print(f"🎯 Using optimized parameters (confidence: {optimization.get('confidence', 0):.3f})")
            except Exception as e:
                print(f"⚠️ Parameter optimization failed: {e}")
        
        return params
    
    def _analyze_basic(self, audio_path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Perform basic audio analysis"""
        if not self.components['basic_agent']:
            return {"error": "Basic agent not available"}
        
        return self.components['basic_agent'].analyze_audio(audio_path)
    
    def _analyze_advanced(self, audio_path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Perform advanced audio analysis"""
        if not self.components['advanced_agent']:
            return {"error": "Advanced agent not available"}
        
        return self.components['advanced_agent'].analyze_audio(audio_path)
    
    def _analyze_neural(self, audio_path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Perform neural network-based analysis"""
        if not self.components['neural_analyzer']:
            return {"error": "Neural analyzer not available"}
        
        return self.components['neural_analyzer'].analyze_audio(audio_path)
    
    def _analyze_optimized(self, audio_path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Perform optimized analysis"""
        if not self.components['optimization_processor']:
            return {"error": "Optimization processor not available"}
        
        return self.components['optimization_processor'].process_audio_file(audio_path)
    
    def _analyze_comprehensive(self, audio_path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive analysis using all available methods"""
        results = {
            'comprehensive_analysis': {},
            'analysis_summary': {},
            'combined_insights': {}
        }
        
        # Run basic analysis
        if self.components['basic_agent']:
            basic_results = self._analyze_basic(audio_path, params)
            results['comprehensive_analysis']['basic'] = basic_results
        
        # Run advanced analysis
        if self.components['advanced_agent']:
            advanced_results = self._analyze_advanced(audio_path, params)
            results['comprehensive_analysis']['advanced'] = advanced_results
        
        # Run neural analysis
        if self.components['neural_analyzer']:
            neural_results = self._analyze_neural(audio_path, params)
            results['comprehensive_analysis']['neural'] = neural_results
        
        # Run optimized analysis
        if self.components['optimization_processor']:
            optimized_results = self._analyze_optimized(audio_path, params)
            results['comprehensive_analysis']['optimized'] = optimized_results
        
        # Combine insights from all analyses
        results['combined_insights'] = self._combine_analysis_insights(results['comprehensive_analysis'])
        
        # Generate summary
        results['analysis_summary'] = self._generate_comprehensive_summary(results)
        
        return results
    
    def _analyze_adaptive(self, audio_path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Perform adaptive analysis that learns and improves over time"""
        if not self.components['meta_learning_system']:
            return {"error": "Meta-learning system not available"}
        
        # Start with comprehensive analysis
        results = self._analyze_comprehensive(audio_path, params)
        
        # Add adaptive learning insights
        learning_report = self.components['meta_learning_system'].generate_learning_report()
        results['adaptive_insights'] = {
            'learning_status': learning_report['system_status'],
            'performance_trends': learning_report['performance_trends'],
            'optimization_recommendations': learning_report['optimization_recommendations'],
            'learning_insights': learning_report['learning_insights']
        }
        
        return results
    
    def _combine_analysis_insights(self, all_analyses: Dict[str, Any]) -> Dict[str, Any]:
        """Combine insights from multiple analysis methods"""
        combined = {
            'consensus_genre': None,
            'consensus_mood': None,
            'confidence_weighted_features': {},
            'analysis_agreement': {},
            'conflicting_results': []
        }
        
        # Extract genre predictions from different analyses
        genre_predictions = []
        mood_predictions = []
        
        for analysis_type, results in all_analyses.items():
            if isinstance(results, dict) and 'error' not in results:
                # Extract genre information
                genre_info = self._extract_genre_info(results)
                if genre_info:
                    genre_predictions.append((analysis_type, genre_info))
                
                # Extract mood information
                mood_info = self._extract_mood_info(results)
                if mood_info:
                    mood_predictions.append((analysis_type, mood_info))
        
        # Determine consensus
        if genre_predictions:
            combined['consensus_genre'] = self._determine_consensus_genre(genre_predictions)
        
        if mood_predictions:
            combined['consensus_mood'] = self._determine_consensus_mood(mood_predictions)
        
        # Calculate agreement scores
        combined['analysis_agreement'] = {
            'genre_agreement': self._calculate_agreement_score(genre_predictions),
            'mood_agreement': self._calculate_agreement_score(mood_predictions),
            'overall_consistency': 0.0  # Will be calculated based on multiple factors
        }
        
        return combined
    
    def _extract_genre_info(self, results: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract genre information from analysis results"""
        # Check various possible locations for genre information
        locations = [
            ['neural_analysis', 'genre_classification'],
            ['advanced_features', 'genre_classification'],
            ['genre_classification'],
            ['musical_structural', 'genre_hints']
        ]
        
        for location in locations:
            current = results
            for key in location:
                if isinstance(current, dict) and key in current:
                    current = current[key]
                else:
                    current = None
                    break
            
            if current and isinstance(current, dict):
                return current
        
        return None
    
    def _extract_mood_info(self, results: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract mood information from analysis results"""
        locations = [
            ['neural_analysis', 'emotion_recognition'],
            ['advanced_features', 'mood_detection'],
            ['mood_detection'],
            ['lyrical_vocal', 'emotion_analysis']
        ]
        
        for location in locations:
            current = results
            for key in location:
                if isinstance(current, dict) and key in current:
                    current = current[key]
                else:
                    current = None
                    break
            
            if current and isinstance(current, dict):
                return current
        
        return None
    
    def _determine_consensus_genre(self, predictions: List[Tuple[str, Dict[str, Any]]]) -> Dict[str, Any]:
        """Determine consensus genre from multiple predictions"""
        genre_votes = {}
        confidence_sum = {}
        
        for analysis_type, prediction in predictions:
            genre = prediction.get('predicted_genre') or prediction.get('genre')
            confidence = prediction.get('confidence', 0.5)
            
            if genre:
                if genre not in genre_votes:
                    genre_votes[genre] = 0
                    confidence_sum[genre] = 0
                
                genre_votes[genre] += 1
                confidence_sum[genre] += confidence
        
        if not genre_votes:
            return {'consensus_genre': 'unknown', 'confidence': 0.0}
        
        # Find most voted genre
        best_genre = max(genre_votes.keys(), key=lambda g: genre_votes[g])
        
        return {
            'consensus_genre': best_genre,
            'confidence': confidence_sum[best_genre] / genre_votes[best_genre],
            'vote_count': genre_votes[best_genre],
            'total_analyses': len(predictions),
            'all_votes': genre_votes
        }
    
    def _determine_consensus_mood(self, predictions: List[Tuple[str, Dict[str, Any]]]) -> Dict[str, Any]:
        """Determine consensus mood from multiple predictions"""
        mood_votes = {}
        confidence_sum = {}
        
        for analysis_type, prediction in predictions:
            mood = prediction.get('predicted_emotion') or prediction.get('mood') or prediction.get('dominant_emotion')
            confidence = prediction.get('confidence', 0.5)
            
            if mood:
                if mood not in mood_votes:
                    mood_votes[mood] = 0
                    confidence_sum[mood] = 0
                
                mood_votes[mood] += 1
                confidence_sum[mood] += confidence
        
        if not mood_votes:
            return {'consensus_mood': 'neutral', 'confidence': 0.0}
        
        best_mood = max(mood_votes.keys(), key=lambda m: mood_votes[m])
        
        return {
            'consensus_mood': best_mood,
            'confidence': confidence_sum[best_mood] / mood_votes[best_mood],
            'vote_count': mood_votes[best_mood],
            'total_analyses': len(predictions),
            'all_votes': mood_votes
        }
    
    def _calculate_agreement_score(self, predictions: List[Tuple[str, Dict[str, Any]]]) -> float:
        """Calculate agreement score between predictions"""
        if len(predictions) < 2:
            return 1.0
        
        # Simple agreement calculation based on most common prediction
        values = []
        for _, prediction in predictions:
            value = (prediction.get('predicted_genre') or 
                    prediction.get('predicted_emotion') or 
                    prediction.get('genre') or 
                    prediction.get('mood'))
            if value:
                values.append(value)
        
        if not values:
            return 0.0
        
        # Calculate the fraction of predictions that agree with the most common one
        most_common = max(set(values), key=values.count)
        agreement_count = values.count(most_common)
        
        return agreement_count / len(values)
    
    def _generate_comprehensive_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a comprehensive summary of all analyses"""
        summary = {
            'overall_confidence': 0.0,
            'key_findings': [],
            'recommendations': [],
            'technical_quality': {},
            'analysis_coverage': {}
        }
        
        # Calculate overall confidence
        confidences = []
        analyses = results.get('comprehensive_analysis', {})
        
        for analysis_type, analysis_results in analyses.items():
            if isinstance(analysis_results, dict) and 'error' not in analysis_results:
                # Extract confidence values from various locations
                conf_values = self._extract_confidence_values(analysis_results)
                confidences.extend(conf_values)
        
        if confidences:
            summary['overall_confidence'] = float(np.mean(confidences))
        
        # Generate key findings
        combined_insights = results.get('combined_insights', {})
        
        if combined_insights.get('consensus_genre'):
            genre_info = combined_insights['consensus_genre']
            summary['key_findings'].append(
                f"Genre: {genre_info.get('consensus_genre', 'unknown')} "
                f"(confidence: {genre_info.get('confidence', 0):.2f})"
            )
        
        if combined_insights.get('consensus_mood'):
            mood_info = combined_insights['consensus_mood']
            summary['key_findings'].append(
                f"Mood: {mood_info.get('consensus_mood', 'unknown')} "
                f"(confidence: {mood_info.get('confidence', 0):.2f})"
            )
        
        # Analysis coverage
        summary['analysis_coverage'] = {
            'total_analyses': len(analyses),
            'successful_analyses': len([a for a in analyses.values() if isinstance(a, dict) and 'error' not in a]),
            'failed_analyses': len([a for a in analyses.values() if isinstance(a, dict) and 'error' in a]),
            'coverage_percentage': 0.0
        }
        
        if summary['analysis_coverage']['total_analyses'] > 0:
            summary['analysis_coverage']['coverage_percentage'] = (
                summary['analysis_coverage']['successful_analyses'] / 
                summary['analysis_coverage']['total_analyses']
            ) * 100
        
        return summary
    
    def _extract_confidence_values(self, results: Dict[str, Any]) -> List[float]:
        """Extract confidence values from analysis results"""
        confidences = []
        
        def extract_recursive(obj, path=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if key == 'confidence' and isinstance(value, (int, float)):
                        confidences.append(float(value))
                    elif isinstance(value, (dict, list)):
                        extract_recursive(value, f"{path}.{key}" if path else key)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    if isinstance(item, (dict, list)):
                        extract_recursive(item, f"{path}[{i}]" if path else f"[{i}]")
        
        extract_recursive(results)
        return confidences
    
    def _save_analysis_results(self, results: Dict[str, Any]):
        """Save analysis results to file"""
        analysis_id = results.get('analysis_metadata', {}).get('analysis_id', 'unknown')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        filename = f"analysis_{analysis_id}_{timestamp}.json"
        filepath = Path(self.config['output_directory']) / filename
        
        try:
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"💾 Results saved: {filepath}")
        except Exception as e:
            print(f"⚠️ Failed to save results: {e}")
    
    def start_realtime_analysis(self, 
                               session_id: Optional[str] = None,
                               enable_visualization: bool = True,
                               callback: Optional[Callable] = None) -> str:
        """Start real-time audio analysis session"""
        if not self.components['realtime_processor']:
            raise RuntimeError("Real-time processor not available")
        
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        if session_id in self.active_realtime_sessions:
            raise RuntimeError(f"Session {session_id} already active")
        
        print(f"🎙️ Starting real-time analysis session: {session_id}")
        
        # Create session data
        session_data = {
            'session_id': session_id,
            'processor': self.components['realtime_processor'],
            'visualizer': None,
            'start_time': time.time(),
            'callbacks': [callback] if callback else []
        }
        
        # Add visualization if requested
        if enable_visualization and HAS_REALTIME:
            try:
                session_data['visualizer'] = RealTimeVisualizer(self.components['realtime_processor'])
                print("📊 Real-time visualization enabled")
            except Exception as e:
                print(f"⚠️ Visualization initialization failed: {e}")
        
        # Start processing
        success = self.components['realtime_processor'].start_processing()
        if not success:
            raise RuntimeError("Failed to start real-time processing")
        
        self.active_realtime_sessions[session_id] = session_data
        print(f"✅ Real-time session {session_id} started")
        
        return session_id
    
    def stop_realtime_analysis(self, session_id: str):
        """Stop real-time audio analysis session"""
        if session_id not in self.active_realtime_sessions:
            raise RuntimeError(f"Session {session_id} not found")
        
        session_data = self.active_realtime_sessions[session_id]
        
        # Stop processor
        session_data['processor'].stop_processing()
        
        # Close visualizer
        if session_data['visualizer']:
            session_data['visualizer'].close()
        
        # Remove from active sessions
        del self.active_realtime_sessions[session_id]
        
        print(f"🛑 Real-time session {session_id} stopped")
    
    def provide_user_feedback(self, analysis_id: str, feedback: Dict[str, float]):
        """Provide user feedback for meta-learning"""
        if not self.components['meta_learning_system']:
            print("⚠️ Meta-learning system not available for feedback")
            return
        
        self.components['meta_learning_system'].receive_user_feedback(analysis_id, feedback)
        print(f"✅ Feedback recorded for analysis {analysis_id}")
    
    def get_system_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive system performance report"""
        report = {
            'system_info': {
                'initialization_status': self.is_initialized,
                'available_modes': self.available_modes,
                'active_components': len([c for c in self.components.values() if c is not None]),
                'total_analyses': len(self.analysis_history),
                'active_realtime_sessions': len(self.active_realtime_sessions)
            },
            'performance_metrics': {},
            'learning_insights': {},
            'component_status': {}
        }
        
        # Component status
        for name, component in self.components.items():
            report['component_status'][name] = {
                'available': component is not None,
                'status': 'active' if component is not None else 'inactive'
            }
        
        # Get meta-learning insights if available
        if self.components['meta_learning_system']:
            learning_report = self.components['meta_learning_system'].generate_learning_report()
            report['learning_insights'] = learning_report
        
        # Analysis performance statistics
        if self.analysis_history:
            durations = [h.get('duration', 0) for h in self.analysis_history if 'duration' in h]
            if durations:
                report['performance_metrics'] = {
                    'average_analysis_time': float(np.mean(durations)),
                    'fastest_analysis': float(np.min(durations)),
                    'slowest_analysis': float(np.max(durations)),
                    'total_analysis_time': float(np.sum(durations))
                }
        
        return report
    
    def export_system_data(self, filepath: str):
        """Export all system data for backup or analysis"""
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'system_config': self.config,
            'analysis_history': self.analysis_history,
            'performance_report': self.get_system_performance_report()
        }
        
        # Add meta-learning data if available
        if self.components['meta_learning_system']:
            try:
                export_data['meta_learning_data'] = self.components['meta_learning_system'].generate_learning_report()
            except Exception as e:
                print(f"⚠️ Failed to export meta-learning data: {e}")
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        print(f"📤 System data exported to: {filepath}")


def main():
    """Main function for ultimate audio system demo"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Ultimate Audio Analysis System")
    
    # Analysis commands
    parser.add_argument("--analyze", help="Analyze audio file")
    parser.add_argument("--mode", default=AnalysisMode.COMPREHENSIVE, 
                       choices=[AnalysisMode.BASIC, AnalysisMode.ADVANCED, AnalysisMode.NEURAL,
                               AnalysisMode.OPTIMIZED, AnalysisMode.COMPREHENSIVE, AnalysisMode.ADAPTIVE],
                       help="Analysis mode")
    
    # Real-time commands
    parser.add_argument("--realtime", action="store_true", help="Start real-time analysis")
    parser.add_argument("--realtime-duration", type=float, default=60.0, help="Real-time duration in seconds")
    parser.add_argument("--no-visualization", action="store_true", help="Disable real-time visualization")
    
    # System commands
    parser.add_argument("--status", action="store_true", help="Show system status")
    parser.add_argument("--performance-report", action="store_true", help="Generate performance report")
    parser.add_argument("--export", help="Export system data to file")
    
    # Feedback commands
    parser.add_argument("--feedback", nargs=3, metavar=('ID', 'CATEGORY', 'SCORE'),
                       help="Provide user feedback: analysis_id category score")
    
    # Configuration
    parser.add_argument("--config", help="Configuration file (JSON)")
    parser.add_argument("--output-dir", help="Output directory")
    
    args = parser.parse_args()
    
    print("🎵 Ultimate Audio Analysis System")
    print("=" * 60)
    
    # Load configuration
    config = {}
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    if args.output_dir:
        config['output_directory'] = args.output_dir
    
    # Initialize system
    system = UltimateAudioSystem(config)
    
    # Handle commands
    if args.status:
        print("\n📊 System Status:")
        print("-" * 30)
        system._print_system_status()
    
    if args.performance_report:
        report = system.get_system_performance_report()
        print("\n📈 Performance Report:")
        print("-" * 30)
        print(json.dumps(report, indent=2, default=str))
    
    if args.analyze:
        if not os.path.exists(args.analyze):
            print(f"❌ File not found: {args.analyze}")
            return
        
        results = system.analyze_audio(args.analyze, mode=args.mode)
        
        # Display summary
        print(f"\n📊 Analysis Results ({args.mode}):")
        print("-" * 30)
        
        metadata = results.get('analysis_metadata', {})
        print(f"Analysis ID: {metadata.get('analysis_id', 'unknown')}")
        print(f"Processing Time: {metadata.get('timing', {}).get('total_time', 0):.2f} seconds")
        
        # Show key findings if available
        if 'analysis_summary' in results:
            summary = results['analysis_summary']
            print(f"Overall Confidence: {summary.get('overall_confidence', 0):.2f}")
            
            key_findings = summary.get('key_findings', [])
            if key_findings:
                print("Key Findings:")
                for finding in key_findings:
                    print(f"  • {finding}")
        
        # Show consensus results if available
        if 'combined_insights' in results:
            insights = results['combined_insights']
            
            if insights.get('consensus_genre'):
                genre_info = insights['consensus_genre']
                print(f"Consensus Genre: {genre_info.get('consensus_genre', 'unknown')} "
                      f"(confidence: {genre_info.get('confidence', 0):.2f})")
            
            if insights.get('consensus_mood'):
                mood_info = insights['consensus_mood']
                print(f"Consensus Mood: {mood_info.get('consensus_mood', 'unknown')} "
                      f"(confidence: {mood_info.get('confidence', 0):.2f})")
    
    if args.realtime:
        try:
            session_id = system.start_realtime_analysis(
                enable_visualization=not args.no_visualization
            )
            
            print(f"🎙️ Real-time analysis running for {args.realtime_duration} seconds...")
            print("Press Ctrl+C to stop early")
            
            time.sleep(args.realtime_duration)
            
            system.stop_realtime_analysis(session_id)
            
        except KeyboardInterrupt:
            print("\n⚠️ Interrupted by user")
            if 'session_id' in locals():
                system.stop_realtime_analysis(session_id)
        except Exception as e:
            print(f"❌ Real-time analysis failed: {e}")
    
    if args.feedback:
        analysis_id, category, score = args.feedback
        try:
            score = float(score)
            system.provide_user_feedback(analysis_id, {category: score})
        except ValueError:
            print(f"❌ Invalid score: {score}. Must be a number.")
    
    if args.export:
        system.export_system_data(args.export)
    
    print("\n✅ Ultimate Audio Analysis System session complete")


if __name__ == "__main__":
    main()