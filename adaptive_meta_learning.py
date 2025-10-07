#!/usr/bin/env python3
"""
Adaptive Meta-Learning Audio Analysis System
Self-improving system that learns from feedback and optimizes performance over time
"""

import os
import sys
import json
import pickle
import numpy as np
import warnings
from typing import Dict, List, Any, Optional, Tuple, Callable
from pathlib import Path
from datetime import datetime, timedelta
from collections import deque, defaultdict
import threading
import time
import hashlib

# Machine learning
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.model_selection import cross_val_score, GridSearchCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, r2_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# Optimization
try:
    from scipy.optimize import minimize, differential_evolution
    from scipy.stats import pearsonr, spearmanr
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# Neural networks for meta-learning
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, optimizers
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False

warnings.filterwarnings('ignore')

class PerformanceMetrics:
    """Track and analyze system performance metrics"""
    
    def __init__(self, max_history: int = 10000):
        self.max_history = max_history
        self.metrics_history = deque(maxlen=max_history)
        self.user_feedback = deque(maxlen=max_history)
        self.parameter_performance = defaultdict(list)
        
        # Performance categories
        self.categories = [
            'accuracy', 'speed', 'memory_usage', 'user_satisfaction',
            'genre_accuracy', 'mood_accuracy', 'tempo_accuracy'
        ]
        
        print("📊 Performance Metrics Tracker Initialized")
        print(f"   Max History: {max_history} entries")
        print(f"   Tracking Categories: {len(self.categories)}")
    
    def record_analysis(self, analysis_id: str, metrics: Dict[str, float], 
                       parameters: Dict[str, Any], results: Dict[str, Any]):
        """Record analysis performance metrics"""
        timestamp = datetime.now()
        
        record = {
            'analysis_id': analysis_id,
            'timestamp': timestamp.isoformat(),
            'metrics': metrics.copy(),
            'parameters': parameters.copy(),
            'results': results.copy(),
            'user_feedback': None  # Will be updated when feedback is received
        }
        
        self.metrics_history.append(record)
        
        # Update parameter performance tracking
        for param_name, param_value in parameters.items():
            self.parameter_performance[param_name].append({
                'value': param_value,
                'metrics': metrics.copy(),
                'timestamp': timestamp
            })
    
    def record_user_feedback(self, analysis_id: str, feedback: Dict[str, float]):
        """Record user feedback for an analysis"""
        # Find the analysis record and update it
        for record in reversed(self.metrics_history):
            if record['analysis_id'] == analysis_id:
                record['user_feedback'] = feedback.copy()
                
                # Update metrics with user feedback
                record['metrics'].update(feedback)
                break
        
        self.user_feedback.append({
            'analysis_id': analysis_id,
            'feedback': feedback.copy(),
            'timestamp': datetime.now().isoformat()
        })
    
    def get_performance_trends(self, category: str, 
                              window_days: int = 30) -> Dict[str, Any]:
        """Analyze performance trends for a category"""
        if category not in self.categories:
            return {"error": f"Unknown category: {category}"}
        
        cutoff_time = datetime.now() - timedelta(days=window_days)
        
        # Filter recent records
        recent_records = [
            r for r in self.metrics_history
            if datetime.fromisoformat(r['timestamp']) >= cutoff_time
            and category in r['metrics']
        ]
        
        if len(recent_records) < 2:
            return {"error": "Insufficient data for trend analysis"}
        
        # Extract values and timestamps
        values = [r['metrics'][category] for r in recent_records]
        timestamps = [datetime.fromisoformat(r['timestamp']) for r in recent_records]
        
        # Calculate trends
        time_deltas = [(t - timestamps[0]).total_seconds() for t in timestamps]
        
        # Linear correlation with time (trend direction)
        if HAS_SCIPY:
            trend_correlation, trend_p_value = pearsonr(time_deltas, values)
        else:
            trend_correlation = 0
            trend_p_value = 1
        
        return {
            'category': category,
            'window_days': window_days,
            'data_points': len(recent_records),
            'mean_performance': float(np.mean(values)),
            'std_performance': float(np.std(values)),
            'min_performance': float(np.min(values)),
            'max_performance': float(np.max(values)),
            'trend_correlation': float(trend_correlation),
            'trend_significant': trend_p_value < 0.05,
            'improvement_rate': float(trend_correlation * np.std(values)),
            'recent_performance': values[-5:] if len(values) >= 5 else values
        }
    
    def analyze_parameter_impact(self, parameter_name: str) -> Dict[str, Any]:
        """Analyze how a parameter affects performance"""
        if parameter_name not in self.parameter_performance:
            return {"error": f"No data for parameter: {parameter_name}"}
        
        param_data = self.parameter_performance[parameter_name]
        if len(param_data) < 10:
            return {"error": "Insufficient data for parameter analysis"}
        
        # Extract parameter values and corresponding performance
        param_values = [d['value'] for d in param_data]
        
        # Analyze impact on each metric category
        impact_analysis = {}
        
        for category in self.categories:
            metric_values = [
                d['metrics'].get(category, 0) for d in param_data
                if category in d['metrics']
            ]
            
            if len(metric_values) < 5:
                continue
            
            corresponding_params = param_values[:len(metric_values)]
            
            # Calculate correlation between parameter and performance
            if HAS_SCIPY and len(set(corresponding_params)) > 1:
                correlation, p_value = pearsonr(corresponding_params, metric_values)
                
                impact_analysis[category] = {
                    'correlation': float(correlation),
                    'significance': p_value < 0.05,
                    'effect_strength': abs(correlation),
                    'direction': 'positive' if correlation > 0 else 'negative',
                    'data_points': len(metric_values)
                }
        
        return {
            'parameter': parameter_name,
            'total_observations': len(param_data),
            'unique_values': len(set(param_values)),
            'value_range': {
                'min': min(param_values),
                'max': max(param_values),
                'mean': float(np.mean(param_values))
            },
            'impact_analysis': impact_analysis
        }
    
    def get_optimization_recommendations(self) -> Dict[str, Any]:
        """Generate recommendations for system optimization"""
        recommendations = {
            'parameter_tuning': [],
            'performance_insights': [],
            'user_feedback_insights': []
        }
        
        # Analyze each parameter's impact
        for param_name in self.parameter_performance.keys():
            impact = self.analyze_parameter_impact(param_name)
            
            if 'impact_analysis' in impact:
                for category, analysis in impact['impact_analysis'].items():
                    if analysis['significance'] and analysis['effect_strength'] > 0.3:
                        recommendations['parameter_tuning'].append({
                            'parameter': param_name,
                            'category': category,
                            'recommendation': f"{'Increase' if analysis['direction'] == 'positive' else 'Decrease'} {param_name} to improve {category}",
                            'confidence': analysis['effect_strength'],
                            'current_range': impact['value_range']
                        })
        
        # Analyze performance trends
        for category in self.categories:
            trend = self.get_performance_trends(category)
            
            if 'trend_correlation' in trend and trend['trend_significant']:
                if trend['trend_correlation'] < -0.3:
                    recommendations['performance_insights'].append({
                        'category': category,
                        'issue': f"{category} performance is declining",
                        'severity': 'high' if trend['trend_correlation'] < -0.5 else 'medium',
                        'suggestion': f"Review and optimize {category} related components"
                    })
                elif trend['trend_correlation'] > 0.3:
                    recommendations['performance_insights'].append({
                        'category': category,
                        'insight': f"{category} performance is improving",
                        'suggestion': f"Continue current approach for {category}"
                    })
        
        # Analyze user feedback patterns
        if self.user_feedback:
            recent_feedback = list(self.user_feedback)[-50:]  # Last 50 feedback entries
            
            # Calculate average satisfaction by category
            feedback_categories = defaultdict(list)
            for fb in recent_feedback:
                for key, value in fb['feedback'].items():
                    feedback_categories[key].append(value)
            
            for category, values in feedback_categories.items():
                avg_satisfaction = np.mean(values)
                if avg_satisfaction < 3.0:  # Assuming 1-5 scale
                    recommendations['user_feedback_insights'].append({
                        'category': category,
                        'issue': f"Low user satisfaction in {category}",
                        'average_score': float(avg_satisfaction),
                        'suggestion': f"Focus improvement efforts on {category}"
                    })
        
        return recommendations


class AdaptiveParameterOptimizer:
    """Automatically optimize system parameters based on performance feedback"""
    
    def __init__(self, metrics_tracker: PerformanceMetrics):
        self.metrics_tracker = metrics_tracker
        self.parameter_ranges = {}
        self.optimization_history = []
        self.current_best_params = {}
        
        # Optimization algorithms
        self.optimizers = {
            'gradient_boosting': None,
            'random_forest': None,
            'neural_network': None
        }
        
        self._initialize_optimizers()
        
        print("🎯 Adaptive Parameter Optimizer Initialized")
        print(f"   Available Optimizers: {list(self.optimizers.keys())}")
    
    def _initialize_optimizers(self):
        """Initialize optimization models"""
        if HAS_SKLEARN:
            self.optimizers['gradient_boosting'] = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
            
            self.optimizers['random_forest'] = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        
        if HAS_TENSORFLOW:
            # Simple neural network for parameter optimization
            self.optimizers['neural_network'] = self._build_optimization_network()
    
    def _build_optimization_network(self):
        """Build neural network for parameter optimization"""
        model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(None,)),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='linear')  # Predict performance score
        ])
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def register_parameter(self, name: str, min_val: float, max_val: float, 
                          param_type: str = 'float'):
        """Register a parameter for optimization"""
        self.parameter_ranges[name] = {
            'min': min_val,
            'max': max_val,
            'type': param_type
        }
        
        print(f"📝 Registered parameter: {name} [{min_val}, {max_val}] ({param_type})")
    
    def suggest_parameters(self, target_metric: str = 'user_satisfaction',
                          method: str = 'gradient_boosting') -> Dict[str, Any]:
        """Suggest optimal parameters based on historical performance"""
        if method not in self.optimizers or self.optimizers[method] is None:
            return self._random_parameters()
        
        # Get training data from metrics history
        X, y = self._prepare_training_data(target_metric)
        
        if len(X) < 10:
            print("⚠️ Insufficient data for optimization, using random parameters")
            return self._random_parameters()
        
        try:
            if method in ['gradient_boosting', 'random_forest']:
                return self._sklearn_optimization(X, y, method, target_metric)
            elif method == 'neural_network':
                return self._neural_optimization(X, y, target_metric)
            else:
                return self._random_parameters()
        
        except Exception as e:
            print(f"⚠️ Optimization failed: {e}, using random parameters")
            return self._random_parameters()
    
    def _prepare_training_data(self, target_metric: str) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data for optimization models"""
        X = []
        y = []
        
        for record in self.metrics_tracker.metrics_history:
            if target_metric in record['metrics']:
                # Extract parameter values
                param_vector = []
                for param_name in sorted(self.parameter_ranges.keys()):
                    if param_name in record['parameters']:
                        value = record['parameters'][param_name]
                        # Normalize to [0, 1]
                        param_range = self.parameter_ranges[param_name]
                        normalized = (value - param_range['min']) / (param_range['max'] - param_range['min'])
                        param_vector.append(normalized)
                    else:
                        param_vector.append(0.5)  # Default to middle of range
                
                if len(param_vector) == len(self.parameter_ranges):
                    X.append(param_vector)
                    y.append(record['metrics'][target_metric])
        
        return np.array(X), np.array(y)
    
    def _sklearn_optimization(self, X: np.ndarray, y: np.ndarray, 
                             method: str, target_metric: str) -> Dict[str, Any]:
        """Optimize using scikit-learn models"""
        model = self.optimizers[method]
        
        # Train the model
        model.fit(X, y)
        
        # Use optimization algorithm to find best parameters
        def objective(params):
            # Predict performance for these parameters
            prediction = model.predict([params])[0]
            return -prediction  # Minimize negative (maximize positive)
        
        # Optimize using scipy
        if HAS_SCIPY:
            result = differential_evolution(
                objective,
                bounds=[(0, 1)] * len(self.parameter_ranges),
                seed=42,
                maxiter=50
            )
            optimal_normalized = result.x
        else:
            # Fallback: grid search
            optimal_normalized = self._grid_search_optimization(model)
        
        # Convert back to original parameter ranges
        suggested_params = {}
        for i, (param_name, param_range) in enumerate(sorted(self.parameter_ranges.items())):
            value = optimal_normalized[i] * (param_range['max'] - param_range['min']) + param_range['min']
            
            if param_range['type'] == 'int':
                value = int(round(value))
            
            suggested_params[param_name] = value
        
        # Predict performance
        predicted_performance = model.predict([optimal_normalized])[0]
        
        return {
            'parameters': suggested_params,
            'predicted_performance': float(predicted_performance),
            'target_metric': target_metric,
            'optimization_method': method,
            'confidence': float(model.score(X, y)) if hasattr(model, 'score') else 0.0
        }
    
    def _neural_optimization(self, X: np.ndarray, y: np.ndarray, 
                            target_metric: str) -> Dict[str, Any]:
        """Optimize using neural network"""
        model = self.optimizers['neural_network']
        
        # Train the model
        model.fit(X, y, epochs=50, batch_size=min(32, len(X)), verbose=0)
        
        # Generate candidate parameter sets and evaluate
        n_candidates = 1000
        candidates = np.random.rand(n_candidates, len(self.parameter_ranges))
        
        # Predict performance for all candidates
        predictions = model.predict(candidates, verbose=0).flatten()
        
        # Select best candidate
        best_idx = np.argmax(predictions)
        optimal_normalized = candidates[best_idx]
        
        # Convert back to original parameter ranges
        suggested_params = {}
        for i, (param_name, param_range) in enumerate(sorted(self.parameter_ranges.items())):
            value = optimal_normalized[i] * (param_range['max'] - param_range['min']) + param_range['min']
            
            if param_range['type'] == 'int':
                value = int(round(value))
            
            suggested_params[param_name] = value
        
        return {
            'parameters': suggested_params,
            'predicted_performance': float(predictions[best_idx]),
            'target_metric': target_metric,
            'optimization_method': 'neural_network',
            'confidence': 0.8  # Neural networks are generally less interpretable
        }
    
    def _grid_search_optimization(self, model) -> np.ndarray:
        """Fallback grid search optimization"""
        best_params = None
        best_score = float('-inf')
        
        # Simple grid search (coarse)
        grid_size = 5
        for i in range(grid_size ** len(self.parameter_ranges)):
            params = []
            temp_i = i
            
            for _ in range(len(self.parameter_ranges)):
                params.append((temp_i % grid_size) / (grid_size - 1))
                temp_i //= grid_size
            
            score = model.predict([params])[0]
            if score > best_score:
                best_score = score
                best_params = params
        
        return np.array(best_params if best_params else [0.5] * len(self.parameter_ranges))
    
    def _random_parameters(self) -> Dict[str, Any]:
        """Generate random parameters within ranges"""
        suggested_params = {}
        
        for param_name, param_range in self.parameter_ranges.items():
            if param_range['type'] == 'int':
                value = np.random.randint(param_range['min'], param_range['max'] + 1)
            else:
                value = np.random.uniform(param_range['min'], param_range['max'])
            
            suggested_params[param_name] = value
        
        return {
            'parameters': suggested_params,
            'predicted_performance': 0.5,  # Unknown
            'target_metric': 'unknown',
            'optimization_method': 'random',
            'confidence': 0.0
        }


class MetaLearningSystem:
    """Meta-learning system that improves analysis performance over time"""
    
    def __init__(self, save_dir: str = "meta_learning_data"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Components
        self.metrics_tracker = PerformanceMetrics()
        self.parameter_optimizer = AdaptiveParameterOptimizer(self.metrics_tracker)
        
        # Learning state
        self.learning_iterations = 0
        self.last_optimization = None
        self.adaptation_threshold = 0.1  # Minimum improvement to adapt
        
        # Load existing data
        self._load_learning_state()
        
        print("🧠 Meta-Learning System Initialized")
        print(f"   Save Directory: {save_dir}")
        print(f"   Learning Iterations: {self.learning_iterations}")
    
    def register_optimizable_parameters(self):
        """Register standard audio analysis parameters for optimization"""
        # Audio processing parameters
        self.parameter_optimizer.register_parameter('sample_rate', 16000, 48000, 'int')
        self.parameter_optimizer.register_parameter('hop_length', 256, 1024, 'int')
        self.parameter_optimizer.register_parameter('n_fft', 1024, 4096, 'int')
        
        # Analysis parameters
        self.parameter_optimizer.register_parameter('analysis_window', 1.0, 10.0, 'float')
        self.parameter_optimizer.register_parameter('smoothing_factor', 0.1, 0.9, 'float')
        
        # ML model parameters
        self.parameter_optimizer.register_parameter('confidence_threshold', 0.1, 0.9, 'float')
        self.parameter_optimizer.register_parameter('feature_importance_cutoff', 0.01, 0.5, 'float')
        
        print("📝 Registered standard optimizable parameters")
    
    def record_analysis_performance(self, analysis_id: str, parameters: Dict[str, Any],
                                   results: Dict[str, Any], timing_info: Dict[str, float]):
        """Record performance of an analysis for learning"""
        # Calculate performance metrics
        metrics = self._calculate_performance_metrics(results, timing_info)
        
        # Record in metrics tracker
        self.metrics_tracker.record_analysis(analysis_id, metrics, parameters, results)
        
        print(f"📊 Recorded performance for analysis: {analysis_id}")
    
    def receive_user_feedback(self, analysis_id: str, feedback: Dict[str, float]):
        """Receive and incorporate user feedback"""
        self.metrics_tracker.record_user_feedback(analysis_id, feedback)
        
        # Trigger learning update if enough new feedback
        if len(self.metrics_tracker.user_feedback) % 10 == 0:
            self._update_learning()
        
        print(f"👤 Received user feedback for: {analysis_id}")
    
    def _calculate_performance_metrics(self, results: Dict[str, Any], 
                                     timing_info: Dict[str, float]) -> Dict[str, float]:
        """Calculate standardized performance metrics"""
        metrics = {}
        
        # Timing metrics
        metrics['processing_time'] = timing_info.get('total_time', 0)
        metrics['speed'] = 1.0 / max(timing_info.get('total_time', 1), 0.001)  # Inverse of time
        
        # Memory usage (if available)
        metrics['memory_usage'] = timing_info.get('memory_mb', 0)
        
        # Analysis quality metrics (heuristic-based)
        if 'advanced_features' in results:
            advanced = results['advanced_features']
            
            # Genre classification confidence
            if 'genre_classification' in advanced:
                metrics['genre_confidence'] = advanced['genre_classification'].get('confidence', 0)
            
            # Mood detection confidence
            if 'mood_detection' in advanced:
                metrics['mood_confidence'] = advanced['mood_detection'].get('confidence', 0)
            
            # Overall analysis completeness
            completed_analyses = len([k for k, v in advanced.items() if 'error' not in v])
            total_analyses = len(advanced)
            metrics['completeness'] = completed_analyses / max(total_analyses, 1)
        
        # Musical analysis quality
        if 'musical_structural' in results:
            musical = results['musical_structural']
            
            # Tempo detection confidence (based on beat consistency)
            if 'rhythm' in musical:
                rhythm = musical['rhythm']
                metrics['tempo_confidence'] = rhythm.get('rhythm_regularity', 0)
            
            # Harmonic analysis quality
            if 'harmony' in musical:
                harmony = musical['harmony']
                metrics['harmony_confidence'] = harmony.get('key_confidence', 0)
        
        # Default values for missing metrics
        default_metrics = {
            'accuracy': 0.7,  # Default assumption
            'user_satisfaction': 3.0,  # Neutral on 1-5 scale
            'genre_accuracy': 0.6,
            'mood_accuracy': 0.6,
            'tempo_accuracy': 0.7
        }
        
        for key, default_value in default_metrics.items():
            if key not in metrics:
                metrics[key] = default_value
        
        return metrics
    
    def get_optimized_parameters(self, target_metric: str = 'user_satisfaction') -> Dict[str, Any]:
        """Get optimized parameters for next analysis"""
        suggestion = self.parameter_optimizer.suggest_parameters(target_metric)
        
        print(f"🎯 Generated optimized parameters for {target_metric}")
        print(f"   Method: {suggestion.get('optimization_method', 'unknown')}")
        print(f"   Predicted Performance: {suggestion.get('predicted_performance', 0):.3f}")
        
        return suggestion
    
    def _update_learning(self):
        """Update learning models with new data"""
        self.learning_iterations += 1
        
        # Get performance trends
        trends = {}
        for category in self.metrics_tracker.categories:
            trend = self.metrics_tracker.get_performance_trends(category)
            if 'mean_performance' in trend:
                trends[category] = trend
        
        # Get optimization recommendations
        recommendations = self.metrics_tracker.get_optimization_recommendations()
        
        # Store learning update
        learning_update = {
            'iteration': self.learning_iterations,
            'timestamp': datetime.now().isoformat(),
            'performance_trends': trends,
            'recommendations': recommendations,
            'data_points': len(self.metrics_tracker.metrics_history)
        }
        
        # Save learning state
        self._save_learning_state(learning_update)
        
        print(f"🧠 Learning update #{self.learning_iterations} completed")
        print(f"   Data points: {learning_update['data_points']}")
        print(f"   Recommendations: {len(recommendations.get('parameter_tuning', []))}")
    
    def _save_learning_state(self, learning_update: Optional[Dict] = None):
        """Save current learning state"""
        state = {
            'learning_iterations': self.learning_iterations,
            'last_update': datetime.now().isoformat(),
            'parameter_ranges': self.parameter_optimizer.parameter_ranges,
            'metrics_count': len(self.metrics_tracker.metrics_history),
            'feedback_count': len(self.metrics_tracker.user_feedback)
        }
        
        if learning_update:
            state['last_learning_update'] = learning_update
        
        state_file = self.save_dir / "learning_state.json"
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        # Save metrics history
        metrics_file = self.save_dir / "metrics_history.pkl"
        with open(metrics_file, 'wb') as f:
            pickle.dump(list(self.metrics_tracker.metrics_history), f)
        
        # Save user feedback
        feedback_file = self.save_dir / "user_feedback.pkl"
        with open(feedback_file, 'wb') as f:
            pickle.dump(list(self.metrics_tracker.user_feedback), f)
    
    def _load_learning_state(self):
        """Load existing learning state"""
        state_file = self.save_dir / "learning_state.json"
        
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
                
                self.learning_iterations = state.get('learning_iterations', 0)
                
                # Restore parameter ranges
                if 'parameter_ranges' in state:
                    self.parameter_optimizer.parameter_ranges = state['parameter_ranges']
                
                print(f"📁 Loaded learning state: {self.learning_iterations} iterations")
                
            except Exception as e:
                print(f"⚠️ Failed to load learning state: {e}")
        
        # Load metrics history
        metrics_file = self.save_dir / "metrics_history.pkl"
        if metrics_file.exists():
            try:
                with open(metrics_file, 'rb') as f:
                    history = pickle.load(f)
                self.metrics_tracker.metrics_history.extend(history)
                print(f"📁 Loaded {len(history)} metrics records")
            except Exception as e:
                print(f"⚠️ Failed to load metrics history: {e}")
        
        # Load user feedback
        feedback_file = self.save_dir / "user_feedback.pkl"
        if feedback_file.exists():
            try:
                with open(feedback_file, 'rb') as f:
                    feedback = pickle.load(f)
                self.metrics_tracker.user_feedback.extend(feedback)
                print(f"📁 Loaded {len(feedback)} feedback records")
            except Exception as e:
                print(f"⚠️ Failed to load user feedback: {e}")
    
    def generate_learning_report(self) -> Dict[str, Any]:
        """Generate comprehensive learning report"""
        report = {
            'system_status': {
                'learning_iterations': self.learning_iterations,
                'total_analyses': len(self.metrics_tracker.metrics_history),
                'user_feedback_count': len(self.metrics_tracker.user_feedback),
                'parameters_tracked': len(self.parameter_optimizer.parameter_ranges)
            },
            'performance_trends': {},
            'optimization_recommendations': {},
            'learning_insights': []
        }
        
        # Get performance trends for all categories
        for category in self.metrics_tracker.categories:
            trend = self.metrics_tracker.get_performance_trends(category)
            if 'mean_performance' in trend:
                report['performance_trends'][category] = trend
        
        # Get optimization recommendations
        report['optimization_recommendations'] = self.metrics_tracker.get_optimization_recommendations()
        
        # Generate learning insights
        insights = []
        
        # Check for improving trends
        improving_categories = [
            cat for cat, trend in report['performance_trends'].items()
            if trend.get('trend_correlation', 0) > 0.3 and trend.get('trend_significant', False)
        ]
        
        if improving_categories:
            insights.append({
                'type': 'positive',
                'message': f"Performance improving in: {', '.join(improving_categories)}",
                'categories': improving_categories
            })
        
        # Check for declining trends
        declining_categories = [
            cat for cat, trend in report['performance_trends'].items()
            if trend.get('trend_correlation', 0) < -0.3 and trend.get('trend_significant', False)
        ]
        
        if declining_categories:
            insights.append({
                'type': 'warning',
                'message': f"Performance declining in: {', '.join(declining_categories)}",
                'categories': declining_categories
            })
        
        # Check parameter optimization opportunities
        param_recommendations = report['optimization_recommendations'].get('parameter_tuning', [])
        high_impact_params = [
            rec for rec in param_recommendations
            if rec.get('confidence', 0) > 0.5
        ]
        
        if high_impact_params:
            insights.append({
                'type': 'optimization',
                'message': f"High-impact parameter optimizations available: {len(high_impact_params)}",
                'parameters': [rec['parameter'] for rec in high_impact_params]
            })
        
        report['learning_insights'] = insights
        
        return report
    
    def export_learning_data(self, filepath: str):
        """Export all learning data for analysis or backup"""
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'learning_report': self.generate_learning_report(),
            'metrics_history': list(self.metrics_tracker.metrics_history),
            'user_feedback': list(self.metrics_tracker.user_feedback),
            'parameter_ranges': self.parameter_optimizer.parameter_ranges,
            'parameter_performance': dict(self.metrics_tracker.parameter_performance)
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        print(f"📤 Exported learning data to: {filepath}")


def main():
    """Main function for meta-learning system demo"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Adaptive Meta-Learning Audio Analysis")
    parser.add_argument("--init", action="store_true", help="Initialize meta-learning system")
    parser.add_argument("--report", action="store_true", help="Generate learning report")
    parser.add_argument("--optimize", help="Get optimized parameters for metric")
    parser.add_argument("--feedback", nargs=3, metavar=('ID', 'CATEGORY', 'SCORE'),
                       help="Add user feedback: analysis_id category score")
    parser.add_argument("--export", help="Export learning data to file")
    
    args = parser.parse_args()
    
    print("🧠 Adaptive Meta-Learning Audio Analysis System")
    print("=" * 50)
    
    # Initialize system
    meta_learner = MetaLearningSystem()
    
    if args.init:
        meta_learner.register_optimizable_parameters()
        print("✅ Meta-learning system initialized with standard parameters")
    
    if args.report:
        report = meta_learner.generate_learning_report()
        
        print("\n📊 LEARNING REPORT")
        print("-" * 30)
        
        status = report['system_status']
        print(f"Learning Iterations: {status['learning_iterations']}")
        print(f"Total Analyses: {status['total_analyses']}")
        print(f"User Feedback Count: {status['user_feedback_count']}")
        print(f"Parameters Tracked: {status['parameters_tracked']}")
        
        print(f"\n🎯 Performance Trends:")
        for category, trend in report['performance_trends'].items():
            direction = "↗️" if trend.get('trend_correlation', 0) > 0.1 else "↘️" if trend.get('trend_correlation', 0) < -0.1 else "→"
            print(f"   {category}: {trend.get('mean_performance', 0):.3f} {direction}")
        
        print(f"\n💡 Learning Insights:")
        for insight in report['learning_insights']:
            icon = "✅" if insight['type'] == 'positive' else "⚠️" if insight['type'] == 'warning' else "🎯"
            print(f"   {icon} {insight['message']}")
    
    if args.optimize:
        suggestion = meta_learner.get_optimized_parameters(args.optimize)
        
        print(f"\n🎯 Optimized Parameters for '{args.optimize}':")
        print(f"Method: {suggestion.get('optimization_method', 'unknown')}")
        print(f"Predicted Performance: {suggestion.get('predicted_performance', 0):.3f}")
        print(f"Confidence: {suggestion.get('confidence', 0):.3f}")
        
        print("\nParameters:")
        for param, value in suggestion.get('parameters', {}).items():
            print(f"   {param}: {value}")
    
    if args.feedback:
        analysis_id, category, score = args.feedback
        try:
            score = float(score)
            meta_learner.receive_user_feedback(analysis_id, {category: score})
            print(f"✅ Recorded feedback: {category} = {score} for analysis {analysis_id}")
        except ValueError:
            print(f"❌ Invalid score: {score}. Must be a number.")
    
    if args.export:
        meta_learner.export_learning_data(args.export)


if __name__ == "__main__":
    main()