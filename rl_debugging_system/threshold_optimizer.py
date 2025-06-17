#!/usr/bin/env python3
"""
Action Threshold Optimizer and Analysis
Critical component for understanding RL performance issues

Key purposes:
1. Find optimal threshold for converting continuous RL actions to binary
2. Analyze action space distribution and sparsity patterns
3. Compare RL action patterns with expert demonstrations
4. Identify if poor performance is due to action conversion issues
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import Dict, List, Any, Tuple
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve, auc
from sklearn.metrics import classification_report, confusion_matrix
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from tqdm import tqdm


class ActionThresholdOptimizer:
    """
    Comprehensive action threshold optimization and analysis.
    
    This helps determine if poor RL performance is due to:
    1. Poor action threshold selection
    2. Wrong action space conversion
    3. Mismatch between RL output and expected binary actions
    4. Action sparsity handling issues
    """
    
    def __init__(self, logger, save_dir: str = "threshold_analysis"):
        self.logger = logger
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Analysis results storage
        self.threshold_results = {}
        self.action_analysis = {}
        self.expert_analysis = {}
        
        self.logger.info(f"🎯 ActionThresholdOptimizer initialized")
        self.logger.info(f"📁 Results will be saved to: {self.save_dir}")
    
    def analyze_rl_model_comprehensive(self, rl_model, test_data: List[Dict], 
                                     model_name: str = "RL_Model") -> Dict[str, Any]:
        """
        Comprehensive analysis of RL model action predictions.
        
        Args:
            rl_model: Trained RL model (PPO, A2C, etc.)
            test_data: List of video dictionaries with expert actions
            model_name: Name for saving results
            
        Returns:
            Comprehensive analysis results
        """
        
        self.logger.info(f"🎯 Analyzing {model_name} action predictions...")
        
        # Step 1: Collect predictions and expert actions
        predictions, expert_actions, metadata = self._collect_model_predictions(
            rl_model, test_data, model_name
        )
        
        if len(predictions) == 0:
            self.logger.error("No predictions collected!")
            return {'status': 'failed', 'error': 'no_predictions'}
        
        # Step 2: Analyze action distributions
        distribution_analysis = self._analyze_action_distributions(
            predictions, expert_actions, model_name
        )
        
        # Step 3: Optimize thresholds for mAP
        threshold_optimization = self._optimize_thresholds_for_map(
            predictions, expert_actions, model_name
        )
        
        # Step 4: Analyze action sparsity patterns
        sparsity_analysis = self._analyze_action_sparsity(
            predictions, expert_actions, model_name
        )
        
        # Step 5: Per-action analysis
        per_action_analysis = self._analyze_per_action_performance(
            predictions, expert_actions, model_name
        )
        
        # Step 6: Create comprehensive visualizations
        self._create_comprehensive_visualizations(
            predictions, expert_actions, model_name, threshold_optimization
        )
        
        # Compile results
        analysis_results = {
            'model_name': model_name,
            'status': 'success',
            'num_predictions': len(predictions),
            'num_videos': len(test_data),
            'distribution_analysis': distribution_analysis,
            'threshold_optimization': threshold_optimization,
            'sparsity_analysis': sparsity_analysis,
            'per_action_analysis': per_action_analysis,
            'metadata': metadata,
            'recommendations': self._generate_recommendations(
                distribution_analysis, threshold_optimization, sparsity_analysis
            )
        }
        
        # Save results
        self._save_analysis_results(analysis_results, model_name)
        
        return analysis_results
    
    def _collect_model_predictions(self, rl_model, test_data: List[Dict], 
                                 model_name: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Collect predictions from RL model on test data."""
        
        self.logger.info(f"📊 Collecting predictions from {model_name}...")
        
        all_predictions = []
        all_expert_actions = []
        all_metadata = []
        
        for video_idx, video in enumerate(tqdm(test_data, desc="Processing videos")):
            video_id = video['video_id']
            frames = video['frame_embeddings']
            expert_actions = video['actions_binaries']
            
            # Sample frames from video (to avoid overwhelming computation)
            num_frames = len(frames)
            sample_indices = np.linspace(0, num_frames-1, min(50, num_frames), dtype=int)
            
            for frame_idx in sample_indices:
                try:
                    # Get model prediction
                    state = frames[frame_idx].reshape(1, -1)
                    action_pred, _ = rl_model.predict(state, deterministic=True)
                    
                    # Process action prediction
                    processed_action = self._process_action_prediction(action_pred)
                    
                    # Store results
                    all_predictions.append(processed_action)
                    all_expert_actions.append(expert_actions[frame_idx])
                    all_metadata.append({
                        'video_id': video_id,
                        'video_idx': video_idx,
                        'frame_idx': frame_idx,
                        'original_frame_idx': frame_idx
                    })
                    
                except Exception as e:
                    self.logger.warning(f"Prediction failed for video {video_id}, frame {frame_idx}: {e}")
                    continue
        
        predictions = np.array(all_predictions)
        expert_actions = np.array(all_expert_actions)
        
        metadata = {
            'total_samples': len(all_predictions),
            'videos_processed': len(test_data),
            'frames_per_video': len(all_predictions) // len(test_data) if test_data else 0,
            'video_details': all_metadata
        }
        
        self.logger.info(f"✅ Collected {len(predictions)} predictions from {len(test_data)} videos")
        return predictions, expert_actions, metadata
    
    def _process_action_prediction(self, action_pred) -> np.ndarray:
        """Process raw RL action prediction to standard format."""
        
        if isinstance(action_pred, torch.Tensor):
            action_pred = action_pred.cpu().numpy()
        
        action_pred = np.array(action_pred).flatten()
        
        # Ensure 100 dimensions
        if len(action_pred) != 100:
            processed = np.zeros(100, dtype=np.float32)
            if len(action_pred) > 0:
                copy_len = min(len(action_pred), 100)
                processed[:copy_len] = action_pred[:copy_len]
            action_pred = processed
        
        # Ensure [0,1] range for continuous actions
        action_pred = np.clip(action_pred, 0.0, 1.0)
        
        return action_pred
    
    def _analyze_action_distributions(self, predictions: np.ndarray, 
                                    expert_actions: np.ndarray, model_name: str) -> Dict:
        """Analyze the distribution of action predictions vs expert actions."""
        
        self.logger.info("📊 Analyzing action distributions...")
        
        analysis = {
            'prediction_stats': {
                'mean': float(np.mean(predictions)),
                'std': float(np.std(predictions)),
                'min': float(np.min(predictions)),
                'max': float(np.max(predictions)),
                'median': float(np.median(predictions)),
                'q25': float(np.percentile(predictions, 25)),
                'q75': float(np.percentile(predictions, 75))
            },
            'expert_stats': {
                'mean': float(np.mean(expert_actions)),
                'sparsity': float(np.mean(expert_actions)),  # Proportion of positive actions
                'avg_actions_per_frame': float(np.mean(np.sum(expert_actions, axis=1))),
                'std_actions_per_frame': float(np.std(np.sum(expert_actions, axis=1)))
            },
            'distribution_comparison': {},
            'action_density_comparison': {}
        }
        
        # Compare action densities at different thresholds
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
        for threshold in thresholds:
            binary_preds = (predictions > threshold).astype(int)
            pred_density = np.mean(np.sum(binary_preds, axis=1))
            expert_density = np.mean(np.sum(expert_actions, axis=1))
            
            analysis['action_density_comparison'][f'threshold_{threshold}'] = {
                'predicted_density': float(pred_density),
                'expert_density': float(expert_density),
                'density_ratio': float(pred_density / expert_density) if expert_density > 0 else 0.0,
                'density_difference': float(abs(pred_density - expert_density))
            }
        
        # Analyze prediction distribution shape
        pred_flat = predictions.flatten()
        analysis['distribution_comparison'] = {
            'predictions_in_0_to_0.2': float(np.mean(pred_flat < 0.2)),
            'predictions_in_0.2_to_0.4': float(np.mean((pred_flat >= 0.2) & (pred_flat < 0.4))),
            'predictions_in_0.4_to_0.6': float(np.mean((pred_flat >= 0.4) & (pred_flat < 0.6))),
            'predictions_in_0.6_to_0.8': float(np.mean((pred_flat >= 0.6) & (pred_flat < 0.8))),
            'predictions_in_0.8_to_1.0': float(np.mean(pred_flat >= 0.8)),
            'prediction_entropy': float(-np.sum(pred_flat * np.log(pred_flat + 1e-8)) / len(pred_flat))
        }
        
        return analysis
    
    def _optimize_thresholds_for_map(self, predictions: np.ndarray, 
                                   expert_actions: np.ndarray, model_name: str) -> Dict:
        """Find optimal thresholds for maximizing mAP."""
        
        self.logger.info("🎯 Optimizing thresholds for mAP...")
        
        # Test range of thresholds
        thresholds = np.arange(0.1, 0.95, 0.05)
        threshold_results = {}
        
        best_threshold = 0.5
        best_map = 0.0
        
        for threshold in tqdm(thresholds, desc="Testing thresholds"):
            # Convert predictions to binary using threshold
            binary_preds = (predictions > threshold).astype(int)
            
            # Calculate metrics
            metrics = self._calculate_threshold_metrics(
                binary_preds, expert_actions, predictions, threshold
            )
            
            threshold_results[f'threshold_{threshold:.2f}'] = metrics
            
            # Track best mAP
            if metrics['mAP'] > best_map:
                best_map = metrics['mAP']
                best_threshold = threshold
        
        # Analyze threshold sensitivity
        threshold_sensitivity = self._analyze_threshold_sensitivity(threshold_results)
        
        optimization_results = {
            'best_threshold': float(best_threshold),
            'best_mAP': float(best_map),
            'threshold_results': threshold_results,
            'threshold_sensitivity': threshold_sensitivity,
            'default_threshold_performance': threshold_results.get('threshold_0.50', {}),
            'improvement_vs_default': float(best_map - threshold_results.get('threshold_0.50', {}).get('mAP', 0.0))
        }
        
        self.logger.info(f"🎯 Optimal threshold: {best_threshold:.2f} (mAP: {best_map:.4f})")
        self.logger.info(f"   Improvement vs 0.5: {optimization_results['improvement_vs_default']:.4f}")
        
        return optimization_results
    
    def _calculate_threshold_metrics(self, binary_preds: np.ndarray, expert_actions: np.ndarray,
                                   continuous_preds: np.ndarray, threshold: float) -> Dict:
        """Calculate comprehensive metrics for a given threshold."""
        
        # Basic metrics
        exact_match = np.mean(np.all(binary_preds == expert_actions, axis=1))
        hamming_accuracy = np.mean(binary_preds == expert_actions)
        
        # Action density metrics
        pred_density = np.mean(np.sum(binary_preds, axis=1))
        expert_density = np.mean(np.sum(expert_actions, axis=1))
        
        # Calculate mAP (focusing on present actions only)
        ap_scores = []
        present_actions = 0
        
        for action_idx in range(expert_actions.shape[1]):
            if np.sum(expert_actions[:, action_idx]) > 0:  # Action is present in dataset
                present_actions += 1
                try:
                    # Use continuous predictions for AP calculation
                    ap = average_precision_score(
                        expert_actions[:, action_idx], 
                        continuous_preds[:, action_idx]
                    )
                    ap_scores.append(ap)
                except:
                    ap_scores.append(0.0)
        
        mAP = np.mean(ap_scores) if ap_scores else 0.0
        
        # Per-class precision, recall, F1
        try:
            from sklearn.metrics import precision_recall_fscore_support
            precision, recall, f1, _ = precision_recall_fscore_support(
                expert_actions.flatten(), binary_preds.flatten(), 
                average='macro', zero_division=0
            )
        except:
            precision = recall = f1 = 0.0
        
        return {
            'threshold': float(threshold),
            'mAP': float(mAP),
            'exact_match': float(exact_match),
            'hamming_accuracy': float(hamming_accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'predicted_density': float(pred_density),
            'expert_density': float(expert_density),
            'density_ratio': float(pred_density / expert_density) if expert_density > 0 else 0.0,
            'present_actions': int(present_actions),
            'total_actions': int(expert_actions.shape[1])
        }
    
    def _analyze_threshold_sensitivity(self, threshold_results: Dict) -> Dict:
        """Analyze how sensitive performance is to threshold choice."""
        
        thresholds = []
        maps = []
        f1s = []
        densities = []
        
        for key, results in threshold_results.items():
            if key.startswith('threshold_'):
                threshold = results['threshold']
                thresholds.append(threshold)
                maps.append(results['mAP'])
                f1s.append(results['f1'])
                densities.append(results['predicted_density'])
        
        # Sort by threshold
        sorted_indices = np.argsort(thresholds)
        thresholds = np.array(thresholds)[sorted_indices]
        maps = np.array(maps)[sorted_indices]
        f1s = np.array(f1s)[sorted_indices]
        densities = np.array(densities)[sorted_indices]
        
        # Calculate sensitivity metrics
        map_variance = np.var(maps)
        map_range = np.max(maps) - np.min(maps)
        
        # Find stable regions (low variance)
        window_size = 3
        if len(maps) >= window_size:
            windowed_variance = []
            for i in range(len(maps) - window_size + 1):
                window_var = np.var(maps[i:i+window_size])
                windowed_variance.append(window_var)
            most_stable_idx = np.argmin(windowed_variance)
            most_stable_threshold = thresholds[most_stable_idx + window_size//2]
        else:
            most_stable_threshold = 0.5
        
        return {
            'mAP_variance': float(map_variance),
            'mAP_range': float(map_range),
            'most_stable_threshold': float(most_stable_threshold),
            'threshold_sensitivity': 'high' if map_range > 0.05 else 'moderate' if map_range > 0.02 else 'low',
            'optimal_range': {
                'min_threshold': float(thresholds[np.argmax(maps) - 1]) if np.argmax(maps) > 0 else float(thresholds[0]),
                'max_threshold': float(thresholds[np.argmax(maps) + 1]) if np.argmax(maps) < len(thresholds)-1 else float(thresholds[-1]),
                'optimal_threshold': float(thresholds[np.argmax(maps)])
            }
        }
    
    def _analyze_action_sparsity(self, predictions: np.ndarray, 
                               expert_actions: np.ndarray, model_name: str) -> Dict:
        """Analyze action sparsity patterns and their impact on performance."""
        
        self.logger.info("🔍 Analyzing action sparsity patterns...")
        
        expert_densities = np.sum(expert_actions, axis=1)
        
        analysis = {
            'expert_sparsity_stats': {
                'mean_actions_per_frame': float(np.mean(expert_densities)),
                'std_actions_per_frame': float(np.std(expert_densities)),
                'min_actions_per_frame': int(np.min(expert_densities)),
                'max_actions_per_frame': int(np.max(expert_densities)),
                'frames_with_0_actions': float(np.mean(expert_densities == 0)),
                'frames_with_1_action': float(np.mean(expert_densities == 1)),
                'frames_with_2_actions': float(np.mean(expert_densities == 2)),
                'frames_with_3_actions': float(np.mean(expert_densities == 3)),
                'frames_with_4_plus_actions': float(np.mean(expert_densities >= 4))
            },
            'prediction_sparsity_analysis': {},
            'sparsity_matching_performance': {}
        }
        
        # Analyze predictions at different thresholds
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
        for threshold in thresholds:
            binary_preds = (predictions > threshold).astype(int)
            pred_densities = np.sum(binary_preds, axis=1)
            
            analysis['prediction_sparsity_analysis'][f'threshold_{threshold}'] = {
                'mean_actions_per_frame': float(np.mean(pred_densities)),
                'std_actions_per_frame': float(np.std(pred_densities)),
                'frames_with_0_actions': float(np.mean(pred_densities == 0)),
                'frames_with_1_action': float(np.mean(pred_densities == 1)),
                'frames_with_2_actions': float(np.mean(pred_densities == 2)),
                'frames_with_3_actions': float(np.mean(pred_densities == 3)),
                'frames_with_4_plus_actions': float(np.mean(pred_densities >= 4))
            }
            
            # Calculate performance for frames with different sparsity levels
            sparsity_performance = {}
            for target_density in [0, 1, 2, 3]:
                mask = expert_densities == target_density
                if np.sum(mask) > 0:
                    subset_expert = expert_actions[mask]
                    subset_pred = binary_preds[mask]
                    subset_accuracy = np.mean(subset_pred == subset_expert)
                    sparsity_performance[f'frames_with_{target_density}_actions'] = float(subset_accuracy)
            
            analysis['sparsity_matching_performance'][f'threshold_{threshold}'] = sparsity_performance
        
        return analysis
    
    def _analyze_per_action_performance(self, predictions: np.ndarray, 
                                      expert_actions: np.ndarray, model_name: str) -> Dict:
        """Analyze performance for individual actions."""
        
        self.logger.info("🔍 Analyzing per-action performance...")
        
        num_actions = expert_actions.shape[1]
        per_action_stats = {}
        
        for action_idx in range(num_actions):
            expert_action = expert_actions[:, action_idx]
            pred_action = predictions[:, action_idx]
            
            # Only analyze actions that appear in the dataset
            if np.sum(expert_action) > 0:
                try:
                    ap = average_precision_score(expert_action, pred_action)
                except:
                    ap = 0.0
                
                # Calculate optimal threshold for this action
                binary_preds_05 = (pred_action > 0.5).astype(int)
                accuracy_05 = np.mean(binary_preds_05 == expert_action)
                
                per_action_stats[f'action_{action_idx:03d}'] = {
                    'frequency': float(np.mean(expert_action)),
                    'total_occurrences': int(np.sum(expert_action)),
                    'average_precision': float(ap),
                    'accuracy_at_0.5': float(accuracy_05),
                    'prediction_mean': float(np.mean(pred_action)),
                    'prediction_std': float(np.std(pred_action))
                }
        
        # Summary statistics
        aps = [stats['average_precision'] for stats in per_action_stats.values()]
        frequencies = [stats['frequency'] for stats in per_action_stats.values()]
        
        summary = {
            'num_actions_analyzed': len(per_action_stats),
            'mean_average_precision': float(np.mean(aps)) if aps else 0.0,
            'std_average_precision': float(np.std(aps)) if aps else 0.0,
            'min_average_precision': float(np.min(aps)) if aps else 0.0,
            'max_average_precision': float(np.max(aps)) if aps else 0.0,
            'mean_action_frequency': float(np.mean(frequencies)) if frequencies else 0.0,
            'actions_with_good_performance': len([ap for ap in aps if ap > 0.1]) if aps else 0,
            'actions_with_poor_performance': len([ap for ap in aps if ap < 0.05]) if aps else 0
        }
        
        return {
            'per_action_stats': per_action_stats,
            'summary': summary
        }
    
    def _create_comprehensive_visualizations(self, predictions: np.ndarray, 
                                           expert_actions: np.ndarray, model_name: str,
                                           threshold_optimization: Dict):
        """Create comprehensive visualizations for threshold analysis."""
        
        self.logger.info("📊 Creating comprehensive visualizations...")
        
        # Create subplot figure
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Threshold vs mAP', 'Action Density Comparison',
                'Prediction Distribution', 'Per-Action Performance',
                'Precision-Recall Curves', 'Threshold Sensitivity'
            ],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. Threshold vs mAP
        threshold_results = threshold_optimization['threshold_results']
        thresholds = []
        maps = []
        for key, results in threshold_results.items():
            if key.startswith('threshold_'):
                thresholds.append(results['threshold'])
                maps.append(results['mAP'])
        
        # Sort by threshold
        sorted_indices = np.argsort(thresholds)
        thresholds = np.array(thresholds)[sorted_indices]
        maps = np.array(maps)[sorted_indices]
        
        fig.add_trace(
            go.Scatter(x=thresholds, y=maps, mode='lines+markers', 
                      name='mAP vs Threshold', line=dict(color='blue', width=2)),
            row=1, col=1
        )
        
        # Mark optimal threshold
        best_threshold = threshold_optimization['best_threshold']
        best_map = threshold_optimization['best_mAP']
        fig.add_trace(
            go.Scatter(x=[best_threshold], y=[best_map], mode='markers',
                      marker=dict(color='red', size=10), name='Optimal Threshold'),
            row=1, col=1
        )
        
        # 2. Action Density Comparison
        expert_densities = np.sum(expert_actions, axis=1)
        pred_densities_05 = np.sum((predictions > 0.5).astype(int), axis=1)
        
        fig.add_trace(
            go.Histogram(x=expert_densities, name='Expert Density', 
                        opacity=0.7, nbinsx=20),
            row=1, col=2
        )
        fig.add_trace(
            go.Histogram(x=pred_densities_05, name='Predicted Density (0.5)', 
                        opacity=0.7, nbinsx=20),
            row=1, col=2
        )
        
        # 3. Prediction Distribution
        pred_flat = predictions.flatten()
        fig.add_trace(
            go.Histogram(x=pred_flat, name='Prediction Distribution', 
                        nbinsx=50, opacity=0.7),
            row=2, col=1
        )
        
        # 4. Per-Action Performance (sample)
        # Show performance for actions with different frequencies
        expert_freqs = np.mean(expert_actions, axis=0)
        present_actions = expert_freqs > 0
        
        if np.sum(present_actions) > 0:
            sample_actions = np.where(present_actions)[0][:20]  # First 20 present actions
            action_aps = []
            
            for action_idx in sample_actions:
                try:
                    ap = average_precision_score(
                        expert_actions[:, action_idx], predictions[:, action_idx]
                    )
                    action_aps.append(ap)
                except:
                    action_aps.append(0.0)
            
            fig.add_trace(
                go.Bar(x=[f'A{i}' for i in sample_actions], y=action_aps,
                      name='Action AP', marker_color='lightblue'),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title_text=f"Comprehensive Threshold Analysis: {model_name}",
            showlegend=True,
            height=900
        )
        
        # Save interactive plot
        fig.write_html(str(self.save_dir / f'{model_name}_threshold_analysis.html'))
        
        # Also create static version
        fig.write_image(str(self.save_dir / f'{model_name}_threshold_analysis.png'), 
                       width=1200, height=900)
        
        self.logger.info(f"📊 Visualizations saved to {self.save_dir}")
    
    def _generate_recommendations(self, distribution_analysis: Dict, 
                                threshold_optimization: Dict, sparsity_analysis: Dict) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        
        recommendations = []
        
        # Threshold recommendations
        best_threshold = threshold_optimization['best_threshold']
        improvement = threshold_optimization['improvement_vs_default']
        
        if improvement > 0.02:  # Significant improvement
            recommendations.append(
                f"Use threshold {best_threshold:.2f} instead of 0.5 (improves mAP by {improvement:.3f})"
            )
        elif improvement > 0.01:
            recommendations.append(
                f"Consider threshold {best_threshold:.2f} for modest improvement ({improvement:.3f})"
            )
        
        # Distribution recommendations
        pred_stats = distribution_analysis['prediction_stats']
        if pred_stats['mean'] < 0.3:
            recommendations.append(
                "Predictions are too conservative (low values) - consider reward function changes"
            )
        elif pred_stats['mean'] > 0.7:
            recommendations.append(
                "Predictions are too aggressive (high values) - consider adding sparsity penalties"
            )
        
        # Sparsity recommendations
        expert_stats = distribution_analysis['expert_stats']
        density_comparison = distribution_analysis['action_density_comparison']
        
        best_density_match = min(density_comparison.values(), 
                               key=lambda x: x['density_difference'])
        best_threshold_for_density = [k for k, v in density_comparison.items() 
                                    if v == best_density_match][0]
        
        if best_density_match['density_difference'] > 1.0:
            recommendations.append(
                f"Action density mismatch is large - consider using {best_threshold_for_density.replace('threshold_', '')} for better density matching"
            )
        
        # Performance recommendations
        best_map = threshold_optimization['best_mAP']
        if best_map < 0.05:
            recommendations.append(
                "Very low mAP (<5%) suggests fundamental issues beyond thresholding"
            )
            recommendations.append(
                "Consider: 1) Reward function alignment, 2) Model architecture, 3) Training data quality"
            )
        elif best_map < 0.1:
            recommendations.append(
                "Low mAP (5-10%) suggests room for improvement in model training or reward design"
            )
        
        return recommendations
    
    def _save_analysis_results(self, analysis_results: Dict, model_name: str):
        """Save comprehensive analysis results."""
        
        # Save JSON results
        json_path = self.save_dir / f'{model_name}_threshold_analysis.json'
        with open(json_path, 'w') as f:
            json.dump(analysis_results, f, indent=2, default=str)
        
        # Save CSV summary for easy analysis
        threshold_results = analysis_results['threshold_optimization']['threshold_results']
        threshold_df = pd.DataFrame(threshold_results).T
        threshold_df.to_csv(self.save_dir / f'{model_name}_threshold_summary.csv')
        
        # Save recommendations as text
        recommendations_path = self.save_dir / f'{model_name}_recommendations.txt'
        with open(recommendations_path, 'w') as f:
            f.write(f"Threshold Analysis Recommendations for {model_name}\n")
            f.write("=" * 50 + "\n\n")
            
            for i, rec in enumerate(analysis_results['recommendations'], 1):
                f.write(f"{i}. {rec}\n")
            
            f.write(f"\nKey Findings:\n")
            f.write(f"- Optimal threshold: {analysis_results['threshold_optimization']['best_threshold']:.2f}\n")
            f.write(f"- Best mAP: {analysis_results['threshold_optimization']['best_mAP']:.4f}\n")
            f.write(f"- Improvement vs 0.5: {analysis_results['threshold_optimization']['improvement_vs_default']:.4f}\n")
        
        self.logger.info(f"💾 Analysis results saved:")
        self.logger.info(f"   JSON: {json_path}")
        self.logger.info(f"   CSV: {self.save_dir / f'{model_name}_threshold_summary.csv'}")
        self.logger.info(f"   Recommendations: {recommendations_path}")


def run_threshold_optimization(rl_model, test_data: List[Dict], 
                             model_name: str, logger, save_dir: str = "threshold_analysis"):
    """
    Main function to run comprehensive threshold optimization and analysis.
    
    This is critical for understanding why RL might be underperforming.
    """
    
    logger.info("🎯 RUNNING COMPREHENSIVE THRESHOLD OPTIMIZATION")
    logger.info("=" * 60)
    logger.info("Goal: Determine if poor RL performance is due to action threshold issues")
    
    # Initialize optimizer
    optimizer = ActionThresholdOptimizer(logger, save_dir)
    
    # Run comprehensive analysis
    analysis_results = optimizer.analyze_rl_model_comprehensive(
        rl_model, test_data, model_name
    )
    
    if analysis_results['status'] == 'success':
        best_threshold = analysis_results['threshold_optimization']['best_threshold']
        best_map = analysis_results['threshold_optimization']['best_mAP']
        improvement = analysis_results['threshold_optimization']['improvement_vs_default']
        
        logger.info("🎯 THRESHOLD OPTIMIZATION COMPLETE!")
        logger.info(f"   Optimal threshold: {best_threshold:.2f}")
        logger.info(f"   Best mAP: {best_map:.4f}")
        logger.info(f"   Improvement vs 0.5: {improvement:.4f}")
        
        if improvement > 0.02:
            logger.info("✅ Significant improvement found - threshold was a major issue!")
        elif improvement > 0.005:
            logger.info("🔶 Moderate improvement - threshold helps but not the main issue")
        else:
            logger.info("❌ Little improvement - threshold not the main issue, look elsewhere")
        
        # Print top recommendations
        logger.info("🔍 Top Recommendations:")
        for i, rec in enumerate(analysis_results['recommendations'][:3], 1):
            logger.info(f"   {i}. {rec}")
    
    else:
        logger.error("❌ Threshold optimization failed!")
    
    return analysis_results


if __name__ == "__main__":
    print("🎯 ACTION THRESHOLD OPTIMIZER")
    print("=" * 60)
    print("🔍 Purpose: Analyze if poor RL performance is due to action thresholding")
    print("📊 Features:")
    print("   ✅ Comprehensive threshold optimization for mAP")
    print("   ✅ Action distribution and sparsity analysis")
    print("   ✅ Per-action performance breakdown")
    print("   ✅ Expert vs predicted action comparison")
    print("   ✅ Interactive visualizations")
    print("   ✅ Actionable recommendations")
    print()
    print("🚀 This will identify if the issue is action conversion or deeper problems!")
