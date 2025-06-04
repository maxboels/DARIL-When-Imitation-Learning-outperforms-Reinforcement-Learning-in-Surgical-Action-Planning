#!/usr/bin/env python3
"""
Enhanced Evaluation Framework for Three-Way Comparison
Evaluates IL, RL+WorldModel, and RL+OfflineVideos on unified metrics
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import Dict, List, Any, Tuple
from sklearn.metrics import average_precision_score, precision_recall_fscore_support
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class UnifiedEvaluationFramework:
    """
    Unified evaluation framework for all three methods using action prediction metrics
    """
    
    def __init__(self, results_dir: str, logger):
        self.results_dir = Path(results_dir)
        self.logger = logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create evaluation subdirectory
        self.eval_dir = self.results_dir / 'unified_evaluation'
        self.eval_dir.mkdir(exist_ok=True)
        
        # Results storage
        self.video_results = {}
        self.aggregate_results = {}
        self.statistical_tests = {}
        
        self.logger.info(f"🔬 Unified Evaluation Framework initialized")
        self.logger.info(f"📁 Results will be saved to: {self.eval_dir}")
    
    def load_all_models(self, experiment_results: Dict) -> Dict:
        """Load all trained models from experiment results"""
        
        models = {}
        
        # 1. Load IL model
        method1 = experiment_results.get('method_1_il_baseline', {})
        if method1.get('status') == 'success' and 'model_path' in method1:
            try:
                from models.dual_world_model import DualWorldModel
                il_model = DualWorldModel.load_model(method1['model_path'], self.device)
                models['IL_Baseline'] = il_model
                self.logger.info("✅ IL model loaded for evaluation")
            except Exception as e:
                self.logger.warning(f"⚠️ Could not load IL model: {e}")
        
        # 2. Load RL + World Model models
        method2 = experiment_results.get('method_2_rl_world_model', {})
        if method2.get('status') == 'success' and 'rl_models' in method2:
            for alg_name, alg_result in method2['rl_models'].items():
                if alg_result.get('status') == 'success' and 'model_path' in alg_result:
                    try:
                        if alg_name.lower() == 'ppo':
                            from stable_baselines3 import PPO
                            models[f'RL_WorldModel_PPO'] = PPO.load(alg_result['model_path'])
                        elif alg_name.lower() == 'a2c':
                            from stable_baselines3 import A2C
                            models[f'RL_WorldModel_A2C'] = A2C.load(alg_result['model_path'])
                        self.logger.info(f"✅ RL+WorldModel {alg_name.upper()} loaded")
                    except Exception as e:
                        self.logger.warning(f"⚠️ Could not load RL+WorldModel {alg_name}: {e}")
        
        # 3. Load RL + Offline Videos models
        method3 = experiment_results.get('method_3_rl_offline_videos', {})
        if method3.get('status') == 'success' and 'rl_models' in method3:
            for alg_name, alg_result in method3['rl_models'].items():
                if alg_result.get('status') == 'success' and 'model_path' in alg_result:
                    try:
                        if alg_name.lower() == 'ppo':
                            from stable_baselines3 import PPO
                            models[f'RL_OfflineVideos_PPO'] = PPO.load(alg_result['model_path'])
                        elif alg_name.lower() == 'a2c':
                            from stable_baselines3 import A2C
                            models[f'RL_OfflineVideos_A2C'] = A2C.load(alg_result['model_path'])
                        self.logger.info(f"✅ RL+OfflineVideos {alg_name.upper()} loaded")
                    except Exception as e:
                        self.logger.warning(f"⚠️ Could not load RL+OfflineVideos {alg_name}: {e}")
        
        self.logger.info(f"📊 Loaded {len(models)} models for unified evaluation")
        return models
    
    def predict_actions_il(self, il_model, video_embeddings: np.ndarray, horizon: int = 15) -> np.ndarray:
        """Predict actions using IL model over specified horizon"""
        
        predictions = []
        video_length = len(video_embeddings)
        
        for t in range(min(video_length - 1, horizon)):
            # Current state at timestep t
            current_state = torch.tensor(
                video_embeddings[t], dtype=torch.float32, device=self.device
            ).unsqueeze(0)  # [1, embedding_dim]
            
            with torch.no_grad():
                # Forward pass through IL model
                outputs = il_model(current_states=current_state.unsqueeze(1))
                
                if 'action_pred' in outputs:
                    action_probs = torch.sigmoid(outputs['action_pred'][:, -1, :])  # [1, 100]
                    action_pred = action_probs.cpu().numpy().flatten()
                else:
                    action_pred = np.zeros(100)
                
                predictions.append(action_pred)
        
        return np.array(predictions)  # [horizon, 100]
    
    def predict_actions_rl(self, rl_model, video_embeddings: np.ndarray, horizon: int = 15) -> np.ndarray:
        """Predict actions using RL model over specified horizon"""
        
        predictions = []
        video_length = len(video_embeddings)
        
        for t in range(min(video_length - 1, horizon)):
            # Current state at timestep t
            current_state = video_embeddings[t].reshape(1, -1)  # [1, embedding_dim]
            
            try:
                # Get action from RL policy
                action_pred, _ = rl_model.predict(current_state, deterministic=True)
                
                # Convert to binary action vector
                if isinstance(action_pred, np.ndarray):
                    action_pred = action_pred.flatten()
                
                # Handle different action formats
                if len(action_pred) == 100:
                    binary_action = (action_pred > 0.5).astype(float)
                elif len(action_pred) == 1:
                    # Single discrete action
                    binary_action = np.zeros(100)
                    action_idx = int(action_pred[0]) % 100
                    binary_action[action_idx] = 1.0
                else:
                    # Pad or truncate
                    binary_action = np.zeros(100)
                    if len(action_pred) > 0:
                        binary_action[:min(len(action_pred), 100)] = (action_pred[:100] > 0.5).astype(float)
                
                predictions.append(binary_action)
                
            except Exception as e:
                self.logger.warning(f"Error predicting with RL model: {e}")
                predictions.append(np.zeros(100))
        
        return np.array(predictions)  # [horizon, 100]
    
    def compute_trajectory_metrics(self, predictions: np.ndarray, ground_truth: np.ndarray) -> Dict[str, List[float]]:
        """Compute metrics over trajectory with cumulative evaluation"""
        
        metrics = {
            'cumulative_mAP': [],
            'cumulative_exact_match': [],
            'cumulative_hamming_accuracy': [],
            'cumulative_precision': [],
            'cumulative_recall': [],
            'cumulative_f1': []
        }
        
        for t in range(1, len(predictions) + 1):
            # Cumulative predictions and ground truth up to timestep t
            pred_cumulative = predictions[:t]  # [t, 100]
            gt_cumulative = ground_truth[:t]   # [t, 100]
            
            # Flatten for metric calculation
            pred_flat = pred_cumulative.reshape(-1, 100)
            gt_flat = gt_cumulative.reshape(-1, 100)
            binary_pred = (pred_flat > 0.5).astype(int)
            
            # 1. mAP calculation
            ap_scores = []
            for action_idx in range(100):
                if np.sum(gt_flat[:, action_idx]) > 0:
                    try:
                        ap = average_precision_score(gt_flat[:, action_idx], pred_flat[:, action_idx])
                        ap_scores.append(ap)
                    except:
                        ap_scores.append(0.0)
                else:
                    # No positive samples - perfect if no false positives
                    ap_scores.append(1.0 if np.sum(binary_pred[:, action_idx]) == 0 else 0.0)
            
            metrics['cumulative_mAP'].append(np.mean(ap_scores))
            
            # 2. Exact match accuracy
            exact_match = np.mean(np.all(binary_pred == gt_flat, axis=1))
            metrics['cumulative_exact_match'].append(exact_match)
            
            # 3. Hamming accuracy
            hamming_acc = np.mean(binary_pred == gt_flat)
            metrics['cumulative_hamming_accuracy'].append(hamming_acc)
            
            # 4. Precision, Recall, F1
            try:
                precision, recall, f1, _ = precision_recall_fscore_support(
                    gt_flat.flatten(), binary_pred.flatten(), average='binary', zero_division=0
                )
                metrics['cumulative_precision'].append(precision)
                metrics['cumulative_recall'].append(recall)
                metrics['cumulative_f1'].append(f1)
            except:
                metrics['cumulative_precision'].append(0.0)
                metrics['cumulative_recall'].append(0.0)
                metrics['cumulative_f1'].append(0.0)
        
        return metrics
    
    def evaluate_single_video(self, models: Dict, video: Dict, horizon: int = 15) -> Dict:
        """Evaluate all models on a single video"""
        
        video_id = video['video_id']
        video_embeddings = video['frame_embeddings'][:horizon+1]  # +1 for ground truth
        ground_truth_actions = video['actions_binaries'][:horizon+1]
        
        self.logger.info(f"📹 Evaluating {video_id} (horizon: {horizon})")
        
        # Ground truth for evaluation (we predict next actions)
        gt_for_evaluation = ground_truth_actions[1:horizon+1]  # [horizon, 100]
        
        video_result = {
            'video_id': video_id,
            'horizon': horizon,
            'predictions': {},
            'metrics': {},
            'summary': {}
        }
        
        # Evaluate each model
        for method_name, model in models.items():
            self.logger.info(f"  🤖 Evaluating {method_name}")
            
            try:
                if 'IL_Baseline' in method_name:
                    predictions = self.predict_actions_il(model, video_embeddings, horizon)
                else:
                    predictions = self.predict_actions_rl(model, video_embeddings, horizon)
                
                # Ensure predictions match ground truth length
                min_len = min(len(predictions), len(gt_for_evaluation))
                predictions = predictions[:min_len]
                gt_adjusted = gt_for_evaluation[:min_len]
                
                # Compute trajectory metrics
                trajectory_metrics = self.compute_trajectory_metrics(predictions, gt_adjusted)
                
                # Store results
                video_result['predictions'][method_name] = predictions.tolist()
                video_result['metrics'][method_name] = trajectory_metrics
                
                # Summary statistics
                video_result['summary'][method_name] = {
                    'final_mAP': trajectory_metrics['cumulative_mAP'][-1] if trajectory_metrics['cumulative_mAP'] else 0.0,
                    'mean_mAP': np.mean(trajectory_metrics['cumulative_mAP']) if trajectory_metrics['cumulative_mAP'] else 0.0,
                    'mAP_degradation': (trajectory_metrics['cumulative_mAP'][0] - trajectory_metrics['cumulative_mAP'][-1]) if len(trajectory_metrics['cumulative_mAP']) > 1 else 0.0,
                    'final_exact_match': trajectory_metrics['cumulative_exact_match'][-1] if trajectory_metrics['cumulative_exact_match'] else 0.0,
                    'mean_exact_match': np.mean(trajectory_metrics['cumulative_exact_match']) if trajectory_metrics['cumulative_exact_match'] else 0.0
                }
                
                self.logger.info(f"    📊 Final mAP: {video_result['summary'][method_name]['final_mAP']:.3f}")
                
            except Exception as e:
                self.logger.error(f"    ❌ Error evaluating {method_name}: {e}")
                import traceback
                traceback.print_exc()
        
        return video_result
    
    def run_unified_evaluation(self, models: Dict, test_data: List[Dict], horizon: int = 15) -> Dict:
        """Run unified evaluation on all test videos"""
        
        self.logger.info("🎯 Running Unified Evaluation Framework")
        self.logger.info(f"📊 Evaluating {len(models)} models on {len(test_data)} videos")
        self.logger.info(f"⏱️ Prediction horizon: {horizon} timesteps")
        self.logger.info("=" * 60)
        
        # Evaluate each video
        for video_idx, video in enumerate(test_data):
            self.logger.info(f"📹 Video {video_idx + 1}/{len(test_data)}")
            
            video_result = self.evaluate_single_video(models, video, horizon)
            self.video_results[video['video_id']] = video_result
        
        # Compute aggregate statistics
        self.aggregate_results = self.compute_aggregate_statistics()
        
        # Perform statistical tests
        self.statistical_tests = self.perform_statistical_tests()
        
        return {
            'video_results': self.video_results,
            'aggregate_results': self.aggregate_results,
            'statistical_tests': self.statistical_tests,
            'evaluation_config': {
                'horizon': horizon,
                'num_videos': len(test_data),
                'num_models': len(models),
                'models_evaluated': list(models.keys())
            }
        }
    
    def compute_aggregate_statistics(self) -> Dict:
        """Compute aggregate statistics across all videos"""
        
        methods = set()
        for video_result in self.video_results.values():
            methods.update(video_result['summary'].keys())
        
        aggregate_stats = {}
        
        for method in methods:
            # Collect metrics across all videos
            final_maps = []
            mean_maps = []
            map_degradations = []
            final_exact_matches = []
            mean_exact_matches = []
            
            # Collect all trajectory points for detailed analysis
            all_map_trajectories = []
            
            for video_result in self.video_results.values():
                if method in video_result['summary']:
                    summary = video_result['summary'][method]
                    final_maps.append(summary['final_mAP'])
                    mean_maps.append(summary['mean_mAP'])
                    map_degradations.append(summary['mAP_degradation'])
                    final_exact_matches.append(summary['final_exact_match'])
                    mean_exact_matches.append(summary['mean_exact_match'])
                    
                    # Collect trajectory
                    if method in video_result['metrics']:
                        trajectory = video_result['metrics'][method]['cumulative_mAP']
                        all_map_trajectories.append(trajectory)
            
            if final_maps:
                aggregate_stats[method] = {
                    'final_mAP': {
                        'mean': np.mean(final_maps),
                        'std': np.std(final_maps),
                        'min': np.min(final_maps),
                        'max': np.max(final_maps)
                    },
                    'mean_mAP': {
                        'mean': np.mean(mean_maps),
                        'std': np.std(mean_maps),
                        'min': np.min(mean_maps),
                        'max': np.max(mean_maps)
                    },
                    'mAP_degradation': {
                        'mean': np.mean(map_degradations),
                        'std': np.std(map_degradations),
                        'min': np.min(map_degradations),
                        'max': np.max(map_degradations)
                    },
                    'exact_match': {
                        'final_mean': np.mean(final_exact_matches),
                        'overall_mean': np.mean(mean_exact_matches)
                    },
                    'trajectory_stability': -np.mean(map_degradations),  # Higher = more stable
                    'num_videos': len(final_maps)
                }
        
        return aggregate_stats
    
    def perform_statistical_tests(self) -> Dict:
        """Perform statistical significance tests between methods"""
        
        methods = list(self.aggregate_results.keys())
        statistical_tests = {}
        
        # Collect final mAP values for each method
        method_final_maps = {}
        for method in methods:
            final_maps = []
            for video_result in self.video_results.values():
                if method in video_result['summary']:
                    final_maps.append(video_result['summary'][method]['final_mAP'])
            method_final_maps[method] = final_maps
        
        # Pairwise comparisons
        for i, method1 in enumerate(methods):
            for method2 in methods[i+1:]:
                
                maps1 = method_final_maps[method1]
                maps2 = method_final_maps[method2]
                
                if len(maps1) > 1 and len(maps2) > 1:
                    # Perform t-test
                    t_stat, p_value = stats.ttest_ind(maps1, maps2)
                    
                    # Effect size (Cohen's d)
                    pooled_std = np.sqrt(((len(maps1) - 1) * np.var(maps1) + 
                                        (len(maps2) - 1) * np.var(maps2)) / 
                                       (len(maps1) + len(maps2) - 2))
                    cohens_d = (np.mean(maps1) - np.mean(maps2)) / pooled_std if pooled_std > 0 else 0
                    
                    statistical_tests[f"{method1}_vs_{method2}"] = {
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'cohens_d': cohens_d,
                        'significant': p_value < 0.05,
                        'mean_diff': np.mean(maps1) - np.mean(maps2),
                        'method1_mean': np.mean(maps1),
                        'method2_mean': np.mean(maps2),
                        'effect_size_interpretation': self._interpret_effect_size(abs(cohens_d))
                    }
        
        return statistical_tests
    
    def _interpret_effect_size(self, cohens_d: float) -> str:
        """Interpret Cohen's d effect size"""
        if cohens_d < 0.2:
            return "negligible"
        elif cohens_d < 0.5:
            return "small"
        elif cohens_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def save_results_to_files(self):
        """Save all results to CSV and JSON files"""
        
        self.logger.info("💾 Saving evaluation results to files...")
        
        # 1. Save video-level results to CSV
        video_rows = []
        for video_id, video_result in self.video_results.items():
            for method, summary in video_result['summary'].items():
                row = {
                    'video_id': video_id,
                    'method': method,
                    'horizon': video_result['horizon'],
                    **summary
                }
                video_rows.append(row)
        
        video_df = pd.DataFrame(video_rows)
        video_csv_path = self.eval_dir / 'video_level_results.csv'
        video_df.to_csv(video_csv_path, index=False)
        self.logger.info(f"  📊 Video results saved to: {video_csv_path}")
        
        # 2. Save aggregate statistics to CSV
        agg_rows = []
        for method, stats in self.aggregate_results.items():
            row = {'method': method}
            for metric_name, metric_data in stats.items():
                if isinstance(metric_data, dict):
                    for sub_name, value in metric_data.items():
                        row[f"{metric_name}_{sub_name}"] = value
                else:
                    row[metric_name] = metric_data
            agg_rows.append(row)
        
        agg_df = pd.DataFrame(agg_rows)
        agg_csv_path = self.eval_dir / 'aggregate_statistics.csv'
        agg_df.to_csv(agg_csv_path, index=False)
        self.logger.info(f"  📊 Aggregate stats saved to: {agg_csv_path}")
        
        # 3. Save trajectory data for plotting
        trajectory_rows = []
        for video_id, video_result in self.video_results.values():
            for method, metrics in video_result['metrics'].items():
                for timestep, map_value in enumerate(metrics['cumulative_mAP']):
                    trajectory_rows.append({
                        'video_id': video_id,
                        'method': method,
                        'timestep': timestep + 1,
                        'cumulative_mAP': map_value,
                        'cumulative_exact_match': metrics['cumulative_exact_match'][timestep],
                        'cumulative_hamming_accuracy': metrics['cumulative_hamming_accuracy'][timestep]
                    })
        
        trajectory_df = pd.DataFrame(trajectory_rows)
        trajectory_csv_path = self.eval_dir / 'trajectory_data.csv'
        trajectory_df.to_csv(trajectory_csv_path, index=False)
        self.logger.info(f"  📊 Trajectory data saved to: {trajectory_csv_path}")
        
        # 4. Save complete results to JSON
        complete_results = {
            'video_results': self.video_results,
            'aggregate_results': self.aggregate_results,
            'statistical_tests': self.statistical_tests
        }
        
        json_path = self.eval_dir / 'complete_evaluation_results.json'
        with open(json_path, 'w') as f:
            json.dump(complete_results, f, indent=2, default=str)
        self.logger.info(f"  📊 Complete results saved to: {json_path}")
        
        return {
            'video_csv': video_csv_path,
            'aggregate_csv': agg_csv_path,
            'trajectory_csv': trajectory_csv_path,
            'complete_json': json_path
        }
    
    def create_comprehensive_visualizations(self):
        """Create comprehensive visualizations"""
        
        self.logger.info("🎨 Creating comprehensive visualizations...")
        
        # Set publication style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create comprehensive figure
        fig = plt.figure(figsize=(20, 16))
        
        # Create grid layout
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # 1. Overall Performance Comparison (top-left)
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_overall_performance(ax1)
        
        # 2. Trajectory Degradation (top-right)
        ax2 = fig.add_subplot(gs[0, 2:])
        self._plot_trajectory_degradation(ax2)
        
        # 3. Method Stability (middle-left)
        ax3 = fig.add_subplot(gs[1, :2])
        self._plot_method_stability(ax3)
        
        # 4. Statistical Significance Heatmap (middle-right)
        ax4 = fig.add_subplot(gs[1, 2:])
        self._plot_statistical_significance(ax4)
        
        # 5. Per-Video Performance (bottom-left)
        ax5 = fig.add_subplot(gs[2, :2])
        self._plot_per_video_performance(ax5)
        
        # 6. Horizon Analysis (bottom-right)
        ax6 = fig.add_subplot(gs[2, 2:])
        self._plot_horizon_analysis(ax6)
        
        # 7. Prediction vs Ground Truth Examples (bottom full width)
        ax7 = fig.add_subplot(gs[3, :])
        self._plot_prediction_examples(ax7)
        
        plt.suptitle('Comprehensive Evaluation: IL vs RL+WorldModel vs RL+OfflineVideos', 
                     fontsize=16, fontweight='bold')
        
        # Save comprehensive figure
        fig_path = self.eval_dir / 'comprehensive_evaluation_results'
        plt.savefig(f'{fig_path}.pdf', bbox_inches='tight', dpi=300)
        plt.savefig(f'{fig_path}.png', bbox_inches='tight', dpi=300)
        plt.close()
        
        # Create individual focused plots
        self._create_individual_plots()
        
        self.logger.info(f"  🎨 Visualizations saved to: {self.eval_dir}")
    
    def _plot_overall_performance(self, ax):
        """Plot overall performance comparison"""
        
        methods = list(self.aggregate_results.keys())
        method_names = [self._format_method_name(m) for m in methods]
        mean_maps = [self.aggregate_results[m]['final_mAP']['mean'] for m in methods]
        std_maps = [self.aggregate_results[m]['final_mAP']['std'] for m in methods]
        
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#4CAF50', '#FF9800'][:len(methods)]
        
        bars = ax.bar(method_names, mean_maps, yerr=std_maps, capsize=5, 
                     color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        
        ax.set_title('(a) Overall Performance Comparison', fontweight='bold')
        ax.set_ylabel('Final mAP')
        ax.set_ylim(0, max(mean_maps) * 1.2)
        
        # Add value labels
        for bar, mean_map in zip(bars, mean_maps):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{mean_map:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax.grid(axis='y', alpha=0.3)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    def _plot_trajectory_degradation(self, ax):
        """Plot trajectory degradation over time"""
        
        methods = list(self.aggregate_results.keys())
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#4CAF50', '#FF9800'][:len(methods)]
        
        for i, method in enumerate(methods):
            # Collect all trajectories for this method
            all_trajectories = []
            for video_result in self.video_results.values():
                if method in video_result['metrics']:
                    trajectory = video_result['metrics'][method]['cumulative_mAP']
                    if trajectory:
                        all_trajectories.append(trajectory)
            
            if all_trajectories:
                # Compute average trajectory
                min_length = min(len(traj) for traj in all_trajectories)
                truncated_trajectories = [traj[:min_length] for traj in all_trajectories]
                mean_trajectory = np.mean(truncated_trajectories, axis=0)
                std_trajectory = np.std(truncated_trajectories, axis=0)
                
                timesteps = np.arange(1, len(mean_trajectory) + 1)
                
                ax.plot(timesteps, mean_trajectory, 
                       label=self._format_method_name(method), 
                       color=colors[i], linewidth=2)
                ax.fill_between(timesteps,
                               mean_trajectory - std_trajectory,
                               mean_trajectory + std_trajectory,
                               alpha=0.2, color=colors[i])
        
        ax.set_title('(b) mAP Degradation Over Prediction Horizon', fontweight='bold')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Cumulative mAP')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_method_stability(self, ax):
        """Plot method stability (lower degradation = more stable)"""
        
        methods = list(self.aggregate_results.keys())
        method_names = [self._format_method_name(m) for m in methods]
        degradations = [self.aggregate_results[m]['mAP_degradation']['mean'] for m in methods]
        
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#4CAF50', '#FF9800'][:len(methods)]
        
        bars = ax.bar(method_names, degradations, color=colors, alpha=0.8)
        ax.set_title('(c) Method Stability (Lower = More Stable)', fontweight='bold')
        ax.set_ylabel('mAP Degradation')
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        
        # Add value labels
        for bar, degradation in zip(bars, degradations):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                   f'{degradation:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax.grid(axis='y', alpha=0.3)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    def _plot_statistical_significance(self, ax):
        """Plot statistical significance heatmap"""
        
        methods = list(self.aggregate_results.keys())
        n_methods = len(methods)
        
        # Create p-value matrix
        p_matrix = np.ones((n_methods, n_methods))
        
        for comparison, results in self.statistical_tests.items():
            method1, method2 = comparison.split('_vs_')
            try:
                idx1 = methods.index(method1)
                idx2 = methods.index(method2)
                p_matrix[idx1, idx2] = results['p_value']
                p_matrix[idx2, idx1] = results['p_value']
            except ValueError:
                continue
        
        im = ax.imshow(p_matrix, cmap='RdYlBu_r', vmin=0, vmax=0.1)
        
        method_names = [self._format_method_name(m) for m in methods]
        ax.set_xticks(range(n_methods))
        ax.set_yticks(range(n_methods))
        ax.set_xticklabels(method_names, rotation=45, ha='right')
        ax.set_yticklabels(method_names)
        ax.set_title('(d) Statistical Significance (p-values)', fontweight='bold')
        
        # Add p-values as text
        for i in range(n_methods):
            for j in range(n_methods):
                if i != j:
                    color = 'white' if p_matrix[i, j] < 0.05 else 'black'
                    ax.text(j, i, f'{p_matrix[i, j]:.3f}', 
                           ha='center', va='center', color=color, fontweight='bold')
        
        plt.colorbar(im, ax=ax, label='p-value')
    
    def _plot_per_video_performance(self, ax):
        """Plot per-video performance comparison"""
        
        video_ids = list(self.video_results.keys())
        methods = list(self.aggregate_results.keys())
        
        x = np.arange(len(video_ids))
        width = 0.15
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#4CAF50', '#FF9800'][:len(methods)]
        
        for i, method in enumerate(methods):
            performances = []
            for video_id in video_ids:
                if method in self.video_results[video_id]['summary']:
                    performances.append(self.video_results[video_id]['summary'][method]['final_mAP'])
                else:
                    performances.append(0)
            
            ax.bar(x + i * width, performances, width, 
                  label=self._format_method_name(method), 
                  color=colors[i], alpha=0.8)
        
        ax.set_title('(e) Per-Video Performance Comparison', fontweight='bold')
        ax.set_xlabel('Video ID')
        ax.set_ylabel('Final mAP')
        ax.set_xticks(x + width * (len(methods) - 1) / 2)
        ax.set_xticklabels([vid.replace('VID', 'V') for vid in video_ids])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    
    def _plot_horizon_analysis(self, ax):
        """Plot performance across prediction horizon"""
        
        methods = list(self.aggregate_results.keys())
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#4CAF50', '#FF9800'][:len(methods)]
        
        # For each method, plot the mean trajectory across all videos
        for i, method in enumerate(methods):
            all_trajectories = []
            for video_result in self.video_results.values():
                if method in video_result['metrics']:
                    trajectory = video_result['metrics'][method]['cumulative_mAP']
                    if trajectory:
                        all_trajectories.append(trajectory)
            
            if all_trajectories:
                min_length = min(len(traj) for traj in all_trajectories)
                truncated_trajectories = [traj[:min_length] for traj in all_trajectories]
                mean_trajectory = np.mean(truncated_trajectories, axis=0)
                
                timesteps = np.arange(1, len(mean_trajectory) + 1)
                ax.plot(timesteps, mean_trajectory, 
                       label=self._format_method_name(method), 
                       color=colors[i], linewidth=2, marker='o', markersize=4)
        
        ax.set_title('(f) Performance Across Prediction Horizon', fontweight='bold')
        ax.set_xlabel('Prediction Timestep')
        ax.set_ylabel('Mean mAP')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_prediction_examples(self, ax):
        """Plot prediction vs ground truth examples"""
        
        # Get first video and first method for example
        first_video = next(iter(self.video_results.values()))
        first_method = next(iter(first_video['predictions'].keys()))
        
        predictions = np.array(first_video['predictions'][first_method])
        
        # Show first 10 timesteps and first 20 action classes for visibility
        pred_subset = predictions[:10, :20]
        
        im = ax.imshow(pred_subset.T, cmap='viridis', aspect='auto')
        ax.set_title(f'(g) Example Predictions: {self._format_method_name(first_method)} on {first_video["video_id"]}', 
                    fontweight='bold')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Action Class')
        ax.set_yticks(range(0, 20, 5))
        ax.set_yticklabels([f'A{i}' for i in range(0, 20, 5)])
        
        plt.colorbar(im, ax=ax, label='Prediction Probability')
    
    def _create_individual_plots(self):
        """Create individual focused plots"""
        
        # Method comparison plot
        fig, ax = plt.subplots(figsize=(10, 6))
        self._plot_overall_performance(ax)
        plt.tight_layout()
        plt.savefig(self.eval_dir / 'method_comparison.pdf', bbox_inches='tight', dpi=300)
        plt.close()
        
        # Trajectory analysis plot
        fig, ax = plt.subplots(figsize=(12, 6))
        self._plot_trajectory_degradation(ax)
        plt.tight_layout()
        plt.savefig(self.eval_dir / 'trajectory_analysis.pdf', bbox_inches='tight', dpi=300)
        plt.close()
    
    def _format_method_name(self, method_name: str) -> str:
        """Format method names for display"""
        return method_name.replace('_', ' ').replace('RL WorldModel', 'RL+WM').replace('RL OfflineVideos', 'RL+OV')
    
    def generate_latex_tables(self) -> str:
        """Generate comprehensive LaTeX tables"""
        
        self.logger.info("📝 Generating LaTeX tables...")
        
        latex_content = []
        
        # Main results table
        latex_content.append(r"""
\begin{table*}[htbp]
\centering
\caption{Comprehensive Comparison: Three-Way Evaluation of Surgical Action Prediction Methods}
\label{tab:main_results}
\begin{tabular}{lccccccc}
\toprule
Method & Final mAP & Mean mAP & mAP Degradation & Stability & Exact Match & Videos & Significance \\
\midrule
""")
        
        # Sort methods by final mAP
        methods_sorted = sorted(self.aggregate_results.items(), 
                              key=lambda x: x[1]['final_mAP']['mean'], reverse=True)
        
        for method, stats in methods_sorted:
            method_name = self._format_method_name(method).replace(' ', ' ')
            
            # Check if method is significantly better than others
            significance = ""
            for comparison, test_result in self.statistical_tests.items():
                if method in comparison and test_result['significant']:
                    if test_result['mean_diff'] > 0 and method == comparison.split('_vs_')[0]:
                        significance = "*"
                        break
                    elif test_result['mean_diff'] < 0 and method == comparison.split('_vs_')[1]:
                        significance = "*"
                        break
            
            latex_content.append(
                f"{method_name} & "
                f"{stats['final_mAP']['mean']:.3f} ± {stats['final_mAP']['std']:.3f} & "
                f"{stats['mean_mAP']['mean']:.3f} & "
                f"{stats['mAP_degradation']['mean']:.3f} & "
                f"{stats['trajectory_stability']:.3f} & "
                f"{stats['exact_match']['final_mean']:.3f} & "
                f"{stats['num_videos']} & "
                f"{significance} \\\\"
            )
        
        latex_content.append(r"""
\bottomrule
\multicolumn{8}{l}{\footnotesize * Statistically significant (p < 0.05) compared to at least one other method} \\
\multicolumn{8}{l}{\footnotesize Stability = -mAP Degradation (higher is better)} \\
\end{tabular}
\end{table*}
""")
        
        # Statistical tests table
        latex_content.append(r"""
\begin{table}[htbp]
\centering
\caption{Statistical Significance Tests: Pairwise Method Comparisons}
\label{tab:statistical_tests}
\begin{tabular}{lcccc}
\toprule
Comparison & Mean Difference & t-statistic & p-value & Effect Size \\
\midrule
""")
        
        for comparison, results in self.statistical_tests.items():
            method1, method2 = comparison.split('_vs_')
            method1_name = self._format_method_name(method1)
            method2_name = self._format_method_name(method2)
            comparison_name = f"{method1_name} vs {method2_name}"
            
            significance = "***" if results['p_value'] < 0.001 else "**" if results['p_value'] < 0.01 else "*" if results['p_value'] < 0.05 else ""
            
            latex_content.append(
                f"{comparison_name} & "
                f"{results['mean_diff']:.3f} & "
                f"{results['t_statistic']:.3f} & "
                f"{results['p_value']:.3f}{significance} & "
                f"{results['cohens_d']:.3f} ({results['effect_size_interpretation']}) \\\\"
            )
        
        latex_content.append(r"""
\bottomrule
\multicolumn{5}{l}{\footnotesize *** p < 0.001, ** p < 0.01, * p < 0.05} \\
\end{tabular}
\end{table}
""")
        
        # Combine all tables
        full_latex = '\n'.join(latex_content)
        
        # Save LaTeX tables
        latex_path = self.eval_dir / 'evaluation_tables.tex'
        with open(latex_path, 'w') as f:
            f.write(full_latex)
        
        self.logger.info(f"  📝 LaTeX tables saved to: {latex_path}")
        
        return full_latex
    
    def print_summary(self):
        """Print evaluation summary"""
        
        print("\n" + "="*70)
        print("🎉 UNIFIED EVALUATION COMPLETED!")
        print("="*70)
        
        print(f"\n📁 Results saved to: {self.eval_dir}/")
        
        print("\n📊 Key Performance Results:")
        methods_sorted = sorted(self.aggregate_results.items(), 
                              key=lambda x: x[1]['final_mAP']['mean'], reverse=True)
        
        for rank, (method, stats) in enumerate(methods_sorted, 1):
            method_name = self._format_method_name(method)
            print(f"  {rank}. {method_name}: {stats['final_mAP']['mean']:.3f} ± {stats['final_mAP']['std']:.3f} mAP")
        
        print("\n🔬 Statistical Significance:")
        significant_count = sum(1 for test in self.statistical_tests.values() if test['significant'])
        total_tests = len(self.statistical_tests)
        print(f"  {significant_count}/{total_tests} comparisons statistically significant (p < 0.05)")
        
        print("\n📄 Generated Files:")
        print("  - video_level_results.csv (Per-video metrics)")
        print("  - aggregate_statistics.csv (Summary statistics)")
        print("  - trajectory_data.csv (Trajectory analysis)")
        print("  - complete_evaluation_results.json (Full results)")
        print("  - comprehensive_evaluation_results.pdf (Main figure)")
        print("  - evaluation_tables.tex (LaTeX tables)")
        
        # Print key insights
        best_method = methods_sorted[0]
        worst_method = methods_sorted[-1]
        
        print(f"\n💡 Key Insights:")
        print(f"  • Best method: {self._format_method_name(best_method[0])} "
              f"({best_method[1]['final_mAP']['mean']:.3f} mAP)")
        print(f"  • Performance range: {worst_method[1]['final_mAP']['mean']:.3f} - "
              f"{best_method[1]['final_mAP']['mean']:.3f} mAP")
        
        # Compare method categories
        il_performance = None
        rl_wm_performances = []
        rl_ov_performances = []
        
        for method, stats in self.aggregate_results.items():
            if 'IL_Baseline' in method:
                il_performance = stats['final_mAP']['mean']
            elif 'RL_WorldModel' in method:
                rl_wm_performances.append(stats['final_mAP']['mean'])
            elif 'RL_OfflineVideos' in method:
                rl_ov_performances.append(stats['final_mAP']['mean'])
        
        if il_performance and rl_wm_performances:
            best_rl_wm = max(rl_wm_performances)
            print(f"  • RL+WorldModel vs IL: {best_rl_wm - il_performance:+.3f} mAP difference")
        
        if il_performance and rl_ov_performances:
            best_rl_ov = max(rl_ov_performances)
            print(f"  • RL+OfflineVideos vs IL: {best_rl_ov - il_performance:+.3f} mAP difference")
        
        print("\n✅ Your comprehensive evaluation is ready for publication!")


# Integration function
def run_enhanced_evaluation(experiment_results: Dict, test_data: List[Dict], 
                          results_dir: str, logger, horizon: int = 15):
    """
    Run enhanced evaluation on all three methods
    
    Args:
        experiment_results: Results from run_experiment_v2.py
        test_data: Test dataset
        results_dir: Directory to save results
        logger: Logger instance
        horizon: Prediction horizon for evaluation
    """
    
    # Initialize evaluation framework
    evaluator = UnifiedEvaluationFramework(results_dir, logger)
    
    # Load all models
    models = evaluator.load_all_models(experiment_results)
    
    if not models:
        logger.error("❌ No models available for evaluation")
        return None
    
    # Run unified evaluation
    results = evaluator.run_unified_evaluation(models, test_data, horizon)
    
    # Save results to files
    file_paths = evaluator.save_results_to_files()
    
    # Create visualizations
    evaluator.create_comprehensive_visualizations()
    
    # Generate LaTeX tables
    latex_tables = evaluator.generate_latex_tables()
    
    # Print summary
    evaluator.print_summary()
    
    return {
        'evaluator': evaluator,
        'results': results,
        'file_paths': file_paths,
        'latex_tables': latex_tables
    }
