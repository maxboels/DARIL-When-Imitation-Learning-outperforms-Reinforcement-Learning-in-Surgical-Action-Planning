#!/usr/bin/env python3
"""
Enhanced RL Debugging and Visualization System
Focus: Understanding why RL can't reach 10% mAP vs supervised learning baseline

Key debugging areas:
1. World model generalization quality
2. Expert action matching during RL training  
3. Reward signal quality and alignment with mAP
4. Action space handling and thresholding
5. Training dynamics and convergence patterns
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from collections import defaultdict, deque
from typing import Dict, List, Any, Tuple
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve
from stable_baselines3.common.callbacks import BaseCallback
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


class RLDebugger:
    """
    Comprehensive RL debugging system focused on expert action matching
    and understanding the gap between RL and supervised learning performance.
    """
    
    def __init__(self, save_dir: str, logger, config: dict):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger
        self.config = config
        
        # Debugging data storage
        self.training_metrics = defaultdict(list)
        self.world_model_metrics = defaultdict(list)
        self.action_analysis = defaultdict(list)
        self.reward_analysis = defaultdict(list)
        self.expert_matching_analysis = defaultdict(list)
        
        # Current episode tracking
        self.current_episode_data = {}
        self.episode_counter = 0
        
        self.logger.info(f"🔍 RL Debugger initialized")
        self.logger.info(f"📁 Debug data will be saved to: {self.save_dir}")
        
    def evaluate_world_model_quality(self, world_model, test_data: List[Dict]) -> Dict[str, Any]:
        """
        Evaluate world model prediction quality to understand if it's generalizing well.
        This helps determine if poor RL performance is due to world model issues.
        """
        
        self.logger.info("🌍 Evaluating World Model Quality...")
        
        world_model.eval()
        
        results = {
            'state_prediction_accuracy': [],
            'reward_prediction_accuracy': [],
            'action_conditioning_quality': [],
            'temporal_consistency': [],
            'per_video_analysis': {}
        }
        
        with torch.no_grad():
            for video_idx, video in enumerate(test_data[:3]):  # Test on first 3 videos
                video_id = video['video_id']
                video_results = self._evaluate_world_model_on_video(world_model, video)
                results['per_video_analysis'][video_id] = video_results
                
                # Aggregate metrics
                results['state_prediction_accuracy'].append(video_results['state_mse'])
                results['reward_prediction_accuracy'].append(video_results['reward_accuracy'])
                results['action_conditioning_quality'].append(video_results['action_conditioning'])
                results['temporal_consistency'].append(video_results['temporal_consistency'])
        
        # Summary statistics
        summary = {
            'avg_state_mse': np.mean(results['state_prediction_accuracy']),
            'avg_reward_accuracy': np.mean(results['reward_prediction_accuracy']),
            'avg_action_conditioning': np.mean(results['action_conditioning_quality']),
            'avg_temporal_consistency': np.mean(results['temporal_consistency']),
            'world_model_quality_score': 0.0
        }
        
        # Overall quality score (0-1, higher is better)
        state_score = max(0, 1.0 - summary['avg_state_mse'] / 10.0)  # Normalize MSE
        reward_score = summary['avg_reward_accuracy']
        conditioning_score = summary['avg_action_conditioning']
        
        summary['world_model_quality_score'] = np.mean([state_score, reward_score, conditioning_score])
        
        results['summary'] = summary
        
        self.logger.info(f"🌍 World Model Quality Score: {summary['world_model_quality_score']:.3f}")
        self.logger.info(f"   State MSE: {summary['avg_state_mse']:.4f}")
        self.logger.info(f"   Reward Accuracy: {summary['avg_reward_accuracy']:.3f}")
        self.logger.info(f"   Action Conditioning: {summary['avg_action_conditioning']:.3f}")
        
        # Save detailed results
        with open(self.save_dir / 'world_model_quality_analysis.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        return results
    
    def _evaluate_world_model_on_video(self, world_model, video: Dict) -> Dict[str, float]:
        """Evaluate world model on a single video."""
        
        frames = video['frame_embeddings']
        actions = video['actions_binaries']
        
        if len(frames) < 10:
            return {'state_mse': 0.0, 'reward_accuracy': 0.0, 'action_conditioning': 0.0, 'temporal_consistency': 0.0}
        
        state_errors = []
        reward_accuracies = []
        action_conditioning_scores = []
        
        # Test on middle frames to avoid boundary effects
        test_indices = range(5, min(len(frames) - 5, 50))
        
        for i in test_indices:
            try:
                current_state = torch.tensor(frames[i], device=world_model.device, dtype=torch.float32)
                action = torch.tensor(actions[i], device=world_model.device, dtype=torch.float32)
                true_next_state = torch.tensor(frames[i + 1], device=world_model.device, dtype=torch.float32)
                
                # Predict next state using world model
                pred_next_state, pred_rewards, _ = world_model.simulate_step(current_state, action)
                
                # State prediction error
                state_mse = torch.mean((pred_next_state - true_next_state) ** 2).item()
                state_errors.append(state_mse)
                
                # Action conditioning quality (how much does action affect prediction)
                random_action = torch.rand_like(action)
                pred_with_random, _, _ = world_model.simulate_step(current_state, random_action)
                
                action_effect = torch.mean((pred_next_state - pred_with_random) ** 2).item()
                action_conditioning_scores.append(action_effect)
                
                # Reward prediction (if available)
                if pred_rewards:
                    reward_accuracies.append(1.0)  # Placeholder
                else:
                    reward_accuracies.append(0.0)
                    
            except Exception as e:
                self.logger.warning(f"World model evaluation failed at frame {i}: {e}")
                continue
        
        # Temporal consistency (how smooth are predictions)
        temporal_consistency = 1.0 / (1.0 + np.std(state_errors)) if state_errors else 0.0
        
        return {
            'state_mse': np.mean(state_errors) if state_errors else float('inf'),
            'reward_accuracy': np.mean(reward_accuracies) if reward_accuracies else 0.0,
            'action_conditioning': np.mean(action_conditioning_scores) if action_conditioning_scores else 0.0,
            'temporal_consistency': temporal_consistency,
            'num_test_frames': len(state_errors)
        }
    
    def create_expert_matching_callback(self, eval_env, expert_data: List[Dict]) -> 'ExpertMatchingCallback':
        """Create a callback that tracks expert action matching during RL training."""
        
        return ExpertMatchingCallback(
            eval_env=eval_env,
            expert_data=expert_data,
            debugger=self,
            eval_freq=500  # Evaluate every 500 steps
        )
    
    def analyze_action_space_handling(self, rl_model, test_data: List[Dict]) -> Dict[str, Any]:
        """
        Analyze how well the RL model handles the sparse binary action space.
        This helps understand if action conversion/thresholding is the issue.
        """
        
        self.logger.info("🎯 Analyzing Action Space Handling...")
        
        analysis = {
            'action_distribution': [],
            'threshold_analysis': {},
            'sparsity_handling': {},
            'expert_vs_predicted': {}
        }
        
        all_predictions = []
        all_expert_actions = []
        
        # Collect predictions and expert actions
        for video in test_data[:2]:  # Test on first 2 videos
            frames = video['frame_embeddings']
            expert_actions = video['actions_binaries']
            
            for i in range(min(50, len(frames))):  # Sample 50 frames per video
                try:
                    state = frames[i].reshape(1, -1)
                    action_pred, _ = rl_model.predict(state, deterministic=True)
                    
                    # Convert to proper format
                    if isinstance(action_pred, np.ndarray):
                        action_pred = action_pred.flatten()
                    
                    # Ensure 100 dimensions
                    if len(action_pred) != 100:
                        padded = np.zeros(100)
                        if len(action_pred) > 0:
                            padded[:min(len(action_pred), 100)] = action_pred[:100]
                        action_pred = padded
                    
                    all_predictions.append(action_pred)
                    all_expert_actions.append(expert_actions[i])
                    
                except Exception as e:
                    self.logger.warning(f"Action prediction failed: {e}")
                    continue
        
        if not all_predictions:
            self.logger.error("No predictions collected for action analysis")
            return analysis
        
        all_predictions = np.array(all_predictions)
        all_expert_actions = np.array(all_expert_actions)
        
        # Analyze action distribution
        analysis['action_distribution'] = {
            'pred_mean': np.mean(all_predictions),
            'pred_std': np.std(all_predictions),
            'pred_min': np.min(all_predictions),
            'pred_max': np.max(all_predictions),
            'expert_sparsity': np.mean(np.sum(all_expert_actions, axis=1)),
            'pred_sparsity_at_0.5': np.mean(np.sum(all_predictions > 0.5, axis=1))
        }
        
        # Threshold analysis - find optimal threshold for mAP
        thresholds = np.arange(0.1, 0.9, 0.1)
        threshold_results = {}
        
        for threshold in thresholds:
            binary_preds = (all_predictions > threshold).astype(int)
            
            # Calculate mAP for this threshold
            ap_scores = []
            for action_idx in range(100):
                if np.sum(all_expert_actions[:, action_idx]) > 0:
                    try:
                        ap = average_precision_score(
                            all_expert_actions[:, action_idx], 
                            all_predictions[:, action_idx]
                        )
                        ap_scores.append(ap)
                    except:
                        ap_scores.append(0.0)
            
            map_score = np.mean(ap_scores) if ap_scores else 0.0
            action_density = np.mean(np.sum(binary_preds, axis=1))
            
            threshold_results[threshold] = {
                'mAP': map_score,
                'action_density': action_density,
                'num_present_actions': len(ap_scores)
            }
        
        analysis['threshold_analysis'] = threshold_results
        
        # Find best threshold
        best_threshold = max(threshold_results.keys(), 
                           key=lambda t: threshold_results[t]['mAP'])
        analysis['optimal_threshold'] = best_threshold
        analysis['optimal_mAP'] = threshold_results[best_threshold]['mAP']
        
        self.logger.info(f"🎯 Action Space Analysis Complete")
        self.logger.info(f"   Optimal threshold: {best_threshold}")
        self.logger.info(f"   Optimal mAP: {analysis['optimal_mAP']:.4f}")
        self.logger.info(f"   Expert action density: {analysis['action_distribution']['expert_sparsity']:.1f}")
        self.logger.info(f"   Predicted density (0.5): {analysis['action_distribution']['pred_sparsity_at_0.5']:.1f}")
        
        # Save analysis
        with open(self.save_dir / 'action_space_analysis.json', 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        return analysis
    
    def create_simplified_reward_environment(self, base_env_class):
        """
        Create a simplified environment that focuses only on expert action matching
        and mAP-aligned rewards. This removes complexity to isolate the core issue.
        """
        
        class SimplifiedExpertMatchingEnv(base_env_class):
            """Simplified environment focused only on expert action matching."""
            
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                
                # Override reward weights to focus only on expert matching
                self.reward_weights = {
                    'expert_matching': 50.0,     # ONLY expert matching (very high weight)
                    'action_sparsity': 5.0,      # Encourage appropriate sparsity
                    'completion_bonus': 2.0      # Small completion bonus
                }
                
                # Debug tracking
                self.debug_rewards = []
                self.debug_expert_matches = []
                
            def _calculate_meaningful_reward(self, action: np.ndarray, predicted_rewards: Dict[str, float]) -> float:
                """SIMPLIFIED: Focus only on expert action matching and sparsity."""
                
                reward = 0.0
                expert_match_score = 0.0
                
                # 1. EXPERT ACTION MATCHING (primary reward)
                if (self.expert_actions_sequence is not None and 
                    self.current_frame_idx < len(self.expert_actions_sequence)):
                    
                    expert_actions = self.expert_actions_sequence[self.current_frame_idx]
                    binary_action = (action > 0.5).astype(int)
                    
                    if len(expert_actions) == len(binary_action):
                        # Focus on positive action prediction (mAP-like reward)
                        positive_mask = expert_actions > 0.5
                        
                        if np.sum(positive_mask) > 0:
                            # Reward for correctly predicting positive actions
                            correct_positives = np.sum(
                                (binary_action[positive_mask] == 1) & (expert_actions[positive_mask] == 1)
                            )
                            total_positives = np.sum(positive_mask)
                            
                            # Precision-like reward
                            predicted_positives = np.sum(binary_action > 0.5)
                            if predicted_positives > 0:
                                precision = correct_positives / predicted_positives
                                recall = correct_positives / total_positives
                                
                                # F1-like reward (similar to mAP optimization)
                                if (precision + recall) > 0:
                                    f1_score = 2 * (precision * recall) / (precision + recall)
                                    expert_match_score = f1_score
                                    reward += self.reward_weights['expert_matching'] * f1_score
                            else:
                                # No predictions when there should be positive actions
                                reward -= 10.0
                        
                        # Small penalty for incorrect negative predictions
                        negative_mask = expert_actions <= 0.5
                        if np.sum(negative_mask) > 0:
                            correct_negatives = np.sum(
                                (binary_action[negative_mask] == 0) & (expert_actions[negative_mask] == 0)
                            )
                            negative_accuracy = correct_negatives / np.sum(negative_mask)
                            reward += 1.0 * negative_accuracy  # Small weight
                
                # 2. ACTION SPARSITY (encourage 1-3 actions like experts)
                action_count = np.sum(action > 0.5)
                if 1 <= action_count <= 3:
                    reward += self.reward_weights['action_sparsity']
                elif action_count == 0:
                    reward -= self.reward_weights['action_sparsity']
                elif action_count > 3:
                    reward -= 2.0 * (action_count - 3)
                
                # 3. COMPLETION BONUS
                if self.current_step >= self.max_episode_steps - 1:
                    reward += self.reward_weights['completion_bonus']
                
                # Debug tracking
                self.debug_rewards.append(reward)
                self.debug_expert_matches.append(expert_match_score)
                
                return np.clip(reward, -20.0, 60.0)
            
            def get_debug_info(self):
                """Get debug information about rewards and expert matching."""
                if not self.debug_rewards:
                    return {}
                
                return {
                    'avg_reward': np.mean(self.debug_rewards[-100:]),
                    'avg_expert_match': np.mean(self.debug_expert_matches[-100:]),
                    'reward_trend': 'improving' if len(self.debug_rewards) > 50 and 
                                   np.mean(self.debug_rewards[-25:]) > np.mean(self.debug_rewards[-50:-25]) 
                                   else 'stable'
                }
        
        return SimplifiedExpertMatchingEnv
    
    def plot_comprehensive_training_analysis(self, training_data: Dict[str, List]) -> None:
        """Create comprehensive plots to visualize training dynamics."""
        
        self.logger.info("📊 Creating comprehensive training analysis plots...")
        
        # Create subplots for different aspects of training
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Reward Progression', 'Expert Action Matching',
                'Action Density vs Expert', 'mAP During Training', 
                'World Model Quality', 'Threshold Optimization'
            ],
            specs=[[{"secondary_y": True}, {"secondary_y": True}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Extract data
        episodes = list(range(len(training_data.get('episode_rewards', []))))
        
        # 1. Reward progression
        if 'episode_rewards' in training_data:
            rewards = training_data['episode_rewards']
            moving_avg = self._calculate_moving_average(rewards, window=20)
            
            fig.add_trace(
                go.Scatter(x=episodes, y=rewards, name='Episode Reward', 
                          line=dict(color='lightblue', width=1), opacity=0.6),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=episodes, y=moving_avg, name='Moving Average', 
                          line=dict(color='blue', width=3)),
                row=1, col=1
            )
        
        # 2. Expert action matching
        if 'expert_matching_scores' in training_data:
            expert_scores = training_data['expert_matching_scores']
            fig.add_trace(
                go.Scatter(x=episodes, y=expert_scores, name='Expert Matching',
                          line=dict(color='green', width=2)),
                row=1, col=2
            )
            fig.add_hline(y=0.5, line_dash="dash", line_color="red", 
                         annotation_text="50% Target", row=1, col=2)
        
        # 3. Action density comparison
        if 'predicted_action_density' in training_data and 'expert_action_density' in training_data:
            fig.add_trace(
                go.Scatter(x=episodes, y=training_data['predicted_action_density'],
                          name='Predicted Density', line=dict(color='orange')),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(x=episodes, y=training_data['expert_action_density'],
                          name='Expert Density', line=dict(color='purple')),
                row=2, col=1
            )
        
        # 4. mAP during training
        if 'training_mAP' in training_data:
            mAP_scores = training_data['training_mAP']
            fig.add_trace(
                go.Scatter(x=episodes, y=mAP_scores, name='Training mAP',
                          line=dict(color='red', width=2)),
                row=2, col=2
            )
            fig.add_hline(y=0.1, line_dash="dash", line_color="green",
                         annotation_text="10% Target", row=2, col=2)
        
        # Update layout
        fig.update_layout(
            title_text="RL Training Analysis: Expert Action Matching Focus",
            showlegend=True,
            height=900
        )
        
        # Save interactive plot
        fig.write_html(str(self.save_dir / 'comprehensive_training_analysis.html'))
        
        # Also create static version
        fig.write_image(str(self.save_dir / 'comprehensive_training_analysis.png'), 
                       width=1200, height=900)
        
        self.logger.info(f"📊 Training analysis plots saved to {self.save_dir}")
    
    def _calculate_moving_average(self, data: List[float], window: int = 20) -> List[float]:
        """Calculate moving average for smoothing plots."""
        if len(data) < window:
            return data
        
        moving_avg = []
        for i in range(len(data)):
            start_idx = max(0, i - window + 1)
            moving_avg.append(np.mean(data[start_idx:i+1]))
        
        return moving_avg
    
    def compare_supervised_vs_rl(self, supervised_results: Dict, rl_results: Dict) -> Dict:
        """
        Compare supervised learning vs RL results to understand the performance gap.
        """
        
        self.logger.info("🔍 Comparing Supervised vs RL Performance...")
        
        comparison = {
            'supervised_mAP': supervised_results.get('mAP', 0.0),
            'rl_mAP': rl_results.get('mAP', 0.0),
            'performance_gap': 0.0,
            'analysis': {},
            'recommendations': []
        }
        
        # Calculate performance gap
        comparison['performance_gap'] = comparison['supervised_mAP'] - comparison['rl_mAP']
        
        # Detailed analysis
        comparison['analysis'] = {
            'relative_performance': comparison['rl_mAP'] / comparison['supervised_mAP'] 
                                  if comparison['supervised_mAP'] > 0 else 0.0,
            'gap_magnitude': 'large' if comparison['performance_gap'] > 0.05 else 'moderate',
            'rl_vs_supervised_ratio': comparison['rl_mAP'] / comparison['supervised_mAP']
                                    if comparison['supervised_mAP'] > 0 else 0.0
        }
        
        # Generate recommendations based on analysis
        if comparison['performance_gap'] > 0.05:  # Large gap
            comparison['recommendations'].extend([
                "Focus on expert action matching rewards",
                "Simplify reward function to align with mAP",
                "Check action space conversion and thresholding",
                "Validate world model prediction quality",
                "Increase expert demonstration matching weight"
            ])
        
        if comparison['analysis']['relative_performance'] < 0.3:  # RL < 30% of supervised
            comparison['recommendations'].extend([
                "Consider behavioral cloning warm-start for RL",
                "Use supervised model to generate additional training data",
                "Implement curriculum learning with increasing difficulty"
            ])
        
        self.logger.info(f"🔍 Performance Comparison Complete")
        self.logger.info(f"   Supervised mAP: {comparison['supervised_mAP']:.4f}")
        self.logger.info(f"   RL mAP: {comparison['rl_mAP']:.4f}")
        self.logger.info(f"   Performance gap: {comparison['performance_gap']:.4f}")
        self.logger.info(f"   Relative performance: {comparison['analysis']['relative_performance']:.2%}")
        
        return comparison
    
    def save_debug_report(self) -> str:
        """Save a comprehensive debug report with all findings and recommendations."""
        
        report_path = self.save_dir / 'rl_debug_report.json'
        
        debug_report = {
            'timestamp': str(pd.Timestamp.now()),
            'config': self.config,
            'training_metrics': dict(self.training_metrics),
            'world_model_metrics': dict(self.world_model_metrics),
            'action_analysis': dict(self.action_analysis),
            'reward_analysis': dict(self.reward_analysis),
            'expert_matching_analysis': dict(self.expert_matching_analysis),
            'summary': {
                'total_episodes_tracked': self.episode_counter,
                'debug_files_created': list(self.save_dir.glob('*.json')),
                'plots_created': list(self.save_dir.glob('*.png')) + list(self.save_dir.glob('*.html'))
            }
        }
        
        with open(report_path, 'w') as f:
            json.dump(debug_report, f, indent=2, default=str)
        
        self.logger.info(f"📋 Debug report saved to: {report_path}")
        return str(report_path)


class ExpertMatchingCallback(BaseCallback):
    """
    Specialized callback for tracking expert action matching during RL training.
    This helps identify if RL is learning to match expert demonstrations.
    """
    
    def __init__(self, eval_env, expert_data: List[Dict], debugger: RLDebugger, eval_freq: int = 500):
        super().__init__()
        self.eval_env = eval_env
        self.expert_data = expert_data
        self.debugger = debugger
        self.eval_freq = eval_freq
        
        # Tracking
        self.episode_count = 0
        self.expert_matching_scores = deque(maxlen=1000)
        self.action_densities = deque(maxlen=1000)
        self.mAP_scores = deque(maxlen=100)
        
        # Current episode data
        self.current_episode_predictions = []
        self.current_episode_expert_actions = []
    
    def _on_step(self) -> bool:
        """Called at each training step."""
        
        # Extract episode information
        if 'infos' in self.locals:
            for info in self.locals.get('infos', []):
                if 'episode' in info:
                    self._process_episode_end(info)
                
                # Track step-level information
                if 'expert_match_score' in info:
                    self.expert_matching_scores.append(info['expert_match_score'])
                
                if 'action_density' in info:
                    self.action_densities.append(info['action_density'])
        
        # Periodic evaluation
        if self.num_timesteps % self.eval_freq == 0:
            self._evaluate_expert_matching()
        
        return True
    
    def _process_episode_end(self, info: Dict):
        """Process end of episode information."""
        self.episode_count += 1
        
        # Log episode summary
        if self.episode_count % 50 == 0:
            self._log_training_progress()
    
    def _evaluate_expert_matching(self):
        """Evaluate how well the RL policy matches expert actions."""
        
        try:
            # Sample some expert data for evaluation
            expert_video = np.random.choice(self.expert_data)
            frames = expert_video['frame_embeddings']
            expert_actions = expert_video['actions_binaries']
            
            predictions = []
            targets = []
            
            # Sample frames for evaluation
            sample_indices = np.random.choice(len(frames), size=min(20, len(frames)), replace=False)
            
            for idx in sample_indices:
                state = frames[idx].reshape(1, -1)
                action_pred, _ = self.model.predict(state, deterministic=True)
                
                # Convert to proper format
                action_pred = self._process_action_prediction(action_pred)
                
                predictions.append(action_pred)
                targets.append(expert_actions[idx])
            
            if predictions and targets:
                # Calculate mAP for this evaluation
                mAP = self._calculate_mAP(np.array(predictions), np.array(targets))
                self.mAP_scores.append(mAP)
                
                # Expert matching accuracy
                expert_match = np.mean([
                    np.mean(pred == target) for pred, target in zip(predictions, targets)
                ])
                
                # Log evaluation results
                self.debugger.logger.info(
                    f"📊 Step {self.num_timesteps}: mAP={mAP:.4f}, Expert Match={expert_match:.3f}"
                )
                
                # Save to debugger
                self.debugger.training_metrics['mAP_during_training'].append(mAP)
                self.debugger.training_metrics['expert_matching_during_training'].append(expert_match)
                self.debugger.training_metrics['training_steps'].append(self.num_timesteps)
        
        except Exception as e:
            self.debugger.logger.warning(f"Expert matching evaluation failed: {e}")
    
    def _process_action_prediction(self, action_pred) -> np.ndarray:
        """Process action prediction to binary format."""
        if isinstance(action_pred, np.ndarray):
            action_pred = action_pred.flatten()
        
        # Ensure 100 dimensions
        if len(action_pred) != 100:
            padded = np.zeros(100)
            if len(action_pred) > 0:
                padded[:min(len(action_pred), 100)] = action_pred[:100]
            action_pred = padded
        
        # Convert to binary using threshold
        return (action_pred > 0.5).astype(int)
    
    def _calculate_mAP(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Calculate mAP for predictions vs targets."""
        ap_scores = []
        
        for action_idx in range(predictions.shape[1]):
            if np.sum(targets[:, action_idx]) > 0:  # Only calculate for present actions
                try:
                    # Use the prediction probabilities for AP calculation
                    # We need to convert binary back to probabilities for AP calculation
                    pred_probs = predictions[:, action_idx].astype(float)
                    ap = average_precision_score(targets[:, action_idx], pred_probs)
                    ap_scores.append(ap)
                except:
                    ap_scores.append(0.0)
        
        return np.mean(ap_scores) if ap_scores else 0.0
    
    def _log_training_progress(self):
        """Log comprehensive training progress."""
        if len(self.expert_matching_scores) < 10:
            return
        
        recent_expert_scores = list(self.expert_matching_scores)[-50:]
        recent_action_densities = list(self.action_densities)[-50:]
        recent_mAP = list(self.mAP_scores)[-10:] if self.mAP_scores else [0.0]
        
        self.debugger.logger.info(f"\n📊 RL Training Progress (Episode {self.episode_count}):")
        self.debugger.logger.info(f"   Expert Matching: {np.mean(recent_expert_scores):.3f}")
        self.debugger.logger.info(f"   Action Density: {np.mean(recent_action_densities):.1f}")
        self.debugger.logger.info(f"   Recent mAP: {np.mean(recent_mAP):.4f}")
        
        # Check for learning progress
        if len(self.mAP_scores) >= 10:
            early_mAP = np.mean(list(self.mAP_scores)[-10:-5])
            late_mAP = np.mean(list(self.mAP_scores)[-5:])
            progress = late_mAP - early_mAP
            
            if progress > 0.01:
                self.debugger.logger.info(f"   📈 Learning progress: +{progress:.3f} mAP")
            elif progress < -0.01:
                self.debugger.logger.info(f"   📉 Performance declining: {progress:.3f} mAP")
            else:
                self.debugger.logger.info(f"   ➡️ Stable performance: {progress:.3f} mAP")


# Usage example
def debug_rl_training_comprehensive(config, logger, world_model, rl_model, train_data, test_data):
    """
    Comprehensive debugging function to understand RL training issues.
    """
    
    logger.info("🔍 STARTING COMPREHENSIVE RL DEBUGGING")
    logger.info("=" * 60)
    
    # Initialize debugger
    debugger = RLDebugger(
        save_dir="debug_output", 
        logger=logger, 
        config=config
    )
    
    # 1. Evaluate world model quality
    logger.info("🌍 Step 1: Evaluating World Model Quality...")
    world_model_analysis = debugger.evaluate_world_model_quality(world_model, test_data)
    
    # 2. Analyze action space handling
    logger.info("🎯 Step 2: Analyzing Action Space Handling...")
    action_analysis = debugger.analyze_action_space_handling(rl_model, test_data)
    
    # 3. Create simplified reward environment for retraining
    logger.info("🎁 Step 3: Creating Simplified Reward Environment...")
    from environment.world_model_env import WorldModelSimulationEnv
    SimplifiedEnv = debugger.create_simplified_reward_environment(WorldModelSimulationEnv)
    
    # 4. Generate comprehensive report
    logger.info("📋 Step 4: Generating Comprehensive Debug Report...")
    
    summary = {
        'world_model_quality': world_model_analysis['summary']['world_model_quality_score'],
        'optimal_action_threshold': action_analysis['optimal_threshold'],
        'optimal_threshold_mAP': action_analysis['optimal_mAP'],
        'recommendations': []
    }
    
    # Generate specific recommendations
    if summary['world_model_quality'] < 0.5:
        summary['recommendations'].append("World model quality is poor - consider retraining world model")
    
    if summary['optimal_threshold_mAP'] < 0.05:
        summary['recommendations'].append("Action space handling is problematic - focus on reward design")
    
    if summary['optimal_action_threshold'] != 0.5:
        summary['recommendations'].append(f"Use threshold {summary['optimal_action_threshold']} instead of 0.5")
    
    logger.info("🔍 DEBUGGING SUMMARY:")
    logger.info(f"   World Model Quality: {summary['world_model_quality']:.3f}")
    logger.info(f"   Optimal Threshold: {summary['optimal_action_threshold']}")
    logger.info(f"   Threshold mAP: {summary['optimal_threshold_mAP']:.4f}")
    logger.info("   Recommendations:")
    for rec in summary['recommendations']:
        logger.info(f"     - {rec}")
    
    report_path = debugger.save_debug_report()
    logger.info(f"📋 Full debug report saved to: {report_path}")
    
    return debugger, summary


if __name__ == "__main__":
    print("🔍 ENHANCED RL DEBUGGING SYSTEM")
    print("=" * 60)
    print("🎯 Focus: Understanding RL vs Supervised Learning performance gap")
    print("📊 Key debugging areas:")
    print("   ✅ World model generalization quality")
    print("   ✅ Expert action matching during training")
    print("   ✅ Reward signal alignment with mAP")
    print("   ✅ Action space handling and thresholding")
    print("   ✅ Training dynamics visualization")
    print("   ✅ Simplified reward environment for testing")
    print()
    print("🚀 This system will help identify exactly why RL can't reach 10% mAP!")
