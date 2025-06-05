#!/usr/bin/env python3
"""
Integrated Enhanced Evaluation for Surgical RL Comparison
Combines the enhanced evaluation framework with rollout saving for visualization
"""

import torch
import numpy as np
import pandas as pd
import json
import warnings
from pathlib import Path
from typing import Dict, List, Any, Tuple
from sklearn.metrics import average_precision_score, precision_recall_fscore_support
from scipy import stats
warnings.filterwarnings('ignore')

class IntegratedEvaluationFramework:
    """
    Enhanced evaluation framework with rollout saving for visualization
    """
    
    def __init__(self, results_dir: str, logger):
        self.results_dir = Path(results_dir)
        self.logger = logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create evaluation subdirectory
        self.eval_dir = self.results_dir / 'integrated_evaluation'
        self.eval_dir.mkdir(exist_ok=True)
        
        # Results storage
        self.video_results = {}
        self.aggregate_results = {}
        self.statistical_tests = {}
        self.rollout_data = {}  # For visualization
        
        self.logger.info(f"🔬 Integrated Evaluation Framework initialized")
        self.logger.info(f"📁 Results will be saved to: {self.eval_dir}")
    
    def load_all_models(self, experiment_results: Dict) -> Dict:
        """Load all trained models from experiment results"""
        
        models = {}
        
        # 1. Load Autoregressive IL model
        method1 = experiment_results.get('method_1_autoregressive_il', {})
        if method1.get('status') == 'success' and 'model_path' in method1:
            try:
                from models.autoregressive_il_model import AutoregressiveILModel
                model = AutoregressiveILModel.load_model(method1['model_path'], device=self.device)
                models['AutoregressiveIL'] = model
                self.logger.info(f"✅ Loaded AutoregressiveIL model")
            except Exception as e:
                self.logger.error(f"❌ Failed to load AutoregressiveIL model: {e}")
        
        # 2. Load Conditional World Model + RL models
        method2 = experiment_results.get('method_2_conditional_world_model', {})
        if method2.get('status') == 'success':
            # Load world model
            if 'world_model_path' in method2:
                try:
                    from models.conditional_world_model import ConditionalWorldModel
                    world_model = ConditionalWorldModel.load_model(method2['world_model_path'], device=self.device)
                    models['ConditionalWorldModel'] = world_model
                    self.logger.info(f"✅ Loaded ConditionalWorldModel")
                except Exception as e:
                    self.logger.error(f"❌ Failed to load ConditionalWorldModel: {e}")
            
            # Load RL models trained with world model
            if 'rl_models' in method2:
                for alg_name, alg_result in method2['rl_models'].items():
                    if alg_result.get('status') == 'success' and 'model_path' in alg_result:
                        try:
                            from stable_baselines3 import PPO, A2C
                            if 'ppo' in alg_name.lower():
                                rl_model = PPO.load(alg_result['model_path'])
                            elif 'a2c' in alg_name.lower():
                                rl_model = A2C.load(alg_result['model_path'])
                            else:
                                continue
                            
                            models[f"WorldModelRL_{alg_name}"] = rl_model
                            self.logger.info(f"✅ Loaded {alg_name} model from world model training")
                        except Exception as e:
                            self.logger.error(f"❌ Failed to load {alg_name} model: {e}")
        
        # 3. Load Direct Video RL models
        method3 = experiment_results.get('method_3_direct_video_rl', {})
        if method3.get('status') == 'success' and 'rl_models' in method3:
            for alg_name, alg_result in method3['rl_models'].items():
                if alg_result.get('status') == 'success' and 'model_path' in alg_result:
                    try:
                        from stable_baselines3 import PPO, A2C
                        if 'ppo' in alg_name.lower():
                            rl_model = PPO.load(alg_result['model_path'])
                        elif 'a2c' in alg_name.lower():
                            rl_model = A2C.load(alg_result['model_path'])
                        else:
                            continue
                        
                        models[f"DirectVideoRL_{alg_name}"] = rl_model
                        self.logger.info(f"✅ Loaded {alg_name} model from direct video training")
                    except Exception as e:
                        self.logger.error(f"❌ Failed to load {alg_name} model: {e}")
        
        self.logger.info(f"📊 Loaded {len(models)} models for integrated evaluation")
        return models
    
    def predict_actions_with_rollout(self, model, video_embeddings: np.ndarray, 
                                   method_name: str, horizon: int = 15) -> Tuple[np.ndarray, Dict]:
        """
        Predict actions and save detailed rollout information for visualization
        """
        
        predictions = []
        rollout_info = {
            'timestep_rollouts': {},  # Rollout at each timestep
            'confidence_scores': [],
            'thinking_process': [],
            'action_probabilities': []
        }
        
        video_length = len(video_embeddings)
        
        for t in range(min(video_length - 1, horizon)):
            # Current state at timestep t
            current_state = torch.tensor(
                video_embeddings[t], dtype=torch.float32, device=self.device
            ).unsqueeze(0)  # [1, embedding_dim]
            
            timestep_rollout = {
                'timestep': t,
                'current_state': video_embeddings[t].tolist(),
                'action_candidates': [],
                'selected_action': None,
                'confidence': 0.0,
                'planning_horizon': [],
                'thinking_steps': []
            }
            
            with torch.no_grad():
                if 'IL_Baseline' in method_name:
                    # IL model prediction with rollout
                    action_pred, rollout = self._predict_il_with_rollout(
                        model, current_state, video_embeddings, t, horizon
                    )
                    timestep_rollout.update(rollout)
                else:
                    # RL model prediction with rollout
                    action_pred, rollout = self._predict_rl_with_rollout(
                        model, current_state, video_embeddings, t, horizon
                    )
                    timestep_rollout.update(rollout)
                
                predictions.append(action_pred)
                rollout_info['timestep_rollouts'][t] = timestep_rollout
                rollout_info['confidence_scores'].append(timestep_rollout['confidence'])
        
        return np.array(predictions), rollout_info
    
    def _predict_il_with_rollout(self, il_model, current_state, video_embeddings, 
                               timestep, horizon):
        """IL prediction with detailed rollout for planning visualization"""
        
        # Forward pass through IL model
        outputs = il_model(frame_embeddings=current_state.unsqueeze(1))
        
        if 'action_pred' in outputs:
            action_probs = outputs['action_pred'][:, -1, :]  # [1, 100]
            action_pred = action_probs.cpu().numpy().flatten()
            
            # Generate planning horizon (simulate future with IL model)
            planning_horizon = []
            current_planning_state = current_state.clone()
            
            for h in range(min(5, horizon - timestep)):
                try:
                    # Use the autoregressive generation capability
                    generation_results = il_model.generate_sequence(
                        initial_frames=current_planning_state.unsqueeze(0).unsqueeze(1),
                        horizon=1,
                        temperature=0.8
                    )
                    
                    next_frame = generation_results['generated_frames'][0, 0]  # [embedding_dim]
                    next_action = generation_results['predicted_actions'][0, 0]  # [num_actions]
                    
                    planning_horizon.append({
                        'step': h + 1,
                        'predicted_frame': next_frame.cpu().numpy().tolist(),
                        'predicted_action': next_action.cpu().numpy().tolist(),
                        'confidence': float(torch.max(next_action))
                    })
                    
                    current_planning_state = next_frame
                    
                except Exception as e:
                    self.logger.warning(f"IL planning failed at step {h}: {e}")
                    break
            
            # Extract thinking process
            thinking_steps = [
                f"Analyzed current surgical context with autoregressive model",
                f"Generated {len(planning_horizon)} future frame predictions",
                f"Identified {int(np.sum(action_pred > 0.5))} potential actions",
                f"Confidence: {np.max(action_pred):.3f}"
            ]
            
            rollout = {
                'selected_action': action_pred.tolist(),
                'confidence': float(np.max(action_pred)),
                'planning_horizon': planning_horizon,
                'thinking_steps': thinking_steps,
                'action_candidates': self._get_top_actions(action_pred, k=5),
                'method_specific': {
                    'model_type': 'AutoregressiveIL',
                    'generation_used': True,
                    'causal_modeling': True
                }
            }
        else:
            action_pred = np.zeros(100)
            rollout = {
                'selected_action': action_pred.tolist(),
                'confidence': 0.0,
                'planning_horizon': [],
                'thinking_steps': ["Model output unavailable"],
                'action_candidates': []
            }
        
        return action_pred, rollout
    
    def _predict_rl_with_rollout(self, rl_model, current_state, video_embeddings, 
                               timestep, horizon):
        """RL prediction with detailed rollout for decision visualization"""
        
        # Get action from RL policy
        state_input = current_state.cpu().numpy().reshape(1, -1)
        action_pred, _ = rl_model.predict(state_input, deterministic=True)
        
        # Convert to binary action vector
        if isinstance(action_pred, np.ndarray):
            action_pred = action_pred.flatten()
        
        # Handle different action formats
        if len(action_pred) == 100:
            binary_action = (action_pred > 0.5).astype(float)
            action_probs = action_pred.copy()
        elif len(action_pred) == 1:
            # Single discrete action
            binary_action = np.zeros(100)
            action_idx = int(action_pred[0]) % 100
            binary_action[action_idx] = 1.0
            action_probs = binary_action.copy()
        else:
            # Pad or truncate
            binary_action = np.zeros(100)
            action_probs = np.zeros(100)
            if len(action_pred) > 0:
                binary_action[:min(len(action_pred), 100)] = (action_pred[:100] > 0.5).astype(float)
                action_probs[:min(len(action_pred), 100)] = action_pred[:100]
        
        # Generate RL planning rollout (multiple action predictions)
        planning_horizon = []
        current_planning_state = state_input.copy()
        
        for h in range(min(5, horizon - timestep)):
            try:
                # Predict multiple candidate actions
                future_action, _ = rl_model.predict(current_planning_state, deterministic=False)
                
                if isinstance(future_action, np.ndarray):
                    future_action = future_action.flatten()
                
                # Convert to binary
                if len(future_action) == 100:
                    future_binary = (future_action > 0.5).astype(float)
                    future_probs = future_action.copy()
                else:
                    future_binary = np.zeros(100)
                    future_probs = np.zeros(100)
                    if len(future_action) > 0:
                        future_binary[:min(len(future_action), 100)] = (future_action[:100] > 0.5).astype(float)
                        future_probs[:min(len(future_action), 100)] = future_action[:100]
                
                # Simulate next state (simple approach)
                if timestep + h + 1 < len(video_embeddings):
                    next_state_embedding = video_embeddings[timestep + h + 1]
                    current_planning_state = next_state_embedding.reshape(1, -1)
                
                planning_horizon.append({
                    'step': h + 1,
                    'predicted_action': future_binary.tolist(),
                    'confidence': float(np.max(future_probs)) if len(future_probs) > 0 else 0.0,
                    'active_actions': int(np.sum(future_binary > 0.5))
                })
            
            except Exception as e:
                self.logger.warning(f"RL planning step {h} failed: {e}")
                break
        
        # RL thinking process
        thinking_steps = [
            f"Evaluated current state with RL policy",
            f"Selected {int(np.sum(binary_action > 0.5))} actions via policy gradient",
            f"Explored {len(planning_horizon)} future decisions",
            f"Policy confidence: {np.max(action_probs):.3f}"
        ]
        
        rollout = {
            'selected_action': binary_action.tolist(),
            'confidence': float(np.max(action_probs)),
            'planning_horizon': planning_horizon,
            'thinking_steps': thinking_steps,
            'action_candidates': self._get_top_actions(action_probs, k=5),
            'method_specific': {
                'rl_algorithm': type(rl_model).__name__,
                'policy_info': {
                    'deterministic_action': action_pred.tolist() if hasattr(action_pred, 'tolist') else [float(action_pred)],
                    'exploration_used': False
                }
            }
        }
        
        return binary_action, rollout
    
    def _get_top_actions(self, action_probs, k=5):
        """Get top k action candidates with probabilities"""
        top_indices = np.argsort(action_probs)[-k:][::-1]
        return [
            {
                'action_id': int(idx),
                'probability': float(action_probs[idx]),
                'rank': i + 1
            }
            for i, idx in enumerate(top_indices)
        ]
    
    def evaluate_single_video_with_rollouts(self, models: Dict, video: Dict, 
                                          horizon: int = 15) -> Dict:
        """Evaluate all models on a single video with detailed rollout saving"""
        
        video_id = video['video_id']
        video_embeddings = video['frame_embeddings'][:horizon+1]
        ground_truth_actions = video['actions_binaries'][:horizon+1]
        
        self.logger.info(f"📹 Evaluating {video_id} with rollouts (horizon: {horizon})")
        
        # Ground truth for evaluation
        gt_for_evaluation = ground_truth_actions[1:horizon+1]
        
        video_result = {
            'video_id': video_id,
            'horizon': horizon,
            'predictions': {},
            'rollouts': {},
            'metrics': {},
            'summary': {},
            'visualization_data': {
                'ground_truth': {
                    'actions': [actions.tolist() for actions in ground_truth_actions],
                    'phases': video.get('phase_binaries', [])[:horizon+1]
                },
                'predictions': {},
                'metadata': {
                    'video_length': len(video_embeddings),
                    'horizon': horizon,
                    'timestamp': str(pd.Timestamp.now())
                }
            }
        }
        
        # Evaluate each model
        for method_name, model in models.items():
            self.logger.info(f"  🤖 Evaluating {method_name} with rollouts")
            
            try:
                # Get predictions with detailed rollouts
                predictions, rollout_info = self.predict_actions_with_rollout(
                    model, video_embeddings, method_name, horizon
                )
                
                # Ensure predictions match ground truth length
                min_len = min(len(predictions), len(gt_for_evaluation))
                predictions = predictions[:min_len]
                gt_adjusted = gt_for_evaluation[:min_len]
                
                # Compute trajectory metrics
                trajectory_metrics = self.compute_trajectory_metrics(predictions, gt_adjusted)
                
                # Store results
                video_result['predictions'][method_name] = predictions.tolist()
                video_result['rollouts'][method_name] = rollout_info
                video_result['metrics'][method_name] = trajectory_metrics
                
                # Summary statistics
                video_result['summary'][method_name] = {
                    'final_mAP': trajectory_metrics['cumulative_mAP'][-1] if trajectory_metrics['cumulative_mAP'] else 0.0,
                    'mean_mAP': np.mean(trajectory_metrics['cumulative_mAP']) if trajectory_metrics['cumulative_mAP'] else 0.0,
                    'mAP_degradation': (trajectory_metrics['cumulative_mAP'][0] - trajectory_metrics['cumulative_mAP'][-1]) if len(trajectory_metrics['cumulative_mAP']) > 1 else 0.0,
                    'final_exact_match': trajectory_metrics['cumulative_exact_match'][-1] if trajectory_metrics['cumulative_exact_match'] else 0.0,
                    'mean_exact_match': np.mean(trajectory_metrics['cumulative_exact_match']) if trajectory_metrics['cumulative_exact_match'] else 0.0,
                    'avg_confidence': np.mean(rollout_info['confidence_scores']) if rollout_info['confidence_scores'] else 0.0
                }
                
                # Add to visualization data
                video_result['visualization_data']['predictions'][method_name] = {
                    'past_actions': predictions.tolist(),
                    'future_rollouts': {
                        str(t): rollout_info['timestep_rollouts'][t]['planning_horizon']
                        for t in rollout_info['timestep_rollouts']
                    },
                    'confidence_timeline': rollout_info['confidence_scores'],
                    'thinking_process': {
                        str(t): rollout_info['timestep_rollouts'][t]['thinking_steps']
                        for t in rollout_info['timestep_rollouts']
                    },
                    'metadata': {
                        'method_type': 'IL' if 'IL' in method_name else 'RL',
                        'avg_confidence': video_result['summary'][method_name]['avg_confidence'],
                        'planning_depth': len(rollout_info['timestep_rollouts'].get(0, {}).get('planning_horizon', []))
                    }
                }
                
                self.logger.info(f"    📊 Final mAP: {video_result['summary'][method_name]['final_mAP']:.3f}")
                
            except Exception as e:
                self.logger.error(f"    ❌ Error evaluating {method_name}: {e}")
                import traceback
                traceback.print_exc()
        
        return video_result
    
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
            pred_cumulative = predictions[:t]
            gt_cumulative = ground_truth[:t]
            
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
    
    def run_integrated_evaluation(self, models: Dict, test_data: List[Dict], 
                                horizon: int = 15) -> Dict:
        """Run integrated evaluation with rollout saving"""
        
        self.logger.info("🎯 Running Integrated Evaluation with Rollout Saving")
        self.logger.info(f"📊 Evaluating {len(models)} models on {len(test_data)} videos")
        self.logger.info(f"⏱️ Prediction horizon: {horizon} timesteps")
        self.logger.info("=" * 60)
        
        # Evaluate each video with rollouts
        for video_idx, video in enumerate(test_data):
            self.logger.info(f"📹 Video {video_idx + 1}/{len(test_data)}")
            
            video_result = self.evaluate_single_video_with_rollouts(models, video, horizon)
            self.video_results[video['video_id']] = video_result
        
        # Compute aggregate statistics
        self.aggregate_results = self.compute_aggregate_statistics()
        
        # Perform statistical tests
        self.statistical_tests = self.perform_statistical_tests()
        
        # Save visualization data
        self.save_visualization_data()
        
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
    
    def save_visualization_data(self):
        """Save data in format compatible with the HTML visualization"""
        
        visualization_data = {
            'ground_truth': {},
            'predictions': {},
            'metadata': {
                'methods': list(self.aggregate_results.keys()),
                'videos': list(self.video_results.keys()),
                'evaluation_timestamp': str(pd.Timestamp.now()),
                'per_method': {}
            }
        }
        
        # Collect data for each video
        for video_id, video_result in self.video_results.items():
            # Ground truth
            visualization_data['ground_truth'][video_id] = video_result['visualization_data']['ground_truth']
            
            # Predictions for each method
            for method_name in video_result['visualization_data']['predictions']:
                if method_name not in visualization_data['predictions']:
                    visualization_data['predictions'][method_name] = {}
                
                visualization_data['predictions'][method_name][video_id] = video_result['visualization_data']['predictions'][method_name]
        
        # Add method-level metadata
        for method_name, stats in self.aggregate_results.items():
            visualization_data['metadata']['per_method'][method_name] = {
                'avg_confidence': stats.get('final_mAP', {}).get('mean', 0.0),
                'performance_rank': 0,  # Will be computed
                'method_type': 'IL' if 'IL' in method_name else 'RL'
            }
        
        # Compute performance rankings
        method_performances = [(method, stats['final_mAP']['mean']) 
                             for method, stats in self.aggregate_results.items()]
        method_performances.sort(key=lambda x: x[1], reverse=True)
        
        for rank, (method, _) in enumerate(method_performances):
            visualization_data['metadata']['per_method'][method]['performance_rank'] = rank + 1
        
        # Save visualization data
        viz_path = self.eval_dir / 'visualization_data.json'
        with open(viz_path, 'w') as f:
            json.dump(visualization_data, f, indent=2, default=str)
        
        self.logger.info(f"📊 Visualization data saved to: {viz_path}")
        
        return viz_path
    
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
            avg_confidences = []
            
            for video_result in self.video_results.values():
                if method in video_result['summary']:
                    summary = video_result['summary'][method]
                    final_maps.append(summary['final_mAP'])
                    mean_maps.append(summary['mean_mAP'])
                    map_degradations.append(summary['mAP_degradation'])
                    final_exact_matches.append(summary['final_exact_match'])
                    mean_exact_matches.append(summary['mean_exact_match'])
                    avg_confidences.append(summary['avg_confidence'])
            
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
                    'confidence': {
                        'mean': np.mean(avg_confidences),
                        'std': np.std(avg_confidences)
                    },
                    'trajectory_stability': -np.mean(map_degradations),
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
    
    def save_all_results(self):
        """Save all results including rollouts and visualization data"""
        
        self.logger.info("💾 Saving integrated evaluation results...")
        
        # 1. Save video-level results with rollouts
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
        video_csv_path = self.eval_dir / 'integrated_video_results.csv'
        video_df.to_csv(video_csv_path, index=False)
        
        # 2. Save detailed rollout data
        rollout_path = self.eval_dir / 'detailed_rollouts.json'
        rollout_data = {
            video_id: {
                method: video_result['rollouts'].get(method, {})
                for method in video_result['rollouts']
            }
            for video_id, video_result in self.video_results.items()
        }
        
        with open(rollout_path, 'w') as f:
            json.dump(rollout_data, f, indent=2, default=str)
        
        # 3. Save aggregate statistics
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
        agg_csv_path = self.eval_dir / 'integrated_aggregate_results.csv'
        agg_df.to_csv(agg_csv_path, index=False)
        
        # 4. Save complete results
        complete_results = {
            'video_results': self.video_results,
            'aggregate_results': self.aggregate_results,
            'statistical_tests': self.statistical_tests,
            'evaluation_metadata': {
                'evaluation_type': 'integrated_with_rollouts',
                'timestamp': str(pd.Timestamp.now()),
                'num_videos': len(self.video_results),
                'num_methods': len(self.aggregate_results)
            }
        }
        
        complete_path = self.eval_dir / 'complete_integrated_results.json'
        with open(complete_path, 'w') as f:
            json.dump(complete_results, f, indent=2, default=str)
        
        self.logger.info(f"📊 Integrated results saved to: {self.eval_dir}")
        
        return {
            'video_csv': video_csv_path,
            'rollout_json': rollout_path,
            'aggregate_csv': agg_csv_path,
            'complete_json': complete_path,
            'visualization_json': self.eval_dir / 'visualization_data.json'
        }


# Integration function for run_experiment_v2.py
def run_integrated_evaluation(experiment_results: Dict, test_data: List[Dict], 
                            results_dir: str, logger, horizon: int = 15):
    """
    Run integrated evaluation with rollout saving
    
    This replaces the _run_comprehensive_evaluation method in run_experiment_v2.py
    """
    
    # Initialize integrated evaluation framework
    evaluator = IntegratedEvaluationFramework(results_dir, logger)
    
    # Load all models
    models = evaluator.load_all_models(experiment_results)
    
    if not models:
        logger.error("❌ No models available for integrated evaluation")
        return None
    
    # Run integrated evaluation with rollouts
    results = evaluator.run_integrated_evaluation(models, test_data, horizon)
    
    # Save all results including visualization data
    file_paths = evaluator.save_all_results()
    
    # Print summary
    logger.info("🎉 INTEGRATED EVALUATION COMPLETED!")
    logger.info("=" * 50)
    logger.info("📊 Key Features:")
    logger.info("  ✅ Unified mAP metrics for all methods")
    logger.info("  ✅ Detailed rollout saving at every timestep")
    logger.info("  ✅ Planning horizon visualization data")
    logger.info("  ✅ Thinking process capture")
    logger.info("  ✅ Statistical significance testing")
    logger.info("  ✅ Visualization-ready data export")
    
    return {
        'evaluator': evaluator,
        'results': results,
        'file_paths': file_paths
    }
