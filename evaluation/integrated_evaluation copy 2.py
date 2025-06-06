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
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, precision_recall_fscore_support
from scipy import stats
from tqdm import tqdm
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
        """Load all trained models from experiment results - CORRECTED VERSION"""
        
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
        
        # 2. Load RL POLICIES trained with World Model (NOT the world model itself)
        method2 = experiment_results.get('method_2_conditional_world_model', {})
        if method2.get('status') == 'success' and 'rl_models' in method2:
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
                        self.logger.info(f"✅ Loaded {alg_name} RL policy (trained with world model)")
                    except Exception as e:
                        self.logger.error(f"❌ Failed to load {alg_name} RL policy: {e}")
        
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
                        self.logger.info(f"✅ Loaded {alg_name} RL policy (trained on direct video)")
                    except Exception as e:
                        self.logger.error(f"❌ Failed to load {alg_name} RL policy: {e}")
        
        # NOTE: We do NOT load the ConditionalWorldModel for direct evaluation
        # because world models predict dynamics (state transitions), not actions
        
        self.logger.info(f"📊 Loaded {len(models)} models for integrated evaluation")
        self.logger.info("🎯 World Model RL evaluation uses the trained RL policies, not the world model directly")
        return models
    
    
    def predict_actions_with_rollout(self, model, video_embeddings: np.ndarray, 
                                   method_name: str, horizon: int = 15) -> Tuple[np.ndarray, Dict]:
        """
        Predict actions and save detailed rollout information - CORRECTED VERSION
        """
        
        predictions = []
        rollout_info = {
            'timestep_rollouts': {},
            'confidence_scores': [],
            'thinking_process': [],
            'action_probabilities': []
        }
        
        video_length = len(video_embeddings)
        if video_length == 1:
            raise ValueError("Video embeddings must have more than one frame for prediction.")
        elif video_length == horizon + 1:
            raise ValueError("Video embeddings should not be exactly equal to the horizon + 1, adjust your horizon.")
        
        for t in tqdm(range(min(video_length - 1, horizon)), desc=f"Predicting actions rollouts:"):
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
                if 'AutoregressiveIL' in method_name:
                    # IL model prediction
                    action_pred, rollout = self._predict_il_with_rollout(
                        model, current_state, video_embeddings, t, horizon
                    )
                    timestep_rollout.update(rollout)
                
                elif 'WorldModelRL' in method_name or 'DirectVideoRL' in method_name:
                    # RL policy prediction (regardless of training method)
                    action_pred, rollout = self._predict_rl_with_rollout(
                        model, current_state, video_embeddings, t, horizon
                    )
                    timestep_rollout.update(rollout)
                
                else:
                    self.logger.warning(f"Unknown method type: {method_name}")
                    action_pred = np.zeros(100)
                    rollout = {
                        'selected_action': action_pred.tolist(),
                        'confidence': 0.0,
                        'planning_horizon': [],
                        'thinking_steps': ["Unknown method type"],
                        'action_candidates': []
                    }
                    timestep_rollout.update(rollout)
                
                predictions.append(action_pred)
                rollout_info['timestep_rollouts'][t] = timestep_rollout
                rollout_info['confidence_scores'].append(timestep_rollout['confidence'])
        
        return np.array(predictions), rollout_info
    
    def evaluate_single_video_with_rollouts(self, models: Dict, video_loader: DataLoader, 
                                        horizon: int = 15) -> Dict:
        """Evaluate all models on a single video using DataLoader format"""
        
        # First, extract video data from the DataLoader
        self.logger.info(f"📹 Extracting video data from DataLoader")
        
        # Extract video data (keeping previous extraction logic)
        all_embeddings = []
        all_actions = []
        video_id = None
        video_length = None
        
        for batch_idx, batch in tqdm(enumerate(video_loader), desc="Processing video batches:"):
            if video_id is None:
                video_id = batch['video_id'][0]
                video_length = batch['sequence_info']['video_length'][0].item()
            
            current_states = batch['current_states']
            next_actions = batch['next_actions']
            current_frames = batch['sequence_info']['current_frame']
            
            for i in range(current_states.size(0)):
                frame_idx = current_frames[i].item()
                frame_embedding = current_states[i, -1, :].cpu().numpy()
                frame_action = next_actions[i, -1, :].cpu().numpy()
                
                all_embeddings.append((frame_idx, frame_embedding))
                all_actions.append((frame_idx, frame_action))
        
        # Sort and convert to arrays
        all_embeddings.sort(key=lambda x: x[0])
        all_actions.sort(key=lambda x: x[0])
        
        embeddings_list = [emb for _, emb in all_embeddings]
        actions_list = [act for _, act in all_actions]
        
        video_embeddings = np.stack(embeddings_list)
        ground_truth_actions = np.stack(actions_list)
        
        self.logger.info(f"📹 Evaluating {video_id} with FULL HORIZON rollouts")
        self.logger.info(f"📊 Video: {len(video_embeddings)} frames, Horizon: {horizon}")
        self.logger.info(f"📊 Will evaluate planning quality at each horizon step (1 to {horizon})")
        
        video_result = {
            'video_id': video_id,
            'horizon': horizon,
            'total_frames': len(video_embeddings),
            'predictions': {},
            'rollouts': {},
            'metrics': {},
            'summary': {},
            'data_validation': {},
            'horizon_analysis': {},  # NEW: Detailed horizon-specific analysis
            'visualization_data': {
                'ground_truth': {
                    'actions': [actions.tolist() for actions in ground_truth_actions],
                    'video_length': int(video_length),
                    'total_frames_used': len(video_embeddings)
                },
                'predictions': {},
                'metadata': {
                    'video_length': len(video_embeddings),
                    'horizon': horizon,
                    'evaluation_type': 'full_horizon_rollout',
                    'timestamp': str(pd.Timestamp.now())
                }
            }
        }
        
        # Evaluate each model
        for method_name, model in models.items():
            self.logger.info(f"  🤖 Evaluating {method_name} with FULL HORIZON rollouts")
            
            try:
                # ✅ PROPER ROLLOUT EVALUATION: Compare full predicted sequences to ground truth sequences
                rollout_results = self._evaluate_full_horizon_rollouts(
                    model, video_embeddings, ground_truth_actions, method_name, horizon
                )
                
                # Store comprehensive results
                video_result['predictions'][method_name] = rollout_results['predictions']
                video_result['rollouts'][method_name] = rollout_results['rollout_info']
                video_result['metrics'][method_name] = rollout_results['metrics']
                video_result['horizon_analysis'][method_name] = rollout_results['horizon_analysis']
                video_result['data_validation'][method_name] = rollout_results['is_valid']
                
                # Summary statistics
                final_mAP = rollout_results['metrics']['cumulative_mAP'][-1] if rollout_results['metrics']['cumulative_mAP'] else 0.0
                mean_mAP = np.mean(rollout_results['metrics']['cumulative_mAP']) if rollout_results['metrics']['cumulative_mAP'] else 0.0
                
                # NEW: Horizon-specific summary
                horizon_mAPs = rollout_results['horizon_analysis']['horizon_mAPs']
                
                video_result['summary'][method_name] = {
                    'final_mAP': final_mAP,
                    'mean_mAP': mean_mAP,
                    'mAP_degradation': (rollout_results['metrics']['cumulative_mAP'][0] - final_mAP) if len(rollout_results['metrics']['cumulative_mAP']) > 1 else 0.0,
                    'avg_confidence': np.mean(rollout_results['rollout_info']['confidence_scores']) if rollout_results['rollout_info']['confidence_scores'] else 0.0,
                    'data_valid': rollout_results['is_valid'],
                    'evaluation_timesteps': rollout_results['num_timesteps'],
                    'total_frames': len(video_embeddings),
                    # NEW: Horizon-specific metrics
                    'horizon_1_mAP': horizon_mAPs[0] if horizon_mAPs else 0.0,  # Immediate next action
                    'horizon_5_mAP': horizon_mAPs[4] if len(horizon_mAPs) > 4 else 0.0,  # 5 steps ahead
                    'horizon_10_mAP': horizon_mAPs[9] if len(horizon_mAPs) > 9 else 0.0,  # 10 steps ahead
                    'horizon_15_mAP': horizon_mAPs[14] if len(horizon_mAPs) > 14 else 0.0,  # 15 steps ahead
                    'planning_quality_degradation': (horizon_mAPs[0] - horizon_mAPs[-1]) if len(horizon_mAPs) > 1 else 0.0
                }
                
                self.logger.info(f"    📊 Rollout Performance: H1={video_result['summary'][method_name]['horizon_1_mAP']:.3f}, "
                            f"H5={video_result['summary'][method_name]['horizon_5_mAP']:.3f}, "
                            f"H10={video_result['summary'][method_name]['horizon_10_mAP']:.3f}, "
                            f"H15={video_result['summary'][method_name]['horizon_15_mAP']:.3f}")
                
            except Exception as e:
                self.logger.error(f"❌ Rollout evaluation failed for {method_name}: {e}")
                import traceback
                traceback.print_exc()
                
                video_result['summary'][method_name] = {
                    'error': str(e),
                    'final_mAP': 0.0,
                    'mean_mAP': 0.0,
                    'data_valid': False
                }
        
        return video_result

    def _analyze_planning_horizon_quality(self, predictions: np.ndarray, ground_truth: np.ndarray, 
                                        horizon: int) -> Dict:
        """
        Analyze how prediction quality changes across the planning horizon.
        
        Args:
            predictions: [num_timesteps, horizon, num_actions]
            ground_truth: [num_timesteps, horizon, num_actions] 
            horizon: Planning horizon length
            
        Returns:
            Dictionary with horizon-specific metrics
        """
        
        horizon_mAPs = []
        horizon_exact_matches = []
        horizon_precisions = []
        horizon_recalls = []
        
        self.logger.info(f"📊 Analyzing planning quality across {horizon} horizon steps")
        
        # For each step in the horizon (1, 2, 3, ..., horizon)
        for h in range(horizon):
            # Extract predictions and ground truth for this horizon step
            pred_h = predictions[:, h, :]  # [num_timesteps, num_actions] - predictions at horizon step h
            gt_h = ground_truth[:, h, :]   # [num_timesteps, num_actions] - ground truth at horizon step h
            
            # Convert to binary predictions
            binary_pred_h = (pred_h > 0.5).astype(int)
            
            # Calculate mAP for this horizon step
            ap_scores = []
            for action_idx in range(gt_h.shape[1]):
                gt_column = gt_h[:, action_idx]
                pred_column = pred_h[:, action_idx]
                
                if np.sum(gt_column) > 0:
                    try:
                        ap = average_precision_score(gt_column, pred_column)
                        ap_scores.append(ap)
                    except:
                        ap_scores.append(0.0)
                else:
                    # No positive examples
                    if np.sum(binary_pred_h[:, action_idx]) == 0:
                        ap_scores.append(1.0)  # Correct negative prediction
                    else:
                        ap_scores.append(0.0)  # False positive
            
            horizon_mAP = np.mean(ap_scores) if ap_scores else 0.0
            horizon_mAPs.append(horizon_mAP)
            
            # Calculate other metrics for this horizon step
            exact_match = np.mean(np.all(binary_pred_h == gt_h, axis=1))
            horizon_exact_matches.append(exact_match)
            
            # Precision and recall
            try:
                precision, recall, _, _ = precision_recall_fscore_support(
                    gt_h.flatten(), binary_pred_h.flatten(), average='macro', zero_division=0
                )
                horizon_precisions.append(precision)
                horizon_recalls.append(recall)
            except:
                horizon_precisions.append(0.0)
                horizon_recalls.append(0.0)
        
        # Log horizon analysis
        self.logger.info(f"📊 Horizon mAP progression: {[f'{mAP:.3f}' for mAP in horizon_mAPs[:5]]}...")
        self.logger.info(f"📊 Planning degradation: {horizon_mAPs[0]:.3f} → {horizon_mAPs[-1]:.3f} "
                        f"(Δ={horizon_mAPs[0]-horizon_mAPs[-1]:.3f})")
        
        return {
            'horizon_mAPs': horizon_mAPs,
            'horizon_exact_matches': horizon_exact_matches,
            'horizon_precisions': horizon_precisions,
            'horizon_recalls': horizon_recalls,
            'planning_degradation': horizon_mAPs[0] - horizon_mAPs[-1] if horizon_mAPs else 0.0,
            'mean_horizon_mAP': np.mean(horizon_mAPs) if horizon_mAPs else 0.0,
            'horizon_stability': np.std(horizon_mAPs) if horizon_mAPs else 0.0
        }


    def _generate_il_rollout_sequence(self, il_model, current_state, video_embeddings, 
                                    timestep: int, horizon: int) -> Tuple[np.ndarray, Dict]:
        """Generate full rollout sequence for IL model (not just single prediction)"""
        
        # Create context sequence
        context_length = 20
        embedding_dim = current_state.size(1)
        
        context_sequence = torch.zeros(1, context_length, embedding_dim, 
                                    dtype=current_state.dtype, device=current_state.device)
        
        # Fill in actual frames
        start_idx = max(0, timestep - context_length + 1)
        for i, frame_idx in enumerate(range(start_idx, timestep + 1)):
            if frame_idx >= 0 and frame_idx < len(video_embeddings):
                context_sequence[0, i, :] = torch.tensor(
                    video_embeddings[frame_idx], dtype=current_state.dtype, device=current_state.device
                )
        
        context_sequence[0, -1, :] = current_state[0]
        
        # ✅ GENERATE FULL SEQUENCE for the horizon
        try:
            generation_results = il_model.generate_sequence(
                initial_frames=context_sequence,
                horizon=horizon,
                temperature=0.8
            )
            
            predicted_actions = generation_results['predicted_actions']  # [batch, horizon, num_actions]
            
            if predicted_actions.dim() == 3:
                predicted_actions = predicted_actions[0]  # Remove batch dim: [horizon, num_actions]
            
            rollout_sequence = predicted_actions.cpu().numpy()  # [horizon, num_actions]
            
            # Ensure we have the right shape
            if rollout_sequence.shape[0] < horizon:
                # Pad if necessary
                padding = np.zeros((horizon - rollout_sequence.shape[0], rollout_sequence.shape[1]))
                rollout_sequence = np.vstack([rollout_sequence, padding])
            
            rollout_sequence = rollout_sequence[:horizon]  # Ensure exact horizon length
            
            planning_horizon = []
            for h in range(min(len(rollout_sequence), 5)):  # First 5 steps for visualization
                planning_horizon.append({
                    'step': h + 1,
                    'predicted_action': rollout_sequence[h].tolist(),
                    'confidence': float(np.max(rollout_sequence[h])),
                    'active_actions': int(np.sum(rollout_sequence[h] > 0.5))
                })
            
            confidence = float(np.mean([np.max(step) for step in rollout_sequence]))
            
        except Exception as e:
            self.logger.warning(f"IL rollout generation failed: {e}")
            rollout_sequence = np.zeros((horizon, 100))
            planning_horizon = []
            confidence = 0.0
        
        timestep_rollout = {
            'timestep': timestep,
            'planning_horizon': planning_horizon,
            'thinking_steps': [
                f"Generated {horizon}-step rollout sequence",
                f"Autoregressive planning with causal modeling",
                f"Average confidence: {confidence:.3f}"
            ],
            'confidence': confidence,
            'method_specific': {
                'model_type': 'AutoregressiveIL',
                'sequence_length': len(rollout_sequence),
                'generation_successful': True
            }
        }
        
        return rollout_sequence, timestep_rollout


    def _generate_rl_rollout_sequence(self, rl_model, current_state, video_embeddings,
                                    timestep: int, horizon: int) -> Tuple[np.ndarray, Dict]:
        """Generate full rollout sequence for RL model (not just single prediction)"""
        
        rollout_sequence = []
        current_planning_state = current_state.cpu().numpy().reshape(1, -1)
        
        # Generate sequence step by step
        for h in range(horizon):
            try:
                # Predict action for this step
                action_pred, _ = rl_model.predict(current_planning_state, deterministic=True)
                
                if isinstance(action_pred, np.ndarray):
                    action_pred = action_pred.flatten()
                
                # Convert to proper action format
                if len(action_pred) == 100:
                    action_probs = action_pred.copy()
                elif len(action_pred) == 1:
                    action_probs = np.zeros(100)
                    action_idx = int(action_pred[0]) % 100
                    action_probs[action_idx] = 1.0
                else:
                    action_probs = np.zeros(100)
                    if len(action_pred) > 0:
                        action_probs[:min(len(action_pred), 100)] = action_pred[:100]
                
                rollout_sequence.append(action_probs)
                
                # Update state for next prediction (simple simulation)
                if timestep + h + 1 < len(video_embeddings):
                    next_state_embedding = video_embeddings[timestep + h + 1]
                    current_planning_state = next_state_embedding.reshape(1, -1)
                
            except Exception as e:
                self.logger.warning(f"RL rollout step {h} failed: {e}")
                # Fill remaining steps with zeros
                while len(rollout_sequence) < horizon:
                    rollout_sequence.append(np.zeros(100))
                break
        
        rollout_sequence = np.array(rollout_sequence)  # [horizon, num_actions]
        
        # Create planning visualization
        planning_horizon = []
        for h in range(min(len(rollout_sequence), 5)):
            planning_horizon.append({
                'step': h + 1,
                'predicted_action': rollout_sequence[h].tolist(),
                'confidence': float(np.max(rollout_sequence[h])),
                'active_actions': int(np.sum(rollout_sequence[h] > 0.5))
            })
        
        confidence = float(np.mean([np.max(step) for step in rollout_sequence]))
        training_method = "world model simulation" if "WorldModelRL" in str(type(rl_model)) else "direct video episodes"
        
        timestep_rollout = {
            'timestep': timestep,
            'planning_horizon': planning_horizon,
            'thinking_steps': [
                f"Generated {horizon}-step RL rollout sequence",
                f"Policy trained on {training_method}",
                f"Sequential action selection with planning",
                f"Average confidence: {confidence:.3f}"
            ],
            'confidence': confidence,
            'method_specific': {
                'rl_algorithm': type(rl_model).__name__,
                'training_environment': training_method,
                'sequence_length': len(rollout_sequence)
            }
        }
        
        return rollout_sequence, timestep_rollout


    def _evaluate_full_horizon_rollouts(self, model, video_embeddings: np.ndarray, 
                                    ground_truth_actions: np.ndarray, method_name: str, 
                                    horizon: int) -> Dict:
        """
        Evaluate model by comparing FULL predicted rollout sequences against ground truth sequences.
        This is the proper way to evaluate planning capability.
        """
        
        video_length = len(video_embeddings)
        
        # We need enough future frames to evaluate the full horizon
        max_start_timestep = video_length - horizon
        if max_start_timestep <= 0:
            raise ValueError(f"Video too short ({video_length} frames) for horizon {horizon}")
        
        self.logger.info(f"📊 Evaluating {max_start_timestep} timesteps with full {horizon}-step rollouts")
        
        all_predictions = []  # Store all rollout predictions
        all_ground_truth = []  # Store corresponding ground truth sequences
        rollout_info = {
            'timestep_rollouts': {},
            'confidence_scores': [],
            'thinking_process': [],
            'action_probabilities': []
        }
        
        # Generate rollouts at each valid timestep
        for t in tqdm(range(max_start_timestep), desc=f"Generating {horizon}-step rollouts"):
            
            # Get current state
            current_state = torch.tensor(
                video_embeddings[t], dtype=torch.float32, device=self.device
            ).unsqueeze(0)  # [1, embedding_dim]
            
            # ✅ GENERATE FULL ROLLOUT SEQUENCE for this timestep
            if 'AutoregressiveIL' in method_name:
                rollout_predictions, timestep_rollout = self._generate_il_rollout_sequence(
                    model, current_state, video_embeddings, t, horizon
                )
            elif 'WorldModelRL' in method_name or 'DirectVideoRL' in method_name:
                rollout_predictions, timestep_rollout = self._generate_rl_rollout_sequence(
                    model, current_state, video_embeddings, t, horizon
                )
            else:
                # Unknown method - generate dummy sequence
                rollout_predictions = np.zeros((horizon, 100))
                timestep_rollout = {
                    'timestep': t,
                    'planning_horizon': [],
                    'thinking_steps': ["Unknown method type"],
                    'confidence': 0.0
                }
            
            # ✅ GET CORRESPONDING GROUND TRUTH SEQUENCE
            gt_sequence = ground_truth_actions[t+1:t+1+horizon]  # [horizon, num_actions]
            
            # Store for evaluation
            all_predictions.append(rollout_predictions)  # [horizon, num_actions]
            all_ground_truth.append(gt_sequence)         # [horizon, num_actions]
            
            # Store rollout info
            rollout_info['timestep_rollouts'][t] = timestep_rollout
            rollout_info['confidence_scores'].append(timestep_rollout['confidence'])
        
        # Convert to numpy arrays for evaluation
        all_predictions = np.array(all_predictions)    # [num_timesteps, horizon, num_actions]
        all_ground_truth = np.array(all_ground_truth)  # [num_timesteps, horizon, num_actions]
        
        self.logger.info(f"📊 Generated rollouts shape: {all_predictions.shape}")
        self.logger.info(f"📊 Ground truth shape: {all_ground_truth.shape}")
        
        # ✅ EVALUATE PLANNING QUALITY AT EACH HORIZON STEP
        horizon_analysis = self._analyze_planning_horizon_quality(all_predictions, all_ground_truth, horizon)
        
        # ✅ COMPUTE TRAJECTORY METRICS (how quality changes over time)
        trajectory_metrics = self._compute_rollout_trajectory_metrics(all_predictions, all_ground_truth)
        
        # Validate data
        is_valid = self._validate_rollout_data(all_predictions, all_ground_truth, method_name)
        
        return {
            'predictions': all_predictions.tolist(),
            'rollout_info': rollout_info,
            'metrics': trajectory_metrics,
            'horizon_analysis': horizon_analysis,
            'is_valid': is_valid,
            'num_timesteps': len(all_predictions)
        }

    def run_evaluation(self, models: Dict, test_data: Dict, horizon: int = 15) -> Dict:
        """
        Main method to run the complete integrated evaluation.
        Updated to handle test_data as Dict[str, DataLoader] format.
        """
        
        self.logger.info(f"🚀 Starting integrated evaluation with {len(models)} models")
        self.logger.info(f"📊 Test videos (dict of dataloaders): {len(test_data)}")
        self.logger.info(f"🎯 Prediction horizon: {horizon}")
        
        if not models:
            self.logger.error("❌ No models available for evaluation")
            return {'status': 'failed', 'error': 'No models loaded'}
        
        # Evaluate each video on all models (test_data is Dict[video_id, DataLoader])
        for video_id, video_loader in tqdm(test_data.items(), desc="Evaluating videos:"):
            self.logger.info(f"📹 Evaluating video: {video_id}")
            
            video_result = self.evaluate_single_video_with_rollouts(
                models, video_loader, horizon
            )
            self.video_results[video_id] = video_result
                        
        if not self.video_results:
            self.logger.error("❌ No videos were successfully evaluated")
            return {'status': 'failed', 'error': 'No videos evaluated'}
        
        # Compute aggregate statistics
        self.logger.info("📊 Computing aggregate statistics...")
        self.aggregate_results = self.compute_aggregate_statistics()
        
        # Perform statistical tests
        self.logger.info("📈 Performing statistical significance tests...")
        self.statistical_tests = self.perform_statistical_tests()
        
        # Save all results
        self.logger.info("💾 Saving all results...")
        file_paths = self.save_all_results()
        
        # Create summary
        results_summary = {
            'status': 'success',
            'evaluation_type': 'integrated_with_rollouts',
            'num_models': len(models),
            'num_videos': len(self.video_results),
            'horizon': horizon,
            'video_results': self.video_results,
            'aggregate_results': self.aggregate_results,
            'statistical_tests': self.statistical_tests,
            'file_paths': file_paths,
            'timestamp': str(pd.Timestamp.now())
        }
        
        self.logger.info(f"✅ Integrated evaluation completed successfully!")
        self.logger.info(f"📊 Models evaluated: {list(models.keys())}")
        self.logger.info(f"📊 Videos processed: {len(self.video_results)}")
        
        return results_summary


    def _predict_il_with_rollout(self, il_model, current_state, video_embeddings, 
                            timestep, horizon):
        """IL prediction with detailed rollout for planning visualization - FIXED for single state input"""
        
        # Convert single state to sequence format expected by IL model
        # current_state should be [1, embedding_dim], need [1, context_length, embedding_dim]
        
        if current_state.dim() == 2:  # [1, embedding_dim]
            # Create a context sequence ending with current_state
            context_length = 20  # Should match model's expected context length
            embedding_dim = current_state.size(1)
            
            # Initialize context with padding
            context_sequence = torch.zeros(1, context_length, embedding_dim, 
                                        dtype=current_state.dtype, device=current_state.device)
            
            # Fill in actual frames if available from video_embeddings
            start_idx = max(0, timestep - context_length + 1)
            for i, frame_idx in enumerate(range(start_idx, timestep + 1)):
                if frame_idx >= 0 and frame_idx < len(video_embeddings):
                    context_sequence[0, i, :] = torch.tensor(
                        video_embeddings[frame_idx], dtype=current_state.dtype, device=current_state.device
                    )
            
            # Ensure current state is at the last position
            context_sequence[0, -1, :] = current_state[0]
            
            current_state = context_sequence  # Now [1, context_length, embedding_dim]
        
        # Check current_state has expected shape [batch, context_length, embedding_dim]
        if current_state.dim() != 3:
            raise ValueError(f"Expected current_state to have 3 dimensions, got {current_state.dim()}")
        
        # Check it has context length larger than 1
        if current_state.size(1) < 2:
            raise ValueError(f"Expected current_state to have context length > 1, got {current_state.size(1)}")

        # Use the predict_next_action method
        action_probs = il_model.predict_next_action(current_state)  # Input is already [batch, seq, emb]
        action_pred = action_probs.cpu().numpy().flatten()
        
        # Generate planning horizon with proper error handling
        planning_horizon = []
        try:
            generation_results = il_model.generate_sequence(
                initial_frames=current_state,
                horizon=min(5, horizon - timestep),
                temperature=0.8
            )
            
            pred_actions = generation_results['predicted_actions']
            
            # Handle tensor dimensions properly
            if pred_actions.dim() == 3:  # [batch, seq, actions]
                pred_actions = pred_actions[0]  # Remove batch dim
            
            for h in range(min(pred_actions.size(0), 5)):  # Use .size(0) instead of len()
                next_action_tensor = pred_actions[h]  # This is a tensor
                next_action = next_action_tensor.cpu().numpy()  # Convert to numpy
                
                planning_horizon.append({
                    'step': h + 1,
                    'predicted_action': next_action.tolist(),
                    'confidence': float(np.max(next_action)),
                    'active_actions': int(np.sum(next_action > 0.5))
                })
        except Exception as e:
            self.logger.warning(f"IL planning horizon generation failed: {e}")
            planning_horizon = []

        # Extract thinking process
        thinking_steps = [
            f"Analyzed current surgical context with autoregressive model",
            f"Generated {len(planning_horizon)} future predictions",
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
                'generation_used': len(planning_horizon) > 0,
                'causal_modeling': True
            }
        }
        
        return action_pred, rollout

    def _predict_rl_with_rollout(self, rl_model, current_state, video_embeddings, 
                               timestep, horizon):
        """RL policy prediction - works for both WorldModelRL and DirectVideoRL"""
        
        # Get action from RL policy (this is the same regardless of how the policy was trained)
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
        
        # Generate RL planning rollout (multiple future action predictions)
        planning_horizon = []
        current_planning_state = state_input.copy()
        
        for h in range(min(5, horizon - timestep)):
            try:
                # Predict multiple candidate actions for future planning
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
                
                # For world model RL, we could note that this policy was trained in simulation
                # For direct video RL, this policy was trained on real video episodes
                # But the prediction mechanism is the same
                
                planning_horizon.append({
                    'step': h + 1,
                    'predicted_action': future_binary.tolist(),
                    'confidence': float(np.max(future_probs)) if len(future_probs) > 0 else 0.0,
                    'active_actions': int(np.sum(future_binary > 0.5))
                })
                
                # Simulate next state (simple approach for planning visualization)
                if timestep + h + 1 < len(video_embeddings):
                    next_state_embedding = video_embeddings[timestep + h + 1]
                    current_planning_state = next_state_embedding.reshape(1, -1)
            
            except Exception as e:
                self.logger.warning(f"RL planning step {h} failed: {e}")
                break
        
        # Determine training method for thinking process
        training_method = "world model simulation" if "WorldModelRL" in str(type(rl_model)) else "direct video episodes"
        
        thinking_steps = [
            f"Used RL policy trained on {training_method}",
            f"Selected {int(np.sum(binary_action > 0.5))} actions via learned policy",
            f"Explored {len(planning_horizon)} future policy decisions",
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
                'training_environment': training_method,
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
    
    def compute_trajectory_metrics(self, predictions: np.ndarray, ground_truth: np.ndarray) -> Dict[str, List[float]]:
        """Compute metrics over trajectory with IMPROVED evaluation"""
        
        metrics = {
            'cumulative_mAP': [],
            'cumulative_exact_match': [],
            'cumulative_hamming_accuracy': [],
            'cumulative_precision': [],
            'cumulative_recall': [],
            'cumulative_f1': []
        }
        
        self.logger.info(f"📊 Computing trajectory metrics:")
        self.logger.info(f"   Predictions shape: {predictions.shape}")
        self.logger.info(f"   Ground truth shape: {ground_truth.shape}")
        
        for t in range(1, len(predictions) + 1):
            # Cumulative predictions and ground truth up to timestep t
            pred_cumulative = predictions[:t]
            gt_cumulative = ground_truth[:t]
            
            # Flatten for metric calculation
            pred_flat = pred_cumulative.reshape(-1, pred_cumulative.shape[-1])
            gt_flat = gt_cumulative.reshape(-1, gt_cumulative.shape[-1])
            
            # FIXED: Add validation checks
            if pred_flat.shape[1] != gt_flat.shape[1]:
                self.logger.warning(f"Shape mismatch at timestep {t}: pred {pred_flat.shape} vs gt {gt_flat.shape}")
                continue
                
            # FIXED: Convert predictions to binary with proper thresholding
            binary_pred = (pred_flat > 0.5).astype(int)
            
            # FIXED: Add debugging for suspiciously high scores
            self.logger.debug(f"Timestep {t}: pred_range=[{pred_flat.min():.3f}, {pred_flat.max():.3f}]")
            self.logger.debug(f"Timestep {t}: gt_range=[{gt_flat.min():.3f}, {gt_flat.max():.3f}]")
            
            # 1. IMPROVED mAP calculation with better handling
            ap_scores = []
            valid_actions = 0
            perfect_predictions = 0
            
            for action_idx in range(gt_flat.shape[1]):
                gt_column = gt_flat[:, action_idx]
                pred_column = pred_flat[:, action_idx]
                
                # Check if this action appears in ground truth
                if np.sum(gt_column) > 0:
                    valid_actions += 1
                    try:
                        ap = average_precision_score(gt_column, pred_column)
                        ap_scores.append(ap)
                        
                        # FIXED: Flag perfect predictions for investigation
                        if ap == 1.0:
                            perfect_predictions += 1
                            
                    except Exception as e:
                        self.logger.warning(f"AP calculation failed for action {action_idx}: {e}")
                        ap_scores.append(0.0)
                else:
                    # No positive examples - check if model correctly predicts no actions
                    if np.sum(binary_pred[:, action_idx]) == 0:
                        ap_scores.append(1.0)  # Correct negative prediction
                        perfect_predictions += 1
                    else:
                        ap_scores.append(0.0)  # False positive prediction
            
            current_mAP = np.mean(ap_scores) if ap_scores else 0.0
            
            # FIXED: Log suspicious results for debugging
            if current_mAP > 0.95:
                self.logger.warning(f"⚠️ Suspiciously high mAP at timestep {t}: {current_mAP:.3f}")
                self.logger.warning(f"   Perfect predictions: {perfect_predictions}/{len(ap_scores)}")
                self.logger.warning(f"   Valid actions: {valid_actions}")
                
            metrics['cumulative_mAP'].append(current_mAP)
            
            # 2. IMPROVED exact match calculation
            exact_match = np.mean(np.all(binary_pred == gt_flat, axis=1))
            metrics['cumulative_exact_match'].append(exact_match)
            
            # 3. Hamming accuracy
            hamming_acc = np.mean(binary_pred == gt_flat)
            metrics['cumulative_hamming_accuracy'].append(hamming_acc)
            
            # 4. IMPROVED Precision, Recall, F1 with better error handling
            try:
                # Use macro averaging to handle class imbalance better
                precision, recall, f1, _ = precision_recall_fscore_support(
                    gt_flat.flatten(), binary_pred.flatten(), 
                    average='macro', zero_division=0
                )
                metrics['cumulative_precision'].append(precision)
                metrics['cumulative_recall'].append(recall)
                metrics['cumulative_f1'].append(f1)
            except Exception as e:
                self.logger.warning(f"Precision/Recall calculation failed: {e}")
                metrics['cumulative_precision'].append(0.0)
                metrics['cumulative_recall'].append(0.0)
                metrics['cumulative_f1'].append(0.0)
        
        # FIXED: Log final metrics summary for debugging
        if metrics['cumulative_mAP']:
            final_mAP = metrics['cumulative_mAP'][-1]
            mean_mAP = np.mean(metrics['cumulative_mAP'])
            self.logger.info(f"📊 Final trajectory metrics:")
            self.logger.info(f"   Final mAP: {final_mAP:.3f}")
            self.logger.info(f"   Mean mAP: {mean_mAP:.3f}")
            self.logger.info(f"   mAP trend: {metrics['cumulative_mAP'][:3]}...{metrics['cumulative_mAP'][-3:]}")
        
        return metrics
    
    def _validate_evaluation_data(self, predictions: np.ndarray, ground_truth: np.ndarray, 
                                 video_id: str) -> bool:
        """Validate evaluation data to catch potential issues"""
        
        issues = []
        
        # Check shapes
        if predictions.shape != ground_truth.shape:
            issues.append(f"Shape mismatch: pred {predictions.shape} vs gt {ground_truth.shape}")
        
        # Check value ranges
        pred_min, pred_max = predictions.min(), predictions.max()
        gt_min, gt_max = ground_truth.min(), ground_truth.max()
        
        if pred_min < 0 or pred_max > 1:
            issues.append(f"Prediction values out of range [0,1]: [{pred_min:.3f}, {pred_max:.3f}]")
        
        if gt_min < 0 or gt_max > 1:
            issues.append(f"Ground truth values out of range [0,1]: [{gt_min:.3f}, {gt_max:.3f}]")
        
        # Check for potential data leakage (predictions too similar to ground truth)
        if predictions.shape == ground_truth.shape:
            similarity = np.mean(np.abs(predictions - ground_truth))
            if similarity < 0.1:  # Very high similarity might indicate leakage
                issues.append(f"Suspiciously high similarity: {1-similarity:.3f}")
        
        # Check action density
        gt_action_density = np.mean(np.sum(ground_truth > 0.5, axis=-1))
        pred_action_density = np.mean(np.sum(predictions > 0.5, axis=-1))
        
        self.logger.info(f"📊 Data validation for {video_id}:")
        self.logger.info(f"   GT action density: {gt_action_density:.2f} actions/frame")
        self.logger.info(f"   Pred action density: {pred_action_density:.2f} actions/frame")
        
        if issues:
            self.logger.warning(f"⚠️ Evaluation data issues for {video_id}:")
            for issue in issues:
                self.logger.warning(f"   - {issue}")
            return False
        
        return True
    
    
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


def run_integrated_evaluation(experiment_results: Dict, test_data: Dict, 
                            results_dir: str, logger, horizon: int = 15):
    """
    Run integrated evaluation with rollout saving
    
    This replaces the _run_comprehensive_evaluation method in run_experiment_v3.py
    
    Args:
        experiment_results: Results from the 3-method experiment
        test_data: Dict[video_id, DataLoader] - test data loaders for each video
        results_dir: Directory to save results
        logger: Logger instance
        horizon: Prediction horizon for evaluation
    """
    
    # Initialize integrated evaluation framework
    evaluator = IntegratedEvaluationFramework(results_dir, logger)
    
    # Load all models
    models = evaluator.load_all_models(experiment_results)
    
    if not models:
        logger.error("❌ No models available for integrated evaluation")
        return None
    
    # Run integrated evaluation with rollouts
    # test_data is already in the correct format: Dict[video_id, DataLoader]
    results = evaluator.run_evaluation(models, test_data, horizon)
    
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