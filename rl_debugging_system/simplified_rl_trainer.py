#!/usr/bin/env python3
"""
Simplified RL Trainer focused on Expert Action Matching
Goal: Get RL to match supervised learning baseline by focusing only on expert demonstration matching

Key simplifications:
1. Reward function focuses only on expert action matching (mAP-aligned)
2. Remove complex reward components that don't directly relate to mAP
3. Enhanced debugging and monitoring throughout training
4. Action space optimization based on expert patterns
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt
import json
from tqdm import tqdm

# Import our debugging system
from rl_debugging_system.rl_debug_system import RLDebugger, ExpertMatchingCallback


class SimplifiedExpertMatchingEnv:
    """
    SIMPLIFIED Environment that focuses ONLY on expert action matching.
    
    This removes all complex reward components and focuses purely on:
    1. Matching expert actions (F1-like reward to align with mAP)
    2. Appropriate action sparsity (1-3 actions like experts)
    3. Nothing else that could confuse the learning
    """
    
    def __init__(self, world_model, video_data: List[Dict], config: Dict, 
                 logger=None, device: str = 'cuda'):
        import gymnasium as gym
        from gymnasium import spaces
        
        self.world_model = world_model
        self.video_data = video_data
        self.config = config
        self.logger = logger
        self.device = device
        
        # Environment parameters
        self.max_episode_steps = config.get('rl_horizon', 20)  # Shorter episodes for faster learning
        
        # SIMPLIFIED action space - continuous [0,1] for each action
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(100,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1024,), dtype=np.float32)
        
        # Current episode state
        self.current_step = 0
        self.current_video_idx = 0
        self.current_frame_idx = 0
        self.current_state = None
        self.episode_reward = 0.0
        
        # Expert data for current episode
        self.current_expert_actions = None
        
        # SIMPLIFIED reward weights - ONLY expert matching
        self.reward_weights = {
            'expert_f1': 100.0,          # F1-like reward (precision + recall)
            'action_sparsity': 5.0,      # Encourage 1-3 actions
            'completion_bonus': 2.0      # Small episode completion bonus
        }
        
        # Debug tracking
        self.debug_info = {
            'episode_expert_f1_scores': [],
            'episode_action_densities': [],
            'episode_rewards': [],
            'action_matching_details': []
        }
        
        self.logger.info("🎯 SimplifiedExpertMatchingEnv initialized")
        self.logger.info("   Focus: ONLY expert action matching + sparsity")
        self.logger.info(f"   Reward weights: {self.reward_weights}")
    
    def reset(self, seed=None, options=None):
        """Reset environment with expert action tracking."""
        
        # Select random video and cache expert actions
        self.current_video_idx = np.random.randint(0, len(self.video_data))
        current_video = self.video_data[self.current_video_idx]
        
        # Cache expert actions for this entire video
        self.current_expert_actions = np.array(current_video['actions_binaries'])
        
        # Start from random position with enough room for episode
        min_start = 5
        max_start = len(current_video['frame_embeddings']) - self.max_episode_steps - 5
        max_start = max(min_start, max_start)
        
        self.current_frame_idx = np.random.randint(min_start, max_start + 1)
        
        # Get initial state
        self.current_state = current_video['frame_embeddings'][self.current_frame_idx].astype(np.float32)
        self.current_state = self._ensure_state_shape(self.current_state)
        
        # Reset episode variables
        self.current_step = 0
        self.episode_reward = 0.0
        
        return self.current_state.copy(), {}
    
    def step(self, action):
        """Step with SIMPLIFIED reward calculation."""
        self.current_step += 1
        
        # Process action
        action = self._process_action(action)
        
        # Check if episode should end
        frames_remaining = len(self.current_expert_actions) - self.current_frame_idx - 1
        done = (self.current_step >= self.max_episode_steps) or (frames_remaining <= 0)
        
        if done:
            # Episode completion
            reward = self.reward_weights['completion_bonus']
            next_state = self.current_state.copy()
        else:
            # Get next state from video or world model
            next_state = self._get_next_state(action)
            
            # SIMPLIFIED reward calculation
            reward = self._calculate_simplified_reward(action)
            
            # Update current state
            self.current_state = next_state.copy()
        
        self.episode_reward += reward
        
        # Debug info for monitoring
        info = {
            'step': self.current_step,
            'episode_reward': self.episode_reward,
            'action_count': float(np.sum(action > 0.5)),
            'frame_idx': self.current_frame_idx,
            'expert_actions_available': self.current_frame_idx < len(self.current_expert_actions),
            'method': 'simplified_expert_matching'
        }
        
        # Add expert matching score for debugging
        if self.current_frame_idx < len(self.current_expert_actions):
            expert_actions = self.current_expert_actions[self.current_frame_idx]
            binary_action = (action > 0.5).astype(int)
            
            if len(expert_actions) == len(binary_action):
                expert_match = np.mean(binary_action == expert_actions)
                info['expert_match_score'] = float(expert_match)
                
                # Calculate F1-like score for debugging
                positive_mask = expert_actions > 0.5
                if np.sum(positive_mask) > 0:
                    predicted_positives = np.sum(binary_action > 0.5)
                    if predicted_positives > 0:
                        correct_positives = np.sum(
                            (binary_action[positive_mask] == 1) & (expert_actions[positive_mask] == 1)
                        )
                        precision = correct_positives / predicted_positives
                        recall = correct_positives / np.sum(positive_mask)
                        
                        if (precision + recall) > 0:
                            f1_score = 2 * (precision * recall) / (precision + recall)
                            info['expert_f1_score'] = float(f1_score)
        
        # Track for episode end
        if done:
            self._record_episode_end(info)
        
        return self.current_state.copy(), reward, done, False, info
    
    def _calculate_simplified_reward(self, action: np.ndarray) -> float:
        """
        SIMPLIFIED reward that focuses ONLY on expert action matching.
        This is designed to be as close as possible to optimizing mAP directly.
        """
        
        reward = 0.0
        
        # 1. EXPERT ACTION MATCHING (primary and only complex reward)
        if self.current_frame_idx < len(self.current_expert_actions):
            expert_actions = self.current_expert_actions[self.current_frame_idx]
            binary_action = (action > 0.5).astype(int)
            
            if len(expert_actions) == len(binary_action):
                # Focus on POSITIVE action prediction (most important for mAP)
                positive_mask = expert_actions > 0.5
                total_positives = np.sum(positive_mask)
                
                if total_positives > 0:
                    # Count correct positive predictions
                    correct_positives = np.sum(
                        (binary_action[positive_mask] == 1) & (expert_actions[positive_mask] == 1)
                    )
                    
                    # Count predicted positives (for precision)
                    predicted_positives = np.sum(binary_action > 0.5)
                    
                    if predicted_positives > 0:
                        # Calculate precision and recall
                        precision = correct_positives / predicted_positives
                        recall = correct_positives / total_positives
                        
                        # F1-like reward (directly aligns with mAP optimization)
                        if (precision + recall) > 0:
                            f1_score = 2 * (precision * recall) / (precision + recall)
                            reward += self.reward_weights['expert_f1'] * f1_score
                    else:
                        # Predicted NO actions when there should be positive actions
                        # This is bad for recall
                        reward -= 20.0
                
                # Small bonus for correct negative predictions (much lower weight)
                negative_mask = expert_actions <= 0.5
                if np.sum(negative_mask) > 0:
                    correct_negatives = np.sum(
                        (binary_action[negative_mask] == 0) & (expert_actions[negative_mask] == 0)
                    )
                    negative_accuracy = correct_negatives / np.sum(negative_mask)
                    reward += 2.0 * negative_accuracy  # Much lower weight than positive matching
        
        # 2. ACTION SPARSITY (encourage expert-like action density)
        action_count = np.sum(action > 0.5)
        expert_action_count = np.sum(self.current_expert_actions[self.current_frame_idx] > 0.5) \
                            if self.current_frame_idx < len(self.current_expert_actions) else 1
        
        # Reward for matching expert action density
        if action_count == expert_action_count:
            reward += self.reward_weights['action_sparsity']
        elif abs(action_count - expert_action_count) == 1:
            reward += self.reward_weights['action_sparsity'] * 0.5
        elif action_count == 0:
            reward -= self.reward_weights['action_sparsity']  # Penalty for no actions
        elif action_count > 5:
            reward -= 2.0 * (action_count - 5)  # Penalty for too many actions
        
        return np.clip(reward, -30.0, 120.0)
    
    def _get_next_state(self, action: np.ndarray) -> np.ndarray:
        """Get next state using world model or direct video frames."""
        
        try:
            if self.world_model is not None:
                # Use world model to predict next state
                current_state_tensor = torch.tensor(
                    self.current_state, dtype=torch.float32, device=self.device
                )
                action_tensor = torch.tensor(
                    action, dtype=torch.float32, device=self.device
                )
                
                next_state, _, _ = self.world_model.simulate_step(
                    current_state_tensor, action_tensor
                )
                
                next_state_np = next_state.cpu().numpy().flatten()
                return self._ensure_state_shape(next_state_np)
            else:
                # Use actual next frame from video
                current_video = self.video_data[self.current_video_idx]
                if self.current_frame_idx + 1 < len(current_video['frame_embeddings']):
                    self.current_frame_idx += 1
                    return current_video['frame_embeddings'][self.current_frame_idx].astype(np.float32)
                else:
                    return self.current_state.copy()
        
        except Exception as e:
            self.logger.warning(f"State prediction failed: {e}")
            # Fallback: add small noise to current state
            noise = np.random.normal(0, 0.01, self.current_state.shape)
            return self.current_state + noise
    
    def _process_action(self, action) -> np.ndarray:
        """Process action to ensure correct format."""
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        
        action = np.array(action, dtype=np.float32).flatten()
        
        # Ensure correct size and range
        if len(action) != 100:
            padded_action = np.zeros(100, dtype=np.float32)
            if len(action) > 0:
                padded_action[:min(len(action), 100)] = action[:100]
            action = padded_action
        
        return np.clip(action, 0.0, 1.0)
    
    def _ensure_state_shape(self, state: np.ndarray) -> np.ndarray:
        """Ensure state has correct shape [1024]."""
        if len(state.shape) > 1:
            state = state.flatten()
        
        if len(state) < 1024:
            padded_state = np.zeros(1024, dtype=np.float32)
            padded_state[:len(state)] = state
            return padded_state
        elif len(state) > 1024:
            return state[:1024].astype(np.float32)
        else:
            return state.astype(np.float32)
    
    def _record_episode_end(self, info: Dict):
        """Record episode end information for debugging."""
        if 'expert_f1_score' in info:
            self.debug_info['episode_expert_f1_scores'].append(info['expert_f1_score'])
        
        self.debug_info['episode_action_densities'].append(info.get('action_count', 0))
        self.debug_info['episode_rewards'].append(self.episode_reward)
    
    def get_debug_info(self) -> Dict:
        """Get debug information for analysis."""
        if not self.debug_info['episode_rewards']:
            return {}
        
        return {
            'avg_episode_reward': np.mean(self.debug_info['episode_rewards'][-20:]),
            'avg_expert_f1': np.mean(self.debug_info['episode_expert_f1_scores'][-20:]) 
                           if self.debug_info['episode_expert_f1_scores'] else 0.0,
            'avg_action_density': np.mean(self.debug_info['episode_action_densities'][-20:]),
            'total_episodes': len(self.debug_info['episode_rewards']),
            'reward_trend': 'improving' if len(self.debug_info['episode_rewards']) > 10 and
                           np.mean(self.debug_info['episode_rewards'][-5:]) > 
                           np.mean(self.debug_info['episode_rewards'][-10:-5]) else 'stable'
        }


class SimplifiedRLTrainer:
    """
    Simplified RL trainer that focuses only on expert action matching.
    Integrates comprehensive debugging to understand training dynamics.
    """
    
    def __init__(self, config: Dict, logger, device: str = 'cuda'):
        self.config = config
        self.logger = logger
        self.device = device
        
        # Create results directory
        self.save_dir = Path(logger.log_dir) / 'simplified_rl_training'
        self.save_dir.mkdir(exist_ok=True)
        
        # Initialize debugger
        self.debugger = RLDebugger(
            save_dir=str(self.save_dir / 'debug'), 
            logger=logger, 
            config=config
        )
        
        self.logger.info("🎯 SimplifiedRLTrainer initialized")
        self.logger.info("   Focus: Expert action matching only")
        self.logger.info(f"   Results: {self.save_dir}")
    
    def train_simplified_ppo(self, world_model, train_data: List[Dict], 
                           timesteps: int = 20000) -> Dict[str, Any]:
        """
        Train PPO with simplified expert matching environment.
        """
        
        self.logger.info("🎯 Training Simplified PPO (Expert Matching Only)")
        self.logger.info("-" * 60)
        
        try:
            # Create simplified environment
            env = self._create_simplified_env(world_model, train_data)
            
            # Test environment
            self.logger.info("🔧 Testing simplified environment...")
            obs = env.reset()
            test_action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(test_action)
            self.logger.info(f"✅ Environment test: Reward={reward:.3f}, Info={info}")
            env.reset()
            
            # Create PPO model with simplified configuration
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=3e-5,      # Conservative learning rate
                n_steps=256,             # More steps for better estimates
                batch_size=64,           # Larger batch size
                n_epochs=10,             # More epochs for better learning
                gamma=0.99,              # Standard discount factor
                gae_lambda=0.95,         # Standard GAE
                clip_range=0.2,          # Standard clip range
                ent_coef=0.01,           # Lower entropy for more focused actions
                vf_coef=0.5,             # Standard value function coefficient
                max_grad_norm=0.5,       # Gradient clipping
                verbose=1,
                device='cpu',  # Use CPU for stability
                policy_kwargs={
                    'net_arch': [256, 256, 128],  # Slightly smaller network
                    'activation_fn': torch.nn.ReLU
                },
                tensorboard_log=str(self.save_dir / 'tensorboard')
            )
            
            # Create expert matching callback for detailed monitoring
            expert_callback = self.debugger.create_expert_matching_callback(
                eval_env=env,
                expert_data=train_data
            )
            
            self.logger.info(f"🚀 Training for {timesteps} timesteps with expert matching focus...")
            
            # Train with comprehensive monitoring
            model.learn(
                total_timesteps=timesteps,
                callback=expert_callback,
                tb_log_name="SimplifiedPPO_ExpertMatching",
                progress_bar=True
            )
            
            # Final evaluation
            self.logger.info("📊 Final evaluation...")
            final_results = self._evaluate_final_performance(model, train_data)
            
            # Save model
            model_path = self.save_dir / 'simplified_ppo_expert_matching.zip'
            model.save(str(model_path))
            
            # Get environment debug info
            base_env = env.envs[0].env
            env_debug_info = base_env.get_debug_info()
            
            # Compile results
            result = {
                'algorithm': 'SimplifiedPPO_ExpertMatching',
                'approach': 'Simplified expert action matching only',
                'model_path': str(model_path),
                'status': 'success',
                'training_timesteps': timesteps,
                'final_evaluation': final_results,
                'environment_debug': env_debug_info,
                'reward_focus': 'expert_f1_score_only',
                'simplifications': [
                    'Removed complex reward components',
                    'Focus only on expert action matching',
                    'F1-like reward aligned with mAP',
                    'Action sparsity matching expert patterns'
                ]
            }
            
            self.logger.info("✅ Simplified PPO training completed!")
            self.logger.info(f"📊 Final mAP: {final_results.get('mAP', 0.0):.4f}")
            self.logger.info(f"📊 Expert F1: {final_results.get('expert_f1', 0.0):.3f}")
            
            # Create comprehensive visualizations
            self._create_training_visualizations(expert_callback)
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Simplified PPO training failed: {e}")
            import traceback
            traceback.print_exc()
            return {'status': 'failed', 'error': str(e)}
    
    def _create_simplified_env(self, world_model, train_data: List[Dict]):
        """Create simplified environment wrapped for stable-baselines3."""
        
        def make_env():
            env = SimplifiedExpertMatchingEnv(
                world_model=world_model,
                video_data=train_data,
                config=self.config.get('rl_training', {}),
                logger=self.logger,
                device=self.device
            )
            return Monitor(env)
        
        return DummyVecEnv([make_env])
    
    def _evaluate_final_performance(self, model, test_data: List[Dict]) -> Dict[str, float]:
        """Evaluate final model performance with mAP calculation."""
        
        self.logger.info("📊 Evaluating final performance...")
        
        all_predictions = []
        all_expert_actions = []
        
        # Collect predictions on test data
        for video in test_data[:3]:  # Evaluate on first 3 videos
            frames = video['frame_embeddings']
            expert_actions = video['actions_binaries']
            
            for i in range(min(50, len(frames))):  # Sample frames per video
                try:
                    state = frames[i].reshape(1, -1)
                    action_pred, _ = model.predict(state, deterministic=True)
                    
                    # Process action prediction
                    action_pred = self._process_action_for_evaluation(action_pred)
                    
                    all_predictions.append(action_pred)
                    all_expert_actions.append(expert_actions[i])
                    
                except Exception as e:
                    self.logger.warning(f"Prediction failed: {e}")
                    continue
        
        if not all_predictions:
            return {'mAP': 0.0, 'expert_f1': 0.0, 'action_density': 0.0}
        
        # Calculate mAP using our evaluation metrics
        from evaluation.evaluation_metrics import calculate_comprehensive_action_metrics
        
        predictions_array = np.array(all_predictions)
        expert_array = np.array(all_expert_actions)
        
        metrics = calculate_comprehensive_action_metrics(
            predictions=predictions_array,
            ground_truth=expert_array,
            method_name="SimplifiedPPO_ExpertMatching"
        )
        
        # Calculate expert F1 score
        expert_f1_scores = []
        for i in range(len(predictions_array)):
            pred = predictions_array[i]
            expert = expert_array[i]
            
            # Calculate F1 for this sample
            positive_mask = expert > 0.5
            if np.sum(positive_mask) > 0:
                predicted_positives = np.sum(pred > 0.5)
                if predicted_positives > 0:
                    correct_positives = np.sum((pred > 0.5) & (expert > 0.5))
                    precision = correct_positives / predicted_positives
                    recall = correct_positives / np.sum(positive_mask)
                    
                    if (precision + recall) > 0:
                        f1 = 2 * (precision * recall) / (precision + recall)
                        expert_f1_scores.append(f1)
        
        avg_expert_f1 = np.mean(expert_f1_scores) if expert_f1_scores else 0.0
        avg_action_density = np.mean(np.sum(predictions_array > 0.5, axis=1))
        
        results = {
            'mAP': metrics['mAP'],
            'expert_f1': avg_expert_f1,
            'action_density': avg_action_density,
            'num_samples': len(predictions_array),
            'detailed_metrics': metrics
        }
        
        return results
    
    def _process_action_for_evaluation(self, action_pred) -> np.ndarray:
        """Process action prediction for evaluation."""
        if isinstance(action_pred, np.ndarray):
            action_pred = action_pred.flatten()
        
        # Ensure 100 dimensions
        if len(action_pred) != 100:
            padded = np.zeros(100, dtype=np.float32)
            if len(action_pred) > 0:
                padded[:min(len(action_pred), 100)] = action_pred[:100]
            action_pred = padded
        
        return np.clip(action_pred, 0.0, 1.0)
    
    def _create_training_visualizations(self, expert_callback):
        """Create comprehensive training visualizations."""
        
        self.logger.info("📊 Creating training visualizations...")
        
        # Extract training data from callback
        training_data = {
            'episode_rewards': list(expert_callback.debugger.training_metrics.get('episode_rewards', [])),
            'expert_matching_scores': list(expert_callback.expert_matching_scores),
            'mAP_during_training': list(expert_callback.mAP_scores),
            'training_steps': list(expert_callback.debugger.training_metrics.get('training_steps', [])),
        }
        
        # Create comprehensive plots
        self.debugger.plot_comprehensive_training_analysis(training_data)
        
        # Save training data
        with open(self.save_dir / 'training_data.json', 'w') as f:
            json.dump(training_data, f, indent=2, default=str)
        
        self.logger.info(f"📊 Visualizations saved to {self.save_dir}")


def run_simplified_rl_debugging(config, logger, world_model, train_data, test_data, timesteps=20000):
    """
    Main function to run simplified RL training with comprehensive debugging.
    This is designed to understand why RL can't reach supervised learning performance.
    """
    
    logger.info("🎯 SIMPLIFIED RL TRAINING WITH EXPERT MATCHING FOCUS")
    logger.info("=" * 60)
    logger.info("Goal: Understand why RL can't reach 10% mAP vs supervised baseline")
    logger.info("Approach: Focus ONLY on expert action matching + comprehensive debugging")
    
    # Initialize trainer
    trainer = SimplifiedRLTrainer(config, logger)
    
    # Step 1: Evaluate world model quality first
    logger.info("\n🌍 Step 1: Evaluating World Model Quality...")
    world_model_analysis = trainer.debugger.evaluate_world_model_quality(world_model, test_data)
    
    if world_model_analysis['summary']['world_model_quality_score'] < 0.3:
        logger.warning("⚠️ World model quality is poor - this may limit RL performance")
    
    # Step 2: Train with simplified approach
    logger.info("\n🎯 Step 2: Training Simplified RL...")
    results = trainer.train_simplified_ppo(world_model, train_data, timesteps)
    
    # Step 3: Compare with supervised baseline (if available)
    logger.info("\n📊 Step 3: Performance Analysis...")
    
    if results['status'] == 'success':
        final_mAP = results['final_evaluation']['mAP']
        expert_f1 = results['final_evaluation']['expert_f1']
        
        logger.info(f"🎯 SIMPLIFIED RL RESULTS:")
        logger.info(f"   Final mAP: {final_mAP:.4f}")
        logger.info(f"   Expert F1: {expert_f1:.3f}")
        logger.info(f"   Action Density: {results['final_evaluation']['action_density']:.1f}")
        
        # Determine if this is closer to supervised performance
        if final_mAP >= 0.05:  # 5% mAP threshold
            logger.info("✅ Simplified RL is showing reasonable performance!")
        elif final_mAP >= 0.02:  # 2% mAP threshold
            logger.info("🔶 Simplified RL is improving but still below target")
        else:
            logger.info("❌ Simplified RL is still far from supervised performance")
            logger.info("   Recommendations:")
            logger.info("     - Check action space conversion and thresholding")
            logger.info("     - Consider behavioral cloning warm-start")
            logger.info("     - Validate reward signal alignment")
    
    # Step 4: Generate comprehensive debug report
    logger.info("\n📋 Step 4: Generating Debug Report...")
    debug_report_path = trainer.debugger.save_debug_report()
    
    logger.info(f"🎯 SIMPLIFIED RL DEBUGGING COMPLETE!")
    logger.info(f"📋 Full debug report: {debug_report_path}")
    logger.info(f"📊 Visualizations: {trainer.save_dir}")
    
    return trainer, results


if __name__ == "__main__":
    print("🎯 SIMPLIFIED RL TRAINER WITH EXPERT MATCHING FOCUS")
    print("=" * 60)
    print("🔍 Purpose: Debug why RL can't reach supervised learning baseline")
    print("🎯 Focus: ONLY expert action matching (F1-like reward aligned with mAP)")
    print("📊 Features:")
    print("   ✅ Simplified reward function (expert matching + sparsity only)")
    print("   ✅ Comprehensive debugging and monitoring")
    print("   ✅ Training visualizations and analysis")
    print("   ✅ World model quality evaluation")
    print("   ✅ Action space optimization")
    print("   ✅ Performance comparison framework")
    print()
    print("🚀 This will help identify exactly where RL training fails!")
