#!/usr/bin/env python3
"""
Fixed RL Training - Continue from Trained Models
Fixes the errors and loads the best trained weights:
- IL Model: epoch 1 (48.3% mAP)
- World Model: epoch 2 (State Loss: 0.1421)

Fixes applied:
1. World model device attribute issue
2. SimplifiedExpertMatchingEnv gymnasium inheritance 
3. Proper error handling for None world models
4. Updated config loading for trained models
"""

import os
import yaml
import torch
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path
from typing import Dict, List, Any, Tuple
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Import existing components  
from models.autoregressive_il_model import AutoregressiveILModel
from models.conditional_world_model import ConditionalWorldModel
from datasets.cholect50 import load_cholect50_data
from utils.logger import SimpleLogger

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FixedSimplifiedExpertMatchingEnv(gym.Env):
    """
    FIXED: Properly inherits from gymnasium.Env
    Simplified Environment that focuses ONLY on expert action matching.
    """
    
    def __init__(self, world_model, video_data: List[Dict], config: Dict, 
                 logger=None, device: str = 'cuda'):
        super().__init__()  # FIXED: Proper gym.Env initialization
        
        self.world_model = world_model
        self.video_data = video_data
        self.config = config
        self.logger = logger
        self.device = device
        
        # Environment parameters
        self.max_episode_steps = config.get('rl_horizon', 20)
        
        # FIXED: Proper gymnasium spaces
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
        
        # Simplified reward weights - ONLY expert matching
        self.reward_weights = {
            'expert_f1': 100.0,
            'action_sparsity': 5.0,
            'completion_bonus': 2.0
        }
        
        # Debug tracking
        self.debug_info = {
            'episode_expert_f1_scores': [],
            'episode_action_densities': [],
            'episode_rewards': []
        }
        
        if self.logger:
            self.logger.info("🎯 FixedSimplifiedExpertMatchingEnv initialized")
            self.logger.info(f"   Reward weights: {self.reward_weights}")
    
    def reset(self, seed=None, options=None):
        """Reset environment with proper gymnasium interface."""
        if seed is not None:
            np.random.seed(seed)
        
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
        """Step with proper gymnasium interface (5-tuple return)."""
        self.current_step += 1
        
        # Process action
        action = self._process_action(action)
        
        # Check if episode should end
        frames_remaining = len(self.current_expert_actions) - self.current_frame_idx - 1
        terminated = self.current_step >= self.max_episode_steps
        truncated = frames_remaining <= 0
        
        if terminated or truncated:
            # Episode completion
            reward = self.reward_weights['completion_bonus']
            next_state = self.current_state.copy()
        else:
            # Get next state from video or world model
            next_state = self._get_next_state(action)
            
            # Calculate reward
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
            'method': 'fixed_simplified_expert_matching'
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
        if terminated or truncated:
            self._record_episode_end(info)
        
        # FIXED: Return 5-tuple for gymnasium
        return self.current_state.copy(), float(reward), terminated, truncated, info
    
    def _calculate_simplified_reward(self, action: np.ndarray) -> float:
        """Simplified reward focusing only on expert action matching."""
        
        reward = 0.0
        
        # Expert action matching (primary reward)
        if self.current_frame_idx < len(self.current_expert_actions):
            expert_actions = self.current_expert_actions[self.current_frame_idx]
            binary_action = (action > 0.5).astype(int)
            
            if len(expert_actions) == len(binary_action):
                # Focus on positive action prediction
                positive_mask = expert_actions > 0.5
                total_positives = np.sum(positive_mask)
                
                if total_positives > 0:
                    correct_positives = np.sum(
                        (binary_action[positive_mask] == 1) & (expert_actions[positive_mask] == 1)
                    )
                    predicted_positives = np.sum(binary_action > 0.5)
                    
                    if predicted_positives > 0:
                        precision = correct_positives / predicted_positives
                        recall = correct_positives / total_positives
                        
                        if (precision + recall) > 0:
                            f1_score = 2 * (precision * recall) / (precision + recall)
                            reward += self.reward_weights['expert_f1'] * f1_score
                    else:
                        reward -= 20.0  # Penalty for no predictions when there should be
                
                # Small bonus for correct negative predictions
                negative_mask = expert_actions <= 0.5
                if np.sum(negative_mask) > 0:
                    correct_negatives = np.sum(
                        (binary_action[negative_mask] == 0) & (expert_actions[negative_mask] == 0)
                    )
                    negative_accuracy = correct_negatives / np.sum(negative_mask)
                    reward += 2.0 * negative_accuracy
        
        # Action sparsity (encourage expert-like action density)
        action_count = np.sum(action > 0.5)
        expert_action_count = np.sum(self.current_expert_actions[self.current_frame_idx] > 0.5) \
                            if self.current_frame_idx < len(self.current_expert_actions) else 1
        
        # Reward for matching expert action density
        if action_count == expert_action_count:
            reward += self.reward_weights['action_sparsity']
        elif abs(action_count - expert_action_count) == 1:
            reward += self.reward_weights['action_sparsity'] * 0.5
        elif action_count == 0:
            reward -= self.reward_weights['action_sparsity']
        elif action_count > 5:
            reward -= 2.0 * (action_count - 5)
        
        return np.clip(reward, -30.0, 120.0)
    
    def _get_next_state(self, action: np.ndarray) -> np.ndarray:
        """Get next state using world model or direct video frames."""
        
        try:
            if self.world_model is not None:
                # FIXED: Use world model with proper device handling
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
            if self.logger:
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


class ExpertMatchingCallback(BaseCallback):
    """
    Callback for tracking expert matching during RL training.
    """
    
    def __init__(self, eval_env, expert_data: List[Dict], eval_freq: int = 500):
        super().__init__()
        self.eval_env = eval_env
        self.expert_data = expert_data
        self.eval_freq = eval_freq
        
        # Tracking
        self.episode_count = 0
        self.training_metrics = {
            'episode_rewards': [],
            'expert_matching_scores': [],
            'action_densities': [],
            'mAP_scores': []
        }
    
    def _on_step(self) -> bool:
        """Called at each training step."""
        
        # Extract episode information
        if 'infos' in self.locals:
            for info in self.locals.get('infos', []):
                if 'episode' in info:
                    self.episode_count += 1
                    ep_reward = info['episode']['r']
                    self.training_metrics['episode_rewards'].append(ep_reward)
                
                # Track step-level information
                if 'expert_match_score' in info:
                    self.training_metrics['expert_matching_scores'].append(info['expert_match_score'])
                
                if 'action_count' in info:
                    self.training_metrics['action_densities'].append(info['action_count'])
        
        # Periodic evaluation
        if self.num_timesteps % self.eval_freq == 0:
            self._evaluate_expert_matching()
        
        return True
    
    def _evaluate_expert_matching(self):
        """Evaluate how well the RL policy matches expert actions."""
        
        try:
            # Sample expert data for evaluation
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
                self.training_metrics['mAP_scores'].append(mAP)
                
                print(f"📊 Step {self.num_timesteps}: mAP={mAP:.4f}, Episodes={self.episode_count}")
        
        except Exception as e:
            print(f"Expert matching evaluation failed: {e}")
    
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
        try:
            from sklearn.metrics import average_precision_score
            
            ap_scores = []
            for action_idx in range(predictions.shape[1]):
                if np.sum(targets[:, action_idx]) > 0:  # Only calculate for present actions
                    try:
                        # Use the prediction probabilities for AP calculation
                        pred_probs = predictions[:, action_idx].astype(float)
                        ap = average_precision_score(targets[:, action_idx], pred_probs)
                        ap_scores.append(ap)
                    except:
                        ap_scores.append(0.0)
            
            return np.mean(ap_scores) if ap_scores else 0.0
        except:
            return 0.0


def load_trained_models(results_dir: str, logger, device: str = 'cuda'):
    """
    Load the best trained models from the previous run.
    
    Based on the log:
    - IL Model: epoch 1 (48.3% mAP)
    - World Model: epoch 2 (State Loss: 0.1421)
    """
    
    logger.info("📂 Loading trained models from previous run...")
    
    models = {}
    
    # Load IL model (epoch 1 - best performance)
    il_model_path = f"{results_dir}/logs/checkpoints/autoregressive_il_best_epoch_1.pt"
    if os.path.exists(il_model_path):
        try:
            il_model = AutoregressiveILModel.load_model(il_model_path, device=device)
            models['il_model'] = il_model
            logger.info(f"✅ Loaded IL model: {il_model_path}")
            logger.info("   Performance: 48.3% mAP on surgical actions")
        except Exception as e:
            logger.error(f"❌ Failed to load IL model: {e}")
    else:
        logger.error(f"❌ IL model not found: {il_model_path}")
    
    # Load World Model (epoch 2 - best performance)
    wm_model_path = f"{results_dir}/logs/checkpoints/world_model_best_epoch_2.pt"
    if os.path.exists(wm_model_path):
        try:
            world_model = ConditionalWorldModel.load_model(wm_model_path, device=device)
            
            # FIXED: Ensure world model has device attribute
            if not hasattr(world_model, 'device'):
                world_model.device = device
            
            models['world_model'] = world_model
            logger.info(f"✅ Loaded World Model: {wm_model_path}")
            logger.info("   Performance: State Loss 0.1421, Reward Loss 0.0594")
        except Exception as e:
            logger.error(f"❌ Failed to load World Model: {e}")
    else:
        logger.error(f"❌ World Model not found: {wm_model_path}")
    
    return models


def train_fixed_simplified_rl(world_model, train_data: List[Dict], config: dict, 
                             logger, timesteps: int = 30000):
    """
    Train simplified RL with all fixes applied.
    """
    
    logger.info("🎯 Training Fixed Simplified RL")
    logger.info("-" * 50)
    
    # Create fixed environment
    def make_env():
        env = FixedSimplifiedExpertMatchingEnv(
            world_model=world_model,
            video_data=train_data,
            config=config.get('rl_training', {}),
            logger=logger,
            device=DEVICE
        )
        return Monitor(env)
    
    # Create vectorized environment
    env = DummyVecEnv([make_env])
    
    # Test environment first
    logger.info("🔧 Testing fixed environment...")
    obs = env.reset()
    test_action = env.action_space.sample()
    obs, reward, done, info = env.step(test_action)
    logger.info(f"✅ Environment test successful - Reward: {reward[0]:.3f}")
    env.reset()
    
    # Create PPO model with conservative settings
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-5,           # Conservative learning rate
        n_steps=256,                  # More steps for better estimates
        batch_size=64,                # Larger batch size
        n_epochs=10,                  # More epochs for learning
        gamma=0.99,                   # Standard discount
        gae_lambda=0.95,              # Standard GAE
        clip_range=0.2,               # Standard clip
        ent_coef=0.01,                # Lower entropy for focused actions
        vf_coef=0.5,                  # Standard value function
        max_grad_norm=0.5,            # Gradient clipping
        verbose=1,
        device='cpu',  # Use CPU for stability
        policy_kwargs={
            'net_arch': [256, 256, 128],
            'activation_fn': torch.nn.ReLU
        }
    )
    
    # Create expert matching callback
    expert_callback = ExpertMatchingCallback(
        eval_env=env,
        expert_data=train_data,
        eval_freq=500
    )
    
    logger.info(f"🚀 Training for {timesteps} timesteps with expert matching focus...")
    
    # Train with monitoring
    model.learn(
        total_timesteps=timesteps,
        callback=expert_callback,
        progress_bar=True
    )
    
    # Final evaluation
    logger.info("📊 Final evaluation...")
    final_mAP = evaluate_rl_model_mAP(model, train_data[:2], logger)  # Quick eval on 2 videos
    
    # Save model
    save_dir = Path("results/fixed_rl_training")
    save_dir.mkdir(parents=True, exist_ok=True)
    model_path = save_dir / 'fixed_simplified_ppo.zip'
    model.save(str(model_path))
    
    # Get environment debug info
    base_env = env.envs[0].env
    env_debug_info = base_env.get_debug_info()
    
    result = {
        'algorithm': 'FixedSimplifiedPPO',
        'model_path': str(model_path),
        'final_mAP': final_mAP,
        'training_timesteps': timesteps,
        'environment_debug': env_debug_info,
        'training_metrics': expert_callback.training_metrics,
        'status': 'success'
    }
    
    logger.info("✅ Fixed RL training completed!")
    logger.info(f"📊 Final mAP: {final_mAP:.4f}")
    logger.info(f"📊 Training metrics: {len(expert_callback.training_metrics['episode_rewards'])} episodes")
    
    return result


def evaluate_rl_model_mAP(rl_model, test_data: List[Dict], logger) -> float:
    """Quick mAP evaluation of RL model."""
    
    from sklearn.metrics import average_precision_score
    
    all_predictions = []
    all_expert_actions = []
    
    for video in test_data:
        frames = video['frame_embeddings']
        expert_actions = video['actions_binaries']
        
        # Sample frames for quick evaluation
        sample_indices = np.linspace(0, len(frames)-1, min(30, len(frames)), dtype=int)
        
        for idx in sample_indices:
            try:
                state = frames[idx].reshape(1, -1)
                action_pred, _ = rl_model.predict(state, deterministic=True)
                
                # Process action prediction
                if isinstance(action_pred, np.ndarray):
                    action_pred = action_pred.flatten()
                
                if len(action_pred) != 100:
                    padded = np.zeros(100)
                    if len(action_pred) > 0:
                        padded[:min(len(action_pred), 100)] = action_pred[:100]
                    action_pred = padded
                
                action_pred = np.clip(action_pred, 0.0, 1.0)
                
                all_predictions.append(action_pred)
                all_expert_actions.append(expert_actions[idx])
                
            except Exception as e:
                logger.warning(f"Prediction failed: {e}")
                continue
    
    if not all_predictions:
        return 0.0
    
    # Calculate mAP
    predictions = np.array(all_predictions)
    expert_actions = np.array(all_expert_actions)
    
    ap_scores = []
    for action_idx in range(100):
        if np.sum(expert_actions[:, action_idx]) > 0:
            try:
                ap = average_precision_score(
                    expert_actions[:, action_idx], 
                    predictions[:, action_idx]
                )
                ap_scores.append(ap)
            except:
                ap_scores.append(0.0)
    
    return np.mean(ap_scores) if ap_scores else 0.0


def main():
    """Main function to continue RL training from trained models."""
    
    print("🔧 FIXED RL TRAINING - CONTINUE FROM TRAINED MODELS")
    print("=" * 70)
    print("🎯 Loading best trained weights:")
    print("   - IL Model: epoch 1 (48.3% mAP)")
    print("   - World Model: epoch 2 (State Loss: 0.1421)")
    print("🔧 All errors fixed:")
    print("   ✅ World model device attribute")
    print("   ✅ SimplifiedExpertMatchingEnv gymnasium inheritance")
    print("   ✅ Proper error handling")
    print()
    
    # Configuration
    config = {
        'experiment': {
            'train': {'max_videos': 40},
            'test': {'max_videos': 10, 'test_on_train': False}
        },
        'data': {
            'context_length': 20,
            'padding_value': 0.0,
            'paths': {
                'data_dir': "/home/maxboels/datasets/CholecT50",
                'fold': 0,
                'metadata_file': "embeddings_f0_swin_bas_129_phase_complet_phase_transit_prog_prob_action_risk_glob_outcome.csv"
            }
        },
        'rl_training': {
            'rl_horizon': 20,
            'timesteps': 30000,
            'reward_weights': {
                'expert_f1': 100.0,
                'action_sparsity': 5.0,
                'completion_bonus': 2.0
            }
        }
    }
    
    # Initialize logger
    results_dir = "results/fixed_rl_continue"
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    logger = SimpleLogger(log_dir=results_dir, name="FixedRL_Continue")
    
    try:
        # Step 1: Load data
        print("📂 Step 1: Loading data...")
        from datasets.cholect50 import load_cholect50_data
        
        train_data = load_cholect50_data(
            config, logger, split='train', max_videos=40
        )
        test_data = load_cholect50_data(
            config, logger, split='test', max_videos=10
        )
        
        print(f"✅ Data loaded: {len(train_data)} train, {len(test_data)} test videos")
        
        # Step 2: Load trained models
        print("\n📂 Step 2: Loading trained models...")
        previous_results_dir = "results/debug_rl_2025-06-17_17-59-26"  # From your log
        models = load_trained_models(previous_results_dir, logger, DEVICE)
        
        if 'world_model' not in models:
            print("⚠️ World model not loaded, proceeding with direct video RL...")
            world_model = None
        else:
            world_model = models['world_model']
            print(f"✅ World model loaded successfully")
        
        # Step 3: Train fixed simplified RL
        print("\n🎯 Step 3: Training fixed simplified RL...")
        rl_results = train_fixed_simplified_rl(
            world_model=world_model,
            train_data=train_data,
            config=config,
            logger=logger,
            timesteps=config['rl_training']['timesteps']
        )
        
        # Step 4: Compare with supervised baseline
        print("\n📊 Step 4: Performance comparison...")
        supervised_mAP = 0.4833  # From your log
        rl_mAP = rl_results['final_mAP']
        
        print(f"🎓 Supervised IL Baseline: {supervised_mAP:.4f} mAP (48.3%)")
        print(f"🎯 Fixed RL Performance: {rl_mAP:.4f} mAP ({rl_mAP:.1%})")
        print(f"📊 Performance Gap: {supervised_mAP - rl_mAP:.4f}")
        print(f"📊 RL vs Supervised Ratio: {rl_mAP / supervised_mAP:.1%}")
        
        if rl_mAP / supervised_mAP > 0.5:
            print("✅ EXCELLENT: RL is competitive with supervised learning!")
        elif rl_mAP / supervised_mAP > 0.2:
            print("🔶 GOOD: RL is learning, continue optimization")
        elif rl_mAP > 0.05:
            print("⚠️ MODERATE: RL shows learning but needs improvement")
        else:
            print("❌ CONCERNING: RL still not learning effectively")
        
        # Save results
        results_summary = {
            'supervised_baseline_mAP': supervised_mAP,
            'fixed_rl_mAP': rl_mAP,
            'performance_gap': supervised_mAP - rl_mAP,
            'performance_ratio': rl_mAP / supervised_mAP,
            'rl_results': rl_results,
            'fixes_applied': [
                'World model device attribute fixed',
                'SimplifiedExpertMatchingEnv proper gymnasium inheritance',
                'Proper error handling for None world models',
                'Updated environment step() to return 5-tuple'
            ]
        }
        
        import json
        with open(f"{results_dir}/fixed_rl_results.json", 'w') as f:
            json.dump(results_summary, f, indent=2, default=str)
        
        print(f"\n🎉 FIXED RL TRAINING COMPLETED!")
        print(f"📁 Results saved to: {results_dir}")
        print(f"📊 Check fixed_rl_results.json for detailed analysis")
        
    except Exception as e:
        print(f"\n❌ Fixed RL training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
