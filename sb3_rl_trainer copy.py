#!/usr/bin/env python3
"""
Stable-Baselines3 RL Trainer for Surgical Action Prediction
Much more reliable than custom implementations!
"""

import torch
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Any, Optional
import logging
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

# Stable-Baselines3 imports
from stable_baselines3 import PPO, SAC, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

class SurgicalActionEnv(gym.Env):
    """
    Gymnasium environment wrapper for surgical action prediction using world model.
    Compatible with Stable-Baselines3.
    """
    
    def __init__(self, world_model, video_data: List[Dict], config: Dict, device: str = 'cuda'):
        super().__init__()
        
        self.world_model = world_model
        self.video_data = video_data
        self.config = config
        self.device = device
        
        # Environment parameters
        self.max_episode_steps = config.get('rl_horizon', 50)
        self.current_step = 0
        self.current_video_idx = 0
        self.current_frame_idx = 0
        
        # Action and observation spaces
        self.action_space = spaces.MultiBinary(100)  # 100 binary surgical actions
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(1024,),  # State embedding dimension
            dtype=np.float32
        )
        
        # Current state
        self.current_state = None
        self.episode_reward = 0
        
        # Reward configuration
        self.reward_config = config.get('reward_weights', {
            'action_accuracy': 1.0,
            'state_consistency': 0.5,
            'progression_bonus': 0.2
        })
    
    def reset(self, seed=None, options=None):
        """Reset environment for new episode."""
        super().reset(seed=seed)
        
        # Select random video and starting frame
        self.current_video_idx = np.random.randint(0, len(self.video_data))
        video = self.video_data[self.current_video_idx]
        
        max_start_frame = len(video['frame_embeddings']) - self.max_episode_steps - 1
        self.current_frame_idx = np.random.randint(0, max(1, max_start_frame))
        
        # Get initial state
        self.current_state = video['frame_embeddings'][self.current_frame_idx].astype(np.float32)
        self.current_step = 0
        self.episode_reward = 0
        
        return self.current_state, {}
    
    def step(self, action):
        """Take environment step."""
        self.current_step += 1
        
        # Get current video
        video = self.video_data[self.current_video_idx]
        
        # Check if episode is done
        done = (
            self.current_step >= self.max_episode_steps or 
            self.current_frame_idx + 1 >= len(video['frame_embeddings'])
        )
        
        if done:
            reward = 0.0
            next_state = self.current_state
        else:
            # Move to next frame
            self.current_frame_idx += 1
            next_state = video['frame_embeddings'][self.current_frame_idx].astype(np.float32)
            
            # Calculate reward based on action accuracy
            if self.current_frame_idx < len(video['actions_binaries']):
                ground_truth_actions = video['actions_binaries'][self.current_frame_idx]
                reward = self._calculate_reward(action, ground_truth_actions)
            else:
                reward = 0.0
        
        self.current_state = next_state
        self.episode_reward += reward
        
        info = {
            'episode_reward': self.episode_reward,
            'video_id': video['video_id'],
            'frame_idx': self.current_frame_idx
        }
        
        return next_state, reward, done, False, info
    
    def _calculate_reward(self, predicted_actions: np.ndarray, ground_truth_actions: np.ndarray) -> float:
        """Calculate reward based on action prediction accuracy."""
        
        # Convert to binary predictions
        pred_binary = (predicted_actions > 0.5).astype(int)
        gt_binary = ground_truth_actions.astype(int)
        
        # Action accuracy reward
        correct_predictions = np.sum(pred_binary == gt_binary)
        total_actions = len(gt_binary)
        accuracy = correct_predictions / total_actions
        
        # Focus on positive actions (more important)
        positive_actions = np.sum(gt_binary)
        if positive_actions > 0:
            positive_correct = np.sum((pred_binary == gt_binary) & (gt_binary == 1))
            positive_accuracy = positive_correct / positive_actions
            
            # Weighted reward favoring positive action accuracy
            reward = (
                0.5 * accuracy + 
                0.5 * positive_accuracy + 
                0.1 * positive_correct  # Bonus for each correct positive
            )
        else:
            reward = accuracy
        
        return float(reward)


class SB3Trainer:
    """
    Stable-Baselines3 trainer for surgical action prediction.
    Much more reliable than custom implementations!
    """
    
    def __init__(self, world_model, config: Dict, logger, device: str = 'cuda'):
        self.world_model = world_model
        self.config = config
        self.logger = logger
        self.device = device
        
        # Create results directory
        self.save_dir = Path(logger.log_dir) / 'sb3_rl_training'
        self.save_dir.mkdir(exist_ok=True)
        
        # RL configuration
        self.rl_config = config.get('rl_training', {})
    
    def create_env(self, train_data: List[Dict], n_envs: int = 1):
        """Create vectorized environment for training."""
        
        def make_env():
            env = SurgicalActionEnv(
                world_model=self.world_model,
                video_data=train_data,
                config=self.rl_config,
                device=self.device
            )
            env = Monitor(env)  # For logging
            return env
        
        if n_envs == 1:
            return DummyVecEnv([make_env])
        else:
            return SubprocVecEnv([make_env for _ in range(n_envs)])
    
    def train_ppo(self, train_data: List[Dict], timesteps: int = 50000) -> Dict[str, Any]:
        """Train PPO agent using Stable-Baselines3."""
        
        print("🤖 Training PPO with Stable-Baselines3")
        print("-" * 50)
        
        try:
            # Create environment
            env = self.create_env(train_data, n_envs=1)
            
            # PPO configuration
            ppo_config = self.rl_config.get('ppo', {})
            
            # Create PPO model
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=ppo_config.get('learning_rate', 3e-4),
                n_steps=ppo_config.get('n_steps', 2048),
                batch_size=ppo_config.get('batch_size', 64),
                n_epochs=ppo_config.get('n_epochs', 10),
                gamma=ppo_config.get('gamma', 0.99),
                gae_lambda=ppo_config.get('gae_lambda', 0.95),
                clip_range=ppo_config.get('clip_range', 0.2),
                ent_coef=ppo_config.get('entropy_coef', 0.01),
                vf_coef=ppo_config.get('value_coef', 0.5),
                verbose=1,
                device=self.device,
                tensorboard_log=str(self.save_dir / 'tensorboard_logs')
            )
            
            # Training callback
            eval_callback = EvalCallback(
                env,
                best_model_save_path=str(self.save_dir / 'ppo_best'),
                log_path=str(self.save_dir / 'ppo_logs'),
                eval_freq=max(timesteps // 10, 1000),
                deterministic=True,
                render=False
            )
            
            # Train the model
            print(f"Training PPO for {timesteps} timesteps...")
            model.learn(
                total_timesteps=timesteps,
                callback=eval_callback,
                tb_log_name="PPO",
                progress_bar=True
            )
            
            # Save final model
            model_path = self.save_dir / 'ppo_final_model.zip'
            model.save(str(model_path))
            
            # Evaluate trained model
            print("Evaluating trained PPO model...")
            mean_reward, std_reward = evaluate_policy(
                model, env, n_eval_episodes=20, deterministic=True
            )
            
            results = {
                'algorithm': 'PPO',
                'library': 'stable-baselines3',
                'training_timesteps': timesteps,
                'mean_reward': float(mean_reward),
                'std_reward': float(std_reward),
                'model_path': str(model_path),
                'best_model_path': str(self.save_dir / 'ppo_best' / 'best_model.zip'),
                'status': 'success'
            }
            
            self.logger.info(f"PPO training completed: {mean_reward:.3f} ± {std_reward:.3f}")
            
            return results
            
        except Exception as e:
            error_msg = f"PPO training failed: {str(e)}"
            self.logger.error(error_msg)
            return {
                'algorithm': 'PPO',
                'library': 'stable-baselines3',
                'status': 'failed',
                'error': error_msg
            }
    
    def train_sac(self, train_data: List[Dict], timesteps: int = 50000) -> Dict[str, Any]:
        """Train SAC agent using Stable-Baselines3."""
        
        print("🤖 Training SAC with Stable-Baselines3")
        print("-" * 50)
        
        try:
            # Create environment
            env = self.create_env(train_data, n_envs=1)
            
            # SAC configuration
            sac_config = self.rl_config.get('sac', {})
            
            # Create SAC model
            model = SAC(
                "MlpPolicy",
                env,
                learning_rate=sac_config.get('learning_rate', 3e-4),
                buffer_size=sac_config.get('buffer_size', 100000),
                learning_starts=sac_config.get('learning_starts', 1000),
                batch_size=sac_config.get('batch_size', 256),
                tau=sac_config.get('tau', 0.005),
                gamma=sac_config.get('gamma', 0.99),
                train_freq=1,
                gradient_steps=1,
                verbose=1,
                device=self.device,
                tensorboard_log=str(self.save_dir / 'tensorboard_logs')
            )
            
            # Training callback
            eval_callback = EvalCallback(
                env,
                best_model_save_path=str(self.save_dir / 'sac_best'),
                log_path=str(self.save_dir / 'sac_logs'),
                eval_freq=max(timesteps // 10, 1000),
                deterministic=True,
                render=False
            )
            
            # Train the model
            print(f"Training SAC for {timesteps} timesteps...")
            model.learn(
                total_timesteps=timesteps,
                callback=eval_callback,
                tb_log_name="SAC",
                progress_bar=True
            )
            
            # Save final model
            model_path = self.save_dir / 'sac_final_model.zip'
            model.save(str(model_path))
            
            # Evaluate trained model
            print("Evaluating trained SAC model...")
            mean_reward, std_reward = evaluate_policy(
                model, env, n_eval_episodes=20, deterministic=True
            )
            
            results = {
                'algorithm': 'SAC',
                'library': 'stable-baselines3',
                'training_timesteps': timesteps,
                'mean_reward': float(mean_reward),
                'std_reward': float(std_reward),
                'model_path': str(model_path),
                'best_model_path': str(self.save_dir / 'sac_best' / 'best_model.zip'),
                'status': 'success'
            }
            
            self.logger.info(f"SAC training completed: {mean_reward:.3f} ± {std_reward:.3f}")
            
            return results
            
        except Exception as e:
            error_msg = f"SAC training failed: {str(e)}"
            self.logger.error(error_msg)
            return {
                'algorithm': 'SAC',
                'library': 'stable-baselines3',
                'status': 'failed',
                'error': error_msg
            }
    
    def train_a2c(self, train_data: List[Dict], timesteps: int = 50000) -> Dict[str, Any]:
        """Train A2C agent using Stable-Baselines3."""
        
        print("🤖 Training A2C with Stable-Baselines3")
        print("-" * 50)
        
        try:
            # Create environment
            env = self.create_env(train_data, n_envs=4)  # A2C benefits from multiple envs
            
            # Create A2C model
            model = A2C(
                "MlpPolicy",
                env,
                learning_rate=3e-4,
                n_steps=5,
                gamma=0.99,
                gae_lambda=1.0,
                ent_coef=0.01,
                vf_coef=0.25,
                verbose=1,
                device=self.device,
                tensorboard_log=str(self.save_dir / 'tensorboard_logs')
            )
            
            # Training callback
            eval_env = self.create_env(train_data, n_envs=1)
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=str(self.save_dir / 'a2c_best'),
                log_path=str(self.save_dir / 'a2c_logs'),
                eval_freq=max(timesteps // 10, 1000),
                deterministic=True,
                render=False
            )
            
            # Train the model
            print(f"Training A2C for {timesteps} timesteps...")
            model.learn(
                total_timesteps=timesteps,
                callback=eval_callback,
                tb_log_name="A2C",
                progress_bar=True
            )
            
            # Save final model
            model_path = self.save_dir / 'a2c_final_model.zip'
            model.save(str(model_path))
            
            # Evaluate trained model
            print("Evaluating trained A2C model...")
            mean_reward, std_reward = evaluate_policy(
                model, eval_env, n_eval_episodes=20, deterministic=True
            )
            
            results = {
                'algorithm': 'A2C',
                'library': 'stable-baselines3',
                'training_timesteps': timesteps,
                'mean_reward': float(mean_reward),
                'std_reward': float(std_reward),
                'model_path': str(model_path),
                'best_model_path': str(self.save_dir / 'a2c_best' / 'best_model.zip'),
                'status': 'success'
            }
            
            self.logger.info(f"A2C training completed: {mean_reward:.3f} ± {std_reward:.3f}")
            
            return results
            
        except Exception as e:
            error_msg = f"A2C training failed: {str(e)}"
            self.logger.error(error_msg)
            return {
                'algorithm': 'A2C',
                'library': 'stable-baselines3',
                'status': 'failed',
                'error': error_msg
            }


def run_sb3_rl_training(config_path: str = 'config.yaml'):
    """Main function to run SB3 RL training."""
    
    # Load configuration  
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup logging
    from utils.logger import SimpleLogger
    logger = SimpleLogger(log_dir="logs", name="sb3_rl_training")
    
    print("🚀 Starting Stable-Baselines3 RL Training")
    print("=" * 60)
    print("✅ Using battle-tested, reliable RL implementations!")
    print("✅ No more tensor conversion errors!")
    print("✅ Standard research-grade algorithms!")
    print("=" * 60)
    
    # Load world model
    from models.dual_world_model import DualWorldModel
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Find your latest supervised model
    model_path = "logs/2025-05-28_16-16-23/checkpoints/supervised_best_epoch_2.pt"
    if not Path(model_path).exists():
        print(f"❌ Supervised model not found at {model_path}")
        print("Please update the path to your trained IL model")
        return None
    
    world_model = DualWorldModel.load_model(model_path, device)
    logger.info(f"Loaded world model from {model_path}")
    
    # Load training data
    from datasets.cholect50 import load_cholect50_data
    train_data = load_cholect50_data(config, logger, split='train', max_videos=1)
    
    # Create SB3 trainer
    trainer = SB3Trainer(world_model, config, logger, device)
    
    # Train algorithms
    results = {}
    
    algorithms = ['ppo', 'sac', 'a2c']  # All standard RL algorithms
    timesteps = 10000  # Reduced for faster testing
    
    for algorithm in algorithms:
        print(f"\n{'='*60}")
        print(f"Training {algorithm.upper()} with Stable-Baselines3")
        print(f"{'='*60}")
        
        if algorithm == 'ppo':
            results['ppo'] = trainer.train_ppo(train_data, timesteps=timesteps)
        elif algorithm == 'sac':
            results['sac'] = trainer.train_sac(train_data, timesteps=timesteps)
        elif algorithm == 'a2c':
            results['a2c'] = trainer.train_a2c(train_data, timesteps=timesteps)
    
    # Save results
    results_path = trainer.save_dir / 'sb3_training_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"SB3 RL training completed. Results saved to {results_path}")
    
    print(f"\n🎉 Stable-Baselines3 RL Training Complete!")
    print("=" * 60)
    
    for algorithm, result in results.items():
        status = result.get('status', 'unknown')
        if status == 'success':
            mean_reward = result.get('mean_reward', 0)
            std_reward = result.get('std_reward', 0)
            print(f"✅ {algorithm.upper()}: {mean_reward:.3f} ± {std_reward:.3f} reward")
        else:
            print(f"❌ {algorithm.upper()}: Failed - {result.get('error', 'Unknown error')}")
    
    print(f"\n📁 Results saved to: {trainer.save_dir}")
    print("📊 Tensorboard logs available for detailed analysis")
    
    return results


if __name__ == "__main__":
    import yaml
    results = run_sb3_rl_training()
