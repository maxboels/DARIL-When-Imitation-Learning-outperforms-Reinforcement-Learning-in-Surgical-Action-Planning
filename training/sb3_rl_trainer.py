#!/usr/bin/env python3
"""
SB3 Trainer - Action space issue resolved
"""

import torch
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Any, Optional
import logging
from datetime import datetime
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

# Stable-Baselines3 imports
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

class DebugCallback(BaseCallback):
    """Debug callback to monitor training progress."""
    
    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.step_count = 0
        self.episode_count = 0
        
    def _on_step(self) -> bool:
        self.step_count += 1
        
        # Log every 1000 steps
        if self.step_count % 1000 == 0:
            print(f"Training step {self.step_count}")
            
        return True
    
    def _on_rollout_end(self) -> None:
        self.episode_count += 1
        if self.episode_count % 10 == 0:
            print(f"Completed {self.episode_count} episodes")


class SurgicalActionEnv(gym.Env):
    """
    FINAL FIXED Gymnasium environment - action space issue resolved.
    """
    
    def __init__(self, world_model, video_data: List[Dict], config: Dict, device: str = 'cuda'):
        super().__init__()
        
        print(f"🔧 Initializing FINAL FIXED environment with {len(video_data)} videos")
        
        self.world_model = world_model
        self.video_data = video_data
        self.config = config
        self.device = device
        
        # FIXED: Ensure videos have sufficient frames
        self._validate_video_data()
        
        # Environment parameters with debugging
        self.max_episode_steps = config.get('rl_horizon', 50)
        print(f"🔧 Max episode steps: {self.max_episode_steps}")
        
        # Current episode state
        self.current_step = 0
        self.current_video_idx = 0
        self.current_frame_idx = 0
        self.current_phase = 0
        self.current_state = None
        self.episode_reward = 0.0
        
        # FIXED: Use Box action space instead of MultiBinary for SB3 compatibility
        # This avoids the MultiBinary sampling issues
        self.action_space = spaces.Box(
            low=0, high=1, shape=(100,), dtype=np.float32
        )
        print(f"🔧 Action space: {self.action_space}")
        
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(1024,),
            dtype=np.float32
        )
        print(f"🔧 Observation space: {self.observation_space}")
        
        # FIXED: Simple but effective reward configuration
        self.reward_weights = {
            'action_accuracy': 1.0,
            'phase_bonus': 2.0,
            'completion_bonus': 5.0
        }
        
        # Episode statistics for debugging
        self.episode_lengths = []
        self.episode_rewards = []
        
        print("✅ FINAL FIXED Environment initialized successfully")

    def _validate_video_data(self):
        """Validate that video data has sufficient frames."""
        min_frames_required = 100  # Minimum frames needed
        
        valid_videos = []
        for video in self.video_data:
            if len(video['frame_embeddings']) >= min_frames_required:
                valid_videos.append(video)
            else:
                print(f"⚠️ Skipping video {video.get('video_id', 'unknown')} - only {len(video['frame_embeddings'])} frames")
        
        if not valid_videos:
            print("❌ No videos with sufficient frames found!")
            raise ValueError("No valid videos found for training")
        
        self.video_data = valid_videos
        print(f"✅ Using {len(self.video_data)} valid videos")

    def reset(self, seed=None, options=None):
        """FIXED reset with comprehensive debugging."""
        super().reset(seed=seed)
        
        # Select random video
        self.current_video_idx = np.random.randint(0, len(self.video_data))
        video = self.video_data[self.current_video_idx]
        
        print(f"🔄 Resetting environment - Video {self.current_video_idx}: {video.get('video_id', 'unknown')}")
        print(f"🔄 Video has {len(video['frame_embeddings'])} frames")
        
        # FIXED: Ensure we have enough frames for full episode
        available_frames = len(video['frame_embeddings'])
        max_start_frame = max(0, available_frames - self.max_episode_steps - 10)  # Leave buffer
        
        if max_start_frame <= 0:
            print(f"⚠️ Video too short, using from frame 0")
            self.current_frame_idx = 0
        else:
            self.current_frame_idx = np.random.randint(0, max_start_frame)
        
        print(f"🔄 Starting from frame {self.current_frame_idx}/{available_frames}")
        
        # Get initial state
        self.current_state = video['frame_embeddings'][self.current_frame_idx].astype(np.float32)
        
        # Reset counters
        self.current_step = 0
        self.episode_reward = 0.0
        self.current_phase = 0
        
        # FIXED: Ensure state has correct shape
        if self.current_state.shape[0] != 1024:
            print(f"⚠️ State shape issue: {self.current_state.shape}, reshaping...")
            self.current_state = self.current_state.flatten()[:1024]
            if len(self.current_state) < 1024:
                self.current_state = np.pad(self.current_state, (0, 1024 - len(self.current_state)))
        
        print(f"✅ Reset complete - Initial state shape: {self.current_state.shape}")
        
        return self.current_state.copy(), {}

    def step(self, action):
        """FINAL FIXED step function with corrected action handling."""
        self.current_step += 1
        
        # # Debug action for first few steps
        # if self.current_step <= 3:
        #     print(f"🔧 Step {self.current_step}: Action type: {type(action)}, shape: {getattr(action, 'shape', 'no shape')}, dtype: {getattr(action, 'dtype', 'no dtype')}")
        
        # FIXED: Robust action handling for Box action space
        try:
            # Ensure action is numpy array
            if isinstance(action, (list, tuple)):
                action = np.array(action, dtype=np.float32)
            elif isinstance(action, torch.Tensor):
                action = action.cpu().numpy().astype(np.float32)
            elif not isinstance(action, np.ndarray):
                # Handle single scalar case
                if np.isscalar(action):
                    action = np.array([action], dtype=np.float32)
                else:
                    action = np.array(action, dtype=np.float32)
            else:
                action = action.astype(np.float32)
            
            # FIXED: Handle different action shapes
            if action.ndim == 0:  # Scalar
                # Create array of zeros with one random active action
                binary_action = np.zeros(100, dtype=int)
                if 0 <= int(action) < 100:
                    binary_action[int(action)] = 1
            elif action.ndim == 1:
                if len(action) == 1:  # Single element array
                    binary_action = np.zeros(100, dtype=int)
                    if 0 <= int(action[0]) < 100:
                        binary_action[int(action[0])] = 1
                elif len(action) == 100:  # Full action vector
                    binary_action = (action > 0.5).astype(int)
                else:  # Wrong size, pad or truncate
                    if len(action) < 100:
                        binary_action = np.zeros(100, dtype=int)
                        binary_action[:len(action)] = (action > 0.5).astype(int)
                    else:
                        binary_action = (action[:100] > 0.5).astype(int)
            else:  # Multi-dimensional, flatten
                action_flat = action.flatten()
                if len(action_flat) >= 100:
                    binary_action = (action_flat[:100] > 0.5).astype(int)
                else:
                    binary_action = np.zeros(100, dtype=int)
                    binary_action[:len(action_flat)] = (action_flat > 0.5).astype(int)
            
            # if self.current_step <= 3:
            #     print(f"🔧 Step {self.current_step}: Processed to binary_action shape: {binary_action.shape}, sum: {np.sum(binary_action)}")
        
        except Exception as e:
            print(f"❌ Error processing action: {e}")
            # Fallback: random binary action
            binary_action = np.random.randint(0, 2, size=100)
        
        # Get current video
        video = self.video_data[self.current_video_idx]
        
        # FIXED: Check termination conditions properly
        frames_remaining = len(video['frame_embeddings']) - self.current_frame_idx - 1
        step_limit_reached = self.current_step >= self.max_episode_steps
        
        done = step_limit_reached or frames_remaining <= 0
        
        # if self.current_step <= 3:
        #     print(f"🔧 Step {self.current_step}: Frames remaining: {frames_remaining}, Step limit: {step_limit_reached}")
        
        if done:
            reward = 0.0
            next_state = self.current_state.copy()
            
            # Episode completion bonus
            if step_limit_reached:
                reward += self.reward_weights['completion_bonus']
                # if self.current_step <= 3:
                #     print(f"🔧 Episode completed - giving completion bonus: {reward}")
        else:
            # Move to next frame
            self.current_frame_idx += 1
            
            # FIXED: Safe frame access with bounds checking
            if self.current_frame_idx < len(video['frame_embeddings']):
                next_state = video['frame_embeddings'][self.current_frame_idx].astype(np.float32)
                
                # FIXED: Ensure consistent state shape
                if next_state.shape[0] != 1024:
                    next_state = next_state.flatten()[:1024]
                    if len(next_state) < 1024:
                        next_state = np.pad(next_state, (0, 1024 - len(next_state)))
            else:
                # Use current state if we somehow exceed bounds
                next_state = self.current_state.copy()
                done = True
            
            # Calculate reward
            reward = self._calculate_reward_fixed(binary_action, video)
            
            # if self.current_step <= 3:
            #     print(f"🔧 Step {self.current_step}: Calculated reward: {reward}")
        
        # Update state
        self.current_state = next_state.copy()
        self.episode_reward += reward
        
        # Enhanced info for debugging
        info = {
            'step': self.current_step,
            'video_id': video.get('video_id', 'unknown'),
            'frame_idx': self.current_frame_idx,
            'episode_reward': self.episode_reward,
            'frames_remaining': frames_remaining,
            'binary_action_sum': int(np.sum(binary_action))
        }
        
        # Log episode completion
        if done:
            self.episode_lengths.append(self.current_step)
            self.episode_rewards.append(self.episode_reward)
            
            print(f"📊 Episode complete - Length: {self.current_step}, Reward: {self.episode_reward:.3f}")
            
            # Keep only last 100 episodes for statistics
            if len(self.episode_lengths) > 100:
                self.episode_lengths = self.episode_lengths[-100:]
                self.episode_rewards = self.episode_rewards[-100:]
        
        return self.current_state.copy(), reward, done, False, info

    def _calculate_reward_fixed(self, predicted_actions: np.ndarray, video: Dict) -> float:
        """FIXED reward calculation - simple but effective."""
        
        reward = 0.0
        
        # 1. Base reward for taking actions (encourages exploration)
        action_count = np.sum(predicted_actions)
        if 1 <= action_count <= 5:  # Reward reasonable action counts
            reward += self.reward_weights['action_accuracy'] * 0.5
        
        # 2. Action accuracy reward (if ground truth available)
        if (self.current_frame_idx < len(video.get('actions_binaries', [])) and 
            len(video['actions_binaries']) > 0):
            
            ground_truth = video['actions_binaries'][self.current_frame_idx]
            
            # Simple accuracy
            if len(ground_truth) == len(predicted_actions):
                accuracy = np.mean(predicted_actions == ground_truth)
                reward += self.reward_weights['action_accuracy'] * accuracy
            
            # Bonus for predicting any true positive actions
            if np.sum(ground_truth) > 0:
                true_positives = np.sum((predicted_actions == 1) & (ground_truth == 1))
                if true_positives > 0:
                    reward += self.reward_weights['phase_bonus'] * (true_positives / np.sum(ground_truth))
        
        # 3. Phase progression bonus (if available)
        if ('next_rewards' in video and 
            '_r_phase_progression' in video['next_rewards'] and
            self.current_frame_idx < len(video['next_rewards']['_r_phase_progression'])):
            
            phase_progress = video['next_rewards']['_r_phase_progression'][self.current_frame_idx]
            if phase_progress > 0:
                reward += self.reward_weights['phase_bonus'] * phase_progress
        
        # 4. Ensure reward is in reasonable range
        reward = np.clip(reward, -5.0, 10.0)
        
        return float(reward)

    def get_episode_stats(self):
        """Get episode statistics for debugging."""
        if not self.episode_lengths:
            return {"avg_length": 0, "avg_reward": 0, "episodes": 0}
        
        return {
            "avg_length": np.mean(self.episode_lengths),
            "avg_reward": np.mean(self.episode_rewards),
            "episodes": len(self.episode_lengths),
            "last_length": self.episode_lengths[-1],
            "last_reward": self.episode_rewards[-1]
        }


class SB3Trainer:
    """
    SB3 trainer with action space issues resolved.
    """
    
    def __init__(self, world_model, config: Dict, logger, device: str = 'cuda'):
        self.world_model = world_model
        self.config = config
        self.logger = logger
        self.device = device
        
        # Create results directory
        self.save_dir = Path(logger.log_dir) / 'rl_training'
        self.save_dir.mkdir(exist_ok=True)
        
        # RL configuration
        self.rl_config = config.get('rl_training', {})
        
        print(f"🔧 FINAL FIXED Trainer initialized - Save dir: {self.save_dir}")

    def create_env(self, train_data: List[Dict], n_envs: int = 1):
        """Create vectorized environment for training with debugging."""
        
        print(f"🔧 Creating FINAL FIXED environment with {len(train_data)} videos")
        
        def make_env():
            try:
                env = SurgicalActionEnv(
                    world_model=self.world_model,
                    video_data=train_data,
                    config=self.rl_config,
                    device=self.device
                )
                env = Monitor(env)  # For logging
                return env
            except Exception as e:
                print(f"❌ Error creating environment: {e}")
                raise
        
        try:
            if n_envs == 1:
                return DummyVecEnv([make_env])
            else:
                return SubprocVecEnv([make_env for _ in range(n_envs)])
        except Exception as e:
            print(f"❌ Error creating vectorized environment: {e}")
            raise

    def train_ppo_final(self, train_data: List[Dict], timesteps: int = 10000) -> Dict[str, Any]:
        """FINAL FIXED PPO training with all issues resolved."""
        
        print("🤖 Training PPO (FINAL FIXED VERSION)")
        print("-" * 50)
        
        try:
            # Create environment with error handling
            print("🔧 Creating FINAL FIXED environment...")
            env = self.create_env(train_data, n_envs=1)
            
            # Test environment thoroughly
            print("🔧 Testing FINAL FIXED environment reset...")
            obs = env.reset()
            print(f"✅ Environment reset successful - Obs shape: {obs.shape}")
            
            # Test environment step with corrected action space
            print("🔧 Testing FINAL FIXED environment step...")
            test_action = env.action_space.sample()
            print(f"🔧 Test action type: {type(test_action)}, shape: {test_action.shape}")
            obs, reward, done, info = env.step(test_action)
            print(f"✅ Environment step successful - Reward: {reward}")
            
            # Reset after test
            env.reset()
            
            # Create PPO with debugging
            print("🔧 Creating FINAL FIXED PPO model...")
            model = PPO(
                "MlpPolicy",  # Use MlpPolicy for Box action space
                env,
                learning_rate=3e-4,
                n_steps=256,  # REDUCED for faster debugging
                batch_size=32,  # REDUCED for faster debugging
                n_epochs=4,    # REDUCED for faster debugging
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                vf_coef=0.5,
                verbose=1,  # Enable verbose output
                device='cpu',  # Use CPU for stability
                tensorboard_log=str(self.save_dir / 'tensorboard_logs')
            )
            print("✅ FINAL FIXED PPO model created successfully")
            
            # Training callback with debugging
            debug_callback = DebugCallback(verbose=1)
            
            eval_callback = EvalCallback(
                env,
                best_model_save_path=str(self.save_dir / 'ppo_best_final'),
                log_path=str(self.save_dir / 'ppo_logs_final'),
                eval_freq=max(timesteps // 5, 500),  # More frequent evaluation
                deterministic=True,
                verbose=1
            )
            
            print(f"🚀 Starting FINAL FIXED PPO training for {timesteps} timesteps...")
            
            # FIXED: Proper training with callbacks
            model.learn(
                total_timesteps=timesteps,
                callback=[debug_callback, eval_callback],
                tb_log_name="PPO_Final_Fixed",
                progress_bar=True  # Enable progress bar
            )
            
            print("🔧 Training completed, saving model...")
            
            # Save and evaluate
            model_path = self.save_dir / 'ppo_model_final_fixed.zip'
            model.save(str(model_path))
            
            print("🔧 Evaluating trained model...")
            mean_reward, std_reward = evaluate_policy(
                model, env, n_eval_episodes=5, deterministic=True
            )
            
            # Get environment statistics
            base_env = env.envs[0].env  # Get the unwrapped environment
            episode_stats = base_env.get_episode_stats()
            
            result = {
                'algorithm': 'PPO_Final_Fixed',
                'mean_reward': float(mean_reward),
                'std_reward': float(std_reward),
                'model_path': str(model_path),
                'status': 'success',
                'episode_stats': episode_stats,
                'training_timesteps': timesteps
            }
            
            print(f"✅ FINAL FIXED PPO training successful!")
            print(f"📊 Mean reward: {mean_reward:.3f} ± {std_reward:.3f}")
            print(f"📊 Episode stats: {episode_stats}")
            
            return result
            
        except Exception as e:
            print(f"❌ FINAL FIXED PPO training failed: {e}")
            import traceback
            traceback.print_exc()
            return {'algorithm': 'PPO_Final_Fixed', 'status': 'failed', 'error': str(e)}

    def train_dqn_final(self, train_data: List[Dict], timesteps: int = 10000) -> Dict[str, Any]:
        """FINAL FIXED DQN training (Note: DQN doesn't work well with continuous action spaces)."""
        
        print("🤖 Training DQN (FINAL FIXED VERSION)")
        print("-" * 50)
        print("⚠️ Note: DQN typically works with discrete actions, but we'll try with continuous")
        
        try:
            print("🔧 Creating environment for DQN...")
            env = self.create_env(train_data, n_envs=1)
            
            # Test environment
            print("🔧 Testing DQN environment...")
            obs = env.reset()
            test_action = env.action_space.sample()
            obs, reward, done, info = env.step(test_action)
            env.reset()
            print("✅ DQN environment test successful")
            
            print("🔧 For DQN, we'll use a discrete action wrapper...")
            # Since DQN needs discrete actions, we'll use A2C instead which handles continuous better
            
            print("🔧 Using A2C instead of DQN for continuous action space...")
            model = A2C(
                "MlpPolicy",
                env,
                learning_rate=1e-4,
                n_steps=32,
                gamma=0.99,
                gae_lambda=1.0,
                ent_coef=0.01,
                vf_coef=0.25,
                verbose=1,
                device='cpu',
                tensorboard_log=str(self.save_dir / 'tensorboard_logs')
            )
            print("✅ A2C model created successfully")
            
            debug_callback = DebugCallback(verbose=1)
            
            eval_callback = EvalCallback(
                env,
                best_model_save_path=str(self.save_dir / 'a2c_best_final'),
                log_path=str(self.save_dir / 'a2c_logs_final'),
                eval_freq=max(timesteps // 5, 500),
                deterministic=True,
                verbose=1
            )
            
            print(f"🚀 Starting A2C training for {timesteps} timesteps...")
            
            model.learn(
                total_timesteps=timesteps,
                callback=[debug_callback, eval_callback],
                tb_log_name="A2C_Final_Fixed",
                progress_bar=True
            )
            
            model_path = self.save_dir / 'a2c_model_final_fixed.zip'
            model.save(str(model_path))
            
            mean_reward, std_reward = evaluate_policy(
                model, env, n_eval_episodes=5, deterministic=True
            )
            
            # Get environment statistics
            base_env = env.envs[0].env
            episode_stats = base_env.get_episode_stats()
            
            result = {
                'algorithm': 'A2C_Final_Fixed',
                'mean_reward': float(mean_reward),
                'std_reward': float(std_reward),
                'model_path': str(model_path),
                'status': 'success',
                'episode_stats': episode_stats,
                'training_timesteps': timesteps
            }
            
            print(f"✅ A2C training successful!")
            print(f"📊 Mean reward: {mean_reward:.3f} ± {std_reward:.3f}")
            
            return result
            
        except Exception as e:
            print(f"❌ A2C training failed: {e}")
            import traceback
            traceback.print_exc()
            return {'algorithm': 'A2C_Final_Fixed', 'status': 'failed', 'error': str(e)}


def test_final_fixed_environment(train_data: List[Dict], config: Dict):
    """Test the final fixed environment."""
    
    print("🧪 FINAL FIXED ENVIRONMENT TEST")
    print("=" * 50)
    
    try:
        # Create environment
        env = SurgicalActionEnv(
            world_model=None,  # We'll mock this
            video_data=train_data,
            config=config.get('rl_training', {}),
            device='cpu'
        )
        
        print("✅ FINAL FIXED Environment created successfully")
        
        # Test action space
        print("🔧 Testing action space sampling...")
        for i in range(3):
            action = env.action_space.sample()
            print(f"  Sample {i+1}: type={type(action)}, shape={action.shape}, dtype={action.dtype}")
        
        # Test reset
        obs, info = env.reset()
        print(f"✅ Reset successful - Obs shape: {obs.shape}")
        
        # Test multiple steps
        total_reward = 0
        for step in range(10):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            
            print(f"Step {step+1}: Reward={reward:.3f}, Done={done}, Action_sum={info.get('binary_action_sum', 'N/A')}")
            
            if done:
                print(f"Episode ended at step {step+1}")
                break
        
        print(f"✅ FINAL FIXED Environment test completed - Total reward: {total_reward:.3f}")
        
        # Get final stats
        stats = env.get_episode_stats()
        print(f"📊 Episode stats: {stats}")
        
        return True
        
    except Exception as e:
        print(f"❌ FINAL FIXED Environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🔧 FINAL FIXED SB3 TRAINER")
    print("Action space issues resolved - ready for training!")
