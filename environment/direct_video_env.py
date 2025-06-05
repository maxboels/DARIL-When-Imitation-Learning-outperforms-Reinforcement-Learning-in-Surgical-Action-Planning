#!/usr/bin/env python3
"""
COMPLETE Direct Video Environment for Method 3: RL with Offline Video Episodes
Steps through actual video frames without world model simulation
"""

import torch
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Any, Optional
import random
from pathlib import Path

class DirectVideoEnvironment(gym.Env):
    """
    Environment that steps through actual video frames for RL training.
    This is Method 3 in your three-way comparison.
    
    Key differences from world model approach:
    - Uses actual video frames, not simulated states
    - Limited to existing video sequences
    - No state prediction/simulation
    - Direct interaction with real data
    """
    
    def __init__(self, video_data: List[Dict], config: Dict, device: str = 'cuda'):
        super().__init__()
        
        self.video_data = video_data
        self.config = config
        self.device = device
        
        # Environment parameters
        self.max_episode_steps = config.get('rl_horizon', 50)
        
        # Current episode state
        self.current_video_idx = 0
        self.current_frame_idx = 0
        self.current_step = 0
        self.episode_reward = 0.0
        
        # Episode statistics
        self.episode_lengths = []
        self.episode_rewards = []
        
        # Action and observation spaces
        self.action_space = spaces.Box(
            low=0, high=1, shape=(100,), dtype=np.float32
        )
        
        # Assuming frame embeddings are 1024-dimensional
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(1024,), dtype=np.float32
        )
        
        # Reward configuration for offline videos
        self.reward_weights = {
            'action_accuracy': 1.0,      # Match expert actions
            'phase_progress': 0.5,       # Progress through surgical phases
            'safety_bonus': 2.0,         # Safe action combinations
            'completion_bonus': 5.0      # Episode completion
        }
        
        print(f"🎬 DirectVideoEnvironment initialized with {len(video_data)} videos")
        print(f"📐 Action space: {self.action_space}")
        print(f"📐 Observation space: {self.observation_space}")
    
    def reset(self, seed=None, options=None):
        """Reset by selecting a random video and starting frame."""
        super().reset(seed=seed)
        
        # Select random video
        self.current_video_idx = random.randint(0, len(self.video_data) - 1)
        video = self.video_data[self.current_video_idx]
        
        # Select random starting frame (ensure enough frames for full episode)
        available_frames = len(video['frame_embeddings'])
        max_start_frame = max(0, available_frames - self.max_episode_steps - 1)
        self.current_frame_idx = random.randint(0, max_start_frame)
        
        # Reset episode variables
        self.current_step = 0
        self.episode_reward = 0.0
        
        # Get initial observation (current frame embedding)
        initial_state = video['frame_embeddings'][self.current_frame_idx].astype(np.float32)
        
        # Ensure correct shape
        if len(initial_state) != 1024:
            if len(initial_state) < 1024:
                initial_state = np.pad(initial_state, (0, 1024 - len(initial_state)))
            else:
                initial_state = initial_state[:1024]
        
        info = {
            'video_id': video.get('video_id', 'unknown'),
            'frame_idx': self.current_frame_idx,
            'available_frames': available_frames
        }
        
        return initial_state, info
    
    def step(self, action):
        """Step through actual video frame."""
        self.current_step += 1
        
        # Get current video
        video = self.video_data[self.current_video_idx]
        
        # Convert action to binary
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        action = np.array(action).flatten()
        
        # Ensure correct action size
        if len(action) != 100:
            if len(action) < 100:
                padded_action = np.zeros(100, dtype=np.float32)
                padded_action[:len(action)] = action
                action = padded_action
            else:
                action = action[:100]
        
        binary_action = (action > 0.5).astype(int)
        
        # Check if episode should end
        frames_remaining = len(video['frame_embeddings']) - self.current_frame_idx - 1
        step_limit_reached = self.current_step >= self.max_episode_steps
        
        terminated = step_limit_reached  # Episode ended due to step limit
        truncated = frames_remaining <= 0  # Episode ended due to video end
        done = terminated or truncated
        
        if done:
            # Episode completion
            reward = self.reward_weights['completion_bonus']
            next_state = video['frame_embeddings'][self.current_frame_idx].astype(np.float32)
        else:
            # Move to next actual frame
            self.current_frame_idx += 1
            next_state = video['frame_embeddings'][self.current_frame_idx].astype(np.float32)
            
            # Calculate reward based on expert actions and other factors
            reward = self._calculate_reward(binary_action, video)
        
        # Ensure state has correct shape
        if len(next_state) != 1024:
            if len(next_state) < 1024:
                next_state = np.pad(next_state, (0, 1024 - len(next_state)))
            else:
                next_state = next_state[:1024]
        
        self.episode_reward += reward
        
        # Enhanced info
        info = {
            'step': self.current_step,
            'video_id': video.get('video_id', 'unknown'),
            'frame_idx': self.current_frame_idx,
            'episode_reward': self.episode_reward,
            'frames_remaining': frames_remaining,
            'binary_action_sum': int(np.sum(binary_action)),
            'using_real_frames': True,  # Key distinguisher from world model approach
            'method': 'direct_video_rl',
            'simulation_based': False
        }
        
        # Log episode completion
        if done:
            self.episode_lengths.append(self.current_step)
            self.episode_rewards.append(self.episode_reward)
            
            # Keep only last 100 episodes for statistics
            if len(self.episode_lengths) > 100:
                self.episode_lengths = self.episode_lengths[-100:]
                self.episode_rewards = self.episode_rewards[-100:]
        
        # Return in gymnasium format: observation, reward, terminated, truncated, info
        return next_state, float(reward), bool(terminated), bool(truncated), info
    
    def _calculate_reward(self, predicted_actions: np.ndarray, video: Dict) -> float:
        """Calculate reward based on expert actions and surgical progress."""
        
        reward = 0.0
        
        # 1. Action accuracy reward (compare with expert actions)
        if (self.current_frame_idx < len(video.get('actions_binaries', [])) and 
            len(video['actions_binaries']) > 0):
            
            expert_actions = video['actions_binaries'][self.current_frame_idx]
            
            if len(expert_actions) == len(predicted_actions):
                # Calculate accuracy
                accuracy = np.mean(predicted_actions == expert_actions)
                reward += self.reward_weights['action_accuracy'] * accuracy
                
                # Bonus for correctly predicting positive actions (surgical actions are sparse)
                if np.sum(expert_actions) > 0:
                    true_positives = np.sum((predicted_actions == 1) & (expert_actions == 1))
                    precision = true_positives / max(np.sum(predicted_actions), 1)
                    recall = true_positives / np.sum(expert_actions)
                    if precision > 0 and recall > 0:
                        f1 = 2 * (precision * recall) / (precision + recall)
                        reward += self.reward_weights['action_accuracy'] * f1
        
        # 2. Phase progression reward (if available)
        if ('next_rewards' in video and 
            '_r_phase_progression' in video['next_rewards'] and
            self.current_frame_idx < len(video['next_rewards']['_r_phase_progression'])):
            
            phase_progress = video['next_rewards']['_r_phase_progression'][self.current_frame_idx]
            if phase_progress > 0:
                reward += self.reward_weights['phase_progress'] * phase_progress
        
        # 3. Safety bonus (avoid unsafe action combinations)
        unsafe_combinations = [[15, 23], [34, 45, 67], [78, 89]]  # Example unsafe patterns
        active_actions = set(np.where(predicted_actions > 0.5)[0])
        
        safety_violation = False
        for unsafe_pattern in unsafe_combinations:
            if all(action in active_actions for action in unsafe_pattern):
                safety_violation = True
                break
        
        if not safety_violation:
            reward += self.reward_weights['safety_bonus'] * 0.1  # Small safety bonus
        else:
            reward -= self.reward_weights['safety_bonus'] * 0.5  # Safety penalty
        
        # 4. Action density reward (encourage appropriate action levels)
        action_count = np.sum(predicted_actions)
        if 1 <= action_count <= 5:  # Reasonable action density
            reward += 0.1
        
        # Clip reward to reasonable range
        return np.clip(reward, -2.0, 5.0)
    
    def get_episode_stats(self):
        """Get episode statistics."""
        if not self.episode_lengths:
            return {"avg_length": 0, "avg_reward": 0, "episodes": 0}
        
        return {
            "avg_length": np.mean(self.episode_lengths),
            "avg_reward": np.mean(self.episode_rewards),
            "episodes": len(self.episode_lengths),
            "last_length": self.episode_lengths[-1],
            "last_reward": self.episode_rewards[-1],
            "using_real_frames": True
        }


class DirectVideoSB3Trainer:
    """
    Trainer for Method 3: RL with Direct Video Episodes
    """
    
    def __init__(self, video_data: List[Dict], config: Dict, logger, device: str = 'cuda'):
        self.video_data = video_data
        self.config = config
        self.logger = logger
        self.device = device
        
        # Create results directory
        self.save_dir = Path(logger.log_dir) / 'direct_video_rl'
        self.save_dir.mkdir(exist_ok=True)
        
        print(f"🎬 DirectVideoSB3Trainer initialized")
        print(f"📁 Save dir: {self.save_dir}")
    
    def create_env(self, n_envs: int = 1):
        """Create direct video environment."""
        
        def make_env():
            env = DirectVideoEnvironment(
                video_data=self.video_data,
                config=self.config.get('rl_training', {}),
                device=self.device
            )
            from stable_baselines3.common.monitor import Monitor
            return Monitor(env)
        
        if n_envs == 1:
            from stable_baselines3.common.vec_env import DummyVecEnv
            return DummyVecEnv([make_env])
        else:
            from stable_baselines3.common.vec_env import SubprocVecEnv
            return SubprocVecEnv([make_env for _ in range(n_envs)])
    
    def train_ppo_direct(self, timesteps: int = 10000) -> Dict[str, Any]:
        """Train PPO on direct video episodes."""
        
        print("🎬 Training PPO with Direct Video Episodes")
        print("-" * 50)
        
        try:
            # Create environment
            env = self.create_env(n_envs=1)
            
            # Test environment
            print("🔧 Testing direct video environment...")
            obs = env.reset()
            test_action = env.action_space.sample()
            # obs, reward, terminated, truncated, info = env.step(test_action)  # Updated for gymnasium
            step_result = env.step(test_action)
            if len(step_result) == 4:
                obs, reward, done, info = step_result
                print(f"✅ Environment test successful (4-value format) - Reward: {reward}")
            elif len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
                print(f"✅ Environment test successful (5-value format) - Reward: {reward}")
            
            
            print(f"✅ Environment test successful - Reward: {reward}")
            env.reset()
            
            # Create PPO model
            from stable_baselines3 import PPO
            from stable_baselines3.common.callbacks import EvalCallback
            
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=3e-4,
                n_steps=256,
                batch_size=32,
                n_epochs=4,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                vf_coef=0.5,
                verbose=1,
                device='cpu',
                tensorboard_log=str(self.save_dir / 'tensorboard_logs')
            )
            
            # Setup callbacks
            eval_callback = EvalCallback(
                env,
                best_model_save_path=str(self.save_dir / 'ppo_best_direct'),
                log_path=str(self.save_dir / 'ppo_logs_direct'),
                eval_freq=max(timesteps // 5, 500),
                deterministic=True,
                verbose=1
            )
            
            print(f"🚀 Training PPO for {timesteps} timesteps...")
            
            # Train model
            model.learn(
                total_timesteps=timesteps,
                callback=[eval_callback],
                tb_log_name="PPO_DirectVideo",
                progress_bar=True
            )
            
            # Save and evaluate
            model_path = self.save_dir / 'ppo_direct_video.zip'
            model.save(str(model_path))
            
            from stable_baselines3.common.evaluation import evaluate_policy
            mean_reward, std_reward = evaluate_policy(
                model, env, n_eval_episodes=5, deterministic=True
            )
            
            # Get environment statistics
            base_env = env.envs[0].env  # Get the unwrapped environment
            episode_stats = base_env.get_episode_stats()
            
            result = {
                'algorithm': 'PPO_DirectVideo',
                'approach': 'Direct RL on video sequences (no world model)',
                'mean_reward': float(mean_reward),
                'std_reward': float(std_reward),
                'model_path': str(model_path),
                'status': 'success',
                'training_timesteps': timesteps,
                'uses_world_model': False,
                'uses_real_frames': True,
                'simulation_based': False,
                'episode_stats': episode_stats
            }
            
            print(f"✅ PPO Direct Video training successful!")
            print(f"📊 Mean reward: {mean_reward:.3f} ± {std_reward:.3f}")
            print(f"📊 Episode stats: {episode_stats}")
            
            return result
            
        except Exception as e:
            print(f"❌ PPO Direct Video training failed: {e}")
            import traceback
            traceback.print_exc()
            return {'algorithm': 'PPO_DirectVideo', 'status': 'failed', 'error': str(e)}
    
    def train_a2c_direct(self, timesteps: int = 10000) -> Dict[str, Any]:
        """Train A2C on direct video episodes."""
        
        print("🎬 Training A2C with Direct Video Episodes")
        print("-" * 50)
        
        try:
            # Create environment
            env = self.create_env(n_envs=1)
            
            # Test environment
            print("🔧 Testing direct video environment...")
            obs = env.reset()
            test_action = env.action_space.sample()
            # obs, reward, terminated, truncated, info = env.step(test_action)  # Updated for gymnasium
            step_result = env.step(test_action)
            if len(step_result) == 4:
                obs, reward, done, info = step_result
                print(f"✅ Environment test successful (4-value format) - Reward: {reward}")
            elif len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
                print(f"✅ Environment test successful (5-value format) - Reward: {reward}")
            
            print(f"✅ Environment test successful - Reward: {reward}")
            env.reset()
            
            # Create A2C model
            from stable_baselines3 import A2C
            from stable_baselines3.common.callbacks import EvalCallback
            
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
            
            # Setup callbacks
            eval_callback = EvalCallback(
                env,
                best_model_save_path=str(self.save_dir / 'a2c_best_direct'),
                log_path=str(self.save_dir / 'a2c_logs_direct'),
                eval_freq=max(timesteps // 5, 500),
                deterministic=True,
                verbose=1
            )
            
            print(f"🚀 Training A2C for {timesteps} timesteps...")
            
            # Train model
            model.learn(
                total_timesteps=timesteps,
                callback=[eval_callback],
                tb_log_name="A2C_DirectVideo",
                progress_bar=True
            )
            
            # Save and evaluate
            model_path = self.save_dir / 'a2c_direct_video.zip'
            model.save(str(model_path))
            
            from stable_baselines3.common.evaluation import evaluate_policy
            mean_reward, std_reward = evaluate_policy(
                model, env, n_eval_episodes=5, deterministic=True
            )
            
            # Get environment statistics
            base_env = env.envs[0].env  # Get the unwrapped environment
            episode_stats = base_env.get_episode_stats()
            
            result = {
                'algorithm': 'A2C_DirectVideo',
                'approach': 'Direct RL on video sequences (no world model)',
                'mean_reward': float(mean_reward),
                'std_reward': float(std_reward),
                'model_path': str(model_path),
                'status': 'success',
                'training_timesteps': timesteps,
                'uses_world_model': False,
                'uses_real_frames': True,
                'simulation_based': False,
                'episode_stats': episode_stats
            }
            
            print(f"✅ A2C Direct Video training successful!")
            print(f"📊 Mean reward: {mean_reward:.3f} ± {std_reward:.3f}")
            print(f"📊 Episode stats: {episode_stats}")
            
            return result
            
        except Exception as e:
            print(f"❌ A2C Direct Video training failed: {e}")
            import traceback
            traceback.print_exc()
            return {'algorithm': 'A2C_DirectVideo', 'status': 'failed', 'error': str(e)}
    
    def train_all_algorithms(self, timesteps: int = 10000) -> Dict[str, Any]:
        """Train all algorithms on direct video sequences."""
        
        results = {}
        
        # Train PPO
        try:
            results['ppo'] = self.train_ppo_direct(timesteps)
        except Exception as e:
            self.logger.error(f"❌ PPO training failed: {e}")
            results['ppo'] = {'status': 'failed', 'error': str(e)}
        
        # Train A2C
        try:
            results['a2c'] = self.train_a2c_direct(timesteps)
        except Exception as e:
            self.logger.error(f"❌ A2C training failed: {e}")
            results['a2c'] = {'status': 'failed', 'error': str(e)}
        
        return results


def test_direct_video_environment(train_data: List[Dict], config: Dict):
    """Test the direct video environment with actual data."""
    
    print("🧪 TESTING DIRECT VIDEO ENVIRONMENT")
    print("=" * 40)
    
    try:
        # Create environment
        env = DirectVideoEnvironment(
            video_data=train_data,
            config=config.get('rl_training', {}),
            device='cpu'
        )
        
        print("✅ DirectVideoEnvironment created successfully")
        
        # Test action space
        print("🔧 Testing action space sampling...")
        for i in range(3):
            action = env.action_space.sample()
            print(f"  Sample {i+1}: type={type(action)}, shape={action.shape}, dtype={action.dtype}")
        
        # Test reset
        obs, info = env.reset()
        print(f"✅ Reset successful - Obs shape: {obs.shape}")
        print(f"📊 Info: {info}")
        
        # Test multiple steps
        total_reward = 0
        for step in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)  # Updated for gymnasium
            total_reward += reward
            
            print(f"Step {step+1}: Reward={reward:.3f}, Terminated={terminated}, Truncated={truncated}, "
                  f"Action_sum={info.get('binary_action_sum', 'N/A')}")
            
            if terminated or truncated:
                print(f"Episode ended at step {step+1}")
                break
        
        print(f"✅ DirectVideoEnvironment test completed - Total reward: {total_reward:.3f}")
        
        # Get final stats
        stats = env.get_episode_stats()
        print(f"📊 Episode stats: {stats}")
        
        return True
        
    except Exception as e:
        print(f"❌ DirectVideoEnvironment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🔧 COMPLETE DIRECT VIDEO ENVIRONMENT")
    print("=" * 50)
    print("✅ Method 3: RL with Offline Video Episodes")
    print("✅ Steps through actual video frames (no simulation)")
    print("✅ Model-free RL on offline data")
    print("✅ Ready for integration with main experiment!")
    print("\nTo test this environment, run:")
    print("python test_method3_standalone.py")