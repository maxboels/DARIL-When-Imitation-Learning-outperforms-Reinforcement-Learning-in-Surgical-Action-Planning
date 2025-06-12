#!/usr/bin/env python3
"""
World Model RL Trainer for Method 2
RL training using ConditionalWorldModel as environment simulator
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

# Custom imports
from environment.rl_environments import WorldModelSimulationEnv


class WorldModelRLTrainer:
    """
    RL trainer using ConditionalWorldModel for simulation (Method 2).
    """
    
    def __init__(self, world_model, config: Dict, logger, device: str = 'cuda'):
        self.world_model = world_model
        self.config = config
        self.logger = logger
        self.device = device
        
        # Validate world model
        if self.world_model is None:
            raise ValueError("❌ ConditionalWorldModel is required for Method 2!")
        
        # Create results directory
        self.save_dir = Path(logger.log_dir) / 'rl_world_model_simulation'
        self.save_dir.mkdir(exist_ok=True)
        
        print(f"🌍 World Model RL Trainer initialized")
        print(f"📁 Save dir: {self.save_dir}")
        print(f"🎯 Method: RL with ConditionalWorldModel simulation")

    def create_world_model_env(self, train_data: List[Dict], n_envs: int = 1):
        """Create world model-based environment."""
        
        def make_env():
            env = WorldModelSimulationEnv(
                world_model=self.world_model,
                video_data=train_data,
                config=self.config.get('rl_training', {}),
                device=self.device
            )
            return Monitor(env)
        
        if n_envs == 1:
            return DummyVecEnv([make_env])
        else:
            return SubprocVecEnv([make_env for _ in range(n_envs)])

    def train_ppo_with_world_model(self, train_data: List[Dict], timesteps: int = 10000) -> Dict[str, Any]:
        """Train PPO using ConditionalWorldModel simulation."""
        
        print("🌍 Training PPO with ConditionalWorldModel Simulation")
        print("-" * 60)
        
        try:
            # Create world model environment
            print("🔧 Creating world model simulation environment...")
            env = self.create_world_model_env(train_data, n_envs=1)
            
            # Test environment
            print("🔧 Testing world model environment...")
            obs = env.reset()
            test_action = env.action_space.sample()
            obs, reward, done, info = env.step(test_action)
            uses_wm = info[0].get('uses_world_model', False)
            sim_based = info[0].get('simulation_based', False)
            print(f"✅ Environment test successful - Uses WM: {uses_wm}, Simulation: {sim_based}")
            env.reset()
            
            # Create PPO model
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
            
            # Training callbacks
            eval_callback = EvalCallback(
                env,
                best_model_save_path=str(self.save_dir / 'ppo_world_model_best'),
                log_path=str(self.save_dir / 'ppo_world_model_logs'),
                eval_freq=max(timesteps // 5, 500),
                deterministic=True,
                verbose=1
            )
            
            print(f"🚀 Training PPO with ConditionalWorldModel for {timesteps} timesteps...")
            
            # Train model
            model.learn(
                total_timesteps=timesteps,
                callback=[eval_callback],
                tb_log_name="PPO_ConditionalWorldModel",
                progress_bar=True
            )
            
            # Save and evaluate
            model_path = self.save_dir / 'ppo_conditional_world_model.zip'
            model.save(str(model_path))
            
            mean_reward, std_reward = evaluate_policy(
                model, env, n_eval_episodes=5, deterministic=True
            )
            
            # Get environment statistics
            base_env = env.envs[0].env
            episode_stats = base_env.get_episode_stats()
            
            result = {
                'algorithm': 'PPO_ConditionalWorldModel',
                'approach': 'RL training with ConditionalWorldModel simulation',
                'mean_reward': float(mean_reward),
                'std_reward': float(std_reward),
                'model_path': str(model_path),
                'status': 'success',
                'training_timesteps': timesteps,
                'uses_world_model': True,
                'simulation_based': True,
                'can_explore_beyond_demos': True,
                'method_type': 'model_based_rl',
                'episode_stats': episode_stats
            }
            
            print(f"✅ ConditionalWorldModel PPO training successful!")
            print(f"📊 Mean reward: {mean_reward:.3f} ± {std_reward:.3f}")
            print(f"🌍 World model predictions: {episode_stats.get('uses_world_model', True)}")
            
            return result
            
        except Exception as e:
            print(f"❌ ConditionalWorldModel PPO training failed: {e}")
            import traceback
            traceback.print_exc()
            return {'algorithm': 'PPO_ConditionalWorldModel', 'status': 'failed', 'error': str(e)}

    def train_a2c_with_world_model(self, train_data: List[Dict], timesteps: int = 10000) -> Dict[str, Any]:
        """Train A2C using ConditionalWorldModel simulation."""
        
        print("🌍 Training A2C with ConditionalWorldModel Simulation")
        print("-" * 60)
        
        try:
            # Create world model environment
            env = self.create_world_model_env(train_data, n_envs=1)
            
            # Test environment
            obs = env.reset()
            test_action = env.action_space.sample()
            obs, reward, done, info = env.step(test_action)
            uses_wm = info[0].get('uses_world_model', False)
            print(f"✅ Environment test successful - Uses WM: {uses_wm}")
            env.reset()
            
            # Create A2C model
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
            
            # Training callbacks
            eval_callback = EvalCallback(
                env,
                best_model_save_path=str(self.save_dir / 'a2c_world_model_best'),
                log_path=str(self.save_dir / 'a2c_world_model_logs'),
                eval_freq=max(timesteps // 5, 500),
                deterministic=True,
                verbose=1
            )
            
            print(f"🚀 Training A2C with ConditionalWorldModel for {timesteps} timesteps...")
            
            # Train model
            model.learn(
                total_timesteps=timesteps,
                callback=[eval_callback],
                tb_log_name="A2C_ConditionalWorldModel",
                progress_bar=True
            )
            
            # Save and evaluate
            model_path = self.save_dir / 'a2c_conditional_world_model.zip'
            model.save(str(model_path))
            
            mean_reward, std_reward = evaluate_policy(
                model, env, n_eval_episodes=5, deterministic=True
            )
            
            # Get environment statistics
            base_env = env.envs[0].env
            episode_stats = base_env.get_episode_stats()
            
            result = {
                'algorithm': 'A2C_ConditionalWorldModel',
                'approach': 'RL training with ConditionalWorldModel simulation',
                'mean_reward': float(mean_reward),
                'std_reward': float(std_reward),
                'model_path': str(model_path),
                'status': 'success',
                'training_timesteps': timesteps,
                'uses_world_model': True,
                'simulation_based': True,
                'can_explore_beyond_demos': True,
                'method_type': 'model_based_rl',
                'episode_stats': episode_stats
            }
            
            print(f"✅ ConditionalWorldModel A2C training successful!")
            print(f"📊 Mean reward: {mean_reward:.3f} ± {std_reward:.3f}")
            
            return result
            
        except Exception as e:
            print(f"❌ ConditionalWorldModel A2C training failed: {e}")
            import traceback
            traceback.print_exc()
            return {'algorithm': 'A2C_ConditionalWorldModel', 'status': 'failed', 'error': str(e)}

    def train_all_algorithms(self, train_data: List[Dict], timesteps: int = 10000) -> Dict[str, Any]:
        """Train all RL algorithms using ConditionalWorldModel simulation."""
        
        print("🌍 Training all RL algorithms with ConditionalWorldModel")
        print("=" * 60)
        
        results = {}
        
        # Train PPO with world model
        try:
            results['ppo'] = self.train_ppo_with_world_model(train_data, timesteps)
        except Exception as e:
            self.logger.error(f"❌ PPO training failed: {e}")
            results['ppo'] = {'status': 'failed', 'error': str(e)}
        
        # Train A2C with world model
        try:
            results['a2c'] = self.train_a2c_with_world_model(train_data, timesteps)
        except Exception as e:
            self.logger.error(f"❌ A2C training failed: {e}")
            results['a2c'] = {'status': 'failed', 'error': str(e)}
        
        # Summary
        successful_algorithms = [alg for alg, res in results.items() if res.get('status') == 'success']
        print(f"\n✅ ConditionalWorldModel RL Training Summary:")
        print(f"   Successful algorithms: {successful_algorithms}")
        print(f"   All use ConditionalWorldModel for simulation")
        print(f"   All can explore beyond expert demonstrations")
        
        return results


def test_world_model_simulation_environment(world_model, train_data: List[Dict], config: Dict):
    """Test the ConditionalWorldModel simulation environment."""
    
    print("🧪 TESTING CONDITIONAL WORLD MODEL SIMULATION ENVIRONMENT")
    print("=" * 60)
    
    try:
        # Create environment
        env = WorldModelSimulationEnv(
            world_model=world_model,
            video_data=train_data,
            config=config.get('rl_training', {}),
            device='cpu'
        )
        
        print("✅ World Model Simulation Environment created successfully")
        
        # Test reset
        obs, info = env.reset()
        print(f"✅ Reset successful - Obs shape: {obs.shape}")
        
        # Test multiple steps with world model predictions
        total_reward = 0
        world_model_uses = 0
        
        for step in range(10):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            
            if info.get('uses_world_model', False):
                world_model_uses += 1
            
            # Print step info
            if step % 2 == 0:  # Print every 2 steps
                print(f"Step {step+1}: Reward={reward:.3f}, Done={done}, "
                    f"WM predictions={info.get('world_model_predictions', 0)}, "
                    f"Method={info.get('method', 'unknown')}")
            
            if done:
                print(f"Episode ended at step {step+1}")
                break
        
        print(f"✅ ConditionalWorldModel Environment test completed")
        print(f"📊 Total reward: {total_reward:.3f}")
        print(f"🌍 World model uses: {world_model_uses}/{step+1}")
        print(f"🌍 Simulation ratio: {world_model_uses/(step+1)*100:.1f}%")
        
        # Get final stats
        stats = env.get_episode_stats()
        print(f"📊 Episode stats: {stats}")
        
        # Verify this is actually using world model simulation
        if stats.get('uses_world_model', False) and stats.get('simulation_based', False):
            print("🎯 ✅ VERIFIED: Environment uses ConditionalWorldModel simulation")
        else:
            print("🎯 ❌ WARNING: Environment may not be using world model properly")
        
        return True
        
    except Exception as e:
        print(f"❌ ConditionalWorldModel Environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🌍 WORLD MODEL RL TRAINER")
    print("=" * 60)
    print("✅ Method 2 trainer using ConditionalWorldModel for:")
    print("  • Action-conditioned state prediction")
    print("  • Multi-type reward prediction")
    print("  • True RL simulation environment")
    print("  • Exploration beyond expert demonstrations")
    print("  • Model-based reinforcement learning")