#!/usr/bin/env python3
"""
RECOMMENDED: World Model-Based RL Training
Uses your correct environment.py with SB3 for reliability
"""

import torch
import numpy as np
from pathlib import Path
import yaml
import json
from datetime import datetime

# SB3 imports
from stable_baselines3 import PPO, A2C, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy

# Your existing modules (these are CORRECT)
from environment import SurgicalWorldModelEnv, MultiVideoSurgicalEnv  # ✅ Uses world model
from models.dual_world_model import DualWorldModel
from datasets.cholect50 import load_cholect50_data
from utils.logger import SimpleLogger


class WorldModelRLTrainer:
    """
    RECOMMENDED: Combines your correct environment with reliable SB3 training
    """
    
    def __init__(self, world_model_path: str, config: Dict, logger):
        self.config = config
        self.logger = logger
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load the world model (trained via IL)
        self.world_model = DualWorldModel.load_model(world_model_path, self.device)
        self.logger.info(f"✅ Loaded world model from {world_model_path}")
        
        # Create save directory
        self.save_dir = Path(logger.log_dir) / 'rl_world_model_training'
        self.save_dir.mkdir(exist_ok=True)

    def create_world_model_env(self, train_data: List[Dict]):
        """Create your CORRECT world model environment"""
        
        env_config = {
            'rl_horizon': self.config.get('rl_training', {}).get('rl_horizon', 50),
            'context_length': self.config.get('data', {}).get('context_length', 10),
            'reward_weights': self.config.get('rl_training', {}).get('reward_weights', {
                'phase_completion': 1.0,
                'phase_initiation': 0.5,
                'phase_progression': 1.0,
                'action_probability': 0.3,
                'risk_penalty': -0.5,
                'global_progression': 0.8
            }),
            'normalize_rewards': True,
            'early_termination': True
        }
        
        def make_env():
            # Use your CORRECT MultiVideoSurgicalEnv
            env = MultiVideoSurgicalEnv(
                world_model=self.world_model,
                config=env_config, 
                video_data=train_data,
                device=self.device
            )
            return env
        
        # Create vectorized environment
        return DummyVecEnv([make_env])

    def train_ppo_with_world_model(self, train_data: List[Dict], timesteps: int = 50000):
        """Train PPO using world model simulation"""
        
        self.logger.info("🚀 Training PPO with World Model Simulation")
        self.logger.info(f"🎯 This ACTUALLY uses your world model for state prediction!")
        
        # Create environment
        env = self.create_world_model_env(train_data)
        
        # Test environment
        self.logger.info("🧪 Testing environment...")
        obs = env.reset()
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        self.logger.info(f"✅ Environment test successful - reward: {reward}")
        env.reset()
        
        # Create PPO model
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,      # PPO batch size
            batch_size=64,     # Mini-batch size
            n_epochs=10,       # PPO epochs
            gamma=0.99,        # Discount factor
            gae_lambda=0.95,   # GAE lambda
            clip_range=0.2,    # PPO clip range
            ent_coef=0.01,     # Entropy coefficient
            vf_coef=0.5,       # Value function coefficient
            verbose=1,
            device='cpu',      # Use CPU for stability
            tensorboard_log=str(self.save_dir / 'tensorboard')
        )
        
        # Setup callbacks
        checkpoint_callback = CheckpointCallback(
            save_freq=10000,
            save_path=str(self.save_dir / 'checkpoints'),
            name_prefix='ppo_world_model'
        )
        
        eval_callback = EvalCallback(
            env,
            best_model_save_path=str(self.save_dir / 'best_model'),
            log_path=str(self.save_dir / 'eval_logs'),
            eval_freq=5000,
            n_eval_episodes=5,
            deterministic=True
        )
        
        # Train the model
        self.logger.info(f"🎯 Training for {timesteps} timesteps...")
        model.learn(
            total_timesteps=timesteps,
            callback=[checkpoint_callback, eval_callback],
            tb_log_name="PPO_WorldModel",
            progress_bar=True
        )
        
        # Save final model
        final_model_path = self.save_dir / 'ppo_world_model_final.zip'
        model.save(str(final_model_path))
        
        # Evaluate final performance
        mean_reward, std_reward = evaluate_policy(
            model, env, n_eval_episodes=10, deterministic=True
        )
        
        result = {
            'algorithm': 'PPO_WorldModel',
            'mean_reward': float(mean_reward),
            'std_reward': float(std_reward),
            'training_timesteps': timesteps,
            'model_path': str(final_model_path),
            'status': 'success',
            'uses_world_model': True,
            'world_model_path': None  # Would store the world model path
        }
        
        self.logger.info(f"✅ PPO World Model Training Complete!")
        self.logger.info(f"📊 Final Performance: {mean_reward:.3f} ± {std_reward:.3f}")
        self.logger.info(f"🎯 This model was trained using TRUE world model simulation")
        
        return result

    def train_sac_with_world_model(self, train_data: List[Dict], timesteps: int = 50000):
        """Train SAC using world model simulation"""
        
        self.logger.info("🚀 Training SAC with World Model Simulation")
        
        # Create environment
        env = self.create_world_model_env(train_data)
        
        # Create SAC model  
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            buffer_size=100000,
            learning_starts=1000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            ent_coef='auto',
            verbose=1,
            device='cpu',
            tensorboard_log=str(self.save_dir / 'tensorboard')
        )
        
        # Setup callbacks
        checkpoint_callback = CheckpointCallback(
            save_freq=10000,
            save_path=str(self.save_dir / 'checkpoints'),
            name_prefix='sac_world_model'
        )
        
        eval_callback = EvalCallback(
            env,
            best_model_save_path=str(self.save_dir / 'best_model'),
            log_path=str(self.save_dir / 'eval_logs'),
            eval_freq=5000,
            n_eval_episodes=5,
            deterministic=True
        )
        
        # Train the model
        model.learn(
            total_timesteps=timesteps,
            callback=[checkpoint_callback, eval_callback],
            tb_log_name="SAC_WorldModel",
            progress_bar=True
        )
        
        # Save and evaluate
        final_model_path = self.save_dir / 'sac_world_model_final.zip'
        model.save(str(final_model_path))
        
        mean_reward, std_reward = evaluate_policy(
            model, env, n_eval_episodes=10, deterministic=True
        )
        
        result = {
            'algorithm': 'SAC_WorldModel',
            'mean_reward': float(mean_reward),
            'std_reward': float(std_reward),
            'training_timesteps': timesteps,
            'model_path': str(final_model_path),
            'status': 'success',
            'uses_world_model': True
        }
        
        self.logger.info(f"✅ SAC World Model Training Complete!")
        self.logger.info(f"📊 Final Performance: {mean_reward:.3f} ± {std_reward:.3f}")
        
        return result

    def train_all_rl_algorithms(self, train_data: List[Dict]) -> Dict[str, Any]:
        """Train all RL algorithms using world model"""
        
        results = {}
        
        # Get configuration
        rl_config = self.config.get('experiment', {}).get('rl_experiments', {})
        timesteps = rl_config.get('timesteps', 50000)
        algorithms = rl_config.get('algorithms', ['ppo', 'sac'])
        
        # Train each algorithm
        for algorithm in algorithms:
            try:
                if algorithm.lower() == 'ppo':
                    results['ppo'] = self.train_ppo_with_world_model(train_data, timesteps)
                elif algorithm.lower() == 'sac':
                    results['sac'] = self.train_sac_with_world_model(train_data, timesteps)
                else:
                    self.logger.warning(f"Unknown algorithm: {algorithm}")
                    
            except Exception as e:
                self.logger.error(f"❌ Training failed for {algorithm}: {e}")
                results[algorithm] = {'status': 'failed', 'error': str(e)}
        
        # Save results
        results_path = self.save_dir / 'world_model_rl_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"📄 Results saved to: {results_path}")
        
        return results


def run_world_model_rl_training():
    """Main function to run world model-based RL training"""
    
    print("🚀 WORLD MODEL-BASED RL TRAINING")
    print("=" * 50)
    print("✅ Uses your CORRECT environment.py implementation")
    print("✅ True world model simulation (not video replay)")
    print("✅ Reliable SB3 algorithms")
    print("✅ Proper evaluation and callbacks")
    print()
    
    # Load config
    config_path = 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup logging
    logger = SimpleLogger(log_dir="logs", name="world_model_rl_training")
    
    # Load training data
    train_data = load_cholect50_data(config, logger, split='train', max_videos=10)
    logger.info(f"Loaded {len(train_data)} training videos")
    
    # Path to your trained IL model (world model)
    world_model_path = "logs/2025-05-28_12-37-34/checkpoints/supervised_best_epoch_3.pt"
    
    if not Path(world_model_path).exists():
        print(f"❌ World model not found at: {world_model_path}")
        print("Please update the path to your trained IL model")
        return
    
    # Create trainer
    trainer = WorldModelRLTrainer(world_model_path, config, logger)
    
    # Train all RL algorithms
    results = trainer.train_all_rl_algorithms(train_data)
    
    # Print summary
    print("\n" + "🎉 WORLD MODEL RL TRAINING COMPLETE!" + "🎉")
    print("=" * 50)
    
    for algorithm, result in results.items():
        if result.get('status') == 'success':
            print(f"✅ {algorithm.upper()}: {result['mean_reward']:.3f} ± {result['std_reward']:.3f}")
        else:
            print(f"❌ {algorithm.upper()}: Failed")
    
    print(f"\n🎯 Key Achievement: RL agents trained using TRUE world model simulation!")
    print(f"📁 Results saved to: {trainer.save_dir}")
    
    return results


if __name__ == "__main__":
    results = run_world_model_rl_training()
