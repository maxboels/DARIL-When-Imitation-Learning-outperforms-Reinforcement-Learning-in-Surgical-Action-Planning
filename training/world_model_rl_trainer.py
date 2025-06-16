#!/usr/bin/env python3
"""
RL Training with proper hyperparameters and monitoring
Addresses the convergence and performance issues in RL training
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
import matplotlib.pyplot as plt
from collections import deque

# Custom imports
from environment.world_model_env import WorldModelSimulationEnv
from environment.direct_video_env import DirectVideoEnvironment


class WorldModelRLTrainer:
    """
    RL Trainer with proper hyperparameters and monitoring.
    """
    
    def __init__(self, config: Dict, logger, device: str = 'cuda'):
        self.config = config
        self.logger = logger
        self.device = device
        
        # Create results directory
        self.save_dir = Path(logger.log_dir) / 'rl_training'
        self.save_dir.mkdir(exist_ok=True)
        
        self.logger.info(f"🔧 RL Trainer initialized with config: {self.config}")
        self.logger.info(f"📂 Results will be saved to: {self.save_dir}")
    
    def create_world_model_env(self, world_model, train_data: List[Dict]):
        """Create world model environment."""
        def make_env():
                        
            env = WorldModelSimulationEnv(
                world_model=world_model,
                video_data=train_data,
                config=self.config.get('rl_training', {}),
                logger=self.logger,
                device=self.device
            )
            return Monitor(env)
        
        return DummyVecEnv([make_env])
    
    def create_direct_video_env(self, train_data: List[Dict]):
        """Create direct video environment."""
        def make_env():

            env = DirectVideoEnvironment(
                video_data=train_data,
                config=self.config.get('rl_training', {}),
                device=self.device
            )
            return Monitor(env)
        
        return DummyVecEnv([make_env])
    
    def train_ppo_world_model(self, world_model, train_data: List[Dict], timesteps: int = 20000) -> Dict[str, Any]:
        """Train PPO with world model environment and proper hyperparameters."""
        
        self.logger.info("-" * 50)
        self.logger.info("🌍 Training PPO with World Model")
        self.logger.info("-" * 50)
        
        try:
            # Create environment
            env = self.create_world_model_env(world_model, train_data)
            eval_env = self.create_world_model_env(world_model, train_data)
            
            # Test environment
            obs = env.reset()
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            self.logger.info(f"✅ Environment test: Reward={reward[0]:.3f}, Action shape={action.shape}")
            env.reset()

            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=1e-5,      # LOWER learning rate for stability
                n_steps=128,             # SMALLER steps for more frequent updates
                batch_size=32,           # SMALLER batch size
                n_epochs=20,             # MORE epochs per update for better learning
                gamma=0.98,              # HIGHER gamma for long-term rewards
                gae_lambda=0.95,         # HIGHER lambda for advantage estimation
                clip_range=0.1,          # LOWER clip range for stability
                ent_coef=0.02,           # LOWER entropy for more focused actions
                vf_coef=0.5,             # Value function coefficient
                max_grad_norm=0.5,       # Gradient clipping
                verbose=1,
                device='cpu',
                policy_kwargs={
                    'net_arch': [256, 256, 128],
                    'activation_fn': torch.nn.ReLU
                },
                tensorboard_log=str(self.save_dir / 'tensorboard')
            )
            
            # Enhanced monitoring callback
            monitor_callback = RLMonitoringCallback(
                eval_env=eval_env,
                save_dir=str(self.save_dir / 'world_model_ppo'),
                eval_freq=max(timesteps // 20, 1000)
            )
            
            self.logger.info(f"🚀 Training for {timesteps} timesteps with enhanced monitoring...")
            
            # Train with monitoring
            model.learn(
                total_timesteps=timesteps,
                callback=monitor_callback,
                tb_log_name="PPO_WorldModel",
                progress_bar=True
            )

            # ADD THIS OPTIMIZATION STEP HERE:
            self.logger.info("🎯 Optimizing action threshold...")
            try:
                optimal_threshold, threshold_map = self.optimize_action_threshold(model, train_data[:2])  # Use 2 videos for speed
            except Exception as e:
                self.logger.info(f"⚠️ Threshold optimization failed: {e}")
                optimal_threshold, threshold_map = 0.5, 0.0

            
            # Evaluation on the environment
            mean_reward, std_reward = evaluate_policy(
                model, eval_env, n_eval_episodes=10, deterministic=True
            )
            
            # Save final model
            final_model_path = self.save_dir / 'ppo_world_model_final.zip'
            model.save(str(final_model_path))
            
            # Get environment statistics
            base_env = env.envs[0].env
            episode_stats = base_env.get_episode_stats()
            
            result = {
                'algorithm': 'PPO_WorldModel',
                'mean_reward': float(mean_reward),
                'std_reward': float(std_reward),
                'model_path': str(final_model_path),
                'status': 'success',
                'training_timesteps': timesteps,
                'episode_stats': episode_stats,
                'monitoring_data': monitor_callback.evaluations,
                'expert_matching_enabled': True,
                'reward_design': 'expert_demonstration_matching',
                'optimal_threshold': optimal_threshold,
                'threshold_map': threshold_map
            }
            
            self.logger.info(f"✅ PPO World Model training completed!")
            self.logger.info(f"📊 Mean Reward: {mean_reward:.3f} ± {std_reward:.3f}")
            self.logger.info(f"📊 Episode stats: {episode_stats}")
            
            # Plot training curves
            self._plot_training_results(monitor_callback, 'world_model_ppo')
            
            return result
            
        except Exception as e:
            self.logger.info(f"❌ PPO World Model training failed: {e}")
            import traceback
            traceback.print_exc()
            return {'algorithm': 'PPO_WorldModel', 'status': 'failed', 'error': str(e)}
    
    def train_ppo_direct_video(self, train_data: List[Dict], timesteps: int = 20000) -> Dict[str, Any]:
        """Train PPO with direct video environment."""
        
        self.logger.info("-" * 50)
        self.logger.info("🎬 Training PPO with Direct Video")
        self.logger.info("-" * 50)
        
        try:
            # Create environment
            env = self.create_direct_video_env(train_data)
            eval_env = self.create_direct_video_env(train_data)
            
            # Test environment
            obs = env.reset()
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            self.logger.info(f"✅ Environment test: Reward={reward[0]:.3f}, Action shape={action.shape}")
            env.reset()
            
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=1e-4,      # LOWER learning rate for stability
                n_steps=256,             # SMALLER steps for more frequent updates
                batch_size=32,           # SMALLER batch size
                n_epochs=20,             # MORE epochs per update for better learning
                gamma=0.98,              # HIGHER gamma for long-term rewards
                gae_lambda=0.95,         # HIGHER lambda for advantage estimation
                clip_range=0.1,          # LOWER clip range for stability
                ent_coef=0.02,           # LOWER entropy for more focused actions
                vf_coef=0.5,             # Value function coefficient
                max_grad_norm=0.5,       # Gradient clipping
                verbose=1,
                device='cpu',
                policy_kwargs={
                    'net_arch': [256, 256, 128],
                    'activation_fn': torch.nn.ReLU
                },
                tensorboard_log=str(self.save_dir / 'tensorboard')
            )
            
            # Enhanced monitoring
            monitor_callback = RLMonitoringCallback(
                eval_env=eval_env,
                save_dir=str(self.save_dir / 'direct_video_ppo'),
                eval_freq=max(timesteps // 20, 1000)
            )
            
            self.logger.info(f"🚀 Training for {timesteps} timesteps...")
            
            # Train
            model.learn(
                total_timesteps=timesteps,
                callback=monitor_callback,
                tb_log_name="PPO_DirectVideo",
                progress_bar=True
            )

            # ADD THIS OPTIMIZATION STEP HERE:
            self.logger.info("🎯 Optimizing action threshold...")
            try:
                optimal_threshold, threshold_map = self.optimize_action_threshold(model, train_data[:2])  # Use 2 videos for speed
            except Exception as e:
                self.logger.info(f"⚠️ Threshold optimization failed: {e}")
                optimal_threshold, threshold_map = 0.5, 0.0

            # Final evaluation
            mean_reward, std_reward = evaluate_policy(
                model, eval_env, n_eval_episodes=10, deterministic=True
            )
            
            # Save final model
            final_model_path = self.save_dir / 'ppo_direct_video_final.zip'
            model.save(str(final_model_path))
            
            # Get environment statistics
            base_env = env.envs[0].env
            episode_stats = base_env.get_episode_stats()
            
            result = {
                'algorithm': 'PPO_DirectVideo',
                'mean_reward': float(mean_reward),
                'std_reward': float(std_reward),
                'model_path': str(final_model_path),
                'status': 'success',
                'training_timesteps': timesteps,
                'episode_stats': episode_stats,
                'monitoring_data': monitor_callback.evaluations,
                'expert_matching_enabled': True,
                'reward_design': 'expert_demonstration_matching',
                'optimal_threshold': optimal_threshold,
                'threshold_map': threshold_map
            }
            
            self.logger.info(f"✅ PPO Direct Video training completed!")
            self.logger.info(f"📊 Mean Reward: {mean_reward:.3f} ± {std_reward:.3f}")
            
            # Plot training curves
            self._plot_training_results(monitor_callback, 'direct_video_ppo')
            
            return result
            
        except Exception as e:
            self.logger.info(f"❌ PPO Direct Video training failed: {e}")
            import traceback
            traceback.print_exc()
            return {'algorithm': 'PPO_DirectVideo', 'status': 'failed', 'error': str(e)}

    def convert_continuous_to_binary(self, action):
        # Use adaptive thresholding instead of fixed 0.5
        threshold = np.percentile(action, 85)  # Top 15% of actions
        threshold = max(threshold, 0.4)  # Minimum threshold
        binary_action = (action > threshold).astype(np.float32)
        
        # Cap at max 3 actions
        if np.sum(binary_action) > 3:
            top_3_indices = np.argsort(action)[-3:]
            binary_action = np.zeros(100)
            binary_action[top_3_indices] = 1.0
        
        return binary_action

    def optimize_action_threshold(self, rl_model, test_data):
        """Find optimal threshold for action prediction after training"""
        
        self.logger.info("🎯 Optimizing action threshold for mAP...")
        
        best_threshold = 0.5
        best_map = 0.0
        
        # Test different thresholds
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
        
        for threshold in thresholds:
            all_preds = []
            all_targets = []
            
            # Collect predictions from test data
            for video_data in test_data:
                states = video_data['frame_embeddings'][:50]  # Sample frames
                actions = video_data['actions_binaries'][:50]
                
                for i in range(len(states)):
                    state = states[i].reshape(1, -1)
                    action_pred, _ = rl_model.predict(state, deterministic=True)
                    
                    # Apply threshold
                    if isinstance(action_pred, np.ndarray):
                        action_pred = action_pred.flatten()
                    
                    # Ensure correct size
                    if len(action_pred) != 100:
                        padded_action = np.zeros(100)
                        if len(action_pred) > 0:
                            copy_len = min(len(action_pred), 100)
                            padded_action[:copy_len] = action_pred[:copy_len]
                        action_pred = padded_action
                    
                    binary_pred = (action_pred > threshold).astype(int)
                    all_preds.append(binary_pred)
                    all_targets.append(actions[i])
            
            # Calculate mAP
            if all_preds and all_targets:
                all_preds = np.array(all_preds)
                all_targets = np.array(all_targets)
                
                from sklearn.metrics import average_precision_score
                ap_scores = []
                for action_idx in range(100):
                    gt_action = all_targets[:, action_idx]
                    if np.sum(gt_action) > 0:
                        try:
                            ap = average_precision_score(gt_action, all_preds[:, action_idx])
                            ap_scores.append(ap)
                        except:
                            ap_scores.append(0.0)
                
                map_score = np.mean(ap_scores) if ap_scores else 0.0
                self.logger.info(f"   Threshold {threshold}: mAP = {map_score:.4f}")
                
                if map_score > best_map:
                    best_map = map_score
                    best_threshold = threshold
        
        self.logger.info(f"✅ Best threshold: {best_threshold} (mAP: {best_map:.4f})")
        return best_threshold, best_map

    def _plot_training_results(self, monitor_callback, method_name: str):
        """Plot comprehensive training results."""
        
        if not monitor_callback.evaluations:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Evaluation rewards over time
        eval_steps = [e['timestep'] for e in monitor_callback.evaluations]
        eval_rewards = [e['mean_reward'] for e in monitor_callback.evaluations]
        eval_stds = [e['std_reward'] for e in monitor_callback.evaluations]
        
        axes[0, 0].plot(eval_steps, eval_rewards, 'b-', linewidth=2, label='Evaluation Reward')
        axes[0, 0].fill_between(eval_steps, 
                               np.array(eval_rewards) - np.array(eval_stds),
                               np.array(eval_rewards) + np.array(eval_stds),
                               alpha=0.3)
        axes[0, 0].set_title('Evaluation Performance')
        axes[0, 0].set_xlabel('Training Steps')
        axes[0, 0].set_ylabel('Mean Reward')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # 2. Episode rewards (if available)
        if monitor_callback.episode_rewards:
            episode_rewards = list(monitor_callback.episode_rewards)
            axes[0, 1].plot(episode_rewards, alpha=0.6, label='Episode Rewards')
            # Moving average
            if len(episode_rewards) > 10:
                moving_avg = []
                for i in range(10, len(episode_rewards)):
                    moving_avg.append(np.mean(episode_rewards[i-10:i]))
                axes[0, 1].plot(range(10, len(episode_rewards)), moving_avg, 
                               'r-', linewidth=2, label='10-Episode Moving Average')
            axes[0, 1].set_title('Episode Rewards')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Reward')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].legend()
        
        # 3. Expert matching scores (if available)
        if monitor_callback.expert_matching_scores:
            expert_scores = list(monitor_callback.expert_matching_scores)
            axes[1, 0].plot(expert_scores, 'g-', alpha=0.7, label='Expert Matching')
            axes[1, 0].axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='50% Match')
            axes[1, 0].set_title('Expert Action Matching')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Matching Score')
            axes[1, 0].set_ylim([0, 1])
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].legend()
        
        # 4. Action density over time
        if monitor_callback.action_densities:
            action_densities = list(monitor_callback.action_densities)
            axes[1, 1].plot(action_densities, 'orange', alpha=0.7, label='Actions per Step')
            axes[1, 1].axhline(y=3, color='b', linestyle='--', alpha=0.5, label='Target Range')
            axes[1, 1].axhline(y=5, color='b', linestyle='--', alpha=0.5)
            axes[1, 1].set_title('Action Density')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Number of Actions')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].legend()
        
        plt.tight_layout()
        plot_path = self.save_dir / f'{method_name}_training_curves.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"📊 Training curves saved to {plot_path}")


class RLMonitoringCallback(BaseCallback):
    """
    Custom callback for monitoring RL training progress with surgical-specific metrics.
    """
    
    def __init__(self, eval_env, save_dir: str, eval_freq: int = 1000):
        super().__init__()
        self.eval_env = eval_env
        self.save_dir = Path(save_dir)
        self.eval_freq = eval_freq
        
        # Metrics tracking
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.expert_matching_scores = deque(maxlen=100)
        self.action_densities = deque(maxlen=100)
        
        # Training progress
        self.evaluations = []
        self.eval_episodes = []
        self.best_mean_reward = -np.inf
        
    def _on_step(self) -> bool:
        """Called at each training step."""
        
        # Extract episode info from the environment
        if 'episode' in self.locals:
            episode_info = self.locals.get('infos', [{}])[0]
            
            if 'episode' in episode_info:
                ep_info = episode_info['episode']
                self.episode_rewards.append(ep_info['r'])
                self.episode_lengths.append(ep_info['l'])
                
                # Extract surgical-specific metrics
                if 'episode_expert_matching' in episode_info:
                    self.expert_matching_scores.append(episode_info['episode_expert_matching'])
                
                if 'action_sum' in episode_info:
                    self.action_densities.append(episode_info['action_sum'])
                
                # Log progress every 50 episodes
                if len(self.episode_rewards) % 50 == 0:
                    self._log_training_progress()
        
        # Evaluation
        if self.num_timesteps % self.eval_freq == 0:
            self._evaluate_model()
        
        return True
    
    def _log_training_progress(self):
        """Log training progress with surgical metrics."""
        if len(self.episode_rewards) < 10:
            return
        
        recent_rewards = list(self.episode_rewards)[-10:]
        recent_lengths = list(self.episode_lengths)[-10:]
        recent_expert_scores = list(self.expert_matching_scores)[-10:] if self.expert_matching_scores else [0]
        recent_action_densities = list(self.action_densities)[-10:] if self.action_densities else [0]
        
        print(f"\n📊 Training Progress (Step {self.num_timesteps}):")
        print(f"   Avg Reward: {np.mean(recent_rewards):.3f} ± {np.std(recent_rewards):.3f}")
        print(f"   Avg Length: {np.mean(recent_lengths):.1f}")
        print(f"   Expert Matching: {np.mean(recent_expert_scores):.3f}")
        print(f"   Avg Actions/Step: {np.mean(recent_action_densities):.1f}")
        
        # Check for learning progress
        if len(self.episode_rewards) >= 50:
            early_rewards = list(self.episode_rewards)[-50:-25]
            late_rewards = list(self.episode_rewards)[-25:]
            improvement = np.mean(late_rewards) - np.mean(early_rewards)
            
            if improvement > 0.1:
                print(f"   📈 Learning progress: +{improvement:.3f}")
            elif improvement < -0.1:
                print(f"   📉 Performance declining: {improvement:.3f}")
            else:
                print(f"   ➡️  Stable performance: {improvement:.3f}")
    
    def _evaluate_model(self):
        """Evaluate model performance."""
        mean_reward, std_reward = evaluate_policy(
            self.model, self.eval_env, n_eval_episodes=5, deterministic=True
        )
        
        self.evaluations.append({
            'timestep': self.num_timesteps,
            'mean_reward': mean_reward,
            'std_reward': std_reward
        })
        
        print(f"🔍 Evaluation: {mean_reward:.3f} ± {std_reward:.3f}")
        
        # Save best model
        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
            best_model_path = self.save_dir / 'best_model.zip'
            self.model.save(str(best_model_path))
            print(f"💾 New best model saved: {mean_reward:.3f}")


# Usage example and integration
def run_rl_training(config, logger, world_model, train_data, timesteps=20000):
    """Run RL training with proper monitoring."""
    
    logger.info("🔧 RUNNING RL TRAINING")
    logger.info("=" * 50)
    
    trainer = WorldModelRLTrainer(config, logger)
    results = {}
    
    # Train World Model RL (Method 2)
    if world_model is not None:
        logger.info("\n🌍 Training World Model RL...")
        results['world_model_rl'] = trainer.train_ppo_world_model(
            world_model, train_data, timesteps
        )
    
    # Train Direct Video RL (Method 3)
    logger.info("\n🎬 Training Direct Video RL...")
    results['direct_video_rl'] = trainer.train_ppo_direct_video(
        train_data, timesteps
    )
    
    # Print summary
    logger.info("\n📊 RL TRAINING SUMMARY:")
    for method, result in results.items():
        if result.get('status') == 'success':
            logger.info(f"✅ {method}: {result['mean_reward']:.3f} ± {result['std_reward']:.3f}")
            if 'episode_stats' in result:
                stats = result['episode_stats']
                if 'avg_expert_matching' in stats:
                    logger.info(f"   Expert matching: {stats['avg_expert_matching']:.3f}")
        else:
            logger.info(f"❌ {method}: Failed - {result.get('error', 'Unknown error')}")
    
    return results


if __name__ == "__main__":
    print("🔧 RL TRAINING")
    print("=" * 50)
    print("✅ Optimized hyperparameters for surgical tasks")
    print("✅ Expert demonstration matching rewards")
    print("✅ Comprehensive monitoring and debugging")
    print("✅ Proper action space handling")
    print("✅ Enhanced convergence detection")
