# ===================================================================
# File: rl_trainer.py
# RL Training Pipeline
# ===================================================================

import os
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
import json
from torch.utils.tensorboard import SummaryWriter

from models import WorldModel, SurgicalWorldModelEnv
import gymnasium as gym
from gymnasium import spaces

class RLExperimentRunner:
    """
    Fixed RL experiment runner with proper action handling
    """
    
    def __init__(self, config: dict, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize tensorboard writer
        self.tb_writer = SummaryWriter(log_dir='./rl_tensorboard_logs')
        
        # Initialize world model and environment
        self.world_model = None
        self.env = None
        
        # Results storage
        self.results = {
            'baseline_imitation': {},
            'rl_algorithms': {}
        }
        
    def load_world_model(self, model_path: str):
        """Load pre-trained world model"""
        self.logger.info(f"Loading world model from {model_path}")
        
        # Load your world model
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Reconstruct model with config
        model_config = self.config['models']['world_model']
        self.world_model = WorldModel(**model_config).to(self.device)
        self.world_model.load_state_dict(checkpoint['model_state_dict'])
        self.world_model.eval()
        
        # Create environment with fixed class
        env_config = {
            'rl_horizon': self.config.get('rl_horizon', 50),
            'context_length': self.config['data']['context_length'],
            'reward_weights': self.config.get('reward_weights', {})
        }
        
        self.env = SurgicalWorldModelEnv(self.world_model, env_config, self.device)
        self.logger.info("World model and environment initialized successfully")
    
    def _evaluate_behavioral_cloning_in_env(self, val_data: List[Dict]) -> Dict:
        """Fixed behavioral cloning evaluation with proper action handling"""
        
        episode_rewards = []
        episode_lengths = []
        
        # Test on a few videos
        test_videos = val_data[:3] if len(val_data) > 3 else val_data  # Reduce for debugging
        
        for video in test_videos:
            video_id = video['video_id']
            self.logger.info(f"Testing behavioral cloning on video: {video_id}")
            
            # Reset environment with this video
            obs, _ = self.env.reset(options={'video_id': video_id})
            
            episode_reward = 0
            episode_length = 0
            
            for step in range(self.config.get('eval_horizon', 30)):
                # Use world model to predict action (behavioral cloning)
                with torch.no_grad():
                    # Ensure state tensor has correct shape
                    state_tensor = torch.tensor(obs, device=self.device, dtype=torch.float32)
                    
                    # Add batch dimension if needed
                    if state_tensor.dim() == 1:
                        state_tensor = state_tensor.unsqueeze(0)  # [1, embedding_dim]
                    
                    # Predict action using world model
                    action_probs = self.world_model.predict_next_action(state_tensor)
                    
                    # Convert to binary action - be careful with dimensions
                    if action_probs.dim() == 3:  # [batch, seq, actions]
                        action_probs = action_probs.squeeze(0).squeeze(0)  # [actions]
                    elif action_probs.dim() == 2:  # [batch, actions] or [seq, actions]
                        action_probs = action_probs.squeeze(0)  # [actions]
                    
                    # Convert to numpy and threshold
                    action = (action_probs.cpu().numpy() > 0.5).astype(np.float32)
                    
                    # Ensure action has correct shape [num_action_classes]
                    if action.shape != (self.world_model.num_action_classes,):
                        self.logger.warning(f"Action shape mismatch: {action.shape}, expected ({self.world_model.num_action_classes},)")
                        # Fix by reshaping or padding
                        action = action.flatten()[:self.world_model.num_action_classes]
                        if len(action) < self.world_model.num_action_classes:
                            action = np.pad(action, (0, self.world_model.num_action_classes - len(action)))
                
                # Take step in environment
                obs, reward, done, truncated, info = self.env.step(action)
                episode_reward += reward
                episode_length += 1
                
                if done or truncated:
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            self.logger.info(f"Video {video_id}: reward={episode_reward:.3f}, length={episode_length}")
        
        return {
            'avg_episode_reward': np.mean(episode_rewards),
            'std_episode_reward': np.std(episode_rewards),
            'avg_episode_length': np.mean(episode_lengths),
            'num_episodes': len(episode_rewards),
            'episode_rewards': episode_rewards  # For debugging
        }
    
    def run_baseline_imitation_learning(self, train_data: List[Dict], val_data: List[Dict]):
        """Run baseline IL with better error handling"""
        self.logger.info("=== Running Baseline Imitation Learning ===")
        
        from datasets.cholect50 import NextFramePredictionDataset
        
        # Create validation dataset
        val_dataset = NextFramePredictionDataset(self.config['data'], val_data)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        
        # Evaluate imitation learning performance
        il_metrics = self._evaluate_imitation_learning(val_loader)
        
        # Test behavioral cloning in the environment
        self.logger.info("Testing behavioral cloning in environment...")
        try:
            env_metrics = self._evaluate_behavioral_cloning_in_env(val_data)
        except Exception as e:
            self.logger.error(f"Error in behavioral cloning evaluation: {e}")
            env_metrics = {
                'avg_episode_reward': 0.0,
                'std_episode_reward': 0.0,
                'avg_episode_length': 0.0,
                'num_episodes': 0,
                'error': str(e)
            }
        
        # Log to tensorboard
        self.tb_writer.add_scalar('Baseline_IL/World_Model_Loss', il_metrics['avg_loss'], 0)
        self.tb_writer.add_scalar('Baseline_IL/Action_Accuracy', il_metrics['action_accuracy'], 0)
        self.tb_writer.add_scalar('Baseline_IL/Env_Avg_Reward', env_metrics['avg_episode_reward'], 0)
        
        self.results['baseline_imitation'] = {
            'world_model_metrics': il_metrics,
            'environment_metrics': env_metrics
        }
        
        self.logger.info(f"Baseline IL Results: {self.results['baseline_imitation']}")
        return self.results['baseline_imitation']
    
    def _evaluate_imitation_learning(self, val_loader: DataLoader) -> Dict:
        """Evaluate imitation learning performance"""
        self.world_model.eval()
        
        total_loss = 0.0
        action_accuracy = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating IL"):
                # Move to device
                current_states = batch['current_states'].to(self.device)
                next_states = batch['next_states'].to(self.device)
                next_actions = batch['next_actions'].to(self.device)
                
                # Forward pass with proper eval mode
                output = self.world_model(
                    current_state=current_states,
                    next_state=next_states,
                    next_actions=next_actions,
                    eval_mode='training'  # Use training mode for loss computation
                )
                
                total_loss += output['total_loss'].item()
                
                # Calculate action prediction accuracy if available
                if '_a_hat' in output and output['_a_hat'] is not None:
                    pred_actions = torch.sigmoid(output['_a_hat'])
                    pred_binary = (pred_actions > 0.5).float()
                    accuracy = (pred_binary == next_actions).float().mean()
                    action_accuracy += accuracy.item()
                
                num_batches += 1
        
        return {
            'avg_loss': total_loss / num_batches,
            'action_accuracy': action_accuracy / num_batches,
            'num_batches': num_batches
        }
    
    def run_rl_experiments(self, train_data: List[Dict], algorithms: List[str] = ['ppo', 'sac']):
        """
        Run RL experiments with different algorithms
        """
        self.logger.info("=== Running RL Experiments ===")
        
        # Set video context for environment
        self.env.set_video_context(train_data)
        
        for algorithm in algorithms:
            self.logger.info(f"Training with {algorithm.upper()}")
            
            try:
                if algorithm.lower() == 'ppo':
                    results = self._train_ppo()
                elif algorithm.lower() == 'sac':
                    results = self._train_sac()
                elif algorithm.lower() == 'td_mpc2':
                    results = self._train_td_mpc2()
                else:
                    self.logger.warning(f"Unknown algorithm: {algorithm}")
                    continue
                    
                self.results['rl_algorithms'][algorithm] = results
                
            except Exception as e:
                self.logger.error(f"Error training {algorithm}: {e}")
                self.results['rl_algorithms'][algorithm] = {'error': str(e)}
        
        return self.results['rl_algorithms']
    
    def _train_ppo(self) -> Dict:
        """Train PPO policy"""
        try:
            from stable_baselines3 import PPO
            from stable_baselines3.common.vec_env import DummyVecEnv
            from stable_baselines3.common.callbacks import EvalCallback
            
            # Wrap environment
            def make_env():
                return self.env
            
            vec_env = DummyVecEnv([make_env])
            
            # Create PPO model
            model = PPO(
                "MlpPolicy", 
                vec_env, 
                verbose=1,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                tensorboard_log="./ppo_surgical_logs/"
            )
            
            # Training
            total_timesteps = self.config.get('rl_timesteps', 50000)
            model.learn(total_timesteps=total_timesteps)
            
            # Save model
            model.save("surgical_ppo_policy")
            
            # Evaluate
            eval_results = self._evaluate_rl_policy(model, algorithm='ppo')
            
            return {
                'training_timesteps': total_timesteps,
                'evaluation': eval_results,
                'model_path': "surgical_ppo_policy.zip"
            }
            
        except ImportError:
            return {'error': 'stable-baselines3 not available'}
    
    def _train_sac(self) -> Dict:
        """Train SAC policy"""
        try:
            from stable_baselines3 import SAC
            from stable_baselines3.common.vec_env import DummyVecEnv
            
            # Convert to continuous action space for SAC
            original_action_space = self.env.action_space
            self.env.action_space = spaces.Box(
                low=0, high=1, 
                shape=(self.world_model.num_action_classes,), 
                dtype=np.float32
            )
            
            def make_env():
                return self.env
            
            vec_env = DummyVecEnv([make_env])
            
            # Create SAC model
            model = SAC(
                "MlpPolicy",
                vec_env,
                verbose=1,
                learning_rate=3e-4,
                buffer_size=100000,
                learning_starts=1000,
                batch_size=256,
                tau=0.005,
                gamma=0.99,
                train_freq=1,
                gradient_steps=1,
                tensorboard_log="./sac_surgical_logs/"
            )
            
            # Training
            total_timesteps = self.config.get('rl_timesteps', 50000)
            model.learn(total_timesteps=total_timesteps)
            
            # Save model
            model.save("surgical_sac_policy")
            
            # Evaluate
            eval_results = self._evaluate_rl_policy(model, algorithm='sac')
            
            # Restore original action space
            self.env.action_space = original_action_space
            
            return {
                'training_timesteps': total_timesteps,
                'evaluation': eval_results,
                'model_path': "surgical_sac_policy.zip"
            }
            
        except ImportError:
            return {'error': 'stable-baselines3 not available'}
    
    def _train_td_mpc2(self) -> Dict:
        """Implement TD-MPC2 training"""
        # This is a simplified TD-MPC2 implementation
        # You would need to implement the full TD-MPC2 algorithm
        
        self.logger.info("TD-MPC2 implementation placeholder")
        return {
            'status': 'not_implemented',
            'note': 'TD-MPC2 requires custom implementation'
        }
    
    def _evaluate_imitation_learning(self, val_loader: DataLoader) -> Dict:
        """Evaluate imitation learning performance"""
        self.world_model.eval()
        
        total_loss = 0.0
        action_accuracy = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating IL"):
                # Move to device
                current_states = batch['current_states'].to(self.device)
                next_states = batch['next_states'].to(self.device)
                next_actions = batch['next_actions'].to(self.device)
                
                # Forward pass
                output = self.world_model(
                    current_state=current_states,
                    next_state=next_states,
                    next_actions=next_actions
                )
                
                total_loss += output['total_loss'].item()
                
                # Calculate action prediction accuracy if available
                if '_a_hat' in output and output['_a_hat'] is not None:
                    pred_actions = torch.sigmoid(output['_a_hat'])
                    pred_binary = (pred_actions > 0.5).float()
                    accuracy = (pred_binary == next_actions).float().mean()
                    action_accuracy += accuracy.item()
                
                num_batches += 1
        
        return {
            'avg_loss': total_loss / num_batches,
            'action_accuracy': action_accuracy / num_batches,
            'num_batches': num_batches
        }
    
    def _evaluate_behavioral_cloning_in_env(self, val_data: List[Dict]) -> Dict:
        """Evaluate behavioral cloning policy in environment"""
        
        episode_rewards = []
        episode_lengths = []
        
        # Test on a few videos
        test_videos = val_data[:5] if len(val_data) > 5 else val_data
        
        for video in test_videos:
            video_id = video['video_id']
            
            # Reset environment with this video
            obs, _ = self.env.reset(options={'video_id': video_id})
            
            episode_reward = 0
            episode_length = 0
            
            for _ in range(self.config.get('eval_horizon', 30)):
                # Use world model to predict action (behavioral cloning)
                with torch.no_grad():
                    state_tensor = torch.tensor(obs, device=self.device).unsqueeze(0)
                    action_probs = self.world_model.predict_next_action(state_tensor)
                    action = (action_probs.squeeze(0).cpu().numpy() > 0.5).astype(np.float32)
                
                obs, reward, done, truncated, info = self.env.step(action)
                episode_reward += reward
                episode_length += 1
                
                if done or truncated:
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        
        return {
            'avg_episode_reward': np.mean(episode_rewards),
            'std_episode_reward': np.std(episode_rewards),
            'avg_episode_length': np.mean(episode_lengths),
            'num_episodes': len(episode_rewards)
        }
    
    def _evaluate_rl_policy(self, model, algorithm: str, n_episodes: int = 10) -> Dict:
        """Evaluate trained RL policy"""
        
        episode_rewards = []
        episode_lengths = []
        episode_info = []
        
        for episode in range(n_episodes):
            obs, _ = self.env.reset()
            episode_reward = 0
            episode_length = 0
            
            while True:
                if algorithm == 'sac':
                    # For SAC, convert binary to continuous
                    action, _ = model.predict(obs)
                    # Convert back to binary
                    action = (action > 0.5).astype(np.float32)
                else:
                    action, _ = model.predict(obs, deterministic=True)
                
                obs, reward, done, truncated, info = self.env.step(action)
                episode_reward += reward
                episode_length += 1
                
                if done or truncated:
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            episode_info.append({
                'reward': episode_reward,
                'length': episode_length,
                'final_phase': info.get('current_phase', -1)
            })
        
        return {
            'avg_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'avg_length': np.mean(episode_lengths),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'episodes': episode_info
        }
    
    def compare_results(self) -> Dict:
        """Compare all approaches and generate summary"""
        
        summary = {
            'comparison': {},
            'statistical_tests': {},
            'recommendations': []
        }
        
        # Extract key metrics
        il_reward = self.results['baseline_imitation']['environment_metrics']['avg_episode_reward']
        
        summary['comparison']['baseline_imitation_learning'] = {
            'avg_reward': il_reward,
            'method': 'Behavioral Cloning with World Model'
        }
        
        # Compare RL algorithms
        for alg_name, alg_results in self.results['rl_algorithms'].items():
            if 'error' not in alg_results and 'evaluation' in alg_results:
                rl_reward = alg_results['evaluation']['avg_reward']
                improvement = ((rl_reward - il_reward) / abs(il_reward)) * 100
                
                summary['comparison'][alg_name] = {
                    'avg_reward': rl_reward,
                    'improvement_over_il': f"{improvement:.2f}%",
                    'absolute_improvement': rl_reward - il_reward
                }
        
        # Generate recommendations
        best_method = max(
            summary['comparison'].items(),
            key=lambda x: x[1]['avg_reward']
        )
        
        summary['recommendations'] = [
            f"Best performing method: {best_method[0]} with avg reward {best_method[1]['avg_reward']:.4f}",
            "RL shows improvement over IL" if any(
                'improvement_over_il' in v and float(v['improvement_over_il'].replace('%', '')) > 0 
                for v in summary['comparison'].values()
            ) else "IL baseline is competitive with RL approaches",
        ]
        
        return summary
    
    def save_results(self, save_path: str = 'experiment_results.json'):
        """Save all results to file"""
        with open(save_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        self.logger.info(f"Results saved to {save_path}")
