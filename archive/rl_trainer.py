#!/usr/bin/env python3
"""
Reinforcement Learning Training Implementation for Surgical Action Prediction
Implements PPO and SAC algorithms using the world model as environment
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from typing import Dict, List, Tuple, Any, Optional
import logging
from collections import deque
import matplotlib.pyplot as plt
from pathlib import Path
import json
from tqdm import tqdm
import time

# Import your existing environment
from rl.environment import SurgicalWorldModelEnv, MultiVideoSurgicalEnv

class PPOAgent:
    """
    Proximal Policy Optimization (PPO) agent for surgical action prediction.
    """
    
    def __init__(self, 
                 observation_dim: int,
                 action_dim: int,
                 hidden_dim: int = 256,
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 clip_ratio: float = 0.2,
                 entropy_coef: float = 0.01,
                 value_coef: float = 0.5,
                 device: str = 'cuda'):
        
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.device = device
        
        # Networks
        self.actor = self._build_actor(observation_dim, action_dim, hidden_dim).to(device)
        self.critic = self._build_critic(observation_dim, hidden_dim).to(device)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        # Training tracking
        self.training_info = {
            'episode_rewards': [],
            'policy_losses': [],
            'value_losses': [],
            'entropy_losses': []
        }
    
    def _build_actor(self, obs_dim: int, act_dim: int, hidden_dim: int) -> nn.Module:
        """Build actor network for policy."""
        return nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim),
            nn.Sigmoid()  # For binary actions
        )
    
    def _build_critic(self, obs_dim: int, hidden_dim: int) -> nn.Module:
        """Build critic network for value function."""
        return nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def get_action(self, observation: np.ndarray, training: bool = True) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        """Get action from policy."""
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_probs = self.actor(obs_tensor)
            value = self.critic(obs_tensor)
            
            if training:
                # Sample actions during training
                action_dist = torch.distributions.Bernoulli(action_probs)
                action = action_dist.sample()
                log_prob = action_dist.log_prob(action).sum(dim=-1)
            else:
                # Use deterministic actions during evaluation
                action = (action_probs > 0.5).float()
                log_prob = torch.zeros(1).to(self.device)
        
        return action.cpu().numpy().flatten(), log_prob, value.squeeze()
    
    def update(self, 
               observations: List[np.ndarray],
               actions: List[np.ndarray],
               rewards: List[float],
               log_probs: List[torch.Tensor],
               values: List[torch.Tensor],
               next_values: List[torch.Tensor],
               dones: List[bool]) -> Dict[str, float]:
        """Update policy using PPO."""
        
        # Convert to tensors
        obs_tensor = torch.FloatTensor(np.array(observations)).to(self.device)
        action_tensor = torch.FloatTensor(np.array(actions)).to(self.device)
        old_log_probs = torch.stack(log_probs).to(self.device)
        old_values = torch.stack(values).to(self.device)
        
        # Compute advantages using GAE
        advantages = self._compute_gae(rewards, values, next_values, dones)
        returns = advantages + old_values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        policy_loss_total = 0
        value_loss_total = 0
        entropy_loss_total = 0
        
        # Multiple epochs of updates
        for _ in range(10):  # PPO typically uses multiple epochs
            # Current policy predictions
            current_action_probs = self.actor(obs_tensor)
            current_values = self.critic(obs_tensor).squeeze()
            
            # Policy loss
            action_dist = torch.distributions.Bernoulli(current_action_probs)
            new_log_probs = action_dist.log_prob(action_tensor).sum(dim=-1)
            entropy = action_dist.entropy().sum(dim=-1).mean()
            
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = nn.MSELoss()(current_values, returns)
            
            # Total loss
            total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
            
            # Update networks
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            total_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()
            
            policy_loss_total += policy_loss.item()
            value_loss_total += value_loss.item()
            entropy_loss_total += entropy.item()
        
        return {
            'policy_loss': policy_loss_total / 10,
            'value_loss': value_loss_total / 10,
            'entropy_loss': entropy_loss_total / 10
        }
    
    def _compute_gae(self, 
                    rewards: List[float], 
                    values: List[torch.Tensor], 
                    next_values: List[torch.Tensor], 
                    dones: List[bool],
                    gae_lambda: float = 0.95) -> torch.Tensor:
        """Compute Generalized Advantage Estimation."""
        advantages = []
        gae = 0
        
        for i in reversed(range(len(rewards))):
            if dones[i]:
                delta = rewards[i] - values[i].item()
                gae = delta
            else:
                delta = rewards[i] + self.gamma * next_values[i].item() - values[i].item()
                gae = delta + self.gamma * gae_lambda * gae
            
            advantages.insert(0, gae)
        
        return torch.FloatTensor(advantages).to(self.device)
    
    def save(self, path: str):
        """Save the model."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'training_info': self.training_info
        }, path)
    
    def load(self, path: str):
        """Load the model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.training_info = checkpoint['training_info']


class SACAgent:
    """
    Soft Actor-Critic (SAC) agent for surgical action prediction.
    """
    
    def __init__(self, 
                 observation_dim: int,
                 action_dim: int,
                 hidden_dim: int = 256,
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 alpha: float = 0.2,
                 device: str = 'cuda'):
        
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.device = device
        
        # Networks
        self.actor = self._build_actor(observation_dim, action_dim, hidden_dim).to(device)
        self.critic1 = self._build_critic(observation_dim, action_dim, hidden_dim).to(device)
        self.critic2 = self._build_critic(observation_dim, action_dim, hidden_dim).to(device)
        self.target_critic1 = self._build_critic(observation_dim, action_dim, hidden_dim).to(device)
        self.target_critic2 = self._build_critic(observation_dim, action_dim, hidden_dim).to(device)
        
        # Copy parameters to target networks
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr)
        
        # Replay buffer
        self.replay_buffer = deque(maxlen=100000)
        
        # Training tracking
        self.training_info = {
            'episode_rewards': [],
            'actor_losses': [],
            'critic_losses': []
        }
    
    def _build_actor(self, obs_dim: int, act_dim: int, hidden_dim: int) -> nn.Module:
        """Build actor network."""
        return nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim),
            nn.Sigmoid()  # For binary actions
        )
    
    def _build_critic(self, obs_dim: int, act_dim: int, hidden_dim: int) -> nn.Module:
        """Build critic network."""
        return nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def get_action(self, observation: np.ndarray, training: bool = True) -> np.ndarray:
        """Get action from policy."""
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_probs = self.actor(obs_tensor)
            
            if training:
                # Add noise for exploration
                noise = torch.randn_like(action_probs) * 0.1
                action_probs = torch.clamp(action_probs + noise, 0, 1)
            
            action = (action_probs > 0.5).float()
        
        return action.cpu().numpy().flatten()
    
    def store_transition(self, 
                        observation: np.ndarray,
                        action: np.ndarray,
                        reward: float,
                        next_observation: np.ndarray,
                        done: bool):
        """Store transition in replay buffer."""
        self.replay_buffer.append((observation, action, reward, next_observation, done))
    
    def update(self, batch_size: int = 256) -> Dict[str, float]:
        """Update networks using SAC."""
        if len(self.replay_buffer) < batch_size:
            return {}
        
        # Sample batch
        batch = list(np.random.choice(self.replay_buffer, batch_size, replace=False))
        observations, actions, rewards, next_observations, dones = zip(*batch)
        
        # Convert to tensors
        obs_tensor = torch.FloatTensor(np.array(observations)).to(self.device)
        action_tensor = torch.FloatTensor(np.array(actions)).to(self.device)
        reward_tensor = torch.FloatTensor(rewards).to(self.device)
        next_obs_tensor = torch.FloatTensor(np.array(next_observations)).to(self.device)
        done_tensor = torch.BoolTensor(dones).to(self.device)
        
        # Update critics
        with torch.no_grad():
            next_actions = self.actor(next_obs_tensor)
            next_q1 = self.target_critic1(torch.cat([next_obs_tensor, next_actions], dim=1))
            next_q2 = self.target_critic2(torch.cat([next_obs_tensor, next_actions], dim=1))
            next_q = torch.min(next_q1, next_q2).squeeze()
            target_q = reward_tensor + self.gamma * next_q * (~done_tensor)
        
        current_q1 = self.critic1(torch.cat([obs_tensor, action_tensor], dim=1)).squeeze()
        current_q2 = self.critic2(torch.cat([obs_tensor, action_tensor], dim=1)).squeeze()
        
        critic1_loss = nn.MSELoss()(current_q1, target_q)
        critic2_loss = nn.MSELoss()(current_q2, target_q)
        
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        # Update actor
        new_actions = self.actor(obs_tensor)
        q1_new = self.critic1(torch.cat([obs_tensor, new_actions], dim=1))
        q2_new = self.critic2(torch.cat([obs_tensor, new_actions], dim=1))
        q_new = torch.min(q1_new, q2_new)
        
        actor_loss = -q_new.mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update target networks
        self._soft_update(self.critic1, self.target_critic1)
        self._soft_update(self.critic2, self.target_critic2)
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': (critic1_loss.item() + critic2_loss.item()) / 2
        }
    
    def _soft_update(self, source: nn.Module, target: nn.Module):
        """Soft update of target network."""
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(
                self.tau * source_param.data + (1 - self.tau) * target_param.data
            )
    
    def save(self, path: str):
        """Save the model."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic1_state_dict': self.critic1.state_dict(),
            'critic2_state_dict': self.critic2.state_dict(),
            'training_info': self.training_info
        }, path)
    
    def load(self, path: str):
        """Load the model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic1.load_state_dict(checkpoint['critic1_state_dict'])
        self.critic2.load_state_dict(checkpoint['critic2_state_dict'])
        self.training_info = checkpoint['training_info']


class RLTrainer:
    """
    Main RL trainer that coordinates training of different RL algorithms.
    """
    
    def __init__(self, 
                 world_model,
                 config: Dict[str, Any],
                 logger,
                 device: str = 'cuda'):
        
        self.world_model = world_model
        self.config = config
        self.logger = logger
        self.device = device
        
        # Create save directory
        self.save_dir = Path(logger.log_dir) / 'rl_training'
        self.save_dir.mkdir(exist_ok=True)
        
        # RL config
        self.rl_config = config.get('rl_training', {})
        
    def train_ppo(self, 
                  train_data: List[Dict],
                  episodes: int = 1000,
                  eval_episodes: int = 10) -> Dict[str, Any]:
        """Train PPO agent."""
        
        print("🤖 Training PPO Agent")
        print("-" * 40)
        
        # Create environment
        env_config = {
            'rl_horizon': self.rl_config.get('rl_horizon', 50),
            'context_length': self.config['data']['context_length'],
            'reward_weights': self.rl_config.get('reward_weights', {}),
            'normalize_rewards': self.rl_config.get('normalize_rewards', True),
            'early_termination': self.rl_config.get('early_termination', True)
        }
        
        env = MultiVideoSurgicalEnv(self.world_model, env_config, train_data, self.device)
        
        # Create PPO agent
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.n
        
        ppo_config = self.rl_config.get('ppo', {})
        agent = PPOAgent(
            observation_dim=obs_dim,
            action_dim=act_dim,
            lr=ppo_config.get('learning_rate', 3e-4),
            gamma=ppo_config.get('gamma', 0.99),
            clip_ratio=ppo_config.get('clip_range', 0.2),
            device=self.device
        )
        
        # Training loop
        episode_rewards = []
        best_reward = float('-inf')
        
        for episode in tqdm(range(episodes), desc="Training PPO"):
            observations, actions, rewards, log_probs, values, next_values, dones = [], [], [], [], [], [], []
            
            obs, info = env.reset()
            episode_reward = 0
            
            for step in range(env_config['rl_horizon']):
                action, log_prob, value = agent.get_action(obs, training=True)
                next_obs, reward, terminated, truncated, info = env.step(action)
                
                # Get next value for GAE
                if not (terminated or truncated):
                    _, _, next_value = agent.get_action(next_obs, training=True)
                else:
                    next_value = torch.zeros(1).to(self.device)
                
                observations.append(obs)
                actions.append(action)
                rewards.append(reward)
                log_probs.append(log_prob)
                values.append(value)
                next_values.append(next_value)
                dones.append(terminated or truncated)
                
                obs = next_obs
                episode_reward += reward
                
                if terminated or truncated:
                    break
            
            # Update agent
            if len(observations) > 1:
                losses = agent.update(observations, actions, rewards, log_probs, values, next_values, dones)
                agent.training_info['policy_losses'].append(losses['policy_loss'])
                agent.training_info['value_losses'].append(losses['value_loss'])
                agent.training_info['entropy_losses'].append(losses['entropy_loss'])
            
            episode_rewards.append(episode_reward)
            agent.training_info['episode_rewards'].append(episode_reward)
            
            # Save best model
            if episode_reward > best_reward:
                best_reward = episode_reward
                agent.save(self.save_dir / 'ppo_best_model.pt')
            
            # Periodic logging
            if episode % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                self.logger.info(f"PPO Episode {episode}: Avg Reward = {avg_reward:.3f}")
        
        # Save final model
        agent.save(self.save_dir / 'ppo_final_model.pt')
        
        # Evaluation
        eval_results = self._evaluate_agent(agent, env, eval_episodes)
        
        results = {
            'algorithm': 'PPO',
            'training_episodes': episodes,
            'final_avg_reward': np.mean(episode_rewards[-100:]),
            'best_reward': best_reward,
            'evaluation': eval_results,
            'model_path': str(self.save_dir / 'ppo_best_model.pt')
        }
        
        # Save training plots
        self._save_training_plots(agent.training_info, 'ppo')
        
        print(f"✅ PPO training completed. Best reward: {best_reward:.3f}")
        return results
    
    def train_sac(self, 
                  train_data: List[Dict],
                  episodes: int = 1000,
                  eval_episodes: int = 10) -> Dict[str, Any]:
        """Train SAC agent."""
        
        print("🤖 Training SAC Agent")
        print("-" * 40)
        
        # Create environment
        env_config = {
            'rl_horizon': self.rl_config.get('rl_horizon', 50),
            'context_length': self.config['data']['context_length'],
            'reward_weights': self.rl_config.get('reward_weights', {}),
            'normalize_rewards': self.rl_config.get('normalize_rewards', True),
            'early_termination': self.rl_config.get('early_termination', True)
        }
        
        env = MultiVideoSurgicalEnv(self.world_model, env_config, train_data, self.device)
        
        # Create SAC agent
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.n
        
        sac_config = self.rl_config.get('sac', {})
        agent = SACAgent(
            observation_dim=obs_dim,
            action_dim=act_dim,
            lr=sac_config.get('learning_rate', 3e-4),
            gamma=sac_config.get('gamma', 0.99),
            tau=sac_config.get('tau', 0.005),
            device=self.device
        )
        
        # Training loop
        episode_rewards = []
        best_reward = float('-inf')
        
        for episode in tqdm(range(episodes), desc="Training SAC"):
            obs, info = env.reset()
            episode_reward = 0
            
            for step in range(env_config['rl_horizon']):
                action = agent.get_action(obs, training=True)
                next_obs, reward, terminated, truncated, info = env.step(action)
                
                # Store transition
                agent.store_transition(obs, action, reward, next_obs, terminated or truncated)
                
                # Update agent
                if len(agent.replay_buffer) > 1000:  # Start updating after some experience
                    losses = agent.update(batch_size=256)
                    if losses:
                        agent.training_info['actor_losses'].append(losses['actor_loss'])
                        agent.training_info['critic_losses'].append(losses['critic_loss'])
                
                obs = next_obs
                episode_reward += reward
                
                if terminated or truncated:
                    break
            
            episode_rewards.append(episode_reward)
            agent.training_info['episode_rewards'].append(episode_reward)
            
            # Save best model
            if episode_reward > best_reward:
                best_reward = episode_reward
                agent.save(self.save_dir / 'sac_best_model.pt')
            
            # Periodic logging
            if episode % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                self.logger.info(f"SAC Episode {episode}: Avg Reward = {avg_reward:.3f}")
        
        # Save final model
        agent.save(self.save_dir / 'sac_final_model.pt')
        
        # Evaluation
        eval_results = self._evaluate_agent(agent, env, eval_episodes)
        
        results = {
            'algorithm': 'SAC',
            'training_episodes': episodes,
            'final_avg_reward': np.mean(episode_rewards[-100:]),
            'best_reward': best_reward,
            'evaluation': eval_results,
            'model_path': str(self.save_dir / 'sac_best_model.pt')
        }
        
        # Save training plots
        self._save_training_plots(agent.training_info, 'sac')
        
        print(f"✅ SAC training completed. Best reward: {best_reward:.3f}")
        return results
    
    def _evaluate_agent(self, agent, env, eval_episodes: int) -> Dict[str, float]:
        """Evaluate trained agent."""
        episode_rewards = []
        episode_lengths = []
        success_rates = []
        
        for episode in range(eval_episodes):
            obs, info = env.reset()
            episode_reward = 0
            episode_length = 0
            
            for step in range(100):  # Max steps for evaluation
                action = agent.get_action(obs, training=False)
                obs, reward, terminated, truncated, info = env.step(action)
                
                episode_reward += reward
                episode_length += 1
                
                if terminated or truncated:
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            # Success rate based on final info
            success_rate = info.get('success_rate', 0.0)
            success_rates.append(success_rate)
        
        return {
            'avg_reward': float(np.mean(episode_rewards)),
            'std_reward': float(np.std(episode_rewards)),
            'avg_length': float(np.mean(episode_lengths)),
            'avg_success_rate': float(np.mean(success_rates))
        }
    
    def _save_training_plots(self, training_info: Dict, algorithm: str):
        """Save training plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Episode rewards
        if 'episode_rewards' in training_info:
            axes[0, 0].plot(training_info['episode_rewards'])
            axes[0, 0].set_title(f'{algorithm.upper()} Episode Rewards')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Reward')
            axes[0, 0].grid(True)
        
        # Loss plots depend on algorithm
        if algorithm.lower() == 'ppo':
            if 'policy_losses' in training_info:
                axes[0, 1].plot(training_info['policy_losses'])
                axes[0, 1].set_title('Policy Loss')
                axes[0, 1].grid(True)
            
            if 'value_losses' in training_info:
                axes[1, 0].plot(training_info['value_losses'])
                axes[1, 0].set_title('Value Loss')
                axes[1, 0].grid(True)
            
            if 'entropy_losses' in training_info:
                axes[1, 1].plot(training_info['entropy_losses'])
                axes[1, 1].set_title('Entropy Loss')
                axes[1, 1].grid(True)
        
        elif algorithm.lower() == 'sac':
            if 'actor_losses' in training_info:
                axes[0, 1].plot(training_info['actor_losses'])
                axes[0, 1].set_title('Actor Loss')
                axes[0, 1].grid(True)
            
            if 'critic_losses' in training_info:
                axes[1, 0].plot(training_info['critic_losses'])
                axes[1, 0].set_title('Critic Loss')
                axes[1, 0].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / f'{algorithm}_training_plots.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Training plots saved for {algorithm}")


def run_rl_training(config_path: str = 'config.yaml'):
    """Main function to run RL training."""
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup logging
    from utils.logger import SimpleLogger
    logger = SimpleLogger(log_dir="logs", name="rl_training")
    
    print("🚀 Starting RL Training for Surgical Action Prediction")
    print("=" * 60)
    
    # Load world model
    from models.dual_world_model import DualWorldModel
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load pre-trained supervised model
    supervised_model_path = "logs/2025-05-28_11-03-37/checkpoints/supervised_best_epoch_1.pt"
    if not Path(supervised_model_path).exists():
        raise FileNotFoundError(f"Supervised model not found at {supervised_model_path}")
    
    world_model = DualWorldModel.load_model(supervised_model_path, device)
    logger.info(f"Loaded world model from {supervised_model_path}")
    
    # Load training data
    from datasets.cholect50 import load_cholect50_data
    train_data = load_cholect50_data(config, logger, split='train', max_videos=10)
    
    # Create RL trainer
    trainer = RLTrainer(world_model, config, logger, device)
    
    # Train different algorithms
    results = {}
    
    algorithms = config.get('experiment', {}).get('rl_experiments', {}).get('algorithms', ['ppo'])
    episodes = config.get('experiment', {}).get('rl_experiments', {}).get('timesteps', 1000)
    
    for algorithm in algorithms:
        if algorithm.lower() == 'ppo':
            results['ppo'] = trainer.train_ppo(train_data, episodes=episodes//50)  # Convert timesteps to episodes
        elif algorithm.lower() == 'sac':
            results['sac'] = trainer.train_sac(train_data, episodes=episodes//50)
    
    # Save results
    results_path = trainer.save_dir / 'rl_training_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"RL training completed. Results saved to {results_path}")
    
    print("\n🎉 RL Training Complete!")
    print("=" * 60)
    
    for algorithm, result in results.items():
        print(f"📊 {algorithm.upper()}: Best Reward = {result['best_reward']:.3f}")
    
    return results


if __name__ == "__main__":
    import yaml
    results = run_rl_training()
