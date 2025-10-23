import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Tuple, Any, Optional, List
import logging

class SurgicalWorldModelEnv(gym.Env):
    """
    Improved gym environment that works with the DualWorldModel.
    Supports both dense and sparse reward configurations.
    """
    
    def __init__(self, 
                 world_model, 
                 config: Dict[str, Any], 
                 device='cuda',
                 reward_mode: str = 'dense'):
        """
        Initialize the environment.
        
        Args:
            world_model: DualWorldModel instance
            config: Configuration dictionary
            device: Device to run on
            reward_mode: 'dense' for continuous rewards, 'sparse' for episode-end rewards
        """
        super().__init__()
        
        self.model = world_model.eval()
        self.config = config
        self.device = device
        self.reward_mode = reward_mode
        
        # Environment parameters
        self.horizon = config.get('rl_horizon', 50)
        self.context_length = config.get('context_length', 10)
        
        # Action and observation spaces
        self.action_space = spaces.MultiBinary(world_model.num_action_classes)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(world_model.embedding_dim,),
            dtype=np.float32
        )
        
        # State tracking
        self.current_state_sequence = None
        self.current_phase = None
        self.step_count = 0
        self.episode_rewards = []
        self.episode_info = []
        
        # Video contexts for realistic initialization
        self.video_contexts = {}
        self.current_video_id = None
        
        # Reward configuration
        self.reward_weights = config.get('reward_weights', {
            'phase_completion': 1.0,
            'phase_initiation': 0.5,
            'phase_progression': 1.0,
            'global_progression': 0.8,
            'action_probability': 0.3,
            'risk_penalty': -0.01,
        })
        
        # Performance tracking
        self.phase_transitions = 0
        self.successful_actions = 0
        self.total_risk_accumulated = 0.0
        
        # Initialize reward normalization (running statistics)
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.reward_count = 0
        
    def set_video_context(self, video_data: List[Dict]):
        """Set context from video data for realistic initialization."""
        self.video_contexts = {video['video_id']: video for video in video_data}
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """
        Reset the environment.
        
        Args:
            seed: Random seed
            options: Reset options (can include 'video_id' for specific video context)
            
        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)
        
        # Reset tracking variables
        self.step_count = 0
        self.episode_rewards = []
        self.episode_info = []
        self.phase_transitions = 0
        self.successful_actions = 0
        self.total_risk_accumulated = 0.0
        
        # Initialize from video context if available
        if options and 'video_id' in options and options['video_id'] in self.video_contexts:
            self._initialize_from_video(options['video_id'])
        else:
            self._initialize_randomly()
        
        # Get initial observation
        observation = self._get_observation()
        
        # Initial info
        info = {
            'step': self.step_count,
            'current_phase': self.current_phase,
            'video_id': self.current_video_id,
            'episode_reward': 0.0
        }
        
        return observation, info
    
    def _initialize_from_video(self, video_id: str):
        """Initialize environment state from a specific video."""
        self.current_video_id = video_id
        video_data = self.video_contexts[video_id]
        embeddings = video_data['frame_embeddings']
        
        # Sample random starting point with enough context
        max_start = len(embeddings) - self.context_length - 1
        start_idx = np.random.randint(0, max(1, max_start))
        
        # Get context sequence
        context_embeddings = embeddings[start_idx:start_idx + self.context_length]
        self.current_state_sequence = torch.tensor(
            context_embeddings, dtype=torch.float32, device=self.device
        ).unsqueeze(0)  # [1, context_length, embedding_dim]
        
        # Set initial phase if available
        if 'phase_binaries' in video_data:
            phase_binary = video_data['phase_binaries'][start_idx + self.context_length - 1]
            self.current_phase = np.argmax(phase_binary)
        else:
            self.current_phase = 0
    
    def _initialize_randomly(self):
        """Initialize environment state randomly."""
        self.current_video_id = 'random'
        self.current_state_sequence = torch.randn(
            1, self.context_length, self.model.embedding_dim, device=self.device
        )
        self.current_phase = 0
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation (last frame in state sequence)."""
        return self.current_state_sequence[0, -1].cpu().numpy()
    
    def _prepare_action_tensor(self, action: np.ndarray) -> torch.Tensor:
        """
        Prepare action tensor with correct dimensions.
        
        Args:
            action: Action array from RL algorithm
            
        Returns:
            Properly shaped action tensor
        """
        # Ensure action is numpy array
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        
        action = np.array(action).flatten()
        
        # Ensure correct size
        if len(action) != self.model.num_action_classes:
            if len(action) < self.model.num_action_classes:
                action = np.pad(action, (0, self.model.num_action_classes - len(action)))
            else:
                action = action[:self.model.num_action_classes]
        
        # Convert to tensor with proper shape
        action_tensor = torch.tensor(action, dtype=torch.float32, device=self.device)
        action_tensor = action_tensor.view(1, 1, self.model.num_action_classes)  # [1, 1, num_actions]
        
        return action_tensor
    
    def _compute_reward(self, 
                       state_pred: torch.Tensor,
                       reward_predictions: Dict[str, torch.Tensor],
                       action: torch.Tensor,
                       info: Dict[str, Any]) -> float:
        """
        Compute reward based on model predictions and environment state.
        
        Args:
            state_pred: Predicted next state
            reward_predictions: Reward predictions from model
            action: Taken action
            info: Additional info for reward computation
            
        Returns:
            Computed reward
        """
        total_reward = 0.0
        reward_breakdown = {}
        
        # Dense rewards from model predictions
        if self.reward_mode == 'dense':
            for reward_type, weight in self.reward_weights.items():
                if reward_type in reward_predictions:
                    # Get reward prediction (negate if it's a loss)
                    pred_reward = reward_predictions[reward_type].squeeze().item()
                    
                    # Apply weight
                    weighted_reward = pred_reward * weight
                    total_reward += weighted_reward
                    reward_breakdown[reward_type] = pred_reward
        
        # Additional reward components
        
        # 1. Action diversity reward (encourage exploration)
        action_diversity = torch.sum(action).item() / action.size(-1)
        diversity_reward = 0.1 * (action_diversity - 0.5)  # Reward balanced action selection
        total_reward += diversity_reward
        reward_breakdown['action_diversity'] = diversity_reward
        
        # 2. State consistency reward (penalize unrealistic state transitions)
        if hasattr(self, '_previous_state'):
            state_change = torch.norm(state_pred - self._previous_state).item()
            # Penalize too large or too small changes
            consistency_reward = -0.1 * abs(state_change - 1.0)  # Expect moderate changes
            total_reward += consistency_reward
            reward_breakdown['state_consistency'] = consistency_reward
        
        self._previous_state = state_pred.clone()
        
        # 3. Phase progression reward
        if 'current_phase' in info and hasattr(self, '_previous_phase'):
            if info['current_phase'] > self._previous_phase:
                phase_progress_reward = 2.0  # Reward phase transitions
                total_reward += phase_progress_reward
                reward_breakdown['phase_progress'] = phase_progress_reward
                self.phase_transitions += 1
        
        self._previous_phase = info.get('current_phase', 0)
        
        # 4. Sparse rewards (if configured)
        if self.reward_mode == 'sparse':
            # Only give rewards at episode end or major milestones
            if self.step_count >= self.horizon - 1:
                # End-of-episode reward based on performance
                episode_performance = (
                    self.phase_transitions * 5.0 +  # Reward phase transitions
                    self.successful_actions * 0.1 -  # Reward good actions
                    self.total_risk_accumulated * 0.5  # Penalize risk
                )
                total_reward = episode_performance
                reward_breakdown['episode_performance'] = episode_performance
        
        # Update running reward statistics for normalization
        self._update_reward_stats(total_reward)
        
        # Normalize reward (optional)
        if self.config.get('normalize_rewards', False):
            total_reward = (total_reward - self.reward_mean) / (self.reward_std + 1e-8)
        
        return total_reward, reward_breakdown
    
    def _update_reward_stats(self, reward: float):
        """Update running statistics for reward normalization."""
        self.reward_count += 1
        delta = reward - self.reward_mean
        self.reward_mean += delta / self.reward_count
        self.reward_std = np.sqrt(
            ((self.reward_count - 1) * self.reward_std ** 2 + delta * (reward - self.reward_mean)) / self.reward_count
        )
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: Action to take
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Prepare action tensor
        action_tensor = self._prepare_action_tensor(action)
        
        # Predict next state and rewards using the world model
        with torch.no_grad():
            # Use RL prediction mode
            outputs = self.model.rl_state_prediction(
                current_states=self.current_state_sequence,
                planned_actions=action_tensor.expand(1, self.context_length, -1),
                return_rewards=True
            )
        
        # Extract predictions
        next_state = outputs['next_states'][:, -1:, :]  # [1, 1, embedding_dim]
        reward_predictions = outputs.get('rewards', {})
        
        # Update phase if predicted
        if 'phases' in outputs:
            phase_probs = outputs['phases'][:, -1]  # [1, num_phases]
            self.current_phase = torch.argmax(phase_probs, dim=-1).item()
        
        # Compute reward
        info = {
            'step': self.step_count,
            'current_phase': self.current_phase,
            'video_id': self.current_video_id
        }
        
        reward, reward_breakdown = self._compute_reward(
            next_state, reward_predictions, action_tensor, info
        )
        
        # Update state sequence (sliding window)
        self.current_state_sequence = torch.cat([
            self.current_state_sequence[:, 1:, :],  # Remove first frame
            next_state  # Add new frame
        ], dim=1)
        
        # Update tracking
        self.step_count += 1
        self.episode_rewards.append(reward)
        
        # Accumulate risk (if available)
        if 'risk_penalty' in reward_predictions:
            self.total_risk_accumulated += abs(reward_predictions['risk_penalty'].item())
        
        # Check if action was "successful" (simple heuristic)
        if reward > 0:
            self.successful_actions += 1
        
        # Episode termination
        terminated = False
        truncated = self.step_count >= self.horizon
        
        # Early termination conditions
        if self.config.get('early_termination', False):
            # Terminate if cumulative reward is very negative
            if sum(self.episode_rewards) < -10.0:
                terminated = True
        
        # Enhanced info
        info.update({
            'reward_breakdown': reward_breakdown,
            'episode_reward': sum(self.episode_rewards),
            'phase_transitions': self.phase_transitions,
            'successful_actions': self.successful_actions,
            'total_risk': self.total_risk_accumulated,
            'success_rate': self.successful_actions / max(self.step_count, 1)
        })
        
        # Get next observation
        observation = self._get_observation()
        
        return observation, reward, terminated, truncated, info
    
    def render(self, mode='human'):
        """Render the environment (placeholder implementation)."""
        if mode == 'human':
            print(f"Step: {self.step_count}, Phase: {self.current_phase}, "
                  f"Episode Reward: {sum(self.episode_rewards):.3f}")
        elif mode == 'rgb_array':
            # Could implement visualization here
            return np.zeros((64, 64, 3), dtype=np.uint8)
    
    def close(self):
        """Clean up resources."""
        pass
    
    def get_episode_metrics(self) -> Dict[str, Any]:
        """Get comprehensive episode metrics."""
        return {
            'episode_length': self.step_count,
            'total_reward': sum(self.episode_rewards),
            'average_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0.0,
            'phase_transitions': self.phase_transitions,
            'successful_actions': self.successful_actions,
            'success_rate': self.successful_actions / max(self.step_count, 1),
            'total_risk': self.total_risk_accumulated,
            'final_phase': self.current_phase,
            'video_id': self.current_video_id
        }


class MultiVideoSurgicalEnv(gym.Env):
    """
    Environment that cycles through multiple videos for more diverse training.
    """
    
    def __init__(self, 
                 world_model,
                 config: Dict[str, Any],
                 video_data: List[Dict],
                 device='cuda'):
        """
        Initialize multi-video environment.
        
        Args:
            world_model: DualWorldModel instance
            config: Configuration dictionary
            video_data: List of video data dictionaries
            device: Device to run on
        """
        super().__init__()
        
        # Create base environment
        self.base_env = SurgicalWorldModelEnv(world_model, config, device)
        self.base_env.set_video_context(video_data)
        
        # Video management
        self.video_ids = [video['video_id'] for video in video_data]
        self.current_video_idx = 0
        
        # Delegate space definitions
        self.action_space = self.base_env.action_space
        self.observation_space = self.base_env.observation_space
        
        # Episode tracking
        self.episode_count = 0
        self.video_episode_counts = {vid: 0 for vid in self.video_ids}
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Reset with video cycling."""
        # Cycle through videos for diversity
        if options is None:
            options = {}
        
        # Select video (round-robin or random)
        if self.episode_count % len(self.video_ids) == 0:
            # Shuffle video order every full cycle
            np.random.shuffle(self.video_ids)
        
        video_id = self.video_ids[self.episode_count % len(self.video_ids)]
        options['video_id'] = video_id
        
        # Update counters
        self.episode_count += 1
        self.video_episode_counts[video_id] += 1
        
        return self.base_env.reset(seed=seed, options=options)
    
    def step(self, action):
        """Forward step to base environment."""
        return self.base_env.step(action)
    
    def render(self, mode='human'):
        """Forward render to base environment."""
        return self.base_env.render(mode)
    
    def close(self):
        """Forward close to base environment."""
        return self.base_env.close()
    
    def get_video_statistics(self) -> Dict[str, Any]:
        """Get statistics about video usage."""
        return {
            'total_episodes': self.episode_count,
            'video_episode_counts': self.video_episode_counts.copy(),
            'current_video': self.video_ids[self.current_video_idx] if self.video_ids else None
        }