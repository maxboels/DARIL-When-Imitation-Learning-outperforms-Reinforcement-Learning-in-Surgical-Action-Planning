#!/usr/bin/env python3
"""
Integration: RL Environments for Your Existing Codebase
Place this in: environment/rl_environments.py
"""

import torch
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Any, Optional, Tuple
import random

class WorldModelSimulationEnv(gym.Env):
    """
    World Model Environment with expert demonstration matching rewards.
    
    Key features:
    1. Continuous action space [0,1]^100 (matches your models)
    2. Expert demonstration matching rewards (primary signal)
    3. Minimal use of risk penalties (as you requested)
    4. Proper episode termination
    """
    
    def __init__(self, world_model, video_data: List[Dict], config: Dict, device: str = 'cuda'):
        super().__init__()
        
        self.world_model = world_model
        self.video_data = video_data
        self.config = config
        self.device = device
        
        # Environment parameters
        self.max_episode_steps = config.get('rl_horizon', 25)  # Shorter episodes for faster training
        self.context_length = config.get('context_length', 10)
        
        # Current episode state
        self.current_step = 0
        self.current_video_idx = 0
        self.current_frame_idx = 0
        self.current_state = None
        self.episode_reward = 0.0
        
        # Proper action space - continuous [0,1] for each action class
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(100,), dtype=np.float32
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(1024,), dtype=np.float32
        )
        
        # Expert demonstration focused rewards (avoiding risky rewards as requested)
        self.reward_weights = {
            'expert_matching': 10.0,        # Primary: match expert actions
            'action_consistency': 2.0,      # Secondary: reasonable action patterns
            'completion_bonus': 5.0,        # Episode completion
            # Note: Removed risk penalties as requested
        }
        
        # Track expert actions for the current video
        self.current_video = None
        self.expert_actions_sequence = None
        
        # Episode statistics for monitoring
        self.episode_rewards = []
        self.episode_lengths = []
        self.expert_matching_scores = []
        
        print("🌍 World Model Environment initialized")
        print(f"📊 Action space: Continuous [0,1]^100 (matches your RL models)")
        print(f"🎯 Focus: Expert demonstration matching (no risky rewards)")

    def reset(self, seed=None, options=None):
        """Reset with proper expert action tracking."""
        super().reset(seed=seed)
        
        # Select random video and cache expert actions
        self.current_video_idx = np.random.randint(0, len(self.video_data))
        self.current_video = self.video_data[self.current_video_idx]
        
        # Cache expert actions using your dataset's key name
        if 'actions_binaries' in self.current_video:
            self.expert_actions_sequence = np.array(self.current_video['actions_binaries'])
        else:
            self.expert_actions_sequence = None
            print(f"⚠️ No expert actions found for video {self.current_video['video_id']}")
        
        # Start from a safe position
        min_start = self.context_length
        max_start = len(self.current_video['frame_embeddings']) - self.max_episode_steps - 5
        max_start = max(min_start, max_start)
        
        if max_start <= min_start:
            self.current_frame_idx = min_start
        else:
            self.current_frame_idx = np.random.randint(min_start, max_start + 1)
        
        # Get initial state
        self.current_state = self.current_video['frame_embeddings'][self.current_frame_idx].astype(np.float32)
        self.current_state = self._fix_state_shape(self.current_state)
        
        # Reset episode variables
        self.current_step = 0
        self.episode_reward = 0.0
        self.episode_expert_scores = []
        
        return self.current_state.copy(), {}

    def step(self, action):
        """Step with expert demonstration matching rewards."""
        self.current_step += 1
        
        # Process action
        action = self._process_action(action)
        
        # Check termination
        frames_remaining = len(self.current_video['frame_embeddings']) - self.current_frame_idx - 1
        done = (self.current_step >= self.max_episode_steps) or (frames_remaining <= 0)
        
        if done:
            # Episode completion bonus
            reward = self.reward_weights['completion_bonus']
            next_state = self.current_state.copy()
        else:
            # Use world model to predict next state
            next_state, predicted_rewards = self._predict_with_world_model(action)
            
            # Calculate expert-focused reward
            reward = self._calculate_expert_focused_reward(action, predicted_rewards)
            
            # Update current state
            self.current_state = next_state.copy()
        
        self.episode_reward += reward
        
        # Info for monitoring
        info = {
            'step': self.current_step,
            'episode_reward': self.episode_reward,
            'action_sum': float(np.sum(action > 0.5)),
            'expert_available': self.expert_actions_sequence is not None,
            'frames_remaining': frames_remaining,
            'method': 'fixed_world_model_rl'
        }
        
        # Log episode completion
        if done:
            self.episode_lengths.append(self.current_step)
            self.episode_rewards.append(self.episode_reward)
            
            # Calculate expert matching score for this episode
            if hasattr(self, 'episode_expert_scores') and self.episode_expert_scores:
                avg_expert_score = np.mean(self.episode_expert_scores)
                self.expert_matching_scores.append(avg_expert_score)
                info['episode_expert_matching'] = avg_expert_score
            
            # Reset episode tracking
            self.episode_expert_scores = []
            
            # Print progress every 20 episodes
            if len(self.episode_rewards) % 20 == 0:
                recent_rewards = self.episode_rewards[-10:]
                recent_expert_scores = self.expert_matching_scores[-10:] if self.expert_matching_scores else [0]
                print(f"📊 Episode {len(self.episode_rewards)}: "
                      f"Avg Reward: {np.mean(recent_rewards):.3f}, "
                      f"Expert Match: {np.mean(recent_expert_scores):.3f}")
        
        return self.current_state.copy(), reward, done, False, info

    def _calculate_expert_focused_reward(self, action: np.ndarray, predicted_rewards: Dict[str, float]) -> float:
        """Calculate reward focused on expert demonstration matching (avoiding risky rewards)."""
        reward = 0.0
        
        # 1. PRIMARY: Expert demonstration matching
        if (self.expert_actions_sequence is not None and 
            self.current_frame_idx < len(self.expert_actions_sequence)):
            
            expert_actions = self.expert_actions_sequence[self.current_frame_idx]
            binary_action = (action > 0.5).astype(int)
            
            if len(expert_actions) == len(binary_action):
                # Basic accuracy
                accuracy = np.mean(binary_action == expert_actions)
                reward += self.reward_weights['expert_matching'] * accuracy
                
                # F1 score bonus for sparse surgical actions
                if np.sum(expert_actions) > 0:
                    true_positives = np.sum((binary_action == 1) & (expert_actions == 1))
                    false_positives = np.sum((binary_action == 1) & (expert_actions == 0))
                    false_negatives = np.sum((binary_action == 0) & (expert_actions == 1))
                    
                    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
                    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
                    
                    if precision > 0 and recall > 0:
                        f1_score = 2 * (precision * recall) / (precision + recall)
                        reward += self.reward_weights['expert_matching'] * f1_score * 0.5  # Bonus weight
                
                # Track for episode summary
                if not hasattr(self, 'episode_expert_scores'):
                    self.episode_expert_scores = []
                self.episode_expert_scores.append(accuracy)
        
        # 2. Action consistency (reasonable surgical action patterns)
        action_count = np.sum(action > 0.5)
        if 1 <= action_count <= 5:  # Reasonable range
            reward += self.reward_weights['action_consistency']
        elif action_count == 0:
            reward -= 0.5  # Small penalty for inaction
        elif action_count > 8:
            reward -= 1.0  # Penalty for too many simultaneous actions
        
        # 3. Use limited world model rewards (but with very low weight)
        world_model_reward = 0.0
        for reward_type, reward_value in predicted_rewards.items():
            if reward_type in ['phase_progression', 'efficiency']:  # Only use safe rewards
                world_model_reward += reward_value
        
        reward += 0.1 * world_model_reward  # Very low weight
        
        # 4. Small exploration bonus
        action_entropy = -np.sum(action * np.log(action + 1e-8) + (1-action) * np.log(1-action + 1e-8))
        reward += 0.01 * action_entropy / 100
        
        return np.clip(reward, -2.0, 20.0)  # Reasonable range

    def _predict_with_world_model(self, action: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        """Use world model for prediction."""
        try:
            current_state_tensor = torch.tensor(self.current_state, dtype=torch.float32, device=self.device)
            action_tensor = torch.tensor(action, dtype=torch.float32, device=self.device)
            
            next_state, rewards, _ = self.world_model.simulate_step(
                current_state_tensor, action_tensor, return_hidden=False
            )
            
            next_state_np = next_state.cpu().numpy().flatten()
            next_state_np = self._fix_state_shape(next_state_np)
            
            return next_state_np, rewards
                
        except Exception as e:
            print(f"⚠️ World model prediction failed: {e}")
            # Fallback: small perturbation
            noise = np.random.normal(0, 0.01, self.current_state.shape)
            return self.current_state + noise, {}

    def _process_action(self, action) -> np.ndarray:
        """Process action to ensure correct format."""
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        
        action = np.array(action, dtype=np.float32).flatten()
        
        if len(action) != 100:
            if len(action) < 100:
                padded_action = np.zeros(100, dtype=np.float32)
                padded_action[:len(action)] = action
                action = padded_action
            else:
                action = action[:100]
        
        return np.clip(action, 0.0, 1.0)

    def _fix_state_shape(self, state: np.ndarray) -> np.ndarray:
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

    def get_episode_stats(self):
        """Get episode statistics for monitoring."""
        if not self.episode_lengths:
            return {"avg_length": 0, "avg_reward": 0, "episodes": 0}
        
        stats = {
            "avg_length": np.mean(self.episode_lengths),
            "avg_reward": np.mean(self.episode_rewards),
            "std_reward": np.std(self.episode_rewards),
            "episodes": len(self.episode_lengths),
            "last_reward": self.episode_rewards[-1],
            "uses_world_model": True,
            "expert_focused": True
        }
        
        if self.expert_matching_scores:
            stats.update({
                "avg_expert_matching": np.mean(self.expert_matching_scores),
                "std_expert_matching": np.std(self.expert_matching_scores),
                "last_expert_matching": self.expert_matching_scores[-1]
            })
        
        return stats


class DirectVideoEnvironment(gym.Env):
    """
    Direct Video Environment with expert demonstration matching.
    """
    
    def __init__(self, video_data: List[Dict], config: Dict, device: str = 'cuda'):
        super().__init__()
        
        self.video_data = video_data
        self.config = config
        self.device = device
        
        # Environment parameters
        self.max_episode_steps = config.get('rl_horizon', 25)
        
        # Current episode state
        self.current_video_idx = 0
        self.current_frame_idx = 0
        self.current_step = 0
        self.episode_reward = 0.0
        
        # Proper action space
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(100,), dtype=np.float32
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(1024,), dtype=np.float32
        )
        
        # Expert-focused reward weights
        self.reward_weights = {
            'expert_matching': 10.0,
            'action_consistency': 2.0,
            'completion_bonus': 5.0
        }
        
        # Episode statistics
        self.episode_lengths = []
        self.episode_rewards = []
        self.expert_matching_scores = []
        
        print(f"🎬 Direct Video Environment initialized")
        print(f"🎯 Focus: Expert demonstration matching on real video frames")
    
    def reset(self, seed=None, options=None):
        """Reset with proper initialization."""
        super().reset(seed=seed)
        
        # Select random video
        self.current_video_idx = random.randint(0, len(self.video_data) - 1)
        video = self.video_data[self.current_video_idx]
        
        # Select random starting frame
        available_frames = len(video['frame_embeddings'])
        max_start_frame = max(0, available_frames - self.max_episode_steps - 1)
        self.current_frame_idx = random.randint(0, max_start_frame)
        
        # Reset episode variables
        self.current_step = 0
        self.episode_reward = 0.0
        self.episode_expert_scores = []
        
        # Get initial observation
        initial_state = video['frame_embeddings'][self.current_frame_idx].astype(np.float32)
        initial_state = self._fix_state_shape(initial_state)
        
        return initial_state, {}
    
    def step(self, action):
        """Step with expert demonstration matching."""
        self.current_step += 1
        video = self.video_data[self.current_video_idx]
        
        # Process action
        action = self._process_action(action)
        
        # Check termination
        frames_remaining = len(video['frame_embeddings']) - self.current_frame_idx - 1
        done = (self.current_step >= self.max_episode_steps) or (frames_remaining <= 0)
        
        if done:
            reward = self.reward_weights['completion_bonus']
            next_state = video['frame_embeddings'][self.current_frame_idx].astype(np.float32)
        else:
            # Move to next frame
            self.current_frame_idx += 1
            next_state = video['frame_embeddings'][self.current_frame_idx].astype(np.float32)
            
            # Calculate expert-focused reward
            reward = self._calculate_expert_focused_reward(action, video)
        
        next_state = self._fix_state_shape(next_state)
        self.episode_reward += reward
        
        info = {
            'step': self.current_step,
            'episode_reward': self.episode_reward,
            'action_sum': float(np.sum(action > 0.5)),
            'frames_remaining': frames_remaining,
            'method': 'fixed_direct_video_rl'
        }
        
        # Log episode completion
        if done:
            self.episode_lengths.append(self.current_step)
            self.episode_rewards.append(self.episode_reward)
            
            if hasattr(self, 'episode_expert_scores') and self.episode_expert_scores:
                avg_expert_score = np.mean(self.episode_expert_scores)
                self.expert_matching_scores.append(avg_expert_score)
                info['episode_expert_matching'] = avg_expert_score
        
        return next_state, reward, done, False, info
    
    def _calculate_expert_focused_reward(self, action: np.ndarray, video: Dict) -> float:
        """Calculate reward based on expert demonstration matching."""
        reward = 0.0
        binary_action = (action > 0.5).astype(int)
        
        # Expert action matching using your dataset's key name
        if ('actions_binaries' in video and 
            self.current_frame_idx < len(video['actions_binaries'])):
            
            expert_actions = video['actions_binaries'][self.current_frame_idx]
            
            if len(expert_actions) == len(binary_action):
                accuracy = np.mean(binary_action == expert_actions)
                reward += self.reward_weights['expert_matching'] * accuracy
                
                # F1 bonus for sparse actions
                if np.sum(expert_actions) > 0:
                    tp = np.sum((binary_action == 1) & (expert_actions == 1))
                    fp = np.sum((binary_action == 1) & (expert_actions == 0))
                    fn = np.sum((binary_action == 0) & (expert_actions == 1))
                    
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    
                    if precision > 0 and recall > 0:
                        f1 = 2 * (precision * recall) / (precision + recall)
                        reward += self.reward_weights['expert_matching'] * f1 * 0.5
                
                # Track for episode summary
                if not hasattr(self, 'episode_expert_scores'):
                    self.episode_expert_scores = []
                self.episode_expert_scores.append(accuracy)
        
        # Action consistency
        action_count = np.sum(binary_action)
        if 1 <= action_count <= 5:
            reward += self.reward_weights['action_consistency']
        elif action_count == 0:
            reward -= 0.5
        elif action_count > 8:
            reward -= 1.0
        
        return np.clip(reward, -2.0, 15.0)
    
    def _process_action(self, action) -> np.ndarray:
        """Process action to ensure correct format."""
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        
        action = np.array(action, dtype=np.float32).flatten()
        
        if len(action) != 100:
            if len(action) < 100:
                padded_action = np.zeros(100, dtype=np.float32)
                padded_action[:len(action)] = action
                action = padded_action
            else:
                action = action[:100]
        
        return np.clip(action, 0.0, 1.0)
    
    def _fix_state_shape(self, state: np.ndarray) -> np.ndarray:
        """Ensure state has correct shape."""
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

    def get_episode_stats(self):
        """Get episode statistics."""
        if not self.episode_lengths:
            return {"avg_length": 0, "avg_reward": 0, "episodes": 0}
        
        stats = {
            "avg_length": np.mean(self.episode_lengths),
            "avg_reward": np.mean(self.episode_rewards),
            "std_reward": np.std(self.episode_rewards),
            "episodes": len(self.episode_lengths),
            "uses_real_frames": True,
            "expert_focused": True
        }
        
        if self.expert_matching_scores:
            stats.update({
                "avg_expert_matching": np.mean(self.expert_matching_scores),
                "last_expert_matching": self.expert_matching_scores[-1]
            })
        
        return stats
