#!/usr/bin/env python3
"""
RL Environment with proper action space and reward design
Addresses the major issues causing poor RL performance
"""

import torch
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Any, Optional, Tuple
import random


class WorldModelSimulationEnv(gym.Env):
    """
    World Model RL Environment with proper reward design and action space.
    
    Key features:
    1. Proper action space (continuous [0,1] per action class)
    2. Meaningful reward function based on expert demonstration matching
    3. Proper episode termination
    4. Better state handling
    """
    
    def __init__(self, world_model, video_data: List[Dict], config: Dict, device: str = 'cuda'):
        super().__init__()
        
        self.world_model = world_model
        self.video_data = video_data
        self.config = config
        self.device = device
        
        # Environment parameters
        self.max_episode_steps = config.get('rl_horizon', 25)  # Shorter episodes
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
        
        # Better reward design based on expert demonstration matching
        self.reward_weights = {
            'expert_matching': 10.0,     # Match expert actions (most important)
            'action_sparsity': 1.0,      # Appropriate number of actions
            'world_model_rewards': 0.5,  # Use world model predictions (lower weight)
            'completion_bonus': 5.0,     # Episode completion
            'consistency_bonus': 1.0     # Consistent action patterns
        }
        
        # Track expert actions for the current video
        self.current_video = None
        self.expert_actions_sequence = None
        
        # Episode statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.expert_matching_scores = []
        
        print("🌍 World Model Simulation Environment initialized")
        print(f"📊 Action space: {self.action_space} (continuous [0,1])")
        print(f"📊 Max episode steps: {self.max_episode_steps}")
        print(f"🎯 Focus: Expert demonstration matching + world model simulation")

    def reset(self, seed=None, options=None):
        """Reset with proper expert action tracking."""
        super().reset(seed=seed)
        
        # Select random video and cache expert actions
        self.current_video_idx = np.random.randint(0, len(self.video_data))
        self.current_video = self.video_data[self.current_video_idx]
        
        # Cache expert actions for this video
        if 'actions_binaries' in self.current_video:
            self.expert_actions_sequence = np.array(self.current_video['actions_binaries'])
        else:
            self.expert_actions_sequence = None
        
        # Start from a position where we have enough context and future frames
        min_start = self.context_length
        max_start = len(self.current_video['frame_embeddings']) - self.max_episode_steps - 5
        max_start = max(min_start, max_start)
        
        if max_start <= min_start:
            # Video too short, use what we have
            self.current_frame_idx = min_start
        else:
            self.current_frame_idx = np.random.randint(min_start, max_start + 1)
        
        # Get initial state
        self.current_state = self.current_video['frame_embeddings'][self.current_frame_idx].astype(np.float32)
        self.current_state = self._fix_state_shape(self.current_state)
        
        # Reset episode variables
        self.current_step = 0
        self.episode_reward = 0.0
        
        return self.current_state.copy(), {}

    def step(self, action):
        """Step with proper reward calculation and expert matching."""
        self.current_step += 1
        
        # Process action to ensure correct format
        action = self._process_action(action)
        
        # Check termination conditions
        frames_remaining = len(self.current_video['frame_embeddings']) - self.current_frame_idx - 1
        done = (self.current_step >= self.max_episode_steps) or (frames_remaining <= 0)
        
        if done:
            # Episode completion
            reward = self.reward_weights['completion_bonus']
            next_state = self.current_state.copy()
        else:
            # Use world model to predict next state
            next_state, predicted_rewards = self._predict_with_world_model(action)
            
            # FIXED: Calculate meaningful reward
            reward = self._calculate_meaningful_reward(action, predicted_rewards)
            
            # Update current state
            self.current_state = next_state.copy()
        
        self.episode_reward += reward
        
        # Enhanced info for debugging
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
            if hasattr(self, 'episode_expert_scores'):
                avg_expert_score = np.mean(self.episode_expert_scores) if self.episode_expert_scores else 0.0
                self.expert_matching_scores.append(avg_expert_score)
                info['episode_expert_matching'] = avg_expert_score
            
            # Reset episode-specific tracking
            self.episode_expert_scores = []
            
            # Print progress every 20 episodes
            if len(self.episode_rewards) % 20 == 0:
                recent_rewards = self.episode_rewards[-10:]
                recent_expert_scores = self.expert_matching_scores[-10:] if self.expert_matching_scores else [0]
                print(f"📊 Episode {len(self.episode_rewards)}: "
                      f"Avg Reward: {np.mean(recent_rewards):.3f}, "
                      f"Expert Match: {np.mean(recent_expert_scores):.3f}")
        
        return self.current_state.copy(), reward, done, False, info


    def _calculate_meaningful_reward(self, action: np.ndarray, predicted_rewards: Dict[str, float]) -> float:
        """FIXED: Calculate reward that aligns with mAP evaluation"""
        reward = 0.0
        
        # 1. MOST CRITICAL: Heavily weight POSITIVE action prediction
        if (self.expert_actions_sequence is not None and 
            self.current_frame_idx < len(self.expert_actions_sequence)):
            
            expert_actions = self.expert_actions_sequence[self.current_frame_idx]
            binary_action = (action > 0.5).astype(int)
            
            if len(expert_actions) == len(binary_action):
                # Separate positive and negative action evaluation
                positive_mask = expert_actions > 0.5
                negative_mask = expert_actions <= 0.5
                
                if np.sum(positive_mask) > 0:
                    # CRITICAL: Heavy reward for correctly predicting positive actions
                    positive_correct = np.sum(
                        (binary_action[positive_mask] == 1) & (expert_actions[positive_mask] == 1)
                    )
                    positive_total = np.sum(positive_mask)
                    positive_recall = positive_correct / positive_total
                    
                    # Check precision (avoid false positives)
                    predicted_positive = np.sum(binary_action > 0.5)
                    if predicted_positive > 0:
                        positive_precision = positive_correct / predicted_positive
                        # F1-like reward focusing on positive actions
                        if (positive_precision + positive_recall) > 0:
                            positive_f1 = 2 * (positive_precision * positive_recall) / (positive_precision + positive_recall)
                            reward += 30.0 * positive_f1  # MUCH higher weight than before
                    else:
                        # Predicted no actions when there should be positive actions = very bad
                        reward -= 15.0
                
                # LOWER weight for negative actions (zeros are easy to predict)
                if np.sum(negative_mask) > 0:
                    negative_correct = np.sum(
                        (binary_action[negative_mask] == 0) & (expert_actions[negative_mask] == 0)
                    )
                    negative_accuracy = negative_correct / np.sum(negative_mask)
                    reward += 1.0 * negative_accuracy  # Much lower weight
                
                # Track expert matching for episode summary
                if not hasattr(self, 'episode_expert_scores'):
                    self.episode_expert_scores = []
                overall_accuracy = np.mean(binary_action == expert_actions)
                self.episode_expert_scores.append(overall_accuracy)
        
        # 2. ENCOURAGE 1-3 ACTIONS (no more, no less)
        action_count = np.sum(action > 0.5)
        if action_count == 0:
            reward -= 4.0  # Penalty for no actions
        elif 1 <= action_count <= 3:
            reward += 5.0  # Strong bonus for ideal range
        elif action_count == 4:
            reward += 2.0  # Small bonus for close
        elif action_count >= 5:
            reward -= 3.0 * (action_count - 4)  # Escalating penalty for too many
        
        # 3. MUCH LOWER weight for world model rewards
        world_model_reward = 0.0
        for reward_type, reward_value in predicted_rewards.items():
            if reward_type != 'value':
                world_model_reward += reward_value
        reward += 0.1 * world_model_reward  # Very small weight
        
        # 4. Action consistency bonus (avoid random switching)
        if hasattr(self, 'previous_action'):
            # But only for reasonable action counts
            if 1 <= action_count <= 3 and 1 <= np.sum(self.previous_action > 0.5) <= 3:
                action_consistency = 1.0 - np.mean(np.abs(action - self.previous_action))
                reward += 1.0 * action_consistency
        
        self.previous_action = action.copy()
        
        return np.clip(reward, -25.0, 40.0)

    # def _calculate_meaningful_reward(self, action: np.ndarray, predicted_rewards: Dict[str, float]) -> float:
    #     """Calculate meaningful reward that drives learning toward expert behavior."""
    #     reward = 0.0
        
    #     # 1. MOST IMPORTANT: Expert demonstration matching
    #     if (self.expert_actions_sequence is not None and 
    #         self.current_frame_idx < len(self.expert_actions_sequence)):
            
    #         expert_actions = self.expert_actions_sequence[self.current_frame_idx]
    #         binary_action = (action > 0.5).astype(int)
            
    #         if len(expert_actions) == len(binary_action):
    #             # Calculate multiple matching metrics
    #             accuracy = np.mean(binary_action == expert_actions)
    #             reward += self.reward_weights['expert_matching'] * accuracy
                
    #             # Bonus for correctly predicting positive actions (surgical actions are sparse)
    #             if np.sum(expert_actions) > 0:
    #                 true_positives = np.sum((binary_action == 1) & (expert_actions == 1))
    #                 false_positives = np.sum((binary_action == 1) & (expert_actions == 0))
    #                 false_negatives = np.sum((binary_action == 0) & (expert_actions == 1))
                    
    #                 # Precision and recall
    #                 precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    #                 recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
                    
    #                 # F1 score bonus
    #                 if precision > 0 and recall > 0:
    #                     f1_score = 2 * (precision * recall) / (precision + recall)
    #                     reward += self.reward_weights['expert_matching'] * f1_score
                
    #             # Track expert matching for episode summary
    #             if not hasattr(self, 'episode_expert_scores'):
    #                 self.episode_expert_scores = []
    #             self.episode_expert_scores.append(accuracy)
        
    #     # 2. Action sparsity reward (surgical actions should be reasonably sparse)
    #     action_count = np.sum(action > 0.5)
    #     if 1 <= action_count <= 5:  # Reasonable range for surgical actions
    #         reward += self.reward_weights['action_sparsity']
    #     elif action_count == 0:
    #         reward -= 0.5  # Small penalty for no actions
    #     elif action_count > 8:
    #         reward -= 1.0  # Penalty for too many simultaneous actions
        
    #     # 3. World model predicted rewards (use but with lower weight)
    #     world_model_reward = 0.0
    #     for reward_type, reward_value in predicted_rewards.items():
    #         if reward_type != 'value':  # Exclude value function
    #             world_model_reward += reward_value
        
    #     reward += self.reward_weights['world_model_rewards'] * world_model_reward
        
    #     # 4. Action consistency bonus (avoid random switching)
    #     if hasattr(self, 'previous_action'):
    #         action_consistency = 1.0 - np.mean(np.abs(action - self.previous_action))
    #         reward += self.reward_weights['consistency_bonus'] * action_consistency
        
    #     self.previous_action = action.copy()
        
    #     # 5. Small exploration bonus to avoid local minima
    #     action_entropy = -np.sum(action * np.log(action + 1e-8) + (1-action) * np.log(1-action + 1e-8))
    #     reward += 0.01 * action_entropy / 100  # Very small exploration bonus
        
    #     # Clip reward to reasonable range
    #     return np.clip(reward, -5.0, 20.0)

    def _predict_with_world_model(self, action: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        """Use world model for prediction with proper error handling."""
        try:
            current_state_tensor = torch.tensor(
                self.current_state, dtype=torch.float32, device=self.device
            )
            action_tensor = torch.tensor(
                action, dtype=torch.float32, device=self.device
            )
            
            # Use world model simulation
            next_state, rewards, _ = self.world_model.simulate_step(
                current_state_tensor, action_tensor, return_hidden=False
            )
            
            # Convert back to numpy
            next_state_np = next_state.cpu().numpy().flatten()
            next_state_np = self._fix_state_shape(next_state_np)
            
            return next_state_np, rewards
                
        except Exception as e:
            print(f"⚠️ World model prediction failed: {e}")
            # Fallback: small random perturbation of current state
            noise = np.random.normal(0, 0.01, self.current_state.shape)
            return self.current_state + noise, {}

    def _process_action(self, action) -> np.ndarray:
        """Process action to ensure correct format."""
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        
        action = np.array(action, dtype=np.float32).flatten()
        
        # Ensure correct size
        if len(action) != 100:
            if len(action) < 100:
                padded_action = np.zeros(100, dtype=np.float32)
                padded_action[:len(action)] = action
                action = padded_action
            else:
                action = action[:100]
        
        # Ensure [0,1] range
        action = np.clip(action, 0.0, 1.0)
        return action

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
        """Get comprehensive episode statistics."""
        if not self.episode_lengths:
            return {"avg_length": 0, "avg_reward": 0, "episodes": 0}
        
        stats = {
            "avg_length": np.mean(self.episode_lengths),
            "avg_reward": np.mean(self.episode_rewards),
            "std_reward": np.std(self.episode_rewards),
            "episodes": len(self.episode_lengths),
            "last_reward": self.episode_rewards[-1],
            "reward_trend": "improving" if len(self.episode_rewards) >= 10 and 
                           np.mean(self.episode_rewards[-5:]) > np.mean(self.episode_rewards[-10:-5]) else "stable"
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
    Direct Video Environment with proper action space and rewards.
    """
    
    def __init__(self, video_data: List[Dict], config: Dict, device: str = 'cuda'):
        super().__init__()
        
        self.video_data = video_data
        self.config = config
        self.device = device
        
        # Environment parameters
        self.max_episode_steps = config.get('rl_horizon', 25)  # Shorter episodes
        
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
        
        # Better reward weights
        self.reward_weights = {
            'expert_matching': 10.0,    # Match expert actions
            'action_density': 1.0,      # Appropriate action density
            'phase_progress': 2.0,      # Progress through phases
            'completion_bonus': 5.0     # Episode completion
        }
        
        # Episode statistics
        self.episode_lengths = []
        self.episode_rewards = []
        self.expert_matching_scores = []
        
        print(f"🎬 DirectVideoEnvironment initialized")
        print(f"📊 Action space: {self.action_space} (continuous [0,1])")
        print(f"🎯 Focus: Expert demonstration matching on real video frames")
    
    def reset(self, seed=None, options=None):
        """Reset with proper initialization."""
        super().reset(seed=seed)
        
        # Select random video
        self.current_video_idx = random.randint(0, len(self.video_data) - 1)
        video = self.video_data[self.current_video_idx]
        
        # Select random starting frame with enough remaining frames
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
        
        info = {
            'video_id': video.get('video_id', 'unknown'),
            'frame_idx': self.current_frame_idx,
            'available_frames': available_frames
        }
        
        return initial_state, info
    
    def step(self, action):
        """Step with meaningful reward calculation."""
        self.current_step += 1
        video = self.video_data[self.current_video_idx]
        
        # Process action
        action = self._process_action(action)
        
        # Check termination
        frames_remaining = len(video['frame_embeddings']) - self.current_frame_idx - 1
        done = (self.current_step >= self.max_episode_steps) or (frames_remaining <= 0)
        
        if done:
            # Episode completion bonus
            reward = self.reward_weights['completion_bonus']
            next_state = video['frame_embeddings'][self.current_frame_idx].astype(np.float32)
        else:
            # Move to next frame
            self.current_frame_idx += 1
            next_state = video['frame_embeddings'][self.current_frame_idx].astype(np.float32)
            
            # Calculate meaningful reward
            reward = self._calculate_meaningful_reward(action, video)
        
        next_state = self._fix_state_shape(next_state)
        self.episode_reward += reward
        
        info = {
            'step': self.current_step,
            'episode_reward': self.episode_reward,
            'action_sum': float(np.sum(action > 0.5)),
            'frames_remaining': frames_remaining,
            'method': 'direct_video_rl'
        }
        
        # Log episode completion
        if done:
            self.episode_lengths.append(self.current_step)
            self.episode_rewards.append(self.episode_reward)
            
            if hasattr(self, 'episode_expert_scores') and self.episode_expert_scores:
                avg_expert_score = np.mean(self.episode_expert_scores)
                self.expert_matching_scores.append(avg_expert_score)
                info['episode_expert_matching'] = avg_expert_score
            
            # Print progress
            if len(self.episode_rewards) % 20 == 0:
                recent_rewards = self.episode_rewards[-10:]
                recent_expert_scores = self.expert_matching_scores[-10:] if self.expert_matching_scores else [0]
                print(f"📊 Episode {len(self.episode_rewards)}: "
                      f"Avg Reward: {np.mean(recent_rewards):.3f}, "
                      f"Expert Match: {np.mean(recent_expert_scores):.3f}")
        
        return next_state, reward, done, False, info

    def _calculate_meaningful_reward(self, action: np.ndarray, predicted_rewards: Dict[str, float]) -> float:
        """FIXED: Calculate reward that aligns with mAP evaluation"""
        reward = 0.0
        
        # 1. MOST CRITICAL: Heavily weight POSITIVE action prediction
        if (self.expert_actions_sequence is not None and 
            self.current_frame_idx < len(self.expert_actions_sequence)):
            
            expert_actions = self.expert_actions_sequence[self.current_frame_idx]
            binary_action = (action > 0.5).astype(int)
            
            if len(expert_actions) == len(binary_action):
                # Separate positive and negative action evaluation
                positive_mask = expert_actions > 0.5
                negative_mask = expert_actions <= 0.5
                
                if np.sum(positive_mask) > 0:
                    # CRITICAL: Heavy reward for correctly predicting positive actions
                    positive_correct = np.sum(
                        (binary_action[positive_mask] == 1) & (expert_actions[positive_mask] == 1)
                    )
                    positive_total = np.sum(positive_mask)
                    positive_recall = positive_correct / positive_total
                    
                    # Check precision (avoid false positives)
                    predicted_positive = np.sum(binary_action > 0.5)
                    if predicted_positive > 0:
                        positive_precision = positive_correct / predicted_positive
                        # F1-like reward focusing on positive actions
                        if (positive_precision + positive_recall) > 0:
                            positive_f1 = 2 * (positive_precision * positive_recall) / (positive_precision + positive_recall)
                            reward += 30.0 * positive_f1  # MUCH higher weight than before
                    else:
                        # Predicted no actions when there should be positive actions = very bad
                        reward -= 15.0
                
                # LOWER weight for negative actions (zeros are easy to predict)
                if np.sum(negative_mask) > 0:
                    negative_correct = np.sum(
                        (binary_action[negative_mask] == 0) & (expert_actions[negative_mask] == 0)
                    )
                    negative_accuracy = negative_correct / np.sum(negative_mask)
                    reward += 1.0 * negative_accuracy  # Much lower weight
                
                # Track expert matching for episode summary
                if not hasattr(self, 'episode_expert_scores'):
                    self.episode_expert_scores = []
                overall_accuracy = np.mean(binary_action == expert_actions)
                self.episode_expert_scores.append(overall_accuracy)
        
        # 2. ENCOURAGE 1-3 ACTIONS (no more, no less)
        action_count = np.sum(action > 0.5)
        if action_count == 0:
            reward -= 8.0  # Strong penalty for no actions
        elif 1 <= action_count <= 3:
            reward += 5.0  # Strong bonus for ideal range
        elif action_count == 4:
            reward += 2.0  # Small bonus for close
        elif action_count >= 5:
            reward -= 3.0 * (action_count - 4)  # Escalating penalty for too many
        
        # 3. MUCH LOWER weight for world model rewards
        world_model_reward = 0.0
        for reward_type, reward_value in predicted_rewards.items():
            if reward_type != 'value':
                world_model_reward += reward_value
        reward += 0.1 * world_model_reward  # Very small weight
        
        # 4. Action consistency bonus (avoid random switching)
        if hasattr(self, 'previous_action'):
            # But only for reasonable action counts
            if 1 <= action_count <= 3 and 1 <= np.sum(self.previous_action > 0.5) <= 3:
                action_consistency = 1.0 - np.mean(np.abs(action - self.previous_action))
                reward += 1.0 * action_consistency
        
        self.previous_action = action.copy()
        
        return np.clip(reward, -25.0, 40.0)

    # def _calculate_meaningful_reward(self, action: np.ndarray, video: Dict) -> float:
    #     """Calculate meaningful reward based on expert matching."""
    #     reward = 0.0
    #     binary_action = (action > 0.5).astype(int)
        
    #     # 1. Expert action matching (most important)
    #     if ('actions_binaries' in video and 
    #         self.current_frame_idx < len(video['actions_binaries'])):
            
    #         expert_actions = video['actions_binaries'][self.current_frame_idx]
            
    #         if len(expert_actions) == len(binary_action):
    #             # Accuracy reward
    #             accuracy = np.mean(binary_action == expert_actions)
    #             reward += self.reward_weights['expert_matching'] * accuracy
                
    #             # F1 score for sparse actions
    #             if np.sum(expert_actions) > 0:
    #                 tp = np.sum((binary_action == 1) & (expert_actions == 1))
    #                 fp = np.sum((binary_action == 1) & (expert_actions == 0))
    #                 fn = np.sum((binary_action == 0) & (expert_actions == 1))
                    
    #                 precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    #                 recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    
    #                 if precision > 0 and recall > 0:
    #                     f1 = 2 * (precision * recall) / (precision + recall)
    #                     reward += self.reward_weights['expert_matching'] * f1
                
    #             # Track for episode summary
    #             if not hasattr(self, 'episode_expert_scores'):
    #                 self.episode_expert_scores = []
    #             self.episode_expert_scores.append(accuracy)
        
    #     # 2. Action density reward
    #     action_count = np.sum(binary_action)
    #     if 1 <= action_count <= 5:
    #         reward += self.reward_weights['action_density']
    #     elif action_count == 0:
    #         reward -= 0.5
    #     elif action_count > 8:
    #         reward -= 1.0
        
    #     # 3. Phase progression (if available)
    #     if ('next_rewards' in video and 
    #         '_r_phase_progression' in video['next_rewards'] and
    #         self.current_frame_idx < len(video['next_rewards']['_r_phase_progression'])):
            
    #         phase_reward = video['next_rewards']['_r_phase_progression'][self.current_frame_idx]
    #         reward += self.reward_weights['phase_progress'] * phase_reward
        
    #     return np.clip(reward, -5.0, 15.0)
    
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
            "last_reward": self.episode_rewards[-1]
        }
        
        if self.expert_matching_scores:
            stats.update({
                "avg_expert_matching": np.mean(self.expert_matching_scores),
                "last_expert_matching": self.expert_matching_scores[-1]
            })
        
        return stats


class WorldModelSimulationEnv_v1(gym.Env):
    """
    RL Environment that uses ConditionalWorldModel for simulation.
    This is the correct implementation of Method 2.
    """
    
    def __init__(self, world_model, video_data: List[Dict], config: Dict, device: str = 'cuda'):
        super().__init__()
        
        print(f"🌍 Initializing World Model Simulation Environment with {len(video_data)} videos")
        
        self.world_model = world_model
        self.video_data = video_data
        self.config = config
        self.device = device
        
        # Validate that we have a world model
        if self.world_model is None:
            raise ValueError("❌ ConditionalWorldModel is required for Method 2!")
        
        # Environment parameters
        self.max_episode_steps = config.get('rl_horizon', 50)
        self.context_length = config.get('context_length', 20)
        
        # Current episode state
        self.current_step = 0
        self.current_video_idx = 0
        self.current_frame_idx = 0
        self.current_state = None
        self.episode_reward = 0.0
        
        # Action and observation spaces
        self.action_space = spaces.Box(
            low=0, high=1, shape=(100,), dtype=np.float32
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(1024,),
            dtype=np.float32
        )
        
        # Episode statistics
        self.episode_lengths = []
        self.episode_rewards = []
        self.world_model_predictions = 0
        self.real_frame_uses = 0
        
        # Reward weights for combining multiple reward types
        self.reward_weights = {
            'phase_progression': 2.0,
            'phase_completion': 3.0,
            'phase_initiation': 1.0,
            'safety': 2.5,
            'efficiency': 1.5,
            'action_probability': 1.0,
            'risk_penalty': -1.0
        }
        
        print("✅ World Model Simulation Environment initialized")
        print(f"📊 Max episode steps: {self.max_episode_steps}")
        print(f"📊 Uses ConditionalWorldModel for simulation")

    def reset(self, seed=None, options=None):
        """Reset with initial real state, then use world model predictions."""
        super().reset(seed=seed)
        
        # Select random video
        self.current_video_idx = np.random.randint(0, len(self.video_data))
        video = self.video_data[self.current_video_idx]
        
        # Start from a position where we have enough context
        min_start = self.context_length
        max_start = len(video['frame_embeddings']) - self.max_episode_steps - 10
        max_start = max(min_start, max_start)
        
        self.current_frame_idx = np.random.randint(min_start, max_start + 1)
        
        # Get initial state from real video (this is the only real frame we use)
        self.current_state = video['frame_embeddings'][self.current_frame_idx].astype(np.float32)
        
        # Reset counters
        self.current_step = 0
        self.episode_reward = 0.0
        self.world_model_predictions = 0
        self.real_frame_uses = 1  # Initial state is real
        
        # Ensure correct state shape
        if self.current_state.shape[0] != 1024:
            self.current_state = self._fix_state_shape(self.current_state)
        
        print(f"🔄 Reset: Video {self.current_video_idx}, Frame {self.current_frame_idx}")
        
        return self.current_state.copy(), {}

    def step(self, action):
        """Use world model to predict next state and rewards."""
        self.current_step += 1
        
        # Process action
        action = self._process_action(action)
        
        # Check termination
        done = self.current_step >= self.max_episode_steps
        
        if done:
            # Episode completed
            reward = 5.0  # Completion bonus
            next_state = self.current_state.copy()
        else:
            # 🌍 CORE: Use world model to predict next state and rewards
            next_state, predicted_rewards = self._predict_with_world_model(action)
            
            # Calculate reward from world model predictions
            reward = self._calculate_reward_from_predictions(predicted_rewards, action)
            
            self.world_model_predictions += 1
        
        # Update current state
        self.current_state = next_state.copy()
        self.episode_reward += reward
        
        # Info for debugging
        info = {
            'step': self.current_step,
            'episode_reward': self.episode_reward,
            'world_model_predictions': self.world_model_predictions,
            'real_frame_uses': self.real_frame_uses,
            'uses_world_model': True,
            'simulation_based': True,
            'method': 'world_model_simulation',
            'simulation_ratio': self.world_model_predictions / max(1, self.current_step)
        }
        
        # Log episode completion
        if done:
            self.episode_lengths.append(self.current_step)
            self.episode_rewards.append(self.episode_reward)
            
            # Only print every 10th episode
            if len(self.episode_lengths) % 100 == 0:
                print(f"📊 Episode {len(self.episode_lengths)} complete - Length: {self.current_step}, "
                    f"Reward: {self.episode_reward:.3f}, "
                    f"WM predictions: {self.world_model_predictions}")
            
            # Keep only last 100 episodes
            if len(self.episode_lengths) > 100:
                self.episode_lengths = self.episode_lengths[-100:]
                self.episode_rewards = self.episode_rewards[-100:]
        
        return self.current_state.copy(), reward, done, False, info

    def _predict_with_world_model(self, next_action: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        """Use ConditionalWorldModel to predict next state and rewards."""
        
        try:
            # Convert to tensors
            current_state_tensor = torch.tensor(
                self.current_state, dtype=torch.float32, device=self.device
            )
            next_action_tensor = torch.tensor(
                next_action, dtype=torch.float32, device=self.device
            )
            
            # Use world model for simulation with the action we want to take
            next_state, rewards, _ = self.world_model.simulate_step(
                current_state_tensor, next_action_tensor, return_hidden=False
            )
            
            # Convert back to numpy
            next_state_np = next_state.cpu().numpy().flatten()
            next_state_np = self._fix_state_shape(next_state_np)
            
            return next_state_np, rewards
                
        except Exception as e:
            print(f"❌ World model prediction failed: {e}")
            # Fallback to current state
            return self.current_state.copy(), {}

    def _calculate_reward_from_predictions(self, predicted_rewards: Dict[str, float], action: np.ndarray) -> float:
        """Calculate reward from world model predictions."""
        
        reward = 0.0
        
        # Combine predicted rewards using weights
        for reward_type, reward_value in predicted_rewards.items():
            if reward_type in self.reward_weights:
                weighted_reward = self.reward_weights[reward_type] * reward_value
                reward += weighted_reward
        
        # Additional reward for reasonable action selection
        action_count = np.sum(action > 0.5)
        if 1 <= action_count <= 5:
            reward += 0.5
        elif action_count == 0:
            reward -= 1.0  # Penalty for no actions
        elif action_count > 10:
            reward -= 0.5  # Penalty for too many actions
        
        # Exploration bonus (small random component)
        exploration_bonus = np.random.normal(0, 0.1)
        reward += exploration_bonus
        
        return np.clip(reward, -10.0, 15.0)

    def _process_action(self, action) -> np.ndarray:
        """Process action to ensure correct format."""
        
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        
        action = np.array(action, dtype=np.float32).flatten()
        
        # Ensure correct size
        if len(action) != 100:
            if len(action) < 100:
                padded_action = np.zeros(100, dtype=np.float32)
                padded_action[:len(action)] = action
                action = padded_action
            else:
                action = action[:100]
        
        return action

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
        """Get episode statistics including world model usage."""
        if not self.episode_lengths:
            return {"avg_length": 0, "avg_reward": 0, "episodes": 0}
        
        return {
            "avg_length": np.mean(self.episode_lengths),
            "avg_reward": np.mean(self.episode_rewards),
            "episodes": len(self.episode_lengths),
            "last_length": self.episode_lengths[-1],
            "last_reward": self.episode_rewards[-1],
            "uses_world_model": True,
            "simulation_based": True,
            "method": "world_model_simulation",
            "can_explore_beyond_demos": True
        }

if __name__ == "__main__":
    print("🔧 RL ENVIRONMENTS")
    print("=" * 50)
    print("✅ Proper continuous action space [0,1]")
    print("✅ Meaningful reward based on expert matching")
    print("✅ Better episode termination")
    print("✅ Comprehensive debugging information")
    print("✅ Action space compatibility")
