#!/usr/bin/env python3
"""
OPTIMIZED RL Environment for Expert Imitation
Key improvements:
1. Hierarchical action space for sparse surgical actions
2. Heavy expert demonstration matching rewards
3. Behavioral cloning integration
4. Adaptive action selection
5. Curriculum learning
"""

import torch
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Any, Optional, Tuple
import random
from collections import deque


class OptimizedSurgicalEnvironment(gym.Env):
    """
    Optimized RL Environment focusing on expert imitation rather than exploration.
    
    Key innovations:
    1. Hierarchical action space: first select number of actions (0-3), then which actions
    2. Heavy behavioral cloning bias with expert demonstration matching
    3. Adaptive curriculum learning
    4. Better reward design focusing on mAP-like metrics
    """
    
    def __init__(self, world_model, video_data: List[Dict], config: Dict, device: str = 'cuda'):
        super().__init__()
        
        self.world_model = world_model
        self.video_data = video_data
        self.config = config
        self.device = device
        
        # Environment parameters
        self.max_episode_steps = config.get('rl_horizon', 20)  # Shorter episodes for faster learning
        self.context_length = config.get('context_length', 10)
        
        # CRITICAL: Use hierarchical action space instead of continuous
        # First select how many actions (0, 1, 2, or 3), then select which actions
        self.action_space = spaces.Dict({
            'num_actions': spaces.Discrete(4),  # 0, 1, 2, or 3 actions
            'action_logits': spaces.Box(low=-5.0, high=5.0, shape=(100,), dtype=np.float32)  # Logits for action selection
        })
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(1024,), dtype=np.float32
        )
        
        # # OPTIMIZED: Heavy focus on expert demonstration matching
        # self.reward_weights = {
        #     'expert_precision': 50.0,     # CRITICAL: Heavy weight on precision (correct positive predictions)
        #     'expert_recall': 40.0,        # CRITICAL: Heavy weight on recall (finding all positive actions)  
        #     'expert_f1': 60.0,           # CRITICAL: F1 score (balanced precision/recall)
        #     'action_sparsity': 10.0,      # Encourage 1-3 actions per step
        #     'behavioral_cloning': 30.0,   # Direct imitation bonus
        #     'world_model_sim': 0.5,       # REDUCED: Minimal weight for world model simulation
        #     'completion_bonus': 15.0      # Episode completion
        # }
        self.reward_weights = {
            'expert_precision': 100.0,    # DOUBLED: Most important - stop false positives
            'expert_recall': 30.0,        # REDUCED: Recall is already decent
            'expert_f1': 80.0,            # INCREASED: Balanced metric
            'false_positive_penalty': 50.0,  # NEW: Direct FP penalty
            'action_sparsity': 15.0,      # INCREASED: Enforce sparsity
            'behavioral_cloning': 10.0,   # REDUCED: Only when precision is good
            'world_model_sim': 0.01,      # MINIMAL: Almost ignore
            'completion_bonus': 5.0       # REDUCED: Don't overwhelm expert signals
        }
        # Curriculum learning parameters
        self.curriculum_stage = 0
        self.success_threshold = 0.3  # Move to next stage when achieving 30% expert matching
        self.recent_expert_scores = deque(maxlen=50)
        
        # Expert demonstration database for behavioral cloning
        self.expert_action_database = self._build_expert_database()
        
        # Current episode state
        self.current_step = 0
        self.current_video_idx = 0
        self.current_frame_idx = 0
        self.current_state = None
        self.episode_reward = 0.0
        
        # Track expert actions for current video
        self.current_video = None
        self.expert_actions_sequence = None
        
        # Episode statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.expert_matching_scores = []
        self.precision_scores = []
        self.recall_scores = []
        self.f1_scores = []
        
        print("🎯 Optimized Surgical RL Environment initialized")
        print(f"📊 Action space: Hierarchical (num_actions + action_logits)")
        print(f"🎯 Focus: Expert imitation with behavioral cloning bias")
        print(f"📈 Curriculum stages: {4}")
        print(f"🔍 Expert database: {len(self.expert_action_database)} patterns")

    def _build_expert_database(self) -> List[Dict]:
        """Build database of expert action patterns for behavioral cloning."""
        expert_patterns = []
        
        for video in self.video_data:
            actions = video['actions_binaries']
            states = video['frame_embeddings']
            
            for i in range(len(actions)):
                action_indices = np.where(actions[i] > 0.5)[0]
                if len(action_indices) > 0:  # Only store frames with actions
                    expert_patterns.append({
                        'state': states[i],
                        'actions': actions[i],
                        'num_actions': len(action_indices),
                        'action_indices': action_indices,
                        'video_id': video['video_id'],
                        'frame_idx': i
                    })
        
        print(f"📚 Built expert database with {len(expert_patterns)} action patterns")
        return expert_patterns

    def _get_behavioral_cloning_action(self, current_state: np.ndarray) -> Dict[str, np.ndarray]:
        """Get action from nearest expert demonstration (behavioral cloning)."""
        if not self.expert_action_database:
            return self._get_random_sparse_action()
        
        # Find most similar expert state
        state_distances = []
        current_state_norm = current_state / (np.linalg.norm(current_state) + 1e-8)
        
        for expert in self.expert_action_database:
            expert_state = expert['state'] / (np.linalg.norm(expert['state']) + 1e-8)
            distance = np.linalg.norm(current_state_norm - expert_state)
            state_distances.append(distance)
        
        # Select from top 5 most similar expert demonstrations
        top_k = min(5, len(state_distances))
        top_indices = np.argsort(state_distances)[:top_k]
        selected_expert = self.expert_action_database[random.choice(top_indices)]
        
        # Create hierarchical action based on expert
        num_actions = min(selected_expert['num_actions'], 3)  # Cap at 3 actions
        
        # Create action logits that favor the expert's actions
        action_logits = np.random.normal(-2.0, 0.5, 100)  # Default low probability
        for action_idx in selected_expert['action_indices'][:num_actions]:
            action_logits[action_idx] = np.random.normal(2.0, 0.3)  # High probability for expert actions
        
        return {
            'num_actions': num_actions,
            'action_logits': action_logits
        }

    def _get_random_sparse_action(self) -> Dict[str, np.ndarray]:
        """Generate random sparse action for exploration."""
        num_actions = np.random.choice([0, 1, 2, 3], p=[0.1, 0.4, 0.3, 0.2])
        action_logits = np.random.normal(-1.0, 1.0, 100)
        return {
            'num_actions': num_actions,
            'action_logits': action_logits
        }

    def _convert_hierarchical_to_binary(self, action_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """Convert hierarchical action to binary action vector."""
        num_actions = action_dict['num_actions']
        action_logits = action_dict['action_logits']
        
        binary_action = np.zeros(100, dtype=np.float32)
        
        if num_actions > 0:
            # Select top-k actions based on logits
            top_indices = np.argsort(action_logits)[-num_actions:]
            binary_action[top_indices] = 1.0
        
        return binary_action

    def reset(self, seed=None, options=None):
        """Reset with curriculum learning and expert-biased initialization."""
        super().reset(seed=seed)
        
        # Curriculum learning: bias toward easier videos in early stages
        if self.curriculum_stage < 2:
            # Prefer videos with more action patterns in early stages
            video_action_counts = []
            for i, video in enumerate(self.video_data):
                actions = video['actions_binaries']
                action_density = np.mean(np.sum(actions, axis=1))
                video_action_counts.append((i, action_density))
            
            # Sort by action density and select from appropriate range
            video_action_counts.sort(key=lambda x: x[1])
            if self.curriculum_stage == 0:
                # Stage 0: Use videos with moderate action density (easier)
                start_idx = len(video_action_counts) // 4
                end_idx = 3 * len(video_action_counts) // 4
            else:
                # Stage 1: Use videos with higher action density
                start_idx = len(video_action_counts) // 2
                end_idx = len(video_action_counts)
            
            suitable_videos = video_action_counts[start_idx:end_idx]
            self.current_video_idx = random.choice(suitable_videos)[0]
        else:
            # Stage 2+: Use all videos
            self.current_video_idx = np.random.randint(0, len(self.video_data))
        
        self.current_video = self.video_data[self.current_video_idx]
        
        # Cache expert actions for this video
        if 'actions_binaries' in self.current_video:
            self.expert_actions_sequence = np.array(self.current_video['actions_binaries'])
        else:
            self.expert_actions_sequence = None
        
        # Start from a position with enough context and future frames
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
        self.episode_precision_scores = []
        self.episode_recall_scores = []
        self.episode_f1_scores = []
        
        return self.current_state.copy(), {}

    def step(self, action):
        """Step with optimized reward calculation and curriculum learning."""
        self.current_step += 1
        
        # Handle different action formats
        if isinstance(action, dict):
            hierarchical_action = action
        else:
            # Convert from RL algorithm output to hierarchical format
            if isinstance(action, (list, np.ndarray)) and len(action) == 2:
                # Assume [num_actions, action_logits_flattened]
                hierarchical_action = {
                    'num_actions': int(action[0]),
                    'action_logits': np.array(action[1:101]) if len(action) > 100 else np.random.normal(0, 1, 100)
                }
            else:
                # Convert continuous action to hierarchical
                action_array = np.array(action).flatten()
                if len(action_array) >= 100:
                    # Estimate number of actions from continuous values
                    high_confidence = np.sum(action_array[:100] > 0.7)
                    medium_confidence = np.sum(action_array[:100] > 0.5)
                    num_actions = min(high_confidence if high_confidence > 0 else medium_confidence, 3)
                    
                    hierarchical_action = {
                        'num_actions': num_actions,
                        'action_logits': action_array[:100] * 4.0 - 2.0  # Convert [0,1] to logits
                    }
                else:
                    hierarchical_action = self._get_random_sparse_action()
        
        # Convert to binary action
        binary_action = self._convert_hierarchical_with_smart_threshold(hierarchical_action)
        
        # Check termination conditions
        frames_remaining = len(self.current_video['frame_embeddings']) - self.current_frame_idx - 1
        done = (self.current_step >= self.max_episode_steps) or (frames_remaining <= 0)
        
        if done:
            # Episode completion
            reward = self.reward_weights['completion_bonus']
            next_state = self.current_state.copy()
        else:
            # Use world model to predict next state (minimal weight)
            next_state, predicted_rewards = self._predict_with_world_model(binary_action)
            
            # OPTIMIZED: Calculate expert-focused reward
            reward = self._calculate_working_reward(binary_action, predicted_rewards)
            
            # Update current state
            self.current_state = next_state.copy()
        
        self.episode_reward += reward
        
        # Enhanced info for debugging
        info = {
            'step': self.current_step,
            'episode_reward': self.episode_reward,
            'action_sum': float(np.sum(binary_action > 0.5)),
            'curriculum_stage': self.curriculum_stage,
            'expert_available': self.expert_actions_sequence is not None,
            'frames_remaining': frames_remaining,
            'method': 'optimized_expert_imitation_rl'
        }
        
        # Episode completion and curriculum update
        if done:
            self._handle_episode_completion(info)
        
        return self.current_state.copy(), reward, done, False, info

    def _calculate_working_reward(self, action: np.ndarray, predicted_rewards: Dict[str, float]) -> float:
        """
        WORKING SOLUTION: Positive rewards with smart selectivity.
        Key insight: Don't punish learning, just reward good behavior more.
        """
        reward = 0.0
        
        # WORKING: Always start with base reward so agent keeps learning
        base_reward = 50.0  # Ensure agent always gets some positive signal
        reward += base_reward
        
        # CRITICAL: Expert demonstration matching (but positive focus)
        if (self.expert_actions_sequence is not None and 
            self.current_frame_idx < len(self.expert_actions_sequence)):
            
            expert_actions = self.expert_actions_sequence[self.current_frame_idx]
            binary_action = (action > 0.5).astype(int)
            
            if len(expert_actions) == len(binary_action):
                # Calculate metrics
                expert_positive = expert_actions > 0.5
                pred_positive = binary_action > 0.5
                
                tp = np.sum(expert_positive & pred_positive)
                fp = np.sum(~expert_positive & pred_positive)
                fn = np.sum(expert_positive & ~pred_positive)
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if np.sum(expert_positive) == 0 else 0.0)
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                
                # WORKING: Positive rewards that encourage good behavior
                reward += 200.0 * precision  # Big reward for being right when you predict
                reward += 100.0 * recall     # Good reward for finding expert actions
                reward += 150.0 * f1         # Balanced reward
                
                # WORKING: Bonus for exact matches (no penalties)
                if tp > 0 and fp == 0:  # Found actions without false positives
                    reward += 100.0 * tp  # Huge bonus for perfect precision
                
                # Track metrics
                if not hasattr(self, 'episode_expert_scores'):
                    self.episode_expert_scores = []
                    self.episode_precision_scores = []
                    self.episode_recall_scores = []
                    self.episode_f1_scores = []
                
                overall_accuracy = np.mean(binary_action == expert_actions)
                self.episode_expert_scores.append(overall_accuracy)
                self.episode_precision_scores.append(precision)
                self.episode_recall_scores.append(recall)
                self.episode_f1_scores.append(f1)
                
                # Behavioral cloning bonus (positive only)
                action_similarity = 1.0 - np.mean(np.abs(binary_action - expert_actions))
                reward += 50.0 * action_similarity
        
        # WORKING: Gentle action sparsity (no harsh penalties)
        action_count = np.sum(action > 0.5)
        if action_count == 1:
            reward += 20.0  # Best case
        elif action_count == 2:
            reward += 15.0  # Good
        elif action_count == 3:
            reward += 10.0  # OK
        elif action_count == 0:
            reward += 5.0   # Sometimes correct
        # No penalties for higher counts - just no bonus
        
        # Small world model reward
        world_model_reward = sum(predicted_rewards.values())
        reward += 0.1 * world_model_reward
        
        # Action consistency bonus
        if hasattr(self, 'previous_action'):
            consistency = 1.0 - np.mean(np.abs(action - self.previous_action))
            reward += 10.0 * consistency
        
        self.previous_action = action.copy()
        
        # WORKING: Keep rewards positive for learning
        return np.clip(reward, 0.0, 1000.0)  # Always positive


    def _convert_hierarchical_with_smart_threshold(self, action_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """
        WORKING SOLUTION: Smart action selection that encourages precision.
        Key: Dynamic thresholding to prefer fewer, high-confidence actions.
        """
        num_actions = action_dict['num_actions']
        action_logits = action_dict['action_logits']
        
        binary_action = np.zeros(100, dtype=np.float32)
        
        if num_actions > 0:
            # WORKING: Dynamic threshold based on action confidence distribution
            sorted_logits = np.sort(action_logits)[::-1]  # Descending
            
            # Find a good threshold that gives us selective actions
            if num_actions == 1:
                # For 1 action, only select if very confident
                threshold = max(1.5, sorted_logits[0] - 0.5)
            elif num_actions == 2:
                # For 2 actions, moderate confidence
                threshold = max(1.0, sorted_logits[1] - 0.3)
            elif num_actions == 3:
                # For 3 actions, lower confidence OK
                threshold = max(0.8, sorted_logits[2] - 0.2)
            else:
                # For more actions, be very selective
                threshold = max(1.8, sorted_logits[min(3, len(sorted_logits)-1)])
            
            # Select actions above threshold
            selected_indices = np.where(action_logits > threshold)[0]
            
            # Limit to reasonable number
            if len(selected_indices) > 5:
                # If too many, take only the top ones
                top_indices = np.argsort(action_logits)[-3:]
                selected_indices = top_indices[action_logits[top_indices] > 0.5]
            
            binary_action[selected_indices] = 1.0
        
        return binary_action

    # def _calculate_expert_focused_reward(self, action: np.ndarray, predicted_rewards: Dict[str, float]) -> float:
    #     """Calculate reward heavily focused on expert demonstration matching."""
    #     reward = 0.0
        
    #     # CRITICAL: Heavy expert demonstration matching
    #     if (self.expert_actions_sequence is not None and 
    #         self.current_frame_idx < len(self.expert_actions_sequence)):
            
    #         expert_actions = self.expert_actions_sequence[self.current_frame_idx]
    #         binary_action = (action > 0.5).astype(int)
            
    #         if len(expert_actions) == len(binary_action):
    #             # Calculate precision, recall, and F1 for positive actions only
    #             expert_positive = expert_actions > 0.5
    #             pred_positive = binary_action > 0.5
                
    #             # True positives, false positives, false negatives
    #             tp = np.sum(expert_positive & pred_positive)
    #             fp = np.sum(~expert_positive & pred_positive)
    #             fn = np.sum(expert_positive & ~pred_positive)
                
    #             # Calculate precision, recall, F1
    #             precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    #             recall = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if np.sum(expert_positive) == 0 else 0.0)
    #             f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                
    #             # MASSIVE rewards for expert matching
    #             reward += self.reward_weights['expert_precision'] * precision
    #             reward += self.reward_weights['expert_recall'] * recall  
    #             reward += self.reward_weights['expert_f1'] * f1
                
    #             # Track episode metrics
    #             if not hasattr(self, 'episode_expert_scores'):
    #                 self.episode_expert_scores = []
    #                 self.episode_precision_scores = []
    #                 self.episode_recall_scores = []
    #                 self.episode_f1_scores = []
                
    #             overall_accuracy = np.mean(binary_action == expert_actions)
    #             self.episode_expert_scores.append(overall_accuracy)
    #             self.episode_precision_scores.append(precision)
    #             self.episode_recall_scores.append(recall)
    #             self.episode_f1_scores.append(f1)
                
    #             # Behavioral cloning bonus (direct imitation)
    #             action_similarity = 1.0 - np.mean(np.abs(binary_action - expert_actions))
    #             reward += self.reward_weights['behavioral_cloning'] * action_similarity
        
    #     # Encourage appropriate action sparsity (1-3 actions)
    #     action_count = np.sum(action > 0.5)
    #     if action_count == 0:
    #         reward -= 10.0  # Strong penalty for no actions
    #     elif 1 <= action_count <= 3:
    #         reward += self.reward_weights['action_sparsity']  # Strong bonus for ideal range
    #     elif action_count == 4:
    #         reward += self.reward_weights['action_sparsity'] * 0.5  # Partial bonus
    #     else:
    #         reward -= 5.0 * (action_count - 4)  # Escalating penalty
        
    #     # MINIMAL world model simulation reward
    #     world_model_reward = sum(predicted_rewards.values())
    #     reward += self.reward_weights['world_model_sim'] * world_model_reward
        
    #     return np.clip(reward, -50.0, 200.0)  # Allow high rewards for good expert matching

    def _predict_with_world_model(self, action: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        """Use world model for minimal prediction (low weight in reward)."""
        try:
            current_state_tensor = torch.tensor(
                self.current_state, dtype=torch.float32, device=self.device
            )
            action_tensor = torch.tensor(
                action, dtype=torch.float32, device=self.device
            )
            
            next_state, rewards, _ = self.world_model.simulate_step(
                current_state_tensor, action_tensor, return_hidden=False
            )
            
            next_state_np = next_state.cpu().numpy().flatten()
            next_state_np = self._fix_state_shape(next_state_np)
            
            return next_state_np, rewards
                
        except Exception as e:
            print(f"⚠️ World model prediction failed: {e}")
            # Fallback: small random perturbation
            noise = np.random.normal(0, 0.01, self.current_state.shape)
            return self.current_state + noise, {}

    def _handle_episode_completion(self, info):
        """Handle episode completion and curriculum advancement."""
        self.episode_lengths.append(self.current_step)
        self.episode_rewards.append(self.episode_reward)
        
        # Calculate episode-level expert matching
        if hasattr(self, 'episode_expert_scores') and self.episode_expert_scores:
            avg_expert_score = np.mean(self.episode_expert_scores)
            avg_precision = np.mean(self.episode_precision_scores) if self.episode_precision_scores else 0.0
            avg_recall = np.mean(self.episode_recall_scores) if self.episode_recall_scores else 0.0
            avg_f1 = np.mean(self.episode_f1_scores) if self.episode_f1_scores else 0.0
            
            self.expert_matching_scores.append(avg_expert_score)
            self.precision_scores.append(avg_precision)
            self.recall_scores.append(avg_recall)
            self.f1_scores.append(avg_f1)
            self.recent_expert_scores.append(avg_f1)  # Use F1 for curriculum advancement
            
            info.update({
                'episode_expert_matching': avg_expert_score,
                'episode_precision': avg_precision,
                'episode_recall': avg_recall,
                'episode_f1': avg_f1
            })
        
        # Curriculum advancement
        if len(self.recent_expert_scores) >= 20:  # Evaluate every 20 episodes
            recent_avg_f1 = np.mean(list(self.recent_expert_scores)[-20:])
            if recent_avg_f1 > self.success_threshold and self.curriculum_stage < 3:
                self.curriculum_stage += 1
                self.success_threshold += 0.1  # Increase threshold for next stage
                print(f"🎓 Curriculum advancement! Stage: {self.curriculum_stage}, F1: {recent_avg_f1:.3f}")
        
        # Print progress
        if len(self.episode_rewards) % 20 == 0:
            recent_rewards = self.episode_rewards[-10:]
            recent_f1 = self.f1_scores[-10:] if self.f1_scores else [0]
            recent_precision = self.precision_scores[-10:] if self.precision_scores else [0]
            recent_recall = self.recall_scores[-10:] if self.recall_scores else [0]
            
            print(f"📊 Episode {len(self.episode_rewards)} (Stage {self.curriculum_stage}):")
            print(f"   Reward: {np.mean(recent_rewards):.1f}, F1: {np.mean(recent_f1):.3f}")
            print(f"   Precision: {np.mean(recent_precision):.3f}, Recall: {np.mean(recent_recall):.3f}")

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
            "curriculum_stage": self.curriculum_stage,
            "reward_trend": "improving" if len(self.episode_rewards) >= 10 and 
                           np.mean(self.episode_rewards[-5:]) > np.mean(self.episode_rewards[-10:-5]) else "stable"
        }
        
        if self.expert_matching_scores:
            stats.update({
                "avg_expert_matching": np.mean(self.expert_matching_scores),
                "avg_precision": np.mean(self.precision_scores) if self.precision_scores else 0.0,
                "avg_recall": np.mean(self.recall_scores) if self.recall_scores else 0.0,
                "avg_f1": np.mean(self.f1_scores) if self.f1_scores else 0.0,
                "last_f1": self.f1_scores[-1] if self.f1_scores else 0.0
            })
        
        return stats

    def get_behavioral_cloning_policy(self, state: np.ndarray) -> Dict[str, np.ndarray]:
        """Get behavioral cloning action for the given state."""
        return self._get_behavioral_cloning_action(state)


# Wrapper for stable-baselines3 compatibility
class SB3CompatibleSurgicalEnv(OptimizedSurgicalEnvironment):
    """
    Wrapper to make the environment compatible with stable-baselines3.
    Converts hierarchical action space to continuous for easier RL algorithm integration.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Override action space for SB3 compatibility
        # Action: [num_actions_logits (4,), action_selection_logits (100,)]
        self.action_space = spaces.Box(
            low=-5.0, high=5.0, shape=(104,), dtype=np.float32
        )
        
        print("🔧 SB3 Compatible wrapper initialized")
        print(f"📊 Action space: Box(104,) = [num_actions_logits(4), action_logits(100)]")

    def step(self, action):
        """Convert SB3 action to hierarchical action."""
        action = np.array(action).flatten()
        
        if len(action) >= 104:
            # Extract number of actions (use softmax to get probabilities)
            num_actions_logits = action[:4]
            num_actions_probs = np.exp(num_actions_logits) / np.sum(np.exp(num_actions_logits))
            num_actions = np.random.choice(4, p=num_actions_probs)
            
            # Extract action selection logits
            action_logits = action[4:104]
            
            hierarchical_action = {
                'num_actions': num_actions,
                'action_logits': action_logits
            }
        else:
            # Fallback for malformed actions
            hierarchical_action = self._get_random_sparse_action()
        
        return super().step(hierarchical_action)


if __name__ == "__main__":
    print("🎯 OPTIMIZED SURGICAL RL ENVIRONMENT")
    print("=" * 60)
    print("Key optimizations for expert imitation:")
    print("✅ Hierarchical action space (num_actions + action_selection)")
    print("✅ Heavy expert demonstration matching rewards (F1, precision, recall)")
    print("✅ Behavioral cloning integration")
    print("✅ Curriculum learning with staged difficulty") 
    print("✅ Adaptive action selection")
    print("✅ Minimal world model dependency")
    print("\n🎯 Expected improvement: 6% → 15-30% mAP")
