# ===================================================================
# File: rl_environment.py 
# Enhanced World Model Environment for RL Training
# ===================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Tuple, Any, Optional, List
import logging

# local imports
from models.world_model import WorldModel


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
    Enhanced gym environment with proper world model integration
    """
    
    def __init__(self, world_model, config: dict, device='cuda'):
        super().__init__()
        self.model = world_model.eval()
        self.config = config
        self.device = device
        self.horizon = config.get('rl_horizon', 50)
        self.context_length = config.get('context_length', 10)

        # Action space: binary actions matching your world model
        self.action_space = spaces.MultiBinary(world_model.num_action_classes)
        
        # Observation space: frame embeddings
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
        
        # Video contexts for realistic initialization
        self.video_contexts = {}
        
        # Reward weights from config
        self.reward_weights = config.get('reward_weights', {
            '_r_phase_completion': 1.0,
            '_r_phase_initiation': 0.5,
            '_r_phase_progression': 1.0,
            '_r_global_progression': 0.8,
            '_r_action_probability': 0.3,
            '_r_risk': -0.5,  # Penalty
        })
        
    def set_video_context(self, video_data: List[Dict]):
        """Set context from your cholect50 video data"""
        self.video_contexts = {video['video_id']: video for video in video_data}
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Enhanced reset with video context initialization"""
        super().reset(seed=seed)
        
        if options and 'video_id' in options and options['video_id'] in self.video_contexts:
            # Initialize from real video context
            video_data = self.video_contexts[options['video_id']]
            embeddings = video_data['frame_embeddings']
            
            # Sample random starting point with enough context
            max_start = len(embeddings) - self.context_length - 1
            start_idx = np.random.randint(0, max_start) if max_start > 0 else 0
            
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
            # Random initialization
            self.current_state_sequence = torch.randn(
                1, self.context_length, self.model.embedding_dim, device=self.device
            )
            self.current_phase = 0
            
        self.step_count = 0
        self.episode_rewards = []
        
        # Return the last frame as observation
        return self.current_state_sequence[0, -1].cpu().numpy(), {}
    
    def step(self, action):
        """Enhanced step function using your world model with proper inference"""
        # Convert action to tensor
        action_tensor = torch.tensor(
            action, dtype=torch.float32, device=self.device
        ).unsqueeze(0).unsqueeze(0)  # [1, 1, num_action_classes]
        
        # Expand action to match context length for world model input
        action_sequence = action_tensor.expand(1, self.context_length, -1)

        with torch.no_grad():
            # Forward pass through your world model in INFERENCE mode
            output = self.model(
                current_state=self.current_state_sequence,
                next_actions=action_sequence,
                eval_mode='basic'  # This tells the model we're doing inference, not training
            )
            
            # Get next state prediction
            next_state = output['_z_hat'][:, -1:, :].detach()  # [1, 1, embedding_dim]
            
            # Aggregate rewards using your reward structure
            total_reward = 0.0
            reward_breakdown = {}
            
            for reward_type, weight in self.reward_weights.items():
                loss_key = f'{reward_type}_loss'
                if loss_key in output:
                    # The model outputs negative values for rewards (as losses)
                    # We negate them to get positive rewards
                    reward_value = -output[loss_key][:, -1].mean().item()
                    weighted_reward = reward_value * weight
                    total_reward += weighted_reward
                    reward_breakdown[reward_type] = reward_value
            
            # Update phase if predicted
            if '_p_hat' in output:
                # Get phase prediction from the model
                phase_logits = output['_p_hat'][:, -1]  # [1, num_phases]
                phase_probs = F.softmax(phase_logits, dim=-1)
                self.current_phase = torch.argmax(phase_probs, dim=-1).item()

        # Update state sequence (slide window)
        self.current_state_sequence = torch.cat([
            self.current_state_sequence[:, 1:, :],  # Remove first frame
            next_state  # Add new frame
        ], dim=1)
        
        self.step_count += 1
        self.episode_rewards.append(total_reward)
        
        # Episode termination
        done = self.step_count >= self.horizon
        
        # Enhanced info
        info = {
            'step': self.step_count,
            'current_phase': self.current_phase,
            'reward_breakdown': reward_breakdown,
            'episode_reward': sum(self.episode_rewards),
        }
        
        obs = next_state.squeeze(0).squeeze(0).cpu().numpy()
        return obs, total_reward, done, False, info

# class SurgicalWorldModelEnv(gym.Env):
#     """
#     Enhanced gym environment that integrates with your existing WorldModel
#     """
    
#     def __init__(self, world_model: WorldModel, config: dict, device='cuda'):
#         super().__init__()
#         self.model = world_model.eval()
#         self.config = config
#         self.device = device
#         self.horizon = config.get('rl_horizon', 50)
#         self.context_length = config.get('context_length', 10)

#         # Action space: binary actions matching your world model
#         self.action_space = spaces.MultiBinary(world_model.num_action_classes)
        
#         # Observation space: frame embeddings
#         self.observation_space = spaces.Box(
#             low=-np.inf, high=np.inf, 
#             shape=(world_model.embedding_dim,), 
#             dtype=np.float32
#         )
        
#         # State tracking
#         self.current_state_sequence = None
#         self.current_phase = None
#         self.step_count = 0
#         self.episode_rewards = []
        
#         # Video contexts for realistic initialization
#         self.video_contexts = {}
        
#         # Reward weights from config
#         self.reward_weights = config.get('reward_weights', {
#             '_r_phase_completion': 1.0,
#             '_r_phase_initiation': 0.5,
#             '_r_phase_progression': 1.0,
#             '_r_global_progression': 0.8,
#             '_r_action_probability': 0.3,
#             '_r_risk': -0.5,  # Penalty
#         })
        
#     def set_video_context(self, video_data: List[Dict]):
#         """Set context from your cholect50 video data"""
#         self.video_contexts = {video['video_id']: video for video in video_data}
        
#     def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
#         """Enhanced reset with video context initialization"""
#         super().reset(seed=seed)
        
#         if options and 'video_id' in options and options['video_id'] in self.video_contexts:
#             # Initialize from real video context
#             video_data = self.video_contexts[options['video_id']]
#             embeddings = video_data['frame_embeddings']
            
#             # Sample random starting point with enough context
#             max_start = len(embeddings) - self.context_length - 1
#             start_idx = np.random.randint(0, max_start) if max_start > 0 else 0
            
#             # Get context sequence
#             context_embeddings = embeddings[start_idx:start_idx + self.context_length]
#             self.current_state_sequence = torch.tensor(
#                 context_embeddings, dtype=torch.float32, device=self.device
#             ).unsqueeze(0)  # [1, context_length, embedding_dim]
            
#             # Set initial phase if available
#             if 'phase_binaries' in video_data:
#                 phase_binary = video_data['phase_binaries'][start_idx + self.context_length - 1]
#                 self.current_phase = np.argmax(phase_binary)
#         else:
#             # Random initialization
#             self.current_state_sequence = torch.randn(
#                 1, self.context_length, self.model.embedding_dim, device=self.device
#             )
#             self.current_phase = 0
            
#         self.step_count = 0
#         self.episode_rewards = []
        
#         # Return the last frame as observation
#         return self.current_state_sequence[0, -1].cpu().numpy(), {}
    
#     def step(self, action):
#         """Enhanced step function using your world model"""
#         # Convert action to tensor
#         action_tensor = torch.tensor(
#             action, dtype=torch.float32, device=self.device
#         ).unsqueeze(0).unsqueeze(0)  # [1, 1, num_action_classes]
        
#         # Expand action to match context length for world model input
#         action_sequence = action_tensor.expand(1, self.context_length, -1)

#         with torch.no_grad():
#             # Forward pass through your world model
#             output = self.model(
#                 current_state=self.current_state_sequence,
#                 next_actions=action_sequence,
#                 eval_mode='basic'
#             )
            
#             # Get next state prediction
#             next_state = output['_z_hat'][:, -1:, :].detach()  # [1, 1, embedding_dim]
            
#             # Aggregate rewards using your reward structure
#             total_reward = 0.0
#             reward_breakdown = {}
            
#             for reward_type, weight in self.reward_weights.items():
#                 loss_key = f'{reward_type}_loss'
#                 if loss_key in output:
#                     # Convert loss to reward (negate since your model outputs losses)
#                     reward_value = -output[loss_key][:, -1].item()
#                     weighted_reward = reward_value * weight
#                     total_reward += weighted_reward
#                     reward_breakdown[reward_type] = reward_value
            
#             # Update phase if predicted
#             if '_p_loss' in output and hasattr(self.model, 'heads') and '_p' in self.model.heads:
#                 # Get phase prediction from the model
#                 phase_logits = self.model.heads['_p'](output['last_hidden_states'][:, -1])
#                 phase_probs = F.softmax(phase_logits, dim=-1)
#                 self.current_phase = torch.argmax(phase_probs, dim=-1).item()

#         # Update state sequence (slide window)
#         self.current_state_sequence = torch.cat([
#             self.current_state_sequence[:, 1:, :],  # Remove first frame
#             next_state  # Add new frame
#         ], dim=1)
        
#         self.step_count += 1
#         self.episode_rewards.append(total_reward)
        
#         # Episode termination
#         done = self.step_count >= self.horizon
        
#         # Enhanced info
#         info = {
#             'step': self.step_count,
#             'current_phase': self.current_phase,
#             'reward_breakdown': reward_breakdown,
#             'episode_reward': sum(self.episode_rewards),
#         }
        
#         obs = next_state.squeeze(0).squeeze(0).cpu().numpy()
#         return obs, total_reward, done, False, info