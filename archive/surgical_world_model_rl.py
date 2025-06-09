import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Tuple, Any, Optional, List
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class WorldModelConfig:
    """Configuration for the world model"""
    latent_dim: int = 512
    action_dim: int = 3  # For triplet actions (instrument, verb, target)
    embed_dim: int = 768  # ViT embedding dimension
    hidden_dim: int = 1024
    num_layers: int = 4
    sequence_length: int = 16
    dropout: float = 0.1
    
class SurgicalTripletEmbedding(nn.Module):
    """Embeds surgical action triplets (instrument, verb, target)"""
    
    def __init__(self, config: WorldModelConfig):
        super().__init__()
        self.config = config
        
        # Embedding dimensions for CholecT50 dataset
        self.instrument_vocab_size = 6  # Number of instruments in CholecT50
        self.verb_vocab_size = 10       # Number of verbs
        self.target_vocab_size = 15     # Number of target anatomies
        
        self.instrument_embed = nn.Embedding(self.instrument_vocab_size, config.embed_dim // 3)
        self.verb_embed = nn.Embedding(self.verb_vocab_size, config.embed_dim // 3)
        self.target_embed = nn.Embedding(self.target_vocab_size, config.embed_dim // 3)
        
        self.projection = nn.Linear(config.embed_dim, config.latent_dim)
        
    def forward(self, triplet_actions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            triplet_actions: (batch_size, 3) - [instrument_id, verb_id, target_id]
        """
        instruments = self.instrument_embed(triplet_actions[:, 0])
        verbs = self.verb_embed(triplet_actions[:, 1])
        targets = self.target_embed(triplet_actions[:, 2])
        
        # Concatenate and project
        combined = torch.cat([instruments, verbs, targets], dim=-1)
        return self.projection(combined)

class SurgicalWorldModel(nn.Module):
    """
    World model for surgical environment using Transformer architecture
    """
    
    def __init__(self, config: WorldModelConfig):
        super().__init__()
        self.config = config
        
        # Input projections
        self.frame_projection = nn.Linear(config.embed_dim, config.latent_dim)
        self.triplet_embedding = SurgicalTripletEmbedding(config)
        
        # Transformer encoder for temporal modeling
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.latent_dim,
            nhead=8,
            dim_feedforward=config.hidden_dim,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, config.num_layers)
        
        # State prediction heads
        self.next_state_predictor = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.latent_dim)
        )
        
        # Reward prediction (for surgical task completion)
        self.reward_predictor = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, 1)
        )
        
        # Done prediction (episode termination)
        self.done_predictor = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, frame_embeddings: torch.Tensor, 
                triplet_actions: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            frame_embeddings: (batch_size, seq_len, embed_dim) - Pre-extracted ViT features
            triplet_actions: (batch_size, seq_len, 3) - Action triplets
            mask: (batch_size, seq_len) - Padding mask
            
        Returns:
            next_states: (batch_size, seq_len, latent_dim)
            rewards: (batch_size, seq_len, 1)
            dones: (batch_size, seq_len, 1)
        """
        batch_size, seq_len = frame_embeddings.shape[:2]
        
        # Project frame embeddings to latent space
        frame_latents = self.frame_projection(frame_embeddings)
        
        # Embed triplet actions
        triplet_latents = self.triplet_embedding(triplet_actions.view(-1, 3))
        triplet_latents = triplet_latents.view(batch_size, seq_len, -1)
        
        # Combine frame and action representations
        combined_input = frame_latents + triplet_latents
        
        # Apply transformer with mask
        if mask is not None:
            # Convert mask to attention mask (True = ignore)
            attn_mask = ~mask.bool()
        else:
            attn_mask = None
            
        transformer_output = self.transformer(combined_input, src_key_padding_mask=attn_mask)
        
        # Predictions
        next_states = self.next_state_predictor(transformer_output)
        rewards = self.reward_predictor(transformer_output)
        dones = self.done_predictor(transformer_output)
        
        return next_states, rewards, dones

class SurgicalWorldModelGym(gym.Env):
    """
    Gym environment wrapper for the surgical world model
    """
    
    def __init__(self, world_model: SurgicalWorldModel, config: WorldModelConfig):
        super().__init__()
        self.world_model = world_model
        self.config = config
        
        # Action space: triplet actions (instrument, verb, target)
        self.action_space = spaces.Box(
            low=np.array([0, 0, 0]), 
            high=np.array([5, 9, 14]),  # Max indices for each category
            dtype=np.int32
        )
        
        # Observation space: latent state representation
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(config.latent_dim,),
            dtype=np.float32
        )
        
        # Internal state
        self.current_state = None
        self.step_count = 0
        self.max_steps = 1000
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        # Initialize with random latent state or from options
        if options and 'initial_state' in options:
            self.current_state = torch.tensor(options['initial_state'], dtype=torch.float32)
        else:
            self.current_state = torch.randn(self.config.latent_dim)
            
        self.step_count = 0
        
        info = {'step': self.step_count}
        return self.current_state.numpy(), info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment"""
        
        # Convert action to tensor
        action_tensor = torch.tensor(action, dtype=torch.long).unsqueeze(0).unsqueeze(0)
        
        # Dummy frame embedding (in practice, you'd use actual frame from current state)
        dummy_frame = torch.randn(1, 1, self.config.embed_dim)
        
        # Predict next state using world model
        with torch.no_grad():
            next_states, rewards, dones = self.world_model(dummy_frame, action_tensor)
            
        # Update state
        self.current_state = next_states.squeeze(0).squeeze(0)
        reward = rewards.squeeze().item()
        done = dones.squeeze().item() > 0.5
        
        self.step_count += 1
        
        # Episode termination conditions
        truncated = self.step_count >= self.max_steps
        terminated = done
        
        info = {
            'step': self.step_count,
            'predicted_reward': reward,
            'predicted_done': done
        }
        
        return self.current_state.numpy(), reward, terminated, truncated, info

class ImitationLearningTrainer:
    """
    Trainer for pre-training the world model using imitation learning
    """
    
    def __init__(self, world_model: SurgicalWorldModel, config: WorldModelConfig):
        self.world_model = world_model
        self.config = config
        self.optimizer = torch.optim.Adam(world_model.parameters(), lr=1e-4)
        
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Single training step on CholecT50 data
        
        Args:
            batch: Dictionary containing:
                - 'frame_embeddings': (batch_size, seq_len, embed_dim)
                - 'triplet_actions': (batch_size, seq_len, 3)
                - 'next_frame_embeddings': (batch_size, seq_len, embed_dim)
                - 'rewards': (batch_size, seq_len, 1) - task completion rewards
                - 'dones': (batch_size, seq_len, 1) - episode termination flags
                - 'mask': (batch_size, seq_len) - sequence padding mask
        """
        
        # Forward pass
        pred_next_states, pred_rewards, pred_dones = self.world_model(
            batch['frame_embeddings'],
            batch['triplet_actions'],
            batch['mask']
        )
        
        # Ground truth next states (from next frame embeddings)
        target_next_states = self.world_model.frame_projection(batch['next_frame_embeddings'])
        
        # Compute losses
        state_loss = F.mse_loss(pred_next_states, target_next_states, reduction='none')
        reward_loss = F.mse_loss(pred_rewards, batch['rewards'], reduction='none')
        done_loss = F.binary_cross_entropy(pred_dones, batch['dones'], reduction='none')
        
        # Apply mask and average
        mask = batch['mask'].unsqueeze(-1)
        state_loss = (state_loss * mask).sum() / mask.sum()
        reward_loss = (reward_loss * mask).sum() / mask.sum()
        done_loss = (done_loss * mask).sum() / mask.sum()
        
        # Total loss
        total_loss = state_loss + 0.1 * reward_loss + 0.1 * done_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), 1.0)
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'state_loss': state_loss.item(),
            'reward_loss': reward_loss.item(),
            'done_loss': done_loss.item()
        }

# Example usage and integration with RL algorithms
def create_surgical_environment(model_path: Optional[str] = None) -> SurgicalWorldModelGym:
    """Create and return the surgical world model environment"""
    
    config = WorldModelConfig()
    world_model = SurgicalWorldModel(config)
    
    if model_path:
        checkpoint = torch.load(model_path, map_location=DEVICE)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            world_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            world_model.load_state_dict(checkpoint)
    world_model.eval()  # Set to evaluation mode for RL training
    
    return SurgicalWorldModelGym(world_model, config)

# Integration example with Stable-Baselines3
def train_rl_policy():
    """Example of training an RL policy using the world model environment"""
    
    try:
        from stable_baselines3 import PPO, SAC
        from stable_baselines3.common.env_util import make_vec_env
        
        # Create environment
        env = create_surgical_environment("pretrained_world_model.pth")
        
        # Train PPO policy
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs/")
        model.learn(total_timesteps=100000)
        
        # Save trained policy
        model.save("surgical_ppo_policy")
        
        return model
        
    except ImportError:
        print("Stable-Baselines3 not available. Install with: pip install stable-baselines3")
        return None

# TD-MPC2 Integration
class SurgicalTDMPC2Environment:
    """
    Environment adapter for TD-MPC2 (Temporal Difference Model-Predictive Control)
    """
    
    def __init__(self, world_model: SurgicalWorldModel, config: WorldModelConfig):
        self.world_model = world_model
        self.config = config
        self.device = next(world_model.parameters()).device
        
    def predict_sequence(self, initial_state: torch.Tensor, 
                        actions: torch.Tensor, 
                        horizon: int = 16) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict a sequence of states, rewards, and dones for MPC planning
        
        Args:
            initial_state: (batch_size, latent_dim) - Current state
            actions: (batch_size, horizon, 3) - Action sequence
            horizon: Planning horizon
            
        Returns:
            states: (batch_size, horizon, latent_dim)
            rewards: (batch_size, horizon, 1)  
            dones: (batch_size, horizon, 1)
        """
        batch_size = initial_state.shape[0]
        
        # Initialize sequences
        states = torch.zeros(batch_size, horizon, self.config.latent_dim, device=self.device)
        rewards = torch.zeros(batch_size, horizon, 1, device=self.device)
        dones = torch.zeros(batch_size, horizon, 1, device=self.device)
        
        current_state = initial_state
        
        with torch.no_grad():
            for t in range(horizon):
                # Create dummy frame embedding from current state
                # In practice, you might use a decoder to generate frame features
                dummy_frame = current_state.unsqueeze(1)  # (batch_size, 1, latent_dim)
                
                # Get action for this timestep
                action_t = actions[:, t:t+1, :]  # (batch_size, 1, 3)
                
                # Predict next state
                next_state, reward, done = self.world_model(dummy_frame, action_t)
                
                # Store predictions
                states[:, t] = next_state.squeeze(1)
                rewards[:, t] = reward.squeeze(1)
                dones[:, t] = done.squeeze(1)
                
                # Update current state
                current_state = next_state.squeeze(1)
                
        return states, rewards, dones
    
    def get_cost_function(self):
        """Return cost function for MPC optimization"""
        def cost_fn(states, actions, rewards, dones):
            # Surgical task-specific cost function
            # Minimize negative rewards (maximize rewards)
            reward_cost = -rewards.sum(dim=1)
            
            # Penalize early termination
            done_penalty = dones.sum(dim=1) * 10.0
            
            # Action smoothness penalty
            if actions.shape[1] > 1:
                action_diff = torch.diff(actions.float(), dim=1)
                smoothness_cost = (action_diff ** 2).sum(dim=[1, 2]) * 0.1
            else:
                smoothness_cost = 0
                
            return reward_cost + done_penalty + smoothness_cost
            
        return cost_fn

if __name__ == "__main__":
    # Example training pipeline
    
    # 1. Create and pre-train world model
    config = WorldModelConfig()
    world_model = SurgicalWorldModel(config)
    trainer = ImitationLearningTrainer(world_model, config)
    
    print("World model created and ready for pre-training on CholecT50 dataset")
    print(f"Model parameters: {sum(p.numel() for p in world_model.parameters()):,}")
    
    # 2. Create gym environment
    env = SurgicalWorldModelGym(world_model, config)
    print(f"Gym environment created:")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # 3. Test environment
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: action={action}, reward={reward:.3f}, done={terminated}")
        
        if terminated or truncated:
            obs, info = env.reset()
            break