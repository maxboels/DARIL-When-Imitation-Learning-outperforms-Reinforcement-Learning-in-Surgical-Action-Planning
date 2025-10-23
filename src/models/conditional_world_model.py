#!/usr/bin/env python3
"""
Conditional World Model for Method 2
Action-conditioned forward simulation: state + action → next_state + rewards
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import os


class ConditionalWorldModel(nn.Module):
    """
    Method 2: Conditional World Model for RL
    
    Architecture:
    1. State + Action Embeddings → Combined Representation
    2. Transformer → Action-Conditioned Hidden States  
    3. Hidden States → Next State + Reward Predictions
    
    Key: Action-conditioned forward simulation for RL training
    """
    
    def __init__(self,
                 hidden_dim: int = 768,
                 embedding_dim: int = 1024,
                 action_embedding_dim: int = 128,
                 n_layer: int = 6,
                 num_action_classes: int = 100,
                 num_phase_classes: int = 7,
                 dropout: float = 0.1,
                 max_sequence_length: int = 512):
        super().__init__()
        
        print("🌍 Initializing Conditional World Model...")
        
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.action_embedding_dim = action_embedding_dim
        self.num_action_classes = num_action_classes
        self.num_phase_classes = num_phase_classes
        self.dropout = dropout
        self.max_sequence_length = max_sequence_length
        
        # State projection: frame embeddings → hidden space
        self.state_projection = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Action embedding: action vector → action embedding
        self.action_embedding = nn.Sequential(
            nn.Linear(num_action_classes, action_embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Combined projection: state + action → unified representation
        self.combined_projection = nn.Sequential(
            nn.Linear(hidden_dim + action_embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Transformer for action-conditioned modeling
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=max(1, hidden_dim // 64),
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True,
            norm_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            transformer_layer,
            num_layers=n_layer,
            enable_nested_tensor=False
        )
        
        # Next state prediction head
        self.next_state_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, embedding_dim)
        )
        
        # Reward prediction heads (multiple types)
        self.reward_heads = nn.ModuleDict({
            'phase_progression': self._create_reward_head(hidden_dim),
            'phase_completion': self._create_reward_head(hidden_dim),
            'phase_initiation': self._create_reward_head(hidden_dim),
            'action_probability': self._create_reward_head(hidden_dim),
            'safety': self._create_reward_head(hidden_dim),
            'efficiency': self._create_reward_head(hidden_dim),
            'risk_penalty': self._create_reward_head(hidden_dim)
        })
        
        # Phase prediction head (for auxiliary task)
        self.phase_prediction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_phase_classes)
        )
        
        # Value prediction head (for RL value function approximation)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Initialize weights
        self._init_weights()
        
        print(f"✅ Conditional World Model initialized")
        print(f"   Architecture: State + Action → Next State + Rewards")
        print(f"   Hidden dim: {hidden_dim}, Embedding dim: {embedding_dim}")
        print(f"   Action embedding dim: {action_embedding_dim}")
        print(f"   Transformer layers: {n_layer}")
        print(f"   Reward types: {list(self.reward_heads.keys())}")
        print(f"   Key: Action-conditioned forward simulation")
    
    def _create_reward_head(self, hidden_dim: int) -> nn.Module:
        """Create a reward prediction head."""
        return nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def _init_weights(self):
        """Initialize model weights."""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if 'head' in name.lower():
                    # Xavier initialization for prediction heads
                    nn.init.xavier_uniform_(module.weight)
                else:
                    # Small normal initialization for other layers
                    nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self,
                current_states: torch.Tensor,
                next_actions: torch.Tensor,  # FIXED: Renamed to clarify this should be next actions
                target_next_states: Optional[torch.Tensor] = None,
                target_rewards: Optional[Dict[str, torch.Tensor]] = None,
                target_phases: Optional[torch.Tensor] = None,
                return_hidden_states: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass for action-conditioned world model.
        
        Args:
            current_states: [batch_size, seq_len, embedding_dim]
            next_actions: [batch_size, seq_len, num_action_classes] - Actions that cause state transitions
            target_next_states: [batch_size, seq_len, embedding_dim] (for training)
            target_rewards: Dict of [batch_size, seq_len, 1] (for training)
            target_phases: [batch_size, seq_len] (for training)
            return_hidden_states: Whether to return all hidden states
            
        Returns:
            Dictionary containing predictions and losses
        """
        
        batch_size, seq_len, _ = current_states.shape
        device = current_states.device
        
        # Project states to hidden space
        state_features = self.state_projection(current_states)
        
        # Embed next actions (the actions that will cause the state transitions)
        action_features = self.action_embedding(next_actions)
        
        # Combine state and action features
        combined_features = torch.cat([state_features, action_features], dim=-1)
        combined_features = self.combined_projection(combined_features)
        
        # Create attention mask (all positions visible for now)
        # Could add causal masking if needed for autoregressive prediction
        src_key_padding_mask = None
        
        # Forward through transformer
        hidden_states = self.transformer(
            combined_features,
            src_key_padding_mask=src_key_padding_mask
        )
        
        # Predict next states (conditioned on next actions)
        next_state_pred = self.next_state_head(hidden_states)
        
        # Predict rewards for each type
        reward_predictions = {}
        for reward_type, head in self.reward_heads.items():
            reward_predictions[f'reward_{reward_type}'] = head(hidden_states)
        
        # Predict phases (auxiliary task)
        phase_logits = self.phase_prediction_head(hidden_states)
        
        # Predict values (for RL)
        value_pred = self.value_head(hidden_states)
        
        # Prepare outputs
        outputs = {
            'next_state_pred': next_state_pred,
            'phase_logits': phase_logits,
            'value_pred': value_pred,
            'hidden_states': hidden_states,
            **reward_predictions
        }
        
        if return_hidden_states:
            outputs['all_hidden_states'] = hidden_states
        
        # Calculate losses if targets are provided
        total_loss = 0.0
        
        # Next state prediction loss
        if target_next_states is not None:
            state_loss = F.mse_loss(next_state_pred, target_next_states)
            outputs['state_loss'] = state_loss
            total_loss += 3.0 * state_loss  # High weight for state prediction
        
        # Reward prediction losses
        if target_rewards is not None:
            total_reward_loss = 0.0
            reward_count = 0
            
            for reward_type, target_reward in target_rewards.items():
                if f'reward_{reward_type}' in reward_predictions:
                    pred_reward = reward_predictions[f'reward_{reward_type}']
                    
                    # Handle different target shapes
                    if target_reward.dim() == 2:  # [batch, seq_len]
                        target_reward = target_reward.unsqueeze(-1)  # [batch, seq_len, 1]
                    
                    r_loss = F.mse_loss(pred_reward, target_reward)
                    outputs[f'reward_{reward_type}_loss'] = r_loss
                    total_reward_loss += r_loss
                    reward_count += 1
            
            if reward_count > 0:
                avg_reward_loss = total_reward_loss / reward_count
                outputs['total_reward_loss'] = avg_reward_loss
                total_loss += 2.0 * avg_reward_loss  # Weight reward prediction
        
        # Phase prediction loss (auxiliary)
        if target_phases is not None:
            phase_loss = F.cross_entropy(
                phase_logits.view(-1, self.num_phase_classes),
                target_phases.view(-1),
                ignore_index=-1
            )
            outputs['phase_loss'] = phase_loss
            total_loss += 0.5 * phase_loss  # Lower weight for auxiliary task
        
        outputs['total_loss'] = total_loss
        return outputs
    
    def simulate_step(self, 
                     current_state: torch.Tensor, 
                     next_action: torch.Tensor,  # FIXED: Renamed to clarify this is the next action
                     return_hidden: bool = False) -> Tuple[torch.Tensor, Dict[str, float], Optional[torch.Tensor]]:
        """
        Single step simulation for RL environment.
        
        Args:
            current_state: [batch_size, embedding_dim] or [embedding_dim]
            next_action: [batch_size, num_action_classes] or [num_action_classes] - Action to take
            return_hidden: Whether to return hidden states
            
        Returns:
            next_state: [batch_size, embedding_dim]
            rewards: Dict of scalar rewards
            hidden_state: Optional hidden state
        """
        
        self.eval()
        with torch.no_grad():
            # Ensure correct shapes
            if current_state.dim() == 1:
                current_state = current_state.unsqueeze(0)  # [1, embedding_dim]
            if next_action.dim() == 1:
                next_action = next_action.unsqueeze(0)  # [1, num_action_classes]
            
            batch_size = current_state.size(0)
            
            # Add sequence dimension
            state_seq = current_state.unsqueeze(1)  # [batch_size, 1, embedding_dim]
            action_seq = next_action.unsqueeze(1)  # [batch_size, 1, num_action_classes]
            
            # Forward simulation
            outputs = self.forward(
                current_states=state_seq,
                next_actions=action_seq,  # FIXED: Use next_actions parameter
                return_hidden_states=return_hidden
            )
            
            # Extract next state prediction
            next_state = outputs['next_state_pred'][:, -1, :]  # [batch_size, embedding_dim]
            
            # Extract reward predictions
            rewards = {}
            for key, value in outputs.items():
                if key.startswith('reward_'):
                    reward_type = key.replace('reward_', '')
                    # Take last timestep and convert to scalar per batch item
                    reward_values = value[:, -1, 0]  # [batch_size]
                    
                    if batch_size == 1:
                        rewards[reward_type] = float(reward_values[0])
                    else:
                        rewards[reward_type] = reward_values.cpu().numpy()
            
            # Extract value prediction
            if 'value_pred' in outputs:
                value = outputs['value_pred'][:, -1, 0]  # [batch_size]
                if batch_size == 1:
                    rewards['value'] = float(value[0])
                else:
                    rewards['value'] = value.cpu().numpy()
            
            hidden_state = None
            if return_hidden:
                hidden_state = outputs['hidden_states'][:, -1, :]  # [batch_size, hidden_dim]
            
            return next_state, rewards, hidden_state
    
    def simulate_trajectory(self,
                           initial_state: torch.Tensor,
                           action_sequence: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Simulate a complete trajectory given initial state and action sequence.
        
        Args:
            initial_state: [embedding_dim]
            action_sequence: [seq_len, num_action_classes]
            
        Returns:
            Dictionary with simulated trajectory
        """
        
        self.eval()
        seq_len = action_sequence.size(0)
        
        # Prepare sequences
        states = [initial_state]
        rewards = {reward_type: [] for reward_type in self.reward_heads.keys()}
        rewards['value'] = []
        
        current_state = initial_state.unsqueeze(0)  # [1, embedding_dim]
        
        with torch.no_grad():
            for t in range(seq_len):
                action = action_sequence[t].unsqueeze(0)  # [1, num_action_classes]
                
                # Simulate one step
                next_state, step_rewards, _ = self.simulate_step(current_state, action)
                
                # Store results
                states.append(next_state.squeeze(0))
                for reward_type, reward_value in step_rewards.items():
                    if reward_type in rewards:
                        rewards[reward_type].append(reward_value)
                
                # Update current state
                current_state = next_state
        
        # Convert to tensors
        result = {
            'states': torch.stack(states[1:], dim=0),  # [seq_len, embedding_dim]
            'all_states': torch.stack(states, dim=0),  # [seq_len+1, embedding_dim]
        }
        
        for reward_type, reward_list in rewards.items():
            if reward_list:
                result[f'rewards_{reward_type}'] = torch.tensor(reward_list)
        
        return result
    
    def save_model(self, path: str):
        """Save the model with configuration."""
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': {
                'hidden_dim': self.hidden_dim,
                'embedding_dim': self.embedding_dim,
                'action_embedding_dim': self.action_embedding_dim,
                'n_layer': len(self.transformer.layers),
                'num_action_classes': self.num_action_classes,
                'num_phase_classes': self.num_phase_classes,
                'dropout': self.dropout,
                'max_sequence_length': self.max_sequence_length
            },
            'reward_types': list(self.reward_heads.keys()),
            'model_type': 'ConditionalWorldModel'
        }, path)
        
        print(f"✅ Conditional World Model saved to: {path}")
    
    @classmethod
    def load_model(cls, path: str, device: Optional[torch.device] = None):
        """Load model from saved checkpoint."""
        
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint['config']
        
        # Create model with saved configuration
        model = cls(**config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        print(f"✅ Conditional World Model loaded from: {path}")
        return model


# Example usage and testing
if __name__ == "__main__":
    print("🌍 CONDITIONAL WORLD MODEL")
    print("=" * 60)
    
    # Test model creation
    model = ConditionalWorldModel(
        hidden_dim=512,
        embedding_dim=1024,
        action_embedding_dim=128,
        n_layer=4,
        num_action_classes=100,
        dropout=0.1
    )
    
    # Test forward pass
    batch_size, seq_len, embedding_dim = 2, 8, 1024
    num_actions = 100
    
    current_states = torch.randn(batch_size, seq_len, embedding_dim)
    actions = torch.randint(0, 2, (batch_size, seq_len, num_actions)).float()
    target_next_states = torch.randn(batch_size, seq_len, embedding_dim)
    
    # Create target rewards
    target_rewards = {
        'phase_progression': torch.randn(batch_size, seq_len, 1),
        'safety': torch.randn(batch_size, seq_len, 1),
        'efficiency': torch.randn(batch_size, seq_len, 1)
    }
    
    print(f"\n🧪 Testing forward pass...")
    print(f"State shape: {current_states.shape}")
    print(f"Action shape: {actions.shape}")
    
    outputs = model(
        current_states=current_states,
        actions=actions,
        target_next_states=target_next_states,
        target_rewards=target_rewards
    )
    
    print(f"✅ Forward pass successful!")
    print(f"Next state pred shape: {outputs['next_state_pred'].shape}")
    print(f"Total loss: {outputs['total_loss'].item():.4f}")
    print(f"Reward predictions: {len([k for k in outputs.keys() if k.startswith('reward_')])}")
    
    # Test single step simulation
    print(f"\n🧪 Testing single step simulation...")
    single_state = torch.randn(embedding_dim)
    single_action = torch.randint(0, 2, (num_actions,)).float()
    
    next_state, rewards, hidden = model.simulate_step(
        single_state, single_action, return_hidden=True
    )
    
    print(f"✅ Single step simulation successful!")
    print(f"Input state shape: {single_state.shape}")
    print(f"Output state shape: {next_state.shape}")
    print(f"Rewards: {list(rewards.keys())}")
    print(f"Hidden state shape: {hidden.shape if hidden is not None else None}")
    
    # Test trajectory simulation
    print(f"\n🧪 Testing trajectory simulation...")
    initial_state = torch.randn(embedding_dim)
    action_sequence = torch.randint(0, 2, (10, num_actions)).float()
    
    trajectory = model.simulate_trajectory(initial_state, action_sequence)
    
    print(f"✅ Trajectory simulation successful!")
    print(f"Initial state shape: {initial_state.shape}")
    print(f"Action sequence shape: {action_sequence.shape}")
    print(f"Simulated states shape: {trajectory['states'].shape}")
    print(f"Trajectory keys: {list(trajectory.keys())}")
    
    print(f"\n🎯 KEY FEATURES:")
    print(f"✅ Action-conditioned forward simulation")
    print(f"✅ Multi-type reward prediction")
    print(f"✅ Single step and trajectory simulation")
    print(f"✅ Value function approximation")
    print(f"✅ Ready for RL environment integration")
    print(f"✅ Perfect for Method 2 in three-way comparison")
