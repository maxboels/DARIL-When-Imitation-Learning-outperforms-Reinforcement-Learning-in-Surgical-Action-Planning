import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import transformers
from transformers import GPT2Config, GPT2LMHeadModel
from typing import Dict, Any, Optional, List, Tuple, Union
import math

class DualWorldModel(nn.Module):
    """
    Enhanced World Model that supports both:
    1. Autoregressive action prediction from predicted latent states
    2. RL-based state prediction conditioned on actions
    
    This model can seamlessly switch between two modes:
    - SUPERVISED: For autoregressive action prediction using expert demonstrations
    - RL: For state/reward prediction conditioned on planned actions
    """
    
    def __init__(self, 
                 hidden_dim: int, 
                 embedding_dim: int, 
                 action_embedding_dim: int, 
                 n_layer: int,
                 num_action_classes: int = 100,
                 num_phase_classes: int = 7,
                 max_length: int = 1024,
                 # Mode-specific parameters
                 enable_autoregressive_prediction: bool = True,
                 enable_rl_prediction: bool = True,
                 enable_reward_prediction: bool = True,
                 # Loss weights
                 loss_weights: Dict[str, float] = None,
                 # Dropout for regularization
                 dropout: float = 0.1,
                 ) -> None:
        """
        Initialize the dual-purpose world model.
        
        Args:
            hidden_dim: Hidden dimension of GPT-2
            embedding_dim: Dimension of input frame embeddings
            action_embedding_dim: Dimension of action embeddings
            n_layer: Number of GPT-2 layers
            num_action_classes: Number of action classes
            num_phase_classes: Number of phase classes
            max_length: Maximum sequence length
            enable_autoregressive_prediction: Enable autoregressive action prediction
            enable_rl_prediction: Enable RL state prediction
            enable_reward_prediction: Enable reward prediction
            loss_weights: Dictionary of loss weights
            dropout: Dropout rate
        """
        super(DualWorldModel, self).__init__()
        
        # Store configuration
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.action_embedding_dim = action_embedding_dim
        self.n_layer = n_layer
        self.num_action_classes = num_action_classes
        self.num_phase_classes = num_phase_classes
        self.max_length = max_length
        self.dropout = dropout
        
        # Mode configuration
        self.enable_autoregressive_prediction = enable_autoregressive_prediction
        self.enable_rl_prediction = enable_rl_prediction
        self.enable_reward_prediction = enable_reward_prediction
        
        # Loss weights
        self.loss_weights = loss_weights if loss_weights else {
            'state': 1.0,
            'action': 1.0,
            'reward': 0.5,
            'phase': 0.5
        }
        
        # Initialize model components
        self._init_gpt2_backbone()
        self._init_embedding_layers()
        self._init_prediction_heads()
        self._init_weights()
    
    def _init_gpt2_backbone(self):
        """Initialize GPT-2 backbone for causal modeling."""
        self.gpt2_config = GPT2Config.from_pretrained('gpt2')
        self.gpt2_config.hidden_size = self.hidden_dim
        self.gpt2_config.num_hidden_layers = self.n_layer
        self.gpt2_config.num_attention_heads = max(1, self.hidden_dim // 64)
        self.gpt2_config.add_cross_attention = False
        self.gpt2_config.output_hidden_states = True
        self.gpt2_config.n_layer = self.n_layer
        self.gpt2_config.n_embd = self.hidden_dim
        self.gpt2_config.vocab_size = 1  # Not using traditional vocab
        self.gpt2_config.attn_pdrop = self.dropout
        self.gpt2_config.resid_pdrop = self.dropout
        
        # Initialize GPT-2 model
        self.gpt2 = GPT2LMHeadModel(self.gpt2_config)
        # Remove the default language modeling head
        self.gpt2.lm_head = nn.Identity()
    
    def _init_embedding_layers(self):
        """Initialize embedding and projection layers."""
        # State embedding projection
        self.state_projection = nn.Linear(self.embedding_dim, self.hidden_dim)
        
        # Action embedding
        self.action_embedding = nn.Linear(self.num_action_classes, self.action_embedding_dim)
        
        # Combined input projection (state + action)
        self.combined_projection = nn.Linear(self.hidden_dim + self.action_embedding_dim, self.hidden_dim)
        
        # Layer normalization
        self.input_ln = nn.LayerNorm(self.hidden_dim)
        
        # Dropout
        self.input_dropout = nn.Dropout(self.dropout)
    
    def _init_prediction_heads(self):
        """Initialize prediction heads for different tasks."""
        self.heads = nn.ModuleDict()
        
        # State prediction head (for both modes)
        self.heads['state'] = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.embedding_dim)
        )
        
        # Action prediction head (for autoregressive mode)
        if self.enable_autoregressive_prediction:
            self.heads['action'] = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim, self.num_action_classes)
            )
        
        # Phase prediction head
        self.heads['phase'] = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, self.num_phase_classes)
        )
        
        # Reward prediction heads (for RL mode)
        if self.enable_reward_prediction:
            reward_heads = [
                'phase_completion',
                'phase_initiation', 
                'phase_progression',
                'action_probability',
                'risk_penalty',
                'global_progression'
            ]
            
            for reward_type in reward_heads:
                self.heads[f'reward_{reward_type}'] = nn.Sequential(
                    nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.hidden_dim // 2, 1)
                )
    
    def _init_weights(self):
        """Initialize weights using Xavier/Kaiming initialization."""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if 'heads' in name:
                    # Use Xavier initialization for prediction heads
                    nn.init.xavier_uniform_(module.weight)
                else:
                    # Use small normal initialization for other layers
                    nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal attention mask for autoregressive generation."""
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        return mask.to(device)
    
    def forward(self, 
                current_states: torch.Tensor,
                actions: Optional[torch.Tensor] = None,
                next_states: Optional[torch.Tensor] = None,
                next_actions: Optional[torch.Tensor] = None,
                next_phases: Optional[torch.Tensor] = None,
                next_rewards: Optional[Dict[str, torch.Tensor]] = None,
                mode: str = 'supervised',
                attention_mask: Optional[torch.Tensor] = None,
                return_hidden_states: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass that supports both supervised and RL modes.
        
        Args:
            current_states: Current frame embeddings [batch_size, seq_len, embedding_dim]
            actions: Action sequence [batch_size, seq_len, num_action_classes]
            next_states: Target next states (for supervised mode)
            next_actions: Target next actions (for supervised mode)
            next_phases: Target next phases
            next_rewards: Target rewards (for RL mode)
            mode: 'supervised' for action prediction, 'rl' for state prediction
            attention_mask: Custom attention mask
            return_hidden_states: Whether to return hidden states
            
        Returns:
            Dictionary containing predictions and losses
        """
        batch_size, seq_len, _ = current_states.shape
        device = current_states.device
        
        # Project states to hidden dimension
        state_embeds = self.state_projection(current_states)
        
        # Handle actions
        if actions is None:
            # Create zero actions if not provided
            actions = torch.zeros(batch_size, seq_len, self.num_action_classes, device=device)
        
        # Create action embeddings
        action_embeds = self.action_embedding(actions)
        
        # Combine state and action embeddings
        combined_embeds = torch.cat([state_embeds, action_embeds], dim=-1)
        combined_embeds = self.combined_projection(combined_embeds)
        combined_embeds = self.input_ln(combined_embeds)
        combined_embeds = self.input_dropout(combined_embeds)
        
        # Create attention mask
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_len, device=device)
        
        # Create causal mask for autoregressive behavior
        causal_mask = self.create_causal_mask(seq_len, device)
        
        # Forward pass through GPT-2
        gpt2_outputs = self.gpt2(
            inputs_embeds=combined_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        # Get hidden states
        hidden_states = gpt2_outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_dim]
        
        # Initialize outputs
        outputs = {
            'hidden_states': hidden_states,
            'attention_mask': attention_mask
        }
        
        # Generate predictions
        state_pred = self.heads['state'](hidden_states)
        outputs['state_pred'] = state_pred
        
        if self.enable_autoregressive_prediction and 'action' in self.heads:
            action_pred = self.heads['action'](hidden_states)
            outputs['action_pred'] = action_pred
        
        phase_pred = self.heads['phase'](hidden_states)
        outputs['phase_pred'] = phase_pred
        
        # Reward predictions for RL mode
        if self.enable_reward_prediction and mode == 'rl':
            for head_name, head in self.heads.items():
                if head_name.startswith('reward_'):
                    reward_type = head_name.replace('reward_', '')
                    outputs[f'reward_{reward_type}'] = head(hidden_states)
        
        # Calculate losses if targets are provided
        total_loss = 0.0
        
        # State prediction loss
        if next_states is not None:
            state_loss = F.mse_loss(state_pred, next_states)
            outputs['state_loss'] = state_loss
            total_loss += self.loss_weights['state'] * state_loss
        
        # Action prediction loss (for supervised mode)
        if next_actions is not None and 'action_pred' in outputs:
            action_loss = F.binary_cross_entropy_with_logits(
                outputs['action_pred'], next_actions
            )
            outputs['action_loss'] = action_loss
            total_loss += self.loss_weights['action'] * action_loss
        
        # Phase prediction loss
        if next_phases is not None:
            # Reshape for cross entropy loss
            phase_pred_flat = phase_pred.view(-1, self.num_phase_classes)
            phase_targets_flat = next_phases.view(-1)
            phase_loss = F.cross_entropy(phase_pred_flat, phase_targets_flat, ignore_index=-1)
            outputs['phase_loss'] = phase_loss
            total_loss += self.loss_weights['phase'] * phase_loss
        
        # Reward prediction losses (for RL mode)
        if next_rewards is not None and mode == 'rl':
            reward_loss = 0.0
            for reward_type, target_reward in next_rewards.items():
                if f'reward_{reward_type}' in outputs:
                    pred_reward = outputs[f'reward_{reward_type}']
                    r_loss = F.mse_loss(pred_reward.squeeze(-1), target_reward.squeeze(-1))
                    outputs[f'reward_{reward_type}_loss'] = r_loss
                    reward_loss += r_loss
            
            outputs['total_reward_loss'] = reward_loss
            total_loss += self.loss_weights['reward'] * reward_loss
        
        outputs['total_loss'] = total_loss
        
        if return_hidden_states:
            outputs['all_hidden_states'] = gpt2_outputs.hidden_states
        
        return outputs
    
    def autoregressive_action_prediction(self, 
                                       initial_states: torch.Tensor,
                                       horizon: int = 15,
                                       temperature: float = 1.0,
                                       top_k: Optional[int] = None,
                                       top_p: Optional[float] = None) -> Dict[str, torch.Tensor]:
        """
        Autoregressive action prediction from predicted latent states.
        
        Args:
            initial_states: Initial frame embeddings [batch_size, context_len, embedding_dim]
            horizon: Number of steps to predict
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling threshold
            
        Returns:
            Dictionary with predicted actions and states
        """
        self.eval()
        batch_size, context_len, _ = initial_states.shape
        device = initial_states.device
        
        # Initialize sequences
        predicted_states = initial_states.clone()
        predicted_actions = []
        
        with torch.no_grad():
            for step in range(horizon):
                # Get the last few states as context
                context_states = predicted_states[:, -context_len:]
                
                # Forward pass to predict next action and state
                outputs = self.forward(
                    current_states=context_states,
                    mode='supervised'
                )
                
                # Extract predictions for the last timestep
                next_state_pred = outputs['state_pred'][:, -1:, :]  # [batch_size, 1, embedding_dim]
                
                if 'action_pred' in outputs:
                    action_logits = outputs['action_pred'][:, -1, :]  # [batch_size, num_action_classes]
                    
                    # Apply temperature
                    if temperature != 1.0:
                        action_logits = action_logits / temperature
                    
                    # Apply top-k sampling
                    if top_k is not None:
                        top_k = min(top_k, action_logits.size(-1))
                        top_k_logits, top_k_indices = torch.topk(action_logits, top_k)
                        action_logits = torch.full_like(action_logits, float('-inf'))
                        action_logits.scatter_(-1, top_k_indices, top_k_logits)
                    
                    # Apply nucleus sampling
                    if top_p is not None:
                        sorted_logits, sorted_indices = torch.sort(action_logits, descending=True)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        indices_to_remove = sorted_indices.gather(-1, sorted_indices_to_remove.long())
                        action_logits.scatter_(-1, indices_to_remove, float('-inf'))
                    
                    # Sample action (using Gumbel-Softmax for differentiability)
                    action_probs = torch.sigmoid(action_logits)
                    sampled_action = torch.bernoulli(action_probs)
                    
                    predicted_actions.append(sampled_action)
                else:
                    # Fallback to random actions
                    sampled_action = torch.bernoulli(torch.ones(batch_size, self.num_action_classes, device=device) * 0.5)
                    predicted_actions.append(sampled_action)
                
                # Append predicted state to sequence
                predicted_states = torch.cat([predicted_states, next_state_pred], dim=1)
        
        return {
            'predicted_states': predicted_states[:, context_len:],  # Exclude initial context
            'predicted_actions': torch.stack(predicted_actions, dim=1),
            'full_state_sequence': predicted_states
        }
    
    def rl_state_prediction(self, 
                           current_states: torch.Tensor,
                           planned_actions: torch.Tensor,
                           return_rewards: bool = True) -> Dict[str, torch.Tensor]:
        """
        Predict next states and rewards given current states and planned actions.
        This is used by RL algorithms to simulate environment transitions.
        
        Args:
            current_states: Current frame embeddings [batch_size, seq_len, embedding_dim]
            planned_actions: Planned actions [batch_size, seq_len, num_action_classes]
            return_rewards: Whether to return reward predictions
            
        Returns:
            Dictionary with predicted next states and rewards
        """
        self.eval()
        
        with torch.no_grad():
            outputs = self.forward(
                current_states=current_states,
                actions=planned_actions,
                mode='rl'
            )
            
            # Extract predictions
            result = {
                'next_states': outputs['state_pred'],
                'hidden_states': outputs['hidden_states']
            }
            
            # Add reward predictions if requested
            if return_rewards and self.enable_reward_prediction:
                rewards = {}
                for key, value in outputs.items():
                    if key.startswith('reward_') and not key.endswith('_loss'):
                        reward_type = key.replace('reward_', '')
                        rewards[reward_type] = value
                result['rewards'] = rewards
            
            # Add phase predictions
            if 'phase_pred' in outputs:
                result['phases'] = F.softmax(outputs['phase_pred'], dim=-1)
            
            return result
    
    def save_model(self, path: str):
        """Save the model state."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': {
                'hidden_dim': self.hidden_dim,
                'embedding_dim': self.embedding_dim,
                'action_embedding_dim': self.action_embedding_dim,
                'n_layer': self.n_layer,
                'num_action_classes': self.num_action_classes,
                'num_phase_classes': self.num_phase_classes,
                'max_length': self.max_length,
                'enable_autoregressive_prediction': self.enable_autoregressive_prediction,
                'enable_rl_prediction': self.enable_rl_prediction,
                'enable_reward_prediction': self.enable_reward_prediction,
                'loss_weights': self.loss_weights,
                'dropout': self.dropout
            }
        }, path)
    
    @classmethod
    def load_model(cls, path: str, device: Optional[torch.device] = None):
        """Load the model from a saved state."""
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint['config']
        
        model = cls(**config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        return model