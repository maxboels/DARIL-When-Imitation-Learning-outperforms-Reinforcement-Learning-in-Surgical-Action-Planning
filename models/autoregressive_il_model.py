#!/usr/bin/env python3
"""
Autoregressive Imitation Learning Model for Method 1
Pure causal frame generation → action prediction (no action conditioning)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import GPT2Config, GPT2LMHeadModel
from typing import Dict, Any, Optional, List, Tuple
import os

class BiLSTMActionRecognizer(nn.Module):
    def __init__(self, input_dim: int, lstm_hidden_dim: int, num_classes: int, dropout: float = 0.1):
        super().__init__()

        self.bi_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(2 * lstm_hidden_dim, lstm_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden_dim, num_classes)
        )

    def forward(self, x):
        """
        x: Tensor of shape (B, T, D) → pre-GPT2 visual features
        """
        lstm_out, _ = self.bi_lstm(x)       # (B, T, 2 * hidden_dim)
        logits = self.classifier(lstm_out)  # (B, T, num_classes)
        return logits



class AutoregressiveILModel(nn.Module):
    """
    Method 1: Pure Autoregressive Imitation Learning
    
    Architecture:
    1. Frame Embeddings → GPT-2 (Causal) → Hidden States
    2. Hidden States → Next Frame Prediction
    3. Hidden States → Action Prediction (from generated frames)
    
    Key: NO action conditioning during frame generation (pure autoregressive)
    """
    
    def __init__(self, 
                 hidden_dim: int = 768,
                 embedding_dim: int = 1024,
                 n_layer: int = 6,
                 num_action_classes: int = 100,
                 num_phase_classes: int = 7,
                 max_length: int = 1024,
                 dropout: float = 0.1):
        super().__init__()
        
        print("🎓 Initializing Autoregressive IL Model...")
        
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.num_action_classes = num_action_classes
        self.num_phase_classes = num_phase_classes
        self.max_length = max_length
        self.dropout = dropout

        # Auxiliary BiLSTM recogniser for Swin features (pre-GPT2)
        self.bilstm_action_recogniser = BiLSTMActionRecognizer(
            input_dim=embedding_dim,        # Input is pre-GPT2 visual features
            lstm_hidden_dim=hidden_dim // 4,  # Small size since Swin features are strong
            num_classes=num_action_classes,
            dropout=dropout
        )

        
        # GPT-2 configuration for causal modeling
        self.gpt2_config = GPT2Config(
            hidden_size=hidden_dim,
            num_hidden_layers=n_layer,
            num_attention_heads=max(1, hidden_dim // 64),
            intermediate_size=hidden_dim * 4,
            max_position_embeddings=max_length,
            vocab_size=1,  # Not using traditional vocabulary
            n_positions=max_length,
            n_embd=hidden_dim,
            n_layer=n_layer,
            attn_pdrop=dropout,
            resid_pdrop=dropout,
            output_hidden_states=True,
            use_cache=False
        )
        
        # Initialize GPT-2 backbone
        self.gpt2 = GPT2LMHeadModel(self.gpt2_config)
        self.gpt2.lm_head = nn.Identity()  # Remove default language modeling head
        
        # Input projection: frame embeddings → hidden space
        self.frame_projection = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Next frame prediction head (autoregressive state generation)
        self.next_frame_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, embedding_dim)
        )
        
        # Action prediction head (from generated frames)
        self.action_prediction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_action_classes)
        )
        
        # Phase prediction head (surgical phase understanding)
        self.phase_prediction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_phase_classes)
        )

        # Action recognition layer
        self.action_recognition_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_action_classes)
        )
        
        # Initialize weights
        self._init_weights()
        
        print(f"✅ Autoregressive IL Model initialized")
        print(f"   Architecture: Frame → Causal Generation → Action")
        print(f"   Hidden dim: {hidden_dim}, Embedding dim: {embedding_dim}")
        print(f"   Layers: {n_layer}, Actions: {num_action_classes}")
        print(f"   Key: NO action conditioning during generation")
    
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
                frame_embeddings: torch.Tensor,
                target_next_frames: Optional[torch.Tensor] = None,
                target_current_actions: Optional[torch.Tensor] = None,
                target_actions: Optional[torch.Tensor] = None,
                target_phases: Optional[torch.Tensor] = None,
                return_hidden_states: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass for autoregressive IL training.
        
        Args:
            frame_embeddings: [batch_size, seq_len, embedding_dim]
            target_next_frames: [batch_size, seq_len, embedding_dim] (for training)
            target_actions: [batch_size, seq_len, num_action_classes] (for training)
            target_phases: [batch_size, seq_len] (for training)
            return_hidden_states: Whether to return all hidden states
            
        Returns:
            Dictionary containing predictions and losses
        """
        
        batch_size, seq_len, _ = frame_embeddings.shape
        device = frame_embeddings.device

        # Action recognition logits from pre-GPT2 features        
        action_rec_logits = self.bilstm_action_recogniser(frame_embeddings)  # [B, T, num_classes]
        # action_rec_logits = self.action_recognition_head(bilstm_action_logits) # [B, T, num_classes]

        # Project frame embeddings to hidden space
        hidden_inputs = self.frame_projection(frame_embeddings)

        # Create attention mask (all positions visible)
        attention_mask = torch.ones(batch_size, seq_len, device=device)
        
        # Forward through GPT-2 (causal autoregressive modeling)
        gpt2_outputs = self.gpt2(
            inputs_embeds=hidden_inputs,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        # Get hidden states from last layer
        hidden_states = gpt2_outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_dim]
        
        # Predict next frames (autoregressive state generation)
        next_frame_pred = self.next_frame_head(hidden_states)
        
        # Predict actions from hidden states (learned from generated frames)
        action_logits = self.action_prediction_head(hidden_states)
        
        # Predict phases
        phase_logits = self.phase_prediction_head(hidden_states)
        
        # Prepare outputs
        outputs = {
            'action_rec_probs': torch.sigmoid(action_rec_logits),  # For compatibility
            'next_frame_pred': next_frame_pred,
            'action_logits': action_logits,
            'action_pred': torch.sigmoid(action_logits),  # For compatibility
            'phase_logits': phase_logits,
            'hidden_states': hidden_states
        }
        
        if return_hidden_states:
            outputs['all_hidden_states'] = gpt2_outputs.hidden_states
        
        # Calculate losses if targets are provided
        total_loss = 0.0

        # Action recognition loss
        if target_current_actions is not None:
            action_rec_loss = F.binary_cross_entropy_with_logits(
                action_rec_logits, target_current_actions
            )
            outputs['action_rec_loss'] = action_rec_loss
            total_loss += 1.5 * action_rec_loss
        
        # Next frame prediction loss (autoregressive)
        if target_next_frames is not None:
            frame_loss = F.mse_loss(next_frame_pred, target_next_frames)
            outputs['frame_loss'] = frame_loss
            total_loss += frame_loss # might overfit on this loss
        
        # Action prediction loss
        if target_actions is not None:
            action_loss = F.binary_cross_entropy_with_logits(
                action_logits, target_actions
            )
            outputs['action_loss'] = action_loss
            total_loss += 2.0 * action_loss  # Weight action loss higher
        
        # Phase prediction loss
        if target_phases is not None:
            phase_loss = F.cross_entropy(
                phase_logits.view(-1, self.num_phase_classes),
                target_phases.view(-1),
                ignore_index=-1
            )
            outputs['phase_loss'] = phase_loss
            total_loss += 0.5 * phase_loss  # Lower weight for phase
        
        outputs['total_loss'] = total_loss
        return outputs
    
    def generate_sequence(self, 
                         initial_frames: torch.Tensor,
                         horizon: int = 15,
                         temperature: float = 1.0,
                         top_p: Optional[float] = None) -> Dict[str, torch.Tensor]:
        """
        Autoregressive sequence generation: frames → next_frames → actions
        
        Args:
            initial_frames: [batch_size, context_len, embedding_dim]
            horizon: Number of steps to generate
            temperature: Sampling temperature for actions
            top_p: Nucleus sampling threshold
            
        Returns:
            Dictionary with generated frames and predicted actions
        """
        
        self.eval()
        batch_size, context_len, _ = initial_frames.shape
        device = initial_frames.device
        
        # Start with initial frames
        generated_frames = initial_frames.clone()
        predicted_actions = []
        predicted_phases = []
                
        with torch.no_grad():
            for step in range(horizon):
                # Get current context (limit to reasonable length)
                max_context = min(self.max_length - 1, generated_frames.size(1))
                current_context = generated_frames[:, -max_context:]
                
                # Forward pass (no action conditioning!)
                outputs = self.forward(current_context)
                
                # Get predictions for the last timestep (causal next token prediction)
                next_frame = outputs['next_frame_pred'][:, -1:, :]  # [batch_size, 1, embedding_dim]
                action_logits = outputs['action_logits'][:, -1, :]  # [batch_size, num_action_classes]
                phase_logits = outputs['phase_logits'][:, -1, :]  # [batch_size, num_phase_classes]
                
                # Sample actions with temperature and nucleus sampling
                if temperature != 1.0:
                    action_logits = action_logits / temperature
                
                if top_p is not None:
                    # Nucleus sampling for actions
                    action_probs = torch.sigmoid(action_logits)
                    sorted_probs, sorted_indices = torch.sort(action_probs, descending=True, dim=-1)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    
                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = 0
                    
                    indices_to_remove = sorted_indices.gather(-1, sorted_indices_to_remove.long())
                    action_logits.scatter_(-1, indices_to_remove, float('-inf'))
                
                # Sample actions
                action_probs = torch.sigmoid(action_logits)
                sampled_actions = torch.bernoulli(action_probs)
                
                # Get phase predictions
                phase_probs = F.softmax(phase_logits, dim=-1)
                
                # Store predictions
                predicted_actions.append(sampled_actions)
                predicted_phases.append(phase_probs)
                
                # Append next frame for continued generation
                generated_frames = torch.cat([generated_frames, next_frame], dim=1)
                
        result = {
            'generated_frames': generated_frames[:, context_len:],  # Exclude initial context
            'predicted_actions': torch.stack(predicted_actions, dim=1),
            'predicted_phases': torch.stack(predicted_phases, dim=1),
            'full_sequence': generated_frames,
            'generation_info': {
                'initial_context_length': context_len,
                'generated_length': horizon,
                'temperature': temperature,
                'top_p': top_p
            }
        }
        
        return result
    
    def predict_next_action(self, frame_sequence: torch.Tensor) -> torch.Tensor:
        """
        Predict next action from frame sequence (for evaluation).
        
        Args:
            frame_sequence: [batch_size, seq_len, embedding_dim]
            
        Returns:
            action_probs: [batch_size, num_action_classes]
        """
        # Check input shape
        batch_size, seq_len, _ = frame_sequence.shape
        if seq_len < 2:
            raise ValueError("Frame sequence must have at least 2 frames for action prediction.")

        self.eval()
        with torch.no_grad():
            outputs = self.forward(frame_sequence)
            action_probs = outputs['action_pred'][:, -1, :]  # Last timestep
            return action_probs
    
    def save_model(self, path: str):
        """Save the model with configuration."""
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': {
                'hidden_dim': self.hidden_dim,
                'embedding_dim': self.embedding_dim,
                'n_layer': self.gpt2_config.num_hidden_layers,
                'num_action_classes': self.num_action_classes,
                'num_phase_classes': self.num_phase_classes,
                'max_length': self.max_length,
                'dropout': self.dropout
            },
            'model_type': 'AutoregressiveILModel'
        }, path)
        
        print(f"✅ Autoregressive IL Model saved to: {path}")
    
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
        
        print(f"✅ Autoregressive IL Model loaded from: {path}")
        return model


# Example usage and testing
if __name__ == "__main__":
    print("🎓 AUTOREGRESSIVE IMITATION LEARNING MODEL")
    