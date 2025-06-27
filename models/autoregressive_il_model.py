#!/usr/bin/env python3
"""
Improved Autoregressive Imitation Learning Model
Dual-path architecture: BiLSTM (recognition) + GPT2 (generation)
Optimized for small datasets with clear temporal alignment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import GPT2Config, GPT2LMHeadModel
from typing import Dict, Any, Optional, List, Tuple
import os

class BiLSTMActionRecognizer(nn.Module):
    """Enhanced BiLSTM for current action recognition with full temporal context"""
    
    def __init__(self, input_dim: int, lstm_hidden_dim: int, num_classes: int, dropout: float = 0.1):
        super().__init__()

        # Enhanced BiLSTM with more layers for better temporal modeling
        self.bi_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=2,  # Increased from 1 for better temporal understanding
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_hidden_dim > 1 else 0.0  # Only apply if multi-layer
        )

        # Enhanced classifier with better regularization
        self.classifier = nn.Sequential(
            nn.LayerNorm(2 * lstm_hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(2 * lstm_hidden_dim, lstm_hidden_dim),
            nn.GELU(),  # Better activation for recognition tasks
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden_dim, num_classes)
        )
        
        # Initialize weights for small dataset training
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stable training on small datasets"""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.zeros_(param.data)
                # Set forget gate bias to 1 for better gradient flow
                if 'bias_ih' in name:
                    n = param.size(0)
                    param.data[n//4:n//2].fill_(1.)

    def forward(self, x):
        """
        x: Tensor of shape (B, T, D) → pre-GPT2 visual features
        Returns: logits for current action recognition at each timestep
        """
        lstm_out, _ = self.bi_lstm(x)       # (B, T, 2 * hidden_dim)
        logits = self.classifier(lstm_out)  # (B, T, num_classes)
        return logits


class AutoregressiveILModel(nn.Module):
    """
    Improved Dual-Path Autoregressive Imitation Learning
    
    Architecture:
    Path 1: Frame Embeddings → BiLSTM (bidirectional) → Current Action Recognition
    Path 2: Frame Embeddings → GPT-2 (causal) → Next Action Generation + Frame Prediction
    
    Key Improvements:
    - Clear temporal alignment (current vs next action prediction)
    - Better regularization for small datasets
    - Enhanced knowledge transfer between paths
    - Curriculum learning support
    """
    
    def __init__(self, 
                 hidden_dim: int = 768,
                 embedding_dim: int = 1024,
                 n_layer: int = 6,
                 num_action_classes: int = 100,
                 num_phase_classes: int = 7,
                 max_length: int = 1024,
                 dropout: float = 0.15,
                 transfer_context: bool = False):  # Increased for small dataset
        super().__init__()
        
        print("🎓 Initializing Improved Autoregressive IL Model...")
        
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.num_action_classes = num_action_classes
        self.num_phase_classes = num_phase_classes
        self.max_length = max_length
        self.dropout = dropout

        # PATH 1: Recognition Path (BiLSTM - Bidirectional for current actions)
        self.bilstm_action_recogniser = BiLSTMActionRecognizer(
            input_dim=embedding_dim,
            lstm_hidden_dim=hidden_dim // 3,  # Optimized size
            num_classes=num_action_classes,
            dropout=dropout
        )
        
        # PATH 2: Generation Path (GPT2 - Causal for next actions)
        self.gpt2_config = GPT2Config(
            hidden_size=hidden_dim,
            num_hidden_layers=n_layer,
            num_attention_heads=max(1, hidden_dim // 64),
            intermediate_size=hidden_dim * 2,  # Reduced for small dataset
            max_position_embeddings=max_length,
            vocab_size=1,
            n_positions=max_length,
            n_embd=hidden_dim,
            n_layer=n_layer,
            attn_pdrop=dropout,
            resid_pdrop=dropout,
            output_hidden_states=True,
            use_cache=False
        )
        
        self.gpt2 = GPT2LMHeadModel(self.gpt2_config)
        self.gpt2.lm_head = nn.Identity()
        
        # Enhanced input projection with residual connection
        self.frame_projection = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Next frame prediction head (autoregressive state generation)
        self.next_frame_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, embedding_dim)
        )
        
        # Next action prediction head (causal - predicts t+1 from t)
        self.next_action_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_action_classes)
        )
        
        # Phase prediction head
        self.phase_prediction_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_phase_classes)
        )

        # Cross-path knowledge transfer (optional enhancement)
        if transfer_context:
            self.knowledge_transfer = nn.Sequential(
                nn.Linear(2 * (hidden_dim // 3), hidden_dim // 4),  # BiLSTM features to GPT2
                nn.Tanh(),
                nn.Dropout(dropout)
            )
        
        # Initialize weights
        self._init_weights()
        
        print(f"✅ Improved Autoregressive IL Model initialized")
        print(f"   Architecture: Dual-Path (BiLSTM + GPT2)")
        print(f"   Recognition: BiLSTM (current actions)")
        print(f"   Generation: GPT2 (next actions + frames)")
        print(f"   Hidden dim: {hidden_dim}, Embedding dim: {embedding_dim}")
        print(f"   Optimized for small datasets (40 videos)")
    
    def _init_weights(self):
        """Initialize model weights for stable small dataset training"""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if 'head' in name.lower() or 'classifier' in name.lower():
                    # Xavier for prediction heads
                    nn.init.xavier_uniform_(module.weight)
                else:
                    # Conservative initialization for small datasets
                    nn.init.normal_(module.weight, std=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, 
                frame_embeddings: torch.Tensor,
                target_next_frames: Optional[torch.Tensor] = None,
                target_current_actions: Optional[torch.Tensor] = None,
                target_actions: Optional[torch.Tensor] = None,  # For backward compatibility
                target_phases: Optional[torch.Tensor] = None,
                return_hidden_states: bool = False,
                epoch: int = 0) -> Dict[str, torch.Tensor]:
        """
        Forward pass with improved dual-path architecture.
        
        Args:
            frame_embeddings: [batch_size, seq_len, embedding_dim]
            target_next_frames: [batch_size, seq_len, embedding_dim] - for next frame prediction
            target_current_actions: [batch_size, seq_len, num_action_classes] - current actions
            target_actions: [batch_size, seq_len, num_action_classes] - for backward compatibility
            target_phases: [batch_size, seq_len] - phase labels
            return_hidden_states: Whether to return all hidden states
            epoch: Current training epoch (for curriculum learning)
            
        Returns:
            Dictionary containing predictions and losses with same interface as before
        """
        
        batch_size, seq_len, _ = frame_embeddings.shape
        device = frame_embeddings.device

        # PATH 1: Recognition Path - Current Action Recognition (BiLSTM)
        action_rec_logits = self.bilstm_action_recogniser(frame_embeddings)  # [B, T, num_classes]
        
        # PATH 2: Generation Path - Causal Modeling (GPT2)
        projected_features = self.frame_projection(frame_embeddings)
        
        # Optional: Add recognition context to generation path
        if hasattr(self, 'knowledge_transfer'):
            lstm_features = self.bilstm_action_recogniser.bi_lstm(frame_embeddings)[0]
            transfer_context = self.knowledge_transfer(lstm_features)
            # Add as residual connection
            projected_features = projected_features + transfer_context
        
        attention_mask = torch.ones(batch_size, seq_len, device=device)
        
        gpt2_outputs = self.gpt2(
            inputs_embeds=projected_features,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        hidden_states = gpt2_outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_dim]
        
        # Next frame prediction (autoregressive)
        next_frame_pred = self.next_frame_head(hidden_states)
        
        # Next action prediction (causal - predict t+1 from t)
        next_action_logits = self.next_action_head(hidden_states)
        
        # Phase prediction
        phase_logits = self.phase_prediction_head(hidden_states)
        
        # Prepare outputs (maintain same interface as original)
        outputs = {
            'action_rec_probs': torch.sigmoid(action_rec_logits),  # Current actions (BiLSTM)
            'next_frame_pred': next_frame_pred,
            'action_logits': next_action_logits,  # Next actions (GPT2) - for compatibility
            'action_pred': torch.sigmoid(next_action_logits),  # For compatibility
            'phase_logits': phase_logits,
            'hidden_states': hidden_states
        }
        
        if return_hidden_states:
            outputs['all_hidden_states'] = gpt2_outputs.hidden_states
        
        # Calculate losses with improved strategy for small datasets
        total_loss = 0.0
        
        # Use target_actions for backward compatibility if target_current_actions not provided
        current_action_targets = target_current_actions if target_current_actions is not None else target_actions

        # Loss 1: Current Action Recognition (BiLSTM)
        if current_action_targets is not None:
            action_rec_loss = F.binary_cross_entropy_with_logits(
                action_rec_logits, current_action_targets
            )
            outputs['action_rec_loss'] = action_rec_loss
            # Higher weight for recognition (easier task, provides strong gradients)
            total_loss += 2.0 * action_rec_loss
        
        # Loss 2: Next Frame Prediction (autoregressive)
        if target_next_frames is not None:
            frame_loss = F.mse_loss(next_frame_pred, target_next_frames)
            outputs['frame_loss'] = frame_loss
            # Moderate weight - helps with representation learning
            total_loss += 0.5 * frame_loss
        
        # Loss 3: Next Action Prediction (causal)
        # Align temporal targets: hidden[t] predicts action[t+1]
        if current_action_targets is not None and seq_len > 1:
            # Shift targets: predict action[t+1] from hidden[t]
            next_action_targets = current_action_targets[:, 1:]  # Actions at t+1
            next_action_preds = next_action_logits[:, :-1]       # Predictions from t
            
            next_action_loss = F.binary_cross_entropy_with_logits(
                next_action_preds, next_action_targets
            )
            outputs['action_loss'] = next_action_loss  # For compatibility
            
            # Curriculum learning: gradually increase next action loss weight
            curriculum_weight = min(1.0, epoch / 20.0)  # Ramp up over 20 epochs
            total_loss += curriculum_weight * 1.5 * next_action_loss
        
        # Loss 4: Phase Prediction
        if target_phases is not None:
            phase_loss = F.cross_entropy(
                phase_logits.view(-1, self.num_phase_classes),
                target_phases.view(-1),
                ignore_index=-1
            )
            outputs['phase_loss'] = phase_loss
            total_loss += 0.3 * phase_loss  # Lower weight
        
        # Consistency loss between recognition and generation (optional)
        if current_action_targets is not None and seq_len > 1:
            # Recognition at t should be similar to generation prediction at t-1
            consistency_loss = F.mse_loss(
                torch.sigmoid(action_rec_logits[:, 1:]),  # Recognition at t+1
                torch.sigmoid(next_action_logits[:, :-1])  # Prediction at t for t+1
            )
            outputs['consistency_loss'] = consistency_loss
            total_loss += 0.1 * consistency_loss  # Very small weight
        
        outputs['total_loss'] = total_loss
        return outputs

    # Add target-specific loss weighting in your code:
    def compute_targeted_loss(self, outputs, targets):
        # Standard losses
        base_loss = outputs['total_loss']
        
        # Extra focus on target (T) actions
        target_action_indices = [/* your target action indices */]
        if len(target_action_indices) > 0:
            target_predictions = outputs['action_pred'][:, :, target_action_indices]
            target_labels = targets[:, :, target_action_indices]
            target_loss = F.binary_cross_entropy_with_logits(
                target_predictions, target_labels
            )
            base_loss += 0.3 * target_loss  # Boost target learning
        
        return base_loss

    def generate_sequence(self, 
                         initial_frames: torch.Tensor,
                         horizon: int = 15,
                         temperature: float = 1.0,
                         top_p: Optional[float] = None) -> Dict[str, torch.Tensor]:
        """
        Autoregressive sequence generation using the GPT2 generation path.
        Interface unchanged for compatibility.
        """
        
        self.eval()
        batch_size, context_len, _ = initial_frames.shape
        device = initial_frames.device
        
        generated_frames = initial_frames.clone()
        predicted_actions = []
        predicted_phases = []
                
        with torch.no_grad():
            for step in range(horizon):
                max_context = min(self.max_length - 1, generated_frames.size(1))
                current_context = generated_frames[:, -max_context:]
                
                outputs = self.forward(current_context)
                
                # Get predictions for next timestep
                next_frame = outputs['next_frame_pred'][:, -1:, :]
                action_logits = outputs['action_logits'][:, -1, :]
                phase_logits = outputs['phase_logits'][:, -1, :]
                
                # Apply temperature and nucleus sampling
                if temperature != 1.0:
                    action_logits = action_logits / temperature
                
                if top_p is not None:
                    action_probs = torch.sigmoid(action_logits)
                    sorted_probs, sorted_indices = torch.sort(action_probs, descending=True, dim=-1)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = 0
                    
                    indices_to_remove = sorted_indices.gather(-1, sorted_indices_to_remove.long())
                    action_logits.scatter_(-1, indices_to_remove, float('-inf'))
                
                action_probs = torch.sigmoid(action_logits)
                sampled_actions = torch.bernoulli(action_probs)
                phase_probs = F.softmax(phase_logits, dim=-1)
                
                predicted_actions.append(sampled_actions)
                predicted_phases.append(phase_probs)
                generated_frames = torch.cat([generated_frames, next_frame], dim=1)
                
        result = {
            'generated_frames': generated_frames[:, context_len:],
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
        Interface unchanged for compatibility.
        """
        batch_size, seq_len, _ = frame_sequence.shape
        if seq_len < 2:
            raise ValueError("Frame sequence must have at least 2 frames for action prediction.")

        self.eval()
        with torch.no_grad():
            outputs = self.forward(frame_sequence)
            # Use the GPT2 generation path for next action prediction
            action_probs = outputs['action_pred'][:, -1, :]
            return action_probs
    
    def save_model(self, path: str):
        """Save model - interface unchanged for compatibility"""
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
        
        print(f"✅ Improved Autoregressive IL Model saved to: {path}")
    
    @classmethod
    def load_model(cls, path: str, device: Optional[torch.device] = None):
        """Load model - interface unchanged for compatibility"""
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint['config']
        
        model = cls(**config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        print(f"✅ Improved Autoregressive IL Model loaded from: {path}")
        return model


# Example usage and testing
if __name__ == "__main__":
    print("🎓 IMPROVED AUTOREGRESSIVE IMITATION LEARNING MODEL")
    print("   - Enhanced dual-path architecture (BiLSTM + GPT2)")
    print("   - Optimized for small datasets (40 videos)")
    print("   - Clear temporal alignment (current vs next actions)")
    print("   - Curriculum learning support")
    print("   - Backward compatible interface")
    
    # Test model initialization
    model = AutoregressiveILModel(
        hidden_dim=512,      # Smaller for 40 videos
        embedding_dim=1024,
        n_layer=4,           # Fewer layers
        num_action_classes=100,
        num_phase_classes=7,
        dropout=0.2          # Higher dropout for regularization
    )
    
    # Test forward pass
    batch_size, seq_len, embedding_dim = 2, 10, 1024
    frame_embeddings = torch.randn(batch_size, seq_len, embedding_dim)
    target_actions = torch.randint(0, 2, (batch_size, seq_len, 100)).float()
    
    outputs = model(frame_embeddings, target_current_actions=target_actions, epoch=0)
    print(f"✅ Forward pass successful")
    print(f"   Total loss: {outputs['total_loss'].item():.4f}")
    print(f"   Output keys: {list(outputs.keys())}")