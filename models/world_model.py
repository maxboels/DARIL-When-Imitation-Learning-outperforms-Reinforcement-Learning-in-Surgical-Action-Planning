import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import transformers
from transformers import GPT2Config, GPT2LMHeadModel
from typing import Dict, Any, Optional, List, Tuple, Union

class WorldModel(nn.Module):
    """
    WORLD MODEL: GPT-2 based model for predicting future frame embeddings.
    """
    def __init__(self, hidden_dim: int, embedding_dim: int, action_embedding_dim: int,
                    n_layer: int, max_length: int = 1024,
                    use_head: bool = False,
                    targets_dims: Dict[str, int] = None,
                    target_heads: List[str] = None,
                    loss_weights: Dict[str, float] = None,
                    num_action_classes: int = 100,
                    num_outcomes: int = 1,
                    eval: Dict[str, Any] = None,
                    imitation_learning: bool = False) -> None:
        """
        Initialize the generative model.
        """
        super(WorldModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.action_embedding_dim = action_embedding_dim
        self.n_layer = n_layer
        self.max_length = max_length
        self.use_head = use_head
        self.targets_dims = targets_dims
        self.target_heads = target_heads
        self.loss_weights = loss_weights
        self.w_z = loss_weights.get('_z', 1.0) if loss_weights else 1.0
        self.w_a = loss_weights.get('_a', 1.0) if loss_weights else 1.0
        self.w_r = loss_weights.get('_r', 1.0) if loss_weights else 1.0
        self.w_q = loss_weights.get('_q', 1.0) if loss_weights else 1.0
        self.w_R = loss_weights.get('_R', 1.0) if loss_weights else 1.0
        self.w_a_temporal_consist = loss_weights.get('_a_temporal_consist', 0.1) if loss_weights else 0.0
        self.num_action_classes = num_action_classes
        self.num_outcomes = num_outcomes
        # Configuration for GPT-2 model
        self.config = GPT2Config.from_pretrained('gpt2')
        self.config.hidden_size = self.hidden_dim
        self.config.num_hidden_layers = self.n_layer
        self.config.num_attention_heads = 8
        self.config.add_cross_attention = False  # Ensure it's purely autoregressive
        self.config.output_hidden_states = True
        self.config.n_layer = self.n_layer
        self.config.n_embd = self.hidden_dim
        self.config.vocab_size = 1  # Not using vocab embeddings in traditional sense

        # Options for imitation learning
        self.imitation_learning = imitation_learning
        
        # Initialize the world model
        self._init_world_model()
        self._init_weights()

    # Fix variable definitions in _init_world_model()
    def _init_world_model(self):
        # GPT-2 model
        self.model = GPT2LMHeadModel(self.config)        
                
        # For binary multi-label actions
        self.action_embedding = nn.Linear(self.num_action_classes, self.action_embedding_dim)

        # Combined frame and action embedding dimensions
        self.combined_dim = self.embedding_dim + self.action_embedding_dim

        # Projection to GPT2 input dimension
        self.input_projection = nn.Linear(self.combined_dim, self.hidden_dim)

        # Output projection: from GPT-2 hidden dimension back to frame embedding dimension
        self.model.lm_head = nn.Linear(self.hidden_dim, self.embedding_dim)
        
        # Head
        if self.target_heads:
            self.heads = nn.ModuleDict()
            for target in self.target_heads:
                target_dim = self.targets_dims[target]
                self.heads[target] = nn.Linear(self.hidden_dim, target_dim)

    def _init_weights(self):
        """Initialize the weights for the custom layers."""
        nn.init.normal_(self.input_projection.weight, std=0.02)
        nn.init.zeros_(self.input_projection.bias)
        nn.init.normal_(self.model.lm_head.weight, std=0.02)
        nn.init.zeros_(self.model.lm_head.bias)
    
    def forward(self, 
                current_state: torch.Tensor, 
                next_state: Optional[torch.Tensor] = None,
                next_actions: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training the model.
        
        Args:
            current_state: Frame embeddings tensor of shape [batch_size, seq_length, embedding_dim]
            next_state: Target frame embeddings tensor of shape [batch_size, seq_length, embedding_dim]
                   If None, will use current_state shifted to the right for teacher forcing
            next_actions: We pass the next actions as input to the model for conditional generation of the next state,
                     we want to learn the effect of future actions on the next state.
            attention_mask: Attention mask of shape [batch_size, seq_length]
        
        Returns:
            Dictionary with loss and predictions
        """
        batch_size, seq_length, _ = current_state.size()
        
        # Get action embeddings from next_actions shape: [batch_size, seq_len, num_action_classes]
        next_action_embeddings = self.action_embedding(next_actions)  # [batch_size, seq_len, action_embedding_dim]
        
        # Concatenate current frame and next action embeddings along the feature dimension
        conditinal_state = torch.cat([current_state, next_action_embeddings], dim=-1)
        
        # Project to GPT2 input dimension
        proj_inputs = self.input_projection(conditinal_state)  # [batch_size, seq_len, gpt2_input_dim]
        
        # Create position IDs
        position_ids = torch.arange(seq_length, dtype=torch.long, device=current_state.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # If no next_state provided, use current_state shifted right for teacher forcing
        if next_state is None:
            # Shift current_state to the right, prepend with zeros as first token
            shifted_inputs = torch.cat([
                torch.zeros(batch_size, 1, self.embedding_dim, device=current_state.device),
                current_state[:, :-1]
            ], dim=1)
            
            # Project the shifted current_state
            shifted_hidden_states = self.input_projection(shifted_inputs)
            
            # Use the original current_state as targets
            _z = current_state
        else:
            # Use provided current_state and next_state (can predict any future frame embeddings)
            shifted_hidden_states = proj_inputs
            _z = next_state
        
        # Pass through GPT-2 model
        # We're using the model differently - we're providing hidden states directly
        # and treating the output as continuous values rather than categorical logits
        outputs = self.model(
            inputs_embeds=shifted_hidden_states,
            position_ids=position_ids,
            attention_mask=attention_mask, # None if not provided
            output_hidden_states=True
        ) # ouputs: logits, past_key_values, hidden_states
        
        # Get the last hidden state
        last_hidden_state = outputs.hidden_states[-1] # D=768
        
        # Project back to embedding dimension
        _z_hat = self.model.lm_head(last_hidden_state) # D=1024
        
        # Calculate MSE loss between predictions and targets
        _z_loss = F.mse_loss(_z_hat, _z)

        # If we want to use imitation learning - predict the next action to mimic expert behavior, we just need
        # to predict the next actions based on the current state.
        other_losses = None
        head_outputs = None
        if self.imitation_learning:
            other_losses = {}
            if self.target_heads:
                head_outputs = {}
                for target in self.target_heads:
                    # Forward pass through the heads
                    head_outputs[target] = self.heads[target](last_hidden_state)
                    
                    # Calculate loss for the heads
                    if target == '_a' and self.w_a is not None:
                        action_logits = head_outputs[target]
                        
                        # Standard BCE loss
                        action_loss = F.binary_cross_entropy_with_logits(action_logits, next_actions)
                        
                        # Add temporal consistency within the sequence
                        # Skip first frame as it has no previous frame
                        if seq_length > 1:
                            # Compute logits for each frame except the last one
                            prev_logits = action_logits[:, :-1, :]
                            # Compute logits for each frame except the first one
                            curr_logits = action_logits[:, 1:, :]
                            
                            # Calculate temporal consistency loss
                            temporal_loss = F.mse_loss(
                                torch.sigmoid(curr_logits),
                                torch.sigmoid(prev_logits)
                            )
                            
                            # Combine losses
                            other_losses['_a_loss'] = self.w_a * (action_loss + self.w_a_temporal_consist * temporal_loss)
                        else:
                            other_losses['_a_loss'] = self.w_a * action_loss
                            
                    elif target == '_R':
                        pass
        
        outputs = {
            "_z_loss": _z_loss,
            "_z_hat": _z_hat,
            "logits": outputs.logits,
            "head_outputs": head_outputs,
            "last_hidden_states": last_hidden_state
        }
        # Add other losses if they exist
        if other_losses:
            for key, value in other_losses.items():
                outputs[key] = value
        
        return outputs
    
    def predict_next_action(self, state):
        """Predict the next action based on the current state"""
        # If you have an action head, use it
        if '_a' in self.heads:
            with torch.no_grad():
                # Project to hidden dimension first
                hidden = self.input_projection(state)
                action_logits = self.heads['_a'](hidden)
                # Return most likely action
                return torch.argmax(action_logits, dim=-1)
        else:
            # Fallback: return random actions
            batch_size = state.shape[0]
            return torch.randint(0, self.num_action_classes, (batch_size,), device=state.device)

    def generate_trajectory(self, initial_state, initial_action, num_steps):
        """Generate a trajectory starting from initial state and action"""
        batch_size = initial_state.shape[0]
        device = initial_state.device
        
        # Initialize containers
        states = [initial_state]
        actions = [initial_action]
        
        current_state = initial_state
        
        for t in range(num_steps):
            # Decide on next action (could be from a policy, random, or provided)
            next_action = predict_next_action(current_state)  # Custom function
            actions.append(next_action)
            
            # Prepare inputs (add sequence dimension if needed)
            if current_state.dim() == 2:  # [batch_size, embedding_dim]
                current_state = current_state.unsqueeze(1)  # [batch_size, 1, embedding_dim]
                next_action = next_action.unsqueeze(1)  # [batch_size, 1]
            
            # Predict next state
            with torch.no_grad():
                next_state = self.forward(current_state, next_action)
                
                # If we used sequence dimension, remove it
                if next_state.shape[1] == 1:
                    next_state = next_state.squeeze(1)
            
            # Store and update
            states.append(next_state)
            current_state = next_state
        
        # Stack results
        states = torch.stack(states, dim=1)  # [batch_size, num_steps+1, embedding_dim]
        actions = torch.stack(actions, dim=1)  # [batch_size, num_steps+1]
        
        return states, actions

    def generate(self, 
                input_embeddings: torch.Tensor,
                horizon: int = 10,
                temperature: float = 1.0,
                top_k: Optional[int] = None,
                top_p: Optional[float] = None,
                repetition_penalty: float = 1.0,
                do_sample: bool = True,
                use_past: bool = True,
                use_memory: bool = False) -> Dict[str, torch.Tensor]:
        """
        Generate future frame embeddings autoregressively.
        
        Args:
            input_embeddings: Initial frame embeddings of shape [batch_size, seq_length, embedding_dim]
            horizon: Number of future frames to generate
            temperature: Sampling temperature for generation
            top_k: Number of highest probability tokens to keep for top-k sampling
            top_p: Cumulative probability for nucleus sampling
            repetition_penalty: Penalty for repeating tokens
            do_sample: Whether to use sampling or greedy decoding
            use_past: Whether to use past key values for faster generation
            
        Returns:
            Dictionary containing generated embeddings and optional head outputs
        """
        self.eval()
        device = input_embeddings.device
        batch_size, seq_length, _ = input_embeddings.size()
        
        # Project input embeddings to GPT-2 hidden dimension
        hidden_states = self.input_projection(input_embeddings)
        
        # Initialize past key values if using them
        past = None
        
        # Initialize the generated sequence with the input embeddings
        generated_embeddings = input_embeddings.clone()
        
        # Create attention mask for the input sequence
        attention_mask = torch.ones(batch_size, seq_length, device=device)
        
        # Initialize containers to store hidden states and head outputs
        all_hidden_states = []
        all_head_outputs = {target: [] for target in self.target_heads} if self.target_heads else None
        
        # Generate new frames autoregressively
        for i in range(horizon):
            # Create position IDs for the current sequence
            position_ids = torch.arange(
                seq_length + i, 
                dtype=torch.long, 
                device=device
            ).unsqueeze(0).expand(batch_size, -1)
            
            # If using past key values, only the last token needs to be processed
            if use_past and past is not None:
                # Get only the last frame embedding
                current_hidden = hidden_states[:, -1:, :]
                current_position_ids = position_ids[:, -1:]
                current_attention_mask = torch.cat([
                    attention_mask,
                    torch.ones(batch_size, i, device=device)
                ], dim=1)
            else:
                # Process the entire sequence
                current_hidden = hidden_states
                current_position_ids = position_ids
                current_attention_mask = attention_mask
            
            # Forward pass through GPT-2
            with torch.no_grad():
                outputs = self.model(
                    inputs_embeds=current_hidden,
                    position_ids=current_position_ids,
                    attention_mask=current_attention_mask,
                    past_key_values=past,
                    use_cache=use_past,
                    output_hidden_states=True
                )
            
            # Update past for next iteration if using past key values
            if use_past:
                past = outputs.past_key_values
            
            # Get the next frame embedding representation
            next_token_hidden = outputs.hidden_states[-1][:, -1, :] # D=768
            all_hidden_states.append(next_token_hidden)
            
            # Project hidden state to embedding dimension
            next_frame_embedding = self.model.lm_head(next_token_hidden) # D=1024
            
            # Apply temperature scaling and optional sampling techniques
            if do_sample:
                # For continuous generation, we add noise scaled by temperature
                noise = torch.randn_like(next_frame_embedding) * temperature
                next_frame_embedding = next_frame_embedding + noise
                
                # Apply repetition penalty by comparing with previous embeddings
                if repetition_penalty != 1.0:
                    # Calculate cosine similarity with previous frames
                    # Higher similarity will lead to penalization
                    similarities = F.cosine_similarity(
                        next_frame_embedding.unsqueeze(1), 
                        generated_embeddings, 
                        dim=2
                    )
                    # Apply penalty by scaling down embeddings based on similarity
                    penalty = torch.pow(similarities.max(dim=1)[0], 1/repetition_penalty).unsqueeze(1)
                    next_frame_embedding = next_frame_embedding * penalty
            
            # Compute head outputs for the current frame
            if self.target_heads:
                for target in self.target_heads:
                    head_output = self.heads[target](next_token_hidden)
                    all_head_outputs[target].append(head_output)
            
            # Reshape for concatenation
            next_frame_embedding = next_frame_embedding.unsqueeze(1)
            
            # Add the generated frame to the sequence
            generated_embeddings = torch.cat([generated_embeddings, next_frame_embedding], dim=1)
            
            # Update hidden states for next iteration
            next_hidden = self.input_projection(next_frame_embedding)
            hidden_states = torch.cat([hidden_states, next_hidden], dim=1)
            
            # Extend attention mask
            attention_mask = torch.cat([
                attention_mask, 
                torch.ones(batch_size, 1, device=device)
            ], dim=1)
        
        # Stack all hidden states
        all_hidden_states = torch.stack(all_hidden_states, dim=1)  # [batch_size, num_frames, hidden_dim]
        
        # Stack future predictions over the sequence dimension
        if self.target_heads:
            for target in self.target_heads:
                all_head_outputs[f"f{target}_seq_hat"] = torch.stack(all_head_outputs[target], dim=1)
        
        # Return the full sequence including the input and generated frames
        result = {
            "z": input_embeddings,
            "generated_embeddings": generated_embeddings[:, input_embeddings.size(1):],  # Rename for consistency
            "_zs_hat": generated_embeddings[:, input_embeddings.size(1):],  # Keep this if needed for backward compatibility
            "last_hidden_states": all_hidden_states,
            **all_head_outputs
        }
        
        return result

    def generate_future(self, 
                        initial_embedding: torch.Tensor, 
                        length: int = 10) -> Dict[str, torch.Tensor]:
        """
        Simplified method to generate future frames from a single embedding.
        
        Args:
            initial_embedding: Starting frame embedding of shape [batch_size, embedding_dim]
            length: Number of future frames to generate
            
        Returns:
            Dictionary containing generated sequence of future frame embeddings and head outputs
        """
        # Ensure the initial embedding has batch and sequence dimensions
        if initial_embedding.dim() == 2:
            # [batch_size, embedding_dim] -> [batch_size, 1, embedding_dim]
            initial_embedding = initial_embedding.unsqueeze(1)
        elif initial_embedding.dim() == 1:
            # [embedding_dim] -> [1, 1, embedding_dim]
            initial_embedding = initial_embedding.unsqueeze(0).unsqueeze(0)
        
        # Generate the sequence
        result = self.generate(
            input_embeddings=initial_embedding,
            horizon=length,
            do_sample=True,
            temperature=0.7,
            use_past=True
        )
        
        # Modify the result to only include generated future frames (exclude the initial frame)
        result["generated_embeddings"] = result["generated_embeddings"][:, 1:]
        
        return result

    def update_memory(self,
                      memory: Dict[str, torch.Tensor],
                      next_state: Optional[torch.Tensor] = None,
                      next_actions: Optional[torch.Tensor] = None,
                      attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Update the memory with new observations and perform a forward pass.
        
        Args:
            memory: Dictionary containing previous memory tensors
            next_state: Target frame embeddings tensor of shape [batch_size, seq_length, embedding_dim]
            next_actions: Target action logits tensor of shape [batch_size, num_action_classes]
            attention_mask: Attention mask of shape [batch_size, seq_length]
        
        Returns:
            Dictionary with loss and predictions
        """
        # Extract previous memory tensors
        current_state = memory["input_embeddings"]
        
        # Perform forward pass
        outputs = self.forward(
            current_state=current_state,
            next_state=next_state,
            next_actions=next_actions,
            attention_mask=attention_mask
        )
        
        return outputs
    
    def save(self, path: str) -> None:
        """Save the model to the specified path."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': {
                'hidden_dim': self.hidden_dim,
                'embedding_dim': self.embedding_dim,
                'n_layer': self.n_layer,
                'max_length': self.max_length
            }
        }, path)
    
    @classmethod
    def load(cls, path: str, device: Optional[torch.device] = None) -> 'WorldModel':
        """Load the model from the specified path."""
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint['config']
        
        model = cls(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        return model

# Example usage:
if __name__ == "__main__":
    # Configuration for a small GPT-2 model
    config = {
        'hidden_dim': 768,      # GPT-2 hidden dimension
        'embedding_dim': 1024,  # Frame embedding dimension
        'n_layer': 6,           # Number of transformer layers
        'max_length': 512       # Maximum sequence length
    }
    
    # Initialize the model
    model = WorldModel(config)
    
    # Example: Generate future frames from an initial frame embedding
    batch_size = 2
    initial_embedding = torch.randn(batch_size, 1, 1024)  # [batch_size, seq_length=1, embedding_dim]
    
    # Generate 10 future frames
    future_frames = model.generate(initial_embedding, horizon=10)
    
    print(f"Generated sequence shape: {future_frames.shape}")  # [batch_size, 11, embedding_dim]
    
    # Training example
    input_seq = torch.randn(batch_size, 5, 1024)  # 5 frames
    outputs = model.forward(input_seq)
    loss = outputs["loss"]
    
    print(f"Training loss: {loss.item()}")