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
    
    This model implements the core functionality for surgical robotics world modeling:
    1. Predicts future states based on current states and actions
    2. Learns from streams of experience rather than isolated examples
    3. Supports auto-regressive generation for planning multiple steps ahead
    4. Optionally learns action prediction for imitation learning
    """
    def __init__(self, hidden_dim: int, embedding_dim: int, action_embedding_dim: int, n_layer: int, 
                    reward_head: bool = False,
                    max_length: int = 1024,
                    use_head: bool = False,
                    targets_dims: Dict[str, int] = None,
                    target_heads: List[str] = None,
                    loss_weights: Dict[str, float] = None,
                    num_action_classes: int = 100,
                    num_phase_classes: int = 7,
                    num_outcomes: int = 1,
                    # inputs
                    action_conditioning: bool = True,
                    # outputs
                    imitation_learning: bool = True,
                    reward_learning: bool = False,
                    action_learning: bool = False,
                    phase_learning: bool = False,
                    outcome_learning: bool = False,
                    ) -> None:
        """
        Initialize the generative model.
        """
        super(WorldModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.action_embedding_dim = action_embedding_dim
        self.n_layer = n_layer
        self.reward_head = reward_head
        self.max_length = max_length
        self.use_head = use_head
        self.targets_dims = targets_dims if targets_dims else {}
        self.target_heads = target_heads if target_heads else []
        self.loss_weights = loss_weights if loss_weights else {}
        self.w_z = self.loss_weights.get('_z', 1.0)
        self.w_a = self.loss_weights.get('_a', 1.0)
        self.w_p = self.loss_weights.get('_p', 1.0)
        self.w_r = self.loss_weights.get('_r', 1.0)
        self.w_q = self.loss_weights.get('_q', 1.0)
        # self.w_R = self.loss_weights.get('_R', 1.0)
        # self.w_a_temporal_consist = self.loss_weights.get('_a_temporal_consist', 0.1)
        self.num_action_classes = num_action_classes
        self.num_phase_classes = num_phase_classes
        self.num_outcomes = num_outcomes
        self.action_conditioning = action_conditioning
        self.imitation_learning = imitation_learning
        self.reward_learning = reward_learning
        self.action_learning = action_learning
        self.phase_learning = phase_learning
        self.outcome_learning = outcome_learning

        # Options for reward prediction
        self.state_reward_dim = 1
        self.outcome_reward_dim = 1
        
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
        
        # Initialize the world model
        self._init_world_model()
        self._init_weights()

    def _init_world_model(self):
        # GPT-2 model
        self.model = GPT2LMHeadModel(self.config)        
                
        if self.action_conditioning:
            self.action_embedding = nn.Linear(self.num_action_classes, self.action_embedding_dim)
            self.input_dim = self.embedding_dim + self.action_embedding_dim
        else:
            self.input_dim = self.embedding_dim

        # Projection to GPT2 input dimension
        self.input_projection = nn.Linear(self.input_dim, self.hidden_dim)

        # Output projection: from GPT-2 hidden dimension back to frame embedding dimension
        self.model.lm_head = nn.Linear(self.hidden_dim, self.embedding_dim)

        # Initialize heads for imitation learning and reward prediction
        if self.imitation_learning or self.reward_learning:
            self.heads = nn.ModuleDict()

        # Imitation learning heads for next action and phase prediction
        if self.imitation_learning:
            if self.action_learning:
                self.heads['_a'] = nn.Linear(self.hidden_dim, self.num_action_classes)
            if self.phase_learning:
                self.heads['_p'] = nn.Linear(self.hidden_dim, self.num_phase_classes)
        
        if self.reward_learning:
            self.heads['_r_phase_completion'] = nn.Linear(self.hidden_dim, self.state_reward_dim)
            self.heads['_r_phase_initiation'] = nn.Linear(self.hidden_dim, self.state_reward_dim)
            self.heads['_r_phase_progression'] = nn.Linear(self.hidden_dim, self.state_reward_dim)
            self.heads['_r_action_probability'] = nn.Linear(self.hidden_dim, self.state_reward_dim)
            self.heads['_r_risk'] = nn.Linear(self.hidden_dim, self.state_reward_dim)
            self.heads['_r_global_progression'] = nn.Linear(self.hidden_dim, self.state_reward_dim)
        
        if self.outcome_learning:
            self.heads['_q'] = nn.Linear(self.hidden_dim, self.num_outcomes) # expected cumulative reward i.e. Q-value or outcome

    def _init_weights(self):
        """Initialize the weights for the custom layers."""
        nn.init.normal_(self.input_projection.weight, std=0.02)
        nn.init.zeros_(self.input_projection.bias)
        nn.init.normal_(self.model.lm_head.weight, std=0.02)
        nn.init.zeros_(self.model.lm_head.bias)
    
    def forward(self, 
                current_state: torch.Tensor, 
                next_state: Optional[torch.Tensor] = None,
                next_rewards: Dict[str, torch.Tensor] = None,
                next_actions: Optional[torch.Tensor] = None,
                next_phases: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                eval_mode: str = 'basic', # 'basic' or 'full'
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training the model with the correct temporal relationship:
        current_state + next_actions -> next_state
        
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
        
        # Ensure next_actions is provided
        if next_actions is None:
            raise ValueError("next_actions must be provided for the forward pass to capture the correct temporal relationship")
        
        # Get action embeddings from next_actions shape: [batch_size, seq_len, num_action_classes]
        if self.action_conditioning:
            next_action_embeddings = self.action_embedding(next_actions)  # [batch_size, seq_len, action_embedding_dim]
            current_state = torch.cat([current_state, next_action_embeddings], dim=-1)
        
        # Project to GPT2 input dimension
        proj_inputs = self.input_projection(current_state)  # [batch_size, seq_len, gpt2_input_dim]
        
        # Create position IDs
        position_ids = torch.arange(seq_length, dtype=torch.long, device=current_state.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # Use provided current_state and next_state
        _z = next_state
        
        # Pass through GPT-2 model
        outputs = self.model(
            inputs_embeds=proj_inputs,
            position_ids=position_ids,
            attention_mask=attention_mask, # None if not provided
            output_hidden_states=True
        ) # outputs: logits, past_key_values, hidden_states
        
        # Get the last hidden state
        last_hidden_state = outputs.hidden_states[-1] # D=768
        
        # Project back to embedding dimension
        _z_hat = self.model.lm_head(last_hidden_state) # D=1024
        
        # Calculate MSE loss between predictions and targets
        _z_loss = F.mse_loss(_z_hat, _z)

        # If we want to use imitation learning - predict the next action
        other_losses = {}
        head_outputs = {}
        if self.imitation_learning:

            if self.action_learning and next_actions is not None:
                action_logits = self.heads['_a'](last_hidden_state)
                # Calculate loss for the action head
                action_loss = F.binary_cross_entropy_with_logits(action_logits, next_actions)
                other_losses['_a_loss'] = action_loss * self.w_a

            if self.phase_learning and next_phases is not None:
                phase_logits = self.heads['_p'](last_hidden_state).permute(0, 2, 1) # [batch_size, num_phase_classes, seq_len]
                # Calculate loss for the phase head (cross-entropy with a softmax output)
                phase_loss = F.cross_entropy(phase_logits, next_phases)
                other_losses['_p_loss'] = phase_loss * self.w_p

        # If predicting rewards associated with the next state
        # TODO: get the ground truth rewards scores: Done
        # the goal is for the model to learn to pick actions that lead to higher rewards in short and long term
        # Also, we want to learn the reward function itself that uses weights for each signal
        # and knows when to focus on which grounded signals or human preferences like safety
        # Ideally, we want to focus more on discovering new pathways/trajectories from raw data and final outcomes, leaving
        # space for the model to come up with better strategies humans might not have thought of.
        if self.reward_learning and next_rewards is not None:
            # Calculate loss for the reward head
            reward_phase_completion = self.heads['_r_phase_completion'](last_hidden_state)
            reward_phase_initiation = self.heads['_r_phase_initiation'](last_hidden_state)
            reward_phase_progression = self.heads['_r_phase_progression'](last_hidden_state)
            reward_action_probability = self.heads['_r_action_probability'](last_hidden_state)
            reward_risk_penalty = self.heads['_r_risk'](last_hidden_state)
            reward_global_progression = self.heads['_r_global_progression'](last_hidden_state)
            # Calculate loss for the reward head
            reward_loss_phase_completion = F.mse_loss(reward_phase_completion, next_rewards['_r_phase_completion'])
            reward_loss_phase_initiation = F.mse_loss(reward_phase_initiation, next_rewards['_r_phase_initiation'])
            reward_loss_phase_progression = F.mse_loss(reward_phase_progression, next_rewards['_r_phase_progression'])
            reward_loss_action_probability = F.mse_loss(reward_action_probability, next_rewards['_r_action_probability'])
            reward_loss_risk_penalty = F.mse_loss(reward_risk_penalty, next_rewards['_r_risk'])
            reward_loss_global_progression = F.mse_loss(reward_global_progression, next_rewards['_r_global_progression'])
            # Store losses
            other_losses['_r_phase_completion_loss'] = reward_loss_phase_completion * self.w_r * 1.0
            other_losses['_r_phase_initiation_loss'] = reward_loss_phase_initiation * self.w_r * 1.0
            other_losses['_r_phase_progression_loss'] = reward_loss_phase_progression * self.w_r * 1.0
            other_losses['_r_action_probability_loss'] = reward_loss_action_probability * self.w_r * 1.0
            other_losses['_r_risk_loss'] = reward_loss_risk_penalty * self.w_r * 0.1
            other_losses['_r_global_progression_loss'] = reward_loss_global_progression * self.w_r * 0.1

        outputs = {
            "_z_loss": _z_loss,
            "_z_hat": _z_hat,
            "_a_hat": action_logits if self.action_learning and self.imitation_learning else None,
            "logits": outputs.logits,
            "head_outputs": head_outputs,
            "last_hidden_states": last_hidden_state
        }
        # Add other losses if they exist
        for key, value in other_losses.items():
            outputs[key] = value
        
        # Calculate total loss
        total_loss = self.w_z * _z_loss
        for loss_name, loss_value in other_losses.items():
            total_loss = total_loss + loss_value
        
        outputs["total_loss"] = total_loss
        
        return outputs
    
    def predict_next_action(self, state: torch.Tensor) -> torch.Tensor:
        """
        Predict the next action based on the current state.
        
        Args:
            state: Current state embeddings of shape [batch_size, embedding_dim] or [batch_size, 1, embedding_dim]
            
        Returns:
            Predicted next action probabilities
        """
        # Ensure state has the right shape [batch_size, seq_len, embedding_dim]
        if state.dim() == 2:
            state = state.unsqueeze(1)  # Add sequence dimension
        
        with torch.no_grad():
            # Project to hidden dimension
            # For action prediction, we don't need to provide action embeddings
            # We're just using the current state to predict what action should come next
            state_proj = self.input_projection(
                torch.cat([
                    state, 
                    torch.zeros(state.shape[0], state.shape[1], self.action_embedding_dim, device=state.device)
                ], dim=-1)
            )
            
            # Get hidden states
            outputs = self.model(
                inputs_embeds=state_proj,
                output_hidden_states=True
            )
            
            last_hidden_state = outputs.hidden_states[-1]
            
            # If we have an action head, use it
            if self.imitation_learning and '_a' in self.heads:
                action_logits = self.heads['_a'](last_hidden_state)
                # Return action probabilities
                return torch.sigmoid(action_logits)
            else:
                # Fallback: return random probabilities
                batch_size = state.shape[0]
                seq_len = state.shape[1]
                return torch.rand(batch_size, seq_len, self.num_action_classes, device=state.device)

    def sample_next_action(self, state: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        Sample the next action based on predicted probabilities.
        
        Args:
            state: Current state embeddings
            temperature: Sampling temperature (higher = more random)
            
        Returns:
            One-hot encoded sampled actions
        """
        action_probs = self.predict_next_action(state)
        
        # Apply temperature
        if temperature != 1.0:
            logits = torch.log(action_probs + 1e-8) / temperature
            action_probs = torch.softmax(logits, dim=-1)
        
        # Sample actions (binary multi-label)
        sampled_actions = torch.bernoulli(action_probs)
        
        return sampled_actions

    def generate_conditional_future_states(self, 
                input_embeddings: torch.Tensor,
                input_actions: Optional[torch.Tensor] = None,
                horizon: int = 10,
                temperature: float = 1.0,
                top_k: Optional[int] = None,
                top_p: Optional[float] = None,
                repetition_penalty: float = 1.0,
                do_sample: bool = True,
                use_past: bool = True) -> Dict[str, torch.Tensor]:
        """
        Generate future frame embeddings (states) autoregressively.
        
        Args:
            input_embeddings: Initial frame embeddings of shape [batch_size, seq_length, embedding_dim]
            input_actions: Initial actions for the first step (optional)
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
        
        # Initialize actions if not provided
        if input_actions is None and self.imitation_learning:
            # Sample initial actions from the current state
            input_actions = self.sample_next_action(input_embeddings)
        elif input_actions is None:
            # Create random actions if we don't have imitation learning
            input_actions = torch.zeros(batch_size, seq_length, self.num_action_classes, device=device)
            input_actions = input_actions.bernoulli(0.5)  # Random binary actions
        
        # Project input embeddings and actions to GPT-2 hidden dimension
        action_embeddings = self.action_embedding(input_actions)
        combined_inputs = torch.cat([input_embeddings, action_embeddings], dim=-1)
        hidden_states = self.input_projection(combined_inputs)
        
        # Initialize past key values if using them
        past = None
        
        # Initialize the generated sequence with the input embeddings
        generated_embeddings = input_embeddings.clone()
        generated_actions = input_actions.clone()
        
        # Create attention mask for the input sequence
        attention_mask = torch.ones(batch_size, seq_length, device=device)
        
        # Initialize containers to store hidden states and head outputs
        all_hidden_states = []
        all_head_outputs = {target: [] for target in self.target_heads} if self.target_heads else {}
        
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
            next_token_hidden = outputs.hidden_states[-1][:, -1, :]  # [batch_size, hidden_dim]
            all_hidden_states.append(next_token_hidden)
            
            # Project hidden state to embedding dimension
            next_frame_embedding = self.model.lm_head(next_token_hidden)  # [batch_size, embedding_dim]
            
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
            
            # Compute head outputs for the current frame and predict next action
            next_action = None
            if self.imitation_learning and self.target_heads:
                for target in self.target_heads:
                    head_output = self.heads[target](next_token_hidden)
                    all_head_outputs[target] = all_head_outputs.get(target, []) + [head_output]
                    
                    # If we have an action head, use it to predict the next action
                    if target == '_a':
                        action_logits = head_output
                        next_action = torch.bernoulli(torch.sigmoid(action_logits))
            
            # If we don't have a predicted action yet, sample one
            if next_action is None:
                next_action = self.sample_next_action(next_frame_embedding.unsqueeze(1)).squeeze(1)
            
            # Reshape for concatenation
            next_frame_embedding = next_frame_embedding.unsqueeze(1)  # [batch_size, 1, embedding_dim]
            next_action = next_action.unsqueeze(1)  # [batch_size, 1, num_action_classes]
            
            # Add the generated frame and action to the sequences
            generated_embeddings = torch.cat([generated_embeddings, next_frame_embedding], dim=1)
            generated_actions = torch.cat([generated_actions, next_action], dim=1)
            
            # Update hidden states for next iteration
            next_action_embedding = self.action_embedding(next_action)
            next_combined = torch.cat([next_frame_embedding, next_action_embedding], dim=-1)
            next_hidden = self.input_projection(next_combined)
            hidden_states = torch.cat([hidden_states, next_hidden], dim=1)
            
            # Extend attention mask
            attention_mask = torch.cat([
                attention_mask, 
                torch.ones(batch_size, 1, device=device)
            ], dim=1)
        
        # Stack all hidden states
        all_hidden_states = torch.stack(all_hidden_states, dim=1)  # [batch_size, horizon, hidden_dim]
        
        # Stack all head outputs
        stacked_head_outputs = {}
        for target, outputs_list in all_head_outputs.items():
            if outputs_list:
                stacked_head_outputs[target] = torch.stack(outputs_list, dim=1)
        
        # Return the full sequence including the input and generated atates and actions
        result = {
            "input_embeddings": input_embeddings,
            "input_actions": input_actions,
            "generated_embeddings": generated_embeddings[:, input_embeddings.size(1):],
            "generated_actions": generated_actions[:, input_actions.size(1):],
            "full_embeddings": generated_embeddings,
            "full_actions": generated_actions,
            "last_hidden_states": all_hidden_states,
            "head_outputs": stacked_head_outputs
        }
        
        return result

    def save(self, path: str) -> None:
        """Save the model to the specified path."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': {
                'hidden_dim': self.hidden_dim,
                'embedding_dim': self.embedding_dim,
                'action_embedding_dim': self.action_embedding_dim,
                'n_layer': self.n_layer,
                'max_length': self.max_length,
                'use_head': self.use_head,
                'targets_dims': self.targets_dims,
                'target_heads': self.target_heads,
                'loss_weights': self.loss_weights,
                'num_action_classes': self.num_action_classes,
                'num_outcomes': self.num_outcomes,
                'imitation_learning': self.imitation_learning
            }
        }, path)
    
    @classmethod
    def load(cls, path: str, device: Optional[torch.device] = None) -> 'WorldModel':
        """Load the model from the specified path."""
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint['config']
        
        model = cls(**config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        return model