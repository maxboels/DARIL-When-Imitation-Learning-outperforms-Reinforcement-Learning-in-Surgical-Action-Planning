import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer models.
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class RecognitionModule(nn.Module):
    """
    Recognition module for detecting and tracking actions and instruments in video frames.
    Uses self-attention blocks for better temporal modeling.
    Can be integrated with the CausalGPT2ForFrameEmbeddings model.
    """
    def __init__(self, 
                 embedding_dim: int, 
                 hidden_dim: int,
                 num_action_classes: int = 100,
                 num_instrument_classes: int = 50,
                 num_attention_heads: int = 8,
                 num_attention_layers: int = 3,
                 dropout: float = 0.1):
        """
        Initialize the recognition module.
        
        Args:
            embedding_dim: Dimension of input frame embeddings
            hidden_dim: Dimension of hidden representations
            num_action_classes: Number of possible action classes to recognize
            num_instrument_classes: Number of possible instrument classes to recognize
            num_attention_heads: Number of attention heads in transformer blocks
            num_attention_layers: Number of transformer layers
            dropout: Dropout probability for regularization
        """
        super(RecognitionModule, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_action_classes = num_action_classes
        self.num_instrument_classes = num_instrument_classes
        
        # Input projection layer
        self.input_projection = nn.Linear(embedding_dim, hidden_dim)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(hidden_dim)
        
        # Self-attention layers for temporal context
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_attention_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_attention_layers
        )
        
        # Layer normalization
        self.layer_norm_input = nn.LayerNorm(hidden_dim)
        self.layer_norm_output = nn.LayerNorm(hidden_dim)
        
        # Recognition heads
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_action_classes)
        )
        
        self.instrument_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_instrument_classes)
        )
        
        # Confidence estimator (for reliability score)
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Attention pooling for global context
        self.attention_pool = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )
        
        # Temporal smoothing parameters (learnable)
        self.smoothing_factor = nn.Parameter(torch.tensor(0.7))
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small random values."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param, std=0.02)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, frame_embeddings, attention_mask=None):
        """
        Forward pass through the recognition module.
        
        Args:
            frame_embeddings: Tensor of shape [batch_size, seq_length, embedding_dim]
            attention_mask: Optional mask to ignore padding (1 for tokens to attend to, 0 for padding)
            
        Returns:
            Dictionary containing action logits, instrument logits, confidence scores
        """
        batch_size, seq_length, _ = frame_embeddings.shape
        
        # Project input embeddings to hidden dimension
        hidden = self.input_projection(frame_embeddings)
        
        # Add positional encoding
        hidden = self.positional_encoding(hidden)
        
        # Apply input layer normalization
        hidden = self.layer_norm_input(hidden)
        
        # Apply transformer encoder for global context
        if attention_mask is not None:
            # Convert boolean mask to attention mask (0 for tokens to attend to, -inf for padding)
            attn_mask = (1 - attention_mask) * -10000.0
            context = self.transformer_encoder(hidden, src_key_padding_mask=attn_mask)
        else:
            context = self.transformer_encoder(hidden)
        
        # Apply output layer normalization
        context = self.layer_norm_output(context)
        
        # Generate predictions for each frame
        action_logits = self.action_head(context)
        instrument_logits = self.instrument_head(context)
        confidence = self.confidence_head(context)
        
        # Apply temporal smoothing if processing a sequence
        if seq_length > 1:
            # Learn optimal smoothing factor
            smooth_factor = torch.clamp(self.smoothing_factor, 0.0, 1.0)
            
            # Apply exponential moving average smoothing along temporal dimension
            # This approach doesn't require previous states from outside
            if seq_length > 2:
                smoothed_actions = action_logits.clone()
                smoothed_instruments = instrument_logits.clone()
                
                # Initialize with first frame
                prev_action = action_logits[:, 0:1, :]
                prev_instrument = instrument_logits[:, 0:1, :]
                
                # Apply smoothing frame by frame
                for t in range(1, seq_length):
                    curr_action = action_logits[:, t:t+1, :]
                    curr_instrument = instrument_logits[:, t:t+1, :]
                    
                    # Weighted average between current and previous
                    smoothed_action = smooth_factor * curr_action + (1 - smooth_factor) * prev_action
                    smoothed_instrument = smooth_factor * curr_instrument + (1 - smooth_factor) * prev_instrument
                    
                    # Store smoothed values
                    smoothed_actions[:, t:t+1, :] = smoothed_action
                    smoothed_instruments[:, t:t+1, :] = smoothed_instrument
                    
                    # Update previous
                    prev_action = smoothed_action
                    prev_instrument = smoothed_instrument
                
                # Replace with smoothed versions
                action_logits = smoothed_actions
                instrument_logits = smoothed_instruments
        
        # Also calculate attention weights for visualization/analysis
        attention_weights = self.attention_pool(context)
        
        # Get global context vector through attention pooling
        global_context = torch.bmm(
            attention_weights.transpose(1, 2),  # [batch_size, 1, seq_length]
            context  # [batch_size, seq_length, hidden_dim]
        )  # [batch_size, 1, hidden_dim]
        
        global_context = global_context.squeeze(1)  # [batch_size, hidden_dim]
        
        return {
            'action_logits': action_logits,
            'instrument_logits': instrument_logits,
            'confidence': confidence,
            'attention_weights': attention_weights.squeeze(-1),  # [batch_size, seq_length]
            'global_context': global_context  # [batch_size, hidden_dim]
        }
    
    def detect_actions(self, frame_embeddings, threshold=0.5, attention_mask=None):
        """
        Detect actions in the given frame embeddings.
        
        Args:
            frame_embeddings: Tensor of shape [batch_size, seq_length, embedding_dim]
            threshold: Confidence threshold for detection
            attention_mask: Optional mask for padding
            
        Returns:
            Dictionary containing detected actions, instruments, and their confidence scores
        """
        outputs = self.forward(frame_embeddings, attention_mask)
        
        # Apply sigmoid to get probabilities
        action_probs = torch.sigmoid(outputs['action_logits'])
        instrument_probs = torch.sigmoid(outputs['instrument_logits'])
        
        # Get detected actions and instruments (binary)
        detected_actions = (action_probs > threshold).float()
        detected_instruments = (instrument_probs > threshold).float()
        
        return {
            'detected_actions': detected_actions,
            'detected_instruments': detected_instruments,
            'action_probs': action_probs,
            'instrument_probs': instrument_probs,
            'confidence': outputs['confidence'],
            'attention_weights': outputs['attention_weights']
        }
    
    def track_sequence(self, frame_embeddings_sequence, window_size=None, attention_mask=None):
        """
        Track actions and instruments over a sequence of frames.
        Using self-attention, we can process the entire sequence at once,
        but optionally allow windowing for very long sequences.
        
        Args:
            frame_embeddings_sequence: Tensor of shape [batch_size, seq_length, embedding_dim]
            window_size: Optional size of sliding window for processing very long sequences
            attention_mask: Optional mask for padding
            
        Returns:
            Time series of detected actions and instruments
        """
        batch_size, seq_length, _ = frame_embeddings_sequence.shape
        
        # If the sequence is short enough or window_size is None, process the whole sequence at once
        if window_size is None or seq_length <= window_size:
            outputs = self.forward(frame_embeddings_sequence, attention_mask)
            return {
                'action_logits': outputs['action_logits'],
                'instrument_logits': outputs['instrument_logits'],
                'confidences': outputs['confidence'],
                'attention_weights': outputs['attention_weights']
            }
        
        # For longer sequences, use a sliding window approach with overlap
        # Initialize storage for tracking results
        all_actions = []
        all_instruments = []
        all_confidences = []
        all_attentions = []
        
        # Process sequence in sliding windows
        for i in range(0, seq_length, window_size // 2):  # 50% overlap between windows
            end_idx = min(i + window_size, seq_length)
            window = frame_embeddings_sequence[:, i:end_idx, :]
            
            # Create attention mask for the window if needed
            window_mask = None
            if attention_mask is not None:
                window_mask = attention_mask[:, i:end_idx]
            
            # Process current window
            outputs = self.forward(window, window_mask)
            
            # Store results (only keep non-overlapping part or last window)
            if i + window_size >= seq_length:
                # Last window - keep all
                all_actions.append(outputs['action_logits'])
                all_instruments.append(outputs['instrument_logits'])
                all_confidences.append(outputs['confidence'])
                all_attentions.append(outputs['attention_weights'])
            else:
                # Keep only first half (non-overlapping part)
                end_keep = window_size // 2
                all_actions.append(outputs['action_logits'][:, :end_keep, :])
                all_instruments.append(outputs['instrument_logits'][:, :end_keep, :])
                all_confidences.append(outputs['confidence'][:, :end_keep, :])
                all_attentions.append(outputs['attention_weights'][:, :end_keep])
        
        # Concatenate results
        action_logits = torch.cat(all_actions, dim=1)
        instrument_logits = torch.cat(all_instruments, dim=1)
        confidences = torch.cat(all_confidences, dim=1)
        attentions = torch.cat(all_attentions, dim=1)
        
        return {
            'action_logits': action_logits,
            'instrument_logits': instrument_logits,
            'confidences': confidences,
            'attention_weights': attentions
        }