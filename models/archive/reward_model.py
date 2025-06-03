import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Positional Encoding for Transformer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sin/cos positional encoding
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # Register buffer (persistent state)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # Add positional encoding to input
        # x shape: [batch_size, seq_len, embedding_dim]
        x = x + self.pe[:, :x.size(1), :]
        return x

# Bidirectional Attention Reward Prediction Model
class RewardPredictor(nn.Module):
    def __init__(self, input_dim, context_length=5, hidden_dim=256, num_heads=4, num_layers=2, dropout=0.1):
        super(RewardPredictor, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.context_length = context_length
        
        # Input projection (if input_dim != hidden_dim)
        self.input_projection = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(hidden_dim)
        
        # Layer normalization before transformer
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        
        # Transformer encoder
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )
        
        # Attention pooling layer to combine transformer outputs
        self.attention_pool = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )
        
        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x):
        """
        Forward pass through the bidirectional attention model.
        
        Args:
            x: Input tensor of shape [batch_size, context_length, embedding_dim]
            
        Returns:
            Predicted survival time
        """
        batch_size, seq_len, _ = x.size()
        
        # Project input to hidden dimension if needed
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Layer normalization
        x = self.norm(x)
        
        # Transformer encoder (bidirectional attention)
        # No mask is applied, allowing bidirectional attention
        encoded = self.transformer_encoder(x)
        
        # Attention pooling over sequence dimension
        # Calculate attention weights
        attn_weights = self.attention_pool(encoded)  # [batch_size, seq_len, 1]
        
        # Apply attention weights to get context vector
        context = torch.bmm(encoded.transpose(1, 2), attn_weights)  # [batch_size, hidden_dim, 1]
        context = context.squeeze(2)  # [batch_size, hidden_dim]
        
        # Alternative simplified approach: Use CLS token or mean pooling
        # context = encoded[:, 0, :]  # CLS token approach
        # or
        # context = encoded.mean(dim=1)  # Mean pooling
        
        # Predict survival time
        out = self.fc(context)
        
        return out.squeeze()  # Remove singleton dimension
    
    def get_attention_weights(self, x):
        """
        Get attention weights for visualization/analysis.
        
        Args:
            x: Input tensor of shape [batch_size, context_length, embedding_dim]
            
        Returns:
            Attention weights over the sequence
        """
        batch_size, seq_len, _ = x.size()
        
        # Project input to hidden dimension if needed
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Layer normalization
        x = self.norm(x)
        
        # Transformer encoder
        encoded = self.transformer_encoder(x)
        
        # Get attention weights
        attn_weights = self.attention_pool(encoded)  # [batch_size, seq_len, 1]
        
        return attn_weights.squeeze(-1)  # [batch_size, seq_len]

