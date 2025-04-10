import torch
import torch.nn as nn
import torch.nn.functional as F

class RecognitionHead(nn.Module):
    """
    Simple recognition head with attention and linear classification layers.
    For instrument and action triplet recognition.
    """
    def __init__(self, cfg, embedding_dim, hidden_dim, num_action_classes, num_instrument_classes=6, dropout=0.1):
        super(RecognitionHead, self).__init__()
        self.cfg = cfg
        self.sequence_model_name = cfg['sequence_model']  # 'transformer', 'lstm', 'gru'
        self.sequence_model = None
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_action_classes = num_action_classes
        self.num_instrument_classes = num_instrument_classes
        
        # Input projection
        self.input_projection = nn.Linear(embedding_dim, hidden_dim)
        
        # Sequence model (Transformer, LSTM, GRU, etc.)
        if self.sequence_model_name=='transformer':
            self.sequence_model = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=cfg['num_heads'], dim_feedforward=hidden_dim*4, dropout=dropout),
                num_layers=cfg['num_layers'],
            )
        elif self.sequence_model_name=='lstm':
            self.sequence_model = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        elif self.sequence_model_name=='gru':
            self.sequence_model = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        else:
            raise ValueError("Unsupported sequence model type. Choose 'transformer', 'lstm', or 'gru'.")
        
        # Classification layers
        self.action_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_action_classes)
        )
        
        if num_instrument_classes > 0:
            self.instrument_classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_instrument_classes)
            )
        
    def forward(self, x):
        """
        Forward pass through the recognition head.
        
        Args:
            x: Input tensor of shape [batch_size, seq_length, embedding_dim]
            
        Returns:
            Dictionary containing action and instrument logits
        """
        batch_size, seq_length, _ = x.shape
        
        # Project to hidden dimension
        hidden = self.input_projection(x)
        
 
        if self.sequence_model is not None:
            if self.sequence_model_name == 'transformer':
                hidden = hidden.permute(1, 0, 2)
                hidden = self.sequence_model(hidden)
                hidden = hidden.permute(1, 0, 2)
            elif self.sequence_model_name == 'lstm':
                hidden, _ = self.sequence_model(hidden)
            elif self.sequence_model_name == 'gru':
                hidden, _ = self.sequence_model(hidden)
        else:
            raise ValueError("Sequence model is not defined.")

        # Classification
        action_logits = self.action_classifier(hidden[:, -1, :])  # Use the last hidden state for classification
        
        outputs = {'action_logits': action_logits}
        
        if self.num_instrument_classes > 0:
            instrument_logits = self.instrument_classifier(hidden[:, -1, :])  # Use the last hidden state for classification
            outputs['instrument_logits'] = instrument_logits
        
        return outputs
    
    def save_model(self, save_dir):
        """Save the model to the specified directory."""
        import os
        from datetime import datetime
        
        os.makedirs(save_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(save_dir, f"recognition_head_{timestamp}.pt")
        
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': {
                'embedding_dim': self.embedding_dim,
                'hidden_dim': self.hidden_dim,
                'num_action_classes': self.num_action_classes,
                'num_instrument_classes': self.num_instrument_classes
            }
        }, save_path)
        
        return save_path
    
    def load_model(self, model_path):
        """Load the model from the specified path."""
        checkpoint = torch.load(model_path)
        self.load_state_dict(checkpoint['model_state_dict'])
        return self