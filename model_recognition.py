import torch
import torch.nn as nn
import torch.nn.functional as F

class RecognitionHead(nn.Module):
    """
    Simple recognition head with attention and linear classification layers.
    For instrument and action triplet recognition.
    """
    def __init__(self, embedding_dim, hidden_dim, num_action_classes, num_instrument_classes=0, dropout=0.1):
        super(RecognitionHead, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_action_classes = num_action_classes
        self.num_instrument_classes = num_instrument_classes
        
        # Input projection
        self.input_projection = nn.Linear(embedding_dim, hidden_dim)
        
        # Attention mechanism for temporal context
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )
        
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
        
        # Apply attention to capture temporal context
        attention_weights = self.attention(hidden)
        context = torch.bmm(attention_weights.transpose(1, 2), hidden)
        context = context.squeeze(1)
        
        # Classification
        action_logits = self.action_classifier(context)
        
        outputs = {'action_logits': action_logits}
        
        if self.num_instrument_classes > 0:
            instrument_logits = self.instrument_classifier(context)
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