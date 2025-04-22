import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Action Policy Model with reward weighting
class ActionPolicyModel(nn.Module):
    def __init__(self, input_dim, context_length=10, num_action_classes=100, hidden_dim=256):
        super(ActionPolicyModel, self).__init__()
        
        # LSTM to process sequence of frame embeddings
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True
        )
        
        # Action prediction head
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_action_classes)
        )
    
    def forward(self, x):
        # x shape: [batch_size, context_length, embedding_dim]
        lstm_out, _ = self.lstm(x)
        
        # Take the last LSTM output
        last_output = lstm_out[:, -1, :]
        
        # Predict action logits
        action_logits = self.action_head(last_output)
        
        return action_logits
