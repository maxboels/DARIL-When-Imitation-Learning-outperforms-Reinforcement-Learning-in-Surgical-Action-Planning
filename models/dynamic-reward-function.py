import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicRewardFunction(nn.Module):
    """
    A flexible reward function that combines multiple grounded signals
    and adapts based on feedback, following the Era of Experience principles.
    
    This implements the bi-level optimization approach where:
    - Low level: Optimizes grounded signals from surgical environment
    - Top level: Adapts based on feedback to align with long-term goals
    """
    def __init__(self, num_signals=10, hidden_dim=128):
        super(DynamicRewardFunction, self).__init__()
        
        # Signal weighting network (adaptable weights for each signal)
        self.signal_weights = nn.Parameter(torch.ones(num_signals) / num_signals)
        
        # Context-dependent reward network
        self.context_reward = nn.Sequential(
            nn.Linear(num_signals, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Reward shaping network (temporal consistency)
        self.reward_shaping = nn.Sequential(
            nn.Linear(num_signals * 2, hidden_dim),  # Current and previous signals
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Signal normalization parameters (learned)
        self.signal_means = nn.Parameter(torch.zeros(num_signals))
        self.signal_stds = nn.Parameter(torch.ones(num_signals))
        
        # Hyperparameters
        self.reward_discount = 0.95  # Temporal discount factor
        self.shaping_weight = 0.2    # Weight for reward shaping
        
    def normalize_signals(self, signals):
        """Normalize signals using learned parameters"""
        return (signals - self.signal_means) / (self.signal_stds + 1e-8)
    
    def forward(self, signals, prev_signals=None, user_feedback=None):
        """
        Calculate reward from multiple grounded signals
        
        Args:
            signals: Tensor of shape [batch_size, num_signals] containing grounded signals
            prev_signals: Previous timestep signals for temporal consistency (optional)
            user_feedback: Optional feedback signal to adapt reward (optional)
            
        Returns:
            Scalar reward value
        """
        batch_size = signals.shape[0]
        
        # 1. Normalize signals
        norm_signals = self.normalize_signals(signals)
        
        # 2. Apply learned weights to signals
        weighted_signals = norm_signals * F.softmax(self.signal_weights, dim=0)
        
        # 3. Calculate base reward using context-dependent network
        base_reward = self.context_reward(weighted_signals)
        
        # 4. Apply reward shaping if previous signals are available
        shaping_term = 0
        if prev_signals is not None:
            norm_prev_signals = self.normalize_signals(prev_signals)
            # Combine current and previous signals
            temporal_input = torch.cat([norm_signals, norm_prev_signals], dim=1)
            # Calculate shaping term (promotes smooth improvement)
            shaping_term = self.reward_shaping(temporal_input) * self.shaping_weight
        
        # 5. Apply user feedback adaptation if available
        if user_feedback is not None:
            # Adjust reward based on user feedback
            # Simple implementation: scale reward by feedback
            base_reward = base_reward * (1.0 + 0.1 * user_feedback)
        
        # 6. Combine all reward components
        total_reward = base_reward + shaping_term
        
        return total_reward
    
    def update_from_feedback(self, signals, rewards, feedback, learning_rate=0.01):
        """
        Update the reward function based on feedback
        
        Args:
            signals: Tensor of signals from past episodes
            rewards: Rewards predicted for those signals
            feedback: Human feedback on those episodes (higher = better)
            learning_rate: Rate of adaptation
        """
        # Simple update rule: adjust weights based on correlation with feedback
        with torch.no_grad():
            # Calculate correlation between each signal and feedback
            norm_signals = self.normalize_signals(signals)
            for i in range(len(self.signal_weights)):
                correlation = torch.mean(norm_signals[:, i] * feedback)
                # Increase weight for positively correlated signals
                self.signal_weights[i] += learning_rate * correlation
            
            # Re-normalize weights
            self.signal_weights.data = F.softmax(self.signal_weights, dim=0)


class BilevelRewardOptimizer:
    """
    Implements the bi-level optimization for reward learning.
    
    This follows the Era of Experience approach where the reward function
    is optimized to:
    1. Encourage expert-like behavior
    2. Adapt based on environment feedback
    """
    def __init__(self, reward_function, policy_model, world_model):
        self.reward_function = reward_function
        self.policy_model = policy_model
        self.world_model = world_model
        
        # Optimizer for reward function
        self.reward_optimizer = torch.optim.Adam(
            reward_function.parameters(), lr=0.001
        )
        
    def optimize_step(self, expert_trajectories, policy_trajectories, user_feedback=None):
        """
        Perform one step of bi-level optimization
        
        Args:
            expert_trajectories: Sample trajectories from expert demonstrations
            policy_trajectories: Sample trajectories from current policy
            user_feedback: Optional feedback on quality (higher = better)
        """
        # Extract signals from trajectories
        expert_signals = self._extract_signals(expert_trajectories)
        policy_signals = self._extract_signals(policy_trajectories)
        
        # Zero gradients
        self.reward_optimizer.zero_grad()
        
        # 1. Calculate rewards for expert and policy trajectories
        expert_rewards = self.reward_function(expert_signals)
        policy_rewards = self.reward_function(policy_signals)
        
        # 2. IRL loss: expert should get higher rewards than policy
        # The irl_margin ensures expert trajectories are sufficiently better
        irl_margin = 0.1
        irl_loss = F.relu(policy_rewards.mean() - expert_rewards.mean() + irl_margin)
        
        # 3. Regularization to prevent reward hacking
        entropy_reg = -torch.sum(F.softmax(self.reward_function.signal_weights, dim=0) * 
                                torch.log(F.softmax(self.reward_function.signal_weights, dim=0) + 1e-8))
        reg_loss = -0.1 * entropy_reg  # Encourage diversity in signal weights
        
        # 4. User feedback loss (if available)
        feedback_loss = 0
        if user_feedback is not None:
            # If we have user feedback on specific trajectories
            # Higher feedback should correspond to higher rewards
            combined_signals = torch.cat([expert_signals, policy_signals], dim=0)
            combined_rewards = torch.cat([expert_rewards, policy_rewards], dim=0)
            combined_feedback = user_feedback
            
            # Encourage correlation between rewards and feedback
            feedback_loss = -0.5 * (torch.sum(combined_rewards * combined_feedback) / 
                                   (torch.norm(combined_rewards) * torch.norm(combined_feedback) + 1e-8))
        
        # 5. Compute total loss and update
        total_loss = irl_loss + reg_loss + feedback_loss
        total_loss.backward()
        self.reward_optimizer.step()
        
        return {
            'irl_loss': irl_loss.item(),
            'reg_loss': reg_loss.item(),
            'feedback_loss': feedback_loss if isinstance(feedback_loss, float) else feedback_loss.item(),
            'total_loss': total_loss.item(),
            'expert_reward': expert_rewards.mean().item(),
            'policy_reward': policy_rewards.mean().item()
        }
    
    def _extract_signals(self, trajectories):
        """Extract relevant signals from trajectories"""
        # This would be implemented according to your specific data format
        # For example, extracting the motion metrics, risk scores, etc.
        return trajectories['signals']