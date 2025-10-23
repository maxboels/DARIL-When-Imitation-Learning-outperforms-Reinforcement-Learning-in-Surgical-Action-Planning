import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import os
import logging
import random
from collections import defaultdict
import matplotlib.pyplot as plt

class ValueNetwork(nn.Module):
    """
    Value network for estimating expected returns from surgical states.
    Used as a critic in TD-MPC and for reward/outcome prediction.
    """
    def __init__(
        self,
        state_dim=1024,   # Frame embedding dimension
        hidden_dim=256,   # Hidden dimension for value network
        num_layers=3,     # Number of hidden layers
        dropout=0.1       # Dropout rate
    ):
        super().__init__()
        
        layers = []
        current_dim = state_dim
        
        # Hidden layers
        for i in range(num_layers - 1):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        
        # Output layer (predicts a single value)
        layers.append(nn.Linear(current_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state):
        """
        Forward pass through the value network
        
        Args:
            state: State representation [batch_size, state_dim]
            
        Returns:
            Value estimate [batch_size, 1]
        """
        return self.network(state).squeeze(-1)


class ActionPolicyNetwork(nn.Module):
    """
    Policy network for selecting surgical actions based on state.
    Uses multi-label classification for selecting multiple actions.
    """
    def __init__(
        self,
        state_dim=1024,       # Frame embedding dimension
        action_dim=100,       # Number of possible action classes
        hidden_dim=256,       # Hidden dimension for policy network
        max_actions=3,        # Maximum number of actions to select
        temperature=1.0       # Temperature for sampling
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_actions = max_actions
        self.temperature = temperature
        
        # Network architecture
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, state):
        """
        Forward pass through the policy network
        
        Args:
            state: State representation [batch_size, state_dim]
            
        Returns:
            Action logits [batch_size, action_dim]
        """
        return self.network(state)
    
    def get_action_probs(self, state, training=False):
        """
        Get action probabilities
        
        Args:
            state: State representation [batch_size, state_dim]
            training: Whether in training mode
            
        Returns:
            Action probabilities [batch_size, action_dim]
        """
        logits = self.forward(state)
        
        # Apply temperature and return probabilities
        scaled_logits = logits / max(0.1, self.temperature)
        
        # Return probabilities
        return torch.sigmoid(scaled_logits)
    
    def sample_actions(self, state, training=False, deterministic=False):
        """
        Sample actions from policy
        
        Args:
            state: State representation [batch_size, state_dim]
            training: Whether in training mode
            deterministic: Whether to sample deterministically
            
        Returns:
            Binary action vector [batch_size, action_dim]
        """
        action_probs = self.get_action_probs(state, training=training)
        batch_size = action_probs.size(0)
        
        if deterministic:
            # Select top-k actions deterministically
            _, top_actions = torch.topk(action_probs, min(self.max_actions, self.action_dim), dim=1)
            actions = torch.zeros_like(action_probs)
            
            for b in range(batch_size):
                actions[b, top_actions[b]] = 1.0
        else:
            # Sample from Bernoulli distribution for each action
            # but ensure at least one and at most max_actions are selected
            actions = torch.zeros_like(action_probs)
            
            for b in range(batch_size):
                # Probabilistically sample actions
                probs = action_probs[b]
                sampled = torch.bernoulli(probs)
                
                # Ensure at least one action (if all zeros, pick highest prob)
                if sampled.sum() == 0:
                    top_action = torch.argmax(probs)
                    sampled[top_action] = 1.0
                
                # Ensure at most max_actions (if too many, keep top ones)
                if sampled.sum() > self.max_actions:
                    _, top_indices = torch.topk(probs * sampled, self.max_actions)
                    new_sampled = torch.zeros_like(sampled)
                    new_sampled[top_indices] = 1.0
                    sampled = new_sampled
                
                actions[b] = sampled
        
        return actions


class TDMPCSurgicalActionPolicy:
    """
    TD-MPC (Temporal Difference Model Predictive Control) policy for surgical actions.
    
    This implements an offline RL approach that uses a world model to simulate
    the effects of different actions and selects the best one according to a learned value function.
    """
    def __init__(
        self,
        world_model,             # Trained world model for predicting next states
        state_dim=1024,          # Frame embedding dimension
        action_dim=100,          # Number of possible action classes
        horizon=10,              # Planning horizon (number of steps to look ahead)
        num_samples=50,          # Number of action sequences to sample
        elite_ratio=0.1,         # Ratio of elite samples to keep
        temperature=1.0,         # Temperature for exploration
        discount=0.99,           # Discount factor for future rewards
        device='cuda'            # Device to run on
    ):
        self.world_model = world_model
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.num_samples = num_samples
        self.elite_ratio = elite_ratio
        self.num_elite = max(1, int(num_samples * elite_ratio))
        self.temperature = temperature
        self.discount = discount
        self.device = device
        
        # Initialize policy and value networks
        self.policy_net = ActionPolicyNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=256,
            temperature=temperature
        ).to(device)
        
        self.value_net = ValueNetwork(
            state_dim=state_dim,
            hidden_dim=256
        ).to(device)
        
        # Initialize target networks
        self.target_value_net = ValueNetwork(
            state_dim=state_dim,
            hidden_dim=256
        ).to(device)
        
        # Copy parameters to target network
        self.update_target_network(tau=1.0)
        
        # Initialize optimizers
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=3e-4)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=3e-4)
        
        # Action rewards cache (for storing estimated action rewards)
        self.action_rewards = defaultdict(float)
        self.action_counts = defaultdict(int)
        
        # For tracking training progress
        self.train_iterations = 0
    
    def update_target_network(self, tau=0.005):
        """
        Update target network with Polyak averaging
        
        Args:
            tau: Polyak averaging coefficient (1.0 for hard update)
        """
        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
    
    def plan_actions(self, current_state, tools=None, recognized_actions=None):
        """
        Plan actions using MPC (Model Predictive Control)
        
        Args:
            current_state: Current state representation [1, state_dim]
            tools: Current tools (optional) [1, tool_dim]
            recognized_actions: Recognized actions from vision model (optional) [1, action_dim]
            
        Returns:
            Selected actions, value of selected action, all candidate actions with values
        """
        self.policy_net.eval()
        self.value_net.eval()
        self.world_model.eval()
        
        with torch.no_grad():
            # Sample candidate actions from policy
            action_candidates = []
            
            # If recognized actions available, use them as one of the candidates
            if recognized_actions is not None:
                action_candidates.append(recognized_actions)
            
            # Sample additional actions from policy
            for _ in range(self.num_samples - len(action_candidates)):
                sampled_action = self.policy_net.sample_actions(
                    current_state, deterministic=False
                )
                action_candidates.append(sampled_action)
            
            # Stack all candidate actions
            action_candidates = torch.cat(action_candidates, dim=0)
            
            # Simulate trajectories for each candidate action
            state_values = []
            
            for i in range(len(action_candidates)):
                # Get candidate action
                action = action_candidates[i:i+1]
                
                # Prepare current tools for simulation
                if tools is not None:
                    current_tools = tools.expand(action.size(0), -1)
                else:
                    current_tools = None
                
                # Simulate trajectory using world model
                predicted_states = self.world_model.predict_next_frames(
                    frames=current_state,
                    actions=action,
                    tools=current_tools,
                    horizon=self.horizon
                )
                
                # Extract final state
                final_state = predicted_states[:, -1]
                
                # Evaluate trajectory value
                state_value = self.value_net(final_state).item()
                state_values.append(state_value)
            
            # Select action with highest value
            best_idx = np.argmax(state_values)
            best_action = action_candidates[best_idx]
            best_value = state_values[best_idx]
            
            # Return best action, its value, and all candidates with values
            return best_action, best_value, {
                "action_candidates": action_candidates.cpu().numpy(),
                "state_values": state_values
            }
    
    def train_step(self, batch, update_policy=True):
        """
        Perform a training step using offline data
        
        Args:
            batch: Batch of data containing states, actions, rewards, next_states
            update_policy: Whether to update policy network
            
        Returns:
            Dictionary of losses and metrics
        """
        states = batch['frames'].to(self.device)
        actions = batch['actions'].to(self.device)
        next_states = batch['next_frames'].to(self.device)
        rewards = batch['rewards'].to(self.device) if 'rewards' in batch else None
        
        # If rewards not provided, use a reward model or calculate surrogate rewards
        if rewards is None:
            # Use reward function that promotes key surgical milestones and safety
            rewards = self.calculate_surrogate_rewards(states, actions, next_states)
        
        # Update value network
        # Calculate target values using target network and rewards
        with torch.no_grad():
            next_values = self.target_value_net(next_states)
            target_values = rewards + self.discount * next_values
        
        # Calculate value loss
        predicted_values = self.value_net(states)
        value_loss = F.mse_loss(predicted_values, target_values)
        
        # Update value network
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
        
        # Update policy network
        policy_loss = torch.tensor(0.0, device=self.device)
        
        if update_policy:
            # Generate actions from policy
            action_logits = self.policy_net(states)
            
            # Calculate policy loss (maximizing Q-value)
            # For a multi-label setting we use sigmoid and binary cross-entropy
            # This is a form of behavior cloning with expert actions
            policy_loss = F.binary_cross_entropy_with_logits(action_logits, actions)
            
            # Update policy network
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()
        
        # Update target network with soft update
        self.update_target_network(tau=0.005)
        
        # Track training progress
        self.train_iterations += 1
        
        return {
            "value_loss": value_loss.item(),
            "policy_loss": policy_loss.item(),
            "mean_predicted_value": predicted_values.mean().item(),
            "mean_target_value": target_values.mean().item(),
            "mean_reward": rewards.mean().item()
        }
    
    def calculate_surrogate_rewards(self, states, actions, next_states):
        """
        Calculate surrogate rewards when true rewards are not available
        
        Args:
            states: Current states [batch_size, state_dim]
            actions: Actions taken [batch_size, action_dim]
            next_states: Next states [batch_size, state_dim]
            
        Returns:
            Rewards [batch_size]
        """
        batch_size = states.shape[0]
        rewards = torch.zeros(batch_size, device=self.device)
        
        # 1. Calculate progress in embedding space
        # Moving in a consistent direction in embedding space is rewarded
        # Use cosine similarity to measure progress
        delta_states = next_states - states
        
        # Normalize to get direction vectors
        state_norms = torch.norm(delta_states, dim=1, keepdim=True)
        state_directions = delta_states / (state_norms + 1e-8)
        
        # If we have a stored mean direction vector, calculate similarity
        if hasattr(self, 'mean_direction_vector'):
            direction_sim = F.cosine_similarity(
                state_directions, 
                self.mean_direction_vector.expand(batch_size, -1),
                dim=1
            )
            
            # Progress reward: +0.1 for consistent movement
            progress_reward = 0.1 * direction_sim
        else:
            # On first call, just initialize mean direction
            self.mean_direction_vector = torch.mean(state_directions, dim=0, keepdim=True)
            progress_reward = torch.zeros_like(rewards)
        
        # Update running mean of direction vector (slow update)
        with torch.no_grad():
            self.mean_direction_vector = 0.95 * self.mean_direction_vector + 0.05 * torch.mean(state_directions, dim=0, keepdim=True)
        
        # 2. Action rewards based on statistics from offline data
        action_reward = torch.zeros_like(rewards)
        
        # Convert actions to binary and compute indices
        binary_actions = (actions > 0.5).int()
        
        for i in range(batch_size):
            # Get indices of active actions
            active_indices = binary_actions[i].nonzero().squeeze(-1).cpu().tolist()
            
            # If single tensor, convert to list
            if isinstance(active_indices, int):
                active_indices = [active_indices]
            
            # Track action statistics and calculate reward
            action_r = 0.0
            
            for act_idx in active_indices:
                if act_idx < self.action_dim:
                    # Reward based on action frequency (rare actions might be more important)
                    self.action_counts[act_idx] += 1
                    
                    # Use cached reward if available, otherwise use a prior
                    if self.action_rewards[act_idx] != 0:
                        action_r += self.action_rewards[act_idx]
                    else:
                        # Prior: rare actions (high risk) are more important
                        # This can be replaced with expert knowledge or statistics
                        action_r += 0.05
            
            # Normalize by number of actions
            if active_indices:
                action_r /= len(active_indices)
            
            action_reward[i] = action_r
        
        # 3. Novelty reward for exploration (in embedding space)
        # Reward states that are different from recently seen states
        # This encourages exploration of diverse states
        if hasattr(self, 'recent_states'):
            # Calculate distance to recent states
            dists = torch.cdist(next_states, self.recent_states)
            min_dists, _ = torch.min(dists, dim=1)
            
            # Normalize distances
            norm_dists = min_dists / (torch.max(min_dists) + 1e-8)
            
            # Novelty reward: +0.05 for novel states
            novelty_reward = 0.05 * norm_dists
            
            # Update recent states buffer (FIFO)
            self.recent_states = torch.cat([
                self.recent_states[batch_size:], 
                next_states.detach()
            ], dim=0)
        else:
            # Initialize recent states buffer
            self.recent_states = next_states.detach()
            novelty_reward = torch.zeros_like(rewards)
        
        # Combine all reward components
        rewards = progress_reward + action_reward + novelty_reward
        
        return rewards
    
    def update_action_rewards(self, action_rewards):
        """
        Update the action rewards cache with new values
        
        Args:
            action_rewards: Dictionary mapping action indices to rewards
        """
        for action_idx, reward in action_rewards.items():
            self.action_rewards[action_idx] = reward
    
    def train(self, dataloader, num_epochs=10, log_interval=100, update_policy_interval=1):
        """
        Train the policy using offline data
        
        Args:
            dataloader: DataLoader providing batches of experience
            num_epochs: Number of epochs to train
            log_interval: How often to log progress
            update_policy_interval: How often to update policy (vs. only critic)
            
        Returns:
            Dictionary of training metrics
        """
        metrics = {
            "value_loss": [],
            "policy_loss": [],
            "mean_predicted_value": [],
            "mean_reward": []
        }
        
        total_steps = 0
        
        for epoch in range(num_epochs):
            epoch_metrics = defaultdict(list)
            
            for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
                # Update policy less frequently than critic for stability
                update_policy = (total_steps % update_policy_interval == 0)
                
                # Perform training step
                step_metrics = self.train_step(batch, update_policy=update_policy)
                
                # Track metrics
                for k, v in step_metrics.items():
                    epoch_metrics[k].append(v)
                
                # Log progress
                if batch_idx % log_interval == 0:
                    log_str = f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(dataloader)}"
                    for k, v in step_metrics.items():
                        log_str += f", {k}: {v:.4f}"
                    
                    print(log_str)
                
                total_steps += 1
            
            # Compute epoch averages
            for k, v in epoch_metrics.items():
                avg_metric = sum(v) / len(v)
                metrics[k].append(avg_metric)
                print(f"Epoch {epoch+1}/{num_epochs}, Average {k}: {avg_metric:.4f}")
        
        return metrics
    
    def save(self, save_dir):
        """
        Save policy, value network, and action rewards
        
        Args:
            save_dir: Directory to save models
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Save networks
        torch.save({
            'policy_state_dict': self.policy_net.state_dict(),
            'value_state_dict': self.value_net.state_dict(),
            'target_value_state_dict': self.target_value_net.state_dict(),
            'action_rewards': dict(self.action_rewards),
            'action_counts': dict(self.action_counts),
            'train_iterations': self.train_iterations
        }, os.path.join(save_dir, 'tdmpc_policy.pt'))
    
    def load(self, load_path):
        """
        Load policy, value network, and action rewards
        
        Args:
            load_path: Path to saved model
        """
        checkpoint = torch.load(load_path, map_location=self.device)
        
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.value_net.load_state_dict(checkpoint['value_state_dict'])
        self.target_value_net.load_state_dict(checkpoint['target_value_state_dict'])
        
        self.action_rewards = defaultdict(float, checkpoint['action_rewards'])
        self.action_counts = defaultdict(int, checkpoint['action_counts'])
        self.train_iterations = checkpoint['train_iterations']


def prepare_offline_rl_dataset(video_data, world_model, value_model=None, device='cuda'):
    """
    Prepare an offline RL dataset from video data
    
    Args:
        video_data: List of video dictionaries with frame_embeddings and actions
        world_model: Trained world model for state representation
        value_model: Optional value model for reward calculation
        device: Device to use
        
    Returns:
        Dictionary of offline RL data (states, actions, next_states, rewards)
    """
    offline_data = {
        'frames': [],
        'actions': [],
        'next_frames': [],
        'rewards': []
    }
    
    for video in tqdm(video_data, desc="Preparing RL dataset"):
        frames = video['frame_embeddings']
        actions = video['actions_binaries']
        
        # Convert to tensors
        frames_tensor = torch.tensor(frames, dtype=torch.float32)
        actions_tensor = torch.tensor(actions, dtype=torch.float32)
        
        # For each frame (except the last one)
        for i in range(len(frames) - 1):
            offline_data['frames'].append(frames[i])
            offline_data['actions'].append(actions[i])
            offline_data['next_frames'].append(frames[i+1])
            
            # If value model provided, compute reward
            if value_model is not None:
                with torch.no_grad():
                    current_frame = torch.tensor(frames[i:i+1], dtype=torch.float32).to(device)
                    next_frame = torch.tensor(frames[i+1:i+2], dtype=torch.float32).to(device)
                    current_action = torch.tensor(actions[i:i+1], dtype=torch.float32).to(device)
                    
                    # Use the world model to get hidden state representation
                    outputs = world_model(current_frame, current_action)
                    current_state = outputs['hidden_states'][:, -1].cpu().numpy()
                    
                    outputs = world_model(next_frame, torch.tensor(actions[i+1:i+2]).to(device) if i+1 < len(actions) else None)
                    next_state = outputs['hidden_states'][:, -1].cpu().numpy()
                    
                    # Calculate value difference as reward
                    current_value = value_model(torch.tensor(current_state, dtype=torch.float32).to(device)).item()
                    next_value = value_model(torch.tensor(next_state, dtype=torch.float32).to(device)).item()
                    
                    # Reward is improvement in value
                    reward = next_value - current_value
                    offline_data['rewards'].append(reward)
            else:
                # If no value model, use zero rewards (will be replaced during training)
                offline_data['rewards'].append(0.0)
    
    # Convert to arrays
    for key in offline_data:
        offline_data[key] = np.array(offline_data[key])
    
    return offline_data


class OfflineRLDataset(torch.utils.data.Dataset):
    """Dataset for offline RL training"""
    
    def __init__(self, data, device='cuda'):
        self.frames = torch.tensor(data['frames'], dtype=torch.float32)
        self.actions = torch.tensor(data['actions'], dtype=torch.float32)
        self.next_frames = torch.tensor(data['next_frames'], dtype=torch.float32)
        self.rewards = torch.tensor(data['rewards'], dtype=torch.float32)
    
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx):
        return {
            'frames': self.frames[idx],
            'actions': self.actions[idx],
            'next_frames': self.next_frames[idx],
            'rewards': self.rewards[idx]
        }


def estimate_action_values(video_data, policy, world_model, horizon=10, device='cuda'):
    """
    Estimate the value of different actions based on policy rollouts
    
    Args:
        video_data: List of video dictionaries
        policy: Trained TDMPC policy
        world_model: Trained world model
        horizon: Planning horizon
        device: Device to use
        
    Returns:
        Dictionary mapping action indices to estimated values
    """
    action_values = defaultdict(list)
    
    # Set models to evaluation mode
    policy.policy_net.eval()
    policy.value_net.eval()
    world_model.eval()
    
    with torch.no_grad():
        for video in tqdm(video_data, desc="Estimating action values"):
            frames = video['frame_embeddings']
            actions = video['actions_binaries']
            
            for i in range(len(frames) - horizon):
                # Current frame and ground truth action
                current_frame = torch.tensor(frames[i:i+1], dtype=torch.float32).to(device)
                true_action = torch.tensor(actions[i:i+1], dtype=torch.float32).to(device)
                
                # Plan actions using policy
                _, _, candidates = policy.plan_actions(current_frame)
                
                candidate_actions = candidates['action_candidates']
                candidate_values = candidates['state_values']
                
                # For each action, update its estimated value
                for j, action in enumerate(candidate_actions):
                    # Find active actions
                    active_indices = np.where(action > 0.5)[1]
                    
                    # Update value for each active action
                    for act_idx in active_indices:
                        action_values[int(act_idx)].append(candidate_values[j])
    
    # Calculate average value for each action
    avg_action_values = {}
    for act_idx, values in action_values.items():
        avg_action_values[act_idx] = sum(values) / len(values) if values else 0.0
    
    return avg_action_values


def evaluate_policy(test_videos, policy, world_model, device='cuda'):
    """
    Evaluate policy performance against ground truth actions
    
    Args:
        test_videos: List of test video dictionaries
        policy: Trained policy
        world_model: Trained world model
        device: Device to use
        
    Returns:
        Dictionary of evaluation metrics
    """
    policy.policy_net.eval()
    policy.value_net.eval()
    world_model.eval()
    
    metrics = {
        'action_agreement': [],  # How often policy agrees with expert
        'value_improvement': [], # Value improvement over expert actions
        'frame_prediction_mse': []  # MSE of frame predictions
    }
    
    with torch.no_grad():
        for video in tqdm(test_videos, desc="Evaluating policy"):
            frames = video['frame_embeddings']
            expert_actions = video['actions_binaries']
            
            video_metrics = {
                'action_agreement': [],
                'value_improvement': [],
                'frame_prediction_mse': []
            }
            
            for i in range(len(frames) - 1):
                # Current frame and expert action
                current_frame = torch.tensor(frames[i:i+1], dtype=torch.float32).to(device)
                expert_action = torch.tensor(expert_actions[i:i+1], dtype=torch.float32).to(device)
                next_frame = torch.tensor(frames[i+1:i+1+1], dtype=torch.float32).to(device)
                
                # Plan action using policy
                policy_action, policy_value, _ = policy.plan_actions(current_frame)
                
                # Compare actions (agreement = Jaccard similarity for multi-label)
                policy_action_binary = policy_action > 0.5
                expert_action_binary = expert_action > 0.5
                
                intersection = torch.sum(policy_action_binary & expert_action_binary).item()
                union = torch.sum(policy_action_binary | expert_action_binary).item()
                
                if union > 0:
                    action_agreement = intersection / union
                else:
                    action_agreement = 1.0  # Both empty
                
                video_metrics['action_agreement'].append(action_agreement)
                
                # Evaluate value of expert action
                expert_outputs = world_model(current_frame, expert_action)
                expert_next_state = expert_outputs['predicted_frame_embeddings'][:, -1]
                expert_value = policy.value_net(expert_next_state).item()
                
                # Value improvement
                value_improvement = policy_value - expert_value
                video_metrics['value_improvement'].append(value_improvement)
                
                # Frame prediction MSE
                policy_outputs = world_model(current_frame, policy_action)
                policy_next_frame = policy_outputs['predicted_frame_embeddings'][:, -1]
                
                frame_mse = F.mse_loss(policy_next_frame, next_frame).item()
                video_metrics['frame_prediction_mse'].append(frame_mse)
            
            # Aggregate video metrics
            for key, values in video_metrics.items():
                if values:
                    metrics[key].extend(values)
    
    # Calculate final metrics
    results = {}
    for key, values in metrics.items():
        if values:
            results[f'mean_{key}'] = sum(values) / len(values)
            results[f'std_{key}'] = np.std(values)
    
    return results


def plot_policy_evaluation(results, save_path=None):
    """
    Plot policy evaluation results
    
    Args:
        results: Dictionary of evaluation results
        save_path: Path to save plot
    """
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    
    # Plot action agreement
    axs[0].hist(results.get('action_agreement', []), bins=20)
    axs[0].set_title('Action Agreement with Expert (Jaccard Similarity)')
    axs[0].set_xlabel('Agreement')
    axs[0].set_ylabel('Frequency')
    
    # Plot value improvement
    axs[1].hist(results.get('value_improvement', []), bins=20)
    axs[1].set_title('Value Improvement over Expert Actions')
    axs[1].axvline(x=0, color='r', linestyle='--')
    axs[1].set_xlabel('Value Improvement')
    axs[1].set_ylabel('Frequency')
    
    # Plot frame prediction MSE
    axs[2].hist(results.get('frame_prediction_mse', []), bins=20)
    axs[2].set_title('Frame Prediction MSE')
    axs[2].set_xlabel('MSE')
    axs[2].set_ylabel('Frequency')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser(description="Train and evaluate TDMPC surgical action policy")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"], help="Mode")
    parser.add_argument("--world_model", type=str, required=True, help="Path to trained world model")
    parser.add_argument("--policy", type=str, default=None, help="Path to saved policy for evaluation")
    args = parser.parse_args()
    
    # Load config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    
    # Set up logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(cfg['training']['log_dir'], "tdmpc.log"))
        ]
    )
    logger = logging.getLogger()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load world model
    from surgical_world_model import MultiLabelSurgicalWorldModel
    
    logger.info(f"Loading world model from {args.world_model}")
    checkpoint = torch.load(args.world_model, map_location=device)
    world_model = MultiLabelSurgicalWorldModel(
        frame_dim=cfg['models']['world_model']['embedding_dim'],
        action_dim=cfg['models']['world_model']['targets_dims']['_a'],
        tool_dim=cfg['models']['recognition']['transformer']['num_instrument_classes'],
        hidden_dim=cfg['models']['world_model']['hidden_dim'],
        n_layer=cfg['models']['world_model']['n_layer']
    ).to(device)
    world_model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load data
    from datasets import load_cholect50_data
    
    train_data = load_cholect50_data(cfg['data'], split='train', max_videos=cfg['experiment']['max_videos'])
    test_data = load_cholect50_data(cfg['data'], split='test', max_videos=cfg['experiment']['max_videos'])
    
    if args.mode == "train":
        # Initialize policy
        logger.info("Initializing TDMPC policy")
        policy = TDMPCSurgicalActionPolicy(
            world_model=world_model,
            state_dim=cfg['models']['world_model']['embedding_dim'],
            action_dim=cfg['models']['world_model']['targets_dims']['_a'],
            horizon=cfg['eval']['world_model']['max_horizon'],
            device=device
        )
        
        # Prepare offline RL dataset
        logger.info("Preparing offline RL dataset")
        offline_data = prepare_offline_rl_dataset(train_data, world_model, device=device)
        
        # Create dataset and dataloader
        dataset = OfflineRLDataset(offline_data)
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=cfg['training']['batch_size'], 
            shuffle=True
        )
        
        # Train policy
        logger.info("Starting policy training")
        metrics = policy.train(dataloader, num_epochs=5)
        
        # Save policy
        save_dir = os.path.join(cfg['training']['checkpoint_dir'], 'tdmpc')
        os.makedirs(save_dir, exist_ok=True)
        policy.save(save_dir)
        logger.info(f"Policy saved to {save_dir}")
        
        # Estimate action values
        logger.info("Estimating action values")
        action_values = estimate_action_values(train_data, policy, world_model, device=device)
        
        # Save action values
        import json
        with open(os.path.join(save_dir, 'action_values.json'), 'w') as f:
            json.dump(action_values, f, indent=2)
        
        logger.info(f"Action values saved to {save_dir}/action_values.json")
    
    elif args.mode == "eval":
        # Load policy
        if args.policy is None:
            logger.error("Policy path must be provided for evaluation")
            exit(1)
        
        logger.info(f"Loading policy from {args.policy}")
        policy = TDMPCSurgicalActionPolicy(
            world_model=world_model,
            state_dim=cfg['models']['world_model']['embedding_dim'],
            action_dim=cfg['models']['world_model']['targets_dims']['_a'],
            horizon=cfg['eval']['world_model']['max_horizon'],
            device=device
        )
        policy.load(args.policy)
        
        # Evaluate policy
        logger.info("Evaluating policy")
        results = evaluate_policy(test_data, policy, world_model, device=device)
        
        # Log results
        for key, value in results.items():
            logger.info(f"{key}: {value:.4f}")
        
        # Plot results
        logger.info("Generating plots")
        output_dir = os.path.join(cfg['training']['log_dir'], 'tdmpc_evaluation')
        os.makedirs(output_dir, exist_ok=True)
        plot_policy_evaluation(results, save_path=os.path.join(output_dir, 'policy_evaluation.png'))
        
        # Save results
        import json
        with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
            json.dump({k: float(v) for k, v in results.items()}, f, indent=2)
        
        logger.info(f"Evaluation results saved to {output_dir}/evaluation_results.json")