import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import defaultdict
import os
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

class SurgicalFeatureExtractor(nn.Module):
    """
    Extracts meaningful surgical features from frame embeddings.
    These features will be used by the reward function to evaluate surgical actions.
    """
    def __init__(self, embedding_dim, hidden_dim=256, feature_dim=64, n_layers=3, dropout=0.1):
        super(SurgicalFeatureExtractor, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim
        
        # Initial projection
        layers = [nn.Linear(embedding_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        
        # Hidden layers
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
        
        # Final projection to feature dimension
        layers.append(nn.Linear(hidden_dim, feature_dim))
        
        self.feature_network = nn.Sequential(*layers)
    
    def forward(self, embeddings):
        """
        Extract features from frame embeddings.
        
        Args:
            embeddings: Frame embeddings of shape [batch_size, seq_length, embedding_dim]
            
        Returns:
            Features of shape [batch_size, seq_length, feature_dim]
        """
        batch_size, seq_length, _ = embeddings.shape
        
        # Reshape for processing
        flat_embeddings = embeddings.reshape(-1, self.embedding_dim)
        
        # Extract features
        flat_features = self.feature_network(flat_embeddings)
        
        # Reshape back
        features = flat_features.reshape(batch_size, seq_length, self.feature_dim)
        
        return features

class RewardHead(nn.Module):
    """
    Maps extracted surgical features to scalar rewards.
    This module estimates the quality of surgical actions and states.
    """
    def __init__(self, feature_dim, hidden_dim=128, n_layers=2, dropout=0.1):
        super(RewardHead, self).__init__()
        
        # Network to process features and output scalar reward
        layers = [nn.Linear(feature_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
        
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.reward_network = nn.Sequential(*layers)
    
    def forward(self, features):
        """
        Calculate rewards from features.
        
        Args:
            features: Surgical features of shape [batch_size, seq_length, feature_dim]
            
        Returns:
            Rewards of shape [batch_size, seq_length, 1]
        """
        batch_size, seq_length, feature_dim = features.shape
        
        # Reshape for processing
        flat_features = features.reshape(-1, feature_dim)
        
        # Calculate rewards
        flat_rewards = self.reward_network(flat_features)
        
        # Reshape back
        rewards = flat_rewards.reshape(batch_size, seq_length, 1)
        
        return rewards

class RewardModel(nn.Module):
    """
    Complete reward model combining feature extraction and reward calculation.
    This model estimates the quality of surgical states and actions.
    """
    def __init__(self, embedding_dim, feature_dim=64, hidden_dim=256, dropout=0.1):
        super(RewardModel, self).__init__()
        
        self.feature_extractor = SurgicalFeatureExtractor(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            feature_dim=feature_dim,
            dropout=dropout
        )
        
        self.reward_head = RewardHead(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim // 2,
            dropout=dropout
        )
    
    def forward(self, embeddings):
        """
        Calculate rewards from embeddings.
        
        Args:
            embeddings: Frame embeddings of shape [batch_size, seq_length, embedding_dim]
            
        Returns:
            Dictionary containing:
            - rewards: Rewards of shape [batch_size, seq_length, 1]
            - features: Extracted features of shape [batch_size, seq_length, feature_dim]
        """
        # Extract features
        features = self.feature_extractor(embeddings)
        
        # Calculate rewards
        rewards = self.reward_head(features)
        
        return {
            'rewards': rewards,
            'features': features
        }
    
    def save(self, path):
        """Save the model."""
        torch.save({
            'feature_extractor': self.feature_extractor.state_dict(),
            'reward_head': self.reward_head.state_dict()
        }, path)
    
    @classmethod
    def load(cls, path, embedding_dim, feature_dim=64, hidden_dim=256, dropout=0.1, device='cuda'):
        """Load the model."""
        model = cls(
            embedding_dim=embedding_dim,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        
        checkpoint = torch.load(path, map_location=device)
        model.feature_extractor.load_state_dict(checkpoint['feature_extractor'])
        model.reward_head.load_state_dict(checkpoint['reward_head'])
        model.to(device)
        
        return model

class PolicyModel(nn.Module):
    """
    Policy model for learning optimal surgical actions.
    This model takes the current state and outputs a distribution over actions.
    """
    def __init__(self, embedding_dim, action_dim, hidden_dim=256, n_layers=3, dropout=0.1):
        super(PolicyModel, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Policy network
        layers = [nn.Linear(embedding_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
        
        self.backbone = nn.Sequential(*layers)
        
        # Action head for multi-label binary classification
        self.action_head = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, embeddings):
        """
        Forward pass to get action logits.
        
        Args:
            embeddings: Frame embeddings of shape [batch_size, seq_length, embedding_dim]
            
        Returns:
            Action logits of shape [batch_size, seq_length, action_dim]
        """
        batch_size, seq_length, _ = embeddings.shape
        
        # Reshape for processing
        flat_embeddings = embeddings.reshape(-1, self.embedding_dim)
        
        # Process through backbone
        flat_features = self.backbone(flat_embeddings)
        
        # Get action logits
        flat_action_logits = self.action_head(flat_features)
        
        # Reshape back
        action_logits = flat_action_logits.reshape(batch_size, seq_length, self.action_dim)
        
        return action_logits
    
    def sample_action(self, embeddings, temperature=1.0):
        """
        Sample actions from policy.
        
        Args:
            embeddings: Frame embeddings of shape [batch_size, seq_length, embedding_dim]
            temperature: Sampling temperature
            
        Returns:
            Sampled actions of shape [batch_size, seq_length, action_dim]
        """
        # Get action logits
        action_logits = self.forward(embeddings)
        
        # Apply temperature
        if temperature != 1.0:
            action_logits = action_logits / temperature
        
        # Sample actions (binary for each class)
        action_probs = torch.sigmoid(action_logits)
        actions = torch.bernoulli(action_probs)
        
        return actions
    
    def save(self, path):
        """Save the model."""
        torch.save(self.state_dict(), path)
    
    @classmethod
    def load(cls, path, embedding_dim, action_dim, hidden_dim=256, n_layers=3, dropout=0.1, device='cuda'):
        """Load the model."""
        model = cls(
            embedding_dim=embedding_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout
        )
        
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        
        return model

def train_reward_model(cfg, logger, model, train_loader, val_loader=None, device='cuda'):
    """
    Train the reward model using expert demonstrations.
    This initial reward function will be refined through bi-level optimization.
    
    Args:
        cfg: Configuration dictionary
        logger: Logger object
        model: RewardModel instance
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        device: Device to train on
        
    Returns:
        Path to the best saved model
    """
    # Extract config
    train_config = cfg['training']
    epochs = train_config['epochs']
    lr = train_config['lr']
    weight_decay = train_config.get('weight_decay', 0.01)
    save_dir = train_config.get('save_dir', 'saved_models/reward')
    eval_interval = train_config.get('eval_interval', 5)
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Setup optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Track best model
    best_val_loss = float('inf')
    best_model_path = None
    
    # Start training
    logger.info(f"Starting reward model training for {epochs} epochs")
    
    for epoch in range(epochs):
        # Training loop
        model.train()
        train_loss = 0.0
        start_time = time.time()
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            # Extract data
            states = batch['states'].to(device)
            expert_scores = batch['expert_scores'].to(device)
            
            # Forward pass
            outputs = model(states)
            predicted_rewards = outputs['rewards']
            
            # Compute loss (MSE between predicted rewards and expert scores)
            loss = F.mse_loss(predicted_rewards.squeeze(-1), expert_scores)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Log loss
            train_loss += loss.item()
        
        # Update scheduler
        scheduler.step()
        
        # Calculate average loss
        train_loss /= len(train_loader)
        epoch_time = time.time() - start_time
        
        logger.info(f"Epoch {epoch+1}/{epochs} completed in {epoch_time:.2f}s | Train loss: {train_loss:.4f}")
        
        # Validation
        if val_loader is not None and ((epoch + 1) % eval_interval == 0 or epoch == epochs - 1):
            val_loss = evaluate_reward_model(model, val_loader, device)
            logger.info(f"Validation loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = os.path.join(save_dir, f"reward_model_best_epoch_{epoch+1}.pt")
                model.save(best_model_path)
                logger.info(f"New best model saved at {best_model_path}")
        
        # Always save latest model
        latest_path = os.path.join(save_dir, "reward_model_latest.pt")
        model.save(latest_path)
    
    return best_model_path or latest_path

def evaluate_reward_model(model, data_loader, device='cuda'):
    """
    Evaluate the reward model on validation data.
    
    Args:
        model: RewardModel instance
        data_loader: DataLoader for validation data
        device: Device to evaluate on
        
    Returns:
        Average validation loss
    """
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        for batch in data_loader:
            # Extract data
            states = batch['states'].to(device)
            expert_scores = batch['expert_scores'].to(device)
            
            # Forward pass
            outputs = model(states)
            predicted_rewards = outputs['rewards']
            
            # Compute loss
            loss = F.mse_loss(predicted_rewards.squeeze(-1), expert_scores)
            
            # Log loss
            val_loss += loss.item()
    
    # Calculate average loss
    val_loss /= len(data_loader)
    
    return val_loss

def train_policy_model(cfg, logger, world_model, reward_model, train_loader, val_loader=None, device='cuda'):
    """
    Train a policy model to maximize the reward function.
    This follows the principles from "The Era of Experience" by learning from interactions.
    
    Args:
        cfg: Configuration dictionary
        logger: Logger object
        world_model: WorldModel instance
        reward_model: RewardModel instance
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        device: Device to train on
        
    Returns:
        Path to the best saved policy model
    """
    # Extract config
    train_config = cfg['training']
    epochs = train_config['epochs']
    lr = train_config['lr']
    weight_decay = train_config.get('weight_decay', 0.01)
    save_dir = train_config.get('save_dir', 'saved_models/policy')
    eval_interval = train_config.get('eval_interval', 5)
    rollout_horizon = train_config.get('rollout_horizon', 10)
    entropy_weight = train_config.get('entropy_weight', 0.01)
    
    # Create policy model
    policy_model = PolicyModel(
        embedding_dim=world_model.embedding_dim,
        action_dim=world_model.num_action_classes,
        hidden_dim=train_config.get('hidden_dim', 256),
        n_layers=train_config.get('n_layers', 3)
    ).to(device)
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Setup optimizer
    optimizer = optim.AdamW(policy_model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Set models to appropriate modes
    world_model.eval()  # World model is fixed
    reward_model.eval()  # Reward model is fixed
    
    # Track best model
    best_val_reward = float('-inf')
    best_model_path = None
    
    # Start training
    logger.info(f"Starting policy model training for {epochs} epochs")
    
    for epoch in range(epochs):
        # Training loop
        policy_model.train()
        train_stats = defaultdict(float)
        start_time = time.time()
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            # Extract initial states
            initial_states = batch['states'][:, 0:1].to(device)  # [batch_size, 1, embedding_dim]
            
            # Policy gradient optimization
            optimizer.zero_grad()
            
            # Sample trajectories using the policy and world model
            trajectories = []
            total_rewards = []
            entropies = []
            
            # Generate multiple trajectories for each initial state
            batch_size = initial_states.shape[0]
            
            for b in range(batch_size):
                # Get initial state for this batch item
                initial_state = initial_states[b:b+1]  # [1, 1, embedding_dim]
                
                # Sample initial action from policy
                action_logits = policy_model(initial_state)
                action_probs = torch.sigmoid(action_logits)
                initial_action = torch.bernoulli(action_probs)  # [1, 1, action_dim]
                
                # Track log probabilities and rewards for this trajectory
                log_probs = []
                rewards = []
                trajectory_entropy = []
                
                # Add log prob of initial action
                log_prob = (action_probs * initial_action + (1 - action_probs) * (1 - initial_action)).log()
                log_probs.append(log_prob)
                
                # Calculate entropy of initial action
                entropy = -(action_probs * action_probs.log() + (1 - action_probs) * (1 - action_probs).log()).sum(-1)
                trajectory_entropy.append(entropy)
                
                # Start with initial state and action
                current_state = initial_state
                current_action = initial_action
                
                # Generate trajectory
                states = [current_state]
                actions = [current_action]
                
                for t in range(rollout_horizon):
                    # Generate next state using world model
                    with torch.no_grad():
                        world_outputs = world_model(
                            current_state=current_state,
                            next_actions=current_action
                        )
                        next_state = world_outputs['_z_hat'].detach()
                    
                    # Sample next action from policy
                    action_logits = policy_model(next_state)
                    action_probs = torch.sigmoid(action_logits)
                    next_action = torch.bernoulli(action_probs)
                    
                    # Add log prob of action
                    log_prob = (action_probs * next_action + (1 - action_probs) * (1 - next_action)).log()
                    log_probs.append(log_prob)
                    
                    # Calculate entropy of action
                    entropy = -(action_probs * action_probs.log() + (1 - action_probs) * (1 - action_probs).log()).sum(-1)
                    trajectory_entropy.append(entropy)
                    
                    # Calculate reward for current state-action pair
                    with torch.no_grad():
                        reward_outputs = reward_model(current_state)
                        reward = reward_outputs['rewards'].detach()
                    
                    rewards.append(reward)
                    
                    # Store states and actions
                    states.append(next_state)
                    actions.append(next_action)
                    
                    # Update for next step
                    current_state = next_state
                    current_action = next_action
                
                # Calculate discounted rewards
                gamma = 0.99
                discounted_rewards = []
                R = 0
                
                for r in reversed(rewards):
                    R = r + gamma * R
                    discounted_rewards.insert(0, R)
                
                discounted_rewards = torch.cat(discounted_rewards, dim=1)
                log_probs = torch.cat(log_probs, dim=1)
                trajectory_entropy = torch.cat(trajectory_entropy, dim=1)
                
                # Store trajectory information
                trajectories.append({
                    'states': torch.cat(states, dim=1),
                    'actions': torch.cat(actions, dim=1),
                    'log_probs': log_probs,
                    'rewards': discounted_rewards,
                    'entropy': trajectory_entropy
                })
                
                # Calculate total reward for this trajectory
                total_reward = discounted_rewards.sum().item()
                total_rewards.append(total_reward)
                
                # Calculate total entropy for this trajectory
                total_entropy = trajectory_entropy.sum().item()
                entropies.append(total_entropy)
            
            # Calculate policy loss
            policy_loss = 0
            
            for trajectory in trajectories:
                # Policy gradient loss
                pg_loss = -(trajectory['log_probs'] * trajectory['rewards']).sum()
                
                # Entropy bonus (to encourage exploration)
                entropy_loss = -entropy_weight * trajectory['entropy'].sum()
                
                # Combine losses
                policy_loss += pg_loss + entropy_loss
            
            # Average loss across batch
            policy_loss /= batch_size
            
            # Backward pass
            policy_loss.backward()
            optimizer.step()
            
            # Log statistics
            train_stats['policy_loss'] += policy_loss.item()
            train_stats['avg_reward'] += sum(total_rewards) / batch_size
            train_stats['avg_entropy'] += sum(entropies) / batch_size
        
        # Update scheduler
        scheduler.step()
        
        # Calculate average statistics
        for key in train_stats:
            train_stats[key] /= len(train_loader)
        
        epoch_time = time.time() - start_time
        
        logger.info(f"Epoch {epoch+1}/{epochs} completed in {epoch_time:.2f}s | "
                  f"Policy loss: {train_stats['policy_loss']:.4f} | "
                  f"Avg reward: {train_stats['avg_reward']:.4f} | "
                  f"Avg entropy: {train_stats['avg_entropy']:.4f}")
        
        # Validation
        if val_loader is not None and ((epoch + 1) % eval_interval == 0 or epoch == epochs - 1):
            val_reward = evaluate_policy_model(
                policy_model=policy_model,
                world_model=world_model,
                reward_model=reward_model,
                data_loader=val_loader,
                rollout_horizon=rollout_horizon,
                device=device
            )
            
            logger.info(f"Validation reward: {val_reward:.4f}")
            
            # Save best model
            if val_reward > best_val_reward:
                best_val_reward = val_reward
                best_model_path = os.path.join(save_dir, f"policy_model_best_epoch_{epoch+1}.pt")
                policy_model.save(best_model_path)
                logger.info(f"New best model saved at {best_model_path}")
        
        # Always save latest model
        latest_path = os.path.join(save_dir, "policy_model_latest.pt")
        policy_model.save(latest_path)
    
    return best_model_path or latest_path

def evaluate_policy_model(policy_model, world_model, reward_model, data_loader, rollout_horizon=10, device='cuda'):
    """
    Evaluate the policy model by generating trajectories and calculating rewards.
    
    Args:
        policy_model: PolicyModel instance
        world_model: WorldModel instance
        reward_model: RewardModel instance
        data_loader: DataLoader for validation data
        rollout_horizon: Number of steps to roll out
        device: Device to evaluate on
        
    Returns:
        Average reward
    """
    policy_model.eval()
    world_model.eval()
    reward_model.eval()
    
    total_reward = 0.0
    num_trajectories = 0
    
    with torch.no_grad():
        for batch in data_loader:
            # Extract initial states
            initial_states = batch['states'][:, 0:1].to(device)  # [batch_size, 1, embedding_dim]
            batch_size = initial_states.shape[0]
            
            # Generate trajectories for each initial state
            for b in range(batch_size):
                # Get initial state for this batch item
                initial_state = initial_states[b:b+1]  # [1, 1, embedding_dim]
                
                # Sample initial action from policy
                action_logits = policy_model(initial_state)
                action_probs = torch.sigmoid(action_logits)
                initial_action = torch.bernoulli(action_probs)  # [1, 1, action_dim]
                
                # Start with initial state and action
                current_state = initial_state
                current_action = initial_action
                
                # Track total rewards
                trajectory_reward = 0.0
                
                # Generate trajectory
                for t in range(rollout_horizon):
                    # Calculate reward for current state
                    reward_outputs = reward_model(current_state)
                    reward = reward_outputs['rewards'].item()
                    trajectory_reward += reward
                    
                    # Generate next state using world model
                    world_outputs = world_model(
                        current_state=current_state,
                        next_actions=current_action
                    )
                    next_state = world_outputs['_z_hat']
                    
                    # Sample next action from policy
                    action_logits = policy_model(next_state)
                    action_probs = torch.sigmoid(action_logits)
                    next_action = torch.bernoulli(action_probs)
                    
                    # Update for next step
                    current_state = next_state
                    current_action = next_action
                
                # Track total reward
                total_reward += trajectory_reward
                num_trajectories += 1
    
    # Calculate average reward per trajectory
    avg_reward = total_reward / num_trajectories if num_trajectories > 0 else 0.0
    
    return avg_reward

def run_bi_level_optimization(cfg, logger, world_model, reward_model, policy_model, train_loader, val_loader=None, device='cuda'):
    """
    Run bi-level optimization to refine the reward model based on expert demonstrations
    while simultaneously learning a policy to maximize the reward.
    
    This implements the key concept from "The Era of Experience" where the reward function
    is learned and refined through interaction with the environment.
    
    Args:
        cfg: Configuration dictionary
        logger: Logger object
        world_model: WorldModel instance
        reward_model: Initial RewardModel instance
        policy_model: Initial PolicyModel instance
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        device: Device to train on
        
    Returns:
        Tuple of (refined reward model, optimized policy model)
    """
    # Extract config
    bi_level_config = cfg['bi_level_optimization']
    num_iterations = bi_level_config['num_iterations']
    reward_lr = bi_level_config.get('reward_lr', 1e-4)
    policy_lr = bi_level_config.get('policy_lr', 1e-4)
    weight_decay = bi_level_config.get('weight_decay', 0.01)
    save_dir = bi_level_config.get('save_dir', 'saved_models/bi_level')
    expert_weight = bi_level_config.get('expert_weight', 1.0)
    policy_weight = bi_level_config.get('policy_weight', 0.5)
    rollout_horizon = bi_level_config.get('rollout_horizon', 10)
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Setup optimizers
    reward_optimizer = optim.AdamW(reward_model.parameters(), lr=reward_lr, weight_decay=weight_decay)
    policy_optimizer = optim.AdamW(policy_model.parameters(), lr=policy_lr, weight_decay=weight_decay)
    
    # Set world model to eval mode (fixed)
    world_model.eval()
    
    # Track best models
    best_val_metric = float('-inf')
    best_reward_path = None
    best_policy_path = None
    
    # Initialize metrics tracking
    metrics_history = defaultdict(list)
    
    # Start bi-level optimization
    logger.info(f"Starting bi-level optimization for {num_iterations} iterations")
    
    for iteration in range(num_iterations):
        ##############################
        # Step 1: Train policy model #
        ##############################
        policy_model.train()
        reward_model.eval()  # Fix reward model while training policy
        
        policy_stats = defaultdict(float)
        
        # Train policy for one epoch
        for batch in tqdm(train_loader, desc=f"Iteration {iteration+1}/{num_iterations} - Policy Training"):
            # Extract initial states
            initial_states = batch['states'][:, 0:1].to(device)  # [batch_size, 1, embedding_dim]
            
            # Policy gradient optimization
            policy_optimizer.zero_grad()
            
            # Sample trajectories using the policy and world model
            trajectories = []
            total_rewards = []
            
            # Generate one trajectory for each initial state
            batch_size = initial_states.shape[0]
            
            for b in range(batch_size):
                # Get initial state for this batch item
                initial_state = initial_states[b:b+1]  # [1, 1, embedding_dim]
                
                # Sample initial action from policy
                with torch.set_grad_enabled(True):
                    action_logits = policy_model(initial_state)
                    action_probs = torch.sigmoid(action_logits)
                    initial_action = torch.bernoulli(action_probs)  # [1, 1, action_dim]
                
                # Track log probabilities and rewards for this trajectory
                log_probs = []
                rewards = []
                
                # Add log prob of initial action
                log_prob = (action_probs * initial_action + (1 - action_probs) * (1 - initial_action)).log()
                log_probs.append(log_prob)
                
                # Start with initial state and action
                current_state = initial_state
                current_action = initial_action
                
                # Generate trajectory
                states = [current_state]
                actions = [current_action]
                
                for t in range(rollout_horizon):
                    # Generate next state using world model
                    with torch.no_grad():
                        world_outputs = world_model(
                            current_state=current_state,
                            next_actions=current_action
                        )
                        next_state = world_outputs['_z_hat'].detach()
                    
                    # Sample next action from policy
                    action_logits = policy_model(next_state)
                    action_probs = torch.sigmoid(action_logits)
                    next_action = torch.bernoulli(action_probs)
                    
                    # Add log prob of action
                    log_prob = (action_probs * next_action + (1 - action_probs) * (1 - next_action)).log()
                    log_probs.append(log_prob)
                    
                    # Calculate reward for current state-action pair
                    with torch.no_grad():
                        reward_outputs = reward_model(current_state)
                        reward = reward_outputs['rewards'].detach()
                    
                    rewards.append(reward)
                    
                    # Store states and actions
                    states.append(next_state)
                    actions.append(next_action)
                    
                    # Update for next step
                    current_state = next_state
                    current_action = next_action
                
                # Calculate discounted rewards
                gamma = 0.99
                discounted_rewards = []
                R = 0
                
                for r in reversed(rewards):
                    R = r + gamma * R
                    discounted_rewards.insert(0, R)
                
                discounted_rewards = torch.cat(discounted_rewards, dim=1)
                log_probs = torch.cat(log_probs, dim=1)
                
                # Store trajectory information
                trajectories.append({
                    'states': torch.cat(states, dim=1),
                    'actions': torch.cat(actions, dim=1),
                    'log_probs': log_probs,
                    'rewards': discounted_rewards
                })
                
                # Calculate total reward for this trajectory
                total_reward = discounted_rewards.sum().item()
                total_rewards.append(total_reward)
            
            # Calculate policy loss
            policy_loss = 0
            
            for trajectory in trajectories:
                # Policy gradient loss
                pg_loss = -(trajectory['log_probs'] * trajectory['rewards']).sum()
                policy_loss += pg_loss
            
            # Average loss across batch
            policy_loss /= batch_size
            
            # Backward pass and optimize
            policy_loss.backward()
            policy_optimizer.step()
            
            # Log statistics
            policy_stats['policy_loss'] += policy_loss.item()
            policy_stats['avg_reward'] += sum(total_rewards) / batch_size
        
        # Calculate average policy statistics
        for key in policy_stats:
            policy_stats[key] /= len(train_loader)
            metrics_history[f"policy_{key}"].append(policy_stats[key])
        
        ###############################
        # Step 2: Update reward model #
        ###############################
        reward_model.train()
        policy_model.eval()  # Fix policy model while updating reward
        
        reward_stats = defaultdict(float)
        
        # Update reward model for one epoch
        for batch in tqdm(train_loader, desc=f"Iteration {iteration+1}/{num_iterations} - Reward Update"):
            # Extract data
            expert_states = batch['states'].to(device)
            expert_scores = batch['expert_scores'].to(device)
            
            # Get initial states
            initial_states = expert_states[:, 0:1]  # [batch_size, 1, embedding_dim]
            
            # Zero gradients
            reward_optimizer.zero_grad()
            
            # Expert demonstration loss
            expert_outputs = reward_model(expert_states)
            expert_predicted_rewards = expert_outputs['rewards']
            expert_loss = F.mse_loss(expert_predicted_rewards.squeeze(-1), expert_scores)
            
            # Generate policy trajectories
            policy_states = []
            batch_size = initial_states.shape[0]
            
            with torch.no_grad():
                for b in range(min(batch_size, 8)):  # Limit number of trajectories for efficiency
                    # Get initial state for this batch item
                    initial_state = initial_states[b:b+1]  # [1, 1, embedding_dim]
                    
                    # Generate trajectory using policy
                    current_state = initial_state
                    states = [current_state]
                    
                    for t in range(rollout_horizon):
                        # Sample action from policy
                        action_logits = policy_model(current_state)
                        action_probs = torch.sigmoid(action_logits)
                        action = torch.bernoulli(action_probs)
                        
                        # Generate next state
                        world_outputs = world_model(
                            current_state=current_state,
                            next_actions=action
                        )
                        next_state = world_outputs['_z_hat']
                        
                        # Store state
                        states.append(next_state)
                        
                        # Update for next step
                        current_state = next_state
                    
                    # Concatenate states
                    trajectory_states = torch.cat(states, dim=1)  # [1, rollout_horizon+1, embedding_dim]
                    policy_states.append(trajectory_states)
            
            # If we have policy trajectories, calculate policy loss
            policy_loss = 0
            
            if policy_states:
                # Concatenate all trajectories
                policy_states = torch.cat(policy_states, dim=0)  # [batch_size, rollout_horizon+1, embedding_dim]
                
                # Calculate rewards for policy trajectories
                policy_outputs = reward_model(policy_states)
                policy_rewards = policy_outputs['rewards']
                
                # Calculate reward gradient constraint (to prevent reward hacking)
                # We want to encourage higher rewards for expert trajectories compared to policy trajectories
                policy_avg_reward = policy_rewards.mean()
                expert_avg_reward = expert_predicted_rewards.mean()
                
                # The reward gap should be positive (expert should get higher rewards)
                reward_gap = expert_avg_reward - policy_avg_reward
                policy_loss = F.relu(-reward_gap)  # Penalize if policy gets higher rewards than expert
            
            # Combine losses
            reward_loss = expert_weight * expert_loss + policy_weight * policy_loss
            
            # Backward pass and optimize
            reward_loss.backward()
            reward_optimizer.step()
            
            # Log statistics
            reward_stats['expert_loss'] += expert_loss.item()
            reward_stats['policy_loss'] += policy_loss.item()
            reward_stats['reward_loss'] += reward_loss.item()
            
            if policy_states:
                reward_stats['expert_avg_reward'] += expert_avg_reward.item()
                reward_stats['policy_avg_reward'] += policy_avg_reward.item()
                reward_stats['reward_gap'] += reward_gap.item()
        
        # Calculate average reward statistics
        for key in reward_stats:
            reward_stats[key] /= len(train_loader)
            metrics_history[f"reward_{key}"].append(reward_stats[key])
        
        # Log progress
        logger.info(f"Iteration {iteration+1}/{num_iterations} | "
                  f"Policy loss: {policy_stats['policy_loss']:.4f} | "
                  f"Avg reward: {policy_stats['avg_reward']:.4f} | "
                  f"Expert loss: {reward_stats['expert_loss']:.4f} | "
                  f"Reward gap: {reward_stats.get('reward_gap', 0):.4f}")
        
        # Validation (if provided)
        if val_loader is not None:
            # Evaluate policy on validation data
            val_reward = evaluate_policy_model(
                policy_model=policy_model,
                world_model=world_model,
                reward_model=reward_model,
                data_loader=val_loader,
                rollout_horizon=rollout_horizon,
                device=device
            )
            
            # Evaluate reward model on validation data
            val_expert_loss = evaluate_reward_model(reward_model, val_loader, device)
            
            # Combined validation metric (higher is better)
            val_metric = val_reward - val_expert_loss
            
            logger.info(f"Validation | Reward: {val_reward:.4f} | Expert loss: {val_expert_loss:.4f}")
            
            # Save best models
            if val_metric > best_val_metric:
                best_val_metric = val_metric
                
                # Save reward model
                best_reward_path = os.path.join(save_dir, f"reward_model_best_iter_{iteration+1}.pt")
                reward_model.save(best_reward_path)
                
                # Save policy model
                best_policy_path = os.path.join(save_dir, f"policy_model_best_iter_{iteration+1}.pt")
                policy_model.save(best_policy_path)
                
                logger.info(f"New best models saved at iteration {iteration+1}")
        
        # Always save latest models
        reward_model.save(os.path.join(save_dir, "reward_model_latest.pt"))
        policy_model.save(os.path.join(save_dir, "policy_model_latest.pt"))
        
        # Save metrics history
        torch.save(metrics_history, os.path.join(save_dir, "metrics_history.pt"))
        
        # Plot training curves
        plot_bi_level_curves(metrics_history, save_dir)
    
    # Return paths to best models
    return best_reward_path, best_policy_path

def plot_bi_level_curves(metrics_history, save_dir):
    """
    Plot training curves for bi-level optimization.
    
    Args:
        metrics_history: Dictionary of metrics history
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot policy metrics
    plt.figure(figsize=(10, 6))
    if 'policy_policy_loss' in metrics_history:
        plt.plot(metrics_history['policy_policy_loss'], label='Policy Loss')
    if 'policy_avg_reward' in metrics_history:
        plt.plot(metrics_history['policy_avg_reward'], label='Avg Reward')
    plt.xlabel('Iteration')
    plt.ylabel('Value')
    plt.title('Policy Training Metrics')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'policy_metrics.png'))
    plt.close()
    
    # Plot reward metrics
    plt.figure(figsize=(10, 6))
    if 'reward_expert_loss' in metrics_history:
        plt.plot(metrics_history['reward_expert_loss'], label='Expert Loss')
    if 'reward_policy_loss' in metrics_history:
        plt.plot(metrics_history['reward_policy_loss'], label='Policy Loss')
    if 'reward_reward_loss' in metrics_history:
        plt.plot(metrics_history['reward_reward_loss'], label='Total Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Reward Model Metrics')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'reward_metrics.png'))
    plt.close()
    
    # Plot reward values
    plt.figure(figsize=(10, 6))
    if 'reward_expert_avg_reward' in metrics_history:
        plt.plot(metrics_history['reward_expert_avg_reward'], label='Expert Reward')
    if 'reward_policy_avg_reward' in metrics_history:
        plt.plot(metrics_history['reward_policy_avg_reward'], label='Policy Reward')
    if 'reward_reward_gap' in metrics_history:
        plt.plot(metrics_history['reward_reward_gap'], label='Reward Gap')
    plt.xlabel('Iteration')
    plt.ylabel('Reward')
    plt.title('Reward Values')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'reward_values.png'))
    plt.close()