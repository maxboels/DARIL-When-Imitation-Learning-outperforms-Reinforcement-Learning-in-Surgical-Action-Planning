"""
This script contains the implementation of a reinforcement learning policy
training and evaluation using a world model for action learning in surgical video analysis.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from datetime import datetime
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18
from torchvision.datasets import ImageFolder


def train_and_evaluate_rl_policy(cfg, logger, world_model, train_loader, test_video_loaders, device='cuda'):
    """
    Train an RL policy using the world model as environment and evaluate against auto-regressive.
    
    Args:
        cfg: Configuration dictionary
        logger: Logger instance
        world_model: Trained world model to use as environment
        train_loader: DataLoader for training data
        test_video_loaders: Dictionary of DataLoaders for test videos
        device: Device to train on
        
    Returns:
        Dictionary of results and trained policy model
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
    import os
    from datetime import datetime
    from tqdm import tqdm
    
    # Create timestamp for results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("results", f"rl_policy_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    logger.info(f"Training RL policy and comparing with auto-regressive, saving to {results_dir}")
    
    # 1. Create policy model
    state_dim = world_model.embedding_dim  # Adjust based on your model
    action_dim = 100  # Adjust based on your dataset (number of action classes)
    
    # Create policy network
    policy_model = PolicyNetwork(state_dim, action_dim).to(device)
    
    # 2. Create world model environment
    world_model_env = WorldModelEnv(
        world_model, 
        reward_weights={
            '_r_phase_completion': 1.0,
            '_r_phase_progression': 0.5,
            '_r_risk': -0.7,
            '_r_action_probability': 0.2,
            '_r_global_progression': 0.3
        },
        max_steps=50
    )
    
    # 3. Create training data sampler for initial states
    initial_state_sampler = InitialStateSampler(train_loader, device)
    
    # 4. Train policy with PPO
    logger.info("Training policy with PPO...")
    policy_trainer = PPOTrainer(
        policy_model,
        world_model_env,
        initial_state_sampler,
        device=device,
        learning_rate=0.0003,
        num_episodes=1000,
        gamma=0.99,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01
    )
    
    # Start training
    training_metrics = policy_trainer.train()
    
    # Save trained policy
    policy_path = os.path.join(results_dir, "policy_model.pt")
    torch.save(policy_model.state_dict(), policy_path)
    logger.info(f"Saved trained policy to {policy_path}")
    
    # 5. Evaluate policy on its own
    logger.info("Evaluating RL policy...")
    rl_metrics = evaluate_rl_policy(world_model_env, policy_model, test_video_loaders, device)
    
    # 6. Compare with auto-regressive prediction
    logger.info("Comparing RL policy with auto-regressive prediction...")
    comparison_metrics = compare_rl_and_autoregressive(
        cfg, world_model, policy_model, test_video_loaders, device, logger
    )
    
    # 7. Combine results
    results = {
        'training_metrics': training_metrics,
        'rl_evaluation': rl_metrics,
        'comparison': comparison_metrics
    }
    
    return results, policy_model

class InitialStateSampler:
    """Sample initial states from the training dataset."""
    def __init__(self, train_loader, device):
        self.train_loader = train_loader
        self.device = device
        self.batches = []
        self._cache_batches()
    
    def _cache_batches(self):
        # Cache a few batches to sample from
        for i, batch in enumerate(self.train_loader):
            if i >= 10:  # Cache 10 batches
                break
            self.batches.append(batch)
    
    def sample(self):
        """Sample a random initial state from the dataset."""
        if not self.batches:
            return None
        
        # Randomly select a batch
        batch = np.random.choice(self.batches)
        
        # Randomly select a sample from the batch
        batch_size = batch['current_states'].size(0)
        sample_idx = np.random.randint(0, batch_size)
        
        # Extract the state (last frame in context window)
        current_states = batch['current_states'][sample_idx:sample_idx+1].to(self.device)
        initial_state = current_states[:, -1]  # [1, embedding_dim]
        
        return initial_state

class PPOTrainer:
    """Trainer for Proximal Policy Optimization."""
    def __init__(self, policy_model, world_model_env, initial_state_sampler, device='cuda',
                 learning_rate=0.0003, num_episodes=1000, gamma=0.99, clip_epsilon=0.2,
                 value_coef=0.5, entropy_coef=0.01):
        
        self.policy_model = policy_model
        self.world_model_env = world_model_env
        self.initial_state_sampler = initial_state_sampler
        self.device = device
        
        # Hyperparameters
        self.learning_rate = learning_rate
        self.num_episodes = num_episodes
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
        # Create optimizer
        self.optimizer = optim.Adam(policy_model.parameters(), lr=learning_rate)
        
        # Track metrics
        self.metrics = {
            'episode_returns': [],
            'episode_lengths': [],
            'losses': []
        }
    
    def train(self):
        """Train the policy using PPO algorithm."""
        import torch.nn.functional as F
        from tqdm import tqdm
        
        self.policy_model.train()
        
        for episode in tqdm(range(self.num_episodes), desc="Training RL policy"):
            # Reset environment with sampled initial state
            initial_state = self.initial_state_sampler.sample()
            if initial_state is None:
                continue
            
            state = self.world_model_env.reset(initial_state)
            
            # Collect trajectory
            states = []
            actions = []
            rewards = []
            action_log_probs = []
            done = False
            
            episode_length = 0
            episode_return = 0
            
            while not done:
                # Get action from policy
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).to(self.device)
                    action_probs = self.policy_model(state_tensor)
                    action = torch.bernoulli(action_probs)
                    action_log_prob = (action * torch.log(action_probs + 1e-10) + 
                                     (1 - action) * torch.log(1 - action_probs + 1e-10)).sum(-1)
                
                # Take step in environment
                next_state, reward, done, _ = self.world_model_env.step(action)
                
                # Store trajectory data
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                action_log_probs.append(action_log_prob)
                
                # Update state
                state = next_state
                
                # Track episode stats
                episode_length += 1
                episode_return += reward
            
            # Track episode metrics
            self.metrics['episode_returns'].append(episode_return)
            self.metrics['episode_lengths'].append(episode_length)
            
            # Calculate returns
            returns = self.calculate_returns(rewards)
            
            # Update policy
            loss = self.update_policy(states, actions, action_log_probs, returns)
            self.metrics['losses'].append(loss)
            
            # Log progress
            if (episode + 1) % 100 == 0:
                recent_returns = self.metrics['episode_returns'][-100:]
                avg_return = sum(recent_returns) / len(recent_returns)
                recent_losses = self.metrics['losses'][-100:]
                avg_loss = sum(recent_losses) / len(recent_losses)
                print(f"Episode {episode+1}/{self.num_episodes} | "
                      f"Avg Return: {avg_return:.4f} | Avg Loss: {avg_loss:.4f}")
        
        return self.metrics
    
    def calculate_returns(self, rewards):
        """Calculate discounted returns."""
        import torch
        
        returns = []
        R = 0
        
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Normalize returns
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        return returns
    
    def update_policy(self, states, actions, old_log_probs, returns):
        """Update policy using PPO objective."""
        import torch
        import torch.nn.functional as F
        
        # Convert lists to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.cat(actions).to(self.device)
        old_log_probs = torch.cat(old_log_probs).detach()
        
        # Get new action probabilities
        action_probs = self.policy_model(states)
        new_log_probs = (actions * torch.log(action_probs + 1e-10) + 
                        (1 - actions) * torch.log(1 - action_probs + 1e-10)).sum(-1)
        
        # Calculate entropy (for exploration)
        entropy = -(action_probs * torch.log(action_probs + 1e-10) + 
                   (1 - action_probs) * torch.log(1 - action_probs + 1e-10)).mean()
        
        # Calculate ratio
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # Calculate surrogate losses
        surrogate1 = ratio * returns
        surrogate2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * returns
        policy_loss = -torch.min(surrogate1, surrogate2).mean()
        
        # Final loss with entropy bonus
        loss = policy_loss - self.entropy_coef * entropy
        
        # Update policy
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

class PolicyNetwork(nn.Module):
    """Policy network for RL-based action learning."""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(PolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Sigmoid()  # For binary actions
        )
        
    def forward(self, state):
        return self.network(state)
    
    def sample_action(self, state, deterministic=False):
        action_probs = self.forward(state)
        if deterministic:
            return (action_probs > 0.5).float()
        else:
            return torch.bernoulli(action_probs)

class WorldModelEnv:
    """Wrapper for the world model to use as an RL environment."""
    def __init__(self, world_model, reward_weights=None, max_steps=50):
        self.world_model = world_model
        self.world_model.eval()  # Set to evaluation mode
        self.max_steps = max_steps
        self.current_step = 0
        self.current_state = None
        
        # Default reward weights if not provided
        self.reward_weights = reward_weights or {
            '_r_phase_completion': 1.0,
            '_r_phase_progression': 0.5,
            '_r_risk': -0.7,  # Negative weight for risk (we want to minimize risk)
            '_r_action_probability': 0.2,
            '_r_global_progression': 0.3
        }
    
    def reset(self, initial_state=None):
        """Reset the environment with an optional initial state."""
        if initial_state is None:
            raise ValueError("Initial state must be provided")
        
        self.current_state = initial_state
        self.current_step = 0
        return self.current_state
    
    def step(self, action):
        """Take a step in the environment using the world model."""
        with torch.no_grad():
            # Format inputs for the world model
            current_state = self.current_state.unsqueeze(0)  # Add batch dimension if needed
            
            # Use the world model to predict the next state and rewards
            outputs = self.world_model(
                current_states=current_state,
                next_actions=action
            )
            
            # Extract next state prediction
            next_state = outputs['_z_hat'].squeeze(0)
            
            # Calculate rewards from the various reward components
            reward = 0
            for reward_key, weight in self.reward_weights.items():
                if 'head_outputs' in outputs and reward_key in outputs['head_outputs']:
                    reward_component = outputs['head_outputs'][reward_key].squeeze().item()
                    reward += weight * reward_component
            
            # Update current state
            self.current_state = next_state
            self.current_step += 1
            
            # Check if done
            done = self.current_step >= self.max_steps
            
            return next_state, reward, done, {}

def evaluate_rl_policy(world_model_env, policy_model, test_data_loaders, device):
    """
    Evaluate a reinforcement learning policy model on test data.
    
    Args:
        world_model_env: WorldModelEnv instance
        policy_model: PolicyNetwork instance
        test_data_loaders: Dictionary of DataLoaders for test videos
        device: Device to evaluate on
        
    Returns:
        Dictionary of evaluation metrics
    """
    import torch
    import numpy as np
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score
    
    policy_model.eval()
    
    # Initialize metrics
    metrics = {
        'accuracy': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'f1': 0.0,
        'per_video': {}
    }
    
    total_correct = 0
    total_samples = 0
    
    all_predictions = []
    all_ground_truth = []
    
    with torch.no_grad():
        for video_id, data_loader in test_data_loaders.items():
            video_predictions = []
            video_ground_truth = []
            
            for batch in data_loader:
                # Move data to device
                current_states = batch['current_states'].to(device)
                next_actions = batch['next_actions'].to(device)
                
                # Get final state from context window
                states = current_states[:, -1]  # [batch_size, embedding_dim]
                
                # Get actions from policy model
                action_probs = policy_model(states)
                predictions = (action_probs > 0.5).float()
                
                # Get ground truth (first action in sequence)
                ground_truth = next_actions[:, 0]
                
                # Store predictions and ground truth
                video_predictions.append(predictions.cpu().numpy())
                video_ground_truth.append(ground_truth.cpu().numpy())
            
            if video_predictions:
                # Concatenate all predictions and ground truth for this video
                video_predictions = np.vstack(video_predictions)
                video_ground_truth = np.vstack(video_ground_truth)
                
                # Calculate metrics for this video
                video_accuracy = np.mean(np.all(video_predictions == video_ground_truth, axis=1))
                
                precision, recall, f1, _ = precision_recall_fscore_support(
                    video_ground_truth.flatten(), video_predictions.flatten(), average='binary'
                )
                
                # Store metrics for this video
                metrics['per_video'][video_id] = {
                    'accuracy': float(video_accuracy),
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1': float(f1),
                    'num_samples': int(video_ground_truth.shape[0])
                }
                
                # Update totals
                all_predictions.append(video_predictions)
                all_ground_truth.append(video_ground_truth)
    
    # Calculate overall metrics
    if all_predictions:
        all_predictions = np.vstack(all_predictions)
        all_ground_truth = np.vstack(all_ground_truth)
        
        # Overall accuracy (exact match across all action classes)
        metrics['accuracy'] = float(np.mean(np.all(all_predictions == all_ground_truth, axis=1)))
        
        # Overall precision, recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_ground_truth.flatten(), all_predictions.flatten(), average='binary'
        )
        
        metrics['precision'] = float(precision)
        metrics['recall'] = float(recall)
        metrics['f1'] = float(f1)
    
    return metrics

# Example usage in main experiment function
def run_cholect50_experiment(cfg):
    """Run the experiment with CholecT50 data."""
    print("Starting CholecT50 experiment for surgical video analysis")
    
    # [Your existing code and data loading...]
    
    # Train the world model
    if cfg_exp['pretrain_world_model']['train']:       
        print("\n[WORLD MODEL] Training next frame prediction model...")
        world_model = WorldModel(**cfg['models']['world_model']).to(device)
        best_model_path = train_world_model(cfg, logger, world_model, train_loader, test_video_loaders, device=device)
        logger.info(f"[WORLD MODEL] Best model saved at: {best_model_path}")
    
    # Load the world model
    if cfg_exp['pretrain_world_model']['inference']:
        logger.info("\n[WORLD MODEL] Loading model for inference...")
        if best_model_path is None:
            best_model_path = cfg_exp['pretrain_world_model']['best_model_path']
            logger.info(f"[WORLD MODEL] Using best model from pre-existing path: {best_model_path}")
        checkpoint = torch.load(best_model_path)
        world_model = WorldModel(**cfg['models']['world_model']).to(device)
        world_model.load_state_dict(checkpoint['model_state_dict'])
        world_model.eval()
        
        # Run enhanced evaluation
        enhanced_results = enhanced_inference_evaluation(cfg, logger, world_model, test_video_loaders, device)
        logger.info("[WORLD MODEL] Enhanced evaluation completed!")
    
    # Train and evaluate RL policy
    if cfg_exp['train_action_policy']:
        logger.info("\n[RL POLICY] Training and evaluating reinforcement learning policy...")
        rl_results, policy_model = train_and_evaluate_rl_policy(
            cfg, logger, world_model, train_loader, test_video_loaders, device
        )
        logger.info("[RL POLICY] Training and evaluation completed!")
    
    # Return results
    return world_model, policy_model, enhanced_results, rl_results
