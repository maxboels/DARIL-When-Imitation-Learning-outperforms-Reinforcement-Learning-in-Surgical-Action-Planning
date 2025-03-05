import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import random
from tqdm import tqdm

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# Constants
NUM_VIDEOS = 40
FRAMES_PER_VIDEO = 100
EMBEDDING_DIM = 1024
NUM_ACTION_CLASSES = 100
CONTEXT_LENGTH = 5
ANTICIPATION_LENGTH = 3
HIDDEN_DIM = 256

# Synthetic data generation functions
def generate_synthetic_data(num_videos, frames_per_video, embedding_dim, num_action_classes):
    """Generate synthetic surgical video data."""
    data = []
    
    for i in range(num_videos):
        # Generate survival time between 10 and 200 weeks
        survival_time = np.random.randint(10, 200)
        
        # Generate frame embeddings, risk scores, and action classes
        frame_embeddings = np.random.randn(frames_per_video, embedding_dim) * 0.1
        risk_scores = np.random.randint(1, 6, size=frames_per_video)
        action_classes = np.random.randint(0, num_action_classes, size=frames_per_video)
        
        data.append({
            "video_id": i,
            "survival_time": survival_time,
            "frame_embeddings": frame_embeddings,
            "risk_scores": risk_scores,
            "action_classes": action_classes
        })
    
    return data

# Define models
class RewardModel(nn.Module):
    """Model to predict expected reward (survival time) based on frame embeddings."""
    def __init__(self, embedding_dim, context_length):
        super(RewardModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.context_length = context_length
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(embedding_dim * context_length, HIDDEN_DIM)
        self.fc2 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 2)
        self.fc3 = nn.Linear(HIDDEN_DIM // 2, 1)
        
    def forward(self, x):
        # x shape: [batch_size, context_length, embedding_dim]
        x = self.flatten(x)  # [batch_size, context_length * embedding_dim]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # Output is unbounded reward prediction
        
        # Constrain output to reasonable survival time (10-200 weeks)
        x = 10 + 190 * torch.sigmoid(x)
        return x

class FramePredictor(nn.Module):
    """Model to predict next frame embedding based on current frame."""
    def __init__(self, embedding_dim):
        super(FramePredictor, self).__init__()
        self.embedding_dim = embedding_dim
        
        self.fc1 = nn.Linear(embedding_dim, HIDDEN_DIM)
        self.fc2 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.fc3 = nn.Linear(HIDDEN_DIM, embedding_dim)
        
    def forward(self, x):
        # x shape: [batch_size, embedding_dim]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def generate_future(self, x, length):
        """Generate sequence of future embeddings."""
        futures = []
        current = x.clone()
        
        for _ in range(length):
            current = self(current)
            futures.append(current)
            
        return torch.stack(futures, dim=1)  # [batch_size, length, embedding_dim]

class ActionPolicyModel(nn.Module):
    """Policy model to recommend optimal actions at each frame."""
    def __init__(self, embedding_dim, context_length, num_action_classes):
        super(ActionPolicyModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.context_length = context_length
        self.num_action_classes = num_action_classes
        
        # Process context frames
        self.context_encoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=HIDDEN_DIM,
            num_layers=2,
            batch_first=True
        )
        
        # Action prediction head
        self.action_head = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, num_action_classes)
        )
    
    def forward(self, x):
        # x shape: [batch_size, context_length, embedding_dim]
        # Encode the context window
        context_encoded, _ = self.context_encoder(x)
        
        # Use the last output for action prediction
        last_hidden = context_encoded[:, -1]
        
        # Predict action logits
        action_logits = self.action_head(last_hidden)
        return action_logits

# Function to calculate reward differences for each action
def calculate_action_rewards(data, frame_predictor, reward_model):
    """Calculate reward differences for each action class."""
    action_rewards = defaultdict(list)
    
    for video in tqdm(data, desc="Calculating action rewards"):
        frame_embeddings = torch.tensor(video["frame_embeddings"], dtype=torch.float32)
        action_classes = video["action_classes"]
        
        for t in range(CONTEXT_LENGTH, len(frame_embeddings) - ANTICIPATION_LENGTH):
            # Get context window
            context = frame_embeddings[t-CONTEXT_LENGTH:t]
            context = context.unsqueeze(0)  # Add batch dimension
            
            # Current reward prediction
            current_reward = reward_model(context).item()
            
            # Generate future embeddings
            current_frame = frame_embeddings[t].unsqueeze(0)  # Add batch dimension
            future_embeddings = frame_predictor.generate_future(current_frame, ANTICIPATION_LENGTH)
            future_embeddings = future_embeddings.squeeze(0)  # Remove batch dimension
            
            # Combine context with future
            combined_context = torch.cat([
                context.squeeze(0)[ANTICIPATION_LENGTH:],
                future_embeddings
            ]).unsqueeze(0)  # Add batch dimension back
            
            # Future reward prediction
            future_reward = reward_model(combined_context).item()
            
            # Calculate reward difference
            reward_diff = future_reward - current_reward
            
            # Store reward difference for this action
            action_class = action_classes[t]
            action_rewards[action_class].append(reward_diff)
    
    # Calculate average reward difference for each action
    avg_action_rewards = {}
    for action, rewards in action_rewards.items():
        avg_action_rewards[action] = np.mean(rewards)
    
    return avg_action_rewards

# Function to train the policy model with action weighting
def train_action_policy(data, avg_action_rewards, frame_predictor, reward_model, epochs=10):
    """Train a policy model that prioritizes high-reward actions."""
    
    # Create training dataset
    X = []  # Context windows
    y = []  # Target actions
    weights = []  # Action weights based on rewards
    
    # Min-max normalize the action rewards to get weights between 0.1 and 10
    min_reward = min(avg_action_rewards.values())
    max_reward = max(avg_action_rewards.values())
    reward_range = max_reward - min_reward
    
    # Function to convert reward to weight (scale from 0.1 to 10)
    def reward_to_weight(reward):
        if reward_range == 0:  # Avoid division by zero
            return 1.0
        normalized = (reward - min_reward) / reward_range
        return 0.1 + 9.9 * normalized  # Scale to 0.1-10 range
    
    # Convert rewards to weights
    action_weights = {action: reward_to_weight(reward) 
                     for action, reward in avg_action_rewards.items()}
    
    # Create action weights dictionary for easy lookup
    print("Top 10 actions with highest weights:")
    top_actions = sorted(action_weights.items(), key=lambda x: x[1], reverse=True)[:10]
    for action, weight in top_actions:
        print(f"Action {action}: Weight {weight:.2f}")
    
    # Prepare training data
    for video in tqdm(data, desc="Preparing training data"):
        frame_embeddings = torch.tensor(video["frame_embeddings"], dtype=torch.float32)
        action_classes = video["action_classes"]
        
        for t in range(CONTEXT_LENGTH, len(frame_embeddings)):
            # Get context window
            context = frame_embeddings[t-CONTEXT_LENGTH:t]
            
            # Target action
            action = action_classes[t]
            
            # Weight for this action
            weight = action_weights.get(action, 1.0)  # Default to 1.0 if not found
            
            X.append(context.numpy())
            y.append(action)
            weights.append(weight)
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    weights = np.array(weights)
    
    # Create PyTorch dataset and dataloader
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    
    dataset = TensorDataset(X_tensor, y_tensor, weights_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Initialize policy model
    policy_model = ActionPolicyModel(EMBEDDING_DIM, CONTEXT_LENGTH, NUM_ACTION_CLASSES)
    
    # Define optimizer
    optimizer = optim.Adam(policy_model.parameters(), lr=0.001)
    
    # Training loop
    policy_model.train()
    losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_X, batch_y, batch_weights in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad()
            
            # Forward pass
            action_logits = policy_model(batch_X)
            
            # Apply weighted loss - multiply sample weight by loss
            # This gives more importance to high-reward actions
            loss = F.cross_entropy(action_logits, batch_y, reduction='none')
            weighted_loss = (loss * batch_weights).mean()
            
            # Backward pass and optimization
            weighted_loss.backward()
            optimizer.step()
            
            epoch_loss += weighted_loss.item()
        
        avg_epoch_loss = epoch_loss / len(dataloader)
        losses.append(avg_epoch_loss)
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_epoch_loss:.4f}")
    
    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs+1), losses)
    plt.xlabel('Epoch')
    plt.ylabel('Weighted Loss')
    plt.title('Training Loss')
    plt.grid(True)
    plt.savefig('policy_training_loss.png')
    
    return policy_model, action_weights

# Function to evaluate the policy model
def evaluate_policy(policy_model, test_data, action_weights, reward_model, frame_predictor):
    """Evaluate the trained policy model."""
    policy_model.eval()
    
    original_rewards = []
    policy_rewards = []
    
    for video in tqdm(test_data, desc="Evaluating policy"):
        frame_embeddings = torch.tensor(video["frame_embeddings"], dtype=torch.float32)
        original_actions = video["action_classes"]
        
        # Track rewards for this video
        video_original_rewards = []
        video_policy_rewards = []
        
        for t in range(CONTEXT_LENGTH, len(frame_embeddings) - ANTICIPATION_LENGTH):
            # Get context window
            context = frame_embeddings[t-CONTEXT_LENGTH:t].unsqueeze(0)  # Add batch dim
            
            # Original action and its reward
            original_action = original_actions[t]
            
            # Predict policy's recommended action
            with torch.no_grad():
                action_logits = policy_model(context)
                policy_action = action_logits.argmax(dim=1).item()
            
            # Calculate rewards for both actions
            # For original action
            future_with_original = predict_future_with_action(
                frame_embeddings[t].unsqueeze(0), 
                original_action, 
                frame_predictor, 
                ANTICIPATION_LENGTH
            )
            
            combined_original = torch.cat([
                context.squeeze(0)[ANTICIPATION_LENGTH:],
                future_with_original.squeeze(0)
            ]).unsqueeze(0)
            
            original_reward = reward_model(combined_original).item()
            
            # For policy's action
            future_with_policy = predict_future_with_action(
                frame_embeddings[t].unsqueeze(0), 
                policy_action, 
                frame_predictor, 
                ANTICIPATION_LENGTH
            )
            
            combined_policy = torch.cat([
                context.squeeze(0)[ANTICIPATION_LENGTH:],
                future_with_policy.squeeze(0)
            ]).unsqueeze(0)
            
            policy_reward = reward_model(combined_policy).item()
            
            video_original_rewards.append(original_reward)
            video_policy_rewards.append(policy_reward)
        
        # Store average rewards for this video
        if video_original_rewards:
            original_rewards.append(np.mean(video_original_rewards))
            policy_rewards.append(np.mean(video_policy_rewards))
    
    # Calculate overall statistics
    avg_original_reward = np.mean(original_rewards)
    avg_policy_reward = np.mean(policy_rewards)
    improvement = avg_policy_reward - avg_original_reward
    percent_improvement = (improvement / avg_original_reward) * 100
    
    print(f"Average Reward with Original Actions: {avg_original_reward:.2f} weeks")
    print(f"Average Reward with Policy Actions: {avg_policy_reward:.2f} weeks")
    print(f"Improvement: {improvement:.2f} weeks ({percent_improvement:.2f}%)")
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    
    # Video-by-video comparison
    plt.subplot(1, 2, 1)
    video_ids = range(len(original_rewards))
    width = 0.35
    plt.bar([i - width/2 for i in video_ids], original_rewards, width, label='Original')
    plt.bar([i + width/2 for i in video_ids], policy_rewards, width, label='Optimized')
    plt.xlabel('Video ID')
    plt.ylabel('Expected Survival (weeks)')
    plt.title('Reward Comparison by Video')
    plt.legend()
    
    # Overall comparison
    plt.subplot(1, 2, 2)
    plt.bar(['Original', 'Optimized'], [avg_original_reward, avg_policy_reward])
    plt.ylabel('Expected Survival (weeks)')
    plt.title(f'Average Reward\n{percent_improvement:.1f}% Improvement')
    
    plt.tight_layout()
    plt.savefig('policy_evaluation.png')
    
    return {
        'original_reward': avg_original_reward,
        'policy_reward': avg_policy_reward,
        'improvement': improvement,
        'percent_improvement': percent_improvement
    }

# Function to predict future embeddings with a specific action
def predict_future_with_action(current_embedding, action, frame_predictor, length):
    """
    Simple simulation of future embeddings given an action.
    In a real implementation, this would use a more sophisticated model.
    """
    # This is a synthetic implementation - in reality, you'd have a model
    # that predicts how an embedding evolves when a specific action is taken
    future_embeddings = frame_predictor.generate_future(current_embedding, length)
    
    # Here we're just simulating that different actions lead to different futures
    # In reality, this would be learned from the actual data
    action_factor = (action / NUM_ACTION_CLASSES) - 0.5  # Range: -0.5 to 0.5
    action_influence = 0.1 * action_factor  # Small influence factor
    
    # Apply action influence to future embeddings
    influenced_embeddings = future_embeddings * (1 + action_influence)
    
    return influenced_embeddings

# Analyze which actions are most beneficial
def analyze_optimal_actions(action_weights, avg_action_rewards):
    """Analyze which actions are most beneficial for patient outcomes."""
    # Sort actions by reward
    sorted_actions = sorted(avg_action_rewards.items(), key=lambda x: x[1], reverse=True)
    
    # Get top 10 and bottom 10 actions
    top_actions = sorted_actions[:10]
    bottom_actions = sorted_actions[-10:]
    
    # Plot top and bottom actions
    plt.figure(figsize=(12, 10))
    
    # Top actions
    plt.subplot(2, 1, 1)
    action_ids = [action for action, _ in top_actions]
    rewards = [reward for _, reward in top_actions]
    colors = ['#1a9850', '#66bd63', '#a6d96a', '#d9ef8b', '#fee08b', 
              '#fdae61', '#f46d43', '#d73027', '#a50026', '#800026']
    
    plt.bar(action_ids, rewards, color=colors)
    plt.xlabel('Action ID')
    plt.ylabel('Average Reward Difference (weeks)')
    plt.title('Top 10 Actions with Highest Positive Impact')
    plt.grid(axis='y', alpha=0.3)
    
    # Bottom actions
    plt.subplot(2, 1, 2)
    action_ids = [action for action, _ in bottom_actions]
    rewards = [reward for _, reward in bottom_actions]
    colors = colors[::-1]  # Reverse color order
    
    plt.bar(action_ids, rewards, color=colors)
    plt.xlabel('Action ID')
    plt.ylabel('Average Reward Difference (weeks)')
    plt.title('Top 10 Actions with Highest Negative Impact')
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('optimal_actions_analysis.png')
    
    return {
        'top_actions': top_actions,
        'bottom_actions': bottom_actions
    }

# Main function to run the entire pipeline
def train_optimal_action_model():
    """Run the entire training and evaluation pipeline."""
    print("===== OPTIMAL SURGICAL ACTION TRAINING =====")
    
    # 1. Generate synthetic data
    print("\nGenerating synthetic data...")
    data = generate_synthetic_data(NUM_VIDEOS, FRAMES_PER_VIDEO, EMBEDDING_DIM, NUM_ACTION_CLASSES)
    
    # Split into training and testing sets (80/20)
    split_idx = int(0.8 * len(data))
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    
    print(f"Training set: {len(train_data)} videos")
    print(f"Testing set: {len(test_data)} videos")
    
    # 2. Initialize models
    frame_predictor = FramePredictor(EMBEDDING_DIM)
    reward_model = RewardModel(EMBEDDING_DIM, CONTEXT_LENGTH)
    
    # 3. Calculate reward differences for each action
    print("\nCalculating action rewards...")
    avg_action_rewards = calculate_action_rewards(train_data, frame_predictor, reward_model)
    
    # 4. Train policy model with action weighting
    print("\nTraining action policy model with reward-based weighting...")
    policy_model, action_weights = train_action_policy(
        train_data, avg_action_rewards, frame_predictor, reward_model, epochs=5)
    
    # 5. Evaluate the policy model
    print("\nEvaluating trained policy model...")
    eval_results = evaluate_policy(
        policy_model, test_data, action_weights, reward_model, frame_predictor)
    
    # 6. Analyze optimal actions
    print("\nAnalyzing which actions are most beneficial...")
    action_analysis = analyze_optimal_actions(action_weights, avg_action_rewards)
    
    # Display top beneficial actions
    print("\nTop 5 Most Beneficial Actions:")
    for i, (action, reward) in enumerate(action_analysis['top_actions'][:5]):
        print(f"#{i+1}: Action {action} - Expected improvement: {reward:.2f} weeks")
    
    # Display top harmful actions
    print("\nTop 5 Most Harmful Actions:")
    for i, (action, reward) in enumerate(action_analysis['bottom_actions'][:5]):
        print(f"#{i+1}: Action {action} - Expected decline: {reward:.2f} weeks")
    
    print("\n===== TRAINING COMPLETE =====")
    print(f"Model achieves {eval_results['percent_improvement']:.2f}% improvement in expected survival time")
    
    return policy_model, action_weights, eval_results, action_analysis

if __name__ == "__main__":
    train_optimal_action_model()