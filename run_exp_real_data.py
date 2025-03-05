import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import glob
import json
import pandas as pd
import re
from tqdm import tqdm

from model import CausalGPT2ForFrameEmbeddings, RewardPredictor

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Configuration
CONTEXT_LENGTH = 5  # c_a
ANTICIPATION_LENGTH = 3  # l_a
BATCH_SIZE = 16
LEARNING_RATE = 0.001
EPOCHS = 20

# Step 1: Data Loading from CholecT50 Dataset
def load_cholect50_data(base_path, metadata_path=None, risk_score_path=None, max_videos=None,
                        frame_risk_agg='max'):
    """
    Load frame embeddings from the CholecT50 dataset.
    
    Args:
        base_path: Path to the folder containing video directories
        metadata_path: Path to metadata CSV file (if available)
        risk_score_path: Path to risk score JSON file (if available)
        max_videos: Maximum number of videos to load (for testing)
        
    Returns:
        List of dictionaries containing video data
    """
    print(f"Loading CholecT50 data from {base_path}")
    
    # Load metadata if available
    metadata = None
    if metadata_path and os.path.exists(metadata_path):
        print(f"Loading metadata from {metadata_path}")
        metadata = pd.read_csv(metadata_path)
        print(f"Metadata shape: {metadata.shape}")
    
    risk_score_root = "/home/maxboels/datasets/CholecT50/instructions/anticipation_5s_with_goals/"
    # Add the risk score for each frame to the metadata correctly
    video_ids_cache = []
    risk_column_name = f'risk_score_{frame_risk_agg}'
    if metadata is not None:
        for i, row in metadata.iterrows():
            video_id = row['video']
            frame_id = row['frame']

            # Load risk scores if available and is a new video in the metadata
            if risk_column_name not in metadata.columns:
                if video_id not in video_ids_cache:
                    print(f"Loading risk scores for video {video_id}")
                    video_ids_cache.append(video_id)
                    risk_scores = None
                    risk_score_path = risk_score_root + f"{video_id}_sorted_with_risk_scores_instructions_with_goals.json" 
                    if risk_score_path and os.path.exists(risk_score_path):
                        print(f"Loading risk scores from {risk_score_path}")
                        with open(risk_score_path, 'r') as f:
                            risk_scores = json.load(f)
                    else:
                        print(f"Risk score path not found, skipping")
                
                # Get risk score for this frame
                current_actions = risk_scores[str(frame_id)]['current_actions']
                frame_risk_scores = []
                for action in current_actions: # it's a list of dictionaries
                    frame_risk_scores.append(action['expert_risk_score'])
                if frame_risk_agg == 'mean':
                    risk_score = np.mean(frame_risk_scores)
                elif frame_risk_agg == 'max':
                    risk_score = np.max(frame_risk_scores)
                else:
                    print(f"Frame risk aggregation method {frame_risk_agg} not supported, skipping")
                # create new column if doesnt exist and add risk score
                # is it better to add it once at the end or during the loop?
                # answer: it is better to add it during the loop
                metadata.loc[i, risk_column_name] = risk_score

        # remove root from embedding path
        remove_root = '/nfs/home/mboels/projects/self-distilled-swin/outputs/embeddings_train_set/'
        if remove_root in metadata['embedding_path'][0]:
            metadata['embedding_path'] = metadata['embedding_path'].apply(lambda x: x.replace(remove_root, '/'))
        else:
            print(f"Root not found in embedding path, skipping")
        # save new version of metadata
        metadata.to_csv(metadata_path, index=False)
        print(f"Saved metadata with risk scores to {metadata_path}")

    # Find all videos in metadata csv file
    if metadata is not None:
        video_ids = metadata['video'].unique()  
    if max_videos:
        video_ids = video_ids[:max_videos]
    
    print(f"Found {len(video_dirs)} video directories")
    
    # Initialize data list
    data = []
    
    # Load frame embeddings for each video from the metadata
    for video_id in tqdm(video_ids, desc="Loading videos"):
        # Filter metadata for this video
        video_metadata = metadata[metadata['video'] == video_id]
        print(f"Found {len(video_metadata)} frames for video {video_id}")

        frame_files = video_metadata['embedding_path'].tolist()
        frame_files = [os.path.join(base_path, f) for f in frame_files]

        # Load frame embeddings
        frame_embeddings = []
        for frame_file in tqdm(frame_files, desc=f"Frames for {video_dir}", leave=False):
            embedding = np.load(frame_file)
            frame_embeddings.append(embedding)
        
        frame_embeddings = np.array(frame_embeddings)
        
        # Check embedding dimension
        embedding_dim = frame_embeddings.shape[1]
        print(f"Embedding dimension: {embedding_dim}")
        
        # For demonstration: Generate random action classes and survival time if metadata not available
        # In a real scenario, you would extract these from metadata
        num_frames = len(frame_embeddings)
        
        # Generate synthetic action classes and risk scores if not in metadata
        action_classes = np.random.randint(0, 100, size=num_frames)
        risk_scores = np.random.randint(1, 6, size=num_frames)
        
        # Random survival time between 50 and 200 weeks
        survival_time = np.random.randint(50, 201)
        
        # TODO: Extract actual action classes, risk scores, and outcomes from metadata if available
        
        # Store video data
        data.append({
            'video_id': video_id,
            'video_dir': video_dir,
            'frame_embeddings': frame_embeddings,
            'action_classes': action_classes,
            'risk_scores': risk_scores,
            'survival_time': survival_time,
            'num_frames': num_frames
        })
    
    if not data:
        raise ValueError("No valid videos loaded!")
        
    print(f"Successfully loaded {len(data)} videos")
    return data

# Custom dataset for frame prediction
class FramePredictionDataset(Dataset):
    def __init__(self, data):
        self.samples = []
        
        for video in data:
            embeddings = video['frame_embeddings']
            
            for i in range(len(embeddings) - 1):
                self.samples.append({
                    'input': embeddings[i],
                    'target': embeddings[i + 1]
                })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return torch.tensor(sample['input'], dtype=torch.float32), torch.tensor(sample['target'], dtype=torch.float32)

# Custom dataset for reward prediction
class RewardPredictionDataset(Dataset):
    def __init__(self, data, context_length=CONTEXT_LENGTH):
        self.samples = []
        self.context_length = context_length
        
        for video in data:
            embeddings = video['frame_embeddings']
            survival_time = video['survival_time']
            
            for i in range(context_length - 1, len(embeddings)):
                context = embeddings[i - (context_length - 1):i + 1]
                self.samples.append({
                    'context': np.array(context),
                    'target': survival_time
                })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return torch.tensor(sample['context'], dtype=torch.float32), torch.tensor(sample['target'], dtype=torch.float32)

# Action Policy Model with reward weighting
class ActionPolicyModel(nn.Module):
    def __init__(self, input_dim, context_length=CONTEXT_LENGTH, num_action_classes=100, hidden_dim=256):
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

# Training function for next frame predictor
def train_next_frame_model(model, data, device, epochs=EPOCHS):
    # Split data
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
    
    # Create datasets and dataloaders
    train_dataset = FramePredictionDataset(train_data)
    val_dataset = FramePredictionDataset(val_data)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} Training"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # For GPT2 model, reshape inputs to include sequence dimension
            inputs_seq = inputs.unsqueeze(1)  # [batch_size, 1, embedding_dim]
            targets_seq = targets.unsqueeze(1)  # [batch_size, 1, embedding_dim]
            
            # Forward pass
            outputs = model(inputs_seq, labels=targets_seq)
            loss = outputs["loss"]
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
        
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} Validation"):
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Add sequence dimension for model input
                inputs_seq = inputs.unsqueeze(1)  # [batch_size, 1, embedding_dim]
                
                # Generate next frame predictions
                future_embeddings = model.generate_future(
                    initial_embedding=inputs_seq,
                    length=1  # Generate one step ahead
                )
                
                # Extract predictions (remove sequence dim if necessary)
                predictions = future_embeddings.squeeze(1)  # [batch_size, embedding_dim]
                
                # Calculate loss between predictions and targets
                loss = criterion(predictions, targets)
                val_loss += loss.item() * inputs.size(0)
        
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    return train_losses, val_losses

# Training function for reward predictor
def train_reward_model(model, data, device, epochs=EPOCHS):
    # Split data
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
    
    # Create datasets and dataloaders
    train_dataset = RewardPredictionDataset(train_data)
    val_dataset = RewardPredictionDataset(val_data)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for contexts, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} Training"):
            contexts, targets = contexts.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(contexts)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * contexts.size(0)
        
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for contexts, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} Validation"):
                contexts, targets = contexts.to(device), targets.to(device)
                
                outputs = model(contexts)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item() * contexts.size(0)
        
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    return train_losses, val_losses

# Step 4: Estimate reward difference
def estimate_reward_difference(reward_model, frame_embeddings, next_frame_model, t, device):
    """Estimate the difference in expected rewards between current and future states."""
    # Get context embeddings (previous c_a frames)
    start_idx = max(0, t - CONTEXT_LENGTH + 1)
    context_embeddings = frame_embeddings[start_idx:t+1]
    
    # If context is shorter than expected, pad it
    if len(context_embeddings) < CONTEXT_LENGTH:
        padding = np.zeros((CONTEXT_LENGTH - len(context_embeddings), context_embeddings.shape[1]))
        context_embeddings = np.vstack([padding, context_embeddings])
    
    # Convert to tensor
    context_tensor = torch.tensor(context_embeddings, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Estimate current expected reward
    reward_model.eval()
    with torch.no_grad():
        current_reward = reward_model(context_tensor).item()
    
    # Generate future embeddings
    current_embedding = torch.tensor(frame_embeddings[t], dtype=torch.float32).to(device)
    future_embeddings = next_frame_model.predict_sequence(current_embedding, ANTICIPATION_LENGTH)
    
    # Convert future embeddings to numpy for easier handling
    future_np = torch.stack(future_embeddings).cpu().numpy()
    
    # Combine context with future for anticipated reward
    # Use only the most recent context frames plus future frames
    combined_embeddings = np.vstack([
        context_embeddings[-(CONTEXT_LENGTH - ANTICIPATION_LENGTH):] if CONTEXT_LENGTH > ANTICIPATION_LENGTH else [],
        future_np
    ])
    
    # Ensure we have the right context length
    if len(combined_embeddings) < CONTEXT_LENGTH:
        padding = np.zeros((CONTEXT_LENGTH - len(combined_embeddings), combined_embeddings.shape[1]))
        combined_embeddings = np.vstack([padding, combined_embeddings])
    elif len(combined_embeddings) > CONTEXT_LENGTH:
        combined_embeddings = combined_embeddings[-CONTEXT_LENGTH:]
    
    # Convert back to tensor
    combined_tensor = torch.tensor(combined_embeddings, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Estimate future expected reward
    with torch.no_grad():
        future_reward = reward_model(combined_tensor).item()
    
    # Return the difference
    return future_reward - current_reward

# Calculate action rewards - find which actions lead to positive reward differences
def calculate_action_rewards(data, next_frame_model, reward_model, device):
    """Calculate average reward difference for each action class."""
    print("Calculating action rewards...")
    
    action_rewards = {}
    action_counts = {}
    
    for video in tqdm(data, desc="Processing videos for action rewards"):
        frame_embeddings = video['frame_embeddings']
        action_classes = video['action_classes']
        
        for t in range(CONTEXT_LENGTH, len(frame_embeddings) - ANTICIPATION_LENGTH):
            # Calculate reward difference for this frame
            reward_diff = estimate_reward_difference(
                reward_model,
                frame_embeddings,
                next_frame_model,
                t,
                device
            )
            
            # Get action class for this frame
            action = int(action_classes[t])
            
            # Update running sum and count for this action
            if action not in action_rewards:
                action_rewards[action] = 0
                action_counts[action] = 0
            
            action_rewards[action] += reward_diff
            action_counts[action] += 1
    
    # Calculate average reward for each action
    avg_action_rewards = {}
    for action, total_reward in action_rewards.items():
        if action_counts[action] > 0:
            avg_action_rewards[action] = total_reward / action_counts[action]
        else:
            avg_action_rewards[action] = 0
    
    return avg_action_rewards

# Create a dataset for action policy training with weighted actions
class ActionPolicyDataset(Dataset):
    def __init__(self, data, action_weights, context_length=CONTEXT_LENGTH):
        self.samples = []
        self.context_length = context_length
        
        for video in data:
            embeddings = video['frame_embeddings']
            actions = video['action_classes']
            
            for i in range(context_length - 1, len(embeddings)):
                context = embeddings[i - (context_length - 1):i + 1]
                action = actions[i]
                
                # Get weight for this action (default to 1.0 if not found)
                weight = action_weights.get(int(action), 1.0)
                
                self.samples.append({
                    'context': np.array(context),
                    'action': int(action),
                    'weight': weight
                })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return (
            torch.tensor(sample['context'], dtype=torch.float32),
            torch.tensor(sample['action'], dtype=torch.long),
            torch.tensor(sample['weight'], dtype=torch.float32)
        )

# Train action policy model with reward weighting
def train_action_policy(data, action_weights, device, input_dim, num_action_classes=100, epochs=10):
    """Train a policy model that prioritizes high-reward actions."""
    print("Training action policy model with reward weighting...")
    
    # Calculate min and max rewards for normalization
    rewards = list(action_weights.values())
    min_reward = min(rewards)
    max_reward = max(rewards)
    reward_range = max_reward - min_reward
    
    # Function to normalize rewards to weights between 0.1 and 10
    def reward_to_weight(reward):
        if reward_range == 0:  # Avoid division by zero
            return 1.0
        normalized = (reward - min_reward) / reward_range
        return 0.1 + 9.9 * normalized  # Scale to 0.1-10 range
    
    # Convert rewards to weights
    normalized_weights = {action: reward_to_weight(reward) 
                         for action, reward in action_weights.items()}
    
    # Print top actions by weight
    print("Top 10 actions with highest weights:")
    top_actions = sorted(normalized_weights.items(), key=lambda x: x[1], reverse=True)[:10]
    for action, weight in top_actions:
        print(f"Action {action}: Weight {weight:.2f}")
    
    # Split data into train and validation sets
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
    
    # Create datasets and dataloaders
    train_dataset = ActionPolicyDataset(train_data, normalized_weights)
    val_dataset = ActionPolicyDataset(val_data, normalized_weights)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # Initialize model
    policy_model = ActionPolicyModel(input_dim, num_action_classes=num_action_classes).to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss(reduction='none')  # Use 'none' to apply sample weights
    optimizer = optim.Adam(policy_model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    losses = []
    
    for epoch in range(epochs):
        # Training
        policy_model.train()
        epoch_loss = 0.0
        
        for contexts, actions, weights in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            contexts = contexts.to(device)
            actions = actions.to(device)
            weights = weights.to(device)
            
            # Forward pass
            logits = policy_model(contexts)
            
            # Calculate loss and apply weights
            loss = criterion(logits, actions)
            weighted_loss = (loss * weights).mean()
            
            # Backward pass and optimize
            optimizer.zero_grad()
            weighted_loss.backward()
            optimizer.step()
            
            epoch_loss += weighted_loss.item() * contexts.size(0)
        
        epoch_loss /= len(train_loader.dataset)
        losses.append(epoch_loss)
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
    
    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs+1), losses)
    plt.title('Action Policy Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Weighted Loss')
    plt.grid(True)
    plt.savefig('action_policy_training_loss.png')
    
    return policy_model, normalized_weights

# Run TD-MPC2 algorithm on the validation dataset
def run_tdmpc(data, next_frame_model, reward_model, policy_model, action_weights, device):
    """Run TD-MPC2 algorithm on the validation dataset."""
    print("Running TD-MPC2...")
    
    results = []
    
    for video in tqdm(data, desc="Evaluating videos"):
        video_results = []
        frame_embeddings = video['frame_embeddings']
        original_actions = video['action_classes']
        
        for t in range(CONTEXT_LENGTH, len(frame_embeddings) - ANTICIPATION_LENGTH):
            # Get context window
            start_idx = max(0, t - CONTEXT_LENGTH + 1)
            context_embeddings = frame_embeddings[start_idx:t+1]
            
            # Pad if needed
            if len(context_embeddings) < CONTEXT_LENGTH:
                padding = np.zeros((CONTEXT_LENGTH - len(context_embeddings), context_embeddings.shape[1]))
                context_embeddings = np.vstack([padding, context_embeddings])
            
            # Convert to tensor
            context_tensor = torch.tensor(context_embeddings, dtype=torch.float32).unsqueeze(0).to(device)
            
            # Calculate reward difference with original action
            reward_diff = estimate_reward_difference(
                reward_model,
                frame_embeddings,
                next_frame_model,
                t,
                device
            )
            
            # Get model's recommended action
            with torch.no_grad():
                action_logits = policy_model(context_tensor)
                recommended_action = action_logits.argmax(dim=1).item()
            
            original_action = int(original_actions[t])
            
            # Add to results
            video_results.append({
                'frame_idx': t,
                'original_action': original_action,
                'recommended_action': recommended_action,
                'reward_difference': reward_diff,
                'original_action_weight': action_weights.get(original_action, 1.0),
                'recommended_action_weight': action_weights.get(recommended_action, 1.0)
            })
        
        results.append({
            'video_id': video['video_id'],
            'survival_time': video['survival_time'],
            'frame_results': video_results
        })
    
    return results

# Analyze and visualize results
def analyze_results(results, action_weights):
    """Analyze and visualize results of the experiment."""
    print("Analyzing results...")
    
    # Calculate statistics
    original_action_weights = []
    recommended_action_weights = []
    reward_diffs = []
    
    for video in results:
        for frame in video['frame_results']:
            original_action_weights.append(frame['original_action_weight'])
            recommended_action_weights.append(frame['recommended_action_weight'])
            reward_diffs.append(frame['reward_difference'])
    
    # Calculate average improvement
    avg_original_weight = np.mean(original_action_weights)
    avg_recommended_weight = np.mean(recommended_action_weights)
    avg_improvement = avg_recommended_weight - avg_original_weight
    percent_improvement = (avg_improvement / avg_original_weight) * 100
    
    print(f"Results Summary:")
    print(f"- Average original action weight: {avg_original_weight:.2f}")
    print(f"- Average recommended action weight: {avg_recommended_weight:.2f}")
    print(f"- Average improvement: {avg_improvement:.2f} ({percent_improvement:.2f}%)")
    print(f"- Average reward difference: {np.mean(reward_diffs):.2f}")
    
    # Visualize results
    plt.figure(figsize=(16, 12))
    
    # Plot 1: Histogram of original vs. recommended action weights
    plt.subplot(2, 2, 1)
    plt.hist(original_action_weights, bins=20, alpha=0.5, label='Original Actions')
    plt.hist(recommended_action_weights, bins=20, alpha=0.5, label='Recommended Actions')
    plt.title('Action Weight Distribution')
    plt.xlabel('Action Weight')
    plt.ylabel('Frequency')
    plt.legend()
    
    # Plot 2: Scatterplot of original vs. recommended weights
    plt.subplot(2, 2, 2)
    plt.scatter(original_action_weights, recommended_action_weights, alpha=0.3)
    plt.plot([0, 10], [0, 10], 'r--')  # Diagonal line
    plt.title('Original vs. Recommended Action Weights')
    plt.xlabel('Original Action Weight')
    plt.ylabel('Recommended Action Weight')
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    
    # Plot 3: Top 10 recommended actions
    recommended_actions = [frame['recommended_action'] for video in results for frame in video['frame_results']]
    action_counts = {}
    for action in recommended_actions:
        if action not in action_counts:
            action_counts[action] = 0
        action_counts[action] += 1
    
    top_actions = sorted(action_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    
    plt.subplot(2, 2, 3)
    plt.bar([f"Action {a[0]}" for a in top_actions], [a[1] for a in top_actions])
    plt.title('Top 10 Recommended Actions')
    plt.xlabel('Action ID')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    
    # Plot 4: Average reward differences by video
    video_reward_diffs = []
    for video in results:
        avg_video_diff = np.mean([frame['reward_difference'] for frame in video['frame_results']])
        video_reward_diffs.append((video['video_id'], avg_video_diff))
    
    video_reward_diffs.sort(key=lambda x: x[0])
    
    plt.subplot(2, 2, 4)
    plt.bar([f"VID{v[0]}" for v in video_reward_diffs], [v[1] for v in video_reward_diffs])
    plt.title('Average Reward Difference by Video')
    plt.xlabel('Video ID')
    plt.ylabel('Avg. Reward Difference')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('results_analysis.png')
    plt.show()
    
    return {
        'avg_original_weight': avg_original_weight,
        'avg_recommended_weight': avg_recommended_weight,
        'percent_improvement': percent_improvement,
        'avg_reward_diff': np.mean(reward_diffs)
    }

# Main function to run the experiment with CholecT50 data
def run_cholect50_experiment(gpt2_config, reward_config,
                            base_path, metadata_path=None, max_videos=None):
    """Run the experiment with CholecT50 data."""
    print("Starting CholecT50 experiment for surgical video analysis")
    
    # Set device (use GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Step 1: Load data
    data = load_cholect50_data(base_path, metadata_path, max_videos)
    
    # Get embedding dimension from the first video
    embedding_dim = data[0]['frame_embeddings'].shape[1]
    print(f"Embedding dimension: {embedding_dim}")
    
    # Get number of unique action classes
    all_actions = np.concatenate([video['action_classes'] for video in data])
    num_action_classes = int(max(all_actions)) + 1
    print(f"Number of action classes: {num_action_classes}")
    
    # Step 2: Pre-train next frame prediction model
    print("\nTraining next frame prediction model...")
    next_frame_predictor = CausalGPT2ForFrameEmbeddings(gpt2_config).to(device)
    train_next_frame_model(next_frame_predictor, data, device, epochs=1)  # Reduced epochs for demonstration

    # Step 3: Train reward prediction model
    print("\nTraining reward prediction model...")
    reward_model = RewardPredictor(**reward_config).to(device)
    train_reward_model(reward_model, data, device, epochs=5)  # Reduced epochs for demonstration
    
    # Calculate action rewards
    print("\nCalculating action rewards...")
    avg_action_rewards = calculate_action_rewards(data, next_frame_model, reward_model, device)
    
    # Train action policy model with reward weighting
    print("\nTraining action policy model...")
    policy_model, action_weights = train_action_policy(
        data, avg_action_rewards, device, embedding_dim, num_action_classes, epochs=5)
    
    # Run TD-MPC2 to evaluate the model
    print("\nRunning TD-MPC2...")
    results = run_tdmpc(data, next_frame_model, reward_model, policy_model, action_weights, device)
    
    # Analyze and visualize results
    print("\nAnalyzing results...")
    analysis = analyze_results(results, action_weights)
    
    return next_frame_model, reward_model, policy_model, action_weights, results, analysis

if __name__ == "__main__":
    # Set paths
    base_path = "/home/maxboels/datasets/CholecT50/embeddings_train_set/fold0"
    metadata_path = os.path.join(base_path, "embeddings_f0_swin_bas_129.csv")
    gpt2_config = {
        'hidden_dim': 768,  # GPT-2 hidden dimension
        'embedding_dim': 1024,  # GPT-2 embedding dimension
        'n_layer': 6  # Number of transformer layers
    }
    # Run the experiment with a subset of videos for faster execution
    max_videos = 5  # Set to None to use all videos

    reward_config = {
        'input_dim': 1024,
        'context_length': 5,
        'hidden_dim': 256,
        'num_heads': 4,
        'num_layers': 2
    }
    
    # Run the experiment
    next_frame_model, reward_model, policy_model, action_weights, results, analysis = run_cholect50_experiment(
        gpt2_config, reward_config, # config dictionaries
        base_path, metadata_path, max_videos, # others
    )
    
    print("\nExperiment completed!")
    print(f"Model performance: {analysis['percent_improvement']:.2f}% improvement in action quality")