
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Training function for reward predictor
def train_reward_model(cfg, model, data, device):
    # Split data
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
    
    # Create datasets and dataloaders
    train_dataset = RewardPredictionDataset(train_data)
    val_dataset = RewardPredictionDataset(val_data)
    
    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg['batch_size'])
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg['learning_rate'])
    
    # Training loop
    train_losses = []
    val_losses = []
    
    for epoch in range(cfg['epochs']):
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
def estimate_reward_difference(cfg, reward_model, frame_embeddings, next_frame_model, t, device):
    context_length = cfg['context_length']
    ANTICIPATION_LENGTH = cfg['anticipation_length']
    """Estimate the difference in expected rewards between current and future states."""
    # Get context embeddings (previous c_a frames)
    start_idx = max(0, t - context_length + 1)
    context_embeddings = frame_embeddings[start_idx:t+1]
    
    # If context is shorter than expected, pad it
    if len(context_embeddings) < context_length:
        padding = np.zeros((context_length - len(context_embeddings), context_embeddings.shape[1]))
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
        context_embeddings[-(context_length - ANTICIPATION_LENGTH):] if context_length > ANTICIPATION_LENGTH else [],
        future_np
    ])
    
    # Ensure we have the right context length
    if len(combined_embeddings) < context_length:
        padding = np.zeros((context_length - len(combined_embeddings), combined_embeddings.shape[1]))
        combined_embeddings = np.vstack([padding, combined_embeddings])
    elif len(combined_embeddings) > context_length:
        combined_embeddings = combined_embeddings[-context_length:]
    
    # Convert back to tensor
    combined_tensor = torch.tensor(combined_embeddings, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Estimate future expected reward
    with torch.no_grad():
        future_reward = reward_model(combined_tensor).item()
    
    # Return the difference
    return future_reward - current_reward

# Calculate action rewards - find which actions lead to positive reward differences
def calculate_action_rewards(data, next_frame_model, reward_model, device, context_length=10, ANTICIPATION_LENGTH=5):
    """Calculate average reward difference for each action class."""
    print("Calculating action rewards...")
    
    action_rewards = {}
    action_counts = {}
    
    for video in tqdm(data, desc="Processing videos for action rewards"):
        frame_embeddings = video['frame_embeddings']
        actions_binaries = video['actions_binaries']
        
        for t in range(context_length, len(frame_embeddings) - ANTICIPATION_LENGTH):
            # Calculate reward difference for this frame
            reward_diff = estimate_reward_difference(cfg, reward_model, frame_embeddings, next_frame_model, t, device)
            
            # Get action class for this frame
            action = int(actions_binaries[t])
            
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


# Train action policy model with reward weighting
def train_action_policy(cfg, data, action_weights, device):
    input_dim = cfg['embedding_dim']
    num_action_classes = cfg['num_action_classes']
    num_epochs = cfg['epochs']
    BATCH_SIZE = cfg['batch_size']
    LEARNING_RATE = cfg['learning_rate']

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
        original_actions = video['actions_binaries']
        
        for t in range(context_length, len(frame_embeddings) - ANTICIPATION_LENGTH):
            # Get context window
            start_idx = max(0, t - context_length + 1)
            context_embeddings = frame_embeddings[start_idx:t+1]
            
            # Pad if needed
            if len(context_embeddings) < context_length:
                padding = np.zeros((context_length - len(context_embeddings), context_embeddings.shape[1]))
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
