import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Step 1: Data Preparation
# Configuration
NUM_VIDEOS = 40
NUM_FRAMES_PER_VIDEO = 1000  # Assuming each video has 1000 frames
EMBEDDING_DIM = 1024
NUM_ACTION_CLASSES = 100
CONTEXT_LENGTH = 5  # c_a
ANTICIPATION_LENGTH = 3  # l_a
BATCH_SIZE = 16
LEARNING_RATE = 0.001
EPOCHS = 20

# Generate synthetic data
def generate_synthetic_data():
    data = []
    
    for video_idx in range(NUM_VIDEOS):
        # Generate survival time (outcome) - between 10 and 200 weeks
        survival_time = np.random.randint(10, 201)
        
        # Generate frame embeddings for this video
        frame_embeddings = np.random.randn(NUM_FRAMES_PER_VIDEO, EMBEDDING_DIM) * 0.5  # Mean 0, std 0.5
        
        # Generate risk scores (1-5)
        risk_scores = np.random.randint(1, 6, size=NUM_FRAMES_PER_VIDEO)
        
        # Generate action classes
        action_classes = np.random.randint(0, NUM_ACTION_CLASSES, size=NUM_FRAMES_PER_VIDEO)
        
        data.append({
            'video_id': video_idx,
            'survival_time': survival_time,
            'frame_embeddings': frame_embeddings,
            'risk_scores': risk_scores,
            'action_classes': action_classes
        })
    
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

# Step 2: GPT-2 like model for next frame prediction
class NextFramePredictor(nn.Module):
    def __init__(self, input_dim=EMBEDDING_DIM, hidden_dim=512):
        super(NextFramePredictor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x):
        return self.model(x)
    
    def predict_sequence(self, initial_embedding, length=ANTICIPATION_LENGTH):
        """Generate a sequence of future embeddings"""
        sequence = [initial_embedding]
        current_embedding = initial_embedding
        
        with torch.no_grad():
            for _ in range(length):
                current_embedding = self.forward(current_embedding)
                sequence.append(current_embedding)
        
        return sequence[1:]  # Return without the initial embedding

# Step 3: Reward Prediction Model
class RewardPredictor(nn.Module):
    def __init__(self, input_dim=EMBEDDING_DIM, context_length=CONTEXT_LENGTH, hidden_dim=256):
        super(RewardPredictor, self).__init__()
        
        # LSTM to process sequence of frame embeddings
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True
        )
        
        # Fully connected layers for prediction
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)  # Single output for survival time
        )
    
    def forward(self, x):
        # x shape: [batch_size, context_length, embedding_dim]
        lstm_out, _ = self.lstm(x)
        
        # Take the last LSTM output
        last_output = lstm_out[:, -1, :]
        
        # Predict survival time
        out = self.fc(last_output)
        
        return out.squeeze()  # Remove singleton dimension

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
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
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
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
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
        
        for contexts, targets in train_loader:
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
            for contexts, targets in val_loader:
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
        padding = np.zeros((CONTEXT_LENGTH - len(context_embeddings), EMBEDDING_DIM))
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
    
    # Concatenate context and future embeddings
    combined_embeddings = np.vstack([context_embeddings, future_np])
    
    # If combined embeddings exceed expected length, trim it
    if len(combined_embeddings) > CONTEXT_LENGTH + ANTICIPATION_LENGTH:
        combined_embeddings = combined_embeddings[-CONTEXT_LENGTH - ANTICIPATION_LENGTH:]
    
    # Convert back to tensor
    combined_tensor = torch.tensor(combined_embeddings, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Estimate future expected reward
    with torch.no_grad():
        future_reward = reward_model(combined_tensor).item()
    
    # Return the difference
    return future_reward - current_reward

# Step 5: TD-MPC2 (Reinforcement Learning) - simplified version
def run_tdmpc(data, next_frame_model, reward_model, device):
    """Run TD-MPC2 algorithm on the validation dataset."""
    print("Running TD-MPC2...")
    
    results = []
    
    for video in data:
        video_results = []
        frame_embeddings = video['frame_embeddings']
        
        for t in range(CONTEXT_LENGTH, len(frame_embeddings) - ANTICIPATION_LENGTH):
            # Calculate reward difference at this timestep
            reward_diff = estimate_reward_difference(
                reward_model,
                frame_embeddings,
                next_frame_model,
                t,
                device
            )
            
            video_results.append({
                'frame_idx': t,
                'predicted_reward_difference': reward_diff,
                'actual_risk_score': video['risk_scores'][t],
                'action_class': video['action_classes'][t]
            })
        
        results.append({
            'video_id': video['video_id'],
            'survival_time': video['survival_time'],
            'frame_results': video_results
        })
    
    return results

# Analyze results
def analyze_results(results):
    """Analyze and visualize results of the experiment."""
    print("Analyzing results...")
    
    # Calculate average reward difference across all videos
    all_reward_diffs = []
    all_risk_scores = []
    
    for video in results:
        for frame in video['frame_results']:
            all_reward_diffs.append(frame['predicted_reward_difference'])
            all_risk_scores.append(frame['actual_risk_score'])
    
    avg_reward_diff = np.mean(all_reward_diffs)
    
    # Analyze correlation between risk scores and reward differences
    correlation = np.corrcoef(all_risk_scores, all_reward_diffs)[0, 1]
    
    print(f"Experiment Results:")
    print(f"- Number of videos: {len(results)}")
    print(f"- Average predicted reward difference: {avg_reward_diff:.2f}")
    print(f"- Correlation between risk scores and rewards: {correlation:.2f}")
    print(f"- Context length used: {CONTEXT_LENGTH}")
    print(f"- Anticipation length used: {ANTICIPATION_LENGTH}")
    
    # Visualize results
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Distribution of reward differences
    plt.subplot(1, 2, 1)
    plt.hist(all_reward_diffs, bins=30, alpha=0.7)
    plt.axvline(avg_reward_diff, color='r', linestyle='--', label=f'Mean: {avg_reward_diff:.2f}')
    plt.title('Distribution of Predicted Reward Differences')
    plt.xlabel('Reward Difference')
    plt.ylabel('Frequency')
    plt.legend()
    
    # Plot 2: Scatter plot of risk scores vs. reward differences
    plt.subplot(1, 2, 2)
    plt.scatter(all_risk_scores, all_reward_diffs, alpha=0.3)
    plt.title(f'Risk Scores vs. Reward Differences (Corr: {correlation:.2f})')
    plt.xlabel('Risk Score')
    plt.ylabel('Reward Difference')
    
    plt.tight_layout()
    plt.savefig('results_analysis.png')
    plt.show()

# Main experiment function
def run_experiment():
    """Run the entire experiment."""
    print("Starting synthetic data experiment for surgical video analysis")
    
    # Set device (use GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Step 1: Generate data
    print("Generating synthetic data...")
    data = generate_synthetic_data()
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
    
    # Step 2: Pre-train next frame prediction model
    print("\nTraining next frame prediction model...")
    next_frame_model = NextFramePredictor().to(device)
    train_next_frame_model(next_frame_model, train_data, device)
    
    # Step 3: Train reward prediction model
    print("\nTraining reward prediction model...")
    reward_model = RewardPredictor().to(device)
    train_reward_model(reward_model, train_data, device)
    
    # Step 4 & 5: Run TD-MPC2 which includes reward difference estimation
    results = run_tdmpc(val_data, next_frame_model, reward_model, device)
    
    # Analyze and visualize results
    analyze_results(results)
    
    return next_frame_model, reward_model, results

if __name__ == "__main__":
    next_frame_model, reward_model, results = run_experiment()