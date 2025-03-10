import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
import os
import glob
import json
import pandas as pd
import re
from tqdm import tqdm
import yaml
import os
from datetime import datetime

from model import CausalGPT2ForFrameEmbeddings, RewardPredictor

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Step 1: Data Loading from CholecT50 Dataset
def load_cholect50_data(cfg, split='train'):
    """
    Load frame embeddings from the CholecT50 dataset for training or validation.

    Returns:
        List of dictionaries containing video data
    """
    # extract paths from config
    paths_config = cfg['paths']
    data_dir = paths_config['data_dir']
    metadata_file = paths_config['metadata_file']
    video_global_outcome_file = paths_config['video_global_outcome_file'] # embeddings_f0_swin_bas_129_with_enhanced_global_metrics.csv
    fold = paths_config['fold']
    print(f"Loading CholecT50 data from {data_dir}")

    # Create metadata path
    split_folder = f"embeddings_{split}_set"    
    metadata_dir = os.path.join(data_dir, split_folder, f"fold{fold}")
    metadata_path = os.path.join(metadata_dir, metadata_file)
    print(f"Metadata path: {metadata_path}")
    
    # Load metadata if available
    metadata = None
    if metadata_path and os.path.exists(metadata_path):
        metadata = pd.read_csv(metadata_path)
        print(f"Metadata loaded with shape: {metadata.shape}")

    # Load video global outcomes if available
    video_global_outcomes = None
    if video_global_outcome_file:
        video_global_outcome_path = os.path.join(metadata_dir, video_global_outcome_file)
        video_global_outcomes = pd.read_csv(video_global_outcome_path)
        print(f"Video global outcomes loaded with shape: {video_global_outcomes.shape}")
    
    # Add correct video_global_outcomes row and with all columns to the metadata file using the video id as unique identifier
    # Here we need to add the video global outcomes to the metadata
    # however, we only have x rows for each video, so we need to find the video id for each frame and then
    # select the value from the other dataframe and pass it to the metadata
    if metadata is not None and video_global_outcomes is not None:
        columns_to_add = ["avg_risk", "max_risk", "risk_std", "critical_risk_events", "critical_risk_percentage", "global_outcome_score"]
        video_ids = metadata['video'].unique().tolist()
        for video_id in video_ids:
            video_global_outcome = video_global_outcomes[video_global_outcomes['video'] == video_id]
            for column in columns_to_add:
                metadata.loc[metadata['video'] == video_id, column] = video_global_outcome[column].values[0]
        print(f"Added video global outcomes to metadata")
        
    risk_score_root = "/home/maxboels/datasets/CholecT50/instructions/anticipation_5s_with_goals/"
    # Add the risk score for each frame to the metadata correctly
    video_ids_cache = []
    all_risk_scores = []
    risk_column_name = f'risk_score_{cfg['frame_risk_agg']}'
    if metadata is not None:
        for i, row in metadata.iterrows():
            video_id = row['video']
            frame_id = row['frame']

            # Add risk score to metadata if not already there or column has nan value
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
                if cfg['frame_risk_agg'] == 'mean':
                    risk_score = np.mean(frame_risk_scores)
                elif cfg['frame_risk_agg'] == 'max':
                    risk_score = np.max(frame_risk_scores)
                else:
                    print(f"Frame risk aggregation method {cfg['frame_risk_agg']} not supported, skipping")
                risk_score = float(risk_score)
                all_risk_scores.append(risk_score)

                # if last frame, add risk score to metadata
                if i == len(metadata) - 1:
                    metadata[risk_column_name] = all_risk_scores
                    print(f"Added risk scores to metadata")

        # remove root from embedding path
        remove_root_1 = f'/nfs/home/mboels/projects/self-distilled-swin/outputs/embeddings_{split}_set/fold0/'
        remove_root_2 = f'/nfs/home/mboels/projects/self-distilled-swin/outputs/embeddings_{split}_setfold0/'
        if remove_root_1 in metadata['embedding_path'][0]:
            metadata['embedding_path'] = metadata['embedding_path'].apply(lambda x: x.replace(remove_root_1, f'fold{fold}/'))
        elif remove_root_2 in metadata['embedding_path'][0]:
            metadata['embedding_path'] = metadata['embedding_path'].apply(lambda x: x.replace(remove_root_2, f'fold{fold}/'))
        elif f'/fold{fold}/' in metadata['embedding_path'][0]:
            metadata['embedding_path'] = metadata['embedding_path'].apply(lambda x: x.replace('/fold0/', 'fold0/'))
        else:
            print(f"Root not found in embedding path, skipping")
        # save new version of metadata
        metadata.to_csv(metadata_path, index=False)
        print(f"Saved metadata with risk scores to {metadata_path}")

    # Find all videos in metadata csv file
    if metadata is not None:
        video_ids = metadata['video'].unique().tolist()
    if cfg['max_videos']:
        video_ids = video_ids[:cfg['max_videos']]
    
    print(f"Found {len(video_ids)} video directories")
    
    # Initialize data list
    data = []
    video_durations = []
    all_videos_mean_risk_scores = []
    all_videos_mean_risk_score_durs = []
    
    # Load frame embeddings for each video from the metadata
    for video_id in tqdm(video_ids, desc="Loading videos"):
        # Filter metadata for this video
        video_metadata = metadata[metadata['video'] == video_id]
        print(f"Found {len(video_metadata)} frames for video {video_id}")

        frame_files = video_metadata['embedding_path'].tolist()

        # Load frame embeddings
        video_frame_embeddings = []
        for frame_file in tqdm(frame_files, desc=f"Frames for {video_id}"):
            embedding = np.load(os.path.join(data_dir, split_folder, frame_file))
            video_frame_embeddings.append(embedding)
        
        video_frame_embeddings = np.array(video_frame_embeddings)
        
        # Check embedding dimension
        embedding_dim = video_frame_embeddings.shape[1]
        print(f"Embedding dimension: {embedding_dim}")
        
        # For demonstration: Generate random action classes and survival time if metadata not available
        # In a real scenario, you would extract these from metadata
        num_frames = len(video_frame_embeddings)
        video_durations.append(num_frames)
        
        video_risk_scores = video_metadata[risk_column_name].values

        # indices from column 'tri0':'tri199'
        action_columns = [f'tri{i}' for i in range(0, 100)]
        video_action_classes = video_metadata[action_columns].values

        # Take the mean of the 
        video_mean_risk_score = np.mean(video_risk_scores)
        all_videos_mean_risk_scores.append(video_mean_risk_score)

        # Multiple average risk score by video length
        video_mean_risk_score_dur = video_mean_risk_score * num_frames
        all_videos_mean_risk_score_durs.append(video_mean_risk_score_dur)

        # Calculate survival time based on risk score
        m = 100 / 5 # 5 is the max risk score
        video_mean_survival_time = 100 - (video_mean_risk_score * m)

        # global outcome score
        columns_to_add = ["avg_risk", "max_risk", "risk_std", "critical_risk_events", "critical_risk_percentage", "global_outcome_score"]
        video_outcome_data = {}
        for column in columns_to_add:
            video_outcome_data[column] = video_metadata[column].values[0]

        # Store video data
        data.append({
            'video_id': video_id,
            'video_dir': os.path.join(data_dir, split_folder, video_id),
            'frame_embeddings': video_frame_embeddings,
            'actions_binaries': video_action_classes,
            'risk_scores': video_risk_scores,
            'video_mean_risk_score': video_mean_risk_score,
            'video_mean_risk_score_dur': video_mean_risk_score_dur,
            'survival_time': video_mean_survival_time,
            'num_frames': num_frames,
            **video_outcome_data        # unpack video_outcome_data
        })

    
    avg_video_duration = np.mean(video_durations)
    print(f"Average video duration: {avg_video_duration:.2f} frames")
    
    if not data:
        raise ValueError("No valid videos loaded!")
    
    # Normalize by average video duration
    all_videos_mean_risk_score_durs = np.array(all_videos_mean_risk_score_durs) / avg_video_duration
    
    print(f"All Videos Mean Risk Score: {np.mean(all_videos_mean_risk_scores):.2f}")
    print(f"Standard Deviation of All Videos Mean Risk Score: {np.std(all_videos_mean_risk_scores):.2f}")
    print(f"All Videos Risk Scores: {all_videos_mean_risk_scores}")
    print(f"All Videos Mean Risk Score Dur: {np.mean(all_videos_mean_risk_score_durs):.2f}")
    print(f"Standard Deviation of All Videos Mean Risk Score Dur: {np.std(all_videos_mean_risk_score_durs):.2f}")
    print(f"All Videos Risk Score Dur: {all_videos_mean_risk_score_durs}")
    print(f"Successfully loaded {len(data)} ({split}ing) videos")
    return data

# Custom dataset for frame prediction with sequence inputs
class NextFramePredictionDataset(Dataset):
    def __init__(self, cfg, data):
        """
        Initialize the dataset with sequences of frame embeddings
        
        Args:
            data: List of video dictionaries containing frame_embeddings and actions_binaries
            context_length: Number of previous frames to include in each input sequence
            padding_value: Value to use for padding when not enough previous frames exist
        """
        self.samples = []
        
        context_length = cfg.get('context_length', 10)
        padding_value = cfg.get('padding_value', 0.0)
        train_shift = cfg.get('train_shift', 1)

        for video in data:
            embeddings = video['frame_embeddings']
            actions = video['actions_binaries']
            
            num_actions = len(actions[0])
            embedding_dim = len(embeddings[0])

            for i in range(len(embeddings) - 1):
                # For position i, take context_length frames from the left (previous frames)
                # This means frames from (i-context_length+1) to i, inclusive
                z_seq = []
                _z_seq = []
                # a_seq: We already trained the vision backbone to recognise the actions
                _a_seq = []
                
                # Add previous frames, using padding if needed
                for j in range(i - context_length + 1, i + 1):
                    if j < 0:
                        # Padding for positions before the start of the video
                        z_seq.append([padding_value] * embedding_dim)
                    else:
                        z_seq.append(embeddings[j])
                
                # Add the shifted next frame and action
                for k in range(i - context_length + 1 + train_shift, i + 1 + train_shift):
                    if k < 0:
                        # Padding for positions before the start of the video
                        _z_seq.append([padding_value] * embedding_dim)
                        _a_seq.append([0] * num_actions)
                        print(f"Padding for frame position {k}/{len(embeddings)}")
                    elif k >= len(embeddings):
                        # Padding for positions after the end of the video
                        _z_seq.append([padding_value] * embedding_dim)
                        _a_seq.append([0] * num_actions)
                        print(f"Padding for position {k}/{len(embeddings)}")
                    else:
                        _z_seq.append(embeddings[k])
                        _a_seq.append(actions[k])

                # Add the sequence to the samples list
                self.samples.append({
                    'z': z_seq,     # Sequence of frames from (i-context_length+1) to i
                    '_z': _z_seq,   # Sequence of frames from (i-context_length+2) to i+1
                    '_a': _a_seq    # Sequence of actions from (i-context_length+2) to i+1
                })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]

        z = torch.tensor(sample['z'], dtype=torch.float32)  # Shape: [context_length, embedding_dim]
        _z = torch.tensor(sample['_z'], dtype=torch.float32)
        _a = torch.tensor(sample['_a'], dtype=torch.float32)

        return z, _z, _a

# Custom dataset for frame prediction
class FramePredictionDataset(Dataset):
    def __init__(self, data):
        self.samples = []

        for video in data:
            embeddings = video['frame_embeddings']
            actions = video['actions_binaries']
            
            for i in range(len(embeddings) - 1):
                self.samples.append({
                    'z': embeddings[i],
                    '_z': embeddings[i + 1],
                    '_a': actions[i + 1]
                })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]

        z = torch.tensor(sample['z'], dtype=torch.float32)
        _z = torch.tensor(sample['_z'], dtype=torch.float32)
        _a = torch.tensor(sample['_a'], dtype=torch.float32)

        return z, _z, _a

# Custom dataset for reward prediction
class RewardPredictionDataset(Dataset):
    def __init__(self, data, context_length=10):
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

# Training function for next frame predictor
def train_next_frame_model(cfg, model, data, val_data=None, device='cuda'):
    # Split data
    if val_data is None:
        train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
    else:
        train_data = data
    
    # Create datasets and dataloaders
    train_dataset = NextFramePredictionDataset(cfg['data'], train_data)
    val_dataset = NextFramePredictionDataset(cfg['data'], val_data)
    
    train_loader = DataLoader(train_dataset, batch_size=cfg['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg['training']['batch_size'])

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg['training']['learning_rate'])
    
    # Training loop
    today = datetime.now().strftime("%Y-%m-%d")
    writer = SummaryWriter(log_dir=cfg['training']['log_dir'] + f"/{today}")
    train_losses = []
    val_losses = []
    epochs = cfg['training']['epochs']
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0

        for batch_idx, (z_seq, _z_seq, _a_seq) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} Training")):
            z_seq, _z_seq, _a_seq = z_seq.to(device), _z_seq.to(device), _a_seq.to(device)

            assert z_seq[:, -1, :].equal(_z_seq[:, -2, :]), "Last frame of z must equal second-to-last frame of _z"
            
            # Forward pass
            outputs = model(z_seq, next_frame=_z_seq, next_actions=_a_seq)
            loss = outputs["_z_loss"]
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * z_seq.size(0)

            # Calculate global step using epoch and batch index
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Loss/World_Model_Train', loss.item(), global_step)
                
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        use_memory = cfg['eval']['world_model']['use_memory']
        horizon = cfg['eval']['world_model']['horizon']
        with torch.no_grad():
            for batch_idx, (z_seq, _z_seq, _a_seq) in enumerate(tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} Validation")):
                z_seq, _z_seq, _a_seq = z_seq.to(device), _z_seq.to(device), _a_seq.to(device)

                # New frame observed
                # We should already have seen the previous frames and stored them memory
                if use_memory:
                    z_seq = z_seq[:, -1:, :] # select current frame
                
                # Generate next frame predictions
                outputs = model.generate(z_seq, horizon=horizon)
                
                # Calculate loss between predictions and targets
                # We didnt pass the future embeddings in out dataloader
                _z_loss = F.mse_loss(outputs['_zs_hat'][:, 0, :], _z_seq[:, -1, :]) # Check if first frame isn't the input frame
                val_loss += _z_loss.item() * z_seq.size(0)

                if '_a_seq_hat' in outputs:
                    _a_loss = F.mse(outputs['_a_seq_hat'], _a_seq[:, :, :])
                    val_loss += _a_loss.item() * z_seq.size(0)

                    # Calculate accuracy with top-k actions over short time horizon to relax a bit the exact timing
                    topk = 5
                    _, topk_actions = outputs['_a_seq_hat'].topk(topk, dim=2)
                    topk_actions = topk_actions.squeeze(0)
                    correct = topk_actions.eq(_a_seq[:, -1, :].argmax(dim=1).unsqueeze(1).expand_as(topk_actions))
                    accuracy = correct.float().mean()
                    writer.add_scalar('Accuracy/World_Model_Action_Prediction', accuracy.item(), epoch * len(val_loader) + len(val_loader))
                
                # Calculate global step using epoch and batch index
                global_step = epoch * len(val_loader) + batch_idx
                writer.add_scalar('Loss/World_Model_Val', _z_loss.item(), global_step)
                 
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    return train_losses, val_losses

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

# Create a dataset for action policy training with weighted actions
class ActionPolicyDataset(Dataset):
    def __init__(self, data, action_weights, context_length=10):
        self.samples = []
        self.context_length = context_length
        
        for video in data:
            embeddings = video['frame_embeddings']
            actions = video['actions_binaries']
            
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
def run_cholect50_experiment(cfg):
    """Run the experiment with CholecT50 data."""
    print("Starting CholecT50 experiment for surgical video analysis")

    # Set outputs to None
    next_frame_model = None
    reward_model = None
    policy_model = None
    action_weights = None
    results = None
    analysis = None

    cfg_exp = cfg['experiment']
    print(f"Running experiment: {cfg_exp}")
    
    # Set device (use GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Step 1: Load data
    print("\nLoading CholecT50 data...")
    train_data = load_cholect50_data(cfg['data'], split='train')
    test_data = load_cholect50_data(cfg['data'], split='test')
    
    # Step 2: Pre-train next frame prediction model
    if cfg_exp['pretrain_next_frame']:        
        print("\nTraining next frame prediction model...")
        world_model = CausalGPT2ForFrameEmbeddings(**cfg['models']['world_model']).to(device)
        train_next_frame_model(cfg, world_model, train_data, test_data, device=device)  # Reduced epochs for demonstration

    # Step 3: Train reward prediction model
    if cfg_exp['pretrain_reward_model']:
        print("\nTraining reward prediction model...")
        reward_model = RewardPredictor(**cfg['models']['reward']).to(device)
        train_reward_model(cfg['reward_model'], train_data, device)
    
    # Calculate action rewards
    if cfg_exp['calculate_action_rewards']:
        print("\nCalculating action rewards...")
        avg_action_rewards = calculate_action_rewards(train_data, next_frame_model, reward_model, device)
    
    # Train action policy model with reward weighting
    if cfg_exp['train_action_policy']:
        print("\nTraining action policy model...")
        policy_model, action_weights = train_action_policy(cfg, train_data, avg_action_rewards, device)
    
    # Run TD-MPC2 to evaluate the model
    if cfg_exp['run_tdmpc']:
        print("\nRunning TD-MPC2...")
        results = run_tdmpc(data, next_frame_model, reward_model, policy_model, action_weights, device)
    
    # Analyze and visualize results
    if cfg_exp['analyze_results']:
        print("\nAnalyzing results...")
        analysis = analyze_results(results, action_weights)
    
    return next_frame_model, reward_model, policy_model, action_weights, results, analysis

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

if __name__ == "__main__":
    # Load config
    config_path = 'config.yaml'
    print(f"Loading configuration from {os.path.abspath(config_path)}")
    cfg = load_config(config_path)
        
    print("\nConfiguration loaded successfully!")
    
    # Run the experiment
    next_frame_model, reward_model, policy_model, action_weights, results, analysis = run_cholect50_experiment(cfg)    
    print("\nExperiment completed!")
    print(f"Model performance: {analysis['percent_improvement']:.2f}% improvement in action quality")