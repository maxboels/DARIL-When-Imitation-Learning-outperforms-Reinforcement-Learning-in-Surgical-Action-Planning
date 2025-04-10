import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import argparse
from collections import defaultdict

class GlobalRewardCalculator:
    """
    Calculates global (video-level) rewards for surgical procedures
    based on completion success, risk management, and efficiency.
    """
    
    def __init__(self, 
                 phase_importance=None,
                 risk_weight=1.0,
                 time_efficiency_weight=0.5,
                 transition_coverage_weight=1.0,
                 critical_risk_threshold=4.0,
                 critical_risk_penalty=10.0):
        """
        Initialize the global reward calculator.
        
        Args:
            phase_importance: Dictionary mapping phase IDs to their importance (default: equal)
            risk_weight: Weight for risk penalty in global score
            time_efficiency_weight: Weight for time efficiency in global score
            transition_coverage_weight: Weight for phase coverage in global score
            critical_risk_threshold: Threshold for critical risk events
            critical_risk_penalty: Penalty for critical risk events
        """
        # Default phase importance if not provided (equal importance)
        self.phase_importance = phase_importance or {i: 1.0 for i in range(7)}
        
        self.risk_weight = risk_weight
        self.time_efficiency_weight = time_efficiency_weight
        self.transition_coverage_weight = transition_coverage_weight
        self.critical_risk_threshold = critical_risk_threshold
        self.critical_risk_penalty = critical_risk_penalty
    
    def calculate_global_reward(self, video_data):
        """
        Calculate a global reward score for a video based on multiple factors.
        
        Args:
            video_data: Dictionary containing:
                - 'phases_df': DataFrame with phase segments (start_frame, end_frame, phase_id)
                - 'risk_scores': Dictionary mapping frame_id to risk score
                - 'frame_count': Total number of frames
                - 'phase_coverage': Set of phase IDs present in the video
        
        Returns:
            global_score: Final score for the video
            component_scores: Dictionary with individual component scores
        """
        phases_df = video_data.get('phases_df')
        risk_scores = video_data.get('risk_scores', {})
        frame_count = video_data.get('frame_count', 0)
        phase_coverage = video_data.get('phase_coverage', set())
        
        component_scores = {}
        
        # 1. Phase Coverage Score
        # Calculate what fraction of important phases are covered
        if phases_df is not None and not phases_df.empty:
            # Get total weight of all possible phases
            total_phase_weight = sum(self.phase_importance.values())
            
            # Get total weight of covered phases
            covered_phase_weight = sum(self.phase_importance.get(phase_id, 0) 
                                      for phase_id in phase_coverage)
            
            phase_coverage_score = (covered_phase_weight / total_phase_weight) * 10.0
        else:
            phase_coverage_score = 0.0
            
        component_scores['phase_coverage'] = phase_coverage_score
        
        # 2. Risk Management Score
        if risk_scores:
            # Average risk score
            avg_risk = np.mean(list(risk_scores.values()))
            
            # Count critical risk events (above threshold)
            critical_risk_events = sum(1 for score in risk_scores.values() 
                                      if score > self.critical_risk_threshold)
            
            # Risk score is 10 - (avg_risk * 2) with penalty for critical events
            risk_score = 10.0 - (avg_risk * 2.0) - (critical_risk_events * self.critical_risk_penalty / len(risk_scores))
            
            # Cap between 0 and 10
            risk_score = max(0.0, min(10.0, risk_score))
        else:
            risk_score = 5.0  # Default middle score if no risk data
            
        component_scores['risk_management'] = risk_score
        
        # 3. Phase Transition Score
        # Assess how well the procedure followed a reasonable sequence of phases
        if phases_df is not None and len(phases_df) > 1:
            # Count phase transitions
            transition_count = len(phases_df) - 1
            
            # Reasonable number of transitions should be around the number of unique phases
            expected_transitions = len(phase_coverage) - 1
            
            # Too many transitions is inefficient (back and forth)
            if transition_count > expected_transitions:
                transition_efficiency = expected_transitions / transition_count
            else:
                transition_efficiency = transition_count / max(1, expected_transitions)
            
            transition_score = transition_efficiency * 10.0
        else:
            transition_score = 0.0  # No transitions
            
        component_scores['transition_efficiency'] = transition_score
        
        # 4. Combine scores with weights
        global_score = (
            (phase_coverage_score * self.transition_coverage_weight) +
            (risk_score * self.risk_weight) +
            (transition_score * self.time_efficiency_weight)
        ) / (self.transition_coverage_weight + self.risk_weight + self.time_efficiency_weight)
        
        return global_score, component_scores


class SurgicalFrameDataset(Dataset):
    """
    Dataset for training models to predict global rewards from frame features.
    """
    
    def __init__(self, features, targets, frame_ids=None):
        """
        Initialize the dataset.
        
        Args:
            features: Tensor or array of frame features (N x feature_dim)
            targets: Tensor or array of global reward targets (N)
            frame_ids: Optional list of frame IDs for reference
        """
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.frame_ids = frame_ids
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return {
            'features': self.features[idx],
            'target': self.targets[idx],
            'frame_id': self.frame_ids[idx] if self.frame_ids is not None else idx
        }


class RewardPredictionModel(nn.Module):
    """
    Neural network to predict global reward from frame features.
    """
    
    def __init__(self, input_dim, hidden_dims=[128, 64], dropout_rate=0.2):
        """
        Initialize the model.
        
        Args:
            input_dim: Dimension of input features
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout probability for regularization
        """
        super(RewardPredictionModel, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Output layer (predict a single reward value)
        layers.append(nn.Linear(prev_dim, 1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x).squeeze()


class ModelBasedRewardFunction:
    """
    Reward function based on a trained model that predicts global outcomes.
    """
    
    def __init__(self, model=None, feature_scaler=None, reward_scaler=None, 
                 smooth_factor=5.0, reward_scale=1.0):
        """
        Initialize the model-based reward function.
        
        Args:
            model: Trained PyTorch model for reward prediction
            feature_scaler: StandardScaler for normalizing input features
            reward_scaler: StandardScaler for normalizing reward values
            smooth_factor: Smoothing factor for reward signals
            reward_scale: Scaling factor for final rewards
        """
        self.model = model
        self.feature_scaler = feature_scaler
        self.reward_scaler = reward_scaler
        self.smooth_factor = smooth_factor
        self.reward_scale = reward_scale
        
        # Storage for previous predictions (for calculating slopes)
        self.previous_predictions = {}
        self.prediction_history = []
    
    def calculate_reward(self, frame_data):
        """
        Calculate reward based on the model's prediction improvement.
        
        Args:
            frame_data: Dictionary containing:
                - 'features': Feature vector for the current frame
                - 'frame_id': ID of the current frame
        
        Returns:
            reward: Calculated reward value
            info: Dictionary with additional information
        """
        frame_id = frame_data.get('frame_id')
        features = frame_data.get('features')
        
        if self.model is None or features is None:
            return 0.0, {'predicted_value': 0.0, 'slope': 0.0}
        
        # Preprocess features
        if self.feature_scaler is not None:
            features = self.feature_scaler.transform([features])[0]
        
        # Convert to tensor
        features_tensor = torch.tensor(features, dtype=torch.float32)
        
        # Get prediction
        self.model.eval()
        with torch.no_grad():
            predicted_value = self.model(features_tensor).item()
        
        # Denormalize prediction if needed
        if self.reward_scaler is not None:
            predicted_value = self.reward_scaler.inverse_transform([[predicted_value]])[0][0]
        
        # Store prediction
        self.previous_predictions[frame_id] = predicted_value
        self.prediction_history.append(predicted_value)
        
        # Calculate slope (improvement in prediction)
        # Use a window of predictions to calculate trend
        window_size = min(10, len(self.prediction_history))
        if window_size >= 2:
            recent_predictions = self.prediction_history[-window_size:]
            x = np.arange(window_size)
            slope = np.polyfit(x, recent_predictions, 1)[0]
        else:
            slope = 0.0
        
        # Scale the slope to get the reward
        reward = slope * self.reward_scale
        
        return reward, {'predicted_value': predicted_value, 'slope': slope}
    
    def reset(self):
        """Reset the reward function state for a new video."""
        self.previous_predictions = {}
        self.prediction_history = []


def extract_frame_features(metadata_df, video_id, risk_column='risk_score_max'):
    """
    Extract features for each frame in a video.
    
    Args:
        metadata_df: DataFrame with metadata
        video_id: ID of the video to process
        risk_column: Column name for risk scores
    
    Returns:
        Dictionary mapping frame_id to feature vector and phase_id
    """
    # Filter for this video
    video_data = metadata_df[metadata_df['video'] == video_id].copy()
    
    if video_data.empty:
        raise ValueError(f"No data found for video {video_id}")
    
    # Find phase columns
    phase_columns = [col for col in video_data.columns if col.startswith('p') and col[1:].isdigit()]
    
    # Extract features for each frame
    frame_features = {}
    
    for _, row in video_data.iterrows():
        frame_id = row['frame']
        
        # Get risk score
        risk_score = row[risk_column] if risk_column in row else 0.0
        
        # Get phase one-hot encoding
        phase_encoding = [row[col] for col in phase_columns]
        
        # Determine phase ID
        phase_id = None
        for i, val in enumerate(phase_encoding):
            if val == 1:
                phase_id = i
                break
        
        # Create feature vector
        # You can add more features here as needed
        features = [risk_score] + phase_encoding
        
        frame_features[frame_id] = {
            'features': features,
            'phase_id': phase_id,
            'risk_score': risk_score
        }
    
    return frame_features


def prepare_dataset_from_videos(metadata_df, video_ids, risk_column='risk_score_max'):
    """
    Prepare a dataset from multiple videos for training the reward prediction model.
    
    Args:
        metadata_df: DataFrame with metadata
        video_ids: List of video IDs to include
        risk_column: Column name for risk scores
    
    Returns:
        feature_array: Array of features (N x feature_dim)
        target_array: Array of global reward targets (N)
        video_frame_map: Dictionary mapping (video_id, frame_id) to index in arrays
        feature_scaler: Fitted StandardScaler for features
    """
    all_features = []
    all_targets = []
    all_video_frames = []
    
    # Create reward calculator
    reward_calculator = GlobalRewardCalculator()
    
    for video_id in tqdm(video_ids, desc="Preparing dataset"):
        try:
            # Extract features for this video
            frame_features = extract_frame_features(metadata_df, video_id, risk_column)
            
            if not frame_features:
                continue
            
            # Extract phase information for global reward calculation
            _, phases_df, _ = load_data_from_metadata(metadata_df, video_id)
            
            # Get risk scores
            risk_scores = {frame_id: data['risk_score'] for frame_id, data in frame_features.items()}
            
            # Calculate global reward for this video
            video_data = {
                'phases_df': phases_df,
                'risk_scores': risk_scores,
                'frame_count': len(frame_features),
                'phase_coverage': {data['phase_id'] for data in frame_features.values() if data['phase_id'] is not None}
            }
            
            global_reward, _ = reward_calculator.calculate_global_reward(video_data)
            
            # Add features and targets for each frame
            for frame_id, data in frame_features.items():
                all_features.append(data['features'])
                all_targets.append(global_reward)
                all_video_frames.append((video_id, frame_id))
                
        except Exception as e:
            print(f"Error processing video {video_id}: {e}")
    
    # Convert to arrays
    feature_array = np.array(all_features)
    target_array = np.array(all_targets)
    
    # Create mapping from (video_id, frame_id) to index
    video_frame_map = {vf: i for i, vf in enumerate(all_video_frames)}
    
    # Scale features
    feature_scaler = StandardScaler()
    feature_array = feature_scaler.fit_transform(feature_array)
    
    return feature_array, target_array, video_frame_map, feature_scaler


def train_reward_prediction_model(features, targets, test_size=0.2, batch_size=64, 
                                 learning_rate=0.001, epochs=50, hidden_dims=[128, 64]):
    """
    Train a model to predict global rewards from frame features.
    
    Args:
        features: Array of frame features (N x feature_dim)
        targets: Array of global reward targets (N)
        test_size: Fraction of data to use for validation
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        epochs: Number of training epochs
        hidden_dims: List of hidden layer dimensions
    
    Returns:
        model: Trained PyTorch model
        reward_scaler: Fitted StandardScaler for reward values
    """
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        features, targets, test_size=test_size, random_state=42
    )
    
    # Scale targets
    reward_scaler = StandardScaler()
    y_train_scaled = reward_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val_scaled = reward_scaler.transform(y_val.reshape(-1, 1)).flatten()
    
    # Create datasets
    train_dataset = SurgicalFrameDataset(X_train, y_train_scaled)
    val_dataset = SurgicalFrameDataset(X_val, y_val_scaled)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model
    input_dim = features.shape[1]
    model = RewardPredictionModel(input_dim, hidden_dims)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            features = batch['features']
            targets = batch['target']
            
            outputs = model(features)
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                features = batch['features']
                targets = batch['target']
                
                outputs = model(features)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
        
        # Print progress
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, reward_scaler


def load_data_from_metadata(metadata_df, video_id, risk_column_names=None):
    """
    Load both risk scores and phase information from metadata DataFrame for a specific video.
    
    Args:
        metadata_df: DataFrame containing metadata with risk scores and phase information
        video_id: ID of the video
        risk_column_names: List of column names containing risk scores. If None, will try to detect risk score columns.
    
    Returns:
        Tuple containing:
        - Dictionary with risk score column names as keys and dictionaries of {frame_id: risk_score} as values
        - DataFrame with phase information (start_frame, end_frame, phase_id)
        - Dictionary mapping frame_id to its phase_id
    """
    # Filter metadata for this video
    video_metadata = metadata_df[metadata_df['video'] == video_id].copy()
    
    if video_metadata.empty:
        raise ValueError(f"No metadata found for video {video_id}")
    
    # Detect risk score columns if not provided
    if risk_column_names is None:
        risk_column_names = [col for col in video_metadata.columns if col.startswith('risk_score_')]
        if not risk_column_names:
            raise ValueError(f"No risk score columns found in metadata for video {video_id}")
    
    # Extract risk scores for each column
    all_risk_scores = {}
    for risk_column in risk_column_names:
        if risk_column not in video_metadata.columns:
            print(f"Warning: Risk score column '{risk_column}' not found in metadata")
            continue
            
        frame_risk_scores = {}
        for _, row in video_metadata.iterrows():
            frame_id = row['frame']
            risk_score = row[risk_column]
            frame_risk_scores[frame_id] = risk_score
            
        all_risk_scores[risk_column] = frame_risk_scores
    
    if not all_risk_scores:
        raise ValueError(f"No valid risk score columns found for video {video_id}")
    
    # Extract phase information
    # Check for phase columns (p0 to p6)
    phase_columns = [f'p{i}' for i in range(7) if f'p{i}' in video_metadata.columns]
    
    if not phase_columns:
        print(f"Warning: No phase columns (p0-p6) found in metadata for video {video_id}")
        return all_risk_scores, None, {}
    
    # For each frame, determine which phase it belongs to
    # The phase columns are binary (1 if the frame belongs to that phase, 0 otherwise)
    frame_phases = {}
    for _, row in video_metadata.iterrows():
        frame_id = row['frame']
        # Find which phase has a value of 1
        phase_id = None
        for i, col in enumerate(phase_columns):
            if row[col] == 1:
                phase_id = i
                break
        frame_phases[frame_id] = phase_id
    
    # Create phases DataFrame by finding continuous segments with the same phase
    phases_data = []
    sorted_frames = sorted(frame_phases.keys())
    if not sorted_frames:
        return all_risk_scores, None, {}
    
    current_phase = frame_phases[sorted_frames[0]]
    start_frame = sorted_frames[0]
    
    for i in range(1, len(sorted_frames)):
        frame_id = sorted_frames[i]
        phase_id = frame_phases[frame_id]
        
        # If phase changes, record the previous phase segment
        if phase_id != current_phase:
            phases_data.append({
                'phase_id': current_phase,
                'start_frame': start_frame,
                'end_frame': sorted_frames[i-1]
            })
            current_phase = phase_id
            start_frame = frame_id
    
    # Add the last phase segment
    phases_data.append({
        'phase_id': current_phase,
        'start_frame': start_frame,
        'end_frame': sorted_frames[-1]
    })
    
    return all_risk_scores, pd.DataFrame(phases_data), frame_phases


def calculate_model_based_rewards(metadata_df, video_id, risk_column, 
                                 model, feature_scaler, reward_scaler, 
                                 smooth_factor=5.0):
    """
    Calculate rewards for a video based on the model's predictions.
    
    Args:
        metadata_df: DataFrame with metadata
        video_id: ID of the video to process
        risk_column: Column name for risk scores
        model: Trained PyTorch model
        feature_scaler: StandardScaler for features
        reward_scaler: StandardScaler for targets
        smooth_factor: Smoothing factor for reward curves
    
    Returns:
        Tuple containing:
        - Dictionary mapping frame_id to reward value
        - Dictionary mapping frame_id to cumulative reward
        - Dictionary mapping frame_id to predicted global reward
        - Dictionary mapping frame_id to reward slope
    """
    # Extract features for the video
    frame_features = extract_frame_features(metadata_df, video_id, risk_column)
    
    # Get frames in order
    sorted_frames = sorted(frame_features.keys())
    
    # Initialize reward function
    reward_function = ModelBasedRewardFunction(
        model=model,
        feature_scaler=feature_scaler,
        reward_scaler=reward_scaler,
        smooth_factor=smooth_factor
    )
    
    # Calculate rewards frame by frame
    frame_rewards = {}
    predicted_values = {}
    reward_slopes = {}
    
    for frame_id in sorted_frames:
        frame_data = {
            'frame_id': frame_id,
            'features': frame_features[frame_id]['features']
        }
        
        reward, info = reward_function.calculate_reward(frame_data)
        
        frame_rewards[frame_id] = reward
        predicted_values[frame_id] = info['predicted_value']
        reward_slopes[frame_id] = info['slope']
    
    # Apply smoothing to rewards if requested
    if smooth_factor > 0:
        rewards_list = [frame_rewards[frame] for frame in sorted_frames]
        smoothed_rewards = gaussian_filter1d(rewards_list, sigma=smooth_factor)
        
        for i, frame in enumerate(sorted_frames):
            frame_rewards[frame] = smoothed_rewards[i]
    
    # Calculate cumulative rewards
    cumulative_rewards = {}
    cumulative_reward = 0.0
    
    for frame in sorted_frames:
        cumulative_reward += frame_rewards[frame]
        cumulative_rewards[frame] = cumulative_reward
    
    return frame_rewards, cumulative_rewards, predicted_values, reward_slopes


def plot_model_based_rewards(video_id, risk_scores_dict, frame_rewards, cumulative_rewards, 
                           predicted_values, reward_slopes, phases_df=None,
                           output_dir=None, smooth_factor=5.0):
    """
    Plot model-based rewards and predictions for a video.
    
    Args:
        video_id: ID of the video
        risk_scores_dict: Dictionary with risk score column names as keys and {frame_id: risk_score} as values
        frame_rewards: Dictionary mapping frame_id to reward value
        cumulative_rewards: Dictionary mapping frame_id to cumulative reward
        predicted_values: Dictionary mapping frame_id to predicted global reward
        reward_slopes: Dictionary mapping frame_id to reward slope
        phases_df: DataFrame with phase information
        output_dir: Directory to save the plot
        smooth_factor: Smoothing factor for display
    """
    # Set up figure
    fig, axs = plt.subplots(5, 1, figsize=(14, 20), sharex=True)
    
    # Sort frames
    all_frames = set()
    for risk_type, risk_scores in risk_scores_dict.items():
        all_frames.update(risk_scores.keys())
    all_frames.update(frame_rewards.keys())
    sorted_frames = sorted(all_frames)
    
    # Create x-axis array and mapping
    x_indices = np.arange(len(sorted_frames))
    frame_to_index = {frame: i for i, frame in enumerate(sorted_frames)}
    
    # Standard colors for surgical phases (p0 to p6)
    phase_colors = {
        0: '#FFD700',  # Phase 0 (Preparation)
        1: '#87CEFA',  # Phase 1
        2: '#98FB98',  # Phase 2
        3: '#FFA07A',  # Phase 3
        4: '#DDA0DD',  # Phase 4
        5: '#D3D3D3',  # Phase 5
        6: '#B0E0E6'   # Phase 6
    }
    
    phase_names = {
        0: "Phase 0",
        1: "Phase 1", 
        2: "Phase 2",
        3: "Phase 3",
        4: "Phase 4",
        5: "Phase 5",
        6: "Phase 6"
    }
    
    # Function to add phase backgrounds to a subplot
    def add_phase_backgrounds(ax, ymin, ymax):
        if phases_df is not None:
            for _, row in phases_df.iterrows():
                phase_id = row['phase_id']
                start_frame = row['start_frame']
                end_frame = row['end_frame']
                
                # Skip phases outside our frame range
                if end_frame < min(sorted_frames) or start_frame > max(sorted_frames):
                    continue
                    
                # Adjust to our frame range
                start_frame = max(start_frame, min(sorted_frames))
                end_frame = min(end_frame, max(sorted_frames))
                
                # Convert to indices in our array
                start_idx = frame_to_index.get(start_frame, 0)
                end_idx = frame_to_index.get(end_frame, len(sorted_frames) - 1)
                
                # Add colored background
                alpha = 0.3  # Transparency
                rect = plt.Rectangle((start_idx, ymin), end_idx - start_idx, ymax - ymin, 
                                    color=phase_colors.get(phase_id, '#CCCCCC'), alpha=alpha)
                ax.add_patch(rect)
    
    # === PLOT 1: RISK SCORES ===
    ax_risk = axs[0]
    
    # Plot each risk score type
    risk_type_colors = plt.cm.tab10.colors
    for i, (risk_type, risk_scores) in enumerate(risk_scores_dict.items()):
        # Convert to arrays for plotting
        risk_x = [frame_to_index[frame] for frame in sorted_frames if frame in risk_scores]
        risk_y = [risk_scores[frame] for frame in sorted_frames if frame in risk_scores]
        
        # Apply smoothing
        if smooth_factor > 0 and len(risk_y) > smooth_factor * 3:
            risk_y = gaussian_filter1d(risk_y, sigma=smooth_factor)
        
        display_name = risk_type.replace('risk_score_', '')
        ax_risk.plot(risk_x, risk_y, label=display_name, color=risk_type_colors[i % len(risk_type_colors)], linewidth=2)
    
    # Set y limits to include 0
    risk_ymin, risk_ymax = ax_risk.get_ylim()
    risk_ymin = min(0, risk_ymin)
    ax_risk.set_ylim(risk_ymin, risk_ymax)
    
    # Add phase backgrounds
    add_phase_backgrounds(ax_risk, risk_ymin, risk_ymax)
    
    # Set labels
    ax_risk.set_ylabel('Risk Score')
    ax_risk.set_title(f'Risk Scores for Video {video_id}')
    ax_risk.grid(True, linestyle='--', alpha=0.7)
    if risk_scores_dict:
        ax_risk.legend(loc='upper right')
    
    # === PLOT 2: PREDICTED GLOBAL REWARD ===
    ax_pred = axs[1]
    
    # Convert predictions to arrays for plotting
    pred_x = [frame_to_index[frame] for frame in sorted_frames if frame in predicted_values]
    pred_y = [predicted_values[frame] for frame in sorted_frames if frame in predicted_values]
    
    # Plot predictions
    ax_pred.plot(pred_x, pred_y, color='purple', linewidth=2)
    
    # Set y limits
    pred_ymin, pred_ymax = ax_pred.get_ylim()
    add_phase_backgrounds(ax_pred, pred_ymin, pred_ymax)
    
    # Set labels
    ax_pred.set_ylabel('Predicted Global Reward')
    ax_pred.set_title(f'Model Predicted Global Reward for Video {video_id}')
    ax_pred.grid(True, linestyle='--', alpha=0.7)
    
    # === PLOT 3: REWARD SLOPES (FRAME REWARDS) ===
    ax_slope = axs[2]
    
    # Convert slopes to arrays for plotting
    slope_x = [frame_to_index[frame] for frame in sorted_frames if frame in reward_slopes]
    slope_y = [reward_slopes[frame] for frame in sorted_frames if frame in reward_slopes]
    
    # Plot slopes
    ax_slope.plot(slope_x, slope_y, color='orange', linewidth=2)
    
    # Add horizontal line at y=0
    ax_slope.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    
    # Set y limits
    slope_ymin, slope_ymax = ax_slope.get_ylim()
    slope_ymin = min(-0.1, slope_ymin)
    slope_ymax = max(0.1, slope_ymax)
    ax_slope.set_ylim(slope_ymin, slope_ymax)
    
    # Add phase backgrounds
    add_phase_backgrounds(ax_slope, slope_ymin, slope_ymax)
    
    # Set labels
    ax_slope.set_ylabel('Reward Slope')
    ax_slope.set_title(f'Reward Slopes for Video {video_id}')
    ax_slope.grid(True, linestyle='--', alpha=0.7)
    
    # === PLOT 4: FRAME REWARDS ===
    ax_reward = axs[3]
    
    # Convert rewards to arrays for plotting
    reward_x = [frame_to_index[frame] for frame in sorted_frames if frame in frame_rewards]
    reward_y = [frame_rewards[frame] for frame in sorted_frames if frame in frame_rewards]
    
    # Plot rewards
    ax_reward.plot(reward_x, reward_y, color='green', linewidth=2)
    
    # Add horizontal line at y=0
    ax_reward.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    
    # Set y limits
    reward_ymin, reward_ymax = ax_reward.get_ylim()
    reward_ymin = min(-0.1, reward_ymin)
    reward_ymax = max(0.1, reward_ymax)
    ax_reward.set_ylim(reward_ymin, reward_ymax)
    
    # Add phase backgrounds
    add_phase_backgrounds(ax_reward, reward_ymin, reward_ymax)
    
    # Set labels
    ax_reward.set_ylabel('Frame Reward')
    ax_reward.set_title(f'Frame Rewards for Video {video_id}')
    ax_reward.grid(True, linestyle='--', alpha=0.7)
    
    # === PLOT 5: CUMULATIVE REWARDS ===
    ax_cumul = axs[4]
    
    # Convert cumulative rewards to arrays for plotting
    cumul_x = [frame_to_index[frame] for frame in sorted_frames if frame in cumulative_rewards]
    cumul_y = [cumulative_rewards[frame] for frame in sorted_frames if frame in cumulative_rewards]
    
    # Plot cumulative rewards
    ax_cumul.plot(cumul_x, cumul_y, color='blue', linewidth=2)
    
    # Add horizontal line at y=0
    ax_cumul.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    
    # Set y limits
    cumul_ymin, cumul_ymax = ax_cumul.get_ylim()
    cumul_ymin = min(-1, cumul_ymin)
    ax_cumul.set_ylim(cumul_ymin, cumul_ymax)
    
    # Add phase backgrounds
    add_phase_backgrounds(ax_cumul, cumul_ymin, cumul_ymax)
    
    # Set labels
    ax_cumul.set_ylabel('Cumulative Reward')
    ax_cumul.set_title(f'Cumulative Rewards for Video {video_id}')
    ax_cumul.grid(True, linestyle='--', alpha=0.7)
    
    # Add phase legend to the bottom subplot
    if phases_df is not None:
        # Add phase legend
        phase_ids = sorted(set(phases_df['phase_id']))
        phase_labels = [phase_names.get(i, f"Phase {i}") for i in phase_ids]
        phase_handles = [plt.Rectangle((0, 0), 1, 1, color=phase_colors.get(i, '#CCCCCC')) 
                         for i in phase_ids]
        
        # Add to the last subplot
        axs[-1].legend(phase_handles, phase_labels, loc='lower right', title="Surgical Phases")
    
    # Set common x-axis label on bottom subplot
    axs[-1].set_xlabel('Frame Number')
    
    # Add frame numbers to the bottom x-axis
    frame_ticks = np.linspace(0, len(sorted_frames)-1, min(10, len(sorted_frames)))
    frame_tick_labels = [sorted_frames[int(i)] for i in frame_ticks]
    axs[-1].set_xticks(frame_ticks)
    axs[-1].set_xticklabels(frame_tick_labels)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show the plot
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{video_id}_model_reward_visualization.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    else:
        plt.tight_layout()
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train reward prediction model and visualize model-based rewards')
    parser.add_argument('--metadata_path', type=str, default="/home/maxboels/datasets/CholecT50/embeddings_train_set/fold0/embeddings_f0_swin_bas_129.csv",
                      help='Path to metadata CSV file containing risk scores and phase information')
    parser.add_argument('--output_dir', type=str, default="model_reward_visualizations",
                      help='Directory to save visualizations')
    parser.add_argument('--train', type=bool, default=True,
                        help='Train a new model (otherwise, load an existing model)')
    parser.add_argument('--model_path', type=str, default="reward_model.pt",
                      help='Path to save or load the model')
    parser.add_argument('--risk_column', type=str, default='risk_score_max',
                      help='Risk score column to use')
    parser.add_argument('--smooth_factor', type=float, default=5.0,
                      help='Smoothing factor for reward curves')
    parser.add_argument('--video_id', type=str, default=None,
                      help='Specific video ID to visualize (if not specified, will process all videos)')
    parser.add_argument('--test_size', type=float, default=0.2,
                      help='Fraction of videos to use for testing')
    parser.add_argument('--epochs', type=int, default=50,
                      help='Number of training epochs')
    
    args = parser.parse_args()
    
    # Check if metadata file exists
    if not os.path.exists(args.metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {args.metadata_path}")
    
    # Load metadata
    print(f"Loading metadata from {args.metadata_path}")
    metadata_df = pd.read_csv(args.metadata_path)
    
    # Get all unique video IDs
    all_video_ids = sorted(metadata_df['video'].unique().tolist())
    print(f"Found {len(all_video_ids)} videos in metadata")
    
    # Split videos into train and test sets
    test_size = min(args.test_size, 0.5)  # Ensure we have enough training data
    n_test = max(1, int(len(all_video_ids) * test_size))
    
    # Use a fixed seed for reproducibility
    np.random.seed(42)
    test_indices = np.random.choice(len(all_video_ids), n_test, replace=False)
    test_video_ids = [all_video_ids[i] for i in test_indices]
    train_video_ids = [vid for i, vid in enumerate(all_video_ids) if i not in test_indices]
    
    print(f"Split videos into {len(train_video_ids)} training and {len(test_video_ids)} test videos")
    
    # Create output directory
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Train or load the model
    if args.train:
        print("Preparing dataset for training...")
        features, targets, video_frame_map, feature_scaler = prepare_dataset_from_videos(
            metadata_df, train_video_ids, args.risk_column
        )
        
        print(f"Dataset prepared with {len(features)} frame samples")
        
        print("Training reward prediction model...")
        model, reward_scaler = train_reward_prediction_model(
            features, targets, test_size=0.2, epochs=args.epochs
        )
        
        # Save model and scalers
        torch.save({
            'model_state_dict': model.state_dict(),
            'feature_scaler': feature_scaler,
            'reward_scaler': reward_scaler,
            'input_dim': features.shape[1],
            'hidden_dims': [128, 64],  # Match what was used in training
        }, args.model_path)
        
        print(f"Model saved to {args.model_path}")
    else:
        # Load the model
        if not os.path.exists(args.model_path):
            raise FileNotFoundError(f"Model file not found: {args.model_path}")
        
        print(f"Loading model from {args.model_path}")
        checkpoint = torch.load(args.model_path)
        
        # Create model with same architecture
        input_dim = checkpoint['input_dim']
        hidden_dims = checkpoint['hidden_dims']
        model = RewardPredictionModel(input_dim, hidden_dims)
        
        # Load weights and scalers
        model.load_state_dict(checkpoint['model_state_dict'])
        feature_scaler = checkpoint['feature_scaler']
        reward_scaler = checkpoint['reward_scaler']
    
    # Process videos for visualization
    videos_to_process = [args.video_id] if args.video_id else test_video_ids
    
    print(f"Processing {len(videos_to_process)} videos for visualization...")
    
    for video_id in tqdm(videos_to_process):
        try:
            # Skip if video not in metadata
            if video_id not in all_video_ids:
                print(f"Warning: Video {video_id} not found in metadata")
                continue
            
            # Load risk scores and phase information
            risk_scores_dict, phases_df, _ = load_data_from_metadata(
                metadata_df, video_id, [args.risk_column]
            )
            
            # Calculate model-based rewards
            frame_rewards, cumulative_rewards, predicted_values, reward_slopes = calculate_model_based_rewards(
                metadata_df, video_id, args.risk_column,
                model, feature_scaler, reward_scaler,
                args.smooth_factor
            )
            
            # Visualize results
            plot_model_based_rewards(
                video_id, risk_scores_dict, frame_rewards, cumulative_rewards,
                predicted_values, reward_slopes, phases_df,
                args.output_dir, args.smooth_factor
            )
            
            print(f"Successfully processed video {video_id}")
            
        except Exception as e:
            print(f"Error processing video {video_id}: {e}")
    
    print(f"\nOutput saved to: {os.path.abspath(args.output_dir)}")

if __name__ == "__main__":
    main()