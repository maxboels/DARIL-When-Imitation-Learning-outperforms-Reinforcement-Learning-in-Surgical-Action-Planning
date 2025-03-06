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

class SurgicalRewardFunction:
    """
    Reward function for surgical procedures that balances task completion with risk assessment.
    Enhanced version that saves reward components for later use.
    """
    
    def __init__(self, 
                 phase_weights=None, 
                 risk_weight=1.0, 
                 time_penalty=0.01,
                 transition_bonus=5.0,
                 completion_bonus=20.0,
                 critical_risk_threshold=4.0,
                 critical_risk_penalty=10.0,
                 default_phase_duration=100,
                 smoothing_factor=3.0):
        # Default phase weights if not provided (equal weight to all phases)
        self.phase_weights = phase_weights or {i: 1.0 for i in range(7)}  # Assuming phases 0-6
        
        self.risk_weight = risk_weight
        self.time_penalty = time_penalty
        self.transition_bonus = transition_bonus
        self.completion_bonus = completion_bonus
        self.critical_risk_threshold = critical_risk_threshold
        self.critical_risk_penalty = critical_risk_penalty
        self.smoothing_factor = smoothing_factor
        self.default_phase_duration = default_phase_duration
        
        # Initialize phase statistics with defaults
        self.phase_duration_stats = {i: {'mean': default_phase_duration, 'std': default_phase_duration/4} 
                                    for i in range(7)}
        
        # Phase tracking
        self.current_phase = None
        self.previous_phase = None
        self.last_processed_frame = None
        
        # Track completed phases (set of phase IDs)
        self.completed_phases = set()
        
        # Track phase segments (for handling multiple segments of the same phase)
        self.phase_segments = None  # Will be set with phase_segments if provided
        self.current_segment_idx = None  # Current phase segment index
        
        # Store frame-based progress within segments
        self.segment_progress = {}  # {segment_idx: progress_value}
        
        # Track the last created phase transition rewards to avoid duplicates
        self.last_transition_source = None
        self.last_transition_target = None
        
        # Store all calculated rewards and components
        self.all_frame_rewards = {}
        self.all_component_rewards = defaultdict(dict)
        self.cumulative_rewards = {}
    
    def calculate_reward(self, frame_data, is_terminal=False, phase_segments=None):
        """
        Calculate the reward for a single frame in the surgical procedure.
        
        Args:
            frame_data: Dictionary containing:
                - 'phase_id': Current surgical phase ID
                - 'risk_scores': Dictionary of risk scores (different types)
                - 'frame_id': Current frame ID
            is_terminal: Whether this is the terminal state (procedure completed)
            phase_segments: DataFrame with phase segment information (start_frame, end_frame, phase_id)
            
        Returns:
            total_reward: The calculated reward value
            reward_components: Dictionary breaking down reward components
        """
        # Extract data
        phase_id = frame_data.get('phase_id')
        risk_scores = frame_data.get('risk_scores', {})
        frame_id = frame_data.get('frame_id', 0)
        
        reward_components = {}
        
        # Store phase segments if provided (first time)
        if self.phase_segments is None and phase_segments is not None:
            self.phase_segments = phase_segments.copy()
            # Initialize segment progress dictionary
            self.segment_progress = {i: 0.0 for i in range(len(phase_segments))}
        
        # --------- FIND CURRENT PHASE SEGMENT ---------
        current_segment_idx = None
        if self.phase_segments is not None:
            # Find which segment this frame belongs to
            for i, row in self.phase_segments.iterrows():
                if frame_id >= row['start_frame'] and frame_id <= row['end_frame']:
                    current_segment_idx = i
                    break
        
        # Update phase tracking
        self.previous_phase = self.current_phase
        self.current_phase = phase_id
        
        # --------- PHASE TRANSITION HANDLING ---------
        phase_transition_reward = 0.0
        
        # Check if phase has changed
        phase_changed = (self.previous_phase is not None and 
                         phase_id is not None and 
                         self.previous_phase != phase_id)
        
        # Handle phase transitions (only when phase actually changes)
        if phase_changed:
            # Avoid duplicate transition rewards
            transition_key = (self.previous_phase, phase_id)
            if transition_key != (self.last_transition_source, self.last_transition_target):
                # Award transition bonus regardless of direction (no assumption about correct sequence)
                phase_transition_reward = self.transition_bonus * self.phase_weights.get(self.previous_phase, 1.0)
                phase_transition_reward = round(phase_transition_reward, 4)
                
                # Mark the source phase as completed
                self.completed_phases.add(self.previous_phase)
                
                # Update last transition to avoid duplicates
                self.last_transition_source, self.last_transition_target = transition_key
        
        # --------- PHASE PROGRESS CALCULATION ---------
        phase_reward = 0.0
        phase_progress = 0.0
        
        if current_segment_idx is not None:
            # Calculate progress within the current segment
            segment = self.phase_segments.iloc[current_segment_idx]
            segment_duration = segment['end_frame'] - segment['start_frame']
            
            if segment_duration > 0:
                # Calculate position within segment
                position_in_segment = (frame_id - segment['start_frame']) / segment_duration
                position_in_segment = round(position_in_segment, 4)
                
                # Ensure progress never decreases within a segment
                current_progress = max(self.segment_progress.get(current_segment_idx, 0.0), position_in_segment)
                
                # Cap at 1.0
                current_progress = min(current_progress, 1.0)
                
                # Store progress
                self.segment_progress[current_segment_idx] = current_progress
                phase_progress = round(current_progress, 4)
                
                # Calculate reward based on progress and phase importance
                segment_phase_id = segment['phase_id']
                phase_reward = round(current_progress * self.phase_weights.get(segment_phase_id, 1.0), 4)
        
        reward_components['phase_progress_reward'] = phase_reward
        reward_components['phase_transition_reward'] = phase_transition_reward
        reward_components['phase_progress'] = phase_progress
        
        # --------- RISK ASSESSMENT CALCULATION ---------
        risk_penalty = 0.0
        risk_value = 0.0
        
        # Calculate average risk score
        if risk_scores:
            # Can be weighted by risk type importance if needed
            risk_value = np.mean(list(risk_scores.values()))
            risk_value = round(risk_value, 4)
            
            # Basic risk penalty proportional to risk score
            risk_penalty = self.risk_weight * risk_value
            
            # Additional penalty for exceeding critical threshold
            if risk_value > self.critical_risk_threshold:
                risk_penalty += self.critical_risk_penalty * (risk_value - self.critical_risk_threshold)
            
            risk_penalty = round(risk_penalty, 4)
        
        reward_components['risk_penalty'] = round(-risk_penalty, 4)  # Negative because it's a penalty
        reward_components['risk_value'] = risk_value
        
        # --------- TIME EFFICIENCY PENALTY ---------
        time_penalty = round(self.time_penalty, 4)
        reward_components['time_penalty'] = round(-time_penalty, 4)  # Negative because it's a penalty
        
        # --------- COMPLETION BONUS ---------
        completion_reward = 0.0
        if is_terminal and self.phase_segments is not None:
            # Calculate completion based on segment coverage
            all_segments_finished = all(progress >= 0.99 for progress in self.segment_progress.values())
            
            if all_segments_finished:
                completion_reward = round(self.completion_bonus, 4)
        
        reward_components['completion_reward'] = completion_reward
        
        # --------- TOTAL REWARD CALCULATION ---------
        total_reward = round(
            phase_reward + 
            phase_transition_reward + 
            -risk_penalty +  # Negative because it's a penalty
            -time_penalty +  # Negative because it's a penalty
            completion_reward,
            4
        )
        
        # Store the rewards and components
        self.all_frame_rewards[frame_id] = total_reward
        for component, value in reward_components.items():
            self.all_component_rewards[component][frame_id] = value
        
        # Update cumulative reward
        prev_cumulative = self.cumulative_rewards.get(self.last_processed_frame, 0.0) if self.last_processed_frame is not None else 0.0
        self.cumulative_rewards[frame_id] = round(prev_cumulative + total_reward, 4)
        
        # Update last processed frame
        self.last_processed_frame = frame_id
        
        return total_reward, reward_components
    
    def get_all_rewards(self):
        """
        Get all stored rewards and components.
        
        Returns:
            Dictionary containing:
            - 'frame_rewards': Dictionary mapping frame_id to reward value
            - 'cumulative_rewards': Dictionary mapping frame_id to cumulative reward
            - 'component_rewards': Dictionary mapping component name to dictionary of {frame_id: value}
        """
        return {
            'frame_rewards': self.all_frame_rewards,
            'cumulative_rewards': self.cumulative_rewards,
            'component_rewards': dict(self.all_component_rewards)
        }
    
    def reset(self):
        """Reset the reward function state for a new procedure"""
        self.current_phase = None
        self.previous_phase = None
        self.last_processed_frame = None
        self.completed_phases = set()
        self.phase_segments = None
        self.current_segment_idx = None
        self.segment_progress = {}
        self.last_transition_source = None
        self.last_transition_target = None
        self.all_frame_rewards = {}
        self.all_component_rewards = defaultdict(dict)
        self.cumulative_rewards = {}


def process_video_with_enhanced_features(metadata_df, video_id, risk_column_name='risk_score_max', 
                                        reward_function=None, smooth_factor=3.0):
    """
    Process a video to calculate rewards and extract enhanced features for each frame.
    
    Args:
        metadata_df: DataFrame containing metadata
        video_id: ID of the video to process
        risk_column_name: Name of the risk score column
        reward_function: Instance of SurgicalRewardFunction or None to create a new one
        smooth_factor: Smoothing factor for reward curves
    
    Returns:
        Tuple containing:
        - Enhanced metadata DataFrame with reward components as new columns
        - Dictionary of global metrics for the video
    """
    # Filter metadata for this video
    video_metadata = metadata_df[metadata_df['video'] == video_id].copy()
    
    if video_metadata.empty:
        raise ValueError(f"No metadata found for video {video_id}")
    
    # Get phase columns
    phase_columns = [f'p{i}' for i in range(7) if f'p{i}' in video_metadata.columns]
    
    # Extract phase segments
    _, phase_segments, _ = load_data_from_metadata(metadata_df, video_id, [risk_column_name])
    
    # Create reward function if not provided
    if reward_function is None:
        reward_function = SurgicalRewardFunction()
    
    # Reset reward function state
    reward_function.reset()
    
    # Calculate rewards for each frame
    for _, row in video_metadata.iterrows():
        frame_id = row['frame']
        
        # Determine current phase
        phase_id = None
        for i, col in enumerate(phase_columns):
            if col in row and row[col] == 1:
                phase_id = i
                break
        
        # Get risk score
        risk_score = row[risk_column_name] if risk_column_name in row else None
        
        # Prepare frame data
        frame_data = {
            'frame_id': frame_id,
            'phase_id': phase_id,
            'risk_scores': {risk_column_name: risk_score} if risk_score is not None else {}
        }
        
        # Calculate reward for this frame
        is_terminal = (frame_id == video_metadata['frame'].iloc[-1])
        reward_function.calculate_reward(frame_data, is_terminal, phase_segments)
    
    # Get all calculated rewards and components
    all_rewards = reward_function.get_all_rewards()
    
    # Add reward components as new columns to the metadata
    frame_rewards = all_rewards['frame_rewards']
    cumulative_rewards = all_rewards['cumulative_rewards']
    component_rewards = all_rewards['component_rewards']
    
    # Add frame rewards
    video_metadata['frame_reward'] = video_metadata['frame'].map(
        lambda x: frame_rewards.get(x, 0.0)
    )
    
    # Add cumulative rewards
    video_metadata['cumulative_reward'] = video_metadata['frame'].map(
        lambda x: cumulative_rewards.get(x, 0.0)
    )
    
    # Add component rewards as separate columns
    for component, values in component_rewards.items():
        video_metadata[component] = video_metadata['frame'].map(
            lambda x: values.get(x, 0.0)
        )
    
    # Apply smoothing to the reward columns if requested
    if smooth_factor > 0:
        columns_to_smooth = ['frame_reward'] + list(component_rewards.keys())
        for col in columns_to_smooth:
            if col in video_metadata.columns:
                values = video_metadata[col].values
                smoothed_values = gaussian_filter1d(values, sigma=smooth_factor)
                video_metadata[f'{col}_smoothed'] = smoothed_values
    
    # Calculate global metrics for the video
    global_metrics = calculate_global_metrics(video_metadata, phase_segments, component_rewards)
    
    return video_metadata, global_metrics


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


def extract_enhanced_frame_features(metadata_df, video_id, risk_column='risk_score_max'):
    """
    Extract enhanced features for each frame in a video, including reward components.
    
    Args:
        metadata_df: DataFrame with metadata (should already have reward components as columns)
        video_id: ID of the video to process
        risk_column: Column name for risk scores
    
    Returns:
        Dictionary mapping frame_id to feature vector and additional info
    """
    # Filter for this video
    video_data = metadata_df[metadata_df['video'] == video_id].copy()
    
    if video_data.empty:
        raise ValueError(f"No data found for video {video_id}")
    
    # Find phase columns
    phase_columns = [col for col in video_data.columns if col.startswith('p') and col[1:].isdigit()]
    
    # Find reward component columns
    reward_columns = [col for col in video_data.columns if any(col.startswith(prefix) for prefix in 
                                                             ['phase_progress', 'risk_', 'time_penalty', 
                                                              'completion_reward', 'frame_reward', 'cumulative_reward'])]
    
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
        
        # Get reward components
        reward_components = {col: row[col] if col in row else 0.0 for col in reward_columns}
        
        # Create base feature vector
        base_features = [risk_score] + phase_encoding
        
        # Create enhanced feature vector with reward components
        enhanced_features = base_features + [reward_components[col] for col in reward_columns if col in reward_components]
        
        frame_features[frame_id] = {
            'base_features': base_features,
            'enhanced_features': enhanced_features,
            'phase_id': phase_id,
            'risk_score': risk_score,
            'reward_components': reward_components
        }
    
    return frame_features


def calculate_global_metrics(video_metadata, phase_segments, component_rewards):
    """
    Calculate global metrics for a video based on frame-level reward components.
    
    Args:
        video_metadata: DataFrame with metadata for a single video (including reward components)
        phase_segments: DataFrame with phase segments information
        component_rewards: Dictionary of component rewards from reward function
    
    Returns:
        Dictionary of global metrics
    """
    metrics = {}
    
    # 1. Phase Coverage
    if phase_segments is not None:
        unique_phases = phase_segments['phase_id'].unique()
        metrics['unique_phase_count'] = len(unique_phases)
        metrics['phase_coverage'] = round(len(unique_phases) / 7, 4)  # Assuming 7 possible phases
        
        # Phase durations
        phase_durations = {}
        for phase_id in unique_phases:
            segments = phase_segments[phase_segments['phase_id'] == phase_id]
            total_frames = sum(row['end_frame'] - row['start_frame'] + 1 for _, row in segments.iterrows())
            phase_durations[f'phase_{phase_id}_frames'] = total_frames
            phase_durations[f'phase_{phase_id}_percentage'] = round(total_frames / len(video_metadata) * 100, 4)
        
        metrics.update(phase_durations)
    
    # 2. Risk Profile
    if 'risk_value' in component_rewards:
        risk_values = list(component_rewards['risk_value'].values())
        metrics['avg_risk'] = round(np.mean(risk_values), 4)
        metrics['max_risk'] = round(np.max(risk_values), 4)
        metrics['risk_std'] = round(np.std(risk_values), 4)
        
        # Critical risk events (risk > 4.0)
        critical_risk_count = sum(1 for risk in risk_values if risk > 4.0)
        metrics['critical_risk_events'] = critical_risk_count
        metrics['critical_risk_percentage'] = round(critical_risk_count / len(risk_values) * 100 if risk_values else 0, 4)
    
    # 3. Phase Transitions
    if 'phase_transition_reward' in component_rewards:
        transition_rewards = list(component_rewards['phase_transition_reward'].values())
        transition_count = sum(1 for reward in transition_rewards if reward > 0)
        metrics['phase_transition_count'] = transition_count
        
        if phase_segments is not None and len(unique_phases) > 0:
            metrics['transition_efficiency'] = round(transition_count / len(unique_phases), 4)
    
    # 4. Progress Metrics
    if 'phase_progress' in component_rewards and phase_segments is not None:
        progress_values = list(component_rewards['phase_progress'].values())
        metrics['avg_phase_progress'] = round(np.mean(progress_values), 4)
        
        # Calculate progress efficiency (how quickly phases are completed)
        if len(unique_phases) > 0:
            progress_by_phase = defaultdict(list)
            for frame_id, progress in component_rewards['phase_progress'].items():
                phase_id = None
                for _, row in phase_segments.iterrows():
                    if frame_id >= row['start_frame'] and frame_id <= row['end_frame']:
                        phase_id = row['phase_id']
                        break
                
                if phase_id is not None:
                    progress_by_phase[phase_id].append(progress)
            
            # Calculate progress rate for each phase
            progress_rates = {}
            for phase_id, progress_list in progress_by_phase.items():
                if len(progress_list) > 1:
                    # Rate of progress per frame
                    progress_rates[f'phase_{phase_id}_progress_rate'] = round((progress_list[-1] - progress_list[0]) / len(progress_list), 4)
            
            metrics.update(progress_rates)
    
    # 5. Reward Summary
    if 'frame_reward' in component_rewards:
        frame_rewards = list(component_rewards['frame_reward'].values())
        metrics['total_reward'] = round(sum(frame_rewards), 4)
        metrics['avg_frame_reward'] = round(np.mean(frame_rewards), 4)
        metrics['reward_std'] = round(np.std(frame_rewards), 4)
        metrics['positive_reward_percentage'] = round(sum(1 for r in frame_rewards if r > 0) / len(frame_rewards) * 100, 4)
    
    # 6. Global Outcome Score (0-10 scale)
    # Combine all the above metrics into a single score
    outcome_score = 0.0
    
    # Phase coverage contributes up to 3 points
    if 'phase_coverage' in metrics:
        outcome_score += metrics['phase_coverage'] * 3
    
    # Risk management contributes up to 3 points
    if 'avg_risk' in metrics:
        # Lower risk is better, scale from 0 to 5
        risk_score = max(0, 3 - (metrics['avg_risk'] / 5) * 3)
        outcome_score += risk_score
    
    # Transition efficiency contributes up to 2 points
    if 'transition_efficiency' in metrics:
        # A ratio of 1.0 is ideal (one transition per phase)
        efficiency = metrics['transition_efficiency']
        if efficiency <= 1.0:
            transition_score = efficiency * 2  # 0-1 scales to 0-2 points
        else:
            # Penalize excessive transitions
            transition_score = max(0, 2 - (efficiency - 1) * 2)
        
        outcome_score += transition_score
    
    # Reward contributes up to 2 points
    if 'total_reward' in metrics:
        # Normalize reward to 0-2 range
        # This might need tuning based on typical reward ranges
        normalized_reward = min(2, max(0, metrics['total_reward'] / 100 + 1))
        outcome_score += normalized_reward
    
    metrics['global_outcome_score'] = round(outcome_score, 4)
    
    return metrics


def save_enhanced_metadata(metadata_df, output_path, videos_to_process=None, risk_column='risk_score_max'):
    """
    Process videos, calculate reward components, and save enhanced metadata.
    
    Args:
        metadata_df: Original metadata DataFrame
        output_path: Path to save enhanced metadata
        videos_to_process: List of video IDs to process (None for all)
        risk_column: Risk score column name
        
    Returns:
        Enhanced metadata DataFrame
    """
    # Get videos to process
    if videos_to_process is None:
        videos_to_process = metadata_df['video'].unique()
    
    # Create reward function
    reward_function = SurgicalRewardFunction()
    
    # Process each video
    all_enhanced_dfs = []
    global_metrics_by_video = {}
    
    for video_id in tqdm(videos_to_process, desc="Processing videos"):
        try:
            # Process video
            enhanced_df, global_metrics = process_video_with_enhanced_features(
                metadata_df, video_id, risk_column, reward_function
            )
            
            all_enhanced_dfs.append(enhanced_df)
            global_metrics_by_video[video_id] = global_metrics
            
        except Exception as e:
            print(f"Error processing video {video_id}: {e}")
    
    # Combine all enhanced data
    enhanced_metadata = pd.concat(all_enhanced_dfs)
    
    # Save enhanced metadata
    enhanced_metadata.to_csv(output_path, index=False)
    print(f"Enhanced metadata saved to {output_path}")
    
    # Create global metrics DataFrame
    global_metrics_df = pd.DataFrame.from_dict(global_metrics_by_video, orient='index')
    global_metrics_df.index.name = 'video'
    global_metrics_df.reset_index(inplace=True)
    
    # Save global metrics
    global_metrics_path = output_path.replace('.csv', '_global_metrics.csv')
    global_metrics_df.to_csv(global_metrics_path, index=False)
    print(f"Global metrics saved to {global_metrics_path}")
    
    return enhanced_metadata, global_metrics_df


class SurgicalDataset(Dataset):
    """
    Dataset for training models to predict global rewards from frame features.
    """
    
    def __init__(self, metadata_df, global_metrics_df, target_column='global_outcome_score', 
                risk_column='risk_score_max', use_enhanced_features=True):
        """
        Initialize the dataset.
        
        Args:
            metadata_df: Enhanced metadata DataFrame with reward components
            global_metrics_df: DataFrame with global metrics for each video
            target_column: Column in global_metrics_df to use as target
            risk_column: Risk score column in metadata_df
            use_enhanced_features: Whether to use enhanced features including reward components
        """
        self.metadata_df = metadata_df
        self.global_metrics_df = global_metrics_df
        self.target_column = target_column
        self.risk_column = risk_column
        self.use_enhanced_features = use_enhanced_features
        
        # Map video IDs to target values
        self.video_targets = dict(zip(global_metrics_df['video'], global_metrics_df[target_column]))
        
        # Extract all frames and features
        self.frames = []
        self.features = []
        self.targets = []
        self.video_ids = []
        
        for video_id in tqdm(metadata_df['video'].unique(), desc="Preparing dataset"):
            target = self.video_targets.get(video_id)
            if target is None:
                continue
                
            # Extract features for this video
            video_features = extract_enhanced_frame_features(metadata_df, video_id, risk_column)
            
            for frame_id, data in video_features.items():
                if self.use_enhanced_features and 'enhanced_features' in data:
                    features = data['enhanced_features']
                else:
                    features = data['base_features']
                
                self.frames.append(frame_id)
                self.features.append(features)
                self.targets.append(target)
                self.video_ids.append(video_id)
        
        # Convert to numpy arrays
        self.features = np.array(self.features, dtype=np.float32)
        self.targets = np.array(self.targets, dtype=np.float32)
        
        # Scale features
        self.feature_scaler = StandardScaler()
        self.features = self.feature_scaler.fit_transform(self.features)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return {
            'features': torch.tensor(self.features[idx], dtype=torch.float32),
            'target': torch.tensor(self.targets[idx], dtype=np.float32),
            'frame_id': self.frames[idx],
            'video_id': self.video_ids[idx]
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


def train_prediction_model(dataset, test_size=0.2, batch_size=64, 
                         learning_rate=0.001, epochs=50, hidden_dims=[128, 64]):
    """
    Train a model to predict global rewards from frame features.
    
    Args:
        dataset: SurgicalDataset instance
        test_size: Fraction of data to use for validation
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        epochs: Number of training epochs
        hidden_dims: List of hidden layer dimensions
    
    Returns:
        Trained model
    """
    # Split data into training and validation sets
    # Need to be careful to keep frames from the same video together
    unique_videos = list(set(dataset.video_ids))
    train_videos, val_videos = train_test_split(unique_videos, test_size=test_size, random_state=42)
    
    # Create indices for train and validation sets
    train_indices = [i for i, video in enumerate(dataset.video_ids) if video in train_videos]
    val_indices = [i for i, video in enumerate(dataset.video_ids) if video in val_videos]
    
    # Create subset datasets
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model
    input_dim = dataset.features.shape[1]
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
            
            train_loss += loss.item() * len(targets)
        
        train_loss /= len(train_loader.dataset)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                features = batch['features']
                targets = batch['target']
                
                outputs = model(features)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item() * len(targets)
        
        val_loss /= len(val_loader.dataset)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Enhanced surgical reward system')
    parser.add_argument('--metadata_path', type=str, default="/home/maxboels/datasets/CholecT50/embeddings_train_set/fold0/embeddings_f0_swin_bas_129.csv",
                      help='Path to metadata CSV file')
    parser.add_argument('--new_metadata_name', type=str, default='embeddings_f0_swin_bas_129_with_enhanced.csv',
                      help='Path to save enhanced metadata CSV file')
    parser.add_argument('--output_dir', type=str, default='enhanced_data',
                      help='Directory to save enhanced data and models')
    parser.add_argument('--risk_column', type=str, default='risk_score_max',
                      help='Risk score column name')
    parser.add_argument('--video_id', type=str, default=None,
                      help='Process only this video ID (for testing)')
    parser.add_argument('--train_model', type=bool, default=False,
                      help='Train a prediction model after processing')
    parser.add_argument('--epochs', type=int, default=50,
                      help='Number of training epochs')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load metadata
    metadata_df = pd.read_csv(args.metadata_path)
    print(f"Loaded metadata with {len(metadata_df)} rows")
    
    # Process videos to extract enhanced metadata
    videos_to_process = [args.video_id] if args.video_id else None
    
    output_path = os.path.join(args.output_dir, args.new_metadata_name)
    enhanced_metadata, global_metrics = save_enhanced_metadata(
        metadata_df, output_path, videos_to_process, args.risk_column
    )
    
    # Train prediction model if requested
    if args.train_model:
        print("Creating dataset for model training...")
        dataset = SurgicalDataset(enhanced_metadata, global_metrics, 
                                 risk_column=args.risk_column, 
                                 use_enhanced_features=True)
        
        print(f"Dataset created with {len(dataset)} samples")
        
        print("Training prediction model...")
        model = train_prediction_model(dataset, epochs=args.epochs)
        
        # Save model
        model_path = os.path.join(args.output_dir, 'reward_prediction_model.pt')
        torch.save({
            'model_state_dict': model.state_dict(),
            'feature_scaler': dataset.feature_scaler,
            'input_dim': dataset.features.shape[1],
            'hidden_dims': [128, 64],  # Same as used in training
        }, model_path)
        
        print(f"Model saved to {model_path}")
    
    print("Done!")

if __name__ == "__main__":
    main()