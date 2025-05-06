import pandas as pd
import numpy as np
from tqdm import tqdm

def compute_phase_completion_rewards(metadata_df, video_id_col='video_id', n_phases=7, 
                                    transition_window=30,
                                    phase_importance=None,
                                    max_reward=1.0,
                                    reward_function='exponential',
                                    reward_distribution='left_sided'):
    """
    Compute rewards for frames near phase transitions and completions in surgical videos.
    
    Args:
        metadata_df: DataFrame with phase indicators (p0-p{n_phases-1})
        video_id_col: Column name containing video identifiers
        n_phases: Number of phases
        transition_window: Number of frames around transition to distribute rewards
        phase_importance: Importance weight for each phase (defaults to equal)
        max_reward: Maximum reward at exact transition point
        reward_function: Shape of reward curve ('linear', 'exponential', 'sigmoid', 'gaussian')
        reward_distribution: Type of distribution ('left_sided' or 'bell_curve')
            - 'left_sided': Rewards only frames before transition (phase completion)
            - 'bell_curve': Rewards frames before and after transition (centered on transition)
        
    Returns:
        DataFrame with added transition_reward column
    """
    import numpy as np
    import pandas as pd
    
    # Create a copy of the DataFrame
    df = metadata_df.copy()
    
    # Default phase importance if not provided
    if phase_importance is None:
        phase_importance = [1.0] * n_phases
    
    # Initialize the reward column
    df['transition_reward'] = 0.0
    
    # Get phase indicators
    phase_cols = [f'p{i}' for i in range(n_phases)]
    
    # Get unique video IDs
    if video_id_col in df.columns:
        video_ids = df[video_id_col].unique()
    else:
        # If no video ID column is found, assume all rows are from a single video
        print(f"Warning: '{video_id_col}' column not found. Treating all data as a single video.")
        df['_temp_video_id'] = 0
        video_id_col = '_temp_video_id'
        video_ids = [0]
    
    # Process each video separately
    for video_id in video_ids:
        # Get data for this video only
        video_mask = df[video_id_col] == video_id
        
        # Skip if no data for this video
        if not any(video_mask):
            continue
        
        # Get indices for this video
        video_indices = df.index[video_mask].tolist()
        
        # Create temporary phase column for this video
        df.loc[video_mask, 'current_phase'] = -1
        for i, col in enumerate(phase_cols):
            df.loc[video_mask & (df[col] == 1), 'current_phase'] = i
        
        # Find phase transition points within this video
        # Use numeric indices relative to the filtered dataframe
        video_df = df.loc[video_mask].copy()
        phase_changes = video_df['current_phase'].diff() != 0
        video_transition_indices = video_df.index[phase_changes].tolist()
        
        # For each transition point, add rewards around it based on distribution type
        for trans_idx in video_transition_indices:
            # Get the position in the original indices list
            try:
                pos = video_indices.index(trans_idx)
                if pos > 0:  # Skip the first frame if it's marked as a transition
                    # Get the phase we're completing
                    prev_idx = video_indices[pos-1]
                    completed_phase = df.loc[prev_idx, 'current_phase'].astype(int)
                    if completed_phase == -1:
                        continue
                    
                    # Scale reward by the importance of this transition
                    phase_scale = phase_importance[completed_phase]
                    
                    # Left side of the distribution (before transition)
                    # Get up to transition_window frames before, but only within this video
                    start_pos = max(0, pos - transition_window)
                    for i, idx in enumerate(video_indices[start_pos:pos]):
                        # Normalized distance to transition (0 = far, 1 = at transition)
                        progress = i / max(1, (pos - start_pos))
                        
                        # Calculate reward based on selected function shape
                        if reward_function == 'linear':
                            reward = max_reward * progress * phase_scale
                        elif reward_function == 'exponential':
                            reward = max_reward * phase_scale * (np.exp(3 * progress) - 1) / (np.exp(3) - 1)
                        elif reward_function == 'sigmoid':
                            # Sigmoid centered at 0.7 progress point
                            reward = max_reward * phase_scale / (1 + np.exp(-10 * (progress - 0.7)))
                        elif reward_function == 'gaussian' and reward_distribution == 'left_sided':
                            # For left_sided gaussian, we use only the left half of the bell curve
                            sigma = 0.3  # Controls width of the curve
                            # Normalize distance: 0 at start of window, 3*sigma at transition
                            x = 3 * sigma * progress
                            # Use left half of gaussian: x ranges from -3*sigma to 0
                            reward = max_reward * phase_scale * np.exp(-((-3*sigma + x)**2) / (2 * sigma**2))
                        else:  # Default to linear
                            reward = max_reward * progress * phase_scale
                        
                        df.loc[idx, 'transition_reward'] = max(df.loc[idx, 'transition_reward'], reward)
                    
                    # Right side of the distribution (after transition) - only for bell_curve
                    if reward_distribution == 'bell_curve':
                        # Get the phase we're entering
                        new_phase = df.loc[trans_idx, 'current_phase']
                        if new_phase == -1:
                            continue
                        
                        # The window of frames following the transition point, only within this video
                        end_pos = min(len(video_indices), pos + transition_window + 1)
                        for i, idx in enumerate(video_indices[pos:end_pos]):
                            # Normalized distance from transition (0 = at transition, 1 = far)
                            distance = i / max(1, (end_pos - pos))
                            
                            # Calculate reward based on selected function shape
                            if reward_function == 'linear':
                                reward = max_reward * (1 - distance) * phase_scale
                            elif reward_function == 'exponential':
                                reward = max_reward * phase_scale * (np.exp(3 * (1 - distance)) - 1) / (np.exp(3) - 1)
                            elif reward_function == 'sigmoid':
                                # Sigmoid with steeper decline
                                reward = max_reward * phase_scale / (1 + np.exp(10 * (distance - 0.3)))
                            elif reward_function == 'gaussian':
                                # True bell curve using Gaussian
                                sigma = 0.3  # Controls width of the bell curve
                                # Normalize distance: 0 at transition, 3*sigma at end of window
                                x = 3 * sigma * distance
                                # Use right half of gaussian: x ranges from 0 to 3*sigma
                                reward = max_reward * phase_scale * np.exp(-(x**2) / (2 * sigma**2))
                            else:  # Default to linear
                                reward = max_reward * (1 - distance) * phase_scale
                            
                            df.loc[idx, 'transition_reward'] = max(df.loc[idx, 'transition_reward'], reward)
                
            except ValueError:
                # This shouldn't happen, but just in case
                continue
        
        # Special case: reward completion of the final phase at the end of each video
        if video_indices:
            final_idx = video_indices[-1]
            final_phase = df.loc[final_idx, 'current_phase'].astype(int)
            if final_phase != -1:
                phase_scale = phase_importance[final_phase]
                
                # Get up to transition_window frames before the end, within this video
                end_pos = len(video_indices)
                start_pos = max(0, end_pos - transition_window)
                
                for i, idx in enumerate(video_indices[start_pos:end_pos]):
                    progress = i / max(1, (end_pos - start_pos))
                    
                    if reward_function == 'linear':
                        reward = max_reward * progress * phase_scale
                    elif reward_function == 'exponential':
                        reward = max_reward * phase_scale * (np.exp(3 * progress) - 1) / (np.exp(3) - 1)
                    elif reward_function == 'sigmoid':
                        reward = max_reward * phase_scale / (1 + np.exp(-10 * (progress - 0.7)))
                    elif reward_function == 'gaussian' and reward_distribution == 'left_sided':
                        sigma = 0.3
                        x = 3 * sigma * progress
                        reward = max_reward * phase_scale * np.exp(-((-3*sigma + x)**2) / (2 * sigma**2))
                    else:
                        reward = max_reward * progress * phase_scale
                    
                    df.loc[idx, 'transition_reward'] = max(df.loc[idx, 'transition_reward'], reward)
    
    # Clean up temporary columns
    if 'current_phase' in df.columns:
        df = df.drop(columns=['current_phase'])
    if '_temp_video_id' in df.columns:
        df = df.drop(columns=['_temp_video_id'])
    
    return df

def visualize_transition_rewards(df, video_id_col='video_id', video_id=None, transition_idx=None, 
                              window_size=60, figsize=(12, 8)):
    """
    Visualize phase transition rewards around a specific transition point.
    
    Args:
        df: DataFrame with transition_reward column
        video_id_col: Column name containing video identifiers
        video_id: Specific video ID to visualize (if None, uses first video)
        transition_idx: Index of a phase transition to visualize (if None, finds one automatically)
        window_size: Number of frames to show before and after transition
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Filter to specific video if needed
    if video_id_col in df.columns:
        if video_id is None:
            video_id = df[video_id_col].iloc[0]
        df_video = df[df[video_id_col] == video_id].copy()
    else:
        df_video = df.copy()
    
    # Reset index for easier plotting
    df_video = df_video.reset_index(drop=True)
    
    # If no transition specified, try to find one
    if transition_idx is None:
        # Look for a significant change in rewards
        reward_diff = df_video['transition_reward'].diff().abs()
        if reward_diff.max() > 0:
            transition_idx = reward_diff.idxmax()
        else:
            # Just use the middle of the dataframe
            transition_idx = len(df_video) // 2
    
    # Define range to plot
    start_idx = max(0, transition_idx - window_size)
    end_idx = min(len(df_video) - 1, transition_idx + window_size)
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Extract phase columns
    phase_cols = [col for col in df_video.columns if col.startswith('p') and len(col) == 2 and col[1].isdigit()]
    
    # Plot phases as background colors
    if phase_cols:
        for i, col in enumerate(phase_cols):
            # Get phase data in window
            phase_data = df_video.loc[start_idx:end_idx, col].astype(float)
            # Plot as filled area
            ax.fill_between(
                range(start_idx, end_idx + 1),
                0, 
                phase_data, 
                alpha=0.2,
                label=f"Phase {i}"
            )
    
    # Plot transition rewards
    rewards = df_video.loc[start_idx:end_idx, 'transition_reward']
    ax.plot(
        range(start_idx, end_idx + 1),
        rewards,
        linewidth=2,
        color='red',
        label='Transition Reward'
    )
    
    # Add vertical line at transition
    ax.axvline(x=transition_idx, color='black', linestyle='--', alpha=0.7)
    ax.text(transition_idx, ax.get_ylim()[1] * 0.9, 'Transition', rotation=90, verticalalignment='top')
    
    # Add frame numbers
    ax.set_xlabel('Frame Number')
    ax.set_ylabel('Reward Value')
    ax.set_title(f'Phase Transition Rewards (Video {video_id})' 
                if video_id_col in df.columns else 'Phase Transition Rewards')
    ax.legend()
    ax.grid(alpha=0.3)
    
    return fig