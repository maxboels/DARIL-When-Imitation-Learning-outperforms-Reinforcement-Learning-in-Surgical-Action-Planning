import pandas as pd
import numpy as np
from tqdm import tqdm


def compute_phase_transition_rewards(metadata_df, video_id_col='video', n_phases=7, 
                                    reward_window=5, 
                                    phase_importance=None,
                                    reward_value=1.0):
    """
    Compute rewards for frames at the beginning of each new phase in surgical videos.
    
    Args:
        metadata_df: DataFrame with phase indicators (p0-p{n_phases-1})
        video_id_col: Column name containing video identifiers
        n_phases: Number of phases
        reward_window: Number of frames at the beginning of a phase to reward
        phase_importance: Importance weight for each phase (defaults to equal)
        reward_value: Reward value to assign at the beginning of each phase
        
    Returns:
        DataFrame with added phase_initiation_reward column
    """
    import numpy as np
    import pandas as pd
    
    # Create a copy of the DataFrame
    df = metadata_df.copy()
    
    # Default phase importance if not provided
    if phase_importance is None:
        phase_importance = [1.0] * n_phases
    
    # Initialize the reward column
    df['phase_initiation_reward'] = 0.0
    
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
    for video_id in tqdm(video_ids, desc="[PHASE TRANSITION] Processing videos"):
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
        video_df = df.loc[video_mask].copy()
        phase_changes = video_df['current_phase'].diff() != 0
        video_transition_indices = video_df.index[phase_changes].tolist()
        
        # First frame of video is also considered a phase initiation
        if len(video_indices) > 0:
            first_idx = video_indices[0]
            first_phase = df.loc[first_idx, 'current_phase']
            
            if first_phase != -1:  # Make sure it's a valid phase
                phase_idx = int(first_phase)
                phase_scale = phase_importance[phase_idx]
                
                # Assign rewards to first frames of video
                end_pos = min(reward_window, len(video_indices))
                for i, idx in enumerate(video_indices[:end_pos]):
                    df.loc[idx, 'phase_initiation_reward'] = reward_value * phase_scale
        
        # For each transition point, add rewards to the beginning frames of new phase
        for trans_idx in video_transition_indices:
            # Get the position in the original indices list
            try:
                pos = video_indices.index(trans_idx)
                
                # Get the new phase we're entering
                new_phase = df.loc[trans_idx, 'current_phase']
                
                if new_phase != -1:  # Make sure it's a valid phase
                    phase_idx = int(new_phase)
                    phase_scale = phase_importance[phase_idx]
                    
                    # Reward the first few frames of the new phase
                    # (including the transition frame itself)
                    end_pos = min(len(video_indices), pos + reward_window)
                    for idx in video_indices[pos:end_pos]:
                        df.loc[idx, 'phase_initiation_reward'] = reward_value * phase_scale
                
            except ValueError:
                # This shouldn't happen, but just in case
                continue
    
    # Clean up temporary columns
    if 'current_phase' in df.columns:
        df = df.drop(columns=['current_phase'])
    if '_temp_video_id' in df.columns:
        df = df.drop(columns=['_temp_video_id'])
    
    return df

def visualize_phase_initiation_rewards(df, video_id_col='video_id', video_id=None, window_range=None, 
                                    figsize=(12, 8)):
    """
    Visualize phase initiation rewards across a video segment.
    
    Args:
        df: DataFrame with phase_initiation_reward column
        video_id_col: Column name containing video identifiers
        video_id: Specific video ID to visualize (if None, uses first video)
        window_range: Tuple of (start_idx, end_idx) for visualization window
                     (if None, shows the entire video)
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
    
    # Define range to plot
    if window_range is None:
        start_idx = 0
        end_idx = len(df_video) - 1
    else:
        start_idx, end_idx = window_range
        start_idx = max(0, start_idx)
        end_idx = min(len(df_video) - 1, end_idx)
    
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
    
    # Plot initiation rewards
    rewards = df_video.loc[start_idx:end_idx, 'phase_initiation_reward']
    ax.plot(
        range(start_idx, end_idx + 1),
        rewards,
        linewidth=2,
        color='green',
        label='Phase Initiation Reward'
    )
    
    # Add vertical lines at phase changes
    if phase_cols:
        current_phase = None
        for i in range(start_idx, end_idx + 1):
            phase_id = None
            for p, col in enumerate(phase_cols):
                if df_video.loc[i, col] == 1:
                    phase_id = p
                    break
            
            if phase_id != current_phase and phase_id is not None:
                ax.axvline(x=i, color='black', linestyle='--', alpha=0.7)
                ax.text(i, ax.get_ylim()[1] * 0.9, f'Phase {phase_id}', rotation=90, verticalalignment='top')
                current_phase = phase_id
    
    # Add frame numbers
    ax.set_xlabel('Frame Number')
    ax.set_ylabel('Reward Value')
    ax.set_title(f'Phase Initiation Rewards (Video {video_id})' 
               if video_id_col in df.columns else 'Phase Initiation Rewards')
    ax.legend()
    ax.grid(alpha=0.3)
    
    return fig