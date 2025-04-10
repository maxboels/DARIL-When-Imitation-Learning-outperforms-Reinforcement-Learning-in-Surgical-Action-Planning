import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from scipy.ndimage import gaussian_filter1d
import pandas as pd
import os
import json
import argparse
from tqdm import tqdm
from collections import defaultdict

# Import the SurgicalRewardFunction from the surgical_reward.py file
# Make sure this file is in the same directory or in your Python path
from surgical_reward import SurgicalRewardFunction

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

def calculate_rewards_for_video(metadata_df, video_id, risk_column_name='risk_score_max', 
                               reward_function=None, smooth_factor=3.0):
    """
    Calculate rewards for each frame in a video using the SurgicalRewardFunction.
    
    Args:
        metadata_df: DataFrame containing metadata with risk scores and phase information
        video_id: ID of the video
        risk_column_name: Column name for risk scores
        reward_function: Instance of SurgicalRewardFunction, or None to create a default one
        smooth_factor: Smoothing factor for reward values
        
    Returns:
        Tuple containing:
        - Dictionary mapping frame_id to reward value
        - Dictionary mapping frame_id to cumulative reward up to that frame
        - Dictionary mapping reward component names to dictionaries of {frame_id: component_value}
    """
    # Filter metadata for this video and sort by frame
    video_metadata = metadata_df[metadata_df['video'] == video_id].sort_values('frame')
    
    if video_metadata.empty:
        raise ValueError(f"No metadata found for video {video_id}")
    
    # Get phase columns
    phase_columns = [f'p{i}' for i in range(7) if f'p{i}' in video_metadata.columns]
    
    # Create reward function if not provided
    if reward_function is None:
        reward_function = SurgicalRewardFunction()
    
    # Reset reward function state
    reward_function.reset()
    
    # Calculate rewards for each frame
    frame_rewards = {}
    cumulative_rewards = {}
    component_rewards = defaultdict(dict)
    
    cumulative_reward = 0.0
    
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
        reward, components = reward_function.calculate_reward(frame_data, is_terminal)
        
        # Store results
        frame_rewards[frame_id] = reward
        cumulative_reward += reward
        cumulative_rewards[frame_id] = cumulative_reward
        
        # Store reward components
        for component, value in components.items():
            component_rewards[component][frame_id] = value
    
    # Apply smoothing to reward values if requested
    if smooth_factor > 0:
        # Sort frames
        sorted_frames = sorted(frame_rewards.keys())
        rewards = [frame_rewards[frame] for frame in sorted_frames]
        
        # Apply smoothing
        smoothed_rewards = gaussian_filter1d(rewards, sigma=smooth_factor)
        
        # Update dictionary
        for i, frame in enumerate(sorted_frames):
            frame_rewards[frame] = smoothed_rewards[i]
        
        # Recalculate cumulative rewards
        cumulative_reward = 0.0
        for frame in sorted_frames:
            cumulative_reward += frame_rewards[frame]
            cumulative_rewards[frame] = cumulative_reward
    
    return frame_rewards, cumulative_rewards, dict(component_rewards)

def plot_rewards_and_risk(video_id, risk_scores_dict, frame_rewards, cumulative_rewards, 
                          component_rewards=None, phases_df=None, frame_phases=None,
                          smooth_factor=5, output_dir=None, show_components=True):
    """
    Plot rewards and risk scores over time with surgical phase coloring.
    
    Args:
        video_id: ID of the video
        risk_scores_dict: Dictionary with risk score column names as keys and dictionaries of 
                          {frame_id: risk_score} as values
        frame_rewards: Dictionary mapping frame_id to reward value
        cumulative_rewards: Dictionary mapping frame_id to cumulative reward
        component_rewards: Optional dictionary mapping component names to dictionaries of
                          {frame_id: component_value}
        phases_df: DataFrame with phase information
        frame_phases: Dictionary mapping frame_id to phase_id
        smooth_factor: Gaussian smoothing factor for display (higher = smoother)
        output_dir: Directory to save the plot
        show_components: Whether to show reward components in separate subplots
    """
    # Set up figure with subplots
    n_plots = 3  # Risk, reward, cumulative reward
    if show_components and component_rewards:
        n_plots += len(component_rewards)
    
    fig, axs = plt.subplots(n_plots, 1, figsize=(14, 4 * n_plots), sharex=True)
    
    # Sort frames
    all_frames = set()
    for risk_type, risk_scores in risk_scores_dict.items():
        all_frames.update(risk_scores.keys())
    all_frames.update(frame_rewards.keys())
    sorted_frames = sorted(all_frames)
    
    # Create x-axis array (frame indices)
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
    
    # Add phase backgrounds
    risk_ymin, risk_ymax = ax_risk.get_ylim()
    add_phase_backgrounds(ax_risk, risk_ymin, risk_ymax)
    
    # Set labels
    ax_risk.set_ylabel('Risk Score')
    ax_risk.set_title(f'Risk Scores for Video {video_id}')
    ax_risk.grid(True, linestyle='--', alpha=0.7)
    ax_risk.legend(loc='upper right')
    
    # === PLOT 2: FRAME REWARDS ===
    ax_reward = axs[1]
    
    # Convert rewards to arrays for plotting
    reward_x = [frame_to_index[frame] for frame in sorted_frames if frame in frame_rewards]
    reward_y = [frame_rewards[frame] for frame in sorted_frames if frame in frame_rewards]
    
    # Plot rewards
    ax_reward.plot(reward_x, reward_y, color='green', linewidth=2)
    
    # Add horizontal line at y=0
    ax_reward.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    
    # Add phase backgrounds
    reward_ymin, reward_ymax = ax_reward.get_ylim()
    add_phase_backgrounds(ax_reward, reward_ymin, reward_ymax)
    
    # Set labels
    ax_reward.set_ylabel('Reward Value')
    ax_reward.set_title(f'Frame Rewards for Video {video_id}')
    ax_reward.grid(True, linestyle='--', alpha=0.7)
    
    # === PLOT 3: CUMULATIVE REWARDS ===
    ax_cumul = axs[2]
    
    # Convert cumulative rewards to arrays for plotting
    cumul_x = [frame_to_index[frame] for frame in sorted_frames if frame in cumulative_rewards]
    cumul_y = [cumulative_rewards[frame] for frame in sorted_frames if frame in cumulative_rewards]
    
    # Plot cumulative rewards
    ax_cumul.plot(cumul_x, cumul_y, color='blue', linewidth=2)
    
    # Add horizontal line at y=0
    ax_cumul.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    
    # Add phase backgrounds
    cumul_ymin, cumul_ymax = ax_cumul.get_ylim()
    add_phase_backgrounds(ax_cumul, cumul_ymin, cumul_ymax)
    
    # Set labels
    ax_cumul.set_ylabel('Cumulative Reward')
    ax_cumul.set_title(f'Cumulative Rewards for Video {video_id}')
    ax_cumul.grid(True, linestyle='--', alpha=0.7)
    
    # === PLOTS 4+: REWARD COMPONENTS (if requested) ===
    if show_components and component_rewards:
        component_colors = {
            'phase_progress_reward': 'green',
            'phase_transition_reward': 'blue',
            'risk_penalty': 'red',
            'time_penalty': 'orange',
            'completion_reward': 'purple'
        }
        
        for i, (component, rewards) in enumerate(component_rewards.items(), start=3):
            ax_comp = axs[i]
            
            # Convert component rewards to arrays for plotting
            comp_x = [frame_to_index[frame] for frame in sorted_frames if frame in rewards]
            comp_y = [rewards[frame] for frame in sorted_frames if frame in rewards]
            
            # Plot component rewards
            color = component_colors.get(component, 'gray')
            ax_comp.plot(comp_x, comp_y, color=color, linewidth=2)
            
            # Add horizontal line at y=0 if there are negative values
            if min(comp_y) < 0:
                ax_comp.axhline(y=0, color='black', linestyle='--', alpha=0.7)
            
            # Add phase backgrounds
            comp_ymin, comp_ymax = ax_comp.get_ylim()
            add_phase_backgrounds(ax_comp, comp_ymin, comp_ymax)
            
            # Set labels
            ax_comp.set_ylabel('Value')
            ax_comp.set_title(f'{component} for Video {video_id}')
            ax_comp.grid(True, linestyle='--', alpha=0.7)
    
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
        output_path = os.path.join(output_dir, f"{video_id}_reward_visualization.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    else:
        plt.show()
    
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Visualize rewards and risk scores for surgical videos')
    parser.add_argument('--video_id', type=str, default=None, 
                      help='Specific video ID to visualize (if not specified, will process all videos in metadata)')
    parser.add_argument('--metadata_path', type=str, default="/home/maxboels/datasets/CholecT50/embeddings_train_set/fold0/embeddings_f0_swin_bas_129.csv",
                      help='Path to metadata CSV file containing risk scores and phase information')
    parser.add_argument('--risk_column', type=str, default='risk_score_max',
                      help='Risk score column to use for reward calculation')
    parser.add_argument('--risk_weight', type=float, default=1.0,
                      help='Weight for risk penalties in reward function')
    parser.add_argument('--phase_importance', type=int, nargs='+', default=None,
                      help='Importance weights for phases (space-separated list of 7 values)')
    parser.add_argument('--smooth_factor', type=float, default=5.0, 
                      help='Smoothing factor for curves (0 = no smoothing)')
    parser.add_argument('--output_dir', type=str, default="reward_visualizations", 
                      help='Directory to save output plots')
    parser.add_argument('--skip_existing', action='store_true',
                      help='Skip videos that already have visualization files')
    parser.add_argument('--show_components', action='store_true',
                      help='Show individual reward components in separate plots')
    
    args = parser.parse_args()
    
    try:
        # Check if metadata file exists
        if not os.path.exists(args.metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {args.metadata_path}")
        
        # Load metadata
        print(f"Loading metadata from {args.metadata_path}")
        metadata_df = pd.read_csv(args.metadata_path)
        
        # Get all unique video IDs from metadata
        video_ids = []
        if args.video_id:
            # Process only the specified video
            video_ids = [args.video_id]
            if args.video_id not in metadata_df['video'].unique():
                raise ValueError(f"Video ID '{args.video_id}' not found in metadata")
        else:
            # Process all videos in metadata
            video_ids = metadata_df['video'].unique().tolist()
            print(f"Found {len(video_ids)} videos in metadata")
        
        # Create output directory
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            print(f"Output directory: {args.output_dir}")
        
        # Set up phase weights if provided
        phase_weights = None
        if args.phase_importance and len(args.phase_importance) == 7:
            phase_weights = {i: float(w) for i, w in enumerate(args.phase_importance)}
            print(f"Using custom phase weights: {phase_weights}")
        
        # Initialize reward function
        reward_function = SurgicalRewardFunction(
            phase_weights=phase_weights,
            risk_weight=args.risk_weight,
            smoothing_factor=args.smooth_factor
        )
        
        # Process each video
        success_count = 0
        error_count = 0
        skip_count = 0
        
        for video_id in tqdm(video_ids, desc="Processing videos"):
            try:
                # Skip if visualization already exists
                output_path = os.path.join(args.output_dir, f"{video_id}_reward_visualization.png")
                if args.skip_existing and os.path.exists(output_path):
                    print(f"Skipping video {video_id} (visualization already exists)")
                    skip_count += 1
                    continue
                
                # Load risk scores and phase information
                risk_scores_dict, phases_df, frame_phases = load_data_from_metadata(
                    metadata_df, video_id, [args.risk_column]
                )
                
                # Calculate rewards
                frame_rewards, cumulative_rewards, component_rewards = calculate_rewards_for_video(
                    metadata_df, video_id, args.risk_column, reward_function, args.smooth_factor
                )
                
                # Plot rewards and risk scores
                plot_rewards_and_risk(
                    video_id, risk_scores_dict, frame_rewards, cumulative_rewards, 
                    component_rewards, phases_df, frame_phases,
                    args.smooth_factor, args.output_dir, args.show_components
                )
                
                success_count += 1
                print(f"Successfully processed video {video_id}")
                
            except Exception as e:
                print(f"Error processing video {video_id}: {e}")
                error_count += 1
        
        # Print summary
        print(f"\nSummary:")
        print(f"  Total videos: {len(video_ids)}")
        print(f"  Successfully processed: {success_count}")
        print(f"  Skipped (already exists): {skip_count}")
        print(f"  Errors: {error_count}")
        
        if success_count > 0:
            print(f"\nOutput saved to: {os.path.abspath(args.output_dir)}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    main()