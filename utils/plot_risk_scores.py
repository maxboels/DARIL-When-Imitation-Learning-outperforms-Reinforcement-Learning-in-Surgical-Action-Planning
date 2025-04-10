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
        return all_risk_scores, None
    
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
        return all_risk_scores, None
    
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
    
    return all_risk_scores, pd.DataFrame(phases_data)

def load_risk_scores_from_json(video_id, risk_score_root, frame_risk_agg='max'):
    """
    Load risk scores for a specific video from JSON files.
    
    Args:
        video_id: ID of the video
        risk_score_root: Root directory for risk score files
        frame_risk_agg: Method to aggregate risk scores ('mean' or 'max')
    
    Returns:
        Dictionary with frame IDs as keys and risk scores as values
    """
    risk_score_path = os.path.join(risk_score_root, f"{video_id}_sorted_with_risk_scores_instructions_with_goals.json")
    
    if not os.path.exists(risk_score_path):
        raise FileNotFoundError(f"Risk score file not found: {risk_score_path}")
    
    with open(risk_score_path, 'r') as f:
        risk_data = json.load(f)
    
    # Extract risk scores for each frame
    frame_risk_scores = {}
    for frame_id, frame_data in risk_data.items():
        current_actions = frame_data['current_actions']
        action_risk_scores = [action['expert_risk_score'] for action in current_actions]
        
        if frame_risk_agg == 'mean':
            risk_score = np.mean(action_risk_scores) if action_risk_scores else 0
        elif frame_risk_agg == 'max':
            risk_score = np.max(action_risk_scores) if action_risk_scores else 0
        else:
            raise ValueError(f"Unsupported frame risk aggregation method: {frame_risk_agg}")
        
        frame_risk_scores[int(frame_id)] = risk_score
    
    return frame_risk_scores

def plot_risk_scores(video_id, risk_scores_dict, phases_df=None, smooth_factor=5, output_dir=None, x_axis_type='frame'):
    """
    Plot risk scores over time with surgical phase coloring.
    
    Args:
        video_id: ID of the video
        risk_scores_dict: Dictionary with risk score column names as keys and dictionaries of 
                          {frame_id: risk_score} as values, or a single dictionary of 
                          {frame_id: risk_score} for backward compatibility
        phases_df: DataFrame with phase information
        smooth_factor: Gaussian smoothing factor (higher = smoother)
        output_dir: Directory to save the plot
        x_axis_type: Type of x-axis ('frame' or 'time' if timestamp data is available)
    """
    # Check if we have multiple risk score types or just one
    if not isinstance(next(iter(risk_scores_dict.values())), dict):
        # Backward compatibility - single risk score dictionary
        risk_scores_dict = {'risk_score': risk_scores_dict}
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Colors for different risk score types (if multiple)
    risk_type_colors = plt.cm.tab10.colors
    
    # Keep track of all scores for y-axis limits
    all_scores = []
    
    # Plot each risk score type
    for i, (risk_type, risk_scores) in enumerate(risk_scores_dict.items()):
        # Sort frames by ID
        frame_ids = sorted(risk_scores.keys())
        scores = [risk_scores[frame_id] for frame_id in frame_ids]
        all_scores.extend(scores)
        
        # Apply smoothing if requested
        if smooth_factor > 0:
            scores = gaussian_filter1d(scores, sigma=smooth_factor)
        
        if len(risk_scores_dict) == 1:
            # For a single risk score type, use color gradient based on risk intensity
            norm = plt.Normalize(0, 5)  # Assuming risk scores range from 0 to 5
            cmap = plt.cm.get_cmap('RdYlGn_r')  # Red-Yellow-Green color map (reversed)
            
            # Create a colored line plot based on risk intensity
            points = np.array([range(len(scores)), scores]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segments, cmap=cmap, norm=norm)
            lc.set_array(np.array(scores))
            lc.set_linewidth(2)
            line = ax.add_collection(lc)
            
            # Add colorbar
            cbar = fig.colorbar(line, ax=ax)
            cbar.set_label('Risk Score Intensity')
        else:
            # For multiple risk score types, use different colors for each type
            display_name = risk_type.replace('risk_score_', '')
            ax.plot(scores, label=display_name, color=risk_type_colors[i % len(risk_type_colors)], linewidth=2)
    
    # If we have phase information, add colored background sections
    if phases_df is not None:
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
        
        # Plot colored background for each phase
        ymin, ymax = 0, max(all_scores) * 1.1
        ax.set_ylim(ymin, ymax)
        
        # Get first frame_ids from the first risk score type
        first_risk_type = next(iter(risk_scores_dict.keys()))
        frame_ids = sorted(risk_scores_dict[first_risk_type].keys())
        
        for _, row in phases_df.iterrows():
            phase_id = row['phase_id']
            start_frame = row['start_frame']
            end_frame = row['end_frame']
            
            # Skip phases outside our frame range
            if end_frame < min(frame_ids) or start_frame > max(frame_ids):
                continue
                
            # Adjust to our frame range
            start_frame = max(start_frame, min(frame_ids))
            end_frame = min(end_frame, max(frame_ids))
            
            # Convert to indices in our array
            start_idx = frame_ids.index(start_frame) if start_frame in frame_ids else 0
            end_idx = frame_ids.index(end_frame) if end_frame in frame_ids else len(frame_ids) - 1
            
            # Add colored background
            alpha = 0.3  # Transparency
            rect = plt.Rectangle((start_idx, ymin), end_idx - start_idx, ymax - ymin, 
                                color=phase_colors.get(phase_id, '#CCCCCC'), alpha=alpha)
            ax.add_patch(rect)
        
        # Add phase legend
        phase_ids = sorted(set(phases_df['phase_id']))
        phase_labels = [phase_names.get(i, f"Phase {i}") for i in phase_ids]
        phase_handles = [plt.Rectangle((0, 0), 1, 1, color=phase_colors.get(i, '#CCCCCC')) 
                         for i in phase_ids]
        
        if len(risk_scores_dict) > 1:
            # Add risk score type legend if we have multiple types
            ax.legend(loc='upper left', title="Risk Score Types")
            
            # Add phase legend in a separate position
            phase_legend = ax.legend(phase_handles, phase_labels, loc='upper right', title="Surgical Phases")
            ax.add_artist(phase_legend)
        else:
            # Just add phase legend if we have only one risk score type
            ax.legend(phase_handles, phase_labels, loc='upper right', title="Surgical Phases")
    
    # Set axis limits and labels
    first_risk_type = next(iter(risk_scores_dict.keys()))
    num_frames = len(sorted(risk_scores_dict[first_risk_type].keys()))
    ax.set_xlim(0, num_frames - 1)
    
    if x_axis_type == 'time':
        ax.set_xlabel('Time (seconds)')
    else:
        ax.set_xlabel('Frame Number')
        
    ax.set_ylabel('Risk Score')
    ax.set_title(f'Risk Score Visualization for Video {video_id}')
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Save or show the plot
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{video_id}_risk_visualization.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    else:
        plt.tight_layout()
        plt.show()
    
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Visualize risk scores for surgical videos')
    parser.add_argument('--video_id', type=str, default=None, 
                      help='Specific video ID to visualize (if not specified, will process all videos in metadata)')
    parser.add_argument('--risk_score_root', type=str, 
                      default="/path/to/risk_scores", 
                      help='Root directory for risk score files')
    parser.add_argument('--metadata_path', type=str, default= "/home/maxboels/datasets/CholecT50/embeddings_train_set/fold0/embeddings_f0_swin_bas_129.csv",
                      help='Path to metadata CSV file containing risk scores and phase information')
    parser.add_argument('--risk_columns', type=str, nargs='+', default=None,
                      help='List of risk score column names to include in visualization')
    parser.add_argument('--smooth_factor', type=float, default=5.0, 
                      help='Smoothing factor for risk score curve (0 = no smoothing)')
    parser.add_argument('--output_dir', type=str, default="risk_visualizations", 
                      help='Directory to save output plots')
    parser.add_argument('--skip_existing', action='store_true',
                      help='Skip videos that already have visualization files')
    parser.add_argument('--x_axis_type', type=str, default='frame',
                      choices=['frame', 'time'],
                      help='Type of x-axis (frame numbers or time in seconds)')
    
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
        
        # Process each video
        success_count = 0
        error_count = 0
        skip_count = 0
        
        for video_id in tqdm(video_ids, desc="Processing videos"):
            try:
                # Skip if visualization already exists
                output_path = os.path.join(args.output_dir, f"{video_id}_risk_visualization.png")
                if args.skip_existing and os.path.exists(output_path):
                    print(f"Skipping video {video_id} (visualization already exists)")
                    skip_count += 1
                    continue
                
                # Use the combined function to load both risk scores and phase information
                risk_scores_dict = None
                phases_df = None
                
                try:
                    risk_scores_dict, phases_df = load_data_from_metadata(
                        metadata_df, video_id, args.risk_columns
                    )
                except Exception as e:
                    print(f"Warning: Could not load data from metadata for video {video_id}: {e}")
                    
                    # Try to load risk scores from JSON files as fallback
                    if risk_scores_dict is None:
                        print("Attempting to load risk scores from JSON files...")
                        try:
                            risk_scores = load_risk_scores_from_json(
                                video_id, args.risk_score_root, "max"
                            )
                            risk_scores_dict = {"risk_score_json": risk_scores}
                        except Exception as e2:
                            print(f"Error: Failed to load risk scores for video {video_id}: {e2}")
                            raise ValueError("Failed to load risk scores from both metadata and JSON files")
                
                # Plot risk scores
                plot_risk_scores(
                    video_id, risk_scores_dict, phases_df, 
                    args.smooth_factor, args.output_dir, args.x_axis_type
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
    metadata_path = "/home/maxboels/datasets/CholecT50/embeddings_train_set/fold0/embeddings_f0_swin_bas_129.csv"
    main()