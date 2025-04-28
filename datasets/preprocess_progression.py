import pandas as pd
import numpy as np
from tqdm import tqdm

def load_data_from_metadata(metadata_df, video_id, num_phases=7):
    """
    Load phase information from metadata DataFrame for a specific video.
    
    Args:
        metadata_df: DataFrame containing metadata with phase information
        video_id: ID of the video
        num_phases: Number of possible phases
    
    Returns:
        Tuple containing:
        - DataFrame with phase segments (start_frame, end_frame, phase_id)
        - Dictionary mapping frame_id to phase_id
    """
    # Filter metadata for this video
    video_metadata = metadata_df[metadata_df['video'] == video_id].copy()
    
    if video_metadata.empty:
        raise ValueError(f"No metadata found for video {video_id}")

    # Extract phase information
    phase_columns = [f'p{i}' for i in range(num_phases) if f'p{i}' in video_metadata.columns]
    
    if not phase_columns:
        print(f"Warning: No phase columns (p0-p6) found in metadata for video {video_id}")
        return None, {}
    
    # Determine phase for each frame
    frame_phases = {}
    for _, row in video_metadata.iterrows():
        frame_id = row['frame']
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
        return None, {}
    
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
    
    return pd.DataFrame(phases_data), frame_phases

def calculate_phase_progression(frame_id, phase_id, phase_segments):
    """
    Calculate the progression (0.0 to 1.0) of a frame within its current phase.
    
    Args:
        frame_id: Current frame ID
        phase_id: Current phase ID
        phase_segments: DataFrame with phase segment information (start_frame, end_frame, phase_id)
        
    Returns:
        progress: Value between 0.0 and 1.0 indicating position in the current phase
    """
    if phase_id is None or phase_segments is None or phase_segments.empty:
        return 0.0
    
    # Get all segments for this phase
    phase_segment = phase_segments[phase_segments['phase_id'] == phase_id]
    
    if phase_segment.empty:
        return 0.0
    
    # Find the specific segment containing this frame
    segment = phase_segment[(phase_segment['start_frame'] <= frame_id) & 
                           (phase_segment['end_frame'] >= frame_id)]
    
    # If frame isn't in any segment of this phase, it might be a boundary case
    if segment.empty:
        return 0.0
    
    # Calculate progress based on position within segment
    start_frame = segment['start_frame'].iloc[0]
    end_frame = segment['end_frame'].iloc[0]
    total_frames = end_frame - start_frame
    
    # Avoid division by zero
    if total_frames <= 0:
        return 0.0
    
    frames_elapsed = frame_id - start_frame
    progress = frames_elapsed / total_frames
    
    # Ensure progress is between 0.0 and 1.0
    return max(0.0, min(1.0, progress))

def calculate_global_progression(frame_id, min_frame, max_frame):
    """
    Calculate the global progression (0.0 to 1.0) of a frame within the entire video.
    
    Args:
        frame_id: Current frame ID
        min_frame: First frame of the video
        max_frame: Last frame of the video
        
    Returns:
        progress: Value between 0.0 and 1.0 indicating position in the entire video
    """
    if max_frame <= min_frame:
        return 0.0
    
    # Calculate progress based on position within video
    total_frames = max_frame - min_frame
    frames_elapsed = frame_id - min_frame
    
    progress = frames_elapsed / total_frames
    
    # Ensure progress is between 0.0 and 1.0
    return max(0.0, min(1.0, progress))

def add_progression_scores(metadata_df, num_phases=7, phase_column_name="phase_prog", 
                          global_column_name="global_prog", add_phase_progression=True,
                          add_global_progression=True):
    """
    Calculate and add phase progression and global progression scores for each frame.
    
    Args:
        metadata_df: DataFrame containing metadata with phase information
        num_phases: Number of possible phases
        phase_column_name: Column name for phase progression scores
        global_column_name: Column name for global progression scores
        add_phase_progression: Whether to calculate and add phase progression scores (default: True)
        add_global_progression: Whether to calculate and add global progression scores (default: True)
        
    Returns:
        Updated metadata DataFrame with requested progression columns
    """
    # Create a copy to avoid modifying the original
    result_df = metadata_df.copy()
    
    # Initialize progression columns
    if add_phase_progression:
        result_df[phase_column_name] = 0.0
        
    if add_global_progression:
        result_df[global_column_name] = 0.0
    
    # If neither progression type is requested, return the original dataframe
    if not add_phase_progression and not add_global_progression:
        return result_df
    
    video_ids = result_df['video'].unique().tolist()
    
    for video_id in tqdm(video_ids, desc="Processing videos"):
        # Get frames for this video
        video_frames = result_df[result_df['video'] == video_id]
        
        if video_frames.empty:
            continue
            
        # Get min and max frame for global progression if needed
        min_frame = max_frame = None
        if add_global_progression:
            min_frame = video_frames['frame'].min()
            max_frame = video_frames['frame'].max()
        
        # Get phase segments for this video if needed
        phase_segments = frame_phases = None
        if add_phase_progression:
            phase_segments, frame_phases = load_data_from_metadata(result_df, video_id, num_phases)
            if phase_segments is None:
                continue
        
        # Calculate progressions for each frame
        for _, row in video_frames.iterrows():
            frame_id = row['frame']
            idx = result_df.index[(result_df['video'] == video_id) & (result_df['frame'] == frame_id)]
            
            # Calculate phase progression if requested
            if add_phase_progression and phase_segments is not None:
                phase_id = frame_phases.get(frame_id)
                phase_progress = calculate_phase_progression(frame_id, phase_id, phase_segments)
                result_df.loc[idx, phase_column_name] = phase_progress
            
            # Calculate global progression if requested
            if add_global_progression and min_frame is not None and max_frame is not None:
                global_progress = calculate_global_progression(frame_id, min_frame, max_frame)
                result_df.loc[idx, global_column_name] = global_progress
    
    return result_df