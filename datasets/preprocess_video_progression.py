


def add_video_progression_scores(metadata_df, n_phases=7):
    """
    Adds 'phase_progress' and 'video_progress' columns to a metadata DataFrame.
    
    Assumes:
      - One-hot phase columns 'p0'...'p{n_phases-1}' are present.
      - 'video' and 'frame' columns are present.
    
    Args:
        metadata_df (pd.DataFrame): Frame-level metadata.
        n_phases (int): Number of surgical phases.
    
    Returns:
        pd.DataFrame: A copy of metadata_df with added columns.
    """
    df = metadata_df.copy()
    
    # Determine phase_id per frame
    phase_cols = [f'p{i}' for i in range(n_phases)]
    df['phase_id'] = df[phase_cols].values.argmax(axis=1)
    
    # Video-level progression: position / (total_frames - 1)
    df['video_progress'] = df.groupby('video')['frame'].transform(
        lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 1.0
    )
    
    # Identify contiguous segments within each video for each phase
    df['phase_change'] = (df['phase_id'] != df.groupby('video')['phase_id'].shift()).astype(int)
    # Cumulative segment index per video
    df['segment_idx'] = df.groupby('video')['phase_change'].cumsum()
    
    # Compute phase progress per segment
    df['phase_progress'] = 0.0
    for (vid, seg), group in df.groupby(['video', 'segment_idx']):
        start = group['frame'].min()
        end = group['frame'].max()
        duration = end - start
        if duration > 0:
            df.loc[group.index, 'phase_progress'] = (group['frame'] - start) / duration
        else:
            df.loc[group.index, 'phase_progress'] = 1.0
    
    # Clean up helper columns if desired
    df.drop(columns=['phase_change', 'segment_idx'], inplace=True)
    
    return df