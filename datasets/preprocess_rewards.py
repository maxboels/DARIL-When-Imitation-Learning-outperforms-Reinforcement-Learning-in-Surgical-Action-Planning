import pandas as pd

import pandas as pd

def add_progression_scores(metadata_df, n_phases=7):
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

def compute_action_phase_distribution(metadata_df, n_phases=7, n_actions=100, add_nonzero=True):
    """
    Compute the distribution of expert actions (tri0…tri99) in each phase (p0…p{n_phases-1})
    across all videos.

    Args:
        metadata_df (pd.DataFrame): your frame‐level metadata with columns:
            - 'p0'...'p{n_phases-1}' (one‐hot phase labels)
            - 'tri0'...'tri{n_actions-1}' (one‐hot action labels)
        n_phases (int): number of surgical phases (default 7)
        n_actions (int): number of action triplets (default 100)
        add_nonzero (bool): if True, add small non-zero probability to all actions (default True)

    Returns:
        pd.DataFrame: shape (n_phases, n_actions), where entry (p,i) is 
                      P_expert(action=i │ phase=p)
    """
    # Build column lists
    phase_cols  = [f"p{p}"   for p in range(n_phases)]
    action_cols = [f"tri{i}" for i in range(n_actions)]

    # Prepare a DataFrame to hold the distributions
    dist_df = pd.DataFrame(
        data=0.0, 
        index=[f"p{p}" for p in range(n_phases)],
        columns=[f"tri{i}" for i in range(n_actions)]
    )

    # For each phase, sum up action counts and normalize
    for p, pcol in enumerate(phase_cols):
        df_phase = metadata_df[metadata_df[pcol] == 1.0]
        if df_phase.empty:
            continue
        counts = df_phase[action_cols].sum(axis=0).astype(float)

        if add_nonzero:
            # Add small non-zero probabilities to all actions
            for i in range(n_actions):
                if counts[f"tri{i}"] == 0:
                    counts[f"tri{i}"] = 1e-8
        # Normalize counts to get probabilities
        total = counts.sum()
        if total > 0:
            dist_df.loc[f"p{p}"] = counts / total
        
    # Fill NaN values with 0.0
    dist_df.fillna(0.0, inplace=True)
    # Convert to float
    dist_df = dist_df.astype(float)
    # Return the distribution DataFrame
    return dist_df