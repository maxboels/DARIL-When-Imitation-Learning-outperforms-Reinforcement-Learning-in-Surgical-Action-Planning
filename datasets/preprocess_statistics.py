import numpy as np
import pandas as pd
from collections import defaultdict

def compute_phase_statistics(metadata_df):
    """
    Compute statistics about phases from training data to guide the reward function.
    
    Args:
        metadata_df: DataFrame containing video metadata with phase information
        
    Returns:
        Dictionary with phase statistics
    """
    # Initialize statistics dictionary
    phase_stats = {
        'phase_durations': {},
        'phase_frequencies': {},
        'phase_transitions': defaultdict(int),
        'instrument_distributions': {},
        'avg_risk_by_phase': {},
        'avg_progression_rate': {}
    }
    
    # Extract phase columns
    phase_cols = [col for col in metadata_df.columns if col.startswith('p') and len(col) == 2 and col[1:].isdigit()]
    if not phase_cols:
        print("Warning: No phase columns found in metadata.")
        return phase_stats
    
    # Count occurrences of each phase
    for col in phase_cols:
        phase_id = int(col[1:])
        count = metadata_df[col].sum()
        phase_stats['phase_frequencies'][phase_id] = count
    
    # Compute average duration of each phase per video
    # Group by video and identify continuous segments of each phase
    for video_id in metadata_df['video'].unique():
        video_data = metadata_df[metadata_df['video'] == video_id].sort_values('frame')
        
        # Initialize tracking variables
        current_phase = None
        segment_start = None
        phase_segments = defaultdict(list)
        
        # Find phase segments
        for _, row in video_data.iterrows():
            # Determine current phase
            phase_id = None
            for col in phase_cols:
                if row[col] == 1:
                    phase_id = int(col[1:])
                    break
            
            # If phase changed or first frame
            if phase_id != current_phase:
                # Record previous segment if it exists
                if current_phase is not None and segment_start is not None:
                    segment_duration = row['frame'] - segment_start
                    phase_segments[current_phase].append(segment_duration)
                    
                    # Record phase transition
                    if phase_id is not None:
                        phase_stats['phase_transitions'][(current_phase, phase_id)] += 1
                
                # Start new segment
                current_phase = phase_id
                segment_start = row['frame']
        
        # Record last segment
        if current_phase is not None and segment_start is not None:
            segment_duration = video_data['frame'].iloc[-1] - segment_start + 1
            phase_segments[current_phase].append(segment_duration)
        
        # Add to global statistics
        for phase_id, durations in phase_segments.items():
            if phase_id not in phase_stats['phase_durations']:
                phase_stats['phase_durations'][phase_id] = []
            phase_stats['phase_durations'][phase_id].extend(durations)
    
    # Calculate average and standard deviation of phase durations
    phase_duration_stats = {}
    for phase_id, durations in phase_stats['phase_durations'].items():
        if durations:
            phase_duration_stats[phase_id] = {
                'mean': np.mean(durations),
                'median': np.median(durations),
                'std': np.std(durations),
                'min': np.min(durations),
                'max': np.max(durations),
                'count': len(durations)
            }
    phase_stats['phase_duration_stats'] = phase_duration_stats
    
    # Compute instrument usage distribution by phase
    instrument_cols = [col for col in metadata_df.columns if col.startswith('inst') and col[4:].isdigit()]
    if instrument_cols:
        for phase_id in range(len(phase_cols)):
            phase_data = metadata_df[metadata_df[f'p{phase_id}'] == 1]
            if not phase_data.empty:
                # Calculate average number of instruments used in this phase
                phase_stats['instrument_distributions'][phase_id] = {
                    'avg_count': phase_data[instrument_cols].sum(axis=1).mean(),
                    'usage_frequency': {int(col[4:]): phase_data[col].mean() for col in instrument_cols}
                }
    
    # Compute average risk score by phase
    if 'risk_score_max' in metadata_df.columns:
        for phase_id in range(len(phase_cols)):
            phase_data = metadata_df[metadata_df[f'p{phase_id}'] == 1]
            if not phase_data.empty:
                phase_stats['avg_risk_by_phase'][phase_id] = phase_data['risk_score_max'].mean()
    
    # Compute progression rate by phase
    if 'phase_progression' in metadata_df.columns:
        for phase_id in range(len(phase_cols)):
            phase_data = metadata_df[metadata_df[f'p{phase_id}'] == 1]
            if not phase_data.empty:
                # Group by video and calculate progression rate
                progression_rates = []
                for vid in phase_data['video'].unique():
                    video_phase_data = phase_data[phase_data['video'] == vid].sort_values('frame')
                    if len(video_phase_data) > 1:
                        # Calculate rate of progression per frame
                        first_progression = video_phase_data['phase_progression'].iloc[0]
                        last_progression = video_phase_data['phase_progression'].iloc[-1]
                        num_frames = len(video_phase_data)
                        if last_progression > first_progression:
                            rate = (last_progression - first_progression) / num_frames
                            progression_rates.append(rate)
                
                if progression_rates:
                    phase_stats['avg_progression_rate'][phase_id] = np.mean(progression_rates)
    
    return phase_stats


def calculate_global_outcome_proxy(video_data, phase_stats):
    """
    Calculate a global outcome proxy value for a video based on risk management,
    phase completion and efficiency.
    
    Args:
        video_data: DataFrame containing single video data
        phase_stats: Dictionary with phase statistics
        
    Returns:
        Float: Global outcome proxy score (0-10 scale)
    """
    outcome_score = 5.0  # Start with neutral score
    components = {}
    
    # 1. Risk Management (0-4 points)
    if 'risk_score_max' in video_data.columns:
        avg_risk = video_data['risk_score_max'].mean()
        max_risk = video_data['risk_score_max'].max()
        
        # Lower risk is better (0-2 points)
        risk_score = max(0, 2 - (avg_risk / 5.0) * 2)
        
        # Risk variability (0-1 points)
        # Less variability usually means better control
        risk_std = video_data['risk_score_max'].std()
        variability_score = max(0, 1 - (risk_std / 2.5))
        
        # Risk trend (0-1 points)
        # Decreasing risk over time is good
        first_half_risk = video_data.iloc[:len(video_data)//2]['risk_score_max'].mean()
        second_half_risk = video_data.iloc[len(video_data)//2:]['risk_score_max'].mean()
        trend_score = 0.5  # Neutral
        if second_half_risk < first_half_risk:
            # Risk decreased - good
            trend_score = 0.5 + min(0.5, (first_half_risk - second_half_risk))
        else:
            # Risk increased - bad
            trend_score = max(0, 0.5 - min(0.5, (second_half_risk - first_half_risk)))
        
        risk_management_score = risk_score + variability_score + trend_score
        components['risk_management'] = risk_management_score
    else:
        risk_management_score = 2.0  # Neutral if no risk data
        components['risk_management'] = risk_management_score
    
    # 2. Phase Completion (0-3 points)
    if 'p0' in video_data.columns:
        # Identify completed phases
        phase_cols = [col for col in video_data.columns if col.startswith('p') and len(col) == 2 and col[1:].isdigit()]
        completed_phases = set()
        
        for col in phase_cols:
            if video_data[col].sum() > 0:
                completed_phases.add(int(col[1:]))
        
        # Score based on number of completed phases
        phase_completion_ratio = len(completed_phases) / len(phase_cols)
        phase_completion_score = 3.0 * phase_completion_ratio
        components['phase_completion'] = phase_completion_score
    else:
        phase_completion_score = 1.5  # Neutral if no phase data
        components['phase_completion'] = phase_completion_score
    
    # 3. Efficiency (0-3 points)
    if 'phase_progression' in video_data.columns and phase_stats.get('avg_progression_rate'):
        # Calculate overall progression rate
        progression_values = video_data['phase_progression'].values
        progression_rate = (progression_values[-1] - progression_values[0]) / len(progression_values)
        
        # Compare to expected rates from phase_stats
        efficiency_scores = []
        for phase_id in range(7):  # Assuming 7 phases
            phase_data = video_data[video_data[f'p{phase_id}'] == 1]
            if not phase_data.empty and phase_id in phase_stats.get('avg_progression_rate', {}):
                expected_rate = phase_stats['avg_progression_rate'][phase_id]
                if expected_rate > 0:
                    actual_rate = (phase_data['phase_progression'].iloc[-1] - 
                                  phase_data['phase_progression'].iloc[0]) / len(phase_data)
                    # Ratio of actual to expected (higher is better, but diminishing returns)
                    ratio = min(2.0, actual_rate / expected_rate) if actual_rate > 0 else 0.5
                    efficiency_scores.append(ratio)
        
        # Average efficiency scores
        if efficiency_scores:
            efficiency_score = min(3.0, sum(efficiency_scores) / len(efficiency_scores) * 3.0)
        else:
            efficiency_score = 1.5  # Neutral if no scores calculated
    else:
        efficiency_score = 1.5  # Neutral if no progression data
    
    components['efficiency'] = efficiency_score
    
    # Calculate total outcome score
    outcome_score = risk_management_score + phase_completion_score + efficiency_score
    
    # Ensure score is in 0-10 range
    outcome_score = max(0, min(10, outcome_score))
    
    return outcome_score, components


def precompute_action_based_rewards(metadata_df):
    """
    Precompute action-based rewards using imitation learning approach but derived
    from data instead of prescriptive rules.
    
    Args:
        metadata_df: DataFrame with metadata
        
    Returns:
        DataFrame with added action reward column
    """
    # Compute phase statistics
    phase_stats = compute_phase_statistics(metadata_df)
    
    # Compute action distributions by phase
    action_dist_df = compute_action_phase_distribution(metadata_df)
    
    # Create a copy of the DataFrame
    df = metadata_df.copy()
    
    # Add action reward column
    df['action_reward'] = 0.0
    
    # Process each row
    for idx, row in df.iterrows():
        # Determine current phase
        phase_id = None
        for i in range(7):
            if f'p{i}' in row and row[f'p{i}'] == 1:
                phase_id = i
                break
                
        if phase_id is None:
            continue
            
        # Get action vector for this frame
        action_vector = np.zeros(100)
        for i in range(100):
            col = f'tri{i}'
            if col in row:
                action_vector[i] = row[col]
        
        # Calculate reward using KL divergence
        if phase_id in action_dist_df.index and np.sum(action_vector) > 0:
            phase_key = f'p{phase_id}'
            expert_dist = action_dist_df.loc[phase_key].values
            
            # Normalize action vector
            action_dist = action_vector / np.sum(action_vector)
            
            # Calculate KL divergence (smaller is better)
            try:
                # Add small epsilon to prevent division by zero or log(0)
                epsilon = 1e-10
                expert_dist = expert_dist + epsilon
                action_dist = action_dist + epsilon
                
                # Normalize again
                expert_dist = expert_dist / np.sum(expert_dist)
                action_dist = action_dist / np.sum(action_dist)
                
                # Calculate KL divergence
                kld = np.sum(expert_dist * np.log(expert_dist / action_dist))
                
                # Convert to reward (negative KL divergence, higher is better)
                # Apply scaling to keep reward in reasonable range
                reward = -min(kld, 10.0)  # Cap at -10 to avoid extreme penalties
                
                # Normalize to a more reasonable range [-1, 1]
                reward = np.clip(reward / 5.0, -1.0, 1.0)
                
                df.at[idx, 'action_reward'] = reward
            except Exception as e:
                print(f"Error calculating KL divergence: {e}")
    
    return df, phase_stats


def compute_action_phase_distribution(metadata_df, n_phases=7, n_actions=100):
    """
    Compute the distribution of expert actions (tri0…tri99) in each phase (p0…p{n_phases-1})
    across all videos.

    Args:
        metadata_df: DataFrame with metadata
        n_phases: Number of phases
        n_actions: Number of action triplets

    Returns:
        DataFrame with action distributions per phase
    """
    # Build column lists
    phase_cols  = [f"p{p}" for p in range(n_phases)]
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
    return dist_df.astype(float)
