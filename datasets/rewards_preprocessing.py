import pandas as pd

def compute_action_phase_distribution(metadata_df, n_phases=7, n_actions=100):
    """
    Compute the distribution of expert actions (tri0…tri99) in each phase (p0…p{n_phases-1})
    across all videos.

    Args:
        metadata_df (pd.DataFrame): your frame‐level metadata with columns:
            - 'p0'...'p{n_phases-1}' (one‐hot phase labels)
            - 'tri0'...'tri{n_actions-1}' (one‐hot action labels)
        n_phases (int): number of surgical phases (default 7)
        n_actions (int): number of action triplets (default 100)

    Returns:
        pd.DataFrame: shape (n_phases, n_actions), where entry (φ,i) is 
                      P_expert(action=i │ phase=φ)
    """
    # Build column lists
    phase_cols  = [f"p{φ}"   for φ in range(n_phases)]
    action_cols = [f"tri{i}" for i in range(n_actions)]

    # Prepare a DataFrame to hold the distributions
    dist_df = pd.DataFrame(
        data=0.0, 
        index=[f"phase_{φ}" for φ in range(n_phases)],
        columns=[f"action_{i}" for i in range(n_actions)]
    )

    # For each phase, sum up action counts and normalize
    for φ, pcol in enumerate(phase_cols):
        df_phase = metadata_df[metadata_df[pcol] == 1]
        if df_phase.empty:
            continue
        counts = df_phase[action_cols].sum(axis=0).astype(float)
        total = counts.sum()
        if total > 0:
            dist_df.loc[f"phase_{φ}"] = counts / total

    return dist_df
