
import os
import json
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, Any

def add_risk_scores(metadata, split, fold, frame_risk_agg='max'):
    """
    Add risk scores to metadata if not already there
    """
    # Check if risk scores are already in metadata
    risk_score_root = "/home/maxboels/datasets/CholecT50/instructions/anticipation_5s_with_goals/"
    video_ids_cache = []
    all_risk_scores = []
    risk_column_name = f"state_risk_{frame_risk_agg}"
    
    # For each frame in metadata, add risk score
    for i, row in metadata.iterrows():
        video_id = row['video']
        frame_id = row['frame']

        # Add risk score per frame to metadata if not already there or column has nan value
        if video_id not in video_ids_cache:
            print(f"Loading risk scores for video {video_id}")
            video_ids_cache.append(video_id)
            risk_scores = None
            risk_score_path = risk_score_root + f"{video_id}_sorted_with_risk_scores_instructions_with_goals.json" 
            if risk_score_path and os.path.exists(risk_score_path):
                print(f"Loading risk scores from {risk_score_path}")
                with open(risk_score_path, 'r') as f:
                    risk_scores = json.load(f)
            else:
                raise ValueError(f"Risk score file {risk_score_path} not found")
        
        # Get risk score for each frame
        current_actions = risk_scores[str(frame_id)]['current_actions']
        frame_risk_scores = []
        for action in current_actions: # it's a list of dictionaries
            frame_risk_scores.append(action['expert_risk_score'])
        if frame_risk_agg == 'mean':
            risk_score = np.mean(frame_risk_scores)
        elif frame_risk_agg == 'max':
            risk_score = np.max(frame_risk_scores)
        else:
            print(f"Frame risk aggregation method {frame_risk_agg} not supported, skipping")
        risk_score = float(risk_score)
        all_risk_scores.append(risk_score)

        # if last frame, add risk score to metadata
        if i == len(metadata) - 1:
            metadata[risk_column_name] = all_risk_scores

    return metadata