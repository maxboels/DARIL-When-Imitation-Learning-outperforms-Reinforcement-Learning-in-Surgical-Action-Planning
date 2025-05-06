
import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset
import torch
import os
import glob
import json
import pandas as pd
import re
from tqdm import tqdm
import yaml
import os
from datetime import datetime

from .preprocess_progression import add_progression_scores
from .preprocess_phase_completion import compute_phase_transition_rewards
from .preprocess_risk_scores import add_risk_scores
from .preprocess_action_scores import precompute_action_based_rewards
from .preprocess_action_scores import compute_action_phase_distribution



# Step 1: Data Loading from CholecT50 Dataset
def load_cholect50_data(cfg, split='train', max_videos=None):
    """
    Load frame embeddings from the CholecT50 dataset for training or validation.

    Returns:
        List of dictionaries containing video data
    """
    cfg_data = cfg['data']
    cfg_rewards = cfg['preprocess']['rewards']
    paths_config = cfg_data['paths']
    data_dir = paths_config['data_dir']
    metadata_file = paths_config['metadata_file']
    fold = paths_config['fold']
    print(f"Loading CholecT50 data from {data_dir}")

    # Create metadata path
    split_folder = f"embeddings_{split}_set"    
    metadata_dir = os.path.join(data_dir, split_folder, f"fold{fold}")
    metadata_path = os.path.join(metadata_dir, metadata_file)
    print(f"Metadata path: {metadata_path}")

    # Global outcome file
    video_global_outcome_file = paths_config['video_global_outcome_file'] # embeddings_f0_swin_bas_129_with_enhanced_global_metrics.csv
    video_global_outcome_path = os.path.join(metadata_dir, video_global_outcome_file)

    # Load metadata if available
    metadata_df = pd.read_csv(metadata_path)
    print(f"Metadata columns (before adding rewards): {metadata_df.columns.tolist()}")

    # Progressive +1 reward near phase transitions
    if cfg_rewards['grounded']['phase_completion']:
        metadata_df = compute_phase_transition_rewards(metadata_df, video_id_col='video_id', n_phases=7, 
                                    transition_window=30,
                                    phase_importance=None,
                                    max_reward=1.0,
                                    reward_function='exponential',
                                    reward_distribution='left_sided')
        metadata_file = metadata_file.replace('.csv', '_phase_complet.csv')
        metadata_df.to_csv(os.path.join(metadata_dir, metadata_file), index=False)
        print(f"Added phase completion rewards to metadata")

    # Compute reward signals for each frame (state)
    if cfg_rewards['grounded']['phase_progression'] or cfg_rewards['grounded']['global_progression']:
        metadata_df = add_progression_scores(metadata_df,
                        add_phase_progression=cfg_rewards['grounded']['phase_progression'],
                        add_global_progression=cfg_rewards['grounded']['global_progression'])
        metadata_file = metadata_file.replace('.csv', '_prog.csv')
        metadata_df.to_csv(os.path.join(metadata_dir, metadata_file), index=False)
        print(f"Added progression scores to metadata")
        
    # Compute action-based rewards for each frame (state) conditioned on phases
    if cfg_rewards['imitation']['action_distribution']:
        metadata_df = precompute_action_based_rewards(metadata_df, n_phases=7, n_actions=100, epsilon=1e-10)
        # The phase_progression values already gives a +1 reward when reaching the phase transitions
        # and gradually increases rewards until the next phase transition (smooth rewards)
        metadata_file = metadata_file.replace('.csv', '_action_dist.csv')
        metadata_df.to_csv(os.path.join(metadata_dir, metadata_file), index=False)
        print(f"Added action-based rewards to metadata")

    if cfg_rewards['expert_knowledge']['risk_score']:
        metadata_df = add_risk_scores(metadata_df, split, fold, 
                        frame_risk_agg=cfg_rewards['expert_knowledge']['frame_risk_agg'])
        metadata_file = metadata_file.replace('.csv', '_risk.csv')
        metadata_df.to_csv(os.path.join(metadata_dir, metadata_file), index=False)
        print(f"Added risk scores to metadata")
    
    # Add correct video_global_outcomes row and with all columns to the metadata file using the video id as unique identifier
    # Here we need to add the video global outcomes to the metadata
    # however, we only have x rows for each video, so we need to find the video id for each frame and then
    # select the value from the other dataframe and pass it to the metadata
    if metadata is not None and video_global_outcomes is not None:
        columns_to_add = ["avg_risk", "max_risk", "risk_std", "critical_risk_events", "critical_risk_percentage", "global_outcome_score"]
        video_ids = metadata['video'].unique().tolist()
        for video_id in video_ids:
            video_global_outcome = video_global_outcomes[video_global_outcomes['video'] == video_id]
            for column in columns_to_add:
                metadata.loc[metadata['video'] == video_id, column] = video_global_outcome[column].values[0]
        print(f"Added video global outcomes to metadata")

    # Find all videos in metadata csv file
    if metadata is not None:
        video_ids = metadata['video'].unique().tolist()
    if max_videos:
        video_ids = video_ids[:max_videos]
    
    print(f"Found {len(video_ids)} video directories")
    
    # Initialize data list
    data = []
    video_durations = []
    all_videos_mean_risk_scores = []
    all_videos_mean_risk_score_durs = []
    
    # Load frame embeddings for each video from the metadata
    for video_id in tqdm(video_ids, desc="Loading videos"):
        # Filter metadata for this video
        video_metadata = metadata[metadata['video'] == video_id]
        print(f"Found {len(video_metadata)} frames for video {video_id}")

        frame_files = video_metadata['embedding_path'].tolist()

        # Load frame embeddings
        video_frame_embeddings = []
        for frame_file in tqdm(frame_files, desc=f"Frames for {video_id}"):
            embedding = np.load(os.path.join(data_dir, split_folder, frame_file))
            video_frame_embeddings.append(embedding)
        
        video_frame_embeddings = np.array(video_frame_embeddings)
        
        # Check embedding dimension
        embedding_dim = video_frame_embeddings.shape[1]
        print(f"Embedding dimension: {embedding_dim}")
        
        # For demonstration: Generate random action classes and survival time if metadata not available
        # In a real scenario, you would extract these from metadata
        num_frames = len(video_frame_embeddings)
        video_durations.append(num_frames)
        
        video_risk_scores = video_metadata[risk_column_name].values

        # indices from column 'tri0':'tri199'
        action_columns = [f'tri{i}' for i in range(0, 100)]
        video_action_classes = video_metadata[action_columns].values

        # indices from column 'inst0':'inst5'
        instrument_columns = [f'inst{i}' for i in range(0, 6)]
        video_instrument_classes = video_metadata[instrument_columns].values

        # verb
        verb_columns = [f'v{i}' for i in range(0, 9)]
        video_verb_classes = video_metadata[verb_columns].values

        # target
        # target_columns = [f't{i}' for i in range(0, 14)]
        # video_target_classes = video_metadata[target_columns].values

        # Take the mean of the 
        video_mean_risk_score = np.mean(video_risk_scores)
        all_videos_mean_risk_scores.append(video_mean_risk_score)

        # Multiple average risk score by video length
        video_mean_risk_score_dur = video_mean_risk_score * num_frames
        all_videos_mean_risk_score_durs.append(video_mean_risk_score_dur)

        # Calculate survival time based on risk score
        m = 100 / 5 # 5 is the max risk score
        video_mean_survival_time = 100 - (video_mean_risk_score * m)

        # global outcome score (value score not directly related to the states)
        columns_to_add = ["avg_risk", "max_risk", "risk_std", "critical_risk_events", "critical_risk_percentage", "global_outcome_score"]
        video_outcome_data = {}
        for column in columns_to_add:
            video_outcome_data[column] = video_metadata[column].values[0]

        # Store video data
        data.append({
            'video_id': video_id,
            'video_dir': os.path.join(data_dir, split_folder, video_id),
            'frame_embeddings': video_frame_embeddings,
            'actions_binaries': video_action_classes,
            'instruments_binaries': video_instrument_classes,
            # 'verb_binaries': video_verb_classes,
            # 'target_binaries': video_target_classes,
            'risk_scores': video_risk_scores,
            'video_mean_risk_score': video_mean_risk_score,
            'video_mean_risk_score_dur': video_mean_risk_score_dur,
            'survival_time': video_mean_survival_time,
            'num_frames': num_frames,
            **video_outcome_data        # unpack video_outcome_data
        })

    
    avg_video_duration = np.mean(video_durations)
    print(f"Average video duration: {avg_video_duration:.2f} frames")
    
    if not data:
        raise ValueError("No valid videos loaded!")
    
    # Normalize by average video duration
    all_videos_mean_risk_score_durs = np.array(all_videos_mean_risk_score_durs) / avg_video_duration
    
    print(f"All Videos Mean Risk Score: {np.mean(all_videos_mean_risk_scores):.2f}")
    print(f"Standard Deviation of All Videos Mean Risk Score: {np.std(all_videos_mean_risk_scores):.2f}")
    print(f"All Videos Risk Scores: {all_videos_mean_risk_scores}")
    print(f"All Videos Mean Risk Score Dur: {np.mean(all_videos_mean_risk_score_durs):.2f}")
    print(f"Standard Deviation of All Videos Mean Risk Score Dur: {np.std(all_videos_mean_risk_score_durs):.2f}")
    print(f"All Videos Risk Score Dur: {all_videos_mean_risk_score_durs}")
    print(f"Successfully loaded {len(data)} ({split}ing) videos")
    return data

from torch.utils.data import DataLoader

def create_video_dataloaders(cfg, data, batch_size=32, shuffle=True, num_workers=4):
    """
    Create a separate dataloader for each video in the dataset.
    
    Args:
        cfg: Configuration dictionary
        data: List of video dictionaries from load_cholect50_data()
        batch_size: Batch size for the dataloaders
        shuffle: Whether to shuffle the samples
        num_workers: Number of worker processes for data loading
        
    Returns:
        Dictionary mapping video IDs to their respective DataLoaders
    """
    video_dataloaders = {}
    
    for video in data:
        video_id = video['video_id']
        
        # Create a dataset with only this video's data
        video_dataset = NextFramePredictionDataset(cfg['data'], [video])
        
        # Create a dataloader for this video
        video_dataloader = DataLoader(
            video_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )
        
        # Store the dataloader with the video ID as the key
        video_dataloaders[video_id] = video_dataloader
    
    return video_dataloaders

# Custom dataset for frame prediction with sequence inputs
class NextFramePredictionDataset(Dataset):
    def __init__(self, cfg, data):
        """
        Initialize the dataset with sequences of frame embeddings

        Lexic:
            "_" prefix indicates next frame
            "f" prefix indicates future frame
        
        Args:
            data: List of video dictionaries containing frame_embeddings and actions_binaries
            context_length: Number of previous frames to include in each input sequence
            padding_value: Value to use for padding when not enough previous frames exist
        """
        self.samples = []
        
        context_length = cfg.get('context_length', 10)
        padding_value = cfg.get('padding_value', 0.0)
        train_shift = cfg.get('train_shift', 1)
        max_horizon = cfg.get('max_horizon', 15)

        for video in data:
            video_id = video['video_id']
            embeddings = video['frame_embeddings']
            actions = video['actions_binaries']
            instruments = video['instruments_binaries']
            
            num_actions = len(actions[0])
            num_instruments = len(instruments[0])
            embedding_dim = len(embeddings[0])

            for i in range(len(embeddings) - 1):
                # For position i, take context_length frames from the left (previous frames)
                # This means frames from (i-context_length+1) to i, inclusive
                z_seq = []
                _z_seq = []
                f_z_seq = [] # future states from i+1 to i+max_horizon
                # a_seq: We already trained the vision backbone to recognise the actions but we need to evaluate it again
                _a_seq = []
                f_a_seq = [] # future actions from i+1 to i+max_horizon

                # current
                c_a = actions[i]
                c_i = instruments[i]
                
                # Add previous frames, using padding if needed
                for j in range(i - context_length + 1, i + 1):
                    if j < 0:
                        # Padding for positions before the start of the video
                        z_seq.append([padding_value] * embedding_dim)
                    else:
                        z_seq.append(embeddings[j])
                
                # Add the shifted next frame and action
                for k in range(i - context_length + 1 + train_shift, i + 1 + train_shift):
                    if k < 0:
                        # Padding for positions before the start of the video
                        _z_seq.append([padding_value] * embedding_dim)
                        _a_seq.append([0] * num_actions)
                        # print(f"Padding for frame position {k}/{len(embeddings)}")
                    elif k >= len(embeddings):
                        # Padding for positions after the end of the video
                        _z_seq.append([padding_value] * embedding_dim)
                        _a_seq.append([0] * num_actions)
                        # print(f"Padding for position {k}/{len(embeddings)}")
                    else:
                        _z_seq.append(embeddings[k])
                        _a_seq.append(actions[k])
                
                # Add future actions and states
                for k in range(i + 1, min(i + 1 + max_horizon, len(embeddings))):
                    f_z_seq.append(embeddings[k])
                    f_a_seq.append(actions[k])

                if len(f_a_seq) < max_horizon:
                    # Padding for positions after the end of the video
                    for _ in range(max_horizon - len(f_a_seq)):
                        f_a_seq.append([0] * num_actions)
                        f_z_seq.append([padding_value] * embedding_dim)

                # Add the sequence to the samples list
                self.samples.append({
                    'v_id': video_id,
                    'z': z_seq,     # Sequence of frames from (i-context_length+1) to i
                    '_z': _z_seq,   # Sequence of frames from (i-context_length+2) to i+1
                    'f_z': f_z_seq,   # Future states from (i+1) to (i+max_horizon)
                    '_a': _a_seq,    # Sequence of actions from (i-context_length+2) to i+1
                    'f_a': f_a_seq,
                    'c_a': c_a,     # Current Action at position i
                    'c_i': c_i,     # Current Instrument at position i
                })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Convert lists to numpy arrays first, then to tensors
        # v_id = torch.tensor(np.array(sample['v_id']), dtype=torch.float32)  # Shape: [context_length, embedding_dim]
        z = torch.tensor(np.array(sample['z']), dtype=torch.float32)  # Shape: [context_length, embedding_dim]
        _z = torch.tensor(np.array(sample['_z']), dtype=torch.float32)
        f_z = torch.tensor(np.array(sample['f_z']), dtype=torch.float32)  # Shape: [max_horizon, embedding_dim]
        _a = torch.tensor(np.array(sample['_a']), dtype=torch.float32)
        f_a = torch.tensor(np.array(sample['f_a']), dtype=torch.float32)

        # current action and instrument
        c_a = torch.tensor(sample['c_a'], dtype=torch.float32)
        c_i = torch.tensor(sample['c_i'], dtype=torch.float32)

        # create dictiornary for batch
        data = {
            'current_states': z,
            'next_states': _z,
            'future_states': f_z,
            'next_actions': _a,
            'future_actions': f_a,
            'current_actions': c_a,
            'current_instruments': c_i,
        }
        return data

# Custom dataset for reward prediction
class RewardPredictionDataset(Dataset):
    def __init__(self, data, context_length=10):
        self.samples = []
        self.context_length = context_length
        
        for video in data:
            embeddings = video['frame_embeddings']
            survival_time = video['survival_time']
            
            for i in range(context_length - 1, len(embeddings)):
                context = embeddings[i - (context_length - 1):i + 1]
                self.samples.append({
                    'context': np.array(context),
                    'target': survival_time
                })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return torch.tensor(sample['context'], dtype=torch.float32), torch.tensor(sample['target'], dtype=torch.float32)

# Create a dataset for action policy training with weighted actions
class ActionPolicyDataset(Dataset):
    def __init__(self, data, action_weights, context_length=10):
        self.samples = []
        self.context_length = context_length
        
        for video in data:
            embeddings = video['frame_embeddings']
            actions = video['actions_binaries']
            
            for i in range(context_length - 1, len(embeddings)):
                context = embeddings[i - (context_length - 1):i + 1]
                action = actions[i]
                
                # Get weight for this action (default to 1.0 if not found)
                weight = action_weights.get(int(action), 1.0)
                
                self.samples.append({
                    'context': np.array(context),
                    'action': int(action),
                    'weight': weight
                })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return (
            torch.tensor(sample['context'], dtype=torch.float32),
            torch.tensor(sample['action'], dtype=torch.long),
            torch.tensor(sample['weight'], dtype=torch.float32)
        )

