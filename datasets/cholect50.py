
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
from .preprocess_phase_completion import compute_phase_completion_rewards
from .preprocess_phase_transition import compute_phase_transition_rewards
from .preprocess_risk_scores import add_risk_scores
from .preprocess_action_scores import precompute_action_based_rewards
from .preprocess_action_scores import compute_action_phase_distribution



# Step 1: Data Loading from CholecT50 Dataset
def load_cholect50_data(cfg, logger, split='train', max_videos=None):
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
    logger.info(f"Loading {split} data from {data_dir} with fold {fold}")

    # Create metadata path
    split_folder = f"embeddings_{split}_set"    
    metadata_dir = os.path.join(data_dir, split_folder, f"fold{fold}")
    metadata_path = os.path.join(metadata_dir, metadata_file)

    # Global outcome file
    video_global_outcome_file = paths_config['video_global_outcome_file'] # embeddings_f0_swin_bas_129_with_enhanced_global_metrics.csv
    video_global_outcome_path = os.path.join(metadata_dir, video_global_outcome_file)

    # Load metadata if available
    metadata_df = pd.read_csv(metadata_path)
    
    logger.info(f"[{split}] Start processing metadata file {metadata_path}")
    if cfg['preprocess']['extract_rewards']:
        # Progressive +1 reward near phase transitions
        if cfg_rewards['grounded']['phase_completion']:
            metadata_df = compute_phase_completion_rewards(metadata_df, video_id_col='video', n_phases=7, 
                                        transition_window=30,
                                        phase_importance=None,
                                        max_reward=1.0,
                                        reward_function='exponential',
                                        reward_distribution='left_sided')
            metadata_file = metadata_file.replace('.csv', '_phase_complet.csv')
            metadata_df.to_csv(os.path.join(metadata_dir, metadata_file), index=False)
            logger.info(f"[{split}] Added phase completion rewards to {metadata_file}") 

        if cfg_rewards['grounded']['phase_transition']:
            metadata_df = compute_phase_transition_rewards(metadata_df, video_id_col='video', n_phases=7, 
                                        reward_window=5, 
                                        phase_importance=None,
                                        reward_value=1.0)
            metadata_file = metadata_file.replace('.csv', '_phase_transit.csv')
            metadata_df.to_csv(os.path.join(metadata_dir, metadata_file), index=False)
            logger.info(f"[{split}] Added phase transition rewards to {metadata_file}")

        # Compute reward signals for each frame (state)
        if cfg_rewards['grounded']['phase_progression'] or cfg_rewards['grounded']['global_progression']:
            metadata_df = add_progression_scores(metadata_df,
                            add_phase_progression=cfg_rewards['grounded']['phase_progression'],
                            add_global_progression=cfg_rewards['grounded']['global_progression'])
            metadata_file = metadata_file.replace('.csv', '_prog.csv')
            metadata_df.to_csv(os.path.join(metadata_dir, metadata_file), index=False)
            logger.info(f"[{split}] Added phase and global progression rewards to {metadata_file}")
            
        # Compute action-based rewards for each frame (state) conditioned on phases
        if cfg_rewards['imitation']['action_distribution']:
            metadata_df = precompute_action_based_rewards(metadata_df, n_phases=7, n_actions=100, epsilon=1e-10)
            # The phase_progression values already gives a +1 reward when reaching the phase transitions
            # and gradually increases rewards until the next phase transition (smooth rewards)
            metadata_file = metadata_file.replace('.csv', '_prob_action.csv')
            metadata_df.to_csv(os.path.join(metadata_dir, metadata_file), index=False)
            logger.info(f"[{split}] Added action distribution rewards to {metadata_file}")

        if cfg_rewards['expert_knowledge']['risk_score']:
            metadata_df = add_risk_scores(metadata_df, cfg_data, split, fold, 
                            frame_risk_agg=cfg_rewards['expert_knowledge']['frame_risk_agg'])
            metadata_file = metadata_file.replace('.csv', '_risk.csv')
            metadata_df.to_csv(os.path.join(metadata_dir, metadata_file), index=False)
            logger.info(f"[{split}] Added risk scores to {metadata_file}")
        
        # Add correct video_global_outcomes row and with all columns to the metadata file using the video id as unique identifier
        # Here we need to add the video global outcomes to the metadata
        # however, we only have x rows for each video, so we need to find the video id for each frame and then
        # select the value from the other dataframe and pass it to the metadata
        if os.path.exists(video_global_outcome_path):
            video_global_outcomes = pd.read_csv(video_global_outcome_path)
            original_columns = ["avg_risk", "max_risk", "risk_std", "critical_risk_events", "critical_risk_percentage", "global_outcome_score"]
            prefixed_columns = [f"glob_{col}" for col in original_columns]
            video_ids = metadata_df['video'].unique().tolist()
            for video_id in video_ids:
                video_global_outcome = video_global_outcomes[video_global_outcomes['video'] == video_id]
                if not video_global_outcome.empty:
                    for orig_col, new_col in zip(original_columns, prefixed_columns):
                        metadata_df.loc[metadata_df['video'] == video_id, new_col] = video_global_outcome[orig_col].values[0]
            metadata_file = metadata_file.replace('.csv', '_glob_outcome.csv')
            metadata_df.to_csv(os.path.join(metadata_dir, metadata_file), index=False)
            logger.info(f"[{split}] Added global outcome scores to {metadata_file}")
        else:
            logger.warning(f"[{split}] Global outcome file {video_global_outcome_path} not found, skipping global outcome scores")
    else:
        logger.info(f"[{split}] Rewards extraction skipped, using metadata file {metadata_path}")

    # Find all videos in metadata csv file
    if metadata_df is not None:
        video_ids = metadata_df['video'].unique().tolist()
    if max_videos:
        video_ids = video_ids[:max_videos]
    
    logger.info(f"[{split}] Found {len(video_ids)} videos in metadata file")
    
    # Initialize data list
    data = []
    video_durations = []
    
    # Load frame embeddings for each video from the metadata
    logger.info(f"[{split}] Loading data for {len(video_ids)} videos")
    for video_id in tqdm(video_ids, desc="Loading frames for videos"):
        # Filter metadata for this video
        video_metadata = metadata_df[metadata_df['video'] == video_id]
        frame_files = video_metadata['embedding_path'].tolist()

        # Load frame embeddings
        video_frame_embeddings = []
        for frame_file in tqdm(frame_files, desc=f"Frames for {video_id}"):
            embedding = np.load(os.path.join(data_dir, split_folder, frame_file))
            video_frame_embeddings.append(embedding)
        video_frame_embeddings = np.array(video_frame_embeddings)        
        embedding_dim = video_frame_embeddings.shape[1]
        
        # For demonstration: Generate random action classes and survival time if metadata_df not available
        # In a real scenario, you would extract these from metadata_df
        num_frames = len(video_frame_embeddings)
        video_durations.append(num_frames)
        
        # Get reward values from metadata_df per video
        rewards = {}
        rewards['_r_risk_penalty'] = video_metadata['risk_score_max'].values
        rewards['_r_phase_completion'] = video_metadata['phase_completion_reward'].values
        rewards['_r_phase_initiation'] = video_metadata['phase_initiation_reward'].values
        rewards['_r_phase_progression'] = video_metadata['phase_prog'].values
        rewards['_r_global_progression'] = video_metadata['global_prog'].values
        rewards['_r_action_probability'] = video_metadata['action_reward'].values

        # Get global outcome scores
        outcomes = {}
        outcomes['q_global_avg_risk'] = video_metadata['glob_avg_risk'].values
        outcomes['q_global_critical_risk'] = video_metadata['glob_critical_risk_events'].values
        outcomes['q_global_outcome_score'] = video_metadata['glob_global_outcome_score'].values

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

        phase_columns = [f'p{i}' for i in range(0, 7)]
        video_phase_classes = video_metadata[phase_columns].values

        # Store video data
        data.append({
            'video_id': video_id,
            'video_dir': os.path.join(data_dir, split_folder, video_id),
            'frame_embeddings': video_frame_embeddings,
            'actions_binaries': video_action_classes,
            'instruments_binaries': video_instrument_classes,
            # 'verb_binaries': video_verb_classes,
            # 'target_binaries': video_target_classes,
            'phase_binaries': video_phase_classes,
            'num_frames': num_frames,
            'next_rewards': rewards,
            'outcomes': outcomes,
        })

    if not data:
        raise ValueError("No valid videos loaded!")
    
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
            phases = video['phase_binaries']

            # rewards and outcomes
            rewards = video['next_rewards']
            phase_completion = rewards['_r_phase_completion']
            phase_initiation = rewards['_r_phase_initiation']
            phase_progression = rewards['_r_phase_progression']
            global_progression = rewards['_r_global_progression']
            action_probability = rewards['_r_action_probability']
            _r_risk_penalty = rewards['_r_risk_penalty']

            # outcomes
            q = video['outcomes']
            
            num_actions = len(actions[0])
            num_instruments = len(instruments[0])
            num_phases = len(phases[0])
            reward_dim = 1
            embedding_dim = len(embeddings[0])

            for i in range(len(embeddings) - 1):
                # For position i, take context_length frames from the left (previous frames)
                # This means frames from (i-context_length+1) to i, inclusive
                z_seq = []
                _z_seq = []
                f_z_seq = [] # future states from i+1 to i+max_horizon
                # a_seq: We already trained the vision backbone to recognise the actions but we need to evaluate it again
                _a_seq = []
                _p_seq = [] # next phases from i+1 to i+max_horizon
                f_a_seq = [] # future actions from i+1 to i+max_horizon

                # rewards
                _r_p_comp_seq = []
                _r_p_init_seq = []
                _r_p_prog_seq = []
                _r_g_prog_seq = []
                _r_a_prob_seq = []
                _r_risk_seq = [] # we need to subtract the risk penalty from the rewards
                
                # outcomes
                q_seq = []

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
                
                # Add the shifted next frame and action, using padding if needed
                for k in range(i - context_length + 1 + train_shift, i + 1 + train_shift):
                    if k < 0:
                        # Padding for positions before the start of the video
                        _z_seq.append([padding_value] * embedding_dim)
                        _a_seq.append([0] * num_actions)
                        _p_seq.append([0] * num_phases)
                        # rewards
                        _r_p_comp_seq.append([0.0] * reward_dim)
                        _r_p_init_seq.append([0.0] * reward_dim)
                        _r_p_prog_seq.append([0.0] * reward_dim)
                        _r_g_prog_seq.append([0.0] * reward_dim)
                        _r_a_prob_seq.append([0.0] * reward_dim)
                        _r_risk_seq.append([1.0] * reward_dim)

                        # print(f"Padding for frame position {k}/{len(embeddings)}")
                    elif k >= len(embeddings):
                        # Padding for positions after the end of the video
                        _z_seq.append([padding_value] * embedding_dim)
                        _a_seq.append([0] * num_actions)
                        _p_seq.append([0] * num_phases)
                        # rewards
                        _r_p_comp_seq.append([0.0] * reward_dim)
                        _r_p_init_seq.append([0.0] * reward_dim)
                        _r_p_prog_seq.append([0.0] * reward_dim)
                        _r_g_prog_seq.append([0.0] * reward_dim)
                        _r_a_prob_seq.append([0.0] * reward_dim)
                        _r_risk_seq.append([1.0] * reward_dim) # 1.0 is the minimum risk score
                    else:
                        _z_seq.append(embeddings[k])
                        _a_seq.append(actions[k])
                        _p_seq.append(phases[k])
                        # rewards (make sure it is a list of lists)
                        _r_p_comp_seq.append([phase_completion[k]])
                        _r_p_init_seq.append([phase_initiation[k]])
                        _r_p_prog_seq.append([phase_progression[k]])
                        _r_g_prog_seq.append([global_progression[k]])
                        _r_a_prob_seq.append([action_probability[k]])
                        _r_risk_seq.append([_r_risk_penalty[k]])
                        # outcomes
                
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
                    '_p': _p_seq,    # Sequence of phases from (i-context_length+2) to i+1
                    'c_a': c_a,     # Current Action at position i
                    'c_i': c_i,     # Current Instrument at position i
                    # rewards
                    '_r': {
                        '_r_phase_completion': _r_p_comp_seq,
                        '_r_phase_initiation': _r_p_init_seq,
                        '_r_phase_progression': _r_p_prog_seq,
                        '_r_global_progression': _r_g_prog_seq,
                        '_r_action_probability': _r_a_prob_seq,
                        '_r_risk': _r_risk_seq,
                    },
                    # outcomes
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
        _p = torch.tensor(np.array(sample['_p']), dtype=torch.float32)  # Shape: [max_horizon, num_phases]

        # rewards
        _r = sample['_r']
        _r_p_comp = torch.tensor(np.array(_r['_r_phase_completion']), dtype=torch.float32)
        _r_p_init = torch.tensor(np.array(_r['_r_phase_initiation']), dtype=torch.float32)
        _r_p_prog = torch.tensor(np.array(_r['_r_phase_progression']), dtype=torch.float32)
        _r_g_prog = torch.tensor(np.array(_r['_r_global_progression']), dtype=torch.float32)
        _r_a_prob = torch.tensor(np.array(_r['_r_action_probability']), dtype=torch.float32)
        _r_risk = torch.tensor(np.array(_r['_r_risk']), dtype=torch.float32)
        _r_dict = {
            '_r_phase_completion': _r_p_comp,
            '_r_phase_initiation': _r_p_init,
            '_r_phase_progression': _r_p_prog,
            '_r_global_progression': _r_g_prog,
            '_r_action_probability': _r_a_prob,
            '_r_risk': _r_risk,
        }
        q_dict = {}


        # current action and instrument
        c_a = torch.tensor(sample['c_a'], dtype=torch.float32)
        c_i = torch.tensor(sample['c_i'], dtype=torch.float32)

        # convert the binary phase to single integer class
        _p = torch.argmax(_p, dim=1)  # Convert to class integer

        # create dictiornary for batch
        data = {
            'current_states': z,
            'next_states': _z,
            'future_states': f_z,
            'next_actions': _a,
            'future_actions': f_a,
            'next_phases': _p,
            'next_rewards': _r_dict,
            'outcomes': q_dict,
            'current_actions': c_a,
            'current_instruments': c_i,
        }
        return data