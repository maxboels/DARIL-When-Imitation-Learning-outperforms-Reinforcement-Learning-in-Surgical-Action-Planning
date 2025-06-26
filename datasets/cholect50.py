#!/usr/bin/env python3
"""
CholecT50 Dataset Loading with Full Preprocessing Pipeline
Restored from original backup with consistent key naming fixes
"""

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
from typing import Dict, List, Any, Optional, Tuple

# RESTORED: All preprocessing imports
try:
    from datasets.preprocess.progression import add_progression_scores
    from datasets.preprocess.phase_completion import compute_phase_completion_rewards
    from datasets.preprocess.phase_transition import compute_phase_transition_rewards
    from datasets.preprocess.risk_scores import add_risk_scores
    from datasets.preprocess.action_scores import precompute_action_based_rewards
    from datasets.preprocess.action_scores import compute_action_phase_distribution
    PREPROCESSING_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Preprocessing modules not available: {e}")
    print("⚠️ Will skip reward preprocessing - please ensure preprocessing modules exist")
    PREPROCESSING_AVAILABLE = False

# Import reward analyzer
try:
    from datasets.reward_visualization_tool import analyze_rewards_during_loading
    REWARD_ANALYZER_AVAILABLE = True
except ImportError:
    print("⚠️ Reward analyzer not available, reward plotting will be skipped")
    REWARD_ANALYZER_AVAILABLE = False


def clean_csv_file_path(metadata, split='train', fold=0):

    remove_root_1 = f'/nfs/home/mboels/projects/self-distilled-swin/outputs/embeddings_{split}_set/fold{fold}/'
    if remove_root_1 in metadata['embedding_path'][0]:
        metadata['embedding_path'] = metadata['embedding_path'].apply(lambda x: x.replace(remove_root_1, f'fold{fold}/'))
        cleaned = True
    elif f'/fold{fold}/' in metadata['embedding_path'][0]:
        metadata['embedding_path'] = metadata['embedding_path'].apply(lambda x: x.replace(f'/fold{fold}/', f'fold{fold}/'))
        cleaned = True
    else:
        print(f"Root not found in embedding path, skipping")
        cleaned = False
    return metadata, cleaned

# Step 1: Data Loading from CholecT50 Dataset
def load_cholect50_data(cfg, logger, split='train', max_videos=None, test_on_train=False):
    """
    Load frame embeddings from the CholecT50 dataset for training or validation.
    RESTORED: Full preprocessing pipeline with proper error handling

    Returns:
        List of dictionaries containing video data
    """
    cfg_data = cfg['data']
    cfg_rewards = cfg.get('preprocess', {}).get('rewards', {})
    paths_config = cfg_data['paths']
    data_dir = paths_config['data_dir']
    metadata_file = paths_config['metadata_file']
    fold = paths_config['fold']

    # Set split to 'train' if test_on_train is True
    if test_on_train and split == 'test':
        split = 'train'
        logger.info("🔄 Test on training data enabled, using training split for loading data")
    
    # Try multiple data directory paths
    data_paths = [
        data_dir,
        data_dir.replace('/home/maxboels/datasets/CholecT50', '/nfs/home/mboels/datasets/CholecT50'),
        data_dir.replace('/nfs/home/mboels/datasets/CholecT50', '/home/maxboels/datasets/CholecT50')
    ]
    
    for path in data_paths:
        if os.path.exists(path):
            data_dir = path
            logger.info(f"✅ Using data directory: {data_dir}")
            break
    else:
        logger.error(f"❌ No data directory found in: {data_paths}")
        return [], []
    
    # Set split paths
    logger.info(f"Loading {split} data from {data_dir} with fold {fold}")
    split_folder = f"embeddings_{split}_set"    
    metadata_dir = os.path.join(data_dir, split_folder, f"fold{fold}")
    metadata_path = os.path.join(metadata_dir, metadata_file)

    # Global outcome file
    video_global_outcome_file = paths_config.get('video_global_outcome_file', 
                                                 'embeddings_f0_swin_bas_129_with_enhanced_global_metrics.csv')
    video_global_outcome_path = os.path.join(metadata_dir, video_global_outcome_file)

    # Check if metadata file exists
    if not os.path.exists(metadata_path):
        logger.error(f"❌ Metadata file not found: {metadata_path}")
        logger.error(f"❌ Expected directory structure:")
        logger.error(f"   {data_dir}/")
        logger.error(f"   ├── {split_folder}/")
        logger.error(f"   │   └── fold{fold}/")
        logger.error(f"   │       ├── {metadata_file}")
        logger.error(f"   │       └── {video_global_outcome_file}")
        
        # Return empty data rather than dummy data to highlight the real issue
        logger.error("❌ Cannot proceed without real data files")
        logger.error("💡 Please ensure your CholecT50 dataset is properly organized")
        return [], []

    # Load metadata if available
    try:
        metadata_df = pd.read_csv(metadata_path)
        logger.info(f"✅ Loaded metadata: {metadata_path} - Shape: {metadata_df.shape}")
    except Exception as e:
        logger.error(f"❌ Failed to load metadata CSV: {e}")
        return [], []

    split_desc = split.capitalize()
    
    logger.info(f"[{split_desc}] Start processing metadata file {metadata_path}")

    # Clean the metadata file path
    metadata_df, cleaned = clean_csv_file_path(metadata_df, split=split, fold=fold)
    if cleaned:
        logger.info(f"[{split_desc}] Cleaned metadata file paths in {metadata_path}")
        # Save cleaned metadata back to file
        metadata_df.to_csv(metadata_path, index=False)
        logger.info(f"[{split_desc}] Saved cleaned metadata to {metadata_path}")
    
    # RESTORED: Full preprocessing pipeline with proper error handling
    if cfg.get('preprocess', {}).get('extract_rewards', False) and PREPROCESSING_AVAILABLE:
        logger.info(f"[{split_desc}] Starting reward preprocessing...")
        
        try:
            # Progressive +1 reward near phase transitions
            if cfg_rewards.get('grounded', {}).get('phase_completion', False):
                logger.info(f"[{split_desc}] Computing phase completion rewards...")
                metadata_df = compute_phase_completion_rewards(
                    metadata_df, 
                    video_id_col='video', 
                    n_phases=7, 
                    transition_window=30,
                    phase_importance=None,
                    max_reward=1.0,
                    reward_function='exponential',
                    reward_distribution='left_sided'
                )
                metadata_file = metadata_file.replace('.csv', '_phase_complet.csv')
                metadata_df.to_csv(os.path.join(metadata_dir, metadata_file), index=False)
                logger.info(f"[{split_desc}] Added phase completion rewards to {metadata_file}") 
        except Exception as e:
            logger.warning(f"[{split_desc}] Phase completion preprocessing failed: {e}")

        try:
            if cfg_rewards.get('grounded', {}).get('phase_transition', False):
                logger.info(f"[{split_desc}] Computing phase transition rewards...")
                metadata_df = compute_phase_transition_rewards(
                    metadata_df, 
                    video_id_col='video', 
                    n_phases=7, 
                    reward_window=5, 
                    phase_importance=None,
                    reward_value=1.0
                )
                metadata_file = metadata_file.replace('.csv', '_phase_transit.csv')
                metadata_df.to_csv(os.path.join(metadata_dir, metadata_file), index=False)
                logger.info(f"[{split_desc}] Added phase transition rewards to {metadata_file}")
        except Exception as e:
            logger.warning(f"[{split_desc}] Phase transition preprocessing failed: {e}")

        try:
            # Compute reward signals for each frame (state)
            if (cfg_rewards.get('grounded', {}).get('phase_progression', False) or 
                cfg_rewards.get('grounded', {}).get('global_progression', False)):
                logger.info(f"[{split_desc}] Computing progression scores...")
                metadata_df = add_progression_scores(
                    metadata_df,
                    add_phase_progression=cfg_rewards.get('grounded', {}).get('phase_progression', False),
                    add_global_progression=cfg_rewards.get('grounded', {}).get('global_progression', False)
                )
                metadata_file = metadata_file.replace('.csv', '_prog.csv')
                metadata_df.to_csv(os.path.join(metadata_dir, metadata_file), index=False)
                logger.info(f"[{split_desc}] Added phase and global progression rewards to {metadata_file}")
        except Exception as e:
            logger.warning(f"[{split_desc}] Progression score preprocessing failed: {e}")
            
        try:
            # Compute action-based rewards for each frame (state) conditioned on phases
            if cfg_rewards.get('imitation', {}).get('action_distribution', False):
                logger.info(f"[{split_desc}] Computing action-based rewards...")
                metadata_df = precompute_action_based_rewards(metadata_df, n_phases=7, n_actions=100, epsilon=1e-10)
                metadata_file = metadata_file.replace('.csv', '_prob_action.csv')
                metadata_df.to_csv(os.path.join(metadata_dir, metadata_file), index=False)
                logger.info(f"[{split_desc}] Added action distribution rewards to {metadata_file}")
        except Exception as e:
            logger.warning(f"[{split_desc}] Action distribution preprocessing failed: {e}")

        try:
            if cfg_rewards.get('expert_knowledge', {}).get('risk_score', False):
                logger.info(f"[{split_desc}] Computing risk scores...")
                metadata_df = add_risk_scores(
                    metadata_df, 
                    cfg_data, 
                    split, 
                    fold, 
                    frame_risk_agg=cfg_rewards.get('expert_knowledge', {}).get('frame_risk_agg', 'max')
                )
                metadata_file = metadata_file.replace('.csv', '_risk.csv')
                metadata_df.to_csv(os.path.join(metadata_dir, metadata_file), index=False)
                logger.info(f"[{split_desc}] Added risk scores to {metadata_file}")
        except Exception as e:
            logger.warning(f"[{split_desc}] Risk score preprocessing failed: {e}")
        
        try:
            # Add correct video_global_outcomes row and with all columns to the metadata file using the video id as unique identifier
            if os.path.exists(video_global_outcome_path):
                logger.info(f"[{split_desc}] Adding global outcome scores...")
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
                logger.info(f"[{split_desc}] Added global outcome scores to {metadata_file}")
            else:
                logger.warning(f"[{split_desc}] Global outcome file {video_global_outcome_path} not found, skipping global outcome scores")
        except Exception as e:
            logger.warning(f"[{split_desc}] Global outcome preprocessing failed: {e}")
            
    elif cfg.get('preprocess', {}).get('extract_rewards', False) and not PREPROCESSING_AVAILABLE:
        logger.warning(f"[{split_desc}] Reward extraction requested but preprocessing modules not available")
        logger.warning(f"[{split_desc}] Please ensure all preprocessing modules are properly installed")
    else:
        logger.info(f"[{split_desc}] Rewards extraction skipped, using metadata file {metadata_path}")

    # Find all videos in metadata csv file
    if metadata_df is not None:
        video_ids = metadata_df['video'].unique().tolist()
    else:
        logger.error("❌ No metadata available")
        return [], []
        
    if max_videos:
        video_ids = video_ids[:max_videos]
    
    logger.info(f"[{split_desc}] Found {len(video_ids)} videos in metadata file")
    
    # Initialize data list
    data = []
    video_durations = []
    
    # RESTORED: Real frame embedding loading
    logger.info(f"[{split_desc}] Loading data for {len(video_ids)} videos")
    for video_id in tqdm(video_ids, desc="Loading frames for videos"):
        try:
            # Filter metadata for this video
            video_metadata = metadata_df[metadata_df['video'] == video_id]
            if video_metadata.empty:
                logger.warning(f"No metadata found for video {video_id}")
                continue
                
            frame_files = video_metadata['embedding_path'].tolist()

            # Load frame embeddings
            video_frame_embeddings = []
            for frame_file in tqdm(frame_files, desc=f"Frames for {video_id}", leave=False):
                embedding_path = os.path.join(data_dir, split_folder, frame_file)
                if os.path.exists(embedding_path):
                    embedding = np.load(embedding_path)
                    video_frame_embeddings.append(embedding)
                else:
                    logger.warning(f"Frame embedding not found: {embedding_path}")
                    
            if not video_frame_embeddings:
                logger.warning(f"No frame embeddings loaded for video {video_id}")
                continue
                
            video_frame_embeddings = np.array(video_frame_embeddings)        
            embedding_dim = video_frame_embeddings.shape[1]
            
            num_frames = len(video_frame_embeddings)
            video_durations.append(num_frames)
            
            # RESTORED: Extract real reward values from metadata_df per video
            rewards = {}
            reward_columns = {
                '_r_risk_penalty': 'risk_score_max',
                '_r_phase_completion': 'phase_completion_reward', 
                '_r_phase_initiation': 'phase_initiation_reward',
                '_r_phase_progression': 'phase_prog',
                '_r_global_progression': 'global_prog',
                '_r_action_probability': 'action_reward'
            }
            
            for reward_key, column_name in reward_columns.items():
                if column_name in video_metadata.columns:
                    rewards[reward_key] = video_metadata[column_name].values
                else:
                    logger.warning(f"Reward column {column_name} not found for {video_id}")
                    rewards[reward_key] = np.zeros(num_frames)

            # RESTORED: Extract real global outcome scores
            outcomes = {}
            outcome_columns = {
                'q_global_avg_risk': 'glob_avg_risk',
                'q_global_critical_risk': 'glob_critical_risk_events', 
                'q_global_outcome_score': 'glob_global_outcome_score'
            }
            
            for outcome_key, column_name in outcome_columns.items():
                if column_name in video_metadata.columns:
                    outcomes[outcome_key] = video_metadata[column_name].values
                else:
                    logger.warning(f"Outcome column {column_name} not found for {video_id}")
                    outcomes[outcome_key] = np.zeros(num_frames)

            # RESTORED: Extract real action, instrument, verb, phase data
            # indices from column 'tri0':'tri99' (actions)
            action_columns = [f'tri{i}' for i in range(0, 100)]
            available_action_cols = [col for col in action_columns if col in video_metadata.columns]
            if available_action_cols:
                video_action_classes = video_metadata[available_action_cols].values
                # Pad to 100 if needed
                if len(available_action_cols) < 100:
                    padding = np.zeros((num_frames, 100 - len(available_action_cols)))
                    video_action_classes = np.hstack([video_action_classes, padding])
            else:
                logger.warning(f"No action columns found for {video_id}")
                video_action_classes = np.zeros((num_frames, 100))

            # indices from column 'inst0':'inst5' (instruments)
            instrument_columns = [f'inst{i}' for i in range(0, 6)]
            available_inst_cols = [col for col in instrument_columns if col in video_metadata.columns]
            if available_inst_cols:
                video_instrument_classes = video_metadata[available_inst_cols].values
                # Pad to 6 if needed
                if len(available_inst_cols) < 6:
                    padding = np.zeros((num_frames, 6 - len(available_inst_cols)))
                    video_instrument_classes = np.hstack([video_instrument_classes, padding])
            else:
                logger.warning(f"No instrument columns found for {video_id}")
                video_instrument_classes = np.zeros((num_frames, 6))

            # verb columns
            verb_columns = [f'v{i}' for i in range(0, 9)]
            available_verb_cols = [col for col in verb_columns if col in video_metadata.columns]
            if available_verb_cols:
                video_verb_classes = video_metadata[available_verb_cols].values
            else:
                video_verb_classes = np.zeros((num_frames, 9))

            # phase columns  
            phase_columns = [f'p{i}' for i in range(0, 7)]
            available_phase_cols = [col for col in phase_columns if col in video_metadata.columns]
            if available_phase_cols:
                video_phase_classes = video_metadata[available_phase_cols].values
                # Pad to 7 if needed
                if len(available_phase_cols) < 7:
                    padding = np.zeros((num_frames, 7 - len(available_phase_cols)))
                    video_phase_classes = np.hstack([video_phase_classes, padding])
            else:
                logger.warning(f"No phase columns found for {video_id}")
                video_phase_classes = np.zeros((num_frames, 7))

            # FIXED: Store video data with consistent naming
            data.append({
                'video_id': video_id,
                'video_dir': os.path.join(data_dir, split_folder, video_id),
                'frame_embeddings': video_frame_embeddings,
                'actions_binaries': video_action_classes,  # FIXED: Consistent naming
                'instruments_binaries': video_instrument_classes,
                'verb_binaries': video_verb_classes,
                'phase_binaries': video_phase_classes,
                'num_frames': num_frames,
                'next_rewards': rewards,  # For compatibility
                'rewards': rewards,       # For compatibility
                'outcomes': outcomes,
            })
            
            logger.info(f"✅ Loaded video {video_id}: {num_frames} frames, {embedding_dim}-dims embeddings")
            
        except Exception as e:
            logger.error(f"❌ Failed to load video {video_id}: {e}")
            continue

    if not data:
        logger.error("❌ No valid videos loaded!")
        logger.error("💡 Please check:")
        logger.error("   1. Data directory structure")
        logger.error("   2. Metadata file format") 
        logger.error("   3. Embedding file paths")
        logger.error("   4. Column names in metadata")
        return []
    
    if test_on_train:
        logger.info(f"🔄 Test on training data enabled, using {len(data)} videos from training split")
    else:
        logger.info(f"📂 Loaded {len(data)} videos from {split} split")
    
    # NEW: Automatic reward analysis after loading
    analyze_rewards = cfg.get('preprocess', {}).get('analyze_rewards', False)
    if analyze_rewards and REWARD_ANALYZER_AVAILABLE and data:
        logger.info(f"📊 Analyzing reward values for {len(data)} videos...")
        try:
            analysis_dir = f"reward_analysis/{split}/{datetime.now().strftime('%Y%m%d_%H%M')}"
            os.makedirs(analysis_dir, exist_ok=True)
            analyze_rewards_during_loading(data, analysis_dir)
            logger.info(f"✅ Reward analysis complete! Check: {analysis_dir}/")
        except Exception as e:
            logger.warning(f"⚠️ Reward analysis failed: {e}")
    elif analyze_rewards and not REWARD_ANALYZER_AVAILABLE:
        logger.warning("⚠️ Reward analysis requested but reward_visualization_tool not available")
    
    return data

# Quick test script functionality
def quick_reward_analysis_test(config_path: str = 'config_dgx_all_v6.yaml'):
    """
    Quick test to analyze reward values immediately.
    """
    
    print("🚀 QUICK REWARD ANALYSIS TEST")
    print("=" * 40)
    
    try:
        import yaml
        
        # Load config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Create simple logger
        class SimpleLogger:
            def info(self, msg): print(f"INFO: {msg}")
            def warning(self, msg): print(f"WARNING: {msg}")
            def error(self, msg): print(f"ERROR: {msg}")
        
        logger = SimpleLogger()
        
        # Load a small amount of data for quick analysis
        print("📂 Loading small dataset for reward analysis...")
        train_data = load_cholect50_data(
            config, logger, 
            split='train', 
            max_videos=2,  # Just 2 videos for quick test
            analyze_rewards=True  # Enable automatic analysis
        )
        
        if train_data:
            print(f"✅ Loaded {len(train_data)} videos")
            print(f"📊 Reward analysis plots should be saved in reward_analysis_* directory")
            
            # Print quick reward summary
            print(f"\n📋 QUICK REWARD SUMMARY:")
            for video in train_data:
                video_id = video.get('video_id', 'unknown')
                rewards = video.get('rewards', {})
                print(f"\n  Video: {video_id}")
                for reward_type, reward_values in rewards.items():
                    values = np.array(reward_values)
                    if values.size > 0:
                        non_zero = np.sum(np.abs(values) > 1e-6)
                        print(f"    {reward_type}: {values.shape[0]} frames, {non_zero} non-zero, range [{np.min(values):.4f}, {np.max(values):.4f}]")
            
            return True
        else:
            print("❌ No data loaded")
            return False
            
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("📊 CHOLECT50 LOADER WITH REWARD ANALYSIS")
    print("=" * 50)
    print("Automatically analyzes and plots reward values during loading")
    print()
    
    # Run quick test
    success = quick_reward_analysis_test()
    
    if success:
        print("\n🎉 SUCCESS!")
        print("Reward analysis plots should be saved in the reward_analysis_* directory")
        print("Check the plots to verify your reward values are correctly scaled")
    else:
        print("\n❌ FAILED!")
        print("Check your data paths and configuration")


# # RESTORED: Real dataset implementation
# from torch.utils.data import DataLoader

# def create_video_dataloaders(cfg, data, batch_size=32, shuffle=True, num_workers=4):
#     """
#     Create a separate dataloader for each video in the dataset.
    
#     Args:
#         cfg: Configuration dictionary
#         data: List of video dictionaries from load_cholect50_data()
#         batch_size: Batch size for the dataloaders
#         shuffle: Whether to shuffle the samples
#         num_workers: Number of worker processes for data loading
        
#     Returns:
#         Dictionary mapping video IDs to their respective DataLoaders
#     """
#     video_dataloaders = {}
    
#     for video in data:
#         video_id = video['video_id']
        
#         # Create a dataset with only this video's data
#         video_dataset = NextFramePredictionDataset(cfg['data'], [video])
        
#         # Create a dataloader for this video
#         video_dataloader = DataLoader(
#             video_dataset,
#             batch_size=batch_size,
#             shuffle=shuffle,
#             num_workers=num_workers
#         )
        
#         # Store the dataloader with the video ID as the key
#         video_dataloaders[video_id] = video_dataloader
    
#     return video_dataloaders


# # RESTORED: Original NextFramePredictionDataset (with key fixes)
# class NextFramePredictionDataset(Dataset):
#     def __init__(self, cfg, data):
#         """
#         Initialize the dataset with sequences of frame embeddings
#         RESTORED from original with key naming fixes

#         Lexic:
#             "_" prefix indicates next frame
#             "f" prefix indicates future frame
        
#         Args:
#             data: List of video dictionaries containing frame_embeddings and actions_binaries
#             context_length: Number of previous frames to include in each input sequence
#             padding_value: Value to use for padding when not enough previous frames exist
#         """
#         self.samples = []
        
#         context_length = cfg.get('context_length', 10)
#         padding_value = cfg.get('padding_value', 0.0)
#         train_shift = cfg.get('train_shift', 1)
#         max_horizon = cfg.get('max_horizon', 15)

#         for video in data:
#             video_id = video['video_id']
#             embeddings = video['frame_embeddings']
#             actions = video['actions_binaries']  # FIXED: Use consistent key
#             instruments = video['instruments_binaries']
#             phases = video['phase_binaries']

#             # rewards and outcomes
#             rewards = video['next_rewards']
#             phase_completion = rewards['_r_phase_completion']
#             phase_initiation = rewards['_r_phase_initiation']
#             phase_progression = rewards['_r_phase_progression']
#             global_progression = rewards['_r_global_progression']
#             action_probability = rewards['_r_action_probability']
#             _r_risk_penalty = rewards['_r_risk_penalty']

#             # outcomes
#             q = video['outcomes']
            
#             num_actions = len(actions[0])
#             num_instruments = len(instruments[0])
#             num_phases = len(phases[0])
#             reward_dim = 1
#             embedding_dim = len(embeddings[0])

#             for i in range(len(embeddings) - 1):
#                 # For position i, take context_length frames from the left (previous frames)
#                 z_seq = []
#                 _z_seq = []
#                 f_z_seq = [] # future states from i+1 to i+max_horizon
#                 _a_seq = []
#                 _p_seq = [] # next phases from i+1 to i+max_horizon
#                 f_a_seq = [] # future actions from i+1 to i+max_horizon

#                 # rewards
#                 _r_p_comp_seq = []
#                 _r_p_init_seq = []
#                 _r_p_prog_seq = []
#                 _r_g_prog_seq = []
#                 _r_a_prob_seq = []
#                 _r_risk_seq = []
                
#                 # outcomes
#                 q_seq = []

#                 # current
#                 c_a = actions[i]
#                 c_i = instruments[i]
                
#                 # Add previous frames, using padding if needed
#                 for j in range(i - context_length + 1, i + 1):
#                     if j < 0:
#                         # Padding for positions before the start of the video
#                         z_seq.append([padding_value] * embedding_dim)
#                     else:
#                         z_seq.append(embeddings[j])
                
#                 # Add the shifted next frame and action, using padding if needed
#                 for k in range(i - context_length + 1 + train_shift, i + 1 + train_shift):
#                     if k < 0:
#                         # Padding for positions before the start of the video
#                         _z_seq.append([padding_value] * embedding_dim)
#                         _a_seq.append([0] * num_actions)
#                         _p_seq.append([0] * num_phases)
#                         # rewards
#                         _r_p_comp_seq.append([0.0] * reward_dim)
#                         _r_p_init_seq.append([0.0] * reward_dim)
#                         _r_p_prog_seq.append([0.0] * reward_dim)
#                         _r_g_prog_seq.append([0.0] * reward_dim)
#                         _r_a_prob_seq.append([0.0] * reward_dim)
#                         _r_risk_seq.append([1.0] * reward_dim)

#                     elif k >= len(embeddings):
#                         # Padding for positions after the end of the video
#                         _z_seq.append([padding_value] * embedding_dim)
#                         _a_seq.append([0] * num_actions)
#                         _p_seq.append([0] * num_phases)
#                         # rewards
#                         _r_p_comp_seq.append([0.0] * reward_dim)
#                         _r_p_init_seq.append([0.0] * reward_dim)
#                         _r_p_prog_seq.append([0.0] * reward_dim)
#                         _r_g_prog_seq.append([0.0] * reward_dim)
#                         _r_a_prob_seq.append([0.0] * reward_dim)
#                         _r_risk_seq.append([1.0] * reward_dim) # 1.0 is the minimum risk score
#                     else:
#                         _z_seq.append(embeddings[k])
#                         _a_seq.append(actions[k])
#                         _p_seq.append(phases[k])
#                         # rewards (make sure it is a list of lists)
#                         _r_p_comp_seq.append([phase_completion[k]])
#                         _r_p_init_seq.append([phase_initiation[k]])
#                         _r_p_prog_seq.append([phase_progression[k]])
#                         _r_g_prog_seq.append([global_progression[k]])
#                         _r_a_prob_seq.append([action_probability[k]])
#                         _r_risk_seq.append([_r_risk_penalty[k]])
                
#                 # Add future actions and states
#                 for k in range(i + 1, min(i + 1 + max_horizon, len(embeddings))):
#                     f_z_seq.append(embeddings[k])
#                     f_a_seq.append(actions[k])

#                 if len(f_a_seq) < max_horizon:
#                     # Padding for positions after the end of the video
#                     for _ in range(max_horizon - len(f_a_seq)):
#                         f_a_seq.append([0] * num_actions)
#                         f_z_seq.append([padding_value] * embedding_dim)

#                 # Add the sequence to the samples list
#                 self.samples.append({
#                     'v_id': video_id,
#                     'z': z_seq,     # Sequence of frames from (i-context_length+1) to i
#                     '_z': _z_seq,   # Sequence of frames from (i-context_length+2) to i+1
#                     'f_z': f_z_seq,   # Future states from (i+1) to (i+max_horizon)
#                     '_a': _a_seq,    # Sequence of actions from (i-context_length+2) to i+1
#                     'f_a': f_a_seq,
#                     '_p': _p_seq,    # Sequence of phases from (i-context_length+2) to i+1
#                     'c_a': c_a,     # Current Action at position i
#                     'c_i': c_i,     # Current Instrument at position i
#                     # rewards
#                     '_r': {
#                         '_r_phase_completion': _r_p_comp_seq,
#                         '_r_phase_initiation': _r_p_init_seq,
#                         '_r_phase_progression': _r_p_prog_seq,
#                         '_r_global_progression': _r_g_prog_seq,
#                         '_r_action_probability': _r_a_prob_seq,
#                         '_r_risk': _r_risk_seq,
#                     },
#                     # outcomes
#                 })
    
#     def __len__(self):
#         return len(self.samples)
    
#     def __getitem__(self, idx):
#         sample = self.samples[idx]

#         # Convert lists to numpy arrays first, then to tensors
#         z = torch.tensor(np.array(sample['z']), dtype=torch.float32)  # Shape: [context_length, embedding_dim]
#         _z = torch.tensor(np.array(sample['_z']), dtype=torch.float32)
#         f_z = torch.tensor(np.array(sample['f_z']), dtype=torch.float32)  # Shape: [max_horizon, embedding_dim]
#         _a = torch.tensor(np.array(sample['_a']), dtype=torch.float32)
#         f_a = torch.tensor(np.array(sample['f_a']), dtype=torch.float32)
#         _p = torch.tensor(np.array(sample['_p']), dtype=torch.float32)  # Shape: [max_horizon, num_phases]

#         # rewards
#         _r = sample['_r']
#         _r_p_comp = torch.tensor(np.array(_r['_r_phase_completion']), dtype=torch.float32)
#         _r_p_init = torch.tensor(np.array(_r['_r_phase_initiation']), dtype=torch.float32)
#         _r_p_prog = torch.tensor(np.array(_r['_r_phase_progression']), dtype=torch.float32)
#         _r_g_prog = torch.tensor(np.array(_r['_r_global_progression']), dtype=torch.float32)
#         _r_a_prob = torch.tensor(np.array(_r['_r_action_probability']), dtype=torch.float32)
#         _r_risk = torch.tensor(np.array(_r['_r_risk']), dtype=torch.float32)
#         _r_dict = {
#             '_r_phase_completion': _r_p_comp,
#             '_r_phase_initiation': _r_p_init,
#             '_r_phase_progression': _r_p_prog,
#             '_r_global_progression': _r_g_prog,
#             '_r_action_probability': _r_a_prob,
#             '_r_risk': _r_risk,
#         }
#         q_dict = {}

#         # current action and instrument
#         c_a = torch.tensor(sample['c_a'], dtype=torch.float32)
#         c_i = torch.tensor(sample['c_i'], dtype=torch.float32)

#         # convert the binary phase to single integer class
#         _p = torch.argmax(_p, dim=1)  # Convert to class integer

#         # create dictionary for batch - FIXED: Use consistent naming
#         data = {
#             'current_states': z,
#             'next_states': _z,
#             'future_states': f_z,
#             'next_actions': _a,      # FIXED: Consistent with world model dataset
#             'future_actions': f_a,
#             'next_phases': _p,
#             'next_rewards': _r_dict,
#             'outcomes': q_dict,
#             'current_actions': c_a,
#             'current_instruments': c_i,
#         }
#         return data


# # Compatibility function for the 3-method comparison
# def load_cholect50_data_for_comparison(config: Dict[str, Any], logger) -> Tuple[List[Dict], List[Dict]]:
#     """
#     Compatibility wrapper for the three-method comparison experiment.
#     This ensures the data format is consistent with the method comparison pipeline.
#     """
#     logger.info("📂 Loading CholecT50 data for method comparison...")
    
#     # Determine if we have training/test splits configured
#     train_config = config.get('experiment', {}).get('train', {})
#     test_config = config.get('experiment', {}).get('test', {})
    
#     max_train_videos = train_config.get('max_videos', config.get('experiment', {}).get('max_videos', 10))
#     max_test_videos = test_config.get('max_videos', config.get('experiment', {}).get('max_videos', 5))
    
#     # Load training data
#     try:
#         train_data = load_cholect50_data(config, logger, split='train', max_videos=max_train_videos)
#         logger.info(f"✅ Loaded {len(train_data)} training videos")
#     except Exception as e:
#         logger.error(f"❌ Failed to load training data: {e}")
#         train_data = []
    
#     # Load test data  
#     try:
#         test_data = load_cholect50_data(config, logger, split='test', max_videos=max_test_videos)  
#         logger.info(f"✅ Loaded {len(test_data)} test videos")
#     except Exception as e:
#         logger.error(f"❌ Failed to load test data: {e}")
#         test_data = []
    
#     # If no real data loaded, provide guidance but don't use dummy data
#     if not train_data and not test_data:
#         logger.error("❌ No real data could be loaded")
#         logger.error("💡 Please ensure:")
#         logger.error("   1. CholecT50 dataset is properly downloaded and organized")
#         logger.error("   2. Preprocessing modules are available") 
#         logger.error("   3. Configuration paths are correct")
#         logger.error("   4. Metadata files exist and have expected columns")
        
#         # Don't return dummy data - this forces addressing the real issue
#         return [], []
    
#     return train_data, test_data


# if __name__ == "__main__":
#     print("📂 CHOLECT50 DATASET LOADER - RESTORED WITH FULL PREPROCESSING")
#     print("=" * 70)
    
#     # Test configuration
#     test_config = {
#         'data': {
#             'context_length': 20,
#             'train_shift': 1,
#             'padding_value': 0.0,
#             'max_horizon': 15,
#             'paths': {
#                 'data_dir': '/home/maxboels/datasets/CholecT50',
#                 'metadata_file': 'embeddings_metadata.csv',
#                 'fold': 0,
#                 'video_global_outcome_file': 'embeddings_f0_swin_bas_129_with_enhanced_global_metrics.csv'
#             },
#             'frame_risk_agg': 'max'
#         },
#         'preprocess': {
#             'extract_rewards': True,
#             'rewards': {
#                 'grounded': {
#                     'phase_completion': True,
#                     'phase_transition': True,
#                     'phase_progression': True,
#                     'global_progression': True
#                 },
#                 'imitation': {
#                     'action_distribution': True
#                 },
#                 'expert_knowledge': {
#                     'risk_score': True,
#                     'frame_risk_agg': 'max'
#                 }
#             }
#         },
#         'experiment': {
#             'train': {'max_videos': 2},
#             'test': {'max_videos': 1}
#         }
#     }
    
#     # Test logger
#     class TestLogger:
#         def info(self, msg): print(f"INFO: {msg}")
#         def warning(self, msg): print(f"WARNING: {msg}")
#         def error(self, msg): print(f"ERROR: {msg}")
    
#     logger = TestLogger()
    
#     print("🧪 Testing data loading with real preprocessing pipeline...")
#     train_data, test_data = load_cholect50_data_for_comparison(test_config, logger)
    
#     if train_data or test_data:
#         print(f"✅ Successfully loaded {len(train_data)} train + {len(test_data)} test videos")
#         if train_data:
#             video = train_data[0]
#             print(f"📊 Sample video: {video['video_id']}")
#             print(f"   Frames: {video['frame_embeddings'].shape}")
#             print(f"   Actions: {video['actions_binaries'].shape}")
#             print(f"   Phases: {video['phase_binaries'].shape}")
#             print(f"   Rewards: {list(video['rewards'].keys())}")
#     else:
#         print("❌ No data loaded - this will force addressing the real data issues")
#         print("💡 This is intentional - dummy data masks real problems")
        
#     print("\n🎯 Key differences from dummy data version:")
#     print("✅ Real preprocessing pipeline restored")
#     print("✅ All reward computation functions included")
#     print("✅ Real file loading with proper error handling") 
#     print("✅ No dummy data fallback - forces fixing real issues")
#     print("✅ Proper data validation and error reporting")