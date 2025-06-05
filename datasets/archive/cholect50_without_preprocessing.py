#!/usr/bin/env python3
"""
CholecT50 Dataset Loading for Surgical RL Comparison
Loads frame embeddings, actions, phases, and rewards for all three methods
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import torch


def load_cholect50_data(config: Dict[str, Any], logger) -> Tuple[List[Dict], List[Dict]]:
    """
    Load CholecT50 data for the three-method comparison.
    
    Args:
        config: Configuration dictionary
        logger: Logger instance
        
    Returns:
        Tuple of (train_data, test_data) where each is a list of video dictionaries
    """
    
    logger.info("📂 Loading CholecT50 dataset...")
    
    # Extract configuration
    data_config = config.get('data', {})
    experiment_config = config.get('experiment', {})
    
    # Get paths
    data_dir = data_config.get('paths', {}).get('data_dir', '/home/maxboels/datasets/CholecT50')
    csv_file = data_config.get('paths', {}).get('video_global_outcome_file', 
                                               'embeddings_f0_swin_bas_129_with_enhanced_global_metrics.csv')
    
    csv_path = os.path.join(data_dir, csv_file)
    
    if not os.path.exists(csv_path):
        logger.error(f"❌ Data file not found: {csv_path}")
        # Return dummy data for testing
        return _create_dummy_data(config, logger)
    
    logger.info(f"📄 Loading data from: {csv_path}")
    
    # Load CSV data
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"✅ CSV loaded successfully - Shape: {df.shape}")
    except Exception as e:
        logger.error(f"❌ Failed to load CSV: {e}")
        return _create_dummy_data(config, logger)
    
    # Process data into video dictionaries
    train_data, test_data = _process_cholect50_data(df, config, logger)
    
    # Apply video limits from config
    max_train_videos = experiment_config.get('train', {}).get('max_videos', 
                                                              experiment_config.get('max_videos', 10))
    max_test_videos = experiment_config.get('test', {}).get('max_videos', 
                                                            experiment_config.get('max_videos', 5))
    
    if max_train_videos and len(train_data) > max_train_videos:
        train_data = train_data[:max_train_videos]
        logger.info(f"🔄 Limited training videos to: {len(train_data)}")
    
    if max_test_videos and len(test_data) > max_test_videos:
        test_data = test_data[:max_test_videos]
        logger.info(f"🔄 Limited test videos to: {len(test_data)}")
    
    logger.info(f"✅ Data loading completed")
    logger.info(f"   Training videos: {len(train_data)}")
    logger.info(f"   Test videos: {len(test_data)}")
    
    return train_data, test_data


def _process_cholect50_data(df: pd.DataFrame, config: Dict, logger) -> Tuple[List[Dict], List[Dict]]:
    """Process the raw CSV data into video dictionaries."""
    
    logger.info("🔄 Processing CholecT50 data...")
    
    # Get unique videos
    video_ids = df['video_id'].unique()
    logger.info(f"📊 Found {len(video_ids)} unique videos")
    
    # Split into train/test (simple split for now)
    split_point = int(len(video_ids) * 0.8)
    train_video_ids = video_ids[:split_point]
    test_video_ids = video_ids[split_point:]
    
    logger.info(f"📊 Train videos: {len(train_video_ids)}, Test videos: {len(test_video_ids)}")
    
    # Process training videos
    train_data = []
    for video_id in train_video_ids:
        video_data = _process_single_video(df, video_id, config, logger)
        if video_data:
            train_data.append(video_data)
    
    # Process test videos
    test_data = []
    for video_id in test_video_ids:
        video_data = _process_single_video(df, video_id, config, logger)
        if video_data:
            test_data.append(video_data)
    
    logger.info(f"✅ Processed {len(train_data)} training and {len(test_data)} test videos")
    
    return train_data, test_data


def _process_single_video(df: pd.DataFrame, video_id: str, config: Dict, logger) -> Optional[Dict]:
    """Process a single video into the required format."""
    
    # Filter data for this video
    video_df = df[df['video_id'] == video_id].sort_values('frame_idx')
    
    if len(video_df) == 0:
        return None
    
    # Extract frame embeddings (assuming they're in columns with 'embedding' or 'feature')
    embedding_cols = [col for col in video_df.columns if 'embedding' in col.lower() or 'feature' in col.lower()]
    
    if not embedding_cols:
        # Create dummy embeddings if none found
        frame_embeddings = np.random.randn(len(video_df), 1024).astype(np.float32)
        logger.warning(f"⚠️ No embedding columns found for {video_id}, using dummy embeddings")
    else:
        frame_embeddings = video_df[embedding_cols].values.astype(np.float32)
        # Pad or truncate to 1024 dimensions
        if frame_embeddings.shape[1] < 1024:
            padding = np.zeros((frame_embeddings.shape[0], 1024 - frame_embeddings.shape[1]))
            frame_embeddings = np.hstack([frame_embeddings, padding])
        elif frame_embeddings.shape[1] > 1024:
            frame_embeddings = frame_embeddings[:, :1024]
    
    # Extract action binaries (assuming columns with 'action' in name)
    action_cols = [col for col in video_df.columns if 'action' in col.lower() and 'binary' in col.lower()]
    
    if not action_cols:
        # Create dummy actions
        actions_binaries = np.random.randint(0, 2, (len(video_df), 100)).astype(np.float32)
        logger.warning(f"⚠️ No action columns found for {video_id}, using dummy actions")
    else:
        actions_binaries = video_df[action_cols].values.astype(np.float32)
        # Pad or truncate to 100 actions
        if actions_binaries.shape[1] < 100:
            padding = np.zeros((actions_binaries.shape[0], 100 - actions_binaries.shape[1]))
            actions_binaries = np.hstack([actions_binaries, padding])
        elif actions_binaries.shape[1] > 100:
            actions_binaries = actions_binaries[:, :100]
    
    # Extract phase information
    phase_cols = [col for col in video_df.columns if 'phase' in col.lower()]
    
    if phase_cols:
        # If we have phase columns, create one-hot encoding
        phase_data = video_df[phase_cols[0]].values
        unique_phases = np.unique(phase_data)
        num_phases = min(len(unique_phases), 7)
        
        phase_binaries = np.zeros((len(video_df), num_phases))
        for i, phase in enumerate(phase_data):
            if phase < num_phases:
                phase_binaries[i, int(phase)] = 1.0
    else:
        # Create dummy phases
        phase_binaries = np.zeros((len(video_df), 7))
        phase_binaries[:, 0] = 1.0  # All frames in phase 0
    
    # Extract or create rewards
    rewards = _extract_rewards(video_df, config)
    
    video_data = {
        'video_id': video_id,
        'frame_embeddings': frame_embeddings,
        'actions_binaries': actions_binaries,
        'phase_binaries': phase_binaries,
        'rewards': rewards,
        'next_rewards': rewards,  # For compatibility
        'metadata': {
            'num_frames': len(video_df),
            'video_length': len(video_df),
            'data_source': 'CholecT50'
        }
    }
    
    return video_data


def _extract_rewards(video_df: pd.DataFrame, config: Dict) -> Dict[str, np.ndarray]:
    """Extract or create reward signals."""
    
    rewards = {}
    
    # Look for existing reward columns
    reward_cols = [col for col in video_df.columns if 'reward' in col.lower() or '_r_' in col]
    
    if reward_cols:
        for col in reward_cols:
            reward_name = col.replace('_r_', '').replace('reward_', '')
            rewards[reward_name] = video_df[col].values.astype(np.float32)
    else:
        # Create dummy rewards for the different types we expect
        num_frames = len(video_df)
        
        rewards = {
            'phase_progression': np.random.normal(0.1, 0.3, num_frames).astype(np.float32),
            'phase_completion': np.random.exponential(0.2, num_frames).astype(np.float32),
            'phase_initiation': np.random.normal(0.05, 0.1, num_frames).astype(np.float32),
            'safety': np.random.normal(0.8, 0.2, num_frames).astype(np.float32),
            'efficiency': np.random.normal(0.5, 0.3, num_frames).astype(np.float32),
            'action_probability': np.random.uniform(0.1, 0.9, num_frames).astype(np.float32),
            'risk_penalty': np.random.normal(-0.1, 0.2, num_frames).astype(np.float32)
        }
    
    return rewards


def _create_dummy_data(config: Dict, logger) -> Tuple[List[Dict], List[Dict]]:
    """Create dummy data for testing when real data is not available."""
    
    logger.warning("⚠️ Creating dummy data for testing")
    
    experiment_config = config.get('experiment', {})
    max_train_videos = experiment_config.get('train', {}).get('max_videos', 2)
    max_test_videos = experiment_config.get('test', {}).get('max_videos', 1)
    
    train_data = []
    test_data = []
    
    # Create training videos
    for i in range(max_train_videos):
        video_data = _create_dummy_video(f"train_video_{i+1}", config)
        train_data.append(video_data)
    
    # Create test videos
    for i in range(max_test_videos):
        video_data = _create_dummy_video(f"test_video_{i+1}", config)
        test_data.append(video_data)
    
    logger.info(f"✅ Created {len(train_data)} dummy training and {len(test_data)} dummy test videos")
    
    return train_data, test_data


def _create_dummy_video(video_id: str, config: Dict) -> Dict:
    """Create a single dummy video."""
    
    # Random video length between 50-200 frames
    num_frames = np.random.randint(50, 201)
    
    # Create dummy embeddings
    frame_embeddings = np.random.randn(num_frames, 1024).astype(np.float32)
    
    # Create dummy actions (sparse binary actions)
    actions_binaries = np.zeros((num_frames, 100), dtype=np.float32)
    for i in range(num_frames):
        # Randomly activate 1-3 actions per frame
        num_active = np.random.randint(0, 4)
        active_indices = np.random.choice(100, num_active, replace=False)
        actions_binaries[i, active_indices] = 1.0
    
    # Create dummy phases (gradually progress through phases)
    phase_binaries = np.zeros((num_frames, 7), dtype=np.float32)
    for i in range(num_frames):
        phase_idx = min(6, i // (num_frames // 7))
        phase_binaries[i, phase_idx] = 1.0
    
    # Create dummy rewards
    rewards = {
        'phase_progression': np.random.normal(0.1, 0.3, num_frames).astype(np.float32),
        'phase_completion': np.random.exponential(0.2, num_frames).astype(np.float32),
        'phase_initiation': np.random.normal(0.05, 0.1, num_frames).astype(np.float32),
        'safety': np.random.normal(0.8, 0.2, num_frames).astype(np.float32),
        'efficiency': np.random.normal(0.5, 0.3, num_frames).astype(np.float32),
        'action_probability': np.random.uniform(0.1, 0.9, num_frames).astype(np.float32),
        'risk_penalty': np.random.normal(-0.1, 0.2, num_frames).astype(np.float32)
    }
    
    return {
        'video_id': video_id,
        'frame_embeddings': frame_embeddings,
        'actions_binaries': actions_binaries,
        'phase_binaries': phase_binaries,
        'rewards': rewards,
        'next_rewards': rewards,  # For compatibility
        'metadata': {
            'num_frames': num_frames,
            'video_length': num_frames,
            'data_source': 'dummy'
        }
    }


def validate_video_data(video_data: Dict, logger) -> bool:
    """Validate that video data has the correct format."""
    
    required_keys = ['video_id', 'frame_embeddings', 'actions_binaries', 'phase_binaries', 'rewards']
    
    for key in required_keys:
        if key not in video_data:
            logger.error(f"❌ Missing required key: {key}")
            return False
    
    # Check shapes
    num_frames = len(video_data['frame_embeddings'])
    
    if video_data['frame_embeddings'].shape != (num_frames, 1024):
        logger.error(f"❌ Incorrect frame_embeddings shape: {video_data['frame_embeddings'].shape}")
        return False
    
    if video_data['actions_binaries'].shape != (num_frames, 100):
        logger.error(f"❌ Incorrect actions_binaries shape: {video_data['actions_binaries'].shape}")
        return False
    
    if video_data['phase_binaries'].shape[0] != num_frames:
        logger.error(f"❌ Incorrect phase_binaries frames: {video_data['phase_binaries'].shape[0]}")
        return False
    
    return True


if __name__ == "__main__":
    print("📂 CHOLECT50 DATASET LOADER")
    print("=" * 50)
    
    # Test loading
    dummy_config = {
        'data': {
            'context_length': 20,
            'train_shift': 1,
            'padding_value': 0.0,
            'max_horizon': 15,
            'paths': {
                'data_dir': '/home/maxboels/datasets/CholecT50',
                'video_global_outcome_file': 'embeddings_f0_swin_bas_129_with_enhanced_global_metrics.csv'
            },
            'frame_risk_agg': 'max'
        },
        'experiment': {
            'train': {'max_videos': 2},
            'test': {'max_videos': 1}
        }
    }
    
    # This would need a logger in real usage
    class DummyLogger:
        def info(self, msg): print(f"INFO: {msg}")
        def warning(self, msg): print(f"WARNING: {msg}")
        def error(self, msg): print(f"ERROR: {msg}")
    
    logger = DummyLogger()
    
    train_data, test_data = load_cholect50_data(dummy_config, logger)
    
    print(f"✅ Loaded {len(train_data)} training and {len(test_data)} test videos")
    
    if train_data:
        video = train_data[0]
        print(f"📊 Sample video: {video['video_id']}")
        print(f"   Frames: {video['frame_embeddings'].shape}")
        print(f"   Actions: {video['actions_binaries'].shape}")
        print(f"   Phases: {video['phase_binaries'].shape}")
        print(f"   Rewards: {list(video['rewards'].keys())}")