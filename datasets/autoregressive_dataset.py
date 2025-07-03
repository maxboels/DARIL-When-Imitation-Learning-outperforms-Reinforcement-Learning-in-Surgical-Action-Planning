#!/usr/bin/env python3
"""
Autoregressive Dataset for Method 1
Pure causal frame generation → action prediction (no action conditioning)
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Any, Optional, Tuple
import random


class AutoregressiveDataset(Dataset):
    """
    Dataset for Autoregressive IL (Method 1).
    Returns frame sequences for causal generation without action conditioning.
    
    Usage during forward pass:
    - input_frames = target_next_frames[:-1]  # For recognition
    - next_frames = target_next_frames[1:]   # For next frame prediction
    - current_actions = target_actions[:-1]  # For recognition  
    - next_actions = target_actions[1:]      # For next frame prediction
    - current_phases = target_phases[:-1]    # For recognition
    - next_phases = target_phases[1:]        # For next frame prediction
    """
    
    def __init__(self, config: Dict, data: List[Dict]):
        """
        Initialize autoregressive dataset.
        
        Args:
            config: Data configuration
            data: List of video dictionaries
        """
        self.samples = []
        
        context_length = config.get('context_length', 20)
        future_length = config.get('future_length', 10)
        padding_value = config.get('padding_value', 0.0)
        
        for video in data:
            video_id = video['video_id']
            embeddings = video['frame_embeddings']
            actions = video['actions_binaries']  # FIXED: Use correct key name
            phases = video.get('phase_binaries', [])
            
            num_frames = len(embeddings)
            embedding_dim = embeddings.shape[1]
            num_actions = actions.shape[1]  # FIXED: Use actions instead of actions_binaries.shape
            
            for i in range(num_frames - 1):
                # Target sequences that will be split during forward pass
                target_next_frames = []
                target_actions = []
                target_phases = []
                target_future_actions = []
                
                # Get the sequence indices
                sequence_indices = list(range(max(0, i - context_length + 1), i + 1))
                
                # Build sequences frame by frame
                for idx, j in enumerate(sequence_indices):
                    # Pad if index is negative
                    if j < 0:
                        target_next_frames.append([padding_value] * embedding_dim)
                        target_actions.append([0] * num_actions)
                        target_phases.append(0)
                    else:
                        # Target next frames: current frame at time j
                        target_next_frames.append(embeddings[j])
                        
                        # Target actions: current action at time j
                        target_actions.append(actions[j])
                        
                        # Target phases: current phase at time j
                        if len(phases) > j:
                            target_phases.append(np.argmax(phases[j]))
                        else:
                            target_phases.append(0)

                # Add the next frame (t+1) to target_next_frames for next frame prediction
                last_frame_idx = sequence_indices[-1] if sequence_indices else i
                if last_frame_idx + 1 < num_frames:
                    target_next_frames.append(embeddings[last_frame_idx + 1])
                else:
                    target_next_frames.append(embeddings[last_frame_idx])

                # Add the next action (t+1) to target_actions for next frame prediction
                if last_frame_idx + 1 < num_frames:
                    target_actions.append(actions[last_frame_idx + 1])
                else:
                    target_actions.append([0] * num_actions)

                # Add the next phase (t+1) to target_phases for next frame prediction
                if last_frame_idx + 1 < num_frames and len(phases) > last_frame_idx + 1:
                    target_phases.append(np.argmax(phases[last_frame_idx + 1]))
                elif len(phases) > last_frame_idx:
                    target_phases.append(np.argmax(phases[last_frame_idx]))
                else:
                    target_phases.append(0)

                # Future actions (t+1, t+2, t+3, etc.)
                for k in range(1, future_length + 1):  # Start from t+1
                    if last_frame_idx + k < num_frames:
                        target_future_actions.append(actions[last_frame_idx + k])
                    else:
                        target_future_actions.append([0] * num_actions)
                
                # Ensure all sequences have the same length
                # target_next_frames, target_actions and target_phases now have context_length + 1 elements (current + next)
                while len(target_next_frames) < context_length + 1:
                    target_next_frames.insert(-1, [padding_value] * embedding_dim)  # Insert before the last (next) frame
                    target_actions.insert(-1, [0] * num_actions)  # Insert before the last (next) action
                    target_phases.insert(-1, 0)  # Insert before the last (next) phase
                
                self.samples.append({
                    'video_id': video_id,
                    'frame_idx': i,
                    'target_next_frames': target_next_frames,
                    'target_next_actions': target_actions,
                    'target_next_phases': target_phases,
                    'target_future_actions': target_future_actions
                })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        return {
            'target_next_frames': torch.tensor(np.array(sample['target_next_frames']), dtype=torch.float32),
            'target_next_actions': torch.tensor(np.array(sample['target_next_actions']), dtype=torch.float32),
            'target_next_phases': torch.tensor(np.array(sample['target_next_phases']), dtype=torch.long),
            'target_future_actions': torch.tensor(np.array(sample['target_future_actions']), dtype=torch.float32),
            'video_id': sample['video_id'],
            'frame_idx': sample['frame_idx']
        }


from typing import Dict, List, Optional, Tuple
from torch.utils.data import DataLoader

def create_autoregressive_dataloaders(config: Dict, 
                                    train_data: Optional[List[Dict]], 
                                    test_data: List[Dict],
                                    batch_size: int = 32,
                                    num_workers: int = 4) -> Tuple[Optional[DataLoader], Dict[str, DataLoader]]:
    """
    Create dataloaders for autoregressive IL.
    
    Args:
        config: Dataset configuration
        train_data: List of training video dictionaries, or None to skip training
        test_data: List of test video dictionaries
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes
        
    Returns:
        Tuple of (train_loader, test_loaders_dict)
    """
    
    # Validate inputs
    if train_data is not None and not isinstance(train_data, list):
        raise ValueError("train_data must be a list or None")
    if not isinstance(test_data, list):
        raise ValueError("test_data must be a list")
    
    # Training dataset
    train_loader = None
    train_samples = 0
    
    if train_data is not None and len(train_data) > 0:
        try:
            train_dataset = AutoregressiveDataset(config, train_data)
            train_samples = len(train_dataset)
            
            if train_samples > 0:
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=num_workers,
                    pin_memory=True,
                    drop_last=True
                )
            else:
                print("⚠️ Training dataset is empty after processing.")
        except Exception as e:
            print(f"❌ Error creating training dataset: {e}")
            raise
    elif train_data is None:
        print("ℹ️ Training skipped (train_data=None).")
    else:
        print("⚠️ No training data provided.")
    
    # Test datasets (one dataloader per video)
    test_loaders = {}
    failed_videos = []
    
    for test_video in test_data:
        video_id = test_video.get('video_id', 'unknown')
        try:
            video_dataset = AutoregressiveDataset(config, [test_video])
            
            if len(video_dataset) > 0:
                test_loaders[video_id] = DataLoader(
                    video_dataset,
                    batch_size=32,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=True
                )
            else:
                print(f"⚠️ Empty dataset for test video {video_id}")
                
        except Exception as e:
            print(f"❌ Failed to create dataset for test video {video_id}: {e}")
            failed_videos.append(video_id)
    
    # Summary
    print(f"✅ Created autoregressive IL dataloaders:")
    print(f"   Training samples: {train_samples}")
    print(f"   Training batches: {len(train_loader) if train_loader else 0}")
    print(f"   Test videos: {len(test_loaders)}")
    
    if failed_videos:
        print(f"   Failed videos: {failed_videos}")
    
    for video_id, loader in test_loaders.items():
        print(f"   Test video {video_id}: {len(loader.dataset)} samples, {len(loader)} batches")

    return train_loader, test_loaders