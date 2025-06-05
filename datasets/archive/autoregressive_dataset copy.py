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
    """
    
    def __init__(self, config: Dict, data: List[Dict]):
        """
        Initialize autoregressive dataset.
        
        Args:
            config: Data configuration
            data: List of video dictionaries
        """
        self.samples = []
        
        context_length = config.get('context_length', 10)
        padding_value = config.get('padding_value', 0.0)
        
        for video in data:
            video_id = video['video_id']
            embeddings = video['frame_embeddings']
            actions = video['actions_binaries']
            phases = video.get('phase_binaries', [])
            
            num_frames = len(embeddings)
            
            for i in range(num_frames - 1):
                # Input: sequence of frames
                input_frames = []
                target_next_frames = []
                target_actions = []
                target_phases = []
                
                # Build sequences
                for j in range(max(0, i - context_length + 1), i + 1):
                    if j < 0:
                        input_frames.append([padding_value] * embeddings.shape[1])
                        target_next_frames.append([padding_value] * embeddings.shape[1])
                        target_actions.append([0] * actions.shape[1])
                        target_phases.append(0)
                    else:
                        input_frames.append(embeddings[j])
                        if j + 1 < num_frames:
                            target_next_frames.append(embeddings[j + 1])
                            target_actions.append(actions[j + 1])
                            if len(phases) > j + 1:
                                target_phases.append(np.argmax(phases[j + 1]))
                            else:
                                target_phases.append(0)
                        else:
                            target_next_frames.append(embeddings[j])
                            target_actions.append(actions[j])
                            if len(phases) > j:
                                target_phases.append(np.argmax(phases[j]))
                            else:
                                target_phases.append(0)
                
                self.samples.append({
                    'video_id': video_id,
                    'frame_idx': i,
                    'input_frames': input_frames,
                    'target_next_frames': target_next_frames,
                    'target_actions': target_actions,
                    'target_phases': target_phases
                })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        return {
            'input_frames': torch.tensor(np.array(sample['input_frames']), dtype=torch.float32),
            'target_next_frames': torch.tensor(np.array(sample['target_next_frames']), dtype=torch.float32),
            'target_actions': torch.tensor(np.array(sample['target_actions']), dtype=torch.float32),
            'target_phases': torch.tensor(np.array(sample['target_phases']), dtype=torch.long),
            'video_id': sample['video_id'],
            'frame_idx': sample['frame_idx']
        }


def create_autoregressive_dataloaders(config: Dict, 
                                    train_data: List[Dict], 
                                    test_data: List[Dict],
                                    batch_size: int = 32,
                                    num_workers: int = 4) -> Tuple[DataLoader, Dict[str, DataLoader]]:
    """Create dataloaders for autoregressive IL."""
    
    # Training dataset
    train_dataset = AutoregressiveDataset(config, train_data)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    # Test datasets (per video)
    test_loaders = {}
    for video in test_data:
        video_dataset = AutoregressiveDataset(config, [video])
        test_loaders[video['video_id']] = DataLoader(
            video_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    
    return train_loader, test_loaders
