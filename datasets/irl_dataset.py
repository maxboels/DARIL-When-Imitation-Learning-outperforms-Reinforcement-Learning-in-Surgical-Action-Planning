#!/usr/bin/env python3
"""
IRL Dataset for Next Action Prediction
Matches the temporal structure of AutoregressiveDataset for consistent training
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Any, Optional, Tuple
from tqdm import tqdm

class IRLNextActionDataset(Dataset):
    """
    IRL Dataset that matches AutoregressiveDataset structure
    Predicts NEXT actions (t+1) from current states (t)
    
    Key alignment with IL approach:
    - Same temporal structure as AutoregressiveDataset
    - Same context_length handling
    - Same target preparation for next action prediction
    - Compatible with sophisticated negative generation
    """
    
    def __init__(self, video_data: List[Dict], config: Dict):
        """
        Args:
            video_data: List of video dictionaries from load_cholect50_data
            config: Data configuration (same as AutoregressiveDataset)
        """
        self.samples = []
        
        # Use same parameters as AutoregressiveDataset
        context_length = config.get('context_length', 10)
        padding_value = config.get('padding_value', 0.0)
        
        print(f"🎯 Creating IRL Dataset with context_length={context_length} (matching IL approach)")
        
        for video in video_data:
            video_id = video['video_id']
            embeddings = video['frame_embeddings']
            actions = video['actions_binaries']  # Use correct key
            phases = video.get('phase_binaries', [])
            
            num_frames = len(embeddings)
            embedding_dim = embeddings.shape[1]
            num_actions = actions.shape[1]
            
            # Same iteration strategy as AutoregressiveDataset
            for i in range(num_frames - 1):  # -1 because we need t+1 for next prediction
                
                # Build current context sequence (for prediction at time t)
                current_context_frames = []
                current_context_actions = []
                current_context_phases = []
                
                # Get sequence indices (same as AutoregressiveDataset)
                sequence_indices = list(range(max(0, i - context_length + 1), i + 1))
                
                # Build current context sequences
                for idx, j in enumerate(sequence_indices):
                    if j < 0:
                        # Padding (same as AutoregressiveDataset)
                        current_context_frames.append([padding_value] * embedding_dim)
                        current_context_actions.append([0] * num_actions)
                        current_context_phases.append(0)
                    else:
                        # Current frame, action, phase at time j
                        current_context_frames.append(embeddings[j])
                        current_context_actions.append(actions[j])
                        
                        if len(phases) > j:
                            current_context_phases.append(np.argmax(phases[j]))
                        else:
                            current_context_phases.append(0)
                
                # Target next action (what we want to predict at t+1)
                if i + 1 < num_frames:
                    target_next_action = actions[i + 1]
                    target_next_phase = np.argmax(phases[i + 1]) if len(phases) > i + 1 else 0
                else:
                    # Edge case: use current action as fallback
                    target_next_action = actions[i]
                    target_next_phase = np.argmax(phases[i]) if len(phases) > i else 0
                
                # Only include samples where target has actions (like AutoregressiveDataset logic)
                if np.sum(target_next_action) > 0:
                    
                    # Ensure all sequences have the same length (like AutoregressiveDataset)
                    while len(current_context_frames) < context_length:
                        current_context_frames.insert(0, [padding_value] * embedding_dim)
                        current_context_actions.insert(0, [0] * num_actions)
                        current_context_phases.insert(0, 0)
                    
                    self.samples.append({
                        'video_id': video_id,
                        'frame_idx': i,
                        'target_frame_idx': i + 1,
                        
                        # Current context (input for prediction)
                        'current_context_frames': current_context_frames,  # [context_length, embedding_dim]
                        'current_context_actions': current_context_actions,  # [context_length, num_actions]
                        'current_context_phases': current_context_phases,   # [context_length]
                        
                        # Current state (for reward computation)
                        'current_state': embeddings[i],                    # [embedding_dim]
                        'current_action': actions[i],                      # [num_actions]
                        'current_phase': phases[i] if len(phases) > i else np.zeros(7),  # [7]
                        
                        # Next targets (what we want to predict)
                        'target_next_action': target_next_action,          # [num_actions]
                        'target_next_phase': target_next_phase,            # scalar
                    })
        
        print(f"✅ IRL Next Action Dataset created: {len(self.samples)} samples")
        print(f"   Temporal structure: current_context → predict next_action")
        print(f"   Compatible with AutoregressiveDataset approach")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        return {
            'video_id': sample['video_id'],
            'frame_idx': sample['frame_idx'],
            'target_frame_idx': sample['target_frame_idx'],
            
            # Current context (for IL model input)
            'current_context_frames': torch.tensor(np.array(sample['current_context_frames']), dtype=torch.float32),
            'current_context_actions': torch.tensor(np.array(sample['current_context_actions']), dtype=torch.float32),
            'current_context_phases': torch.tensor(np.array(sample['current_context_phases']), dtype=torch.long),
            
            # Current state (for IRL reward computation)
            'current_state': torch.tensor(sample['current_state'], dtype=torch.float32),
            'current_action': torch.tensor(sample['current_action'], dtype=torch.float32),
            'current_phase': torch.tensor(sample['current_phase'], dtype=torch.float32),
            
            # Target next action (what we want to predict)
            'target_next_action': torch.tensor(sample['target_next_action'], dtype=torch.float32),
            'target_next_phase': torch.tensor(sample['target_next_phase'], dtype=torch.long),
        }


def create_irl_next_action_dataloaders(train_data: List[Dict], 
                                       test_data: List[Dict],
                                       config: Dict,
                                       batch_size: int = 32,
                                       num_workers: int = 4) -> Tuple[DataLoader, Dict[str, DataLoader]]:
    """
    Create DataLoaders for IRL Next Action Prediction
    Matches create_autoregressive_dataloaders interface
    
    Args:
        train_data: Training video data
        test_data: Test video data
        config: Data configuration (same as AutoregressiveDataset)
        batch_size: Batch size for training
        num_workers: Number of worker processes
        
    Returns:
        Tuple of (train_loader, test_loaders_dict)
    """
    
    print("🎯 Creating IRL DataLoaders for Next Action Prediction")
    
    # Training dataset
    train_dataset = IRLNextActionDataset(train_data, config)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Important for IRL training
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # Ensure consistent batch sizes
    )
    
    # Test datasets (one per video for detailed analysis)
    test_loaders = {}
    for test_video in test_data:
        video_id = test_video['video_id']
        video_dataset = IRLNextActionDataset([test_video], config)
        
        if len(video_dataset) > 0:
            test_loaders[video_id] = DataLoader(
                video_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            )
    
    print(f"✅ IRL Next Action DataLoaders created:")
    print(f"   Training samples: {len(train_dataset)}")
    print(f"   Training batches: {len(train_loader)}")
    print(f"   Test videos: {len(test_loaders)}")
    print(f"   Task: Predict next_action(t+1) from current_context(t)")
    
    return train_loader, test_loaders