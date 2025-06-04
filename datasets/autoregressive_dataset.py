#!/usr/bin/env python3
"""
Autoregressive Dataset for Method 1 (IL)
Focus: Frame sequences for causal generation → action prediction
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Any, Tuple
from tqdm import tqdm


class AutoregressiveDataset(Dataset):
    """
    Dataset for Autoregressive Imitation Learning (Method 1).
    
    Creates sequences for causal frame generation:
    - Input: Frame sequence [t-n, ..., t]
    - Target: Next frame sequence [t-n+1, ..., t+1] 
    - Target Actions: Actions at each timestep
    
    Key: NO action conditioning during training (pure autoregressive)
    """
    
    def __init__(self, config: Dict, video_data: List[Dict]):
        """
        Initialize autoregressive dataset.
        
        Args:
            config: Data configuration
            video_data: List of video dictionaries from load_cholect50_data()
        """
        
        print("🎓 Initializing Autoregressive Dataset for IL...")
        
        self.config = config
        self.video_data = video_data
        
        # Dataset parameters
        self.context_length = config.get('context_length', 20)
        self.max_horizon = config.get('max_horizon', 15) 
        self.padding_value = config.get('padding_value', 0.0)
        
        # Build samples for autoregressive training
        self.samples = []
        self._build_autoregressive_samples()
        
        print(f"✅ Autoregressive Dataset created")
        print(f"   Total samples: {len(self.samples)}")
        print(f"   Context length: {self.context_length}")
        print(f"   Max horizon: {self.max_horizon}")
        print(f"   Sample type: Frame sequences for causal generation")
    
    def _build_autoregressive_samples(self):
        """Build samples for autoregressive frame generation."""
        
        print("🔧 Building autoregressive samples...")
        
        for video in tqdm(self.video_data, desc="Processing videos"):
            video_id = video['video_id']
            frame_embeddings = video['frame_embeddings']
            action_binaries = video['actions_binaries']
            phase_binaries = video['phase_binaries']
            
            num_frames = len(frame_embeddings)
            embedding_dim = frame_embeddings.shape[1]
            
            # Create sequences starting from frame 0 (use padding for early frames)
            for target_idx in range(num_frames):
                
                # Calculate input sequence indices [target_idx - context_length + 1, ..., target_idx]
                input_start = target_idx - self.context_length + 1
                input_end = target_idx + 1
                
                # Calculate target sequence indices [target_idx - context_length + 2, ..., target_idx + 1]
                target_start = target_idx - self.context_length + 2
                target_end = target_idx + 2
                
                # Skip if we don't have a next frame for target
                if target_end > num_frames:
                    continue
                
                # Build input frame sequence with padding
                input_frames = []
                for i in range(input_start, input_end):
                    if i < 0:
                        # Padding for positions before video start
                        input_frames.append(np.full(embedding_dim, self.padding_value, dtype=np.float32))
                    else:
                        input_frames.append(frame_embeddings[i])
                
                # Build target next frame sequence with padding
                target_next_frames = []
                target_actions = []
                target_phases = []
                
                for i in range(target_start, target_end):
                    if i < 0:
                        # Padding for positions before video start
                        target_next_frames.append(np.full(embedding_dim, self.padding_value, dtype=np.float32))
                        target_actions.append(np.zeros(action_binaries.shape[1], dtype=np.float32))
                        target_phases.append(np.zeros(phase_binaries.shape[1], dtype=np.float32))
                    elif i >= num_frames:
                        # This shouldn't happen with our bounds checking above
                        continue
                    else:
                        target_next_frames.append(frame_embeddings[i])
                        target_actions.append(action_binaries[i])  
                        target_phases.append(phase_binaries[i])
                
                # Ensure all sequences have the correct length
                if (len(input_frames) == self.context_length and 
                    len(target_next_frames) == self.context_length and
                    len(target_actions) == self.context_length and
                    len(target_phases) == self.context_length):
                    
                    # Convert phase binaries to class indices
                    phase_indices = np.argmax(target_phases, axis=1)
                    
                    self.samples.append({
                        'video_id': video_id,
                        'target_idx': target_idx,
                        'input_frames': np.array(input_frames, dtype=np.float32),
                        'target_next_frames': np.array(target_next_frames, dtype=np.float32),
                        'target_actions': np.array(target_actions, dtype=np.float32),
                        'target_phases': phase_indices.astype(np.int64),
                        'sequence_info': {
                            'video_length': num_frames,
                            'target_frame': target_idx,
                            'padding_frames': max(0, -input_start)
                        }
                    })
    
    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample for autoregressive training.
        
        Returns:
            Dictionary with input frames and targets
        """
        
        sample = self.samples[idx]
        
        # Convert to tensors
        input_frames = torch.tensor(sample['input_frames'], dtype=torch.float32)
        target_next_frames = torch.tensor(sample['target_next_frames'], dtype=torch.float32)
        target_actions = torch.tensor(sample['target_actions'], dtype=torch.float32)
        target_phases = torch.tensor(sample['target_phases'], dtype=torch.long)
        
        return {
            'video_id': sample['video_id'],
            'input_frames': input_frames,  # [context_length, embedding_dim]
            'target_next_frames': target_next_frames,  # [context_length, embedding_dim]  
            'target_actions': target_actions,  # [context_length, num_actions]
            'target_phases': target_phases,  # [context_length]
            'sequence_info': sample['sequence_info']
        }
    
    def get_generation_context(self, video_idx: int, start_frame: int, 
                              context_length: Optional[int] = None) -> torch.Tensor:
        """
        Get context frames for autoregressive generation.
        
        Args:
            video_idx: Index of video in video_data
            start_frame: Starting frame index
            context_length: Length of context (defaults to self.context_length)
            
        Returns:
            Context frames tensor [context_length, embedding_dim]
        """
        
        if context_length is None:
            context_length = self.context_length
        
        video = self.video_data[video_idx]
        frame_embeddings = video['frame_embeddings']
        
        # Extract context frames
        end_frame = start_frame + context_length
        if end_frame > len(frame_embeddings):
            # Pad if necessary
            context_frames = frame_embeddings[start_frame:]
            padding_needed = context_length - len(context_frames)
            
            if padding_needed > 0:
                embedding_dim = frame_embeddings.shape[1]
                padding = np.full((padding_needed, embedding_dim), self.padding_value, dtype=np.float32)
                context_frames = np.concatenate([context_frames, padding], axis=0)
        else:
            context_frames = frame_embeddings[start_frame:end_frame]
        
        return torch.tensor(context_frames, dtype=torch.float32)
    
    def get_video_stats(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        
        total_frames = sum(len(video['frame_embeddings']) for video in self.video_data)
        video_lengths = [len(video['frame_embeddings']) for video in self.video_data]
        
        action_stats = []
        for video in self.video_data:
            actions = video['actions_binaries']
            action_density = np.mean(np.sum(actions, axis=1))
            action_stats.append(action_density)
        
        return {
            'num_videos': len(self.video_data),
            'total_frames': total_frames,
            'avg_video_length': np.mean(video_lengths),
            'min_video_length': np.min(video_lengths),
            'max_video_length': np.max(video_lengths),
            'avg_action_density': np.mean(action_stats),
            'total_samples': len(self.samples),
            'context_length': self.context_length,
            'sequence_overlap': True
        }


def create_autoregressive_dataloaders(config: Dict, 
                                    train_data: List[Dict],
                                    test_data: List[Dict],
                                    batch_size: int = 32,
                                    num_workers: int = 4) -> Tuple[DataLoader, Dict[str, DataLoader]]:
    """
    Create dataloaders for autoregressive IL training.
    
    Args:
        config: Data configuration
        train_data: Training video data
        test_data: Test video data  
        batch_size: Batch size for training
        num_workers: Number of worker processes
        
    Returns:
        train_loader: Training dataloader
        test_loaders: Dict of test dataloaders (one per video)
    """
    
    print("🔧 Creating autoregressive dataloaders...")
    
    # Create training dataset and loader
    train_dataset = AutoregressiveDataset(config, train_data)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    # Create test dataloaders (one per video for detailed evaluation)
    test_loaders = {}
    for video in test_data:
        video_dataset = AutoregressiveDataset(config, [video])
        video_loader = DataLoader(
            video_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        test_loaders[video['video_id']] = video_loader
    
    print(f"✅ Created autoregressive dataloaders")
    print(f"   Training samples: {len(train_dataset)}")
    print(f"   Training batches: {len(train_loader)}")
    print(f"   Test videos: {len(test_loaders)}")
    
    # Print dataset statistics
    stats = train_dataset.get_video_stats()
    print(f"📊 Dataset Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.2f}")
        else:
            print(f"   {key}: {value}")
    
    return train_loader, test_loaders


class AutoregressiveEvaluationDataset(Dataset):
    """
    Special dataset for autoregressive evaluation with longer contexts.
    """
    
    def __init__(self, config: Dict, video_data: List[Dict], 
                 evaluation_horizons: List[int] = [1, 3, 5, 10, 15]):
        """
        Initialize evaluation dataset.
        
        Args:
            config: Data configuration
            video_data: Video data for evaluation
            evaluation_horizons: Different horizons to evaluate
        """
        
        self.config = config
        self.video_data = video_data
        self.evaluation_horizons = evaluation_horizons
        self.context_length = config.get('context_length', 20)
        
        # Build evaluation samples
        self.samples = []
        self._build_evaluation_samples()
        
        print(f"🔍 Autoregressive Evaluation Dataset created")
        print(f"   Evaluation samples: {len(self.samples)}")
        print(f"   Horizons: {evaluation_horizons}")
    
    def _build_evaluation_samples(self):
        """Build samples for evaluation at different horizons."""
        
        for video in self.video_data:
            video_id = video['video_id']
            frame_embeddings = video['frame_embeddings']
            action_binaries = video['actions_binaries']
            
            num_frames = len(frame_embeddings)
            max_horizon = max(self.evaluation_horizons)
            
            # Create evaluation samples with sufficient future frames
            for start_idx in range(0, num_frames - self.context_length - max_horizon, 5):
                # Skip some frames to avoid too many samples
                
                context_end = start_idx + self.context_length
                
                # Get context frames
                context_frames = frame_embeddings[start_idx:context_end]
                
                # Get ground truth for all horizons
                horizon_targets = {}
                for horizon in self.evaluation_horizons:
                    if context_end + horizon <= num_frames:
                        horizon_targets[horizon] = {
                            'actions': action_binaries[context_end:context_end + horizon],
                            'frames': frame_embeddings[context_end:context_end + horizon]
                        }
                
                if horizon_targets:  # Only add if we have at least one valid horizon
                    self.samples.append({
                        'video_id': video_id,
                        'start_idx': start_idx,
                        'context_frames': np.array(context_frames, dtype=np.float32),
                        'horizon_targets': horizon_targets
                    })
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        
        # Convert context to tensor
        context_frames = torch.tensor(sample['context_frames'], dtype=torch.float32)
        
        # Convert horizon targets to tensors
        horizon_targets = {}
        for horizon, targets in sample['horizon_targets'].items():
            horizon_targets[horizon] = {
                'actions': torch.tensor(targets['actions'], dtype=torch.float32),
                'frames': torch.tensor(targets['frames'], dtype=torch.float32)
            }
        
        return {
            'video_id': sample['video_id'],
            'start_idx': sample['start_idx'],
            'context_frames': context_frames,
            'horizon_targets': horizon_targets
        }


# Testing and example usage
if __name__ == "__main__":
    print("🎓 AUTOREGRESSIVE DATASET FOR IL")
    print("=" * 60)
    
    # Mock configuration
    config = {
        'context_length': 10,
        'max_horizon': 5,
        'padding_value': 0.0
    }
    
    # Mock video data
    video_data = []
    for i in range(3):
        video_data.append({
            'video_id': f'video_{i}',
            'frame_embeddings': np.random.randn(50, 1024).astype(np.float32),
            'actions_binaries': np.random.randint(0, 2, (50, 100)).astype(np.float32),
            'phase_binaries': np.random.randint(0, 2, (50, 7)).astype(np.float32)
        })
    
    print("🧪 Testing dataset creation...")
    
    # Create dataset
    dataset = AutoregressiveDataset(config, video_data)
    
    print(f"✅ Dataset created with {len(dataset)} samples")
    
    # Test sample retrieval
    sample = dataset[0]
    print(f"✅ Sample shapes:")
    print(f"   Input frames: {sample['input_frames'].shape}")
    print(f"   Target next frames: {sample['target_next_frames'].shape}")
    print(f"   Target actions: {sample['target_actions'].shape}")
    print(f"   Target phases: {sample['target_phases'].shape}")
    
    # Test dataloader
    print(f"\n🧪 Testing dataloader...")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    batch = next(iter(dataloader))
    print(f"✅ Batch shapes:")
    print(f"   Input frames: {batch['input_frames'].shape}")
    print(f"   Target next frames: {batch['target_next_frames'].shape}")
    print(f"   Target actions: {batch['target_actions'].shape}")
    print(f"   Target phases: {batch['target_phases'].shape}")
    
    # Test generation context
    print(f"\n🧪 Testing generation context...")
    context = dataset.get_generation_context(0, 5, context_length=8)
    print(f"✅ Generation context shape: {context.shape}")
    
    # Get statistics
    stats = dataset.get_video_stats()
    print(f"\n📊 Dataset Statistics:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print(f"\n🎯 KEY FEATURES:")
    print(f"✅ Autoregressive frame sequences (no action conditioning)")
    print(f"✅ Input: frame[t-n:t] → Target: frame[t-n+1:t+1]")
    print(f"✅ Action targets aligned with generated frames")
    print(f"✅ Overlapping sequences for better coverage")
    print(f"✅ Ready for Method 1 training")
