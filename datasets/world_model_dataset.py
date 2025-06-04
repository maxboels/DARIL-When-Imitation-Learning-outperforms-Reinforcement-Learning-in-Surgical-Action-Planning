#!/usr/bin/env python3
"""
World Model Dataset for Method 2 (RL)
Focus: State-Action-NextState-Reward tuples for conditional prediction
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Any, Tuple, Optional
from tqdm import tqdm


class WorldModelDataset(Dataset):
    """
    Dataset for Conditional World Model training (Method 2).
    
    Creates action-conditioned samples:
    - Input: Current state + Action
    - Target: Next state + Rewards
    
    Key: Action conditioning for forward simulation training
    """
    
    def __init__(self, config: Dict, video_data: List[Dict]):
        """
        Initialize world model dataset.
        
        Args:
            config: Data configuration  
            video_data: List of video dictionaries from load_cholect50_data()
        """
        
        print("🌍 Initializing World Model Dataset for RL...")
        
        self.config = config
        self.video_data = video_data
        
        # Dataset parameters
        self.context_length = config.get('context_length', 20)
        self.padding_value = config.get('padding_value', 0.0)
        
        # Build samples for action-conditioned training
        self.samples = []
        self._build_world_model_samples()
        
        print(f"✅ World Model Dataset created")
        print(f"   Total samples: {len(self.samples)}")
        print(f"   Context length: {self.context_length}")
        print(f"   Sample type: State-Action-NextState-Reward tuples")
    
    def _build_world_model_samples(self):
        """Build samples for action-conditioned world model training."""
        
        print("🔧 Building world model samples...")
        
        for video in tqdm(self.video_data, desc="Processing videos"):
            video_id = video['video_id']
            frame_embeddings = video['frame_embeddings']
            action_binaries = video['actions_binaries']
            phase_binaries = video['phase_binaries']
            
            # Extract reward signals
            rewards = video.get('next_rewards', {})
            
            num_frames = len(frame_embeddings)
            
            # Create state-action-next_state-reward tuples
            for start_idx in range(num_frames - self.context_length):
                
                end_idx = start_idx + self.context_length
                if end_idx >= num_frames:
                    continue
                
                # Current states sequence [start_idx, ..., start_idx + context_length - 1]
                current_states = frame_embeddings[start_idx:end_idx]
                
                # Action sequence (actions taken at each state)
                action_sequence = action_binaries[start_idx:end_idx]
                
                # Next states sequence [start_idx + 1, ..., start_idx + context_length]
                if end_idx < num_frames:
                    next_states = frame_embeddings[start_idx + 1:end_idx + 1]
                else:
                    continue  # Skip if we don't have enough next states
                
                # Phase sequences
                current_phases = phase_binaries[start_idx:end_idx]
                next_phases = phase_binaries[start_idx + 1:end_idx + 1]
                
                # Extract reward sequences for each reward type
                reward_sequences = {}
                for reward_type, reward_values in rewards.items():
                    if len(reward_values) > end_idx:
                        # Get rewards for next states
                        reward_seq = reward_values[start_idx + 1:end_idx + 1]
                        reward_sequences[reward_type.replace('_r_', '')] = reward_seq
                
                # Ensure all sequences have the correct length
                if (len(current_states) == self.context_length and 
                    len(next_states) == self.context_length and
                    len(action_sequence) == self.context_length):
                    
                    # Convert phase binaries to class indices
                    current_phase_indices = np.argmax(current_phases, axis=1)
                    next_phase_indices = np.argmax(next_phases, axis=1)
                    
                    self.samples.append({
                        'video_id': video_id,
                        'start_idx': start_idx,
                        'current_states': np.array(current_states, dtype=np.float32),
                        'actions': np.array(action_sequence, dtype=np.float32),
                        'next_states': np.array(next_states, dtype=np.float32),
                        'current_phases': current_phase_indices.astype(np.int64),
                        'next_phases': next_phase_indices.astype(np.int64),
                        'rewards': {k: np.array(v, dtype=np.float32) for k, v in reward_sequences.items()},
                        'sequence_info': {
                            'video_length': num_frames,
                            'context_start': start_idx,
                            'context_end': end_idx
                        }
                    })
    
    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample for world model training.
        
        Returns:
            Dictionary with current states, actions, targets
        """
        
        sample = self.samples[idx]
        
        # Convert to tensors
        current_states = torch.tensor(sample['current_states'], dtype=torch.float32)
        actions = torch.tensor(sample['actions'], dtype=torch.float32)
        next_states = torch.tensor(sample['next_states'], dtype=torch.float32)
        current_phases = torch.tensor(sample['current_phases'], dtype=torch.long)
        next_phases = torch.tensor(sample['next_phases'], dtype=torch.long)
        
        # Convert rewards to tensors
        rewards = {}
        for reward_type, reward_values in sample['rewards'].items():
            rewards[reward_type] = torch.tensor(reward_values, dtype=torch.float32)
            # Ensure reward has shape [seq_len, 1]
            if rewards[reward_type].dim() == 1:
                rewards[reward_type] = rewards[reward_type].unsqueeze(-1)
        
        return {
            'video_id': sample['video_id'],
            'current_states': current_states,  # [context_length, embedding_dim]
            'actions': actions,  # [context_length, num_actions]
            'next_states': next_states,  # [context_length, embedding_dim]
            'current_phases': current_phases,  # [context_length]
            'next_phases': next_phases,  # [context_length]
            'rewards': rewards,  # Dict of [context_length, 1] tensors
            'sequence_info': sample['sequence_info']
        }
    
    def get_single_step_sample(self, video_idx: int, frame_idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single step sample for simulation testing.
        
        Args:
            video_idx: Index of video in video_data
            frame_idx: Frame index in video
            
        Returns:
            Single step state-action-next_state-reward sample
        """
        
        video = self.video_data[video_idx]
        
        if frame_idx >= len(video['frame_embeddings']) - 1:
            raise ValueError(f"Frame index {frame_idx} too large for video of length {len(video['frame_embeddings'])}")
        
        current_state = torch.tensor(video['frame_embeddings'][frame_idx], dtype=torch.float32)
        action = torch.tensor(video['actions_binaries'][frame_idx], dtype=torch.float32)
        next_state = torch.tensor(video['frame_embeddings'][frame_idx + 1], dtype=torch.float32)
        
        # Extract rewards if available
        rewards = {}
        if 'next_rewards' in video:
            for reward_type, reward_values in video['next_rewards'].items():
                if frame_idx + 1 < len(reward_values):
                    reward_key = reward_type.replace('_r_', '')
                    rewards[reward_key] = torch.tensor([reward_values[frame_idx + 1]], dtype=torch.float32)
        
        return {
            'current_state': current_state,  # [embedding_dim]
            'action': action,  # [num_actions]
            'next_state': next_state,  # [embedding_dim]
            'rewards': rewards  # Dict of scalar tensors
        }
    
    def get_dataset_stats(self) -> Dict[str, Any]:
        """Get comprehensive dataset statistics."""
        
        total_frames = sum(len(video['frame_embeddings']) for video in self.video_data)
        video_lengths = [len(video['frame_embeddings']) for video in self.video_data]
        
        # Action statistics
        action_stats = []
        for video in self.video_data:
            actions = video['actions_binaries']
            action_density = np.mean(np.sum(actions, axis=1))
            action_stats.append(action_density)
        
        # Reward statistics
        reward_stats = {}
        for video in self.video_data:
            if 'next_rewards' in video:
                for reward_type, reward_values in video['next_rewards'].items():
                    reward_key = reward_type.replace('_r_', '')
                    if reward_key not in reward_stats:
                        reward_stats[reward_key] = []
                    reward_stats[reward_key].extend(reward_values)
        
        reward_summary = {}
        for reward_type, values in reward_stats.items():
            reward_summary[reward_type] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        
        return {
            'num_videos': len(self.video_data),
            'total_frames': total_frames,
            'avg_video_length': np.mean(video_lengths),
            'min_video_length': np.min(video_lengths),
            'max_video_length': np.max(video_lengths),
            'avg_action_density': np.mean(action_stats),
            'total_samples': len(self.samples),
            'context_length': self.context_length,
            'reward_types': list(reward_summary.keys()),
            'reward_statistics': reward_summary
        }


class WorldModelSimulationDataset(Dataset):
    """
    Special dataset for world model simulation and RL training.
    Provides state-action pairs for forward simulation.
    """
    
    def __init__(self, config: Dict, video_data: List[Dict], 
                 simulation_length: int = 50):
        """
        Initialize simulation dataset.
        
        Args:
            config: Data configuration
            video_data: Video data
            simulation_length: Length of simulation episodes
        """
        
        self.config = config
        self.video_data = video_data
        self.simulation_length = simulation_length
        
        # Create starting points for simulation
        self.simulation_starts = []
        self._build_simulation_starts()
        
        print(f"🎮 World Model Simulation Dataset created")
        print(f"   Simulation starts: {len(self.simulation_starts)}")
        print(f"   Simulation length: {simulation_length}")
    
    def _build_simulation_starts(self):
        """Build starting points for simulation episodes."""
        
        for video_idx, video in enumerate(self.video_data):
            frame_embeddings = video['frame_embeddings']
            num_frames = len(frame_embeddings)
            
            # Create multiple starting points per video
            for start_frame in range(0, num_frames - self.simulation_length, 10):
                self.simulation_starts.append({
                    'video_idx': video_idx,
                    'video_id': video['video_id'],
                    'start_frame': start_frame,
                    'max_frames': num_frames
                })
    
    def __len__(self) -> int:
        return len(self.simulation_starts)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        start_info = self.simulation_starts[idx]
        
        video = self.video_data[start_info['video_idx']]
        start_frame = start_info['start_frame']
        
        # Get initial state
        initial_state = torch.tensor(
            video['frame_embeddings'][start_frame], 
            dtype=torch.float32
        )
        
        # Get ground truth sequence for comparison (if available)
        end_frame = min(start_frame + self.simulation_length, len(video['frame_embeddings']))
        
        ground_truth_states = torch.tensor(
            video['frame_embeddings'][start_frame:end_frame],
            dtype=torch.float32
        )
        
        ground_truth_actions = torch.tensor(
            video['actions_binaries'][start_frame:end_frame],
            dtype=torch.float32
        )
        
        return {
            'video_id': start_info['video_id'],
            'video_idx': start_info['video_idx'],
            'start_frame': start_frame,
            'initial_state': initial_state,
            'ground_truth_states': ground_truth_states,
            'ground_truth_actions': ground_truth_actions,
            'simulation_length': len(ground_truth_states)
        }


def create_world_model_dataloaders(config: Dict,
                                 train_data: List[Dict],
                                 test_data: List[Dict],
                                 batch_size: int = 32,
                                 num_workers: int = 4) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create dataloaders for world model training and evaluation.
    
    Args:
        config: Data configuration
        train_data: Training video data
        test_data: Test video data
        batch_size: Batch size
        num_workers: Number of worker processes
        
    Returns:
        train_loader: Training dataloader
        test_loader: Test dataloader  
        simulation_loader: Simulation dataloader
    """
    
    print("🔧 Creating world model dataloaders...")
    
    # Training dataset and loader
    train_dataset = WorldModelDataset(config, train_data)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    # Test dataset and loader
    test_dataset = WorldModelDataset(config, test_data)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Simulation dataset for RL training
    simulation_dataset = WorldModelSimulationDataset(config, train_data)
    simulation_loader = DataLoader(
        simulation_dataset,
        batch_size=1,  # One simulation at a time
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"✅ Created world model dataloaders")
    print(f"   Training samples: {len(train_dataset)}")
    print(f"   Training batches: {len(train_loader)}")
    print(f"   Test samples: {len(test_dataset)}")
    print(f"   Simulation starts: {len(simulation_dataset)}")
    
    # Print dataset statistics
    stats = train_dataset.get_dataset_stats()
    print(f"📊 Dataset Statistics:")
    for key, value in stats.items():
        if key == 'reward_statistics':
            print(f"   {key}:")
            for reward_type, reward_stats in value.items():
                print(f"     {reward_type}: mean={reward_stats['mean']:.3f}, std={reward_stats['std']:.3f}")
        elif isinstance(value, float):
            print(f"   {key}: {value:.2f}")
        else:
            print(f"   {key}: {value}")
    
    return train_loader, test_loader, simulation_loader


# Testing and example usage
if __name__ == "__main__":
    print("🌍 WORLD MODEL DATASET FOR RL")
    print("=" * 60)
    
    # Mock configuration
    config = {
        'context_length': 8,
        'padding_value': 0.0
    }
    
    # Mock video data with rewards
    video_data = []
    for i in range(2):
        num_frames = 30
        video_data.append({
            'video_id': f'video_{i}',
            'frame_embeddings': np.random.randn(num_frames, 1024).astype(np.float32),
            'actions_binaries': np.random.randint(0, 2, (num_frames, 100)).astype(np.float32),
            'phase_binaries': np.random.randint(0, 2, (num_frames, 7)).astype(np.float32),
            'next_rewards': {
                '_r_phase_progression': np.random.randn(num_frames).astype(np.float32),
                '_r_safety': np.random.randn(num_frames).astype(np.float32),
                '_r_efficiency': np.random.randn(num_frames).astype(np.float32)
            }
        })
    
    print("🧪 Testing world model dataset creation...")
    
    # Create dataset
    dataset = WorldModelDataset(config, video_data)
    
    print(f"✅ Dataset created with {len(dataset)} samples")
    
    # Test sample retrieval
    sample = dataset[0]
    print(f"✅ Sample shapes:")
    print(f"   Current states: {sample['current_states'].shape}")
    print(f"   Actions: {sample['actions'].shape}")
    print(f"   Next states: {sample['next_states'].shape}")
    print(f"   Rewards: {list(sample['rewards'].keys())}")
    for reward_type, reward_tensor in sample['rewards'].items():
        print(f"     {reward_type}: {reward_tensor.shape}")
    
    # Test dataloader
    print(f"\n🧪 Testing dataloader...")
    dataloader = DataLoader(dataset, batch_size=3, shuffle=True)
    
    batch = next(iter(dataloader))
    print(f"✅ Batch shapes:")
    print(f"   Current states: {batch['current_states'].shape}")
    print(f"   Actions: {batch['actions'].shape}")
    print(f"   Next states: {batch['next_states'].shape}")
    print(f"   Batch rewards: {list(batch['rewards'].keys())}")
    
    # Test single step sample
    print(f"\n🧪 Testing single step sample...")
    single_step = dataset.get_single_step_sample(0, 5)
    print(f"✅ Single step shapes:")
    print(f"   Current state: {single_step['current_state'].shape}")
    print(f"   Action: {single_step['action'].shape}")
    print(f"   Next state: {single_step['next_state'].shape}")
    print(f"   Rewards: {list(single_step['rewards'].keys())}")
    
    # Test simulation dataset
    print(f"\n🧪 Testing simulation dataset...")
    sim_dataset = WorldModelSimulationDataset(config, video_data, simulation_length=10)
    sim_sample = sim_dataset[0]
    print(f"✅ Simulation sample shapes:")
    print(f"   Initial state: {sim_sample['initial_state'].shape}")
    print(f"   Ground truth states: {sim_sample['ground_truth_states'].shape}")
    print(f"   Ground truth actions: {sim_sample['ground_truth_actions'].shape}")
    
    # Get statistics
    stats = dataset.get_dataset_stats()
    print(f"\n📊 Dataset Statistics:")
    for key, value in stats.items():
        if key == 'reward_statistics':
            print(f"   {key}:")
            for reward_type, reward_stats in value.items():
                print(f"     {reward_type}: mean={reward_stats['mean']:.3f}")
        elif isinstance(value, (int, float)):
            print(f"   {key}: {value}")
        else:
            print(f"   {key}: {value}")
    
    print(f"\n🎯 KEY FEATURES:")
    print(f"✅ Action-conditioned state-reward prediction")
    print(f"✅ Input: state + action → Target: next_state + rewards")
    print(f"✅ Multiple reward types support")
    print(f"✅ Single step and simulation modes")
    print(f"✅ Ready for Method 2 world model training")
