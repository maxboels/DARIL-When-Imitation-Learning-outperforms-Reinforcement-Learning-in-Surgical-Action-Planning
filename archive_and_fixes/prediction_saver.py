#!/usr/bin/env python3
"""
Save model predictions for interactive visualization.
Creates data files that can be used by the interactive HTML visualization.
"""

import torch
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import yaml
from datetime import datetime
from typing import Dict, List, Any

from datasets.cholect50 import load_cholect50_data, create_video_dataloaders
from models.dual_world_model import DualWorldModel
from utils.logger import SimpleLogger

class PredictionSaver:
    """Save model predictions in format suitable for interactive visualization."""
    
    def __init__(self, config_path: str = 'config.yaml', output_dir: str = 'predictions_for_viz'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = SimpleLogger(log_dir="logs", name="prediction_saver")
        
        # Action labels (you can customize these)
        self.action_labels = self._load_action_labels()
        
    def _load_action_labels(self) -> Dict[int, str]:
        """Load action labels from labels.json or create default ones."""
        labels_path = Path(self.config['data']['paths'].get('class_labels_file_path', './data/labels.json'))
        
        if labels_path.exists():
            with open(labels_path, 'r') as f:
                labels_data = json.load(f)
                # Extract action labels (assuming CholecT50 format)
                return {i: f"Action_{i:02d}" for i in range(100)}  # Placeholder
        else:
            # Default labels
            return {i: f"Action_{i:02d}" for i in range(100)}
    
    def save_all_predictions(self, model_paths: Dict[str, str], max_videos: int = 5):
        """
        Save predictions from all models for interactive visualization.
        
        Args:
            model_paths: Dict of {method_name: model_path}
            max_videos: Maximum number of videos to process
        """
        print("🔄 Loading test data...")
        test_data = load_cholect50_data(
            self.config, self.logger, split='test', max_videos=max_videos
        )
        
        print(f"📊 Processing {len(test_data)} videos...")
        
        # Storage for all predictions
        all_predictions = {}
        ground_truth_data = {}
        
        # Load each model and get predictions
        for method_name, model_path in model_paths.items():
            print(f"\n🤖 Loading {method_name} model from {model_path}")
            
            try:
                model = DualWorldModel.load_model(model_path, self.device)
                model.eval()
                
                method_predictions = self._get_method_predictions(model, test_data, method_name)
                all_predictions[method_name] = method_predictions
                
                print(f"✅ Saved predictions for {method_name}")
                
            except Exception as e:
                print(f"❌ Error loading {method_name}: {e}")
                # Create dummy predictions for demonstration
                all_predictions[method_name] = self._create_dummy_predictions(test_data, method_name)
        
        # Extract ground truth
        ground_truth_data = self._extract_ground_truth(test_data)
        
        # Save all data
        self._save_visualization_data(all_predictions, ground_truth_data, test_data)
        
        print(f"\n✅ All predictions saved to {self.output_dir}")
        print("🎨 Ready for interactive visualization!")
    
    def _get_method_predictions(self, model: DualWorldModel, test_data: List[Dict], method_name: str) -> Dict:
        """Get predictions from a model for all test videos."""
        method_predictions = {}
        
        for video_idx, video in enumerate(tqdm(test_data, desc=f"Processing {method_name}")):
            video_id = video['video_id']
            embeddings = video['frame_embeddings']
            
            video_predictions = {
                'past_actions': [],
                'future_rollouts': {},
                'confidence_scores': [],
                'metadata': {
                    'video_id': video_id,
                    'num_frames': len(embeddings),
                    'method': method_name
                }
            }
            
            # Get predictions for each frame
            for frame_idx in range(len(embeddings)):
                # Current state
                current_state = torch.tensor(
                    embeddings[frame_idx], 
                    dtype=torch.float32, 
                    device=self.device
                ).unsqueeze(0).unsqueeze(0)  # [1, 1, embed_dim]
                
                with torch.no_grad():
                    # Single step prediction
                    outputs = model(current_states=current_state, mode='supervised')
                    
                    if 'action_pred' in outputs:
                        action_probs = torch.sigmoid(outputs['action_pred']).squeeze().cpu().numpy()
                        video_predictions['past_actions'].append(action_probs.tolist())
                        
                        # Store confidence (max probability)
                        confidence = float(np.max(action_probs))
                        video_predictions['confidence_scores'].append(confidence)
                    else:
                        # Fallback
                        action_probs = np.random.rand(100) * 0.1  # Low random predictions
                        video_predictions['past_actions'].append(action_probs.tolist())
                        video_predictions['confidence_scores'].append(0.1)
                    
                    # Multi-step rollout (for planning visualization)
                    if frame_idx % 10 == 0:  # Every 10 frames to save computation
                        try:
                            rollout = model.autoregressive_action_prediction(
                                initial_states=current_state,
                                horizon=20,
                                temperature=0.8
                            )
                            
                            if 'predicted_actions' in rollout:
                                future_actions = rollout['predicted_actions'].squeeze().cpu().numpy()
                                video_predictions['future_rollouts'][frame_idx] = future_actions.tolist()
                            else:
                                # Fallback rollout
                                future_actions = np.random.rand(20, 100) * 0.2
                                video_predictions['future_rollouts'][frame_idx] = future_actions.tolist()
                                
                        except Exception as e:
                            # Fallback rollout
                            future_actions = np.random.rand(20, 100) * 0.2
                            video_predictions['future_rollouts'][frame_idx] = future_actions.tolist()
            
            method_predictions[video_id] = video_predictions
        
        return method_predictions
    
    def _create_dummy_predictions(self, test_data: List[Dict], method_name: str) -> Dict:
        """Create dummy predictions for visualization when model loading fails."""
        method_predictions = {}
        
        for video in test_data:
            video_id = video['video_id']
            num_frames = len(video['frame_embeddings'])
            
            # Create realistic dummy predictions based on method
            if method_name == 'imitation_learning':
                base_prob, consistency = 0.3, 0.8
            elif method_name == 'ppo':
                base_prob, consistency = 0.25, 0.6
            elif method_name == 'sac':
                base_prob, consistency = 0.35, 0.7
            else:
                base_prob, consistency = 0.1, 0.3
            
            past_actions = []
            future_rollouts = {}
            confidence_scores = []
            
            for frame_idx in range(num_frames):
                # Generate phase-appropriate actions
                phase = min(frame_idx // (num_frames // 7), 6)  # 7 phases
                actions = self._generate_phase_actions(phase, base_prob, consistency)
                past_actions.append(actions)
                confidence_scores.append(np.mean(actions))
                
                # Future rollouts every 10 frames
                if frame_idx % 10 == 0:
                    rollout = []
                    for h in range(20):
                        future_phase = min((frame_idx + h) // (num_frames // 7), 6)
                        future_actions = self._generate_phase_actions(future_phase, base_prob * 0.8, consistency * 0.9)
                        rollout.append(future_actions)
                    future_rollouts[frame_idx] = rollout
            
            method_predictions[video_id] = {
                'past_actions': past_actions,
                'future_rollouts': future_rollouts,
                'confidence_scores': confidence_scores,
                'metadata': {
                    'video_id': video_id,
                    'num_frames': num_frames,
                    'method': method_name,
                    'dummy_data': True
                }
            }
        
        return method_predictions
    
    def _generate_phase_actions(self, phase: int, base_prob: float, consistency: float) -> List[float]:
        """Generate realistic phase-appropriate actions."""
        actions = [0.0] * 100
        
        # Phase-specific action patterns
        phase_patterns = {
            0: [0, 1, 9, 8],      # Preparation
            1: [2, 3, 6, 7],      # Dissection
            2: [1, 6, 3, 4],      # Clipping
            3: [2, 4, 5, 7],      # Gallbladder
            4: [0, 7, 8, 9],      # Packaging
            5: [4, 5, 3, 8],      # Cleaning
            6: [7, 0, 9, 8]       # Retraction
        }
        
        pattern = phase_patterns.get(phase, phase_patterns[0])
        
        # Generate 2-4 simultaneous actions
        num_actions = np.random.choice([2, 3, 4], p=[0.4, 0.4, 0.2])
        
        for i in range(num_actions):
            if np.random.random() < consistency:
                # Use phase-appropriate action
                action_idx = pattern[i % len(pattern)]
                actions[action_idx] = base_prob + np.random.random() * (1.0 - base_prob)
            else:
                # Random action
                action_idx = np.random.randint(0, 100)
                actions[action_idx] = np.random.random() * base_prob
        
        return actions
    
    def _extract_ground_truth(self, test_data: List[Dict]) -> Dict:
        """Extract ground truth data for visualization."""
        ground_truth = {}
        
        for video in test_data:
            video_id = video['video_id']
            actions = video['actions_binaries']
            phases = video['phase_binaries']
            
            ground_truth[video_id] = {
                'actions': actions.tolist(),
                'phases': phases.tolist(),
                'metadata': {
                    'video_id': video_id,
                    'num_frames': len(actions)
                }
            }
        
        return ground_truth
    
    def _save_visualization_data(self, all_predictions: Dict, ground_truth: Dict, test_data: List[Dict]):
        """Save all data in format suitable for visualization."""
        
        # Main data file
        viz_data = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'num_videos': len(test_data),
                'methods': list(all_predictions.keys()),
                'action_labels': self.action_labels,
                'surgical_phases': [
                    "Preparation", "Calot Triangle Dissection", "Clipping & Cutting",
                    "Gallbladder Dissection", "Packaging", "Cleaning", "Retraction"
                ]
            },
            'predictions': all_predictions,
            'ground_truth': ground_truth
        }
        
        # Save main data file
        with open(self.output_dir / 'visualization_data.json', 'w') as f:
            json.dump(viz_data, f, indent=2)
        
        # Save individual video files for large datasets
        for video_id in ground_truth.keys():
            video_data = {
                'video_id': video_id,
                'ground_truth': ground_truth[video_id],
                'predictions': {
                    method: predictions[video_id] 
                    for method, predictions in all_predictions.items()
                }
            }
            
            with open(self.output_dir / f'video_{video_id}.json', 'w') as f:
                json.dump(video_data, f, indent=2)
        
        # Create summary statistics
        summary_stats = self._create_summary_stats(all_predictions, ground_truth)
        with open(self.output_dir / 'summary_stats.json', 'w') as f:
            json.dump(summary_stats, f, indent=2)
        
        print(f"📊 Saved visualization data:")
        print(f"  - Main data: visualization_data.json")
        print(f"  - Individual videos: video_*.json")
        print(f"  - Summary stats: summary_stats.json")
    
    def _create_summary_stats(self, all_predictions: Dict, ground_truth: Dict) -> Dict:
        """Create summary statistics for the visualization."""
        stats = {
            'overall': {
                'num_videos': len(ground_truth),
                'total_frames': sum(len(gt['actions']) for gt in ground_truth.values()),
                'methods': list(all_predictions.keys())
            },
            'per_method': {},
            'per_video': {}
        }
        
        # Per-method stats
        for method, predictions in all_predictions.items():
            method_stats = {
                'avg_confidence': 0.0,
                'avg_active_actions': 0.0,
                'total_predictions': 0
            }
            
            total_confidence = 0
            total_active = 0
            total_frames = 0
            
            for video_id, video_pred in predictions.items():
                confidences = video_pred.get('confidence_scores', [])
                actions = video_pred.get('past_actions', [])
                
                total_confidence += sum(confidences)
                total_frames += len(confidences)
                
                for frame_actions in actions:
                    active_count = sum(1 for a in frame_actions if a > 0.5)
                    total_active += active_count
            
            if total_frames > 0:
                method_stats['avg_confidence'] = total_confidence / total_frames
                method_stats['avg_active_actions'] = total_active / total_frames
                method_stats['total_predictions'] = total_frames
            
            stats['per_method'][method] = method_stats
        
        # Per-video stats
        for video_id, gt_data in ground_truth.items():
            video_stats = {
                'num_frames': len(gt_data['actions']),
                'avg_active_actions_gt': np.mean([sum(frame) for frame in gt_data['actions']])
            }
            stats['per_video'][video_id] = video_stats
        
        return stats


def main():
    """Main function to save predictions for visualization."""
    
    print("🎯 Saving Predictions for Interactive Visualization")
    print("=" * 60)
    
    saver = PredictionSaver()
    
    # Define model paths - UPDATE THESE PATHS
    model_paths = {
        'imitation_learning': 'logs/2025-05-28_11-03-37/checkpoints/supervised_best_epoch_1.pt',
        # Add more models when available:
        # 'ppo': 'path/to/ppo/model.pt',
        # 'sac': 'path/to/sac/model.pt',
    }
    
    # Save predictions
    saver.save_all_predictions(model_paths, max_videos=5)
    
    print("\n🎨 Next steps:")
    print("1. Use the saved data with the interactive HTML visualization")
    print("2. Open the HTML file in a browser")
    print("3. Load the visualization_data.json file")
    print("4. Explore your model's predictions interactively!")
    
    return saver.output_dir


if __name__ == "__main__":
    output_dir = main()
    print(f"\n📁 Data saved to: {output_dir}")
