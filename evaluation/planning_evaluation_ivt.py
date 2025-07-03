#!/usr/bin/env python3
"""
Planning Evaluation for Autoregressive IL Model using IVT Metrics
Evaluates multi-step action prediction capabilities over different time horizons
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import os

# Import IVT metrics
try:
    import ivtmetrics
    IVT_AVAILABLE = True
except ImportError:
    IVT_AVAILABLE = False
    print("⚠️ ivtmetrics not available. Install with: pip install ivtmetrics")


class AutoregressivePlanningEvaluator:
    """
    Evaluates planning capabilities of autoregressive model using IVT metrics.
    Tests multi-step action prediction over different time horizons.
    """
    
    def __init__(self, 
                 model,
                 device: torch.device,
                 logger=None,
                 fps: int = 1):
        """
        Initialize planning evaluator.
        
        Args:
            model: Trained autoregressive model
            device: Torch device
            logger: Logger instance
            fps: Frames per second of dataset (default: 1 for CholecT50)
        """
        self.model = model
        self.device = device
        self.logger = logger
        self.fps = fps
        
        # Planning horizons to evaluate (in seconds and frames)
        self.planning_horizons = {
            '1s': 1 * fps,   # 1 frame at 1fps
            '2s': 2 * fps,  # 2 frames at 1fps  
            '3s': 3 * fps,  # 3 frames at 1fps
            '5s': 5 * fps,   # 5 frames at 1fps
            '10s': 10 * fps
        }

        # Results storage
        self.planning_results = {}
        self.detailed_results = {}
        
        if logger:
            logger.info(f"🎯 Planning Evaluator initialized")
            logger.info(f"   Planning horizons: {list(self.planning_horizons.keys())}")
            logger.info(f"   IVT available: {IVT_AVAILABLE}")
    
    def evaluate_planning_on_video(self, 
                                  video_dataloader,
                                  video_id: str,
                                  context_length: int = 20,
                                  future_length: int = 10,
                                  temperature: float = 0.1,
                                  deterministic: bool = True) -> Dict[str, Any]:
        """
        Evaluate planning capabilities on a single video.
        
        Args:
            video_dataloader: DataLoader for single video
            video_id: Video identifier
            context_length: Number of context frames to use
            future_length: Number of context frames to use
            temperature: Sampling temperature (lower = more deterministic)
            deterministic: Whether to use deterministic prediction
            
        Returns:
            Dictionary with planning evaluation results

        Expected batch format:
            'target_next_frames': torch.tensor(np.array(sample['target_next_frames']), dtype=torch.float32),
            'target_next_actions': torch.tensor(np.array(sample['target_next_actions']), dtype=torch.float32),
            'target_next_phases': torch.tensor(np.array(sample['target_next_phases']), dtype=torch.long),
            'target_future_actions': torch.tensor(np.array(sample['target_future_actions']), dtype=torch.float32),
            'video_id': sample['video_id'],
            'frame_idx': sample['frame_idx']
        """
        
        self.model.eval()
        
        # Storage for predictions and ground truth for each horizon
        horizon_predictions = {name: [] for name in self.planning_horizons.keys()}
        horizon_ground_truth = {name: [] for name in self.planning_horizons.keys()}
        
        # Video-level statistics
        total_sequences = 0
        successful_predictions = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(video_dataloader, desc=f"Planning eval {video_id}")):
                # For recognition (current state) - slice sequence dimension 
                input_frames = batch['target_next_frames'][:, :-1].to(self.device)  # [batch_size, context_length, embedding_dim]
                target_actions = batch['target_future_actions'].to(self.device)  # [batch, seq, actions]
                
                batch_size, seq_len, _ = input_frames.shape
                _, target_seq_len, num_actions = target_actions.shape
                
                # For each sample in batch
                # for sample_idx in range(batch_size):
                sample_frames = input_frames#[sample_idx:sample_idx+1]  # [1, seq, dim]
                sample_targets = target_actions#[sample_idx]  # [seq, actions]
                
                # Use last context_length frames as context
                context_frames = sample_frames[:, -context_length:, :] # [1, context_length, dim]
                
                # Get maximum horizon we can evaluate
                max_horizon = target_seq_len
                
                # Generate sequence for maximum horizon
                generation_result = self.model.generate_sequence(
                    initial_frames=context_frames,
                    horizon=max_horizon,
                    temperature=temperature if not deterministic else 0.1
                )
                
                predicted_actions = generation_result['predicted_actions']  # [1, horizon, actions]
                # predicted_actions = predicted_actions[0]  # [horizon, actions]

                # FIXED: Evaluate only the target step for each horizon
                for horizon_name, horizon_frames in self.planning_horizons.items():
                    # Get prediction for ONLY the target step
                    target_step_idx = horizon_frames - 1  # 0-indexed
                    horizon_preds = predicted_actions[:, target_step_idx:target_step_idx+1]  # [batch_size, actions]
                    
                    # Get ground truth for ONLY the target step  
                    gt_target_idx = horizon_frames - 1
                    horizon_gt = sample_targets[:, gt_target_idx:gt_target_idx+1]  # [batch_size, actions]
                    
                    # Store predictions and ground truth
                    horizon_predictions[horizon_name].append(horizon_preds.cpu().numpy())
                    horizon_ground_truth[horizon_name].append(horizon_gt.cpu().numpy())
            
        # Convert lists to numpy arrays
        for horizon_name in self.planning_horizons.keys():
            horizon_predictions[horizon_name] = np.concatenate(horizon_predictions[horizon_name], axis=0)
            horizon_ground_truth[horizon_name] = np.concatenate(horizon_ground_truth[horizon_name], axis=0)
                    
        # Compute planning metrics for each horizon
        video_results = {
            'video_id': video_id,
            'horizon_results': {}
        }
        
        for horizon_name in self.planning_horizons.keys():
            self.logger.info(f"Evaluating horizon {horizon_name} for video {video_id}...")
            horizon_result = self._compute_horizon_metrics(
                predictions=horizon_predictions[horizon_name],
                ground_truth=horizon_ground_truth[horizon_name],
                horizon_name=horizon_name,
                video_id=video_id
            )
            video_results['horizon_results'][horizon_name] = horizon_result
        
        return video_results
    
    def _compute_horizon_metrics(self, 
                                predictions: np.
                                ground_truth:  
                                horizon_name: str,
                                video_id: str) -> Dict[str, Any]:
        """
        Compute metrics for a specific planning horizon.
        
        Args:
            predictions: np.array of prediction arrays [horizon_frames, actions]
            ground_truth: np.array of ground truth arrays [horizon_frames, actions]
            horizon_name: Name of the planning horizon
            video_id: Video identifier
            
        Returns:
            Dictionary with horizon-specific metrics
        """
        
        if len(predictions.shape) == 3:
            predictions = np.concatenate(predictions, axis=0)  # [total_frames, actions]
            ground_truth = np.concatenate(ground_truth, axis=0)  # [total_frames, actions]
        
        # Convert ground truth to binary
        ground_truth_binary = (ground_truth > 0.5).astype(int)
        
        horizon_result = {
            'num_sequences': len(predictions),
            'total_frames': predictions.shape[0],
            'horizon_frames': self.planning_horizons[horizon_name],
            'avg_frames_per_sequence': predictions.shape[0] / len(predictions)
        }
        
        # Compute IVT metrics if available
        # Use fresh IVT recognizer for this horizon
        ivt_rec = ivtmetrics.Recognition(num_class=100)
        
        # Update IVT recognizer with all predictions
        ivt_rec.update(ground_truth_binary, predictions)
        ivt_rec.video_end()
        
        # Get IVT results
        ivt_result = ivt_rec.compute_video_AP("ivt")
        horizon_result['ivt_mAP'] = ivt_result["mAP"]
        
        # Component-wise results
        for component in ['i', 'v', 't', 'iv', 'it']:
            comp_result = ivt_rec.compute_video_AP(component)
            horizon_result[f'ivt_{component}_mAP'] = comp_result["mAP"]
                        
        # Compute additional planning-specific metrics
        horizon_result.update(self._compute_planning_specific_metrics(
            predictions, ground_truth_binary
        ))
        
        return horizon_result
    
    def _compute_planning_specific_metrics(self, 
                                         predictions: np.ndarray,
                                         ground_truth: np.ndarray) -> Dict[str, float]:
        """
        Compute planning-specific metrics beyond standard IVT metrics.
        
        Args:
            predictions: [total_frames, actions] prediction probabilities
            ground_truth: [total_frames, actions] binary ground truth
            
        Returns:
            Dictionary with planning metrics
        """
        
        # Binary predictions
        binary_predictions = (predictions > 0.5).astype(int)
        
        # Exact sequence match rate
        exact_match = np.mean(np.all(binary_predictions == ground_truth, axis=1))
        
        # Hamming accuracy (per-frame accuracy)
        hamming_accuracy = np.mean(binary_predictions == ground_truth)
        
        # Action consistency (how stable are predictions over time)
        if len(predictions) > 1:
            prediction_changes = np.sum(np.abs(np.diff(binary_predictions, axis=0)))
            consistency = 1.0 - (prediction_changes / (len(predictions) - 1) / predictions.shape[1])
        else:
            consistency = 1.0
        
        # Temporal smoothness (prediction probability changes)
        if len(predictions) > 1:
            prob_changes = np.mean(np.abs(np.diff(predictions, axis=0)))
            smoothness = np.exp(-prob_changes)  # Higher = smoother
        else:
            smoothness = 1.0
        
        # Action sparsity similarity (how well does predicted sparsity match ground truth)
        pred_sparsity = np.mean(binary_predictions)
        gt_sparsity = np.mean(ground_truth)
        sparsity_similarity = 1.0 - abs(pred_sparsity - gt_sparsity)
        
        return {
            'exact_match_rate': exact_match,
            'hamming_accuracy': hamming_accuracy,
            'action_consistency': consistency,
            'temporal_smoothness': smoothness,
            'pred_sparsity': pred_sparsity,
            'gt_sparsity': gt_sparsity,
            'sparsity_similarity': sparsity_similarity
        }
    
    def evaluate_planning_on_dataset(self, 
                                    test_loaders: Dict[str, torch.utils.data.DataLoader],
                                    context_length: int = 20,
                                    temperature: float = 0.1) -> Dict[str, Any]:
        """
        Evaluate planning capabilities on entire dataset.
        
        Args:
            test_loaders: Dictionary of test data loaders by video
            context_length: Number of context frames
            temperature: Sampling temperature
            
        Returns:
            Comprehensive planning evaluation results
        """
        
        if self.logger:
            self.logger.info("🎯 Starting comprehensive planning evaluation...")
        
        # Evaluate each video
        video_results = {}
        for video_id, dataloader in test_loaders.items():
            self.logger.info(f"📹 Evaluating planning on video {video_id}...")
            
            video_result = self.evaluate_planning_on_video(
                dataloader, video_id, context_length, temperature
            )
            video_results[video_id] = video_result
        
        # Aggregate results across videos
        aggregated_results = self._aggregate_planning_results(video_results)
        
        # Store results
        self.planning_results = aggregated_results
        self.detailed_results = video_results
        
        if self.logger:
            self.logger.info("✅ Planning evaluation completed")
            self._log_planning_summary()
        
        return {
            'aggregated_results': aggregated_results,
            'detailed_video_results': video_results,
            'evaluation_settings': {
                'context_length': context_length,
                'temperature': temperature,
                'planning_horizons': self.planning_horizons,
                'num_videos': len(test_loaders)
            }
        }
    
    def _aggregate_planning_results(self, video_results: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate planning results across all videos."""
        
        aggregated = {
            'num_videos': len(video_results),
            'overall_success_rate': np.mean([v['success_rate'] for v in video_results.values()]),
            'horizon_aggregated': {}
        }
        
        # Aggregate for each planning horizon
        for horizon_name in self.planning_horizons.keys():
            horizon_metrics = []
            
            for video_result in video_results.values():
                if horizon_name in video_result.get('horizon_results', {}):
                    horizon_metrics.append(video_result['horizon_results'][horizon_name])
            
            if horizon_metrics:
                # Aggregate metrics across videos
                aggregated_horizon = {
                    'num_videos_evaluated': len(horizon_metrics),
                    'planning_horizon_frames': self.planning_horizons[horizon_name],
                    'planning_horizon_seconds': self.planning_horizons[horizon_name] / self.fps
                }
                
                # Average metrics across videos
                metric_names = ['ivt_mAP', 'exact_match_rate', 'hamming_accuracy', 
                               'action_consistency', 'temporal_smoothness', 'sparsity_similarity']
                
                for metric in metric_names:
                    values = [hm.get(metric, 0.0) for hm in horizon_metrics]
                    aggregated_horizon[f'mean_{metric}'] = np.mean(values)
                    aggregated_horizon[f'std_{metric}'] = np.std(values)
                
                # Component-wise IVT metrics
                for component in ['i', 'v', 't', 'iv', 'it']:
                    values = [hm.get(f'ivt_{component}_mAP', 0.0) for hm in horizon_metrics]
                    aggregated_horizon[f'mean_ivt_{component}_mAP'] = np.mean(values)
                
                aggregated['horizon_aggregated'][horizon_name] = aggregated_horizon
        
        return aggregated
    
    def _log_planning_summary(self):
        """Log summary of planning evaluation results."""
        
        if not self.logger or not self.planning_results:
            return
        
        self.logger.info("🎯 PLANNING EVALUATION SUMMARY")
        self.logger.info("=" * 60)
        
        overall = self.planning_results
        self.logger.info(f"📊 Overall Results:")
        self.logger.info(f"   Videos evaluated: {overall['num_videos']}")
        
        self.logger.info(f"📈 Planning Horizon Results:")
        for horizon_name, horizon_data in overall['horizon_aggregated'].items():
            ivt_map = horizon_data.get('mean_ivt_mAP', 0)
            self.logger.info(f"   Planning at {horizon_name} mAP: {ivt_map:.3f}")
    
    def save_planning_results(self, output_dir: str):
        """Save planning evaluation results."""
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save aggregated results
        results_path = os.path.join(output_dir, 'planning_evaluation_results.json')
        with open(results_path, 'w') as f:
            json.dump({
                'aggregated_results': self.planning_results,
                'detailed_video_results': self.detailed_results
            }, f, indent=2, default=str)
        
        # Generate planning visualization
        self._generate_planning_plots(output_dir)
        
        if self.logger:
            self.logger.info(f"💾 Planning results saved to: {output_dir}")
    
    def _generate_planning_plots(self, output_dir: str):
        """Generate visualization plots for planning evaluation."""
        
        if not self.planning_results:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: IVT mAP vs Planning Horizon
        horizons = []
        ivt_maps = []
        
        for horizon_name, horizon_data in self.planning_results['horizon_aggregated'].items():
            seconds = horizon_data['planning_horizon_seconds']
            ivt_map = horizon_data.get('mean_ivt_mAP', 0)
            horizons.append(seconds)
            ivt_maps.append(ivt_map)
        
        axes[0, 0].plot(horizons, ivt_maps, 'bo-', linewidth=2, markersize=8)
        axes[0, 0].set_xlabel('Planning Horizon (seconds)')
        axes[0, 0].set_ylabel('IVT mAP')
        axes[0, 0].set_title('Planning Performance vs Horizon')
        axes[0, 0].grid(True)
        
        # Plot 2: Action Consistency vs Planning Horizon
        consistencies = []
        for horizon_name, horizon_data in self.planning_results['horizon_aggregated'].items():
            consistency = horizon_data.get('mean_action_consistency', 0)
            consistencies.append(consistency)
        
        axes[0, 1].plot(horizons, consistencies, 'ro-', linewidth=2, markersize=8)
        axes[0, 1].set_xlabel('Planning Horizon (seconds)')
        axes[0, 1].set_ylabel('Action Consistency')
        axes[0, 1].set_title('Action Consistency vs Horizon')
        axes[0, 1].grid(True)
        
        # Plot 3: Component-wise planning performance
        if horizons:
            components = ['i', 'v', 't', 'iv', 'it', 'ivt']
            # Use middle horizon for component comparison
            mid_horizon = list(self.planning_results['horizon_aggregated'].keys())[len(horizons)//2]
            mid_data = self.planning_results['horizon_aggregated'][mid_horizon]
            
            comp_maps = [mid_data.get(f'mean_ivt_{comp}_mAP', 0) for comp in components]
            
            axes[1, 0].bar(components, comp_maps, alpha=0.7)
            axes[1, 0].set_xlabel('Component')
            axes[1, 0].set_ylabel('mAP')
            axes[1, 0].set_title(f'Component Planning Performance ({mid_horizon})')
            axes[1, 0].grid(True, axis='y')
        
        # Plot 4: Multiple metrics comparison
        metrics = ['mean_ivt_mAP', 'mean_exact_match_rate', 'mean_hamming_accuracy']
        metric_labels = ['IVT mAP', 'Exact Match', 'Hamming Accuracy']
        
        for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
            values = []
            for horizon_data in self.planning_results['horizon_aggregated'].values():
                values.append(horizon_data.get(metric, 0))
            axes[1, 1].plot(horizons, values, 'o-', label=label, linewidth=2, markersize=6)
        
        axes[1, 1].set_xlabel('Planning Horizon (seconds)')
        axes[1, 1].set_ylabel('Performance')
        axes[1, 1].set_title('Multi-Metric Planning Performance')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'planning_evaluation_plots.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        if self.logger:
            self.logger.info(f"📊 Planning plots saved to: {plot_path}")


# Integration with existing trainer
def add_planning_evaluation_to_trainer(trainer_instance, test_loaders, context_length=10):
    """
    Add planning evaluation to existing autoregressive trainer.
    
    Args:
        trainer_instance: AutoregressiveILTrainer instance
        test_loaders: Test data loaders
        context_length: Context length for planning
        
    Returns:
        Planning evaluation results
    """
    
    # Create planning evaluator
    planning_evaluator = AutoregressivePlanningEvaluator(
        model=trainer_instance.model,
        device=trainer_instance.device,
        logger=trainer_instance.logger,
        fps=1  # CholecT50 is 1fps
    )
    
    # Run planning evaluation
    planning_results = planning_evaluator.evaluate_planning_on_dataset(
        test_loaders=test_loaders,
        context_length=context_length,
        temperature=0.1  # Deterministic for evaluation
    )
    
    # Save results
    planning_output_dir = os.path.join(trainer_instance.log_dir, 'planning_evaluation')
    planning_evaluator.save_planning_results(planning_output_dir)
    
    return planning_results


if __name__ == "__main__":
    print("🎯 AUTOREGRESSIVE PLANNING EVALUATION")
    print("=" * 50)
    print("✅ Multi-step action prediction evaluation")
    print("✅ IVT metrics for planning capabilities")
    print("✅ Different time horizons (1-5 seconds)")
    print("✅ Planning-specific metrics")
    print("✅ Temporal consistency analysis")
    
    print("\n📝 Usage:")
    print("# Add to your trainer's evaluate_model method:")
    print("planning_results = add_planning_evaluation_to_trainer(")
    print("    trainer_instance=self,")
    print("    test_loaders=test_loaders,")
    print("    context_length=10")
    print(")")
    
    if IVT_AVAILABLE:
        print("\n✅ IVT metrics available for planning evaluation")
    else:
        print("\n❌ Install ivtmetrics: pip install ivtmetrics")
