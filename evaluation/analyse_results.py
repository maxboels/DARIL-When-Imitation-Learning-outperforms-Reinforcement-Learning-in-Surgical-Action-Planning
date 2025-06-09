#!/usr/bin/env python3
"""
Detailed Metric Debugging Script
Step-by-step analysis of how evaluation metrics are computed
"""
import os
import yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Tuple
from sklearn.metrics import average_precision_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns

# Import models
from models.autoregressive_il_model import AutoregressiveILModel
from models.conditional_world_model import ConditionalWorldModel
from datasets.cholect50 import load_cholect50_data
from datasets.world_model_dataset import create_world_model_dataloaders
from utils.logger import SimpleLogger
from .extended_clinical_evaluation import ClinicalSurgicalEvaluator, generate_clinical_evaluation_report

plt.style.use('seaborn-v0_8-whitegrid')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MetricDebugger:
    """
    Detailed metric debugging for understanding evaluation computation.
    """
    
    def __init__(self, config_path: str = 'config_local_debug.yaml', pretrained_dir: str = None):
        print("🔬 Initializing Metric Debugger")
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set pretrained directory
        if pretrained_dir is None:
            results_base = Path("results")
            if results_base.exists():
                subdirs = [d for d in results_base.iterdir() if d.is_dir()]
                if subdirs:
                    latest_dir = max(subdirs, key=lambda d: d.stat().st_mtime)
                    pretrained_dir = str(latest_dir)
        
        self.pretrained_dir = Path(pretrained_dir)
        
        # Create debug directory
        self.debug_dir = Path("debug_metrics")
        self.debug_dir.mkdir(exist_ok=True)
        
        # Initialize logger
        self.logger = SimpleLogger(
            log_dir=str(self.debug_dir),
            name="MetricDebug",
            use_shared_timestamp=True
        )
        
        # INITIALIZE CLINICAL EVALUATOR HERE
        self.clinical_evaluator = ClinicalSurgicalEvaluator('./data/labels.json')
        self.logger.info("🏥 Clinical evaluator initialized")
        
        print(f"📁 Loading from: {self.pretrained_dir}")
        print(f"📁 Debug output: {self.debug_dir}")
    
    def debug_all_metrics(self):
        """Debug all metrics step by step."""
        
        self.logger.info("🔬 Starting Comprehensive Metric Debugging")
        self.logger.info("=" * 60)

        train_data = load_cholect50_data(
            self.config, self.logger, split='train', max_videos=1
        )
        self.logger.info(f"Loaded {len(train_data)} training videos for debugging")
        
        # Load test data (small sample for detailed analysis)
        test_data = load_cholect50_data(
            self.config, self.logger, split='test', max_videos=2
        )
        
        # Load one model for detailed analysis
        model = self._load_autoregressive_model()
        
        if model is None:
            self.logger.error("❌ Could not load model for debugging")
            return
        
        # Create test loader
        _, test_loaders, _ = create_world_model_dataloaders(
            config=self.config['data'],
            train_data=train_data,
            test_data=test_data,
            batch_size=8,  # Small batch for debugging
            num_workers=0  # No multiprocessing for debugging
        )
        
        # Debug each metric
        for video_id, test_loader in test_loaders.items():
            self.logger.info(f"🎥 Debugging metrics for video: {video_id}")
            self._debug_single_video_metrics(model, test_loader, video_id)
            break  # Only debug first video for detailed analysis
    
    def _load_autoregressive_model(self):
        """Load autoregressive model for debugging."""
        
        try:
            checkpoints_dir = self.pretrained_dir / "2025-06-07_18-44" / "checkpoints"
            model_path = checkpoints_dir / "autoregressive_il_best_epoch_1.pt"
            
            if not model_path.exists():
                model_path = checkpoints_dir / "autoregressive_il_final.pt"
            
            if not model_path.exists():
                self.logger.error(f"Model not found in {checkpoints_dir}")
                return None
            
            self.logger.info(f"📦 Loading model from: {model_path}")
            
            model = AutoregressiveILModel(
                hidden_dim=self.config['models']['autoregressive_il']['hidden_dim'],
                embedding_dim=self.config['models']['autoregressive_il']['embedding_dim'],
                n_layer=self.config['models']['autoregressive_il']['n_layer'],
                num_action_classes=self.config['models']['autoregressive_il']['num_action_classes'],
                dropout=self.config['models']['autoregressive_il']['dropout']
            ).to(DEVICE)
            
            # Load world model checkpoint with metadata
            checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
            
            # Extract configuration and state dict
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                saved_config = checkpoint.get('config', {})
                reward_types = checkpoint.get('reward_types', [])
                self.logger.info(f"📋 Using saved config: {saved_config}")
                self.logger.info(f"📋 Reward types: {reward_types}")
                
                model_config = saved_config if saved_config else self.config['models']['autoregressive_il']
            else:
                state_dict = checkpoint
                model_config = self.config['models']['autoregressive_il']
                self.logger.info(f"📋 Using current config (no saved config found)")
            
            # Load state dict
            model.load_state_dict(state_dict, strict=False)
            self.logger.info("✅ Model loaded successfully")
            model.eval()
            
            return model
            
        except Exception as e:
            self.logger.error(f"❌ Model loading failed: {e}")
            return None
    
    def _debug_single_video_metrics(self, model, test_loader, video_id: str):
        """Debug metrics computation for a single video in detail."""
        
        self.logger.info(f"🔍 Detailed metric analysis for {video_id}")
        
        # Collect predictions and ground truth
        all_predictions = []
        all_ground_truth = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                # if batch_idx >= 5:  # Only process first 5 batches for debugging
                #     break
                
                # Extract data
                current_states = batch['current_states'].to(DEVICE)  # [batch, seq, emb]
                next_actions = batch['next_actions'].to(DEVICE)      # [batch, seq, actions]
                
                # self.logger.info(f"  Batch {batch_idx}: {current_states.shape[0]} samples")
                
                # Get predictions (using last timestep)
                batch_preds = []
                batch_gt = []
                
                for i in range(current_states.shape[0]):
                    # Single sequence
                    seq_states = current_states[i:i+1]  # [1, seq, emb]
                    seq_gt = next_actions[i, -1]        # [actions] - last timestep
                    
                    # Get model prediction
                    outputs = model(frame_embeddings=seq_states)
                    action_pred = outputs['action_pred'][0]  # Remove batch dim
                    action_pred = action_pred[-1]  # Take only last timestep: [num_actions]
                    
                    batch_preds.append(action_pred.cpu().numpy())
                    batch_gt.append(seq_gt.cpu().numpy())
                
                all_predictions.extend(batch_preds)
                all_ground_truth.extend(batch_gt)
        
        # Convert to arrays
        predictions = np.array(all_predictions)      # [n_samples, n_actions]
        ground_truth = np.array(all_ground_truth)    # [n_samples, n_actions]
        
        self.logger.info(f"📊 Collected data shape:")
        self.logger.info(f"   Predictions: {predictions.shape}")
        self.logger.info(f"   Ground truth: {ground_truth.shape}")
        
        # 🔥 ADD CLINICAL EVALUATION HERE - AFTER COLLECTING DATA
        self.logger.info("🏥 Running Clinical Evaluation")
        self.logger.info("-" * 40)
        
        # Identify occurring actions
        occurring_actions = np.sum(ground_truth, axis=0) > 0
        
        # Run clinical evaluation
        try:
            clinical_results = self.clinical_evaluator.evaluate_clinical_performance(
                predictions, ground_truth, occurring_actions
            )
            
            # Generate and log clinical report
            clinical_report = generate_clinical_evaluation_report(clinical_results, self.clinical_evaluator)
            
            # Log the clinical report
            self.logger.info("📋 CLINICAL EVALUATION REPORT:")
            for line in clinical_report.split('\n'):
                self.logger.info(line)
            
            # Save clinical results to file
            clinical_report_path = self.debug_dir / f"{video_id}_clinical_evaluation.txt"
            with open(clinical_report_path, 'w') as f:
                f.write(clinical_report)
            self.logger.info(f"💾 Clinical report saved to: {clinical_report_path}")
            
        except Exception as e:
            self.logger.error(f"❌ Clinical evaluation failed: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Continue with original debug methods
        self._debug_map_computation(predictions, ground_truth, video_id)
        self._debug_exact_match_computation(predictions, ground_truth, video_id)
        self._debug_action_distribution(predictions, ground_truth, video_id)
        self._create_debug_visualizations(predictions, ground_truth, video_id)
    
    def _debug_map_computation(self, predictions: np.ndarray, ground_truth: np.ndarray, video_id: str):
        """Debug mAP computation step by step."""
        
        self.logger.info("📊 DEBUGGING mAP COMPUTATION")
        self.logger.info("-" * 40)
        
        ap_scores = []
        class_stats = []
        
        for action_idx in range(min(100, predictions.shape[1])):  # Debug all 100 actions
            gt_column = ground_truth[:, action_idx]
            pred_column = predictions[:, action_idx]
            
            n_positive = np.sum(gt_column)
            n_negative = len(gt_column) - n_positive
            
            if action_idx < 10:  # Only log details for first 10
                self.logger.info(f"Action {action_idx}:")
                self.logger.info(f"  Positive samples: {n_positive}")
                self.logger.info(f"  Negative samples: {n_negative}")
                self.logger.info(f"  Pred range: [{pred_column.min():.3f}, {pred_column.max():.3f}]")
            
            if n_positive > 0:
                ap = average_precision_score(gt_column, pred_column)
                ap_scores.append(ap)
                if action_idx < 10:
                    self.logger.info(f"  Average Precision: {ap:.4f}")
                
                # Detailed analysis for first few actions
                if action_idx < 3:
                    self._analyze_single_action_ap(gt_column, pred_column, action_idx)
            else:
                if action_idx < 10:
                    self.logger.info(f"  No positive samples - using binary accuracy")
                binary_pred = (pred_column > 0.5).astype(int)
                acc = 1.0 if np.sum(binary_pred) == 0 else 0.0
                ap_scores.append(acc)
            
            class_stats.append({
                'action_idx': action_idx,
                'n_positive': n_positive,
                'n_negative': n_negative,
                'ap_score': ap_scores[-1],
                'pred_mean': pred_column.mean(),
                'pred_std': pred_column.std()
            })
        
        overall_map = np.mean(ap_scores)
        
        # Compute Fair mAP (only occurring actions)
        occurring_ap_scores = [stats['ap_score'] for stats in class_stats if stats['n_positive'] > 0]
        fair_map = np.mean(occurring_ap_scores) if occurring_ap_scores else 0.0
        
        self.logger.info(f"Standard mAP (all 100 actions): {overall_map:.4f}")
        self.logger.info(f"Fair mAP (only occurring actions): {fair_map:.4f}")
        self.logger.info(f"Actions evaluated: {len(occurring_ap_scores)}/100")
        
        # Save detailed stats
        stats_df = pd.DataFrame(class_stats)
        stats_path = self.debug_dir / f"{video_id}_action_stats.csv"
        stats_df.to_csv(stats_path, index=False)
        self.logger.info(f"💾 Action stats saved to: {stats_path}")
    
    def _analyze_single_action_ap(self, gt: np.ndarray, pred: np.ndarray, action_idx: int):
        """Detailed analysis of AP computation for a single action."""
        
        if np.sum(gt) == 0:
            return
        
        # Sort by prediction confidence
        sorted_indices = np.argsort(pred)[::-1]
        sorted_gt = gt[sorted_indices]
        sorted_pred = pred[sorted_indices]
        
        # Compute precision-recall curve manually
        tp = 0
        fp = 0
        precisions = []
        recalls = []
        
        total_positives = np.sum(gt)
        
        self.logger.info(f"    Detailed AP analysis for action {action_idx}:")
        self.logger.info(f"    Total positives: {total_positives}")
        
        for i, (is_positive, confidence) in enumerate(zip(sorted_gt, sorted_pred)):
            if is_positive:
                tp += 1
            else:
                fp += 1
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / total_positives if total_positives > 0 else 0
            
            precisions.append(precision)
            recalls.append(recall)
            
            if i < 5:  # Show first 5 steps
                self.logger.info(f"      Step {i+1}: conf={confidence:.3f}, +={is_positive}, "
                               f"P={precision:.3f}, R={recall:.3f}")
        
        # Compute AP using trapezoidal rule
        ap_manual = 0
        for i in range(1, len(recalls)):
            ap_manual += (recalls[i] - recalls[i-1]) * precisions[i]
        
        self.logger.info(f"    Manual AP calculation: {ap_manual:.4f}")
    
    def _debug_exact_match_computation(self, predictions: np.ndarray, ground_truth: np.ndarray, video_id: str):
        """Debug exact match computation."""
        
        self.logger.info("🎯 DEBUGGING EXACT MATCH COMPUTATION")
        self.logger.info("-" * 40)
        
        # Convert predictions to binary
        binary_preds = (predictions > 0.5).astype(int)
        
        # Compute exact matches
        exact_matches = np.all(binary_preds == ground_truth, axis=1)
        exact_match_rate = np.mean(exact_matches)
        
        self.logger.info(f"Threshold: 0.5")
        self.logger.info(f"Binary predictions shape: {binary_preds.shape}")
        self.logger.info(f"Number of predicted actions / ground truth actions:"
                        f" {np.sum(binary_preds)} / {int(np.sum(ground_truth))}")
        self.logger.info(f"Exact matches: {np.sum(exact_matches)}/{len(exact_matches)}")
        self.logger.info(f"Exact match rate: {exact_match_rate:.4f}")
        
        # Analyze first few samples
        self.logger.info("Sample-by-sample analysis (first 5):")
        for i in range(min(5, len(predictions))):
            n_pred_actions = np.sum(binary_preds[i])
            n_gt_actions = np.sum(ground_truth[i])
            is_match = exact_matches[i]
            
            self.logger.info(f"  Sample {i}: pred_actions={n_pred_actions}, "
                           f"gt_actions={n_gt_actions}, match={is_match}")
        
        # Try different thresholds
        self.logger.info("Exact match rates at different thresholds:")
        for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
            binary_preds_t = (predictions > threshold).astype(int)
            exact_matches_t = np.all(binary_preds_t == ground_truth, axis=1)
            rate_t = np.mean(exact_matches_t)
            self.logger.info(f"  Threshold {threshold}: {rate_t:.4f}")
    
    def _debug_action_distribution(self, predictions: np.ndarray, ground_truth: np.ndarray, video_id: str):
        """Debug action distribution patterns."""
        
        self.logger.info("📈 DEBUGGING ACTION DISTRIBUTION")
        self.logger.info("-" * 40)
        
        # Overall statistics
        pred_mean = predictions.mean()
        pred_std = predictions.std()
        gt_sparsity = np.mean(np.sum(ground_truth, axis=1))
        
        self.logger.info(f"Prediction statistics:")
        self.logger.info(f"  Mean: {pred_mean:.4f}")
        self.logger.info(f"  Std: {pred_std:.4f}")
        self.logger.info(f"  Min: {predictions.min():.4f}")
        self.logger.info(f"  Max: {predictions.max():.4f}")
        
        self.logger.info(f"Ground truth statistics:")
        self.logger.info(f"  Mean actions per sample: {gt_sparsity:.2f}")
        self.logger.info(f"  Total positive examples: {np.sum(ground_truth)}")
        self.logger.info(f"  Sparsity: {1 - np.mean(ground_truth):.4f}")
        
        # Most/least predicted actions
        action_pred_means = np.mean(predictions, axis=0)
        action_gt_means = np.mean(ground_truth, axis=0)
        
        top_pred_actions = np.argsort(action_pred_means)[-5:]
        top_gt_actions = np.argsort(action_gt_means)[-5:]
        
        self.logger.info("Top 5 most predicted actions:")
        for i, action_idx in enumerate(top_pred_actions):
            self.logger.info(f"  Action {action_idx}: pred={action_pred_means[action_idx]:.3f}, "
                           f"gt={action_gt_means[action_idx]:.3f}")
        
        self.logger.info("Top 5 most frequent ground truth actions:")
        for i, action_idx in enumerate(top_gt_actions):
            self.logger.info(f"  Action {action_idx}: gt={action_gt_means[action_idx]:.3f}, "
                           f"pred={action_pred_means[action_idx]:.3f}")
    
    def _create_debug_visualizations(self, predictions: np.ndarray, ground_truth: np.ndarray, video_id: str):
        """Create debugging visualizations."""
        
        self.logger.info("📊 Creating debug visualizations...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Prediction distribution
        axes[0, 0].hist(predictions.flatten(), bins=50, alpha=0.7, label='Predictions')
        axes[0, 0].axvline(0.5, color='red', linestyle='--', label='Threshold')
        axes[0, 0].set_xlabel('Prediction Value')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Prediction Distribution')
        axes[0, 0].legend()
        
        # 2. Ground truth distribution
        axes[0, 1].hist(ground_truth.flatten(), bins=2, alpha=0.7, label='Ground Truth')
        axes[0, 1].set_xlabel('Ground Truth Value')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Ground Truth Distribution')
        axes[0, 1].legend()
        
        # 3. Actions per sample
        actions_per_sample_pred = np.sum(predictions > 0.5, axis=1)
        actions_per_sample_gt = np.sum(ground_truth, axis=1)
        
        axes[0, 2].scatter(actions_per_sample_gt, actions_per_sample_pred, alpha=0.6)
        axes[0, 2].plot([0, max(actions_per_sample_gt)], [0, max(actions_per_sample_gt)], 
                       'r--', label='Perfect prediction')
        axes[0, 2].set_xlabel('GT Actions per Sample')
        axes[0, 2].set_ylabel('Predicted Actions per Sample')
        axes[0, 2].set_title('Actions per Sample Comparison')
        axes[0, 2].legend()
        
        # 4. Action frequency correlation
        action_pred_freq = np.mean(predictions, axis=0)
        action_gt_freq = np.mean(ground_truth, axis=0)
        
        axes[1, 0].scatter(action_gt_freq, action_pred_freq, alpha=0.6)
        axes[1, 0].plot([0, max(action_gt_freq)], [0, max(action_gt_freq)], 'r--')
        axes[1, 0].set_xlabel('GT Action Frequency')
        axes[1, 0].set_ylabel('Predicted Action Frequency')
        axes[1, 0].set_title('Action Frequency Correlation')
        
        # 5. Confidence vs accuracy
        # Sample a subset for clarity
        sample_indices = np.random.choice(len(predictions), min(1000, len(predictions)), replace=False)
        sample_preds = predictions[sample_indices]
        sample_gt = ground_truth[sample_indices]
        
        # For each sample, calculate max confidence and whether it's correct
        max_confidences = np.max(sample_preds, axis=1)
        is_correct = np.all((sample_preds > 0.5) == sample_gt, axis=1)
        
        axes[1, 1].scatter(max_confidences, is_correct.astype(float), alpha=0.6)
        axes[1, 1].set_xlabel('Max Confidence')
        axes[1, 1].set_ylabel('Exact Match (0/1)')
        axes[1, 1].set_title('Confidence vs Correctness')
        
        # 6. Precision-Recall curve for top action
        if np.sum(ground_truth) > 0:
            # Find most frequent action for PR curve
            action_frequencies = np.sum(ground_truth, axis=0)
            top_action = np.argmax(action_frequencies)
            
            if action_frequencies[top_action] > 0:
                from sklearn.metrics import precision_recall_curve
                precision, recall, thresholds = precision_recall_curve(
                    ground_truth[:, top_action], predictions[:, top_action]
                )
                
                axes[1, 2].plot(recall, precision, linewidth=2)
                axes[1, 2].set_xlabel('Recall')
                axes[1, 2].set_ylabel('Precision')
                axes[1, 2].set_title(f'PR Curve (Action {top_action})')
                axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        viz_path = self.debug_dir / f"{video_id}_debug_visualizations.png"
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"📊 Visualizations saved to: {viz_path}")


def main():
    """Main function for metric debugging."""
    
    print("🔬 METRIC DEBUGGING TOOL WITH CLINICAL EVALUATION")
    print("=" * 50)
    print("Purpose: Understand evaluation metrics step-by-step")
    print("🏥 Includes clinical evaluation framework")
    print()
    
    # Configuration and pretrained model directory
    config_path = 'config_eval_debug.yaml'
    pretrained_dir = '/home/maxboels/projects/surl/results/2025-06-07_18-44-58'
    
    # Allow command line override
    import sys
    if len(sys.argv) > 1:
        pretrained_dir = sys.argv[1]
    
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        return
    
    try:
        debugger = MetricDebugger(config_path, pretrained_dir)
        debugger.debug_all_metrics()
        
        print("\n🎉 METRIC DEBUGGING COMPLETED!")
        print("=" * 40)
        print(f"📁 Debug files saved to: {debugger.debug_dir}")
        print("🔬 Check logs and visualizations for detailed analysis")
        print("🏥 Clinical evaluation report included!")

    except Exception as e:
        print(f"\n❌ DEBUGGING FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()