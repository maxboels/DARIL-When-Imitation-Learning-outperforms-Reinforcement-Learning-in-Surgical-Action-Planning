#!/usr/bin/env python3
"""
Shared Evaluation Metrics for Surgical RL/IL Training and Testing
Ensures consistency between training and testing evaluation metrics
"""

import numpy as np
from typing import Dict, List, Any, Tuple
from sklearn.metrics import average_precision_score, precision_recall_fscore_support
import warnings

warnings.filterwarnings('ignore')


class SurgicalEvaluationMetrics:
    """
    Shared evaluation metrics for surgical action prediction tasks.
    
    Handles:
    - Action sparsity (common in surgical tasks)
    - Multiple mAP calculation strategies
    - Comprehensive single-step and sequence metrics
    - Consistent metric calculation across training and testing
    """
    
    def __init__(self, logger=None):
        self.logger = logger
    
    def calculate_comprehensive_action_metrics(self, 
                                             predictions: np.ndarray, 
                                             ground_truth: np.ndarray, 
                                             method_name: str = "unknown",
                                             exclude_last_n: int = 6,
                                             main_metric: str = 'mAP_present_actions_only'
                                             ) -> Dict[str, float]:
        """
        Calculate comprehensive single-step action prediction metrics.

        
        This is the function used by both training and testing stages.
        
        Args:
            predictions: [num_samples, num_actions] - action probabilities
            ground_truth: [num_samples, num_actions] - binary ground truth
            method_name: Name of the method for logging
            exclude_last_n: Number of last actions to exclude in subset evaluation
            
        Returns:
            Dictionary with comprehensive metrics
        """
        
        def _compute_map_scores(preds, gt, handle_sparsity='present_only'):
            """
            Compute mAP scores with different sparsity handling strategies.
            
            Args:
                handle_sparsity: 'standard', 'present_only', 'frequency_weighted', 'sample_wise'
            """
            binary_preds = (preds > 0.5).astype(int)
            
            if handle_sparsity == 'sample_wise':
                # Compute AP per sample instead of per action
                sample_aps = []
                for sample_idx in range(len(gt)):
                    gt_sample = gt[sample_idx]
                    pred_sample = preds[sample_idx]
                    
                    if np.sum(gt_sample) > 0:  # Only if sample has positive actions
                        try:
                            ap = average_precision_score(gt_sample, pred_sample)
                            sample_aps.append(ap)
                        except:
                            sample_aps.append(0.0)
                
                return np.mean(sample_aps) if sample_aps else 0.0
            
            # Action-wise approaches
            ap_scores = []
            action_frequencies = []
            
            for i in range(gt.shape[1]):  # iterate over actions
                gt_class_i = gt[:, i]
                pred_class_i = preds[:, i]
                action_freq = np.sum(gt_class_i) / len(gt_class_i)  # frequency of this action
                
                if np.sum(gt_class_i) > 0:
                    try:
                        ap = average_precision_score(gt_class_i, pred_class_i)
                        ap_scores.append(ap)
                        action_frequencies.append(action_freq)
                    except:
                        ap_scores.append(0.0)
                        action_frequencies.append(action_freq)
                else:
                    # Action not present in this batch/video
                    if handle_sparsity == 'standard':
                        # Original behavior
                        if np.sum(binary_preds[:, i]) == 0:
                            ap_scores.append(1.0)  # Perfect score for absent action
                        else:
                            ap_scores.append(0.0)  # False positive
                        action_frequencies.append(action_freq)
                    elif handle_sparsity == 'present_only':
                        # Skip absent actions entirely
                        continue
                    # For frequency_weighted, we'll skip absent actions too
            
            if not ap_scores:
                return 0.0
                
            if handle_sparsity == 'frequency_weighted' and action_frequencies:
                # Weight by action frequency to reduce impact of rare actions
                weights = np.array(action_frequencies)
                weights = weights / np.sum(weights) if np.sum(weights) > 0 else np.ones_like(weights)
                return np.average(ap_scores, weights=weights)
            else:
                return np.mean(ap_scores)
        
        def _compute_sequence_metrics(preds, gt):
            """Helper to compute sequence-level metrics."""
            binary_preds = (preds > 0.5).astype(int)
            
            exact_match = np.mean(np.all(binary_preds == gt, axis=1))
            hamming_accuracy = np.mean(binary_preds == gt)
            
            try:
                precision, recall, f1, _ = precision_recall_fscore_support(
                    gt.flatten(), binary_preds.flatten(), 
                    average='macro', zero_division=0
                )
            except:
                precision = recall = f1 = 0.0
                
            return exact_match, hamming_accuracy, precision, recall, f1
        
        # Compute different mAP variants on all actions
        mAP_standard = _compute_map_scores(predictions, ground_truth, 'standard')
        mAP_present_only = _compute_map_scores(predictions, ground_truth, 'present_only')
        mAP_freq_weighted = _compute_map_scores(predictions, ground_truth, 'frequency_weighted')
        mAP_sample_wise = _compute_map_scores(predictions, ground_truth, 'sample_wise')
        
        # Other metrics on all actions
        exact_match_all, hamming_all, precision_all, recall_all, f1_all = _compute_sequence_metrics(
            predictions, ground_truth)
        
        # Count present actions for context
        present_actions = np.sum(np.sum(ground_truth, axis=0) > 0)
        total_actions = ground_truth.shape[1]
        
        results = {
            # Different mAP variants - choose the one that fits your needs best
            'mAP_standard': mAP_standard,           # Original (inflated by sparsity)
            'mAP_present_only': mAP_present_only,   # Only actions with positive examples
            'mAP_freq_weighted': mAP_freq_weighted, # Weighted by action frequency
            'mAP_sample_wise': mAP_sample_wise,     # AP per sample, not per action
            
            # Use present_only as main mAP (recommended for sparse data)
            'mAP': mAP_present_only,
            
            # Other metrics
            'exact_match': exact_match_all,
            'hamming_accuracy': hamming_all,
            'precision': precision_all,
            'recall': recall_all,
            'f1': f1_all,
            
            # Metadata
            'num_predictions': len(predictions),
            'num_actions_total': total_actions,
            'num_actions_present': present_actions,
            'action_sparsity': 1 - (present_actions / total_actions),
            'task': 'single_step_action_prediction',
            'method_name': method_name
        }
        
        # Add metrics excluding last N actions if we have enough actions
        if predictions.shape[1] > exclude_last_n:
            preds_subset = predictions[:, :-exclude_last_n]
            gt_subset = ground_truth[:, :-exclude_last_n]
            
            mAP_subset_present = _compute_map_scores(preds_subset, gt_subset, 'present_only')
            results[f'mAP_excluding_last_{exclude_last_n}'] = mAP_subset_present
            
            present_actions_subset = np.sum(np.sum(gt_subset, axis=0) > 0)
            results[f'num_actions_evaluated_subset'] = preds_subset.shape[1]
            results[f'num_actions_present_subset'] = present_actions_subset
        else:
            results[f'mAP_excluding_last_{exclude_last_n}'] = None
            results[f'num_actions_evaluated_subset'] = 0
            results[f'num_actions_present_subset'] = 0
        
        return results
    
    def calculate_planning_stability(self, planning_sequences: np.ndarray) -> float:
        """Calculate planning stability across sequences."""
        if len(planning_sequences) == 0:
            return 0.0
        
        stabilities = []
        for seq in planning_sequences:
            # Measure variance across planning horizon
            action_variance = np.var(seq, axis=0)
            stability = 1.0 / (1.0 + np.mean(action_variance))
            stabilities.append(stability)
        
        return np.mean(stabilities)
    
    def calculate_horizon_performance(self, predictions: np.ndarray, ground_truth: np.ndarray) -> List[float]:
        """Calculate performance at each horizon step."""
        if len(predictions) == 0 or len(ground_truth) == 0:
            return []
        
        horizon = predictions.shape[1]
        horizon_maps = []
        
        for h in range(horizon):
            pred_h = predictions[:, h, :]
            gt_h = ground_truth[:, h, :]
            
            # Calculate mAP for this horizon step using present_only strategy
            ap_scores = []
            for action_idx in range(gt_h.shape[1]):
                gt_column = gt_h[:, action_idx]
                pred_column = pred_h[:, action_idx]
                
                if np.sum(gt_column) > 0:
                    try:
                        ap = average_precision_score(gt_column, pred_column)
                        ap_scores.append(ap)
                    except:
                        ap_scores.append(0.0)
                # Skip absent actions (present_only strategy)
            
            horizon_map = np.mean(ap_scores) if ap_scores else 0.0
            horizon_maps.append(horizon_map)
        
        return horizon_maps
    
    def calculate_sequence_coherence(self, sequences: np.ndarray) -> float:
        """Calculate sequence coherence (for IL models)."""
        coherence_scores = []
        for seq in sequences:
            transitions = np.diff(seq, axis=0)
            smoothness = 1.0 / (1.0 + np.mean(np.abs(transitions)))
            coherence_scores.append(smoothness)
        return np.mean(coherence_scores)
    
    def calculate_long_term_consistency(self, sequences: np.ndarray) -> float:
        """Calculate long-term consistency (for World Model RL)."""
        consistency_scores = []
        for seq in sequences:
            # Measure how consistent actions are over time
            action_consistency = 1.0 - np.std(np.mean(seq, axis=1))
            consistency_scores.append(max(0, action_consistency))
        return np.mean(consistency_scores)
    
    def calculate_immediate_focus(self, sequences: np.ndarray) -> float:
        """Calculate immediate focus (for Direct Video RL)."""
        # For direct video RL, measure how focused on immediate actions
        focus_scores = []
        for seq in sequences:
            # Since it repeats the same action, measure action strength
            first_action = seq[0]
            action_strength = np.max(first_action)
            focus_scores.append(action_strength)
        return np.mean(focus_scores)
    
    def compare_method_metrics(self, method_results: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Compare metrics across different methods.
        
        Args:
            method_results: Dict[method_name, metrics_dict]
            
        Returns:
            Comparison summary with rankings and statistical insights
        """
        
        comparison = {
            'method_ranking': {},
            'metric_summary': {},
            'insights': {}
        }
        
        # Rank methods by main mAP
        method_maps = {method: results.get('mAP', 0.0) 
                      for method, results in method_results.items()}
        
        sorted_methods = sorted(method_maps.items(), key=lambda x: x[1], reverse=True)
        comparison['method_ranking']['by_mAP'] = sorted_methods
        
        # Summary statistics
        all_maps = list(method_maps.values())
        comparison['metric_summary'] = {
            'mean_mAP_across_methods': np.mean(all_maps),
            'std_mAP_across_methods': np.std(all_maps),
            'best_method': sorted_methods[0] if sorted_methods else None,
            'worst_method': sorted_methods[-1] if sorted_methods else None
        }
        
        # Insights
        if len(sorted_methods) >= 2:
            best_score = sorted_methods[0][1]
            second_score = sorted_methods[1][1]
            gap = best_score - second_score
            
            comparison['insights'] = {
                'performance_gap': gap,
                'significant_difference': gap > 0.05,  # 5% threshold
                'evaluation_consistency': 'metrics_calculated_consistently_across_methods'
            }
        
        return comparison
    
    def log_metrics_summary(self, metrics: Dict[str, float], context: str = ""):
        """Log a summary of metrics."""
        if self.logger:
            self.logger.info(f"📊 Metrics Summary {context}:")
            self.logger.info(f"   mAP (present_only): {metrics.get('mAP', 0):.4f}")
            self.logger.info(f"   mAP (standard): {metrics.get('mAP_standard', 0):.4f}")
            self.logger.info(f"   Exact Match: {metrics.get('exact_match', 0):.4f}")
            self.logger.info(f"   Action Sparsity: {metrics.get('action_sparsity', 0):.4f}")
            self.logger.info(f"   Present Actions: {metrics.get('num_actions_present', 0)}/{metrics.get('num_actions_total', 0)}")


# Global instance for easy access
surgical_metrics = SurgicalEvaluationMetrics()


def calculate_comprehensive_action_metrics(predictions: np.ndarray, 
                                         ground_truth: np.ndarray, 
                                         method_name: str = "unknown",
                                         exclude_last_n: int = 6) -> Dict[str, float]:
    """
    Convenience function for calculating comprehensive action metrics.
    
    This is the main function that should be imported and used by both
    training and testing code to ensure consistency.
    """
    return surgical_metrics.calculate_comprehensive_action_metrics(
        predictions, ground_truth, method_name, exclude_last_n
    )


def calculate_planning_metrics(planning_sequences: np.ndarray,
                             ground_truth_sequences: np.ndarray = None) -> Dict[str, float]:
    """Convenience function for planning metrics."""
    
    metrics = {
        'planning_stability': surgical_metrics.calculate_planning_stability(planning_sequences),
        'sequence_coherence': surgical_metrics.calculate_sequence_coherence(planning_sequences)
    }
    
    if ground_truth_sequences is not None:
        metrics['horizon_performance'] = surgical_metrics.calculate_horizon_performance(
            planning_sequences, ground_truth_sequences
        )
    
    return metrics


if __name__ == "__main__":
    print("🔧 SHARED SURGICAL EVALUATION METRICS")
    print("=" * 50)
    print("✅ Comprehensive action prediction metrics")
    print("✅ Handles action sparsity properly")
    print("✅ Multiple mAP calculation strategies")
    print("✅ Planning and sequence evaluation")
    print("✅ Consistent across training and testing")
    print()
    print("📋 Usage:")
    print("   from utils.evaluation_metrics import calculate_comprehensive_action_metrics")
    print("   metrics = calculate_comprehensive_action_metrics(predictions, ground_truth)")
    print()
    print("🎯 This ensures IDENTICAL metrics between training and testing!")
