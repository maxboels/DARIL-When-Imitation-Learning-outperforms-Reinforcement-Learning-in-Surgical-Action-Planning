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
                                             exclude_last_n: int = 6) -> Dict[str, float]:
        """
        Calculate comprehensive single-step action prediction metrics.
        
        🔧 UPDATED: Main mAP is now calculated from present actions only, 
        excluding the last 6 classes (no action classes).
        
        This is the function used by both training and testing stages.
        
        Args:
            predictions: [num_samples, num_actions] - action probabilities
            ground_truth: [num_samples, num_actions] - binary ground truth
            method_name: Name of the method for logging
            exclude_last_n: Number of last actions to exclude (default: 6 for no-action classes)
            
        Returns:
            Dictionary with comprehensive metrics where 'mAP' is present actions only from subset
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
        
        # 🔧 MAIN mAP CALCULATION: Present actions only from subset excluding last N
        main_mAP = 0.0
        main_mAP_info = {
            'calculated_from': 'subset_excluding_last_classes_present_actions_only',
            'excluded_classes': exclude_last_n,
            'subset_size': 0,
            'present_actions_in_subset': 0
        }
        
        if predictions.shape[1] > exclude_last_n:
            # Create subset excluding last N actions (no-action classes)
            preds_subset = predictions[:, :-exclude_last_n]
            gt_subset = ground_truth[:, :-exclude_last_n]
            
            # Calculate mAP on present actions only from this subset
            main_mAP = _compute_map_scores(preds_subset, gt_subset, 'present_only')
            
            # Update info
            present_actions_subset = np.sum(np.sum(gt_subset, axis=0) > 0)
            main_mAP_info.update({
                'subset_size': preds_subset.shape[1],
                'present_actions_in_subset': present_actions_subset,
                'subset_action_sparsity': 1 - (present_actions_subset / preds_subset.shape[1]) if preds_subset.shape[1] > 0 else 1.0
            })
        else:
            # Fallback: if we don't have enough actions, use present_only on all actions
            main_mAP = _compute_map_scores(predictions, ground_truth, 'present_only')
            main_mAP_info.update({
                'calculated_from': 'all_actions_present_only_fallback',
                'reason': f'insufficient_actions_for_exclusion_({predictions.shape[1]}_<=_{exclude_last_n})'
            })
        
        # 📊 ALTERNATIVE mAP calculations for comparison
        mAP_standard_all = _compute_map_scores(predictions, ground_truth, 'standard')
        mAP_present_only_all = _compute_map_scores(predictions, ground_truth, 'present_only')
        mAP_freq_weighted_all = _compute_map_scores(predictions, ground_truth, 'frequency_weighted')
        mAP_sample_wise_all = _compute_map_scores(predictions, ground_truth, 'sample_wise')
        
        # Other metrics on all actions
        exact_match_all, hamming_all, precision_all, recall_all, f1_all = _compute_sequence_metrics(
            predictions, ground_truth)
        
        # Count present actions for context (on all actions)
        present_actions_all = np.sum(np.sum(ground_truth, axis=0) > 0)
        total_actions = ground_truth.shape[1]
        
        results = {
            # 🎯 MAIN mAP: Present actions only from subset excluding last N classes (NORMAL)
            'mAP': main_mAP,
            'mAP_info': main_mAP_info,
            
            # Alternative mAP variants for comparison (calculated on all actions INCLUDING null verbs)
            'mAP_standard_with_null_verb': mAP_standard_all,           # Original (inflated by sparsity)
            'mAP_present_only_with_null_verb': mAP_present_only_all,   # Only actions with positive examples
            'mAP_freq_weighted_with_null_verb': mAP_freq_weighted_all, # Weighted by action frequency
            'mAP_sample_wise_with_null_verb': mAP_sample_wise_all,     # AP per sample, not per action
            
            # Legacy aliases for backwards compatibility (will be deprecated)
            'mAP_standard_all_actions': mAP_standard_all,
            'mAP_present_only_all_actions': mAP_present_only_all,
            'mAP_freq_weighted_all_actions': mAP_freq_weighted_all,
            'mAP_sample_wise_all_actions': mAP_sample_wise_all,
            
            # Other metrics (calculated on all actions INCLUDING null verbs)
            'exact_match_with_null_verb': exact_match_all,
            'hamming_accuracy_with_null_verb': hamming_all,
            'precision_with_null_verb': precision_all,
            'recall_with_null_verb': recall_all,
            'f1_with_null_verb': f1_all,
            
            # Metadata (all actions including null verbs)
            'num_predictions': len(predictions),
            'num_actions_total_with_null_verb': total_actions,
            'num_actions_present_with_null_verb': present_actions_all,
            'action_sparsity_with_null_verb': 1 - (present_actions_all / total_actions),
            'task': 'single_step_action_prediction',
            'method_name': method_name,
            'exclude_last_n': exclude_last_n
        }
        
        # 🔧 SUBSET METRICS: Add comprehensive subset analysis (NORMAL metrics - no prefix)
        if predictions.shape[1] > exclude_last_n:
            preds_subset = predictions[:, :-exclude_last_n]
            gt_subset = ground_truth[:, :-exclude_last_n]
            
            # All mAP variants on subset (NORMAL - no prefix)
            mAP_subset_standard = _compute_map_scores(preds_subset, gt_subset, 'standard')
            mAP_subset_present = _compute_map_scores(preds_subset, gt_subset, 'present_only')
            mAP_subset_freq_weighted = _compute_map_scores(preds_subset, gt_subset, 'frequency_weighted')
            
            # Other metrics on subset (NORMAL - no prefix)
            exact_match_subset, hamming_subset, precision_subset, recall_subset, f1_subset = _compute_sequence_metrics(
                preds_subset, gt_subset)
            
            present_actions_subset = np.sum(np.sum(gt_subset, axis=0) > 0)
            
            # Add subset metrics as NORMAL metrics (clean names)
            subset_metrics = {
                'mAP_standard': mAP_subset_standard,
                'mAP_present_only': mAP_subset_present,  # Same as main mAP but explicit
                'mAP_freq_weighted': mAP_subset_freq_weighted,
                'exact_match': exact_match_subset,
                'hamming_accuracy': hamming_subset,
                'precision': precision_subset,
                'recall': recall_subset,
                'f1': f1_subset,
                'num_actions_total': preds_subset.shape[1],
                'num_actions_present': present_actions_subset,
                'action_sparsity': 1 - (present_actions_subset / preds_subset.shape[1]) if preds_subset.shape[1] > 0 else 1.0
            }
            
            results.update(subset_metrics)
        else:
            # Add placeholder values when subset is not available
            results.update({
                'mAP_standard': None,
                'mAP_present_only': None,
                'mAP_freq_weighted': None,
                'exact_match': None,
                'hamming_accuracy': None,
                'precision': None,
                'recall': None,
                'f1': None,
                'num_actions_total': 0,
                'num_actions_present': 0,
                'action_sparsity': 1.0
            })
        
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
        
        # Rank methods by main mAP (now present actions only from subset)
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
            'worst_method': sorted_methods[-1] if sorted_methods else None,
            'mAP_calculation': 'present_actions_only_excluding_last_6_classes'
        }
        
        # Insights
        if len(sorted_methods) >= 2:
            best_score = sorted_methods[0][1]
            second_score = sorted_methods[1][1]
            gap = best_score - second_score
            
            comparison['insights'] = {
                'performance_gap': gap,
                'significant_difference': gap > 0.05,  # 5% threshold
                'evaluation_consistency': 'metrics_calculated_consistently_across_methods',
                'main_metric_type': 'present_actions_only_from_action_subset'
            }
        
        return comparison
    
    def log_metrics_summary(self, metrics: Dict[str, float], context: str = ""):
        """Log a summary of metrics with focus on main mAP calculation."""
        if self.logger:
            main_mAP = metrics.get('mAP', 0)
            mAP_info = metrics.get('mAP_info', {})
            
            self.logger.info(f"📊 Metrics Summary {context}:")
            self.logger.info(f"   🎯 Main mAP: {main_mAP:.4f} ({mAP_info.get('calculated_from', 'unknown')})")
            
            if 'subset_size' in mAP_info:
                excluded = mAP_info.get('excluded_classes', 6)
                self.logger.info(f"   📋 Evaluated: {mAP_info['present_actions_in_subset']}/{mAP_info['subset_size']} surgical actions (excluded {excluded} null verb classes)")
            
            self.logger.info(f"   📊 Alternatives:")
            self.logger.info(f"     Standard (surgical actions): {metrics.get('mAP_standard', 0):.4f}")
            self.logger.info(f"     With null verbs: {metrics.get('mAP_present_only_with_null_verb', 0):.4f}")
            
            # Show normal metrics (subset) vs with null verbs
            subset_sparsity = metrics.get('action_sparsity', 0)
            with_null_sparsity = metrics.get('action_sparsity_with_null_verb', 0)
            self.logger.info(f"   📈 Sparsity: Surgical actions={subset_sparsity:.3f}, With null verbs={with_null_sparsity:.3f}")
            self.logger.info(f"   📈 Exact Match: {metrics.get('exact_match', 0):.4f} (surgical actions only)")


# Global instance for easy access
surgical_metrics = SurgicalEvaluationMetrics()


def calculate_comprehensive_action_metrics(predictions: np.ndarray, 
                                         ground_truth: np.ndarray, 
                                         method_name: str = "unknown",
                                         exclude_last_n: int = 6) -> Dict[str, float]:
    """
    🔧 UPDATED: Convenience function for calculating comprehensive action metrics.
    
    🎯 NAMING CONVENTION:
    - Main metrics (no prefix): Surgical actions only (excluding last 6 null verb classes) 
    - "_with_null_verb" suffix: All actions including null verb classes
    
    Main mAP is calculated from present surgical actions only, excluding null verb classes.
    
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
    print("🔧 UPDATED SHARED SURGICAL EVALUATION METRICS")
    print("=" * 60)
    print("✅ Comprehensive action prediction metrics")
    print("✅ Handles action sparsity properly")
    print("✅ Multiple mAP calculation strategies")
    print("✅ Planning and sequence evaluation")
    print("✅ Consistent across training and testing")
    print("🎯 MAIN mAP: Present actions only from subset excluding last 6 classes")
    print()
    print("📋 Usage:")
    print("   from evaluation.evaluation_metrics import calculate_comprehensive_action_metrics")
    print("   metrics = calculate_comprehensive_action_metrics(predictions, ground_truth)")
    print("   main_mAP = metrics['mAP']  # Present actions only, excluding last 6 classes")
    print()
    print("🎯 This ensures IDENTICAL metrics between training and testing!")
    print("📊 Main mAP focuses on actual surgical actions (excludes no-action classes)")