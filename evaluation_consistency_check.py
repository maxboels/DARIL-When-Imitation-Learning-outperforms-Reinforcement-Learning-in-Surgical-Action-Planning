#!/usr/bin/env python3
"""
Evaluation Consistency Check
Fix the discrepancy between training and test evaluation

Key issues to investigate:
1. Training vs test evaluation methodology differences  
2. Data distribution mismatch
3. Model calibration problems
4. World model vs direct video evaluation differences
"""

import numpy as np
import torch
from pathlib import Path
import json
from sklearn.metrics import average_precision_score
from stable_baselines3 import PPO

# Import your existing components
from datasets.cholect50 import load_cholect50_data
from utils.logger import SimpleLogger


class EvaluationConsistencyChecker:
    """
    Check and fix inconsistencies between training and test evaluation.
    """
    
    def __init__(self, rl_model_path, logger):
        self.rl_model_path = rl_model_path
        self.logger = logger
        self.model = PPO.load(rl_model_path)
        
        self.logger.info("🔍 Evaluation Consistency Checker initialized")
        
    def compare_training_vs_test_evaluation(self, train_data, test_data):
        """Compare evaluation on training data vs test data to identify issues."""
        
        self.logger.info("🔍 Comparing training vs test evaluation...")
        
        # Evaluate on training data (same as used during training)
        train_eval = self._evaluate_model_on_data(train_data[:2], "TRAINING_DATA")
        
        # Evaluate on test data (same as optimization analysis)
        test_eval = self._evaluate_model_on_data(test_data, "TEST_DATA")
        
        comparison = {
            'training_data_evaluation': train_eval,
            'test_data_evaluation': test_eval,
            'performance_gap': {
                'mAP_difference': train_eval['mAP'] - test_eval['mAP'],
                'mAP_ratio': test_eval['mAP'] / train_eval['mAP'] if train_eval['mAP'] > 0 else 0,
                'density_difference': train_eval['avg_action_density'] - test_eval['avg_action_density']
            },
            'diagnosis': []
        }
        
        # Diagnose issues
        mAP_ratio = comparison['performance_gap']['mAP_ratio']
        if mAP_ratio < 0.5:
            comparison['diagnosis'].append("Severe overfitting: Test performance <50% of training")
        elif mAP_ratio < 0.8:
            comparison['diagnosis'].append("Moderate overfitting: Test performance 50-80% of training")
        
        density_diff = abs(comparison['performance_gap']['density_difference'])
        if density_diff > 5:
            comparison['diagnosis'].append("Large action density difference between train/test")
        
        self.logger.info("🔍 Evaluation Comparison:")
        self.logger.info(f"   Training data mAP: {train_eval['mAP']:.4f}")
        self.logger.info(f"   Test data mAP: {test_eval['mAP']:.4f}")
        self.logger.info(f"   Performance ratio: {mAP_ratio:.2f}")
        
        return comparison
    
    def _evaluate_model_on_data(self, data, data_name, max_samples=300):
        """Evaluate model on given data using consistent methodology."""
        
        self.logger.info(f"📊 Evaluating on {data_name}...")
        
        all_predictions = []
        all_expert_actions = []
        action_densities = []
        
        sample_count = 0
        for video in data:
            if sample_count >= max_samples:
                break
                
            frames = video['frame_embeddings']
            expert_actions = video['actions_binaries']
            
            # Sample frames consistently
            num_samples = min(30, len(frames), max_samples - sample_count)
            indices = np.linspace(0, len(frames)-1, num_samples, dtype=int)
            
            for idx in indices:
                if sample_count >= max_samples:
                    break
                    
                try:
                    state = frames[idx].reshape(1, -1)
                    action_pred, _ = self.model.predict(state, deterministic=True)
                    
                    # Process prediction consistently
                    if isinstance(action_pred, np.ndarray):
                        action_pred = action_pred.flatten()
                    
                    if len(action_pred) != 100:
                        padded = np.zeros(100)
                        if len(action_pred) > 0:
                            padded[:min(len(action_pred), 100)] = action_pred[:100]
                        action_pred = padded
                    
                    action_pred = np.clip(action_pred, 0.0, 1.0)
                    
                    all_predictions.append(action_pred)
                    all_expert_actions.append(expert_actions[idx])
                    
                    # Track action density at different thresholds
                    action_densities.append({
                        'threshold_0.5': np.sum(action_pred > 0.5),
                        'threshold_0.3': np.sum(action_pred > 0.3),
                        'expert': np.sum(expert_actions[idx] > 0.5)
                    })
                    
                    sample_count += 1
                    
                except Exception as e:
                    self.logger.warning(f"Prediction failed: {e}")
                    continue
        
        if not all_predictions:
            return {'mAP': 0.0, 'avg_action_density': 0.0, 'samples': 0}
        
        # Calculate mAP consistently
        predictions = np.array(all_predictions)
        expert_actions = np.array(all_expert_actions)
        
        ap_scores = []
        for action_idx in range(100):
            if np.sum(expert_actions[:, action_idx]) > 0:
                try:
                    ap = average_precision_score(
                        expert_actions[:, action_idx], 
                        predictions[:, action_idx]
                    )
                    ap_scores.append(ap)
                except:
                    ap_scores.append(0.0)
        
        mAP = np.mean(ap_scores) if ap_scores else 0.0
        
        # Calculate average action densities
        avg_density_05 = np.mean([d['threshold_0.5'] for d in action_densities])
        avg_density_03 = np.mean([d['threshold_0.3'] for d in action_densities])
        avg_expert_density = np.mean([d['expert'] for d in action_densities])
        
        return {
            'mAP': mAP,
            'avg_action_density': avg_density_05,
            'avg_action_density_0.3': avg_density_03,
            'avg_expert_density': avg_expert_density,
            'samples': len(predictions),
            'prediction_stats': {
                'mean': float(np.mean(predictions)),
                'std': float(np.std(predictions)),
                'median': float(np.median(predictions)),
                'max': float(np.max(predictions))
            }
        }
    
    def diagnose_model_calibration(self, test_data):
        """Diagnose model calibration issues."""
        
        self.logger.info("🔍 Diagnosing model calibration...")
        
        # Collect predictions and analyze distribution
        all_predictions = []
        
        for video in test_data[:3]:  # Use first 3 videos
            frames = video['frame_embeddings']
            
            # Sample 20 frames per video
            indices = np.linspace(0, len(frames)-1, 20, dtype=int)
            
            for idx in indices:
                try:
                    state = frames[idx].reshape(1, -1)
                    action_pred, _ = self.model.predict(state, deterministic=True)
                    
                    if isinstance(action_pred, np.ndarray):
                        action_pred = action_pred.flatten()[:100]
                    
                    all_predictions.extend(action_pred)
                    
                except Exception as e:
                    continue
        
        if not all_predictions:
            return {'status': 'failed'}
        
        predictions = np.array(all_predictions)
        
        # Analyze prediction distribution
        calibration_analysis = {
            'total_predictions': len(predictions),
            'distribution': {
                'zeros': float(np.sum(predictions == 0.0)),
                'near_zero': float(np.sum(predictions < 0.01)),
                'low': float(np.sum((predictions >= 0.01) & (predictions < 0.1))),
                'medium': float(np.sum((predictions >= 0.1) & (predictions < 0.5))),
                'high': float(np.sum(predictions >= 0.5))
            },
            'statistics': {
                'mean': float(np.mean(predictions)),
                'std': float(np.std(predictions)),
                'median': float(np.median(predictions)),
                'q95': float(np.percentile(predictions, 95)),
                'max': float(np.max(predictions))
            },
            'calibration_issues': []
        }
        
        # Identify calibration issues
        zero_ratio = calibration_analysis['distribution']['zeros'] / len(predictions)
        if zero_ratio > 0.5:
            calibration_analysis['calibration_issues'].append(
                f"Model outputs too many zeros: {zero_ratio:.1%}"
            )
        
        near_zero_ratio = calibration_analysis['distribution']['near_zero'] / len(predictions)
        if near_zero_ratio > 0.8:
            calibration_analysis['calibration_issues'].append(
                f"Model is too conservative: {near_zero_ratio:.1%} predictions near zero"
            )
        
        if calibration_analysis['statistics']['max'] < 0.8:
            calibration_analysis['calibration_issues'].append(
                f"Model never confident: max prediction only {calibration_analysis['statistics']['max']:.3f}"
            )
        
        self.logger.info("🔍 Calibration Analysis:")
        self.logger.info(f"   Zero predictions: {zero_ratio:.1%}")
        self.logger.info(f"   Near-zero predictions: {near_zero_ratio:.1%}")
        self.logger.info(f"   Max confidence: {calibration_analysis['statistics']['max']:.3f}")
        
        return calibration_analysis
    
    def suggest_fixes(self, evaluation_comparison, calibration_analysis):
        """Suggest specific fixes based on analysis."""
        
        fixes = {
            'immediate_actions': [],
            'training_adjustments': [],
            'evaluation_fixes': [],
            'model_improvements': []
        }
        
        # Based on evaluation comparison
        mAP_ratio = evaluation_comparison['performance_gap']['mAP_ratio']
        
        if mAP_ratio < 0.5:
            fixes['immediate_actions'].append("Severe overfitting detected - reduce model complexity")
            fixes['training_adjustments'].extend([
                "Add stronger regularization (dropout, weight decay)",
                "Reduce training episodes/timesteps",
                "Use early stopping based on validation performance"
            ])
        
        # Based on calibration analysis
        if 'calibration_issues' in calibration_analysis:
            for issue in calibration_analysis['calibration_issues']:
                if "too many zeros" in issue or "too conservative" in issue:
                    fixes['model_improvements'].extend([
                        "Retrain with different initialization",
                        "Use behavioral cloning warm-start to improve calibration",
                        "Adjust reward function to encourage more confident predictions"
                    ])
                
                if "never confident" in issue:
                    fixes['training_adjustments'].extend([
                        "Increase exploration during training",
                        "Use different activation functions (e.g., sigmoid instead of tanh)",
                        "Scale rewards to encourage stronger signals"
                    ])
        
        # Evaluation methodology fixes
        fixes['evaluation_fixes'] = [
            "Use identical evaluation methodology for training and testing",
            "Ensure same data preprocessing and thresholding",
            "Cross-validate results on multiple data splits",
            "Compare against supervised baseline on same test set"
        ]
        
        return fixes
    
    def save_analysis(self, save_dir, evaluation_comparison, calibration_analysis, suggested_fixes):
        """Save comprehensive analysis results."""
        
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        full_analysis = {
            'evaluation_comparison': evaluation_comparison,
            'calibration_analysis': calibration_analysis,
            'suggested_fixes': suggested_fixes,
            'summary': {
                'training_vs_test_mAP_ratio': evaluation_comparison['performance_gap']['mAP_ratio'],
                'main_issues': [],
                'priority_fixes': []
            }
        }
        
        # Summarize main issues
        if evaluation_comparison['performance_gap']['mAP_ratio'] < 0.5:
            full_analysis['summary']['main_issues'].append("Severe overfitting")
        
        if calibration_analysis.get('calibration_issues'):
            full_analysis['summary']['main_issues'].extend(calibration_analysis['calibration_issues'])
        
        # Priority fixes
        full_analysis['summary']['priority_fixes'] = suggested_fixes['immediate_actions'][:3]
        
        # Save results
        with open(save_path / 'evaluation_consistency_analysis.json', 'w') as f:
            json.dump(full_analysis, f, indent=2, default=str)
        
        # Human-readable summary
        with open(save_path / 'evaluation_fix_summary.txt', 'w') as f:
            f.write("EVALUATION CONSISTENCY ANALYSIS\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("PERFORMANCE COMPARISON:\n")
            f.write(f"Training data mAP: {evaluation_comparison['training_data_evaluation']['mAP']:.4f}\n")
            f.write(f"Test data mAP: {evaluation_comparison['test_data_evaluation']['mAP']:.4f}\n")
            f.write(f"Performance ratio: {evaluation_comparison['performance_gap']['mAP_ratio']:.2f}\n\n")
            
            f.write("MAIN ISSUES:\n")
            for issue in full_analysis['summary']['main_issues']:
                f.write(f"- {issue}\n")
            
            f.write(f"\nPRIORITY FIXES:\n")
            for fix in full_analysis['summary']['priority_fixes']:
                f.write(f"- {fix}\n")
        
        self.logger.info(f"💾 Analysis saved to {save_path}")


def main():
    """Main consistency checking pipeline."""
    
    print("🔍 EVALUATION CONSISTENCY CHECK")
    print("=" * 50)
    print("Goal: Fix the 3x mAP discrepancy between training and test")
    print()
    
    # Configuration
    config = {
        'data': {
            'paths': {
                'data_dir': "/home/maxboels/datasets/CholecT50",
                'fold': 0,
                'metadata_file': "embeddings_f0_swin_bas_129_phase_complet_phase_transit_prog_prob_action_risk_glob_outcome.csv"
            }
        }
    }
    
    # Initialize logger
    logger = SimpleLogger(log_dir="evaluation_consistency", name="EvalChecker")
    
    try:
        # Load data
        print("📂 Loading data...")
        train_data = load_cholect50_data(
            config, logger, split='train', max_videos=5  # Just first 5 for speed
        )
        test_data = load_cholect50_data(
            config, logger, split='test', max_videos=10
        )
        
        # Initialize checker
        rl_model_path = "results/fixed_rl_training_continued/fixed_simplified_ppo.zip"
        checker = EvaluationConsistencyChecker(rl_model_path, logger)
        
        # Step 1: Compare training vs test evaluation
        print("\n🔍 Step 1: Comparing training vs test evaluation...")
        evaluation_comparison = checker.compare_training_vs_test_evaluation(train_data, test_data)
        
        # Step 2: Diagnose model calibration
        print("\n🔍 Step 2: Diagnosing model calibration...")
        calibration_analysis = checker.diagnose_model_calibration(test_data)
        
        # Step 3: Suggest fixes
        print("\n💡 Step 3: Generating fix recommendations...")
        suggested_fixes = checker.suggest_fixes(evaluation_comparison, calibration_analysis)
        
        # Step 4: Save analysis
        print("\n💾 Step 4: Saving analysis...")
        checker.save_analysis("evaluation_consistency_results", 
                            evaluation_comparison, calibration_analysis, suggested_fixes)
        
        # Summary
        print("\n🎉 CONSISTENCY CHECK COMPLETE!")
        print("=" * 50)
        
        mAP_ratio = evaluation_comparison['performance_gap']['mAP_ratio']
        print(f"📊 Performance ratio (test/train): {mAP_ratio:.2f}")
        
        if mAP_ratio < 0.5:
            print("🚨 SEVERE OVERFITTING DETECTED")
        elif mAP_ratio < 0.8:
            print("⚠️ Moderate overfitting")
        else:
            print("✅ Reasonable generalization")
        
        print("\nTop Priority Fixes:")
        for fix in suggested_fixes['immediate_actions'][:3]:
            print(f"- {fix}")
        
        print(f"\n📁 Full analysis saved to: evaluation_consistency_results/")
        
    except Exception as e:
        print(f"❌ Consistency check failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
