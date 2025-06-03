#!/usr/bin/env python3
"""
Complete IL vs RL Comparison Script for Surgical Action Prediction - FIXED
This script implements a comprehensive comparison between:
1. Imitation Learning (Supervised approach) for Expert Behavior Cloning
2. Reinforcement Learning (PPO and SAC with world model) for Autonomous Exploration
"""
 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
import os
import glob
import json
import pandas as pd
import re
from tqdm import tqdm
import yaml
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging

# Import your existing modules
from datasets.cholect50 import load_cholect50_data, NextFramePredictionDataset, create_video_dataloaders
from utils.logger import SimpleLogger

# Import model components
from models.dual_world_model import DualWorldModel
from trainer.dual_trainer import DualTrainer, train_dual_world_model
from evaluation.dual_evaluator import DualModelEvaluator

# New components for RL training
from trainer.specific_rl_improvements import (
    OutcomeBasedRewardFunction,
    FairEvaluationMetrics,
    ImprovedRLEnvironment
)

# Suppress warnings from sklearn
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ComparisonExperiment:
    """
    Main class for conducting IL vs RL comparison experiment.
    """
    
    def __init__(self, config_path: str = 'config.yaml'):
        """Initialize the comparison experiment."""
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup logging
        self.logger = SimpleLogger(log_dir="logs", name="il_vs_rl_comparison")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Results storage
        self.results = {
            'il_results': None,
            'rl_results': {},
            'comparison_results': None,
            'model_paths': {},
            'config': self.config
        }
        
        # Create results directory
        self.results_dir = Path(self.logger.log_dir) / 'comparison_results'
        self.results_dir.mkdir(exist_ok=True)
        
        self.logger.info("🚀 Starting IL vs RL Comparison Experiment")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Results will be saved to: {self.results_dir}")
    
    def run_complete_comparison(self) -> Dict[str, Any]:
        """Run the complete IL vs RL comparison."""
        
        try:
            # Step 1: Load data
            self.logger.info("📊 Loading dataset...")
            train_data, test_data = self._load_data()
            
            # Step 2: Train Imitation Learning model
            if self.config.get('experiment', {}).get('il_experiments', {}).get('enabled', True):
                self.logger.info("🎓 Training Imitation Learning Model...")
                il_model_path = self._train_imitation_learning(train_data, test_data)
                self.results['model_paths']['imitation_learning'] = il_model_path
            else:
                self.logger.warning("⚠️ Imitation Learning experiments are disabled in config")
                il_model_path = None
                self.results['model_paths']['imitation_learning'] = None
            
            # Step 3: Train RL models
            if self.config.get('experiment', {}).get('rl_experiments', {}).get('enabled', False):
                self.logger.info("🤖 Training RL Models...")
                rl_results = self._train_rl_models(train_data, il_model_path)
                self.results['rl_results'] = rl_results
            else:
                self.logger.warning("⚠️ RL experiments are disabled in config")
                self.results['rl_results'] = {}
            
            # Step 4: Comprehensive evaluation - FIXED
            self.logger.info("📈 Running comprehensive evaluation...")
            evaluation_results = self._run_comprehensive_evaluation(test_data)
            self.results['evaluation_results'] = evaluation_results
            
            # Step 5: Statistical comparison
            self.logger.info("🔬 Performing statistical analysis...")
            comparison_results = self._perform_statistical_comparison()
            self.results['comparison_results'] = comparison_results
            
            # Step 6: Generate reports and visualizations
            self.logger.info("📝 Generating reports and visualizations...")
            self._generate_final_report()
            
            # Step 7: Save everything
            self._save_complete_results()
            
            self.logger.info("✅ Complete IL vs RL comparison finished successfully!")
            return self.results
            
        except Exception as e:
            self.logger.error(f"❌ Comparison experiment failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'error': str(e), 'partial_results': self.results}
    
    def _load_data(self) -> Tuple[List[Dict], List[Dict]]:
        """Load and prepare training and test data."""
        
        # Load training data
        train_videos = self.config.get('experiment', {}).get('train', {}).get('max_videos', 20)
        train_data = load_cholect50_data(
            self.config, self.logger, split='train', max_videos=train_videos
        )
        
        # Load test data
        test_videos = self.config.get('experiment', {}).get('test', {}).get('max_videos', 10)
        test_data = load_cholect50_data(
            self.config, self.logger, split='test', max_videos=test_videos
        )
        
        self.logger.info(f"Loaded {len(train_data)} training videos and {len(test_data)} test videos")
        return train_data, test_data
    
    def _train_imitation_learning(self, train_data: List[Dict], test_data: List[Dict]) -> str:
        """Train the imitation learning model."""
        
        self.logger.info("Training supervised imitation learning model")
        
        # Create datasets and dataloaders
        train_dataset = NextFramePredictionDataset(self.config['data'], train_data)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=self.config['training']['num_workers'],
            pin_memory=self.config['training']['pin_memory']
        )
        
        test_video_loaders = create_video_dataloaders(
            self.config, test_data, batch_size=16, shuffle=False
        )
        
        # Initialize model
        model_config = self.config['models']['dual_world_model']
        model = DualWorldModel(**model_config).to(self.device)
        
        # Log model parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        # Train model in supervised mode
        original_mode = self.config.get('training_mode', 'supervised')
        self.config['training_mode'] = 'supervised'
        
        il_model_path = train_dual_world_model(
            self.config, self.logger, model, train_loader, test_video_loaders, self.device
        )
        
        # Restore original mode
        self.config['training_mode'] = original_mode
        
        self.logger.info(f"✅ IL training completed. Model saved: {il_model_path}")
        return il_model_path

    def _train_rl_models(self, train_data, world_model_path):
        """Train RL models using Stable-Baselines3."""
        # Import RL training components
        from trainer.sb3_rl_trainer import SB3Trainer # Stable Baselines3 trainer
        
        world_model = DualWorldModel.load_model(world_model_path, self.device)
        
        # Create SB3 trainer
        sb3_trainer = SB3Trainer(world_model, self.config, self.logger, self.device)
        
        rl_results = {}
        algorithms = ['ppo', 'dqn']  # Reliable algorithms
        timesteps = 10000
        
        for algorithm in algorithms:
            if algorithm == 'ppo':
                rl_results['ppo'] = sb3_trainer.train_ppo(train_data, timesteps)
            elif algorithm == 'dqn':
                rl_results['dqn'] = sb3_trainer.train_dqn(train_data, timesteps)
        
        return rl_results

    # FIXED: Add missing methods for evaluation
    def _load_il_model(self):
        """Load the trained IL model."""
        if 'imitation_learning' in self.results['model_paths']:
            il_model_path = self.results['model_paths']['imitation_learning']
            if il_model_path and os.path.exists(il_model_path):
                return DualWorldModel.load_model(il_model_path, self.device)
        return None
    
    def _load_rl_models(self) -> Dict:
        """Load the trained RL models."""
        rl_models = {}
        
        # Load each RL model if it exists
        for algorithm, result in self.results['rl_results'].items():
            if result.get('status') == 'success' and 'model_path' in result:
                model_path = result['model_path']
                if os.path.exists(model_path):
                    try:
                        # Import SB3 models
                        from stable_baselines3 import PPO, DQN, A2C
                        
                        if 'ppo' in algorithm.lower():
                            model = PPO.load(model_path)
                        elif 'dqn' in algorithm.lower():
                            model = DQN.load(model_path)
                        elif 'a2c' in algorithm.lower():
                            model = A2C.load(model_path)
                        else:
                            continue
                        
                        rl_models[algorithm] = model
                    except Exception as e:
                        self.logger.warning(f"Failed to load {algorithm} model: {e}")
        
        return rl_models
    
    def _load_world_model(self):
        """Load the world model."""
        if 'imitation_learning' in self.results['model_paths']:
            world_model_path = self.results['model_paths']['imitation_learning']
            if world_model_path and os.path.exists(world_model_path):
                return DualWorldModel.load_model(world_model_path, self.device)
        return None

    def _run_comprehensive_evaluation(self, test_data):
        """Run comprehensive evaluation with proper error handling."""
        
        evaluation_results = {}
        
        # 1. Evaluate IL model if available
        il_model = self._load_il_model()
        if il_model is not None:
            self.logger.info("📊 Evaluating Imitation Learning model...")
            il_eval_results = self._evaluate_il_model(il_model, test_data)
            evaluation_results['imitation_learning'] = il_eval_results
        else:
            self.logger.warning("⚠️ IL model not available for evaluation")
            evaluation_results['imitation_learning'] = {'status': 'not_available'}
        
        # 2. Evaluate RL models if available
        rl_models = self._load_rl_models()
        if rl_models:
            for algorithm, rl_model in rl_models.items():
                self.logger.info(f"📊 Evaluating {algorithm.upper()} model...")
                try:
                    rl_eval_results = self._evaluate_rl_model(rl_model, test_data, algorithm)
                    evaluation_results[algorithm] = rl_eval_results
                except Exception as e:
                    self.logger.error(f"❌ {algorithm.upper()} evaluation failed: {e}")
                    evaluation_results[algorithm] = {'status': 'failed', 'error': str(e)}
        else:
            self.logger.warning("⚠️ No RL models available for evaluation")
        
        # 3. Comparison analysis
        if len(evaluation_results) > 1:
            comparison_analysis = self._create_comparison_analysis(evaluation_results)
            evaluation_results['comparison_analysis'] = comparison_analysis
        
        return evaluation_results
    
    def _evaluate_il_model(self, il_model, test_data) -> Dict[str, Any]:
        """Evaluate IL model with both traditional and clinical metrics."""
        
        # Create test dataloaders
        test_video_loaders = create_video_dataloaders(
            self.config, test_data, batch_size=16, shuffle=False
        )
        
        # Create evaluator
        evaluator = DualModelEvaluator(il_model, self.config, self.device, self.logger)
        
        # Evaluate
        il_eval_results = evaluator.evaluate_both_modes(
            test_video_loaders,
            save_results=True,
            save_dir=self.results_dir / "il_evaluation"
        )
        
        return il_eval_results
    
    def _evaluate_rl_model(self, rl_model, test_data: List[Dict], algorithm: str) -> Dict[str, Any]:
        """Evaluate RL model on test data."""
        
        # Create simple RL evaluation
        from trainer.sb3_rl_trainer import SurgicalActionEnv
        
        # Create environment for evaluation
        env = SurgicalActionEnv(
            world_model=self._load_world_model(),
            video_data=test_data,
            config=self.config.get('rl_training', {}),
            device=self.device
        )
        
        # Run evaluation episodes
        n_eval_episodes = 5  # Small number for quick evaluation
        episode_rewards = []
        
        for episode in range(n_eval_episodes):
            obs, _ = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action, _ = rl_model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                episode_reward += reward
                
                if truncated:
                    break
            
            episode_rewards.append(episode_reward)
        
        return {
            'algorithm': algorithm,
            'mean_reward': float(np.mean(episode_rewards)),
            'std_reward': float(np.std(episode_rewards)),
            'min_reward': float(np.min(episode_rewards)),
            'max_reward': float(np.max(episode_rewards)),
            'n_episodes': n_eval_episodes,
            'status': 'success'
        }
    
    def _create_comparison_analysis(self, evaluation_results: Dict) -> Dict[str, Any]:
        """Create comparison analysis between IL and RL results."""
        
        analysis = {
            'methods_evaluated': list(evaluation_results.keys()),
            'il_performance': {},
            'rl_performance': {},
            'comparison_summary': {}
        }
        
        # Extract IL performance
        if 'imitation_learning' in evaluation_results:
            il_results = evaluation_results['imitation_learning']
            if 'supervised' in il_results:
                sup_results = il_results['supervised']
                if 'action_prediction' in sup_results:
                    mAP = sup_results['action_prediction'].get('single_step_action_exact_match', {}).get('mean', 0)
                    analysis['il_performance'] = {
                        'mAP': mAP,
                        'metric_type': 'action_matching',
                        'status': 'success'
                    }
        
        # Extract RL performance
        rl_results = {}
        for method, results in evaluation_results.items():
            if method != 'imitation_learning' and method != 'comparison_analysis':
                if results.get('status') == 'success':
                    rl_results[method] = {
                        'mean_reward': results.get('mean_reward', 0),
                        'std_reward': results.get('std_reward', 0),
                        'metric_type': 'episode_reward'
                    }
        
        analysis['rl_performance'] = rl_results
        
        # Create summary
        if analysis['il_performance'] and rl_results:
            il_score = analysis['il_performance'].get('mAP', 0)
            best_rl = max(rl_results.values(), key=lambda x: x['mean_reward']) if rl_results else None
            
            if best_rl:
                analysis['comparison_summary'] = {
                    'il_best_score': il_score,
                    'rl_best_score': best_rl['mean_reward'],
                    'rl_best_method': [k for k, v in rl_results.items() if v == best_rl][0],
                    'note': 'Different metrics - direct comparison requires outcome-based evaluation'
                }
        
        return analysis
    
    def _perform_statistical_comparison(self) -> Dict[str, Any]:
        """Perform statistical comparison between methods."""
        
        comparison_results = {
            'methods_compared': [],
            'primary_metric': 'mAP',
            'statistical_tests': {},
            'rankings': {},
            'summary': {}
        }
        
        # Extract performance metrics from evaluation results
        evaluation_results = self.results.get('evaluation_results', {})
        
        methods_performance = {}
        
        # IL performance
        if 'imitation_learning' in evaluation_results:
            il_results = evaluation_results['imitation_learning']
            if 'supervised' in il_results and 'action_prediction' in il_results['supervised']:
                il_map = il_results['supervised']['action_prediction'].get('single_step_action_exact_match', {}).get('mean', 0)
                methods_performance['Imitation Learning'] = il_map
        
        # RL performance - convert rewards to comparable scores
        for algorithm in ['ppo', 'dqn', 'a2c']:
            if algorithm in evaluation_results:
                rl_results = evaluation_results[algorithm]
                if rl_results.get('status') == 'success':
                    rl_reward = rl_results.get('mean_reward', 0)
                    # Normalize reward to [0,1] range for comparison
                    normalized_score = min(max(rl_reward / 100.0, 0), 1)  # Assume max reward ~100
                    methods_performance[f'{algorithm.upper()} (RL)'] = normalized_score
        
        # Perform comparisons
        comparison_results['methods_compared'] = list(methods_performance.keys())
        comparison_results['performance_scores'] = methods_performance
        
        # Rank methods
        if methods_performance:
            ranked_methods = sorted(methods_performance.items(), key=lambda x: x[1], reverse=True)
            comparison_results['rankings'] = {
                'ranking': [method for method, score in ranked_methods],
                'scores': [score for method, score in ranked_methods]
            }
            
            # Generate summary
            best_method, best_score = ranked_methods[0]
            comparison_results['summary'] = {
                'best_method': best_method,
                'best_score': best_score,
                'total_methods': len(ranked_methods),
                'evaluation_note': 'Comparison uses different metrics - outcome-based evaluation needed for fair comparison'
            }
        
        return comparison_results
    
    def _generate_final_report(self):
        """Generate final comparison report."""
        
        report_content = []
        
        # Header
        report_content.append("# Imitation Learning vs Reinforcement Learning Comparison")
        report_content.append("## Surgical Action Prediction on CholecT50 Dataset")
        report_content.append(f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_content.append("")
        
        # Executive Summary
        report_content.append("## Executive Summary")
        
        comparison_results = self.results.get('comparison_results', {})
        if 'summary' in comparison_results:
            summary = comparison_results['summary']
            report_content.append(f"- **Best performing method:** {summary.get('best_method', 'N/A')}")
            report_content.append(f"- **Best score:** {summary.get('best_score', 0):.4f}")
            report_content.append(f"- **Methods compared:** {summary.get('total_methods', 0)}")
            if 'evaluation_note' in summary:
                report_content.append(f"- **Note:** {summary['evaluation_note']}")
        
        report_content.append("")
        
        # Detailed Results
        report_content.append("## Detailed Results")
        
        # IL Results
        if 'imitation_learning' in self.results.get('evaluation_results', {}):
            report_content.append("### Imitation Learning")
            il_results = self.results['evaluation_results']['imitation_learning']
            if 'supervised' in il_results:
                sup_results = il_results['supervised']
                if 'action_prediction' in sup_results:
                    action_acc = sup_results['action_prediction'].get('single_step_action_exact_match', {}).get('mean', 0)
                    report_content.append(f"- **Action Prediction Accuracy:** {action_acc:.4f}")
            report_content.append("")
        
        # RL Results
        for algorithm in ['ppo', 'dqn', 'a2c']:
            if algorithm in self.results.get('evaluation_results', {}):
                report_content.append(f"### {algorithm.upper()} (Reinforcement Learning)")
                rl_results = self.results['evaluation_results'][algorithm]
                if rl_results.get('status') == 'success':
                    report_content.append(f"- **Mean Episode Reward:** {rl_results.get('mean_reward', 0):.4f}")
                    report_content.append(f"- **Std Episode Reward:** {rl_results.get('std_reward', 0):.4f}")
                    report_content.append(f"- **Evaluation Episodes:** {rl_results.get('n_episodes', 0)}")
                else:
                    report_content.append(f"- **Status:** {rl_results.get('status', 'unknown')}")
                    if 'error' in rl_results:
                        report_content.append(f"- **Error:** {rl_results['error']}")
                report_content.append("")
        
        # Method Comparison
        if 'performance_scores' in comparison_results:
            report_content.append("## Method Comparison")
            report_content.append("| Method | Performance Score |")
            report_content.append("|--------|------------------|")
            
            for method, score in comparison_results['performance_scores'].items():
                report_content.append(f"| {method} | {score:.4f} |")
            
            report_content.append("")
        
        # Conclusions
        report_content.append("## Conclusions")
        
        if 'rankings' in comparison_results and comparison_results['rankings']['ranking']:
            best_method = comparison_results['rankings']['ranking'][0]
            report_content.append(f"1. **{best_method}** achieved the best performance in our evaluation.")
            
            if len(comparison_results['rankings']['ranking']) > 1:
                report_content.append("2. Performance differences between methods suggest different approaches have varying strengths.")
            
            report_content.append("3. Both IL and RL approaches show promise for surgical action prediction.")
        
        report_content.append("")
        report_content.append("## Recommendations")
        report_content.append("1. Implement outcome-based evaluation for fair comparison between IL and RL.")
        report_content.append("2. Consider hybrid approaches combining IL and RL strengths.")
        report_content.append("3. Evaluate on additional surgical datasets for generalization.")
        
        # Save report
        report_path = self.results_dir / 'comparison_report.md'
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_content))
        
        self.logger.info(f"📄 Final report saved to: {report_path}")
        
        # Console summary
        print("\n" + "="*80)
        print("🏆 FINAL COMPARISON RESULTS")
        print("="*80)
        
        if 'rankings' in comparison_results and comparison_results['rankings']['ranking']:
            print("🥇 METHOD RANKINGS:")
            for i, (method, score) in enumerate(zip(
                comparison_results['rankings']['ranking'],
                comparison_results['rankings']['scores']
            )):
                medal = ["🥇", "🥈", "🥉"][i] if i < 3 else f"{i+1}."
                print(f"   {medal} {method}: {score:.4f}")
        
        print("="*80)
    
    def _save_complete_results(self):
        """Save all results to files."""
        
        # Convert any numpy types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, tuple):
                return tuple(convert_numpy_types(item) for item in obj)
            else:
                return obj
        
        # Convert results
        converted_results = convert_numpy_types(self.results)
        
        # Save main results
        results_path = self.results_dir / 'complete_comparison_results.json'
        with open(results_path, 'w') as f:
            json.dump(converted_results, f, indent=2)
        
        # Save summary
        summary = {
            'experiment_date': datetime.now().isoformat(),
            'total_methods': len(self.results.get('evaluation_results', {})),
            'best_method': self.results.get('comparison_results', {}).get('summary', {}).get('best_method', 'N/A'),
            'config_used': self.config,
            'model_paths': self.results.get('model_paths', {}),
            'results_directory': str(self.results_dir)
        }
        
        summary_path = self.results_dir / 'experiment_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"💾 Complete results saved to: {results_path}")
        self.logger.info(f"📋 Experiment summary saved to: {summary_path}")


def main():
    """Main function to run the complete IL vs RL comparison."""
    
    print("🚀 Starting Complete IL vs RL Comparison for Surgical Action Prediction")
    print("=" * 80)
    
    # Check if fixed config exists
    config_path = 'config_local_debug.yaml'  # Use the fixed config we created
    if not os.path.exists(config_path):
        config_path = 'config.yaml'  # Fallback to original
        print(f"⚠️ Using original config: {config_path}")
        print("💡 For best results, use the configuration with RL enabled")
    else:
        print(f"✅ Using config: {config_path}")
    
    try:
        # Create and run experiment
        experiment = ComparisonExperiment(config_path)
        results = experiment.run_complete_comparison()
        
        # Print final summary
        if 'error' not in results:
            print("\n🎉 Comparison completed successfully!")
            
            comparison_results = results.get('comparison_results', {})
            if 'summary' in comparison_results:
                summary = comparison_results['summary']
                print(f"🏆 Best method: {summary.get('best_method', 'N/A')}")
                print(f"📊 Best score: {summary.get('best_score', 0):.4f}")
        else:
            print(f"\n❌ Comparison failed: {results['error']}")
            return 1
    
    except Exception as e:
        print(f"\n💥 Experiment crashed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())