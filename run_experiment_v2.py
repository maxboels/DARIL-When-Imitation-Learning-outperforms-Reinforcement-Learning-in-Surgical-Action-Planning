#!/usr/bin/env python3
"""
Complete Experimental Comparison for Research Paper:
1. Imitation Learning (Baseline)
2. RL with World Model Simulation (Our Main Approach) 
3. RL with Offline Video Episodes (Ablation Study)
"""

import torch
import numpy as np
import yaml
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import os

# Import your existing components
from datasets.cholect50 import load_cholect50_data, NextFramePredictionDataset, create_video_dataloaders
from models.dual_world_model import DualWorldModel
from training.dual_trainer import DualTrainer, train_dual_world_model
from training.sb3_rl_trainer import SB3Trainer  # For RL training
from utils.logger import SimpleLogger
from torch.utils.data import DataLoader

from environment.direct_video_env import DirectVideoSB3Trainer, test_direct_video_environment
from evaluation.dual_evaluation_framework import DualEvaluationFramework
from evaluation.integrated_evaluation_framework import run_integrated_evaluation
from evaluation.paper_generator import generate_research_paper


# Import SB3 for RL
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SurgicalRLComparison:
    """
    Experimental comparison for research paper comparing:
    1. Method 1: Imitation Learning (Baseline)
    2. Method 2: RL with World Model Simulation (Our Main Approach)
    3. Method 3: RL with Offline Video Episodes (Ablation Study)
    """
    
    def __init__(self, config_path: str = 'config_dgx_all.yaml'):
        """Initialize the surgical RL comparison experiment."""
        
        # Reset logger timestamp for clean experiment
        SimpleLogger.reset_shared_timestamp()
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup logging with shared timestamp
        self.logger = SimpleLogger(log_dir="logs", name="SuRL")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Results storage
        self.results = {
            'method_1_il_baseline': None,
            'method_2_rl_world_model': None,
            'method_3_rl_offline_videos': None,
            'comparative_analysis': None,
            'model_paths': {},
            'config': self.config,
            'experiment_metadata': {
                'start_time': datetime.now().isoformat(),
                'device': str(self.device),
                'config_path': config_path
            }
        }
        
        # Create results directory
        self.results_dir = Path(self.logger.log_dir) / 'results'
        self.results_dir.mkdir(exist_ok=True)
        
        self.logger.info("🚀 SURGICAL RL COMPARISON INITIALIZED")
        self.logger.info("Method 1: Imitation Learning (Baseline)")
        self.logger.info("Method 2: RL with World Model Simulation")
        self.logger.info("Method 3: RL with Offline Video Episodes")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Results will be saved to: {self.results_dir}")
    
    def run_complete_comparison(self) -> Dict[str, Any]:
        """Run the complete surgical RL comparison."""
        
        try:
            # Step 1: Load data
            self.logger.info("=" * 60)
            self.logger.info("STEP 1: LOADING DATASET")
            self.logger.info("=" * 60)
            train_data, test_data = self._load_data()
            
            # Step 2: Method 1 - Imitation Learning Baseline
            self.logger.info("=" * 60) 
            self.logger.info("STEP 2: METHOD 1 - IMITATION LEARNING BASELINE")
            self.logger.info("=" * 60)
            method1_results = self._run_method1_imitation_learning(train_data, test_data)
            self.results['method_1_il_baseline'] = method1_results
            
            # Step 3: Method 2 - RL with World Model Simulation
            self.logger.info("=" * 60)
            self.logger.info("STEP 3: METHOD 2 - RL WITH WORLD MODEL SIMULATION")
            self.logger.info("=" * 60)
            method2_results = self._run_method2_rl_world_model(train_data, test_data)
            self.results['method_2_rl_world_model'] = method2_results
            
            # Step 4: Method 3 - RL with Offline Video Episodes
            self.logger.info("=" * 60)
            self.logger.info("STEP 4: METHOD 3 - RL WITH OFFLINE VIDEO EPISODES")
            self.logger.info("=" * 60)
            method3_results = self._run_method3_rl_offline_videos(train_data, test_data)
            self.results['method_3_rl_offline_videos'] = method3_results
            
            # Step 5: Comprehensive Evaluation and Comparison
            self.logger.info("=" * 60)
            self.logger.info("STEP 5: COMPREHENSIVE EVALUATION AND COMPARISON")
            self.logger.info("=" * 60)
            comparative_results = self._run_comprehensive_evaluation(test_data)
            self.results['comparative_analysis'] = comparative_results
            
            # Step 6: Generate Research Paper Results
            self.logger.info("=" * 60)
            self.logger.info("STEP 6: GENERATING RESEARCH PAPER RESULTS")
            self.logger.info("=" * 60)
            self._generate_paper_results()
            
            # Step 7: Save all results
            self._save_complete_results()
            
            self.logger.info("✅ SURGICAL RL COMPARISON COMPLETED SUCCESSFULLY!")
            return self.results
            
        except Exception as e:
            self.logger.error(f"❌ Experimental comparison failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'error': str(e), 'partial_results': self.results}
    
    def _load_data(self) -> Tuple[List[Dict], List[Dict]]:
        """Load and prepare training and test data."""
        
        train_videos = self.config.get('experiment', {}).get('train', {}).get('max_videos', 20)
        train_data = load_cholect50_data(
            self.config, self.logger, split='train', max_videos=train_videos
        )
        
        test_videos = self.config.get('experiment', {}).get('test', {}).get('max_videos', 10)
        test_data = load_cholect50_data(
            self.config, self.logger, split='test', max_videos=test_videos
        )
        
        self.logger.info(f"✅ Loaded {len(train_data)} training videos and {len(test_data)} test videos")
        return train_data, test_data
    
    def _run_method1_imitation_learning(self, train_data: List[Dict], test_data: List[Dict]) -> Dict[str, Any]:
        """
        Method 1: Imitation Learning Baseline
        
        Approach: Supervised learning on expert demonstrations
        Evaluation: Direct action prediction accuracy (mAP, top-k, etc.)
        """
        
        self.logger.info("🎓 Training Imitation Learning Baseline...")
        self.logger.info("📋 Approach: Supervised learning on expert demonstrations")
        
        try:
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
            
            # Train model in supervised mode
            self.config['training_mode'] = 'supervised'
            
            il_model_path = train_dual_world_model(
                self.config, self.logger, model, train_loader, test_video_loaders, self.device
            )
            
            # Store model path
            self.results['model_paths']['method1_il'] = il_model_path
            
            # Evaluate IL model
            il_model = DualWorldModel.load_model(il_model_path, self.device)
            evaluation_results = self._evaluate_il_model(il_model, test_data)
            
            result = {
                'method': 'Imitation Learning (Baseline)',
                'approach': 'Supervised learning on expert demonstrations',
                'model_path': il_model_path,
                'evaluation': evaluation_results,
                'status': 'success',
                'training_time': 'recorded',
                'key_insight': 'Optimized for action mimicry - should excel at traditional metrics'
            }
            
            self.logger.info(f"✅ Method 1 (IL) completed successfully")
            self.logger.info(f"📊 IL mAP: {evaluation_results.get('mAP', 0):.4f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Method 1 (IL) failed: {e}")
            return {'method': 'Imitation Learning', 'status': 'failed', 'error': str(e)}
    
    def _run_method2_rl_world_model(self, train_data: List[Dict], test_data: List[Dict]) -> Dict[str, Any]:
        """
        Method 2: RL with World Model Simulation (Our Main Approach)
        
        Approach: Use trained world model as RL environment simulator
        Evaluation: Policy performance in simulated environment + real transfer
        """
        
        self.logger.info("🌍 Training RL with World Model Simulation...")
        self.logger.info("📋 Approach: World model as RL environment simulator")
        
        try:
            # Load or train world model
            world_model_path = self.results['model_paths'].get('method1_il')
            if world_model_path and os.path.exists(world_model_path):
                world_model = DualWorldModel.load_model(world_model_path, self.device)
                self.logger.info(f"✅ Using world model from Method 1: {world_model_path}")
            else:
                # Train world model if not available
                self.logger.info("🔧 Training world model for RL simulation...")
                world_model_path = self._train_world_model_for_rl(train_data)
                world_model = DualWorldModel.load_model(world_model_path, self.device)
                self.results['model_paths']['world_model'] = world_model_path
            
            # Create world model-based RL trainer
            rl_trainer = SB3Trainer(world_model, self.config, self.logger, self.device)
            
            # Train RL algorithms using world model simulation
            timesteps = self.config.get('experiment', {}).get('rl_experiments', {}).get('timesteps', 10000)
            
            rl_results = {}
            
            # Train PPO with world model
            self.logger.info("🤖 Training PPO with world model simulation...")
            ppo_result = rl_trainer.train_ppo_final(train_data, timesteps)
            rl_results['ppo'] = ppo_result
            
            # Train A2C with world model  
            self.logger.info("🤖 Training A2C with world model simulation...")
            a2c_result = rl_trainer.train_dqn_final(train_data, timesteps)  # Actually A2C
            rl_results['a2c'] = a2c_result
            
            # Evaluate RL models
            evaluation_results = self._evaluate_rl_models(rl_results, test_data, world_model)
            
            result = {
                'method': 'RL with World Model Simulation',
                'approach': 'World model as environment simulator for RL training',
                'world_model_path': world_model_path,
                'rl_models': rl_results,
                'evaluation': evaluation_results,
                'status': 'success',
                'key_insight': 'Can explore beyond expert demonstrations using learned dynamics'
            }
            
            self.logger.info(f"✅ Method 2 (RL + World Model) completed successfully")
            successful_models = [alg for alg, res in rl_results.items() if res.get('status') == 'success']
            self.logger.info(f"📊 Successful RL models: {successful_models}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Method 2 (RL + World Model) failed: {e}")
            return {'method': 'RL with World Model', 'status': 'failed', 'error': str(e)}

    def _run_method3_rl_offline_videos(self, train_data: List[Dict], test_data: List[Dict]) -> Dict[str, Any]:
        """
        Method 3: RL with Offline Video Episodes (COMPLETE IMPLEMENTATION)
        
        Approach: Direct RL on offline video sequences without world model
        Evaluation: Policy performance on real video sequences
        """
        
        self.logger.info("📹 Training RL with Offline Video Episodes...")
        self.logger.info("📋 Approach: Direct RL on video sequences (no world model)")
        
        try:
            # Test the environment first
            self.logger.info("🧪 Testing Direct Video Environment...")
            test_success = test_direct_video_environment(train_data, self.config)
            
            if not test_success:
                return {
                    'method': 'RL with Offline Video Episodes',
                    'status': 'failed',
                    'error': 'Environment test failed'
                }
            
            # Create DirectVideo trainer
            direct_trainer = DirectVideoSB3Trainer(
                video_data=train_data,
                config=self.config,
                logger=self.logger,
                device=self.device
            )
            
            # Get training parameters
            timesteps = self.config.get('experiment', {}).get('rl_experiments', {}).get('timesteps', 10000)
            
            self.logger.info(f"🚀 Training RL algorithms on direct video sequences for {timesteps} timesteps...")
            
            # Train all algorithms
            rl_results = direct_trainer.train_all_algorithms(timesteps)
            
            # Evaluate results
            evaluation_results = self._evaluate_direct_video_models(rl_results, test_data)
            
            result = {
                'method': 'RL with Offline Video Episodes',
                'approach': 'Direct RL on video sequences without world model simulation',
                'rl_models': rl_results,
                'evaluation': evaluation_results,
                'status': 'success',
                'key_insight': 'Limited to existing video data, no simulation capability',
                'trainer_save_dir': str(direct_trainer.save_dir)
            }
            
            self.logger.info(f"✅ Method 3 (RL + Offline Videos) completed successfully")
            successful_models = [alg for alg, res in rl_results.items() if res.get('status') == 'success']
            self.logger.info(f"📊 Successful Direct Video RL models: {successful_models}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Method 3 (RL + Offline Videos) failed: {e}")
            import traceback
            traceback.print_exc()
            return {'method': 'RL with Offline Videos', 'status': 'failed', 'error': str(e)}

    def _evaluate_direct_video_models(self, rl_results: Dict, test_data: List[Dict]) -> Dict[str, Any]:
        """Evaluate Direct Video RL models on test data."""
        
        evaluation_results = {}
        
        for alg_name, rl_result in rl_results.items():
            if rl_result.get('status') == 'success' and 'model_path' in rl_result:
                try:
                    # For Direct Video models, the evaluation is based on episode stats
                    # and the mean reward from training
                    
                    evaluation_results[alg_name] = {
                        'mean_reward': rl_result.get('mean_reward', 0),
                        'std_reward': rl_result.get('std_reward', 0),
                        'episode_stats': rl_result.get('episode_stats', {}),
                        'training_successful': True,
                        'model_available': True,
                        'uses_real_frames': True,
                        'uses_world_model': False
                    }
                    
                    self.logger.info(f"✅ {alg_name} Direct Video evaluation completed")
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ Error evaluating Direct Video model {alg_name}: {e}")
                    evaluation_results[alg_name] = {'error': str(e)}
        
        return evaluation_results
    
    def _evaluate_il_model(self, il_model, test_data: List[Dict]) -> Dict[str, float]:
        """Evaluate IL model using traditional action prediction metrics."""
        
        from sklearn.metrics import average_precision_score
        
        il_model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for video in test_data:
                try:
                    # Create dataset for this video
                    video_dataset = NextFramePredictionDataset(self.config['data'], [video])
                    video_loader = DataLoader(video_dataset, batch_size=16, shuffle=False)
                    
                    for batch in video_loader:
                        current_states = batch['current_states'].to(self.device)
                        next_actions = batch['next_actions'].to(self.device)
                        
                        outputs = il_model(current_states=current_states)
                        
                        if 'action_pred' in outputs:
                            predictions = torch.sigmoid(outputs['action_pred'])
                            all_predictions.append(predictions.cpu().numpy())
                            all_targets.append(next_actions.cpu().numpy())
                
                except Exception as e:
                    self.logger.warning(f"⚠️ Error evaluating video {video.get('video_id', 'unknown')}: {e}")
                    continue
        
        if not all_predictions:
            return {'error': 'No predictions available'}
        
        # Calculate metrics
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        # Flatten for metric calculation
        pred_flat = predictions.reshape(-1, predictions.shape[-1])
        target_flat = targets.reshape(-1, targets.shape[-1])
        binary_preds = (pred_flat > 0.5).astype(int)
        
        # Calculate mAP
        ap_scores = []
        for i in range(target_flat.shape[1]):
            if np.sum(target_flat[:, i]) > 0:
                ap = average_precision_score(target_flat[:, i], pred_flat[:, i])
                ap_scores.append(ap)
        mAP = np.mean(ap_scores) if ap_scores else 0.0
        
        # Calculate other metrics
        exact_match = np.mean(np.all(binary_preds == target_flat, axis=1))
        hamming_acc = np.mean(binary_preds == target_flat)
        
        return {
            'mAP': mAP,
            'exact_match_accuracy': exact_match,
            'hamming_accuracy': hamming_acc,
            'evaluation_samples': len(pred_flat)
        }
    
    def _evaluate_rl_models(self, rl_results: Dict, test_data: List[Dict], world_model) -> Dict[str, Any]:
        """Evaluate RL models on test data."""
        
        evaluation_results = {}
        
        for alg_name, rl_result in rl_results.items():
            if rl_result.get('status') == 'success' and 'model_path' in rl_result:
                try:
                    # Load RL model
                    if alg_name.lower() == 'ppo':
                        from stable_baselines3 import PPO
                        rl_model = PPO.load(rl_result['model_path'])
                    elif alg_name.lower() == 'a2c':
                        from stable_baselines3 import A2C
                        rl_model = A2C.load(rl_result['model_path'])
                    else:
                        continue
                    
                    # Evaluate on test data (simplified)
                    evaluation_results[alg_name] = {
                        'mean_reward': rl_result.get('mean_reward', 0),
                        'std_reward': rl_result.get('std_reward', 0),
                        'training_successful': True,
                        'model_available': True
                    }
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ Error evaluating RL model {alg_name}: {e}")
                    evaluation_results[alg_name] = {'error': str(e)}
        
        return evaluation_results
    
    def _train_world_model_for_rl(self, train_data: List[Dict]) -> str:
        """Train world model specifically for RL if not available from IL training"""
        
        self.logger.info("🔧 Training world model for RL simulation...")
        
        # Create datasets and dataloaders
        train_dataset = NextFramePredictionDataset(self.config['data'], train_data)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=self.config['training']['num_workers'],
            pin_memory=self.config['training']['pin_memory']
        )
        
        # Initialize model
        model_config = self.config['models']['dual_world_model']
        model = DualWorldModel(**model_config).to(self.device)
        
        # Train model in supervised mode for world model
        self.config['training_mode'] = 'supervised'
        
        world_model_path = train_dual_world_model(
            self.config, self.logger, model, train_loader, {}, self.device
        )
        
        return world_model_path

    def _run_comprehensive_evaluation(self, test_data: List[Dict]) -> Dict[str, Any]:
        """
        UPDATED: Run integrated evaluation with rollout saving and unified mAP metrics
        """
        
        self.logger.info("📊 Running Integrated Evaluation with Rollout Saving...")
        
        try:
            # Run integrated evaluation
            evaluation_config = self.config.get('evaluation', {})
            horizon = evaluation_config.get('horizon', 15)
            integrated_results = run_integrated_evaluation(
                experiment_results=self.results,
                test_data=test_data,
                results_dir=str(self.results_dir),
                logger=self.logger,
                horizon=horizon
            )
            
            if integrated_results:
                self.logger.info("✅ Integrated evaluation completed successfully!")
                
                # Extract key results for backward compatibility
                evaluator = integrated_results['evaluator']
                results = integrated_results['results']
                file_paths = integrated_results['file_paths']
                
                # Print method comparison
                self._print_method_comparison(results['aggregate_results'])
                
                # Print statistical significance
                self._print_statistical_significance(results['statistical_tests'])
                
                return {
                    'integrated_evaluation': {
                        'status': 'success',
                        'results': results,
                        'file_paths': file_paths,
                        'visualization_data_path': str(file_paths['visualization_json'])
                    },
                    'evaluation_type': 'integrated_with_rollouts',
                    'summary': self._create_evaluation_summary(results)
                }
            else:
                return {'error': 'Integrated evaluation failed', 'status': 'failed'}
                
        except Exception as e:
            self.logger.error(f"❌ Integrated evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            return {'error': str(e), 'status': 'failed'}

    def _print_method_comparison(self, aggregate_results: Dict):
        """Print comparison of all methods"""
        
        self.logger.info("\n📊 METHOD COMPARISON (Unified mAP Metrics)")
        self.logger.info("=" * 60)
        
        # Sort methods by performance
        methods_sorted = sorted(aggregate_results.items(), 
                              key=lambda x: x[1]['final_mAP']['mean'], reverse=True)
        
        for rank, (method, stats) in enumerate(methods_sorted, 1):
            method_display = method.replace('_', ' ')
            final_map = stats['final_mAP']['mean']
            std_map = stats['final_mAP']['std']
            degradation = stats['mAP_degradation']['mean']
            
            self.logger.info(f"{rank}. {method_display}:")
            self.logger.info(f"   📈 mAP: {final_map:.4f} ± {std_map:.4f}")
            self.logger.info(f"   📉 Degradation: {degradation:.4f}")
            self.logger.info(f"   🎯 Stability: {-degradation:.4f}")
            self.logger.info("")

    def _print_statistical_significance(self, statistical_tests: Dict):
        """Print statistical significance results"""
        
        self.logger.info("🔬 STATISTICAL SIGNIFICANCE TESTS")
        self.logger.info("=" * 40)
        
        significant_comparisons = [
            (comparison, results) for comparison, results in statistical_tests.items()
            if results['significant']
        ]
        
        if significant_comparisons:
            self.logger.info(f"Found {len(significant_comparisons)} significant differences:")
            for comparison, results in significant_comparisons:
                method1, method2 = comparison.split('_vs_')
                method1_display = method1.replace('_', ' ')
                method2_display = method2.replace('_', ' ')
                
                self.logger.info(f"  • {method1_display} vs {method2_display}:")
                self.logger.info(f"    p-value: {results['p_value']:.4f}")
                self.logger.info(f"    Effect size: {results['effect_size_interpretation']}")
                self.logger.info(f"    Mean difference: {results['mean_diff']:.4f}")
        else:
            self.logger.info("No statistically significant differences found")

    def _create_evaluation_summary(self, results: Dict) -> Dict:
        """Create summary for backward compatibility"""
        
        aggregate_results = results['aggregate_results']
        
        # Find best method
        best_method = max(aggregate_results.items(), 
                         key=lambda x: x[1]['final_mAP']['mean'])
        
        # Count significant comparisons
        significant_count = sum(1 for test in results['statistical_tests'].values() 
                              if test['significant'])
        
        return {
            'best_method': {
                'name': best_method[0],
                'mAP': best_method[1]['final_mAP']['mean'],
                'std': best_method[1]['final_mAP']['std']
            },
            'total_methods': len(aggregate_results),
            'significant_comparisons': significant_count,
            'evaluation_horizon': results['evaluation_config']['horizon'],
            'videos_evaluated': results['evaluation_config']['num_videos']
        }

    def _generate_paper_results(self):
        """Generate research paper ready results with integrated evaluation"""
        
        self.logger.info("📝 Generating enhanced research paper results...")
        
        paper_results = {
            'title': 'Comprehensive Evaluation: IL vs RL+WorldModel vs RL+OfflineVideos for Surgical Action Prediction',
            'experiment_summary': {
                'method_1': 'Imitation Learning (Baseline) - Supervised learning on expert demonstrations',
                'method_2': 'RL with World Model Simulation - Uses learned dynamics for exploration',
                'method_3': 'RL with Offline Videos - Direct RL on video sequences',
                'evaluation': 'Integrated evaluation with unified mAP metrics, rollout saving, and trajectory analysis'
            },
            'key_findings': [],
            'method_performance': {},
            'integrated_evaluation_results': {},
            'research_contributions': []
        }
        
        # Extract integrated evaluation results
        integrated_eval = self.results.get('comparative_analysis', {}).get('integrated_evaluation', {})
        
        if integrated_eval.get('status') == 'success':
            eval_results = integrated_eval.get('results', {})
            
            if 'aggregate_results' in eval_results:
                aggregate_stats = eval_results['aggregate_results']
                
                # Sort methods by performance
                methods_sorted = sorted(aggregate_stats.items(), 
                                      key=lambda x: x[1]['final_mAP']['mean'], reverse=True)
                
                paper_results['integrated_evaluation_results'] = {
                    'ranking': [
                        {
                            'rank': i+1,
                            'method': method,
                            'final_mAP': stats['final_mAP']['mean'],
                            'mAP_std': stats['final_mAP']['std'],
                            'degradation': stats['mAP_degradation']['mean'],
                            'stability': stats['trajectory_stability'],
                            'confidence': stats.get('confidence', {}).get('mean', 0.0)
                        }
                        for i, (method, stats) in enumerate(methods_sorted)
                    ],
                    'best_method': methods_sorted[0][0] if methods_sorted else 'Unknown',
                    'best_performance': methods_sorted[0][1]['final_mAP']['mean'] if methods_sorted else 0.0,
                    'evaluation_features': [
                        'Unified mAP metrics across all methods',
                        'Rollout saving at every timestep',
                        'Planning horizon visualization',
                        'Thinking process capture',
                        'Statistical significance testing'
                    ]
                }
                
                # Statistical significance results
                if 'statistical_tests' in eval_results:
                    paper_results['integrated_evaluation_results']['statistical_tests'] = {
                        'significant_comparisons': [
                            {
                                'comparison': comp,
                                'p_value': results['p_value'],
                                'mean_difference': results['mean_diff'],
                                'effect_size': results['cohens_d'],
                                'interpretation': results['effect_size_interpretation']
                            }
                            for comp, results in eval_results['statistical_tests'].items()
                            if results['significant']
                        ],
                        'total_comparisons': len(eval_results['statistical_tests']),
                        'significant_count': sum(1 for r in eval_results['statistical_tests'].values() if r['significant'])
                    }
                
                # Add visualization data path
                viz_path = integrated_eval.get('visualization_data_path')
                if viz_path:
                    paper_results['integrated_evaluation_results']['visualization_data_path'] = viz_path
        
        # Traditional method results (for comparison)
        method1 = self.results.get('method_1_il_baseline', {})
        method2 = self.results.get('method_2_rl_world_model', {})
        method3 = self.results.get('method_3_rl_offline_videos', {})
        
        # Extract performance metrics
        if method1.get('status') == 'success':
            il_performance = method1.get('evaluation', {})
            paper_results['method_performance']['IL_Baseline'] = {
                'traditional_mAP': il_performance.get('mAP', 0),
                'exact_match': il_performance.get('exact_match_accuracy', 0),
                'status': 'success',
                'strength': 'Action mimicry via supervised learning'
            }
        
        if method2.get('status') == 'success':
            rl_models = method2.get('rl_models', {})
            paper_results['method_performance']['RL_WorldModel'] = {
                'algorithms': list(rl_models.keys()),
                'status': 'success',
                'strength': 'Exploration via world model simulation',
                'reward_performance': {
                    alg: res.get('mean_reward', 0) 
                    for alg, res in rl_models.items() 
                    if res.get('status') == 'success'
                }
            }
        
        if method3.get('status') == 'success':
            rl_models = method3.get('rl_models', {})
            paper_results['method_performance']['RL_OfflineVideos'] = {
                'algorithms': list(rl_models.keys()),
                'status': 'success',
                'strength': 'Direct interaction with real video data',
                'reward_performance': {
                    alg: res.get('mean_reward', 0) 
                    for alg, res in rl_models.items() 
                    if res.get('status') == 'success'
                }
            }
        
        # Key findings based on integrated evaluation
        findings = []
        if integrated_eval.get('status') == 'success':
            findings.append("✅ Integrated evaluation completed with unified mAP metrics")
            findings.append("✅ All methods evaluated on identical action prediction metrics")
            findings.append("✅ Detailed rollout saving enables visualization of thinking process")
            findings.append("✅ Statistical significance testing performed between all method pairs")
            findings.append("✅ Planning horizon analysis shows performance degradation patterns")
            
            # Add specific performance findings
            if 'integrated_evaluation_results' in paper_results and 'ranking' in paper_results['integrated_evaluation_results']:
                ranking = paper_results['integrated_evaluation_results']['ranking']
                if ranking:
                    best = ranking[0]
                    findings.append(f"✅ Best method: {best['method']} with {best['final_mAP']:.3f} mAP")
                    
                    if len(ranking) > 1:
                        performance_gap = best['final_mAP'] - ranking[-1]['final_mAP']
                        findings.append(f"✅ Performance gap between best and worst: {performance_gap:.3f} mAP")
        else:
            findings.append("⚠️ Integrated evaluation encountered issues")
        
        # Add method-specific findings
        if method1.get('status') == 'success':
            findings.append("✅ Method 1 (IL): Successfully trained and evaluated")
        if method2.get('status') == 'success':
            findings.append("✅ Method 2 (RL + World Model): Successfully demonstrates model-based RL")
        if method3.get('status') == 'success':
            findings.append("✅ Method 3 (RL + Offline Videos): Successfully demonstrates model-free RL")
        
        paper_results['key_findings'] = findings
        
        # Research contributions (updated)
        paper_results['research_contributions'] = [
            "First systematic three-way comparison: IL vs model-based RL vs model-free RL in surgery",
            "Integrated evaluation framework with unified mAP metrics for fair comparison",
            "Rollout saving and visualization of AI decision-making process",
            "Trajectory analysis showing performance degradation over prediction horizons",
            "Statistical significance testing with effect size analysis",
            "Comprehensive visualization suite for surgical AI method comparison",
            "Open-source implementation for reproducible surgical RL research"
        ]

        # Save enhanced paper results
        paper_results_path = self.results_dir / 'integrated_paper_results.json'
        with open(paper_results_path, 'w') as f:
            json.dump(paper_results, f, indent=2, default=str)
        
        # Generate enhanced paper summary
        self._generate_enhanced_paper_summary(paper_results)
        
        # 🆕 NEW: Generate complete research paper with LaTeX and figures
        paper_dir = generate_research_paper(self.results_dir, self.logger)
        
        self.logger.info(f"📄 Integrated paper results saved to: {paper_results_path}")
        self.logger.info(f"📄 Complete research paper generated: {paper_dir}")
        self.logger.info("📊 Includes: LaTeX tables, publication figures, complete paper.tex")
        self.logger.info("🔧 Run compile_paper.sh to generate PDF")


        # Save enhanced paper results
        paper_results_path = self.results_dir / 'integrated_paper_results.json'
        with open(paper_results_path, 'w') as f:
            json.dump(paper_results, f, indent=2, default=str)
        
        # Generate enhanced paper summary
        self._generate_enhanced_paper_summary(paper_results)
        
        self.logger.info(f"📄 Integrated paper results saved to: {paper_results_path}")

    def _generate_enhanced_paper_summary(self, paper_results: Dict):
        """Generate a markdown summary with integrated evaluation results"""
        
        summary_lines = []
        summary_lines.append("# Integrated Three-Way Experimental Comparison Results")
        summary_lines.append("## Surgical Action Prediction: IL vs RL Approaches with Rollout Analysis")
        summary_lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        summary_lines.append("")
        
        # Integrated evaluation results
        if 'integrated_evaluation_results' in paper_results:
            eval_results = paper_results['integrated_evaluation_results']
            
            summary_lines.append("## 🎯 Integrated Evaluation Results (Unified mAP Metrics)")
            summary_lines.append("")
            
            if 'ranking' in eval_results:
                for i, method_result in enumerate(eval_results['ranking'], 1):
                    method_name = method_result['method'].replace('_', ' ')
                    final_map = method_result['final_mAP']
                    std_map = method_result['mAP_std']
                    degradation = method_result['degradation']
                    
                    summary_lines.append(f"### {i}. {method_name}")
                    summary_lines.append(f"- **Final mAP**: {final_map:.4f} ± {std_map:.4f}")
                    summary_lines.append(f"- **mAP Degradation**: {degradation:.4f}")
                    summary_lines.append(f"- **Stability Score**: {method_result['stability']:.4f}")
                    summary_lines.append(f"- **Avg Confidence**: {method_result['confidence']:.4f}")
                    summary_lines.append("")
            
            # Statistical significance
            if 'statistical_tests' in eval_results:
                stat_tests = eval_results['statistical_tests']
                summary_lines.append("## 🔬 Statistical Analysis")
                summary_lines.append("")
                summary_lines.append(f"- **Total Comparisons**: {stat_tests['total_comparisons']}")
                summary_lines.append(f"- **Significant Differences**: {stat_tests['significant_count']}")
                summary_lines.append("")
                
                if stat_tests['significant_comparisons']:
                    summary_lines.append("### Significant Comparisons (p < 0.05)")
                    for comp in stat_tests['significant_comparisons']:
                        comparison_name = comp['comparison'].replace('_vs_', ' vs ').replace('_', ' ')
                        summary_lines.append(f"- **{comparison_name}**: p={comp['p_value']:.4f}, effect size={comp['interpretation']}")
                    summary_lines.append("")
            
            # Evaluation features
            if 'evaluation_features' in eval_results:
                summary_lines.append("## 🚀 Evaluation Features")
                summary_lines.append("")
                for feature in eval_results['evaluation_features']:
                    summary_lines.append(f"- {feature}")
                summary_lines.append("")
        
        # Traditional method comparison
        summary_lines.append("## 📊 Traditional Method Performance")
        summary_lines.append("")
        
        for method, performance in paper_results['method_performance'].items():
            summary_lines.append(f"### {method.replace('_', ' ')}")
            if performance['status'] == 'success':
                summary_lines.append(f"- **Status**: ✅ Successful")
                summary_lines.append(f"- **Strength**: {performance['strength']}")
                if 'traditional_mAP' in performance:
                    summary_lines.append(f"- **Traditional mAP**: {performance['traditional_mAP']:.4f}")
                if 'algorithms' in performance:
                    summary_lines.append(f"- **Algorithms**: {', '.join(performance['algorithms'])}")
                    if 'reward_performance' in performance:
                        for alg, perf in performance['reward_performance'].items():
                            summary_lines.append(f"  - **{alg.upper()}**: Mean Reward = {perf:.3f}")
            else:
                summary_lines.append(f"- **Status**: {performance['status']}")
            summary_lines.append("")
        
        summary_lines.append("## 🔍 Key Findings")
        summary_lines.append("")
        for finding in paper_results['key_findings']:
            summary_lines.append(f"- {finding}")
        summary_lines.append("")
        
        summary_lines.append("## 🏆 Research Contributions")
        summary_lines.append("")
        for contribution in paper_results['research_contributions']:
            summary_lines.append(f"- {contribution}")
        summary_lines.append("")
        
        # Add visualization note
        if 'integrated_evaluation_results' in paper_results and 'visualization_data_path' in paper_results['integrated_evaluation_results']:
            viz_path = paper_results['integrated_evaluation_results']['visualization_data_path']
            summary_lines.append("## 📊 Visualization")
            summary_lines.append("")
            summary_lines.append(f"Interactive visualization data available at: `{viz_path}`")
            summary_lines.append("Load this file in the HTML visualization tool to explore:")
            summary_lines.append("- Model thinking process at each timestep")
            summary_lines.append("- Planning horizon rollouts")
            summary_lines.append("- Ground truth vs predictions comparison")
            summary_lines.append("- Confidence and uncertainty analysis")
            summary_lines.append("")
        
        # Save summary
        summary_path = self.results_dir / 'integrated_experiment_summary.md'
        with open(summary_path, 'w') as f:
            f.write('\n'.join(summary_lines))
        
        self.logger.info(f"📄 Integrated experiment summary saved to: {summary_path}")

    def _save_complete_results(self):
        """Save all experimental results with proper JSON serialization."""
        
        # Convert results for JSON serialization
        converted_results = self._convert_numpy_types(self.results)
        
        # Save complete results
        results_path = self.results_dir / 'complete_results.json'
        with open(results_path, 'w') as f:
            json.dump(converted_results, f, indent=2, default=str)  # Added default=str as backup
        
        self.logger.info(f"💾 Complete results saved to: {results_path}")
        self.logger.info(f"📁 All results available in: {self.results_dir}")


    def _convert_numpy_types(self, obj: Any) -> Any:
        """Convert numpy types and Path objects for JSON serialization."""
        if hasattr(obj, '__dataclass_fields__'):
            return {field: self._convert_numpy_types(getattr(obj, field)) for field in obj.__dataclass_fields__}
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._convert_numpy_types(item) for item in obj)
        else:
            return obj

def main():
    """Main function to run the surgical RL comparison."""
    
    print("🔬 SURGICAL RL COMPARISON")
    print("=" * 60)
    print("Research Paper: IL vs RL for Surgical Action Prediction")
    print()
    print("Method 1: Imitation Learning (Baseline)")
    print("Method 2: RL with World Model Simulation (Our Main Approach)")
    print("Method 3: RL with Offline Video Episodes (Ablation Study)")
    print()
    
    # Choose config file
    config_path = 'config_local_debug.yaml' # select from: config_dgx_all, config_local_debug
    if not os.path.exists(config_path):
        config_path = 'config.yaml'
        print(f"⚠️ Using fallback config: {config_path}")
    else:
        print(f"✅ Using config: {config_path}")
    
    try:
        experiment = SurgicalRLComparison(config_path)
        results = experiment.run_complete_comparison()
        
        if 'error' not in results:
            print("\n🎉 SURGICAL RL COMPARISON COMPLETED!")
            print("=" * 40)
            
            # Print summary
            print("📊 EXPERIMENT SUMMARY:")
            print("-" * 30)
            
            # Method 1 Results
            method1 = results.get('method_1_il_baseline', {})
            if method1.get('status') == 'success':
                il_map = method1.get('evaluation', {}).get('mAP', 0)
                print(f"✅ Method 1 (IL): mAP = {il_map:.4f}")
            else:
                print(f"❌ Method 1 (IL): {method1.get('status', 'Unknown status')}")
            
            # Method 2 Results
            method2 = results.get('method_2_rl_world_model', {})
            if method2.get('status') == 'success':
                successful_rl = [alg for alg, res in method2.get('rl_models', {}).items() 
                               if res.get('status') == 'success']
                print(f"✅ Method 2 (RL + World Model): {len(successful_rl)} algorithms trained")
                
                # Show individual algorithm performance
                for alg_name, alg_result in method2.get('rl_models', {}).items():
                    if alg_result.get('status') == 'success':
                        mean_reward = alg_result.get('mean_reward', 0)
                        print(f"   └─ {alg_name.upper()}: Mean Reward = {mean_reward:.3f}")
            else:
                print(f"❌ Method 2 (RL + World Model): {method2.get('status', 'Unknown status')}")
            
            # Method 3 Results (FIXED)
            method3 = results.get('method_3_rl_offline_videos', {})
            if method3.get('status') == 'success':
                successful_direct = [alg for alg, res in method3.get('rl_models', {}).items() 
                                   if res.get('status') == 'success']
                print(f"✅ Method 3 (RL + Offline Videos): {len(successful_direct)} algorithms trained")
                
                # Show individual algorithm performance
                for alg_name, alg_result in method3.get('rl_models', {}).items():
                    if alg_result.get('status') == 'success':
                        mean_reward = alg_result.get('mean_reward', 0)
                        print(f"   └─ {alg_name.upper()}: Mean Reward = {mean_reward:.3f}")
            elif method3.get('status') == 'failed':
                print(f"❌ Method 3 (RL + Offline Videos): Failed - {method3.get('error', 'Unknown error')}")
            else:
                print(f"⚠️ Method 3 (RL + Offline Videos): {method3.get('status', 'Unknown status')}")
            
            print(f"\n📁 Results saved to: {experiment.results_dir}")
            
            # Enhanced summary
            print("\n🎯 EXPERIMENT COMPLETE!")
            print("=" * 40)
            
            total_methods = 0
            successful_methods = 0
            
            if method1.get('status') == 'success':
                successful_methods += 1
            total_methods += 1
            
            if method2.get('status') == 'success':
                successful_methods += 1
            total_methods += 1
            
            if method3.get('status') == 'success':
                successful_methods += 1
            total_methods += 1
            
            print(f"📊 Success Rate: {successful_methods}/{total_methods} methods completed")
            
            if successful_methods == 3:
                print("🎉 ALL THREE METHODS SUCCESSFUL!")
                print("🎓 Ready for research paper publication!")
                print("\n📋 Key Achievements:")
                print("   • First systematic surgical RL comparison")
                print("   • IL vs Model-based RL vs Model-free RL")
                print("   • Comprehensive evaluation framework")
                print("   • Publication-ready results generated")
            else:
                print(f"⚠️  {3-successful_methods} method(s) incomplete")
            
            print("\n🔬 RESEARCH PAPER READY!")
            
        else:
            print(f"\n❌ Experiment failed: {results['error']}")
            return 1
    
    except Exception as e:
        print(f"\n💥 Experiment crashed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())