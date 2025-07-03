#!/usr/bin/env python3
"""
Updated Experiment Runner with Comprehensive RL Debugging
Focus: Understanding why RL performance is poor vs supervised learning

Key additions:
1. Comprehensive RL debugging and monitoring
2. Simplified expert matching environment
3. World model quality evaluation
4. Action space analysis and optimization
5. Training visualizations and analysis
"""

import os
import yaml
import warnings
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

# Import existing models and components
from models.autoregressive_il_model import AutoregressiveILModel
from models.conditional_world_model import ConditionalWorldModel
from datasets.autoregressive_dataset import create_autoregressive_dataloaders
from datasets.world_model_dataset import create_world_model_dataloaders
from training.autoregressive_il_trainer import AutoregressiveILTrainer
from training.world_model_trainer import WorldModelTrainer
from datasets.cholect50 import load_cholect50_data
from evaluation.integrated_evaluation import run_integrated_evaluation
from utils.logger import SimpleLogger

# Import our new debugging components
from rl_debugging_system.rl_debug_system import RLDebugger, debug_rl_training_comprehensive
from rl_debugging_system.simplified_rl_trainer import SimplifiedRLTrainer, run_simplified_rl_debugging

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DebuggingExperimentRunner:
    """
    Updated experiment runner with comprehensive RL debugging.
    Focuses on understanding and fixing the RL vs supervised learning performance gap.
    """
    
    def __init__(self, config_path: str = 'config_dgx_all_v7.yaml'):
        print("🔍 Initializing RL Debugging Experiment")
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Create experiment directory with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.experiment_name = f"debug_rl_{timestamp}"
        self.results_dir = Path("results") / self.experiment_name
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logger
        self.logger = SimpleLogger(
            log_dir=str(self.results_dir),
            name="RL_Debug_Experiment",
            use_shared_timestamp=True
        )
        
        # Results storage
        self.results = {
            'experiment_name': self.experiment_name,
            'experiment_type': 'rl_debugging_and_optimization',
            'config': self.config,
            'timestamp': timestamp,
            'results_dir': str(self.results_dir),
        }
        
        self.logger.info(f"🔍 RL Debugging Experiment: {self.experiment_name}")
        self.logger.info(f"📁 Results dir: {self.results_dir}")
        self.logger.info(f"🎯 Focus: Understanding RL vs supervised performance gap")
    
    def run_debugging_comparison(self) -> Dict[str, Any]:
        """Run the complete comparison with enhanced RL debugging."""
        
        self.logger.info("🚀 Starting RL Debugging Comparison")
        self.logger.info("=" * 60)
        
        # Load data
        train_data, test_data = self._load_data()
        
        # Method 1: Supervised IL (baseline for comparison)
        method1_results = self._run_method1_supervised_baseline(train_data, test_data)
        self.results['method_1_supervised_baseline'] = method1_results
        
        # Method 2: World Model + Debugging
        method2_results = self._run_method2_world_model_with_debugging(train_data, test_data)
        self.results['method_2_world_model_debug'] = method2_results
        
        # Method 3: Simplified RL (focus on expert matching)
        method3_results = self._run_method3_simplified_rl(train_data, test_data)
        self.results['method_3_simplified_rl'] = method3_results
        
        # Comprehensive debugging analysis
        debug_analysis = self._run_comprehensive_debugging_analysis()
        self.results['debugging_analysis'] = debug_analysis
        
        # Performance comparison and insights
        self._analyze_performance_gaps()
        
        # Save results
        self._save_debugging_results()
        
        return self.results
    
    def _load_data(self) -> Tuple[List[Dict], List[Dict]]:
        """Load training and test data with focus on 40 train + 10 test videos."""
        
        self.logger.info("📂 Loading CholecT50 data for debugging...")
        
        # Use config-specified video counts
        train_videos = self.config.get('experiment', {}).get('train', {}).get('max_videos', 40)
        test_videos = self.config.get('experiment', {}).get('test', {}).get('max_videos', 10)
        test_on_train = self.config.get('experiment', {}).get('test', {}).get('test_on_train', False)
        
        self.logger.info(f"   Training videos: {train_videos}")
        self.logger.info(f"   Test videos: {test_videos}")
        
        # Load data
        train_data = load_cholect50_data(
            self.config, self.logger, split='train', max_videos=train_videos
        )
        test_data = load_cholect50_data(
            self.config, self.logger, split='test', max_videos=test_videos, test_on_train=test_on_train
        )
        
        self.logger.info(f"✅ Data loaded successfully")
        self.logger.info(f"   Training videos: {len(train_data)}")
        self.logger.info(f"   Test videos: {len(test_data)}")
        
        return train_data, test_data
    
    def _run_method1_supervised_baseline(self, train_data: List[Dict], test_data: List[Dict]) -> Dict[str, Any]:
        """Method 1: Supervised IL baseline to compare against."""
        
        self.logger.info("🎓 Method 1: Supervised IL Baseline")
        self.logger.info("-" * 40)
        
        try:
            # Check for pretrained model
            il_config = self.config.get('experiment', {}).get('autoregressive_il', {})
            il_model_path = il_config.get('il_model_path', None)
            
            if il_model_path and os.path.exists(il_model_path):
                self.logger.info(f"📂 Using pretrained IL model: {il_model_path}")
                
                # Load pretrained model
                model = AutoregressiveILModel.load_model(il_model_path, device=DEVICE)
                
                # Create test datasets for evaluation
                _, test_loaders = create_autoregressive_dataloaders(
                    config=self.config['data'],
                    train_data=None,
                    test_data=test_data,
                    batch_size=self.config['training']['batch_size'],
                    num_workers=self.config['training']['num_workers']
                )
                
                # Create trainer for evaluation
                trainer = AutoregressiveILTrainer(
                    model=model,
                    config=self.config,
                    logger=self.logger,
                    device=DEVICE
                )
                
                # Evaluate model
                evaluation_results = trainer.evaluate_model(test_loaders)
                
                return {
                    'status': 'success',
                    'model_path': il_model_path,
                    'model_type': 'AutoregressiveILModel',
                    'approach': 'Supervised IL baseline (pretrained)',
                    'evaluation': evaluation_results,
                    'baseline_mAP': evaluation_results.get('overall_metrics', {}).get('action_mAP', 0.0),
                    'method_description': 'Supervised imitation learning baseline for comparison'
                }
            else:
                self.logger.info("🏋️ Training IL model from scratch as baseline...")
                
                # Train from scratch
                model = AutoregressiveILModel(
                    hidden_dim=self.config['models']['autoregressive_il']['hidden_dim'],
                    embedding_dim=self.config['models']['autoregressive_il']['embedding_dim'],
                    n_layer=self.config['models']['autoregressive_il']['n_layer'],
                    num_action_classes=self.config['models']['autoregressive_il']['num_action_classes'],
                    dropout=self.config['models']['autoregressive_il']['dropout']
                ).to(DEVICE)
                
                # Create datasets
                train_loader, test_loaders = create_autoregressive_dataloaders(
                    config=self.config['data'],
                    train_data=train_data,
                    test_data=test_data,
                    batch_size=self.config['training']['batch_size'],
                    num_workers=self.config['training']['num_workers']
                )
                
                # Create trainer
                trainer = AutoregressiveILTrainer(
                    model=model,
                    config=self.config,
                    logger=self.logger,
                    device=DEVICE
                )
                
                # Train model
                best_model_path = trainer.train(train_loader, test_loaders)
                
                # Evaluate model
                evaluation_results = trainer.evaluate_model(test_loaders)
                
                return {
                    'status': 'success',
                    'model_path': best_model_path,
                    'model_type': 'AutoregressiveILModel',
                    'approach': 'Supervised IL baseline (trained)',
                    'evaluation': evaluation_results,
                    'baseline_mAP': evaluation_results.get('overall_metrics', {}).get('action_mAP', 0.0),
                    'method_description': 'Supervised imitation learning baseline for comparison'
                }
                
        except Exception as e:
            self.logger.error(f"❌ Method 1 failed: {e}")
            return {'status': 'failed', 'error': str(e), 'baseline_mAP': 0.0}
    
    def _run_method2_world_model_with_debugging(self, train_data: List[Dict], test_data: List[Dict]) -> Dict[str, Any]:
        """Method 2: World Model with comprehensive debugging."""
        
        self.logger.info("🌍 Method 2: World Model + Comprehensive Debugging")
        self.logger.info("-" * 40)
        
        try:
            # Check for pretrained world model
            wm_config = self.config.get('experiment', {}).get('world_model', {})
            wm_model_path = wm_config.get('wm_model_path', None)
            
            # Handle world model loading/training
            if wm_model_path and os.path.exists(wm_model_path):
                self.logger.info(f"📂 Using pretrained world model: {wm_model_path}")
                world_model = ConditionalWorldModel.load_model(wm_model_path, device=DEVICE)
                train_data_for_loader = None
            else:
                self.logger.info("🏋️ Training world model from scratch...")
                world_model = ConditionalWorldModel(
                    hidden_dim=self.config['models']['conditional_world_model']['hidden_dim'],
                    embedding_dim=self.config['models']['conditional_world_model']['embedding_dim'],
                    action_embedding_dim=self.config['models']['conditional_world_model']['action_embedding_dim'],
                    n_layer=self.config['models']['conditional_world_model']['n_layer'],
                    num_action_classes=self.config['models']['conditional_world_model']['num_action_classes'],
                    dropout=self.config['models']['conditional_world_model']['dropout']
                ).to(DEVICE)
                train_data_for_loader = train_data
            
            # Create datasets
            train_loader, test_loaders, simulation_loader = create_world_model_dataloaders(
                config=self.config['data'],
                train_data=train_data_for_loader,
                test_data=test_data,
                batch_size=self.config['training']['batch_size'],
                num_workers=self.config['training']['num_workers']
            )
            
            # Store test loaders for evaluation
            self.test_loaders = test_loaders
            
            # Train or evaluate world model
            world_model_trainer = WorldModelTrainer(
                model=world_model,
                config=self.config,
                logger=self.logger,
                device=DEVICE
            )
            
            if train_data_for_loader is None:
                # Just evaluate pretrained model
                world_model_evaluation = world_model_trainer.evaluate_model(test_loaders)
                best_world_model_path = wm_model_path
            else:
                # Train world model
                best_world_model_path = world_model_trainer.train(train_loader, test_loaders)
                world_model_evaluation = world_model_trainer.evaluate_model(test_loaders)
            
            # Initialize debugging system
            debugger = RLDebugger(
                save_dir=str(self.results_dir / 'world_model_debug'),
                logger=self.logger,
                config=self.config
            )
            
            # Step 1: Evaluate world model quality
            self.logger.info("🔍 Evaluating world model quality...")
            world_model_analysis = debugger.evaluate_world_model_quality(world_model, test_data)
            
            # Step 2: Run simplified RL training with debugging
            self.logger.info("🎯 Running simplified RL with debugging...")
            simplified_trainer = SimplifiedRLTrainer(self.config, self.logger)
            
            # Get training timesteps
            timesteps = self.config.get('rl_training', {}).get('timesteps', 20000)
            
            # Train simplified RL
            rl_results = simplified_trainer.train_simplified_ppo(
                world_model, train_data, timesteps
            )
            
            return {
                'status': 'success',
                'world_model_path': best_world_model_path,
                'world_model_evaluation': world_model_evaluation,
                'world_model_analysis': world_model_analysis,
                'simplified_rl_results': rl_results,
                'model_type': 'ConditionalWorldModel + SimplifiedRL',
                'approach': 'World model with comprehensive RL debugging',
                'method_description': 'World model + simplified expert matching RL',
                'debugging_applied': True,
                'world_model_quality_score': world_model_analysis['summary']['world_model_quality_score']
            }
            
        except Exception as e:
            self.logger.error(f"❌ Method 2 failed: {e}")
            import traceback
            traceback.print_exc()
            return {'status': 'failed', 'error': str(e)}
    
    def _run_method3_simplified_rl(self, train_data: List[Dict], test_data: List[Dict]) -> Dict[str, Any]:
        """Method 3: Simplified RL without world model (direct video + expert matching)."""
        
        self.logger.info("🎯 Method 3: Simplified RL (Direct Video + Expert Matching)")
        self.logger.info("-" * 40)
        
        try:
            # Run simplified RL debugging without world model
            trainer, results = run_simplified_rl_debugging(
                config=self.config,
                logger=self.logger,
                world_model=None,  # No world model for this approach
                train_data=train_data,
                test_data=test_data,
                timesteps=self.config.get('rl_training', {}).get('timesteps', 20000)
            )
            
            return {
                'status': 'success',
                'simplified_rl_results': results,
                'model_type': 'SimplifiedDirectVideoRL',
                'approach': 'Direct video RL with simplified expert matching',
                'method_description': 'Model-free RL on video sequences with expert action focus',
                'debugging_applied': True,
                'uses_world_model': False,
                'trainer': trainer
            }
            
        except Exception as e:
            self.logger.error(f"❌ Method 3 failed: {e}")
            import traceback
            traceback.print_exc()
            return {'status': 'failed', 'error': str(e)}
    
    def _run_comprehensive_debugging_analysis(self) -> Dict[str, Any]:
        """Run comprehensive debugging analysis across all methods."""
        
        self.logger.info("🔍 Running Comprehensive Debugging Analysis")
        self.logger.info("-" * 40)
        
        analysis = {
            'performance_comparison': {},
            'debugging_insights': {},
            'recommendations': [],
            'next_steps': []
        }
        
        # Extract performance metrics
        supervised_mAP = self.results.get('method_1_supervised_baseline', {}).get('baseline_mAP', 0.0)
        
        # Method 2 results
        method2 = self.results.get('method_2_world_model_debug', {})
        if method2.get('status') == 'success':
            method2_rl = method2.get('simplified_rl_results', {})
            method2_mAP = method2_rl.get('final_evaluation', {}).get('mAP', 0.0) if method2_rl.get('status') == 'success' else 0.0
            world_model_quality = method2.get('world_model_quality_score', 0.0)
        else:
            method2_mAP = 0.0
            world_model_quality = 0.0
        
        # Method 3 results
        method3 = self.results.get('method_3_simplified_rl', {})
        if method3.get('status') == 'success':
            method3_rl = method3.get('simplified_rl_results', {})
            method3_mAP = method3_rl.get('final_evaluation', {}).get('mAP', 0.0) if method3_rl.get('status') == 'success' else 0.0
        else:
            method3_mAP = 0.0
        
        # Performance comparison
        analysis['performance_comparison'] = {
            'supervised_baseline_mAP': supervised_mAP,
            'world_model_rl_mAP': method2_mAP,
            'direct_video_rl_mAP': method3_mAP,
            'supervised_vs_wm_rl_gap': supervised_mAP - method2_mAP,
            'supervised_vs_direct_rl_gap': supervised_mAP - method3_mAP,
            'rl_performance_relative_to_supervised': {
                'world_model_rl': (method2_mAP / supervised_mAP) if supervised_mAP > 0 else 0.0,
                'direct_video_rl': (method3_mAP / supervised_mAP) if supervised_mAP > 0 else 0.0
            }
        }
        
        # Generate insights and recommendations
        if supervised_mAP > 0.10:  # Good supervised baseline
            if max(method2_mAP, method3_mAP) < 0.05:  # Poor RL performance
                analysis['debugging_insights']['main_issue'] = 'large_rl_vs_supervised_gap'
                analysis['recommendations'].extend([
                    'Focus on reward function alignment with mAP',
                    'Consider behavioral cloning warm-start for RL',
                    'Validate action space conversion and thresholding',
                    'Increase expert demonstration matching weight'
                ])
                
                if world_model_quality < 0.5:
                    analysis['recommendations'].append('World model quality is poor - consider retraining or using direct video RL')
            
            elif max(method2_mAP, method3_mAP) >= 0.05:  # Reasonable RL performance
                analysis['debugging_insights']['main_issue'] = 'moderate_rl_vs_supervised_gap'
                analysis['recommendations'].extend([
                    'RL is showing promise - continue optimization',
                    'Fine-tune hyperparameters and reward weights',
                    'Consider longer training or curriculum learning'
                ])
        else:
            analysis['debugging_insights']['main_issue'] = 'poor_supervised_baseline'
            analysis['recommendations'].extend([
                'Supervised baseline is poor - check data quality and model training',
                'Validate action labeling and preprocessing',
                'Consider different supervised learning approaches'
            ])
        
        # Next steps based on findings
        best_rl_mAP = max(method2_mAP, method3_mAP)
        if best_rl_mAP < 0.02:
            analysis['next_steps'].extend([
                'Implement behavioral cloning initialization',
                'Simplify action space or use different action representation',
                'Validate that expert demonstrations are learnable'
            ])
        elif best_rl_mAP < 0.05:
            analysis['next_steps'].extend([
                'Continue with current approach but optimize hyperparameters',
                'Implement curriculum learning',
                'Add more sophisticated reward shaping'
            ])
        else:
            analysis['next_steps'].extend([
                'Scale up training with more timesteps',
                'Experiment with different RL algorithms',
                'Consider ensemble methods'
            ])
        
        self.logger.info("🔍 Debugging Analysis Complete")
        self.logger.info(f"   Supervised mAP: {supervised_mAP:.4f}")
        self.logger.info(f"   World Model RL mAP: {method2_mAP:.4f}")
        self.logger.info(f"   Direct Video RL mAP: {method3_mAP:.4f}")
        self.logger.info(f"   Main issue: {analysis['debugging_insights'].get('main_issue', 'unknown')}")
        
        return analysis
    
    def _analyze_performance_gaps(self):
        """Analyze and log performance gaps between methods."""
        
        self.logger.info("📊 PERFORMANCE GAP ANALYSIS")
        self.logger.info("=" * 50)
        
        # Extract mAP values
        supervised_mAP = self.results.get('method_1_supervised_baseline', {}).get('baseline_mAP', 0.0)
        
        method2 = self.results.get('method_2_world_model_debug', {})
        method2_mAP = 0.0
        if method2.get('status') == 'success':
            method2_rl = method2.get('simplified_rl_results', {})
            if method2_rl.get('status') == 'success':
                method2_mAP = method2_rl.get('final_evaluation', {}).get('mAP', 0.0)
        
        method3 = self.results.get('method_3_simplified_rl', {})
        method3_mAP = 0.0
        if method3.get('status') == 'success':
            method3_rl = method3.get('simplified_rl_results', {})
            if method3_rl.get('status') == 'success':
                method3_mAP = method3_rl.get('final_evaluation', {}).get('mAP', 0.0)
        
        self.logger.info(f"🎓 Supervised IL Baseline: {supervised_mAP:.4f} mAP")
        self.logger.info(f"🌍 World Model + RL: {method2_mAP:.4f} mAP")
        self.logger.info(f"🎯 Direct Video RL: {method3_mAP:.4f} mAP")
        
        # Calculate gaps
        if supervised_mAP > 0:
            wm_gap = supervised_mAP - method2_mAP
            direct_gap = supervised_mAP - method3_mAP
            
            self.logger.info(f"")
            self.logger.info(f"📉 Performance Gaps:")
            self.logger.info(f"   Supervised vs World Model RL: {wm_gap:.4f} ({wm_gap/supervised_mAP:.1%})")
            self.logger.info(f"   Supervised vs Direct Video RL: {direct_gap:.4f} ({direct_gap/supervised_mAP:.1%})")
            
            # Determine which RL approach is better
            if method2_mAP > method3_mAP:
                self.logger.info(f"🏆 Best RL approach: World Model RL")
            elif method3_mAP > method2_mAP:
                self.logger.info(f"🏆 Best RL approach: Direct Video RL")
            else:
                self.logger.info(f"🤝 RL approaches perform similarly")
            
            # Overall assessment
            best_rl = max(method2_mAP, method3_mAP)
            if best_rl / supervised_mAP > 0.8:
                self.logger.info(f"✅ RL is competitive with supervised learning!")
            elif best_rl / supervised_mAP > 0.5:
                self.logger.info(f"🔶 RL shows promise but needs optimization")
            elif best_rl / supervised_mAP > 0.2:
                self.logger.info(f"⚠️ RL is learning but significant gap remains")
            else:
                self.logger.info(f"❌ RL is far from supervised performance - major issues")
    
    def _save_debugging_results(self):
        """Save all debugging results with comprehensive analysis."""
        
        # Convert results to JSON-serializable format
        json_results = self._convert_for_json(self.results)
        
        # Save detailed results
        import json
        results_path = self.results_dir / 'debugging_results.json'
        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
        
        # Save summary
        summary = self._create_debugging_summary()
        summary_path = self.results_dir / 'debugging_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self.logger.info(f"💾 Debugging results saved to: {self.results_dir}")
        self.logger.info(f"📄 Detailed results: {results_path}")
        self.logger.info(f"📄 Summary: {summary_path}")
    
    def _create_debugging_summary(self) -> Dict:
        """Create a comprehensive debugging summary."""
        
        # Extract key metrics
        supervised_mAP = self.results.get('method_1_supervised_baseline', {}).get('baseline_mAP', 0.0)
        
        # Get RL results
        method2_success = self.results.get('method_2_world_model_debug', {}).get('status') == 'success'
        method3_success = self.results.get('method_3_simplified_rl', {}).get('status') == 'success'
        
        method2_mAP = 0.0
        method3_mAP = 0.0
        
        if method2_success:
            method2_rl = self.results['method_2_world_model_debug'].get('simplified_rl_results', {})
            if method2_rl.get('status') == 'success':
                method2_mAP = method2_rl.get('final_evaluation', {}).get('mAP', 0.0)
        
        if method3_success:
            method3_rl = self.results['method_3_simplified_rl'].get('simplified_rl_results', {})
            if method3_rl.get('status') == 'success':
                method3_mAP = method3_rl.get('final_evaluation', {}).get('mAP', 0.0)
        
        summary = {
            'experiment_type': 'rl_debugging_and_performance_analysis',
            'timestamp': self.results['timestamp'],
            
            'performance_summary': {
                'supervised_baseline_mAP': supervised_mAP,
                'world_model_rl_mAP': method2_mAP,
                'direct_video_rl_mAP': method3_mAP,
                'best_rl_mAP': max(method2_mAP, method3_mAP),
                'rl_vs_supervised_ratio': max(method2_mAP, method3_mAP) / supervised_mAP if supervised_mAP > 0 else 0.0
            },
            
            'debugging_applied': {
                'world_model_quality_evaluation': method2_success,
                'simplified_expert_matching_rewards': True,
                'action_space_analysis': True,
                'comprehensive_monitoring': True,
                'training_visualizations': True
            },
            
            'key_findings': [],
            'recommendations': [],
            'success_metrics': {
                'supervised_baseline_established': supervised_mAP > 0.05,
                'rl_learning_demonstrated': max(method2_mAP, method3_mAP) > 0.01,
                'debugging_systems_functional': method2_success or method3_success
            }
        }
        
        # Generate findings and recommendations based on results
        if supervised_mAP > 0.10:
            summary['key_findings'].append("Strong supervised baseline established")
        elif supervised_mAP > 0.05:
            summary['key_findings'].append("Reasonable supervised baseline")
        else:
            summary['key_findings'].append("Weak supervised baseline - data quality issues?")
        
        best_rl = max(method2_mAP, method3_mAP)
        if best_rl > 0.05:
            summary['key_findings'].append("RL showing reasonable performance")
        elif best_rl > 0.02:
            summary['key_findings'].append("RL learning but below target performance")
        else:
            summary['key_findings'].append("RL struggling to learn - major issues identified")
        
        # Add recommendations from debugging analysis
        debug_analysis = self.results.get('debugging_analysis', {})
        if 'recommendations' in debug_analysis:
            summary['recommendations'].extend(debug_analysis['recommendations'])
        
        return summary
    
    def _convert_for_json(self, obj: Any) -> Any:
        """Convert objects to JSON-serializable format."""
        
        if isinstance(obj, dict):
            return {k: self._convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(item) for item in obj]
        elif isinstance(obj, (torch.Tensor, np.ndarray)):
            return obj.tolist()
        elif isinstance(obj, np.number):
            return obj.item()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, Path):
            return str(obj)
        elif hasattr(obj, '__dict__') and not callable(obj):
            return str(obj)
        else:
            return obj


def main():
    """Main function to run the RL debugging experiment."""
    
    print("🔍 RL DEBUGGING EXPERIMENT")
    print("=" * 60)
    print("Goal: Understand why RL can't reach supervised learning performance")
    print("Focus: Expert action matching + comprehensive debugging")
    print()
    print("🎓 Method 1: Supervised IL baseline")
    print("🌍 Method 2: World Model + Debugging")
    print("🎯 Method 3: Simplified RL (expert matching only)")
    print()
    
    # Choose config file
    config_path = 'config_dgx_all_v7.yaml'
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        return
    else:
        print(f"📄 Using config: {config_path}")
    
    try:
        import argparse
        parser = argparse.ArgumentParser(description="Run RL debugging experiment")
        parser.add_argument('--config', type=str, default=config_path, help="Path to config file")
        args = parser.parse_args()
        
        # Run debugging experiment
        experiment = DebuggingExperimentRunner(args.config)
        results = experiment.run_debugging_comparison()
        
        print("\n🎉 RL DEBUGGING EXPERIMENT COMPLETED!")
        print("=" * 50)
        print(f"📁 Results saved to: {experiment.results_dir}")
        
        # Print key findings
        debug_analysis = results.get('debugging_analysis', {})
        performance_comparison = debug_analysis.get('performance_comparison', {})
        
        print(f"📊 Performance Summary:")
        print(f"   Supervised: {performance_comparison.get('supervised_baseline_mAP', 0.0):.4f} mAP")
        print(f"   World Model RL: {performance_comparison.get('world_model_rl_mAP', 0.0):.4f} mAP")
        print(f"   Direct Video RL: {performance_comparison.get('direct_video_rl_mAP', 0.0):.4f} mAP")
        
        print(f"🔍 Key Insights:")
        insights = debug_analysis.get('debugging_insights', {})
        if 'main_issue' in insights:
            print(f"   Main issue: {insights['main_issue']}")
        
        recommendations = debug_analysis.get('recommendations', [])
        if recommendations:
            print(f"   Recommendations:")
            for rec in recommendations[:3]:  # Show top 3
                print(f"     - {rec}")
        
        print(f"🚀 This experiment provides comprehensive insights into RL training issues!")
        
    except Exception as e:
        print(f"\n❌ RL DEBUGGING EXPERIMENT FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
