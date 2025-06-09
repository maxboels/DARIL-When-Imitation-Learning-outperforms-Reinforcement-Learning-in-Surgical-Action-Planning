#!/usr/bin/env python3
"""
Corrected Evaluation-Only Script for Surgical RL Comparison
Based on diagnostic results - handles checkpoint metadata format correctly
"""
import os
import yaml
import warnings
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

# Import the separate models and their components
from models.autoregressive_il_model import AutoregressiveILModel
from models.conditional_world_model import ConditionalWorldModel

# Import separate datasets
from datasets.world_model_dataset import create_world_model_dataloaders

# Import existing components for Method 3 and evaluation
from datasets.cholect50 import load_cholect50_data

# Import evaluation framework
from evaluation.integrated_evaluation import run_integrated_evaluation
from utils.logger import SimpleLogger

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="stable_baselines3")

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EvaluationOnlyRunner:
    """
    Corrected evaluation-only runner that properly handles checkpoint metadata format.
    """
    
    def __init__(self, config_path: str = 'config_local_debug.yaml', pretrained_dir: str = None):
        print("🔍 Initializing CORRECTED EVALUATION-ONLY Mode")
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set pretrained directory
        if pretrained_dir is None:
            # Use the latest results directory
            results_base = Path("results")
            if results_base.exists():
                subdirs = [d for d in results_base.iterdir() if d.is_dir()]
                if subdirs:
                    latest_dir = max(subdirs, key=lambda d: d.stat().st_mtime)
                    pretrained_dir = str(latest_dir)
                    print(f"📁 Auto-detected latest results: {pretrained_dir}")
        
        self.pretrained_dir = Path(pretrained_dir) if pretrained_dir else None
        
        if not self.pretrained_dir or not self.pretrained_dir.exists():
            raise ValueError(f"❌ Pretrained directory not found: {self.pretrained_dir}")
        
        # Create evaluation directory with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.experiment_name = f"eval_corrected_{timestamp}"
        self.results_dir = Path("results") / self.experiment_name
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logger
        self.logger = SimpleLogger(
            log_dir=str(self.results_dir),
            name="SuRL_Eval_Corrected",
            use_shared_timestamp=True
        )
        
        # Results storage
        self.results = {
            'experiment_name': self.experiment_name,
            'config': self.config,
            'timestamp': timestamp,
            'results_dir': str(self.results_dir),
            'pretrained_dir': str(self.pretrained_dir),
            'mode': 'evaluation_only_corrected'
        }
        
        self.logger.info(f"🔍 Corrected Evaluation Experiment: {self.experiment_name}")
        self.logger.info(f"📁 Loading models from: {self.pretrained_dir}")
        self.logger.info(f"📁 Saving results to: {self.results_dir}")
    
    def run_evaluation_only(self) -> Dict[str, Any]:
        """Run corrected evaluation-only pipeline using pre-trained models."""
        
        self.logger.info("🔍 Starting CORRECTED EVALUATION-ONLY Comparison")
        self.logger.info("=" * 60)
        self.logger.info("🎯 Goal: Debug evaluation metrics using properly loaded models")
        
        # Load data
        train_data, test_data = self._load_data()
        
        # Load pre-trained models with corrected format handling
        self.logger.info("📦 Loading Pre-trained Models (Corrected Format)")
        method1_results = self._load_method1_corrected()
        self.results['method_1_autoregressive_il'] = method1_results
        
        method2_results = self._load_method2_corrected()
        self.results['method_2_conditional_world_model'] = method2_results
        
        method3_results = self._load_method3_corrected()
        self.results['method_3_direct_video_rl'] = method3_results
        
        # Create test loaders for evaluation
        self.logger.info("🔧 Creating test data loaders...")
        self.test_loaders = self._create_test_loaders_corrected(test_data)
        
        # Comprehensive evaluation
        self.logger.info("📊 Running Comprehensive Evaluation")
        evaluation_results = self._run_comprehensive_evaluation()
        self.results['comprehensive_evaluation'] = evaluation_results
        
        # Analysis and comparison
        self.logger.info("🏆 Analyzing Results and Metrics")
        self._print_evaluation_analysis(self.results)
        
        # Save results
        self._save_evaluation_results()
        
        return self.results
    
    def _load_data(self) -> Tuple[List[Dict], List[Dict]]:
        """Load test data for evaluation."""
        
        self.logger.info("📂 Loading CholecT50 data for evaluation...")
        
        # Use smaller datasets for debugging
        train_videos = 2
        train_data = load_cholect50_data(
            self.config, self.logger, split='train', max_videos=train_videos
        )
        
        test_videos = 3
        test_data = load_cholect50_data(
            self.config, self.logger, split='test', max_videos=test_videos
        )
        
        self.logger.info(f"✅ Data loaded for evaluation")
        self.logger.info(f"   Training videos: {len(train_data)} (for context)")
        self.logger.info(f"   Test videos: {len(test_data)} (for evaluation)")
        
        return train_data, test_data
    
    def _load_method1_corrected(self) -> Dict[str, Any]:
        """Load pre-trained Autoregressive IL model with corrected format handling."""
        
        self.logger.info("🎓 Loading Method 1: Autoregressive IL (Corrected)")
        
        try:
            # Find the best model
            checkpoints_dir = self.pretrained_dir / "2025-06-07_18-44" / "checkpoints"
            best_model_path = checkpoints_dir / "autoregressive_il_best_epoch_1.pt"
            
            if not best_model_path.exists():
                best_model_path = checkpoints_dir / "autoregressive_il_final.pt"
            
            if not best_model_path.exists():
                raise FileNotFoundError(f"No autoregressive IL model found in {checkpoints_dir}")
            
            self.logger.info(f"📦 Loading AutoregressiveIL from: {best_model_path}")
            
            # Load checkpoint with metadata
            checkpoint = torch.load(best_model_path, map_location=DEVICE, weights_only=False)
            
            # Extract configuration and state dict
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                saved_config = checkpoint.get('config', {})
                self.logger.info(f"📋 Using saved config: {saved_config}")
                
                # Use saved config if available, otherwise use current config
                model_config = saved_config if saved_config else self.config['models']['autoregressive_il']
            else:
                # Fallback if checkpoint format is different
                state_dict = checkpoint
                model_config = self.config['models']['autoregressive_il']
                self.logger.info(f"📋 Using current config (no saved config found)")
            
            # Create model instance with correct config
            model = AutoregressiveILModel(
                hidden_dim=model_config['hidden_dim'],
                embedding_dim=model_config['embedding_dim'],
                n_layer=model_config['n_layer'],
                num_action_classes=model_config['num_action_classes'],
                dropout=model_config.get('dropout', 0.1)
            ).to(DEVICE)
            
            # Load state dict
            model.load_state_dict(state_dict)
            model.eval()
            
            self.logger.info(f"✅ Method 1 loaded successfully with corrected format")
            
            return {
                'status': 'success',
                'model': model,  # Store the actual model for evaluation
                'model_path': str(best_model_path),
                'model_type': 'AutoregressiveILModel',
                'approach': 'Pure causal frame generation → action prediction',
                'method_description': 'Autoregressive IL without action conditioning',
                'loaded_from_pretrained': True,
                'config_used': model_config
            }
            
        except Exception as e:
            self.logger.error(f"❌ Method 1 loading failed: {e}")
            import traceback
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            return {'status': 'failed', 'error': str(e)}
    
    def _load_method2_corrected(self) -> Dict[str, Any]:
        """Load pre-trained Conditional World Model + RL models with corrected format."""
        
        self.logger.info("🌍 Loading Method 2: Conditional World Model + RL (Corrected)")
        
        try:
            # Load world model
            checkpoints_dir = self.pretrained_dir / "2025-06-07_18-44" / "checkpoints"
            world_model_path = checkpoints_dir / "world_model_best_epoch_2.pt"
            
            if not world_model_path.exists():
                world_model_path = checkpoints_dir / "world_model_final.pt"
            
            if not world_model_path.exists():
                raise FileNotFoundError(f"No world model found in {checkpoints_dir}")
            
            self.logger.info(f"📦 Loading ConditionalWorldModel from: {world_model_path}")
            
            # Load world model checkpoint with metadata
            checkpoint = torch.load(world_model_path, map_location=DEVICE, weights_only=False)
            
            # Extract configuration and state dict
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                saved_config = checkpoint.get('config', {})
                reward_types = checkpoint.get('reward_types', [])
                self.logger.info(f"📋 Using saved config: {saved_config}")
                self.logger.info(f"📋 Reward types: {reward_types}")
                
                model_config = saved_config if saved_config else self.config['models']['conditional_world_model']
            else:
                state_dict = checkpoint
                model_config = self.config['models']['conditional_world_model']
                self.logger.info(f"📋 Using current config (no saved config found)")
            
            # Create world model instance
            world_model = ConditionalWorldModel(
                hidden_dim=model_config['hidden_dim'],
                embedding_dim=model_config['embedding_dim'],
                action_embedding_dim=model_config['action_embedding_dim'],
                n_layer=model_config['n_layer'],
                num_action_classes=model_config['num_action_classes'],
                dropout=model_config.get('dropout', 0.1)
            ).to(DEVICE)
            
            # Load world model weights
            world_model.load_state_dict(state_dict)
            world_model.eval()
            
            self.logger.info(f"✅ World model loaded successfully")
            
            # Load RL models
            rl_dir = self.pretrained_dir / "2025-06-07_18-44" / "rl_world_model_simulation"
            rl_models = {}
            
            # Load PPO model
            ppo_path = rl_dir / "ppo_conditional_world_model.zip"
            if ppo_path.exists():
                from stable_baselines3 import PPO
                ppo_model = PPO.load(str(ppo_path), device='cpu')  # Load on CPU to avoid warnings
                rl_models['ppo'] = {
                    'status': 'success',
                    'model': ppo_model,
                    'model_path': str(ppo_path),
                    'mean_reward': -400.0  # From logs
                }
                self.logger.info(f"✅ PPO model loaded")
            
            # Load A2C model
            a2c_path = rl_dir / "a2c_conditional_world_model.zip"
            if a2c_path.exists():
                from stable_baselines3 import A2C
                a2c_model = A2C.load(str(a2c_path), device='cpu')  # Load on CPU to avoid warnings
                rl_models['a2c'] = {
                    'status': 'success',
                    'model': a2c_model,
                    'model_path': str(a2c_path),
                    'mean_reward': -405.0  # From logs
                }
                self.logger.info(f"✅ A2C model loaded")
            
            return {
                'status': 'success',
                'world_model': world_model,  # Store the actual world model
                'world_model_path': str(world_model_path),
                'rl_models': rl_models,
                'model_type': 'ConditionalWorldModel',
                'approach': 'Action-conditioned world model + RL simulation',
                'method_description': 'World model-based RL with action conditioning',
                'loaded_from_pretrained': True,
                'config_used': model_config
            }
            
        except Exception as e:
            self.logger.error(f"❌ Method 2 loading failed: {e}")
            import traceback
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            return {'status': 'failed', 'error': str(e)}
    
    def _load_method3_corrected(self) -> Dict[str, Any]:
        """Load pre-trained Direct Video RL models."""
        
        self.logger.info("📹 Loading Method 3: Direct Video RL")
        
        try:
            rl_dir = self.pretrained_dir / "2025-06-07_18-44" / "direct_video_rl"
            rl_models = {}
            
            # Load PPO model
            ppo_path = rl_dir / "ppo_direct_video.zip"
            if ppo_path.exists():
                from stable_baselines3 import PPO
                ppo_model = PPO.load(str(ppo_path), device='cpu')  # Load on CPU to avoid warnings
                rl_models['ppo'] = {
                    'status': 'success',
                    'model': ppo_model,
                    'model_path': str(ppo_path),
                    'mean_reward': 79.5  # From logs
                }
                self.logger.info(f"✅ PPO model loaded")
            
            # Load A2C model
            a2c_path = rl_dir / "a2c_direct_video.zip"
            if a2c_path.exists():
                from stable_baselines3 import A2C
                a2c_model = A2C.load(str(a2c_path), device='cpu')  # Load on CPU to avoid warnings
                rl_models['a2c'] = {
                    'status': 'success',
                    'model': a2c_model,
                    'model_path': str(a2c_path),
                    'mean_reward': 76.5  # From logs
                }
                self.logger.info(f"✅ A2C model loaded")
            
            return {
                'status': 'success',
                'rl_models': rl_models,
                'model_type': 'DirectVideoRL',
                'approach': 'Direct RL on video sequences (no world model)',
                'method_description': 'Model-free RL on offline video episodes',
                'loaded_from_pretrained': True
            }
            
        except Exception as e:
            self.logger.error(f"❌ Method 3 loading failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def _create_test_loaders_corrected(self, test_data: List[Dict]) -> Dict[str, Any]:
        """Create test data loaders with corrected approach to avoid empty dataset issues."""
        
        # Use minimal training data (first test video) to avoid empty dataset sampler issues
        minimal_train_data = test_data[:1] if test_data else []
        
        # Create world model test loaders (which handles individual video loaders correctly)
        _, world_model_test_loaders, _ = create_world_model_dataloaders(
            config=self.config['data'],
            train_data=minimal_train_data,  # Use minimal data instead of empty
            test_data=test_data,
            batch_size=self.config['training']['batch_size'],
            num_workers=0  # Disable multiprocessing for debugging
        )
        
        self.logger.info(f"✅ Created test loaders for {len(test_data)} videos")
        self.logger.info(f"📊 Available test videos: {list(world_model_test_loaders.keys())}")
        
        return world_model_test_loaders
    
    def _run_comprehensive_evaluation(self) -> Dict[str, Any]:
        """Run comprehensive evaluation with loaded models."""
        
        self.logger.info("📊 Running comprehensive evaluation with corrected model loading...")
        
        # Run the same evaluation as the original experiment
        evaluation_results = run_integrated_evaluation(
            experiment_results=self.results,
            test_data=self.test_loaders,
            results_dir=str(self.results_dir),
            logger=self.logger,
            horizon=self.config['evaluation']['prediction_horizon']
        )
        
        return evaluation_results
    
    def _print_evaluation_analysis(self, results: Dict):
        """Print detailed evaluation analysis for debugging."""
        
        self.logger.info("🔍 CORRECTED EVALUATION ANALYSIS")
        self.logger.info("=" * 60)
        
        # Print model loading status
        self.logger.info("📦 MODEL LOADING STATUS:")
        
        method1 = results.get('method_1_autoregressive_il', {})
        if method1.get('status') == 'success':
            self.logger.info(f"   ✅ Method 1 (Autoregressive IL): Loaded successfully")
            self.logger.info(f"      Config: {method1.get('config_used', {})}")
        else:
            self.logger.info(f"   ❌ Method 1: {method1.get('error', 'Unknown error')}")
        
        method2 = results.get('method_2_conditional_world_model', {})
        if method2.get('status') == 'success':
            self.logger.info(f"   ✅ Method 2 (World Model + RL): Loaded successfully")
            rl_models = method2.get('rl_models', {})
            for alg, model_info in rl_models.items():
                if model_info.get('status') == 'success':
                    self.logger.info(f"      ✅ {alg.upper()}: Loaded")
        else:
            self.logger.info(f"   ❌ Method 2: {method2.get('error', 'Unknown error')}")
        
        method3 = results.get('method_3_direct_video_rl', {})
        if method3.get('status') == 'success':
            self.logger.info(f"   ✅ Method 3 (Direct Video RL): Loaded successfully")
            rl_models = method3.get('rl_models', {})
            for alg, model_info in rl_models.items():
                if model_info.get('status') == 'success':
                    self.logger.info(f"      ✅ {alg.upper()}: Loaded")
        else:
            self.logger.info(f"   ❌ Method 3: {method3.get('error', 'Unknown error')}")
        
        # Extract evaluation results if available
        eval_results = results.get('comprehensive_evaluation', {})
        if eval_results and 'results' in eval_results:
            self.logger.info("")
            self.logger.info("📊 EVALUATION RESULTS:")
            
            if 'aggregate_results' in eval_results['results']:
                aggregate = eval_results['results']['aggregate_results']
                
                # Single-step results
                single_step = aggregate.get('single_step_comparison', {})
                if single_step:
                    self.logger.info("   🎯 Single-step Action Prediction:")
                    for method, stats in single_step.items():
                        mAP = stats.get('mean_mAP', 0)
                        std = stats.get('std_mAP', 0)
                        self.logger.info(f"      {method}: {mAP:.4f} ± {std:.4f} mAP")
                
                # Planning results
                planning = aggregate.get('planning_analysis', {})
                if planning:
                    self.logger.info("   🚀 Planning Stability:")
                    for method, stats in planning.items():
                        stability = stats.get('mean_planning_stability', 0)
                        self.logger.info(f"      {method}: {stability:.4f}")
        else:
            self.logger.warning("⚠️ No evaluation results found - check if evaluation completed successfully")
    
    def _save_evaluation_results(self):
        """Save evaluation results with debugging information."""
        
        # Convert results to JSON-serializable format
        json_results = self._convert_for_json(self.results)
        
        # Save detailed results
        import json
        results_path = self.results_dir / 'corrected_evaluation_results.json'
        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
        
        self.logger.info(f"💾 Corrected evaluation results saved:")
        self.logger.info(f"   📄 Results: {results_path}")
    
    def _convert_for_json(self, obj):
        """Convert objects to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(item) for item in obj]
        elif isinstance(obj, (torch.Tensor, np.ndarray)):
            return obj.tolist()
        elif isinstance(obj, Path):
            return str(obj)
        elif hasattr(obj, '__dict__') and not callable(obj):
            return str(obj)
        else:
            return obj

def main():
    """Main function to run evaluation-only comparison."""
    
    print("🔍 EVALUATION-ONLY SURGICAL RL COMPARISON")
    print("=" * 60)
    print("Purpose: Debug evaluation metrics using pre-trained models")
    print()
    print("🎯 Benefits:")
    print("   ✅ Skip long training times")
    print("   ✅ Debug evaluation metrics")
    print("   ✅ Understand metric computation")
    print("   ✅ Fast iteration for debugging")
    print()
    
    # Configuration
    config_path = 'config_eval_debug.yaml'
    pretrained_dir = '/home/maxboels/projects/surl/results/2025-06-07_18-44-58'
    
    # Allow command line override
    import sys
    if len(sys.argv) > 1:
        pretrained_dir = sys.argv[1]
        print(f"📁 Using specified pretrained dir: {pretrained_dir}")
    
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        return
    
    try:
        # Run evaluation-only comparison
        evaluator = EvaluationOnlyRunner(config_path, pretrained_dir)
        results = evaluator.run_evaluation_only()
        
        print("\n🎉 EVALUATION-ONLY COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        print(f"📁 Results saved to: {evaluator.results_dir}")
        print(f"🔬 Evaluation metrics computed and analyzed")
        print(f"🎯 Perfect for debugging and understanding metrics!")
        
    except Exception as e:
        print(f"\n❌ EVALUATION-ONLY FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
