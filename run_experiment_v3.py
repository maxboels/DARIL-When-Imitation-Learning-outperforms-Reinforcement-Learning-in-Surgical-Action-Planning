#!/usr/bin/env python3
"""
UPDATED Complete Experimental Comparison with Separate Models:
1. Autoregressive Imitation Learning (Method 1) - Pure causal generation → actions
2. RL with ConditionalWorldModel Simulation (Method 2) - Action-conditioned simulation
3. RL with Offline Video Episodes (Method 3) - Direct video interaction
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
from datasets.autoregressive_dataset import create_autoregressive_dataloaders
from datasets.world_model_dataset import create_world_model_dataloaders

# Import separate trainers
from training.autoregressive_il_trainer import AutoregressiveILTrainer
from training.world_model_trainer import WorldModelTrainer
from training.world_model_rl_trainer import WorldModelRLTrainer

# Import existing components for Method 3 and evaluation
from datasets.cholect50 import load_cholect50_data
from environment.direct_video_env import DirectVideoSB3Trainer, test_direct_video_environment
from evaluation.integrated_evaluation_framework import run_integrated_evaluation
from evaluation.paper_generator import generate_research_paper
from utils.logger import SimpleLogger

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SeparateModelsSurgicalComparison:
    """
    UPDATED Experimental comparison using separate models for each method:
    1. Method 1: AutoregressiveILModel (frames → causal generation → actions)
    2. Method 2: ConditionalWorldModel (state + action → next_state + rewards)
    3. Method 3: Direct Video RL (no model, real video interaction)
    """
    
    def __init__(self, config_path: str = 'config_local_debug.yaml'):
        print("🏗️ Initializing Separate Models Surgical Comparison")
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Create experiment directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = f"separate_models_comparison_{timestamp}"
        self.results_dir = Path("results") / self.experiment_name
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logger
        self.logger = SimpleLogger(
            log_dir=str(self.results_dir),
            name="SeparateModelsComparison",
            use_shared_timestamp=False  # Each experiment gets its own timestamp
        )
        
        # Results storage
        self.results = {
            'experiment_name': self.experiment_name,
            'config': self.config,
            'timestamp': timestamp,
            'results_dir': str(self.results_dir)
        }
        
        self.logger.info(f"🎯 Experiment: {self.experiment_name}")
        self.logger.info(f"📁 Results dir: {self.results_dir}")
    
    def run_complete_comparison(self) -> Dict[str, Any]:
        """Run the complete three-method comparison."""
        
        self.logger.info("🚀 Starting Complete Separate Models Comparison")
        self.logger.info("=" * 60)
        
        try:
            # Load data
            train_data, test_data = self._load_data()
            
            # Method 1: Autoregressive IL
            self.logger.info("🎓 Running Method 1: Autoregressive IL")
            method1_results = self._run_method1_autoregressive_il(train_data, test_data)
            self.results['method_1_autoregressive_il'] = method1_results
            
            # Method 2: Conditional World Model + RL
            self.logger.info("🌍 Running Method 2: Conditional World Model + RL")
            method2_results = self._run_method2_conditional_world_model(train_data, test_data)
            self.results['method_2_conditional_world_model'] = method2_results
            
            # Method 3: Direct Video RL
            self.logger.info("📹 Running Method 3: Direct Video RL")
            method3_results = self._run_method3_direct_video_rl(train_data, test_data)
            self.results['method_3_direct_video_rl'] = method3_results
            
            # Comprehensive evaluation
            self.logger.info("📊 Running Comprehensive Evaluation")
            evaluation_results = self._run_comprehensive_evaluation(test_data)
            self.results['comprehensive_evaluation'] = evaluation_results
            
            # Analysis and comparison
            self._print_method_comparison(self.results)
            self._print_architectural_insights()
            
            # Save results
            self._save_complete_results()
            
            return self.results
            
        except Exception as e:
            self.logger.error(f"❌ Experiment failed: {e}")
            import traceback
            traceback.print_exc()
            return {'status': 'failed', 'error': str(e)}
    
    # def _load_data(self) -> Tuple[List[Dict], List[Dict]]:
    #     """Load and prepare data for all methods."""
        
    #     self.logger.info("📂 Loading CholecT50 data...")
        
    #     # Load using existing function with logger argument
    #     train_data, test_data = load_cholect50_data(self.config, self.logger)
        
    #     self.logger.info(f"✅ Data loaded successfully")
    #     self.logger.info(f"   Training videos: {len(train_data)}")
    #     self.logger.info(f"   Test videos: {len(test_data)}")
        
    #     return train_data, test_data
    
    def _load_data(self) -> Tuple[List[Dict], List[Dict]]:
        """Load and prepare training and test data."""

        self.logger.info("📂 Loading CholecT50 data...")
        
        train_videos = self.config.get('experiment', {}).get('train', {}).get('max_videos', 2)
        train_data = load_cholect50_data(
            self.config, self.logger, split='train', max_videos=train_videos
        )
        
        test_videos = self.config.get('experiment', {}).get('test', {}).get('max_videos', 1)
        test_data = load_cholect50_data(
            self.config, self.logger, split='test', max_videos=test_videos
        )
        self.logger.info(f"✅ Data loaded successfully")
        self.logger.info(f"   Training videos: {len(train_data)}")
        self.logger.info(f"   Test videos: {len(test_data)}")

        return train_data, test_data
    
    def _run_method1_autoregressive_il(self, train_data: List[Dict], test_data: List[Dict]) -> Dict[str, Any]:
        """Method 1: Autoregressive Imitation Learning."""
        
        self.logger.info("🎓 Method 1: Autoregressive IL Training")
        self.logger.info("-" * 40)
        
        try:
            # Create model
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
                'approach': 'Pure causal frame generation → action prediction',
                'evaluation': evaluation_results,
                'method_description': 'Autoregressive IL without action conditioning'
            }
            
        except Exception as e:
            self.logger.error(f"❌ Method 1 failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def _run_method2_conditional_world_model(self, train_data: List[Dict], test_data: List[Dict]) -> Dict[str, Any]:
        """Method 2: Conditional World Model + RL."""
        
        self.logger.info("🌍 Method 2: Conditional World Model + RL")
        self.logger.info("-" * 40)
        
        try:
            # Create world model
            world_model = ConditionalWorldModel(
                hidden_dim=self.config['models']['conditional_world_model']['hidden_dim'],
                embedding_dim=self.config['models']['conditional_world_model']['embedding_dim'],
                action_embedding_dim=self.config['models']['conditional_world_model']['action_embedding_dim'],
                n_layer=self.config['models']['conditional_world_model']['n_layer'],
                num_action_classes=self.config['models']['conditional_world_model']['num_action_classes'],
                dropout=self.config['models']['conditional_world_model']['dropout']
            ).to(DEVICE)
            
            # Create datasets for world model training
            train_loader, test_loader, simulation_loader = create_world_model_dataloaders(
                config=self.config['data'],
                train_data=train_data,
                test_data=test_data,
                batch_size=self.config['training']['batch_size'],
                num_workers=self.config['training']['num_workers']
            )
            
            # Step 1: Train world model
            world_model_trainer = WorldModelTrainer(
                model=world_model,
                config=self.config,
                logger=self.logger,
                device=DEVICE
            )
            
            best_world_model_path = world_model_trainer.train(train_loader, test_loader)
            world_model_evaluation = world_model_trainer.evaluate_model(test_loader)
            
            # Step 2: Train RL agents using world model
            rl_trainer = WorldModelRLTrainer(
                world_model=world_model,
                config=self.config,
                logger=self.logger,
                device=DEVICE
            )
            
            rl_results = rl_trainer.train_all_algorithms(
                train_data=train_data,
                timesteps=self.config.get('rl_training', {}).get('timesteps', 10000)
            )
            
            return {
                'status': 'success',
                'world_model_path': best_world_model_path,
                'world_model_evaluation': world_model_evaluation,
                'rl_models': rl_results,
                'model_type': 'ConditionalWorldModel',
                'approach': 'Action-conditioned world model + RL simulation',
                'method_description': 'World model-based RL with action conditioning'
            }
            
        except Exception as e:
            self.logger.error(f"❌ Method 2 failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def _run_method3_direct_video_rl(self, train_data: List[Dict], test_data: List[Dict]) -> Dict[str, Any]:
        """Method 3: Direct Video RL (no world model)."""
        
        self.logger.info("📹 Method 3: Direct Video RL")
        self.logger.info("-" * 40)
        
        try:
            # Test environment first
            env_works = test_direct_video_environment(train_data, self.config)
            if not env_works:
                raise RuntimeError("Direct video environment test failed")
            
            # Create RL trainer for direct video interaction
            rl_trainer = DirectVideoSB3Trainer(
                video_data=train_data,
                config=self.config,
                logger=self.logger,
                device=DEVICE
            )
            
            # Train RL algorithms directly on videos
            rl_results = rl_trainer.train_all_algorithms(
                timesteps=self.config.get('rl_training', {}).get('timesteps', 10000)
            )
            
            return {
                'status': 'success',
                'rl_models': rl_results,
                'model_type': 'DirectVideoRL',
                'approach': 'Direct RL on video sequences (no world model)',
                'method_description': 'Model-free RL on offline video episodes'
            }
            
        except Exception as e:
            self.logger.error(f"❌ Method 3 failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def _run_comprehensive_evaluation(self, test_data: List[Dict]) -> Dict[str, Any]:
        """Run comprehensive evaluation across all methods."""
        
        self.logger.info("📊 Running Comprehensive Cross-Method Evaluation")
        self.logger.info("-" * 50)
        
        try:
            # Use the integrated evaluation framework
            evaluation_results = run_integrated_evaluation(
                experiment_results=self.results,
                test_data=test_data,
                results_dir=str(self.results_dir),
                logger=self.logger,
                horizon=self.config['evaluation']['prediction_horizon']
            )
            
            return evaluation_results
            
        except Exception as e:
            self.logger.error(f"❌ Comprehensive evaluation failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def _print_method_comparison(self, aggregate_results: Dict):
        """Print comparison of all three methods."""
        
        self.logger.info("🏆 THREE-METHOD COMPARISON RESULTS")
        self.logger.info("=" * 60)
        
        # Method 1 results
        method1 = aggregate_results.get('method_1_autoregressive_il', {})
        if method1.get('status') == 'success':
            eval_results = method1.get('evaluation', {}).get('overall_metrics', {})
            self.logger.info(f"🎓 Method 1 (Autoregressive IL):")
            self.logger.info(f"   Status: ✅ Success")
            self.logger.info(f"   Action mAP: {eval_results.get('action_mAP', 0):.4f}")
            self.logger.info(f"   Exact Match: {eval_results.get('action_exact_match', 0):.4f}")
            self.logger.info(f"   Approach: Pure causal generation → actions")
        else:
            self.logger.info(f"🎓 Method 1: ❌ Failed - {method1.get('error', 'Unknown')}")
        
        # Method 2 results
        method2 = aggregate_results.get('method_2_conditional_world_model', {})
        if method2.get('status') == 'success':
            self.logger.info(f"🌍 Method 2 (Conditional World Model + RL):")
            self.logger.info(f"   Status: ✅ Success")
            
            # World model performance
            wm_eval = method2.get('world_model_evaluation', {}).get('overall_metrics', {})
            self.logger.info(f"   World Model State Loss: {wm_eval.get('state_loss', 0):.4f}")
            self.logger.info(f"   World Model Reward Loss: {wm_eval.get('total_reward_loss', 0):.4f}")
            
            # RL performance
            rl_models = method2.get('rl_models', {})
            successful_rl = [alg for alg, res in rl_models.items() if res.get('status') == 'success']
            self.logger.info(f"   RL Algorithms: {successful_rl}")
            for alg in successful_rl:
                reward = rl_models[alg].get('mean_reward', 0)
                self.logger.info(f"     {alg}: {reward:.3f} reward")
            self.logger.info(f"   Approach: Action-conditioned simulation + RL")
        else:
            self.logger.info(f"🌍 Method 2: ❌ Failed - {method2.get('error', 'Unknown')}")
        
        # Method 3 results
        method3 = aggregate_results.get('method_3_direct_video_rl', {})
        if method3.get('status') == 'success':
            self.logger.info(f"📹 Method 3 (Direct Video RL):")
            self.logger.info(f"   Status: ✅ Success")
            
            rl_models = method3.get('rl_models', {})
            successful_rl = [alg for alg, res in rl_models.items() if res.get('status') == 'success']
            self.logger.info(f"   RL Algorithms: {successful_rl}")
            for alg in successful_rl:
                reward = rl_models[alg].get('mean_reward', 0)
                self.logger.info(f"     {alg}: {reward:.3f} reward")
            self.logger.info(f"   Approach: Model-free RL on real video frames")
        else:
            self.logger.info(f"📹 Method 3: ❌ Failed - {method3.get('error', 'Unknown')}")
        
        # Summary
        successful_methods = []
        if method1.get('status') == 'success':
            successful_methods.append("Method 1 (IL)")
        if method2.get('status') == 'success':
            successful_methods.append("Method 2 (World Model RL)")
        if method3.get('status') == 'success':
            successful_methods.append("Method 3 (Direct RL)")
        
        self.logger.info(f"")
        self.logger.info(f"🎯 Successful Methods: {successful_methods}")
        self.logger.info(f"📊 Total Methods Tested: 3")
        self.logger.info(f"✅ Success Rate: {len(successful_methods)}/3")
    
    def _get_architectural_info(self, method: str) -> str:
        """Get architectural information for each method."""
        
        architectures = {
            'method_1': 'GPT-2 Autoregressive → Frame Generation → Action Prediction',
            'method_2': 'Transformer + Action Conditioning → State/Reward Prediction → RL',
            'method_3': 'Direct Policy Learning on Video Frames (No Model)'
        }
        return architectures.get(method, 'Unknown Architecture')
    
    def _print_architectural_insights(self):
        """Print architectural insights and research contributions."""
        
        self.logger.info("")
        self.logger.info("🏗️ ARCHITECTURAL INSIGHTS")
        self.logger.info("=" * 50)
        
        self.logger.info("🎓 Method 1 (Autoregressive IL):")
        self.logger.info("   • Pure causal modeling without action conditioning")
        self.logger.info("   • Actions emerge from learned frame representations")
        self.logger.info("   • Good for capturing temporal patterns in demonstrations")
        self.logger.info("   • Limited exploration beyond expert behavior")
        
        self.logger.info("🌍 Method 2 (Conditional World Model):")
        self.logger.info("   • Explicit action conditioning enables true simulation")
        self.logger.info("   • Can explore beyond expert demonstrations")
        self.logger.info("   • Learns forward dynamics: state + action → next_state")
        self.logger.info("   • Enables model-based RL and planning")
        
        self.logger.info("📹 Method 3 (Direct Video RL):")
        self.logger.info("   • Model-free approach using real video frames")
        self.logger.info("   • No simulation, limited to existing video sequences")
        self.logger.info("   • Direct policy optimization on offline data")
        self.logger.info("   • Baseline for comparison with model-based approaches")
        
        # Research implications
        benefits = self._analyze_architectural_benefits()
        self.logger.info("")
        self.logger.info("🔬 RESEARCH IMPLICATIONS:")
        for method, benefit_list in benefits.items():
            self.logger.info(f"   {method}:")
            for benefit in benefit_list:
                self.logger.info(f"     • {benefit}")
    
    def _analyze_architectural_benefits(self) -> Dict[str, Any]:
        """Analyze architectural benefits of each approach."""
        
        return {
            'Autoregressive IL': [
                'Simple training objective (supervised learning)',
                'Leverages pre-trained language model knowledge',
                'Good for dense action sequence modeling',
                'Interpretable causal generation process'
            ],
            'Conditional World Model': [
                'Enables true counterfactual simulation',
                'Can explore beyond expert demonstrations',
                'Supports model-based planning and control',
                'Explicit action conditioning for controllability'
            ],
            'Direct Video RL': [
                'No model bias or simulation errors',
                'Direct optimization on real data',
                'Simple implementation and debugging',
                'Good baseline for model-based comparisons'
            ]
        }
    
    def _create_evaluation_summary(self, results: Dict) -> Dict:
        """Create evaluation summary for paper generation."""
        
        summary = {
            'experiment_type': 'three_method_architectural_comparison',
            'methods_tested': ['autoregressive_il', 'conditional_world_model', 'direct_video_rl'],
            'key_findings': [],
            'performance_ranking': [],
            'architectural_insights': self._analyze_architectural_benefits()
        }
        
        # Add performance ranking based on results
        method_performances = []
        
        # Method 1 performance
        method1 = results.get('method_1_autoregressive_il', {})
        if method1.get('status') == 'success':
            eval_results = method1.get('evaluation', {}).get('overall_metrics', {})
            method_performances.append(('Autoregressive IL', eval_results.get('action_mAP', 0)))
        
        # Method 2 performance (use world model state loss as proxy)
        method2 = results.get('method_2_conditional_world_model', {})
        if method2.get('status') == 'success':
            wm_eval = method2.get('world_model_evaluation', {}).get('overall_metrics', {})
            state_loss = wm_eval.get('state_loss', float('inf'))
            # Convert loss to performance score (lower is better)
            performance = 1.0 / (1.0 + state_loss) if state_loss < float('inf') else 0
            method_performances.append(('Conditional World Model', performance))
        
        # Method 3 performance
        method3 = results.get('method_3_direct_video_rl', {})
        if method3.get('status') == 'success':
            rl_models = method3.get('rl_models', {})
            avg_reward = np.mean([res.get('mean_reward', 0) for res in rl_models.values() 
                                if res.get('status') == 'success'])
            method_performances.append(('Direct Video RL', avg_reward))
        
        # Sort by performance
        method_performances.sort(key=lambda x: x[1], reverse=True)
        summary['performance_ranking'] = method_performances
        
        return summary
    
    def _generate_paper_results(self):
        """Generate research paper with results."""
        
        try:
            summary = self._create_evaluation_summary(self.results)
            
            paper_path = generate_research_paper(
                results=self.results,
                summary=summary,
                output_dir=str(self.results_dir)
            )
            
            self.logger.info(f"📄 Research paper generated: {paper_path}")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Paper generation failed: {e}")
    
    def _save_complete_results(self):
        """Save all experimental results."""
        
        # Convert results to JSON-serializable format
        json_results = self._convert_for_json(self.results)
        
        # Save detailed results
        import json
        results_path = self.results_dir / 'complete_results.json'
        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
        
        # Save summary
        summary = self._create_evaluation_summary(self.results)
        summary_path = self.results_dir / 'experiment_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Generate paper
        self._generate_paper_results()
        
        self.logger.info(f"💾 All results saved to: {self.results_dir}")
        self.logger.info(f"📄 Complete results: {results_path}")
        self.logger.info(f"📄 Summary: {summary_path}")
    
    def _convert_for_json(self, obj: Any) -> Any:
        """Convert objects to JSON-serializable format."""
        
        if isinstance(obj, dict):
            return {k: self._convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(item) for item in obj]
        elif isinstance(obj, (torch.Tensor, np.ndarray)):
            return obj.tolist()
        elif isinstance(obj, Path):
            return str(obj)
        elif hasattr(obj, '__dict__'):
            return str(obj)
        else:
            return obj


def main():
    """Main function to run the separate models surgical RL comparison."""
    
    print("🏗️ SEPARATE MODELS SURGICAL RL COMPARISON")
    print("=" * 60)
    print("Research Paper: Optimal Architectures for IL vs RL in Surgery")
    print()
    print("🎓 Method 1: AutoregressiveILModel")
    print("   → Pure causal frame generation → action prediction")
    print("   → No action conditioning during training")
    print()
    print("🌍 Method 2: ConditionalWorldModel + RL")
    print("   → Action-conditioned forward simulation")
    print("   → RL training in simulated environment")
    print()
    print("📹 Method 3: Model-free RL on Video")
    print("   → Direct interaction with real video data")
    print("   → No model, no simulation")
    print()
    
    # Choose config file
    config_path = 'config_local_debug.yaml'
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        print("Please ensure config file exists or update the path")
        return
    else:
        print(f"📄 Using config: {config_path}")
    
    try:
        # Run comparison
        comparison = SeparateModelsSurgicalComparison(config_path)
        results = comparison.run_complete_comparison()
        
        print("\n🎉 EXPERIMENT COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        print(f"📁 Results saved to: {comparison.results_dir}")
        print(f"📊 Methods compared: 3")
        print(f"🎯 Focus: Architectural differences in surgical RL")
        
    except Exception as e:
        print(f"\n❌ EXPERIMENT FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
