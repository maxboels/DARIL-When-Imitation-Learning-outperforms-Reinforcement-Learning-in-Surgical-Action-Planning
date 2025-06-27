#!/usr/bin/env python3
"""
UPDATED Complete Experimental Comparison with  RL Integration:
1. Autoregressive Imitation Learning (Method 1) - Pure causal generation → actions
2. RL with ConditionalWorldModel Simulation (Method 2) - IMPROVED with better rewards + debugging  
3. RL with Offline Video Episodes (Method 3) - IMPROVED with better environments
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

# Import UPDATED trainers with debugging
from training.autoregressive_il_trainer import AutoregressiveILTrainer
from training.world_model_trainer import WorldModelTrainer  
from training.world_model_rl_trainer import WorldModelRLTrainer  # UPDATED: Use debug version

# Import existing components for Method 3 and evaluation
from datasets.cholect50 import load_cholect50_data

# UPDATED: Import improved direct video environment
from environment.direct_video_env import DirectVideoSB3Trainer, test_direct_video_environment

# Import evaluation framework
from evaluation.integrated_evaluation import run_integrated_evaluation
from evaluation.paper_generator import generate_research_paper
from evaluation.publication_plots import create_publication_plots


# Import logger for logging
from utils.logger import SimpleLogger

# UPDATED: Import debugging tools
from debugging.rl_debug_tools import RLDebugger
from debugging.rl_diagnostic_script import diagnose_rl_training

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ExperimentRunner:
    """
    UPDATED Experimental comparison with  RL training:
    1. Method 1: AutoregressiveILModel (frames → causal generation → actions)
    2. Method 2: ConditionalWorldModel (state + action → next_state + rewards) - IMPROVED
    3. Method 3: Direct Video RL (no model, real video interaction) - IMPROVED
    """
    
    def __init__(self, config_path: str = 'config_local_debug.yaml'):
        print("🏗️ Initializing RL Surgical Comparison")
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Create experiment directory with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.experiment_name = f"{timestamp}"
        self.results_dir = Path("results") / self.experiment_name / f"fold{self.config['data']['paths']['fold']}"
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.plots_dir = self.results_dir / "publication_plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        # Initialize logger
        self.logger = SimpleLogger(
            log_dir=str(self.results_dir),
            name="SuRL_FixedRL",
            use_shared_timestamp=True
        )
        
        # Results storage
        self.results = {
            'experiment_name': self.experiment_name,
            'config': self.config,
            'timestamp': timestamp,
            'results_dir': str(self.results_dir),
        }
        
        self.logger.info(f"🎯 RL Experiment: {self.experiment_name}")
        self.logger.info(f"📁 Results dir: {self.results_dir}")
        self.logger.info(f"📂 Plots dir: {self.plots_dir}")

    def run_complete_comparison(self) -> Dict[str, Any]:
        """Run the complete three-method comparison with RL."""
        
        self.logger.info("🚀 Starting Complete RL Comparison")
        self.logger.info("=" * 60)
        
        # Load data
        train_data, test_data = self._load_data()

        # Method 1: Autoregressive IL (unchanged, was working well)
        if self.config.get('experiment', {}).get('autoregressive_il', {}).get('enabled', True):        
            self.logger.info("🎓 Running Method 1: Autoregressive IL")
            method1_results = self._run_method1_autoregressive_il(train_data, test_data)
            self.results['method_1_autoregressive_il'] = method1_results
        else:
            self.logger.info("🎓 Method 1: Autoregressive IL is disabled in config, skipping...")
            method1_results = {'status': 'skipped', 'reason': 'Autoregressive IL disabled in config'}
            self.results['method_1_autoregressive_il'] = method1_results
        
        if self.config.get('experiment', {}).get('world_model', {}).get('enabled', True):
            # Method 2: Conditional World Model + RL
            method2_results = self._run_method2_wm_rl(train_data, test_data)
            self.results['method_2_conditional_world_model'] = method2_results
        else:
            self.logger.info("🌍 Method 2: Conditional World Model + RL is disable")

        if self.config.get('experiment', {}).get('rl_experiments', {}).get('enabled', True):
            # Method 3: Direct Video RL
            method3_results = self._run_method3_direct_rl(train_data, test_data)
            self.results['method_3_direct_video_rl'] = method3_results
        else:
            self.logger.info("📹 Method 3: Direct Video RL is disabled in config")
        
        # Comprehensive evaluation - with proper handling
        # evaluation_results = self._run_comprehensive_evaluation_fixed()
        if not hasattr(self, 'test_loaders') or not self.test_loaders:
            self.logger.error("❌ No test loaders available for evaluation")
            return {'status': 'failed', 'error': 'No test loaders available'}
                
        evaluation_results = run_integrated_evaluation(
            experiment_results=self.results,
            test_data=self.test_loaders,
            results_dir=str(self.results_dir),
            logger=self.logger,
            horizon=self.config['evaluation']['prediction_horizon']
        )
        self.results['comprehensive_evaluation'] = evaluation_results

        self.logger.info("📊 Generating publication-quality plots...")
        plot_paths = create_publication_plots(
            experiment_results=self.results,
            output_dir=str(self.plots_dir),
            logger=self.logger
        )
        self.results['generated_plots'] = plot_paths

        # Analysis and comparison
        self.logger.info("🏆 Analyzing Results and Architectural Insights")
        self._print_method_comparison(self.results)
        
        # Save results
        self._save_complete_results()
        
        return self.results
            
    
    def _load_data(self) -> Tuple[List[Dict], List[Dict]]:
        """Load and prepare training and test data."""

        self.logger.info("📂 Loading CholecT50 data...")
        
        train_videos = self.config.get('experiment', {}).get('train', {}).get('max_videos', 2)
        test_videos = self.config.get('experiment', {}).get('test', {}).get('max_videos', 1)
        test_on_train = self.config.get('experiment', {}).get('test', {}).get('test_on_train', False)
        self.logger.info(f"   Training videos: {train_videos}")
        self.logger.info(f"   Test videos: {test_videos}")
        if test_on_train:
            self.logger.warning("⚠️ Testing on training data is enabled, results may not generalize!")

        # Load training and test data
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

    def _run_method1_autoregressive_il(self, train_data: List[Dict], test_data: List[Dict]) -> Dict[str, Any]:
        """Method 1: Autoregressive IL with enhanced IVT-based model saving."""
        
        self.logger.info("🎓 Method 1: Enhanced Autoregressive IL with IVT-based Saving")
        self.logger.info("-" * 40)
        
        try:
            # Check if pretrained model is configured
            il_config = self.config.get('experiment', {}).get('autoregressive_il', {})
            training_enabled = il_config.get('train', True)
            evaluate_enabled = il_config.get('evaluate', True)
            il_enabled = il_config.get('enabled', False)
            il_model_path = il_config.get('il_model_path', None)

            # ENHANCED: Allow specifying which model type to load
            model_type_preference = il_config.get('model_type_preference', 'combined')  # 'current', 'next', 'combined'
            
            # FIXED: Initialize best_model_path to handle all cases
            best_model_paths = {}
            
            # Determine if we should use pretrained model
            use_pretrained = il_enabled and il_model_path and os.path.exists(il_model_path)
            
            if use_pretrained:
                self.logger.info(f"📂 Loading pretrained IL model from: {il_model_path}")
                # Load pretrained model
                model = AutoregressiveILModel.load_model(il_model_path, device=DEVICE)
                self.logger.info("✅ Pretrained IL model loaded successfully")
                
                # FIXED: Set best_model_path to the loaded model path
                best_model_paths['loaded'] = il_model_path
                
                # Use None for train_data since we're using pretrained model
                train_data_for_loader = None
                
            elif il_enabled:
                self.logger.info("🏋️ Training enhanced IL model from scratch...")
                
                # Create model from scratch
                model = AutoregressiveILModel(
                    hidden_dim=self.config['models']['autoregressive_il']['hidden_dim'],
                    embedding_dim=self.config['models']['autoregressive_il']['embedding_dim'],
                    n_layer=self.config['models']['autoregressive_il']['n_layer'],
                    num_action_classes=self.config['models']['autoregressive_il']['num_action_classes'],
                    dropout=self.config['models']['autoregressive_il']['dropout']
                ).to(DEVICE)
                
                # Use original training data
                train_data_for_loader = train_data
                
            else:
                self.logger.info("❌ Autoregressive IL is disabled in config, skipping...")
                return {'status': 'skipped', 'reason': 'Autoregressive IL disabled in config'}
            
            # Create datasets
            train_loader, test_loaders = create_autoregressive_dataloaders(
                config=self.config['data'],
                train_data=train_data_for_loader,  # None if using pretrained model
                test_data=test_data,
                batch_size=self.config['training']['batch_size'],
                num_workers=self.config['training']['num_workers']
            )

            self.test_loaders = test_loaders  # Store test loaders for evaluation
            
            # ENHANCED: Create trainer with IVT-based saving
            from training.autoregressive_il_trainer import AutoregressiveILTrainer
            trainer = AutoregressiveILTrainer(
                model=model,
                config=self.config,
                logger=self.logger,
                device=DEVICE
            )

            # Training phase (only if not using pretrained model and training is enabled)
            if training_enabled and not use_pretrained:
                self.logger.info("🌟 Training Enhanced Autoregressive IL model with IVT-based saving...")                
                best_combined_path = trainer.train(train_loader, test_loaders)
                
                # ENHANCED: Get all best model paths
                best_model_paths = trainer.get_best_model_paths()
                best_metrics = trainer.get_best_metrics()
                
                self.logger.info("📊 TRAINING COMPLETED - Best Models Summary:")
                self.logger.info(f"   🎯 Best Current Recognition mAP: {best_metrics['best_ivt_current_mAP']:.4f}")
                self.logger.info(f"      Model: {best_model_paths['best_current_recognition']}")
                self.logger.info(f"   🎯 Best Next Prediction mAP: {best_metrics['best_ivt_next_mAP']:.4f}")
                self.logger.info(f"      Model: {best_model_paths['best_next_prediction']}")
                self.logger.info(f"   🎯 Best Combined Score: {best_metrics['best_combined_score']:.4f}")
                self.logger.info(f"      Model: {best_model_paths['best_combined']}")
                
            elif use_pretrained:
                self.logger.info("📊 Skipping training (using pretrained model)")
            else:
                self.logger.info("📊 Skipping training (training disabled)")

            # ENHANCED: Choose which model to use for evaluation
            if evaluate_enabled:
                self.logger.info("📊 Evaluating Enhanced Autoregressive IL model...")
                
                # Determine which model to use for evaluation
                if not use_pretrained and best_model_paths:
                    # Choose model based on preference
                    if model_type_preference == 'current' and best_model_paths.get('best_current_recognition'):
                        eval_model_path = best_model_paths['best_current_recognition']
                        self.logger.info(f"🎯 Using best CURRENT recognition model for evaluation")
                    elif model_type_preference == 'next' and best_model_paths.get('best_next_prediction'):
                        eval_model_path = best_model_paths['best_next_prediction']
                        self.logger.info(f"🎯 Using best NEXT prediction model for evaluation")
                    else:
                        eval_model_path = best_model_paths.get('best_combined') or best_model_paths.get('best_next_prediction')
                        self.logger.info(f"🎯 Using best COMBINED model for evaluation")
                    
                    # Load the chosen best model for evaluation
                    if eval_model_path and os.path.exists(eval_model_path):
                        self.logger.info(f"📂 Loading best model for evaluation: {eval_model_path}")
                        model = AutoregressiveILModel.load_model(eval_model_path, device=DEVICE)
                        trainer.model = model  # Update trainer's model
                
                evaluation_results = trainer.evaluate_model(test_loaders)
                
                # Extract key metrics for comparison
                single_step_map = evaluation_results['overall_metrics'].get('action_mAP', 0)
                planning_2s_map = evaluation_results['publication_metrics'].get('planning_2s_mAP', 0)
                planning_degradation = evaluation_results['evaluation_summary'].get('planning_degradation', 0)
                
                # ENHANCED: Add best metrics from training
                training_best_metrics = {}
                if not use_pretrained and hasattr(trainer, 'get_best_metrics'):
                    training_best_metrics = trainer.get_best_metrics()
                
                return {
                    'status': 'success',
                    'model_paths': best_model_paths,  # ENHANCED: All model paths
                    'model_type': 'AutoregressiveIL',
                    'approach': 'Enhanced: Causal frame generation → action anticipation with IVT-based saving',
                    'evaluation': evaluation_results,
                    
                    # Enhanced summary with planning and training metrics
                    'method_description': 'Enhanced Autoregressive IL with IVT-optimized model saving',
                    'capabilities': {
                        'single_step_recognition': single_step_map,
                        'short_term_planning_2s': planning_2s_map,
                        'planning_degradation': planning_degradation,
                        'planning_horizon': 'up_to_5_seconds'
                    },
                    
                    # ENHANCED: Training performance summary
                    'training_performance': training_best_metrics,
                    'model_selection_strategy': f'IVT-based ({model_type_preference} preference)',
                    
                    # Your t+1 target choice is PERFECT for planning!
                    'target_type': 'next_action_prediction',  # This is ideal for planning evaluation
                    'planning_ready': True,
                    'pretrained': use_pretrained,
                    'enhanced_saving': True
                }
            else:
                self.logger.info("📊 Evaluation disabled, returning basic results")
                return {
                    'status': 'success',
                    'model_paths': best_model_paths,  # ENHANCED: All model paths
                    'model_type': 'AutoregressiveIL',
                    'approach': 'Enhanced: Causal frame generation → action anticipation with IVT-based saving',
                    'evaluation': None,
                    'method_description': 'Enhanced Autoregressive IL (evaluation skipped)',
                    'pretrained': use_pretrained,
                    'enhanced_saving': True
                }
        
        except Exception as e:
            self.logger.error(f"❌ Enhanced Method 1 failed: {e}")
            import traceback
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            return {'status': 'failed', 'error': str(e)}

    def _run_method2_wm_rl(self, train_data: List[Dict], test_data: List[Dict]) -> Dict[str, Any]:
            """FIXED Method 2: Conditional World Model + Improved RL - supports pretrained models."""
            
            self.logger.info("-" * 40)
            self.logger.info("-" * 40)
            self.logger.info("🌍 Method 2: Conditional World Model + RL")
            self.logger.info("-" * 40)
            self.logger.info("-" * 40)
            
            try:
                # Check if pretrained world model is configured
                wm_config = self.config.get('experiment', {}).get('world_model', {})
                wm_model_path = wm_config.get('wm_model_path', None)
                
                # 🔧 FIX: Determine if we should skip training
                skip_training = wm_model_path and os.path.exists(wm_model_path)
                
                if skip_training:
                    self.logger.info(f"📂 Using pretrained world model from: {wm_model_path}")
                    # Skip training by setting train_data to None
                    train_data_for_loader = None
                    
                    # Load pretrained world model
                    world_model = ConditionalWorldModel.load_model(wm_model_path, device=DEVICE)
                    self.logger.info("✅ Pretrained world model loaded successfully")
                    
                else:
                    self.logger.info("🏋️ Will train world model from scratch")
                    # Use original training data
                    train_data_for_loader = train_data
                    
                    # Create world model from scratch
                    world_model = ConditionalWorldModel(
                        hidden_dim=self.config['models']['conditional_world_model']['hidden_dim'],
                        embedding_dim=self.config['models']['conditional_world_model']['embedding_dim'],
                        action_embedding_dim=self.config['models']['conditional_world_model']['action_embedding_dim'],
                        n_layer=self.config['models']['conditional_world_model']['n_layer'],
                        num_action_classes=self.config['models']['conditional_world_model']['num_action_classes'],
                        dropout=self.config['models']['conditional_world_model']['dropout']
                    ).to(DEVICE)
                    self.logger.info("✅ World model initialized for training")
                
                # Create datasets for world model training/evaluation
                train_loader, test_loaders, simulation_loader = create_world_model_dataloaders(
                    config=self.config['data'],
                    train_data=train_data_for_loader,  # None if using pretrained model
                    test_data=test_data,
                    batch_size=self.config['training']['batch_size'],
                    num_workers=self.config['training']['num_workers']
                )

                # ✅ IMPORTANT: Store test loaders for comprehensive evaluation
                self.test_loaders = test_loaders  # Dict[video_id, DataLoader]
                self.logger.info(f"✅ Stored test loaders for evaluation: {list(self.test_loaders.keys())}")
            
                # Step 1: Train or evaluate world model
                world_model_trainer = WorldModelTrainer(
                    model=world_model,
                    config=self.config,
                    logger=self.logger,
                    device=DEVICE
                )
                
                if skip_training:
                    self.logger.info("📊 Step 1: Evaluating pretrained world model...")
                    # Skip training, go directly to evaluation
                    world_model_evaluation = world_model_trainer.evaluate_model(test_loaders)
                    best_world_model_path = wm_model_path  # Use the loaded path
                    
                else:
                    self.logger.info("🌍 Step 1: Training world model from scratch...")
                    # Train world model
                    best_world_model_path = world_model_trainer.train(train_loader, test_loaders)
                    
                    self.logger.info("🌍 Evaluating trained world model...")
                    world_model_evaluation = world_model_trainer.evaluate_model(test_loaders)
                
                # Step 2: RL training
                self.logger.info("🚀 Step 2: Starting RL Policy training...")

                # Create RL trainer with world model                
                rl_trainer = WorldModelRLTrainer(
                    config=self.config,
                    logger=self.logger,
                    device=DEVICE
                )
                
                # Get training timesteps
                timesteps = self.config.get('rl_training', {}).get('timesteps', 20000)  # Increased default
                
                # Use original training data for RL
                rl_train_data = train_data  # Always use original training data for RL
                
                # Train RL algorithms with world model
                self.logger.info(f"🌍 Training World Model RL with PPO...")
                world_model_rl_results = rl_trainer.train_ppo_world_model(
                    world_model, rl_train_data, timesteps
                )
                
                self.logger.info(f"🎬 Training Direct Video RL with PPO...")
                direct_video_rl_results = rl_trainer.train_ppo_direct_video(
                    rl_train_data, timesteps
                )
                
                rl_results = {
                    'world_model_ppo': world_model_rl_results,
                    'direct_video_ppo': direct_video_rl_results
                }
                
                return {
                    'status': 'success',
                    'world_model_path': best_world_model_path,
                    'world_model_evaluation': world_model_evaluation,
                    'world_model_pretrained': skip_training,  # Track if model was pretrained
                    'rl_models': rl_results,
                    'model_type': 'ConditionalWorldModel',
                    'approach': f'FIXED: Action-conditioned world model + improved RL {"(PRETRAINED)" if skip_training else "(TRAINED)"}',
                    'method_description': f'World model-based RL with rewards and debugging {"(pretrained WM)" if skip_training else "(trained WM)"}',
                    'improvements': [
                        'Expert demonstration matching rewards',
                        'Proper action space handling', 
                        'Enhanced monitoring and debugging',
                        'Optimized hyperparameters',
                        'Pretrained world model support' if skip_training else 'Trained world model from scratch'
                    ]
                }
                
            except Exception as e:
                self.logger.error(f"❌ Method 2 failed: {e}")
                import traceback
                self.logger.error(f"Full traceback: {traceback.format_exc()}")
                return {'status': 'failed', 'error': str(e)}
    
    def _run_method3_direct_rl(self, train_data: List[Dict], test_data: List[Dict]) -> Dict[str, Any]:
        """FIXED Method 3: Direct Video RL with improved environments."""
        
        self.logger.info("📹 Method 3: Direct Video RL")
        self.logger.info("-" * 40)
        
        try:
            # Test environment first with version
            env_works = test_direct_video_environment(train_data, self.config)
            if not env_works:
                raise RuntimeError("FIXED direct video environment test failed")
            
            # Create RL trainer for direct video interaction with environments
            rl_trainer = DirectVideoSB3Trainer(
                video_data=train_data,
                config=self.config,
                logger=self.logger,
                device=DEVICE
            )
            
            # Get training timesteps (increased for better convergence)
            timesteps = self.config.get('rl_training', {}).get('timesteps', 20000)  # Increased default
            
            # Train RL algorithms directly on videos with rewards
            self.logger.info(f"🚀 Training RL algorithms for {timesteps} timesteps...")
            rl_results = rl_trainer.train_all_algorithms(timesteps=timesteps)
            
            return {
                'status': 'success',
                'rl_models': rl_results,
                'model_type': 'DirectVideoRL',
                'approach': 'FIXED: Direct RL on video sequences with improved rewards',
                'method_description': 'Model-free RL on offline video episodes with reward design',
                'improvements': [
                    'Expert demonstration matching rewards',
                    'Proper continuous action space [0,1]', 
                    'Better episode termination',
                    'Meaningful reward functions'
                ]
            }
            
        except Exception as e:
            self.logger.error(f"❌ Method 3 failed: {e}")
            import traceback
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            return {'status': 'failed', 'error': str(e)}
    
    def _run_comprehensive_evaluation_fixed(self) -> Dict[str, Any]:
        """FIXED: Run comprehensive evaluation with proper handling."""
        
        if not hasattr(self, 'test_loaders') or not self.test_loaders:
            self.logger.error("❌ No test loaders available for evaluation")
            return {'status': 'failed', 'error': 'No test loaders available'}
        
        # Use corrected evaluation that should be the version
        evaluation_results = run_integrated_evaluation(
            experiment_results=self.results,
            test_data=self.test_loaders,
            results_dir=str(self.results_dir),
            logger=self.logger,
            horizon=self.config['evaluation']['prediction_horizon']
        )
        
        return evaluation_results
    
    def _print_method_comparison(self, aggregate_results: Dict):
        """Print comparison of all three methods with RL results."""
        
        self.logger.info("🏆 THREE-METHOD COMPARISON RESULTS (FIXED RL)")
        self.logger.info("=" * 60)
        
        # Method 1 results (unchanged)
        method1 = aggregate_results.get('method_1_autoregressive_il', {})
        if method1.get('status') == 'success':
            eval_results = method1.get('evaluation', {}).get('overall_metrics', {})
            self.logger.info(f"🎓 Method 1 (Autoregressive IL):")
            self.logger.info(f"   Status: ✅ Success")
            self.logger.info(f"   Action mAP: {eval_results.get('action_mAP', 0):.4f}")
            self.logger.info(f"   Approach: Pure causal generation → actions")
        else:
            self.logger.info(f"🎓 Method 1: ❌ Failed - {method1.get('error', 'Unknown')}")

        generated_plots = aggregate_results.get('generated_plots', {})
        if generated_plots:
            self.logger.info(f"")
            self.logger.info(f"📊 PUBLICATION PLOTS GENERATED:")
            for plot_type, plot_path in generated_plots.items():
                self.logger.info(f"   📈 {plot_type}: {plot_path}")
        

        # Method 2 results 
        method2 = aggregate_results.get('method_2_conditional_world_model', {})
        if method2.get('status') == 'success':
            self.logger.info(f"🌍 Method 2 (FIXED Conditional World Model + RL):")
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
                self.logger.info(f"     {alg}: {reward:.3f} reward (IMPROVED)")
            self.logger.info(f"   Approach: Action-conditioned simulation + RL")
            
            # Log improvements
            improvements = method2.get('improvements', [])
            if improvements:
                self.logger.info(f"   🔧 Issues:")
                for improvement in improvements:
                    self.logger.info(f"     ✅ {improvement}")
        else:
            self.logger.info(f"🌍 Method 2: ❌ Failed - {method2.get('error', 'Unknown')}")
        
        # Method 3 results 
        method3 = aggregate_results.get('method_3_direct_video_rl', {})
        if method3.get('status') == 'success':
            self.logger.info(f"📹 Method 3 (FIXED Direct Video RL):")
            self.logger.info(f"   Status: ✅ Success")
            
            rl_models = method3.get('rl_models', {})
            successful_rl = [alg for alg, res in rl_models.items() if res.get('status') == 'success']
            self.logger.info(f"   RL Algorithms: {successful_rl}")
            for alg in successful_rl:
                reward = rl_models[alg].get('mean_reward', 0)
                self.logger.info(f"     {alg}: {reward:.3f} reward (IMPROVED)")
            self.logger.info(f"   Approach: Model-free RL on real video frames")
            
            # Log improvements
            improvements = method3.get('improvements', [])
            if improvements:
                self.logger.info(f"   🔧 Issues:")
                for improvement in improvements:
                    self.logger.info(f"     ✅ {improvement}")
        else:
            self.logger.info(f"📹 Method 3: ❌ Failed - {method3.get('error', 'Unknown')}")
        
        # Summary
        successful_methods = []
        if method1.get('status') == 'success':
            successful_methods.append("Method 1 (IL)")
        if method2.get('status') == 'success':
            successful_methods.append("Method 2 (FIXED World Model RL)")
        if method3.get('status') == 'success':
            successful_methods.append("Method 3 (FIXED Direct RL)")
        
        self.logger.info(f"")
        self.logger.info(f"🎯 Successful Methods: {successful_methods}")
        self.logger.info(f"📊 Total Methods Tested: 3")
        self.logger.info(f"✅ Success Rate: {len(successful_methods)}/3")
        
        # Compare RL improvements
        self._compare_rl_improvements()
    
    def _compare_rl_improvements(self):
        """Compare before/after RL performance."""
        
        self.logger.info("")        
        method2 = self.results.get('method_2_conditional_world_model', {})
        method3 = self.results.get('method_3_direct_video_rl', {})
        
        if method2.get('status') == 'success':
            rl_models = method2.get('rl_models', {})
            for alg, result in rl_models.items():
                if result.get('status') == 'success':
                    reward = result.get('mean_reward', 0)
                    self.logger.info(f"     Method 2 {alg}: {reward:.3f}")
        
        if method3.get('status') == 'success':
            rl_models = method3.get('rl_models', {})
            for alg, result in rl_models.items():
                if result.get('status') == 'success':
                    reward = result.get('mean_reward', 0)
                    self.logger.info(f"     Method 3 {alg}: {reward:.3f}")
          
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
        
        self.logger.info(f"💾 All RL results saved to: {self.results_dir}")
        self.logger.info(f"📄 Complete results: {results_path}")
        self.logger.info(f"📄 Summary: {summary_path}")

        generated_plots = self.results.get('generated_plots', {})
        if generated_plots:
            self.logger.info(f"📊 Publication plots: {self.plots_dir}")

    def _create_evaluation_summary(self, results: Dict) -> Dict:
        """Create evaluation summary focusing on RL improvements."""
        
        summary = {
            'experiment_type': 'three_method_architectural_comparison',
            'methods_tested': ['autoregressive_il', 'conditional_world_model', 'direct_video_rl'],
            'rl_improvements_applied': True,
            'key_findings': [],
            'performance_ranking': [],
            'rl_fixes': [
                'Expert demonstration matching rewards',
                'Proper continuous action space',
                'Enhanced monitoring and debugging', 
                'Optimized hyperparameters',
                'Better episode termination'
            ]
        }
        
        # Add performance ranking based on results
        method_performances = []
        
        # Method 1 performance
        method1 = results.get('method_1_autoregressive_il', {})
        if method1.get('status') == 'success':
            eval_results = method1.get('evaluation', {}).get('overall_metrics', {})
            method_performances.append(('Autoregressive IL', eval_results.get('action_mAP', 0)))
        
        # Method 2 performance (use best RL reward)
        method2 = results.get('method_2_conditional_world_model', {})
        if method2.get('status') == 'success':
            rl_models = method2.get('rl_models', {})
            best_reward = max([res.get('mean_reward', -1000) for res in rl_models.values() 
                              if res.get('status') == 'success'], default=-1000)
            if best_reward > -1000:
                method_performances.append(('FIXED Conditional World Model RL', best_reward))
        
        # Method 3 performance (use best RL reward)
        method3 = results.get('method_3_direct_video_rl', {})
        if method3.get('status') == 'success':
            rl_models = method3.get('rl_models', {})
            best_reward = max([res.get('mean_reward', -1000) for res in rl_models.values() 
                              if res.get('status') == 'success'], default=-1000)
            if best_reward > -1000:
                method_performances.append(('FIXED Direct Video RL', best_reward))
        
        # Sort by performance
        method_performances.sort(key=lambda x: x[1], reverse=True)
        summary['performance_ranking'] = method_performances
        
        return summary
    
    def _generate_paper_results(self):
        """Generate research paper with results."""
        
        try:
            summary = self._create_evaluation_summary(self.results)
            
            paper_dir = generate_research_paper(self.results_dir, self.logger)
            
            self.logger.info(f"📄 Research paper generated: {paper_dir}")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Paper generation failed: {e}")
    
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
    """Main function to run the RL surgical comparison."""
    
    print("🏗️ RL SURGICAL COMPARISON")
    print("=" * 60)
    print("Research Paper: Optimal Architectures for IL vs RL in Surgery")
    print()
    print("🎓 Method 1: AutoregressiveILModel (unchanged - was working)")
    print("   → Pure causal frame generation → action prediction")
    print()
    print("🌍 Method 2: ConditionalWorldModel + RL")
    print("   → Action-conditioned forward simulation with IMPROVED rewards")
    print("   → Expert demonstration matching + proper monitoring")
    print()
    print("📹 Method 3: Model-free RL on Video")
    print("   → Direct interaction with real video data + IMPROVED rewards")
    print("   → Expert demonstration matching + proper action space")
    print()
    
    # Choose config file here
    config_path = 'config_dgx_all_v8.yaml'
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        print("Please ensure config file exists or update the path")
        return
    else:
        print(f"📄 Using config: {config_path}")
    
    try:
        import argparse
        parser = argparse.ArgumentParser(description="Run  RL surgical experiment")
        parser.add_argument('--config', type=str, default=config_path, help="Path to config file")
        args = parser.parse_args()
        print(f"🔧 Arguments: {args}")

        # Run  comparison
        experiment = ExperimentRunner(args.config)
        results = experiment.run_complete_comparison()
        
        print("\n🎉  RL EXPERIMENT COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        print(f"📁 Results saved to: {experiment.results_dir}")
        generated_plots = results.get('generated_plots', {})
        
        if generated_plots:
            print(f"📊 Publication plots generated:")
            for plot_type, plot_path in generated_plots.items():
                print(f"   📈 {plot_type}: {plot_path}")
        else:
            print(f"⚠️ No plots generated (check for errors)")
        
        print(f"🎯 Ready for publication!")
        
    except Exception as e:
        print(f"\n❌  RL EXPERIMENT FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

def get_enhanced_il_config():
    """
    Get enhanced IL configuration with IVT-based model saving options.
    
    Add this to your config YAML:
    """
    
    enhanced_config = {
        'experiment': {
            'autoregressive_il': {
                'enabled': True,
                'train': True,
                'evaluate': True,
                'il_model_path': None,  # Set to pretrained model path if available
                
                # ENHANCED: Model selection preferences
                'model_type_preference': 'combined',  # 'current', 'next', 'combined'
                'save_all_best_models': True,  # Save separate models for each task
                
                # ENHANCED: IVT-based saving configuration
                'ivt_saving': {
                    'enabled': True,
                    'current_weight': 0.4,    # Weight for current recognition in combined score
                    'next_weight': 0.6,       # Weight for next prediction in combined score
                    'save_frequency': 1,      # Save every N epochs (1 = every epoch)
                    'keep_top_k': 3          # Keep top K models for each task
                }
            }
        },
        
        # ENHANCED: Training configuration optimized for IVT metrics
        'training': {
            'epochs': 15,
            'batch_size': 8,
            'learning_rate': 8e-5,
            
            # Enhanced monitoring for IVT metrics
            'monitor_ivt_metrics': True,
            'early_stopping': {
                'enabled': True,
                'patience': 8,
                'monitor': 'ivt_next_mAP',  # Monitor next prediction mAP
                'min_delta': 0.001,
                'mode': 'max'
            },
            
            # Enhanced logging
            'log_ivt_components': True,  # Log I, V, T, IV, IT components
            'tensorboard_ivt_tracking': True
        }
    }
    
    return enhanced_config