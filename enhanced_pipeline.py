#!/usr/bin/env python3
"""
Enhanced Experiment Pipeline with Unified Evaluation
Integrates the enhanced evaluation framework into the existing pipeline
"""

import sys
import os
from pathlib import Path

# Add the enhanced evaluation to your existing pipeline
def add_enhanced_evaluation_to_pipeline():
    """
    Integration function to add enhanced evaluation to run_experiment_v2.py
    """
    
    # The enhanced evaluation code that should be added to your SurgicalRLComparison class
    enhanced_evaluation_code = '''
    def _run_enhanced_evaluation(self, test_data: List[Dict]) -> Dict[str, Any]:
        """
        Run enhanced unified evaluation with mAP metrics for all methods
        
        This replaces the basic evaluation with comprehensive action prediction analysis
        """
        
        self.logger.info("🔬 Running Enhanced Unified Evaluation Framework")
        self.logger.info("📊 Evaluating all methods on action prediction with mAP metrics")
        
        try:
            # Import the enhanced evaluation framework
            from enhanced_evaluation_framework import UnifiedEvaluationFramework
            
            # Initialize evaluation framework
            evaluator = UnifiedEvaluationFramework(self.logger.log_dir, self.logger)
            
            # Load all trained models
            models = evaluator.load_all_models(self.results)
            
            if not models:
                self.logger.warning("⚠️ No models available for enhanced evaluation")
                return {'error': 'No models available'}
            
            # Run unified evaluation with prediction horizon
            horizon = self.config.get('evaluation', {}).get('horizon', 15)
            enhanced_results = evaluator.run_unified_evaluation(models, test_data, horizon)
            
            # Save results to files (CSV, JSON)
            file_paths = evaluator.save_results_to_files()
            
            # Create comprehensive visualizations
            evaluator.create_comprehensive_visualizations()
            
            # Generate LaTeX tables
            latex_tables = evaluator.generate_latex_tables()
            
            # Print summary
            evaluator.print_summary()
            
            # Return comprehensive results
            return {
                'enhanced_evaluation': enhanced_results,
                'file_paths': file_paths,
                'latex_tables': latex_tables,
                'evaluator': evaluator,
                'status': 'success'
            }
            
        except Exception as e:
            self.logger.error(f"❌ Enhanced evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            return {'error': str(e), 'status': 'failed'}
    '''
    
    # Updated comprehensive evaluation method
    updated_comprehensive_evaluation = '''
    def _run_comprehensive_evaluation(self, test_data: List[Dict]) -> Dict[str, Any]:
        """
        Run both the original dual evaluation AND the enhanced unified evaluation
        """
        
        results = {}
        
        # 1. Run enhanced unified evaluation (NEW - primary evaluation)
        self.logger.info("=" * 60)
        self.logger.info("ENHANCED EVALUATION: Unified mAP Analysis")
        self.logger.info("=" * 60)
        
        enhanced_results = self._run_enhanced_evaluation(test_data)
        results['enhanced_evaluation'] = enhanced_results
        
        # 2. Run original dual evaluation (OPTIONAL - for comparison)
        if self.config.get('evaluation', {}).get('run_dual_evaluation', False):
            self.logger.info("=" * 60)
            self.logger.info("DUAL EVALUATION: Traditional vs Clinical Metrics")
            self.logger.info("=" * 60)
            
            # Load models for evaluation
            models_for_evaluation = {}
            rl_models = {}
            
            # Load IL model
            il_path = self.results['model_paths'].get('method1_il')
            if il_path and os.path.exists(il_path):
                from models.dual_world_model import DualWorldModel
                models_for_evaluation['IL_Baseline'] = DualWorldModel.load_model(il_path, self.device)
                self.logger.info("✅ Loaded IL model for dual evaluation")
            
            # Load RL models for dual evaluation
            method2_results = self.results.get('method_2_rl_world_model', {})
            if 'rl_models' in method2_results:
                for alg_name, rl_result in method2_results['rl_models'].items():
                    if rl_result.get('status') == 'success' and 'model_path' in rl_result:
                        try:
                            if alg_name.lower() == 'ppo':
                                from stable_baselines3 import PPO
                                rl_models[f'RL_WM_{alg_name.upper()}'] = PPO.load(rl_result['model_path'])
                            elif alg_name.lower() == 'a2c':
                                from stable_baselines3 import A2C
                                rl_models[f'RL_WM_{alg_name.upper()}'] = A2C.load(rl_result['model_path'])
                            self.logger.info(f"✅ Loaded {alg_name.upper()} for dual evaluation")
                        except Exception as e:
                            self.logger.warning(f"⚠️ Could not load {alg_name}: {e}")
            
            # World model for dual evaluation
            world_model_path = self.results['model_paths'].get('method1_il')
            world_model = None
            if world_model_path and os.path.exists(world_model_path):
                from models.dual_world_model import DualWorldModel
                world_model = DualWorldModel.load_model(world_model_path, self.device)
            
            # Run dual evaluation if we have models
            if models_for_evaluation or rl_models:
                try:
                    from evaluation.dual_evaluation_framework import DualEvaluationFramework
                    evaluator = DualEvaluationFramework(self.config, self.logger)
                    
                    il_model = models_for_evaluation.get('IL_Baseline') if models_for_evaluation else None
                    
                    dual_results = evaluator.evaluate_comprehensively(
                        il_model, rl_models, test_data, world_model
                    )
                    
                    results['dual_evaluation'] = dual_results
                    
                except Exception as e:
                    self.logger.error(f"❌ Dual evaluation failed: {e}")
                    results['dual_evaluation'] = {'error': str(e)}
            else:
                self.logger.warning("⚠️ No models available for dual evaluation")
                results['dual_evaluation'] = {'error': 'No models available'}
        
        return results
    '''
    
    # Updated paper results generation
    updated_paper_results = '''
    def _generate_paper_results(self):
        """Generate research paper ready results with enhanced evaluation"""
        
        self.logger.info("📝 Generating enhanced research paper results...")
        
        paper_results = {
            'title': 'Comprehensive Evaluation: IL vs RL+WorldModel vs RL+OfflineVideos for Surgical Action Prediction',
            'experiment_summary': {
                'method_1': 'Imitation Learning (Baseline) - Supervised learning on expert demonstrations',
                'method_2': 'RL with World Model Simulation - Uses learned dynamics for exploration',
                'method_3': 'RL with Offline Videos - Direct RL on video sequences',
                'evaluation': 'Unified action prediction evaluation with mAP metrics and trajectory analysis'
            },
            'key_findings': [],
            'method_performance': {},
            'enhanced_evaluation_results': {},
            'research_contributions': []
        }
        
        # Extract enhanced evaluation results
        enhanced_eval = self.results.get('comparative_analysis', {}).get('enhanced_evaluation', {})
        
        if enhanced_eval.get('status') == 'success':
            enhanced_results = enhanced_eval.get('enhanced_evaluation', {})
            
            if 'aggregate_results' in enhanced_results:
                aggregate_stats = enhanced_results['aggregate_results']
                
                # Sort methods by performance
                methods_sorted = sorted(aggregate_stats.items(), 
                                      key=lambda x: x[1]['final_mAP']['mean'], reverse=True)
                
                paper_results['enhanced_evaluation_results'] = {
                    'ranking': [
                        {
                            'rank': i+1,
                            'method': method,
                            'final_mAP': stats['final_mAP']['mean'],
                            'mAP_std': stats['final_mAP']['std'],
                            'degradation': stats['mAP_degradation']['mean'],
                            'stability': stats['trajectory_stability']
                        }
                        for i, (method, stats) in enumerate(methods_sorted)
                    ],
                    'best_method': methods_sorted[0][0] if methods_sorted else 'Unknown',
                    'best_performance': methods_sorted[0][1]['final_mAP']['mean'] if methods_sorted else 0.0
                }
                
                # Statistical significance results
                if 'statistical_tests' in enhanced_results:
                    paper_results['enhanced_evaluation_results']['statistical_tests'] = {
                        'significant_comparisons': [
                            {
                                'comparison': comp,
                                'p_value': results['p_value'],
                                'mean_difference': results['mean_diff'],
                                'effect_size': results['cohens_d']
                            }
                            for comp, results in enhanced_results['statistical_tests'].items()
                            if results['significant']
                        ]
                    }
        
        # Traditional method results (for comparison)
        method1 = self.results.get('method_1_il_baseline', {})
        method2 = self.results.get('method_2_rl_world_model', {})
        method3 = self.results.get('method_3_rl_offline_videos', {})
        
        # Extract traditional performance metrics
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
        
        # Key findings based on enhanced evaluation
        findings = []
        if enhanced_eval.get('status') == 'success':
            findings.append("✅ Enhanced unified evaluation completed with mAP trajectory analysis")
            findings.append("✅ All methods evaluated on identical action prediction metrics")
            findings.append("✅ Statistical significance testing performed between all method pairs")
            findings.append("✅ Comprehensive visualizations and LaTeX tables generated")
            
            # Add specific performance findings
            if 'enhanced_evaluation_results' in paper_results and 'ranking' in paper_results['enhanced_evaluation_results']:
                ranking = paper_results['enhanced_evaluation_results']['ranking']
                if ranking:
                    best = ranking[0]
                    findings.append(f"✅ Best method: {best['method']} with {best['final_mAP']:.3f} mAP")
        else:
            findings.append("⚠️ Enhanced evaluation encountered issues")
        
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
            "Unified evaluation framework using action prediction mAP metrics for fair comparison",
            "Trajectory analysis showing performance degradation over prediction horizons",
            "Statistical significance testing with effect size analysis",
            "Comprehensive visualization suite for surgical AI method comparison",
            "World model effectiveness analysis for surgical action prediction",
            "Open-source implementation for reproducible surgical RL research"
        ]
        
        # Save enhanced paper results
        paper_results_path = self.results_dir / 'enhanced_paper_results.json'
        with open(paper_results_path, 'w') as f:
            json.dump(paper_results, f, indent=2, default=str)
        
        # Generate enhanced paper summary
        self._generate_enhanced_paper_summary(paper_results)
        
        self.logger.info(f"📄 Enhanced paper results saved to: {paper_results_path}")
    '''
    
    # Print integration instructions
    print("🔗 INTEGRATION INSTRUCTIONS")
    print("=" * 50)
    print()
    print("To integrate the enhanced evaluation into your existing pipeline,")
    print("add the following methods to your SurgicalRLComparison class in run_experiment_v2.py:")
    print()
    print("1. Add the enhanced evaluation framework file to your project")
    print("2. Add the _run_enhanced_evaluation method")
    print("3. Replace the _run_comprehensive_evaluation method")
    print("4. Update the _generate_paper_results method")
    print()
    print("The enhanced evaluation will provide:")
    print("✅ Unified mAP metrics for all three methods")
    print("✅ Trajectory analysis over prediction horizons")
    print("✅ Statistical significance testing")
    print("✅ Comprehensive visualizations")
    print("✅ Per-video results in CSV/JSON format")
    print("✅ LaTeX tables ready for publication")
    print()
    
    return {
        'enhanced_evaluation_method': enhanced_evaluation_code,
        'updated_comprehensive_evaluation': updated_comprehensive_evaluation,
        'updated_paper_results': updated_paper_results
    }

# Example usage for updating configuration
def update_config_for_enhanced_evaluation():
    """Add enhanced evaluation configuration options"""
    
    enhanced_config = '''
# Add to your config YAML file:

evaluation:
  # Enhanced evaluation settings
  horizon: 15                    # Prediction horizon for trajectory analysis
  run_dual_evaluation: false     # Whether to also run the original dual evaluation
  max_videos: 10                 # Maximum videos to evaluate (for speed)
  
  # Visualization settings
  create_individual_plots: true  # Create individual focused plots
  save_trajectory_data: true     # Save detailed trajectory data
  
  # Statistical analysis
  significance_level: 0.05       # p-value threshold
  effect_size_threshold: 0.2     # Cohen's d threshold
  
  # Output formats
  save_formats: ['pdf', 'png']   # Figure formats to save
  create_latex_tables: true      # Generate LaTeX tables
  create_csv_outputs: true       # Generate CSV files for analysis
'''
    
    print("📝 CONFIGURATION UPDATE")
    print("=" * 30)
    print(enhanced_config)
    
    return enhanced_config

if __name__ == "__main__":
    print("🚀 ENHANCED EVALUATION INTEGRATION")
    print("=" * 50)
    
    # Show integration instructions
    integration_code = add_enhanced_evaluation_to_pipeline()
    
    # Show configuration updates
    config_updates = update_config_for_enhanced_evaluation()
    
    print("\n🎯 NEXT STEPS:")
    print("1. Save the enhanced_evaluation_framework.py file to your project")
    print("2. Update your run_experiment_v2.py with the new methods")
    print("3. Update your config file with the enhanced evaluation settings")
    print("4. Run your experiment - it will now include comprehensive evaluation!")
    print()
    print("The enhanced evaluation will provide unified mAP metrics for all three methods,")
    print("allowing for fair comparison instead of comparing mAP vs rewards.")
