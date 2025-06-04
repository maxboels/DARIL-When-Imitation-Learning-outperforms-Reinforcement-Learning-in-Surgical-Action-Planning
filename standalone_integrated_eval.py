#!/usr/bin/env python3
"""
Standalone Integrated Evaluation Runner
Run the integrated evaluation with rollout saving on existing results
"""

import json
import yaml
import sys
from pathlib import Path
import torch
from datasets.cholect50 import load_cholect50_data
from utils.logger import SimpleLogger

def run_standalone_integrated_evaluation(
    results_json_path: str = 'logs/2025-06-03_20-36/surgical_rl_results/complete_surgical_rl_results.json',
    config_path: str = 'config_dgx_all.yaml',
    output_dir: str = 'integrated_evaluation_results',
    horizon: int = 15,
    max_videos: int = 10
):
    """
    Run integrated evaluation with rollout saving on existing experimental results
    
    Args:
        results_json_path: Path to your complete experimental results JSON
        config_path: Path to your configuration file
        output_dir: Directory to save integrated evaluation results
        horizon: Prediction horizon for evaluation
        max_videos: Maximum number of test videos to evaluate
    """
    
    print("🚀 STANDALONE INTEGRATED EVALUATION WITH ROLLOUT SAVING")
    print("=" * 70)
    print(f"📁 Results: {results_json_path}")
    print(f"⚙️ Config: {config_path}")
    print(f"📊 Output: {output_dir}")
    print(f"⏱️ Horizon: {horizon}")
    print(f"🎥 Max Videos: {max_videos}")
    print("=" * 70)
    
    try:
        # 1. Load experimental results
        print("📚 Loading experimental results...")
        with open(results_json_path, 'r') as f:
            experiment_results = json.load(f)
        
        print(f"✅ Loaded results from: {results_json_path}")
        
        # 2. Load configuration
        print("⚙️ Loading configuration...")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"✅ Loaded config from: {config_path}")
        
        # 3. Setup logger
        logger = SimpleLogger(log_dir=output_dir, name="IntegratedEval")
        
        # 4. Load test data
        print("📊 Loading test data...")
        test_data = load_cholect50_data(
            config, logger, split='test', max_videos=max_videos
        )
        
        print(f"✅ Loaded {len(test_data)} test videos")
        
        # 5. Import and run integrated evaluation
        print("🔬 Running integrated evaluation with rollout saving...")
        
        # Import the integrated evaluation framework
        from integrated_evaluation_framework import run_integrated_evaluation
        
        # Run integrated evaluation
        integrated_results = run_integrated_evaluation(
            experiment_results=experiment_results,
            test_data=test_data,
            results_dir=output_dir,
            logger=logger,
            horizon=horizon
        )
        
        if integrated_results:
            print("\n🎉 INTEGRATED EVALUATION COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            
            # Print summary of what was generated
            file_paths = integrated_results['file_paths']
            results = integrated_results['results']
            
            print("📊 Generated Files:")
            for file_type, file_path in file_paths.items():
                print(f"  • {file_type}: {file_path}")
            
            print(f"\n📈 Performance Summary:")
            aggregate_results = results['aggregate_results']
            methods_sorted = sorted(aggregate_results.items(), 
                                  key=lambda x: x[1]['final_mAP']['mean'], reverse=True)
            
            for rank, (method, stats) in enumerate(methods_sorted, 1):
                method_display = method.replace('_', ' ')
                final_map = stats['final_mAP']['mean']
                std_map = stats['final_mAP']['std']
                
                print(f"  {rank}. {method_display}: {final_map:.4f} ± {std_map:.4f} mAP")
            
            print(f"\n🔬 Statistical Analysis:")
            significant_tests = [test for test in results['statistical_tests'].values() 
                               if test['significant']]
            print(f"  • {len(significant_tests)} significant differences found (p < 0.05)")
            
            print(f"\n📊 Visualization Data:")
            viz_path = file_paths.get('visualization_json', 'Not found')
            print(f"  • Load {viz_path} in the HTML visualization tool")
            print(f"  • Contains rollout data for {len(test_data)} videos")
            print(f"  • Shows thinking process for all {len(aggregate_results)} methods")
            
            print(f"\n🎯 Next Steps:")
            print(f"  1. Open the HTML visualization tool")
            print(f"  2. Load the visualization_data.json file")
            print(f"  3. Explore AI decision-making at each timestep")
            print(f"  4. Use results for research paper")
            
            return integrated_results
        else:
            print("❌ Integrated evaluation failed!")
            return None
            
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        print("Make sure the results JSON and config files exist.")
        return None
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def analyze_existing_results(results_json_path: str):
    """
    Analyze existing experimental results to see what models are available
    """
    
    print("🔍 ANALYZING EXISTING RESULTS FOR INTEGRATED EVALUATION")
    print("=" * 60)
    
    try:
        with open(results_json_path, 'r') as f:
            results = json.load(f)
        
        print(f"📁 Results file: {results_json_path}")
        print()
        
        models_available = 0
        
        # Analyze Method 1 (IL)
        method1 = results.get('method_1_il_baseline', {})
        print("📊 Method 1 (IL Baseline):")
        if method1.get('status') == 'success' and 'model_path' in method1:
            print(f"  ✅ Status: Success")
            print(f"  📈 mAP: {method1.get('evaluation', {}).get('mAP', 'N/A')}")
            print(f"  💾 Model: {method1.get('model_path', 'Not found')}")
            models_available += 1
        else:
            print(f"  ❌ Status: {method1.get('status', 'Unknown')}")
        print()
        
        # Analyze Method 2 (RL + World Model)
        method2 = results.get('method_2_rl_world_model', {})
        print("📊 Method 2 (RL + World Model):")
        if method2.get('status') == 'success':
            print(f"  ✅ Status: Success")
            rl_models = method2.get('rl_models', {})
            for alg, alg_result in rl_models.items():
                if alg_result.get('status') == 'success' and 'model_path' in alg_result:
                    print(f"  🤖 {alg.upper()}: Reward {alg_result.get('mean_reward', 'N/A'):.3f}")
                    print(f"    💾 Model: {alg_result.get('model_path', 'Not found')}")
                    models_available += 1
        else:
            print(f"  ❌ Status: {method2.get('status', 'Unknown')}")
        print()
        
        # Analyze Method 3 (RL + Offline Videos)
        method3 = results.get('method_3_rl_offline_videos', {})
        print("📊 Method 3 (RL + Offline Videos):")
        if method3.get('status') == 'success':
            print(f"  ✅ Status: Success")
            rl_models = method3.get('rl_models', {})
            for alg, alg_result in rl_models.items():
                if alg_result.get('status') == 'success' and 'model_path' in alg_result:
                    print(f"  🤖 {alg.upper()}: Reward {alg_result.get('mean_reward', 'N/A'):.3f}")
                    print(f"    💾 Model: {alg_result.get('model_path', 'Not found')}")
                    models_available += 1
        else:
            print(f"  ❌ Status: {method3.get('status', 'Unknown')}")
        print()
        
        print("🎯 INTEGRATED EVALUATION READINESS:")
        print(f"  • {models_available} models available for evaluation")
        if models_available >= 2:
            print("  ✅ Ready for integrated evaluation!")
            print("  ✅ Will provide unified mAP metrics")
            print("  ✅ Will save rollout data for visualization")
            print("  ✅ Will perform statistical significance testing")
        else:
            print("  ⚠️ Need at least 2 successful models for meaningful comparison")
        print()
        
        return results, models_available
        
    except Exception as e:
        print(f"❌ Error analyzing results: {e}")
        return None, 0

def create_evaluation_config(
    horizon: int = 15,
    max_videos: int = 10,
    save_detailed_rollouts: bool = True,
    statistical_testing: bool = True
):
    """
    Create configuration for integrated evaluation
    """
    
    config = {
        'integrated_evaluation': {
            'horizon': horizon,
            'max_videos': max_videos,
            'save_detailed_rollouts': save_detailed_rollouts,
            'statistical_testing': statistical_testing,
            
            # Rollout saving options
            'rollout_options': {
                'save_thinking_process': True,
                'save_action_candidates': True,
                'save_planning_horizon': True,
                'save_confidence_scores': True,
                'max_planning_steps': 5
            },
            
            # Visualization options
            'visualization': {
                'create_data_file': True,
                'include_ground_truth': True,
                'include_phase_info': True,
                'max_actions_to_show': 100
            },
            
            # Statistical analysis options
            'statistical_analysis': {
                'significance_level': 0.05,
                'effect_size_calculation': True,
                'pairwise_comparisons': True,
                'correction_method': 'bonferroni'
            }
        }
    }
    
    return config

def main():
    """Main function with command line arguments"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Integrated Evaluation with Rollout Saving')
    parser.add_argument('--results', default='logs/latest/surgical_rl_results/complete_surgical_rl_results.json',
                       help='Path to experimental results JSON')
    parser.add_argument('--config', default='config_dgx_all.yaml',
                       help='Path to configuration file')
    parser.add_argument('--output', default='integrated_evaluation_results',
                       help='Output directory for integrated evaluation')
    parser.add_argument('--horizon', type=int, default=15,
                       help='Prediction horizon for evaluation')
    parser.add_argument('--max-videos', type=int, default=10,
                       help='Maximum number of test videos to evaluate')
    parser.add_argument('--analyze-only', action='store_true',
                       help='Only analyze current results without running evaluation')
    
    args = parser.parse_args()
    
    # Check if results file exists
    if not Path(args.results).exists():
        print(f"❌ Results file not found: {args.results}")
        print("Available options:")
        logs_dir = Path('logs')
        if logs_dir.exists():
            for log_dir in sorted(logs_dir.iterdir(), reverse=True):
                results_file = log_dir / 'surgical_rl_results' / 'complete_surgical_rl_results.json'
                if results_file.exists():
                    print(f"  • {results_file}")
        return 1
    
    if args.analyze_only:
        # Just analyze current results
        results, models_count = analyze_existing_results(args.results)
        
        if models_count >= 2:
            print("\n🚀 To run integrated evaluation:")
            print(f"python {__file__} --results {args.results} --output {args.output}")
        
        return 0
    else:
        # Run full integrated evaluation
        integrated_results = run_standalone_integrated_evaluation(
            results_json_path=args.results,
            config_path=args.config,
            output_dir=args.output,
            horizon=args.horizon,
            max_videos=args.max_videos
        )
        
        if integrated_results:
            print("\n🎉 SUCCESS! Your integrated evaluation is complete.")
            print("🎯 Next step: Open the HTML visualization tool and load the visualization_data.json file")
            return 0
        else:
            print("\n❌ FAILED! Check the error messages above.")
            return 1

if __name__ == "__main__":
    exit(main())
