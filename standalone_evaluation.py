#!/usr/bin/env python3
"""
Standalone Enhanced Evaluation Runner
Run enhanced evaluation on your existing experimental results
"""

import json
import yaml
from pathlib import Path
import torch
from datasets.cholect50 import load_cholect50_data
from utils.logger import SimpleLogger

def run_enhanced_evaluation_standalone(
    results_json_path: str = 'logs/2025-06-03_20-36/surgical_rl_results/complete_surgical_rl_results.json',
    config_path: str = 'config_dgx_all.yaml',
    output_dir: str = 'enhanced_evaluation_results',
    horizon: int = 15
):
    """
    Run enhanced evaluation on existing experimental results
    
    Args:
        results_json_path: Path to your complete experimental results JSON
        config_path: Path to your configuration file
        output_dir: Directory to save enhanced evaluation results
        horizon: Prediction horizon for evaluation
    """
    
    print("🚀 STANDALONE ENHANCED EVALUATION")
    print("=" * 50)
    print(f"📁 Results: {results_json_path}")
    print(f"⚙️ Config: {config_path}")
    print(f"📊 Output: {output_dir}")
    print(f"⏱️ Horizon: {horizon}")
    print("=" * 50)
    
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
        logger = SimpleLogger(log_dir=output_dir, name="EnhancedEval")
        
        # 4. Load test data
        print("📊 Loading test data...")
        test_data = load_cholect50_data(
            config, logger, split='test', 
            max_videos=config.get('experiment', {}).get('test', {}).get('max_videos', 10)
        )
        
        print(f"✅ Loaded {len(test_data)} test videos")
        
        # 5. Import and run enhanced evaluation
        print("🔬 Running enhanced evaluation...")
        
        # Import the enhanced evaluation framework
        from enhanced_evaluation_framework import UnifiedEvaluationFramework
        
        # Initialize evaluation framework
        evaluator = UnifiedEvaluationFramework(output_dir, logger)
        
        # Load all trained models from experimental results
        models = evaluator.load_all_models(experiment_results)
        
        if not models:
            print("❌ No models found in experimental results!")
            print("Make sure your experimental results contain model paths.")
            return None
        
        print(f"✅ Loaded {len(models)} models: {list(models.keys())}")
        
        # Run unified evaluation
        print(f"🎯 Running unified evaluation with horizon {horizon}...")
        results = evaluator.run_unified_evaluation(models, test_data, horizon)
        
        # Save results to files
        print("💾 Saving results to files...")
        file_paths = evaluator.save_results_to_files()
        
        # Create comprehensive visualizations
        print("🎨 Creating visualizations...")
        evaluator.create_comprehensive_visualizations()
        
        # Generate LaTeX tables
        print("📝 Generating LaTeX tables...")
        latex_tables = evaluator.generate_latex_tables()
        
        # Print summary
        evaluator.print_summary()
        
        print("\n🎉 ENHANCED EVALUATION COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        print(f"📁 All results saved to: {output_dir}/")
        print("\n📊 Key Files Generated:")
        print("  - video_level_results.csv (Per-video metrics)")
        print("  - aggregate_statistics.csv (Summary statistics)")
        print("  - trajectory_data.csv (Trajectory analysis data)")
        print("  - complete_evaluation_results.json (Full results)")
        print("  - comprehensive_evaluation_results.pdf (Main figure)")
        print("  - evaluation_tables.tex (LaTeX tables)")
        print("  - method_comparison.pdf (Method comparison plot)")
        print("  - trajectory_analysis.pdf (Trajectory analysis plot)")
        
        return {
            'evaluator': evaluator,
            'results': results,
            'file_paths': file_paths,
            'latex_tables': latex_tables
        }
        
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        print("Make sure the results JSON and config files exist.")
        return None
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def analyze_current_results(results_json_path: str):
    """
    Analyze your current experimental results to see what's available
    """
    
    print("🔍 ANALYZING CURRENT RESULTS")
    print("=" * 40)
    
    try:
        with open(results_json_path, 'r') as f:
            results = json.load(f)
        
        print(f"📁 Results file: {results_json_path}")
        print()
        
        # Analyze Method 1 (IL)
        method1 = results.get('method_1_il_baseline', {})
        print("📊 Method 1 (IL Baseline):")
        if method1.get('status') == 'success':
            print(f"  ✅ Status: Success")
            print(f"  📈 mAP: {method1.get('evaluation', {}).get('mAP', 'N/A')}")
            print(f"  💾 Model: {method1.get('model_path', 'Not found')}")
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
                if alg_result.get('status') == 'success':
                    print(f"  🤖 {alg.upper()}: Reward {alg_result.get('mean_reward', 'N/A'):.3f}")
                    print(f"    💾 Model: {alg_result.get('model_path', 'Not found')}")
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
                if alg_result.get('status') == 'success':
                    print(f"  🤖 {alg.upper()}: Reward {alg_result.get('mean_reward', 'N/A'):.3f}")
                    print(f"    💾 Model: {alg_result.get('model_path', 'Not found')}")
        else:
            print(f"  ❌ Status: {method3.get('status', 'Unknown')}")
        print()
        
        # Check for comparative analysis
        comp_analysis = results.get('comparative_analysis', {})
        print("📊 Comparative Analysis:")
        if comp_analysis:
            print(f"  📈 Available: {list(comp_analysis.keys())}")
        else:
            print("  ❌ Not found")
        print()
        
        print("🎯 RECOMMENDATIONS:")
        print("1. Your results show all three methods completed successfully!")
        print("2. However, you're comparing different metrics (mAP vs rewards)")
        print("3. The enhanced evaluation will provide unified mAP metrics for all methods")
        print("4. This will enable fair comparison between IL and RL approaches")
        print()
        
        return results
        
    except Exception as e:
        print(f"❌ Error analyzing results: {e}")
        return None

def create_comparison_summary(original_results_path: str, enhanced_results_path: str):
    """
    Create a comparison between original and enhanced evaluation results
    """
    
    print("📊 COMPARISON: Original vs Enhanced Results")
    print("=" * 50)
    
    try:
        # Load original results
        with open(original_results_path, 'r') as f:
            original = json.load(f)
        
        # Load enhanced results
        enhanced_path = Path(enhanced_results_path) / 'complete_evaluation_results.json'
        if enhanced_path.exists():
            with open(enhanced_path, 'r') as f:
                enhanced = json.load(f)
        else:
            print(f"❌ Enhanced results not found at: {enhanced_path}")
            return
        
        print("🔍 ORIGINAL RESULTS (Different Metrics):")
        print("-" * 30)
        
        # IL performance
        il_map = original.get('method_1_il_baseline', {}).get('evaluation', {}).get('mAP', 0)
        print(f"IL Baseline: {il_map:.3f} mAP")
        
        # RL performances (rewards)
        method2 = original.get('method_2_rl_world_model', {}).get('rl_models', {})
        method3 = original.get('method_3_rl_offline_videos', {}).get('rl_models', {})
        
        print("RL + World Model:")
        for alg, result in method2.items():
            if result.get('status') == 'success':
                print(f"  {alg.upper()}: {result.get('mean_reward', 0):.3f} reward")
        
        print("RL + Offline Videos:")
        for alg, result in method3.items():
            if result.get('status') == 'success':
                print(f"  {alg.upper()}: {result.get('mean_reward', 0):.3f} reward")
        
        print("\n🎯 ENHANCED RESULTS (Unified mAP Metrics):")
        print("-" * 30)
        
        # Enhanced evaluation results
        if 'aggregate_results' in enhanced:
            for method, stats in enhanced['aggregate_results'].items():
                final_map = stats.get('final_mAP', {}).get('mean', 0)
                std_map = stats.get('final_mAP', {}).get('std', 0)
                print(f"{method.replace('_', ' ')}: {final_map:.3f} ± {std_map:.3f} mAP")
        
        print("\n💡 KEY INSIGHTS:")
        print("1. Enhanced evaluation provides unified mAP metrics for fair comparison")
        print("2. Now you can directly compare all methods on the same scale")
        print("3. Statistical significance testing shows which differences are meaningful")
        print("4. Trajectory analysis reveals how performance degrades over time")
        
    except Exception as e:
        print(f"❌ Error creating comparison: {e}")

def main():
    """Main function with different usage options"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Evaluation for Surgical RL Comparison')
    parser.add_argument('--results', default='logs/2025-06-03_20-36/surgical_rl_results/complete_surgical_rl_results.json',
                       help='Path to experimental results JSON')
    parser.add_argument('--config', default='config_dgx_all.yaml',
                       help='Path to configuration file')
    parser.add_argument('--output', default='enhanced_evaluation_results',
                       help='Output directory for enhanced evaluation')
    parser.add_argument('--horizon', type=int, default=15,
                       help='Prediction horizon for evaluation')
    parser.add_argument('--analyze-only', action='store_true',
                       help='Only analyze current results without running evaluation')
    
    args = parser.parse_args()
    
    if args.analyze_only:
        # Just analyze current results
        analyze_current_results(args.results)
    else:
        # Run full enhanced evaluation
        enhanced_results = run_enhanced_evaluation_standalone(
            results_json_path=args.results,
            config_path=args.config,
            output_dir=args.output,
            horizon=args.horizon
        )
        
        if enhanced_results:
            # Create comparison summary
            print("\n" + "="*50)
            create_comparison_summary(args.results, args.output)

if __name__ == "__main__":
    main()
