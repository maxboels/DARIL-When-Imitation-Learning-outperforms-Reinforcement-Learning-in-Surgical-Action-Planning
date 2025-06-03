# ===================================================================
# File: run_action_analysis.py  
# Main script to run action analysis
# ===================================================================


import torch
from pathlib import Path
from action_analysis import SurgicalActionAnalyzer

def run_complete_action_analysis(config_path: str = 'config_rl.yaml'):
    """
    Run complete action analysis comparing predicted vs ground truth actions
    """
    
    print("🔍 Starting Comprehensive Action Analysis")
    print("=" * 60)
    
    # Load configuration and data
    import yaml
    from datasets.cholect50 import load_cholect50_data
    from models import WorldModel
    import logging
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load test data
    print("📚 Loading test data...")
    test_data = load_cholect50_data(config, logger, split='test', max_videos=3)
    
    # Load models
    print("🤖 Loading trained models...")
    models = {}
    
    # Load world model for imitation learning baseline
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        world_model_path = config['experiment']['world_model']['best_model_path']
        checkpoint = torch.load(world_model_path, map_location=device)
        
        model_config = config['models']['world_model']
        world_model = WorldModel(**model_config).to(device)
        world_model.load_state_dict(checkpoint['model_state_dict'])
        world_model.eval()
        
        models['imitation_learning'] = world_model
        print("  ✅ World model loaded for imitation learning")
        
    except Exception as e:
        print(f"  ❌ Error loading world model: {e}")
    
    # Load RL models (if available)
    try:
        from stable_baselines3 import PPO, SAC
        
        if Path('surgical_ppo_policy.zip').exists():
            ppo_model = PPO.load('surgical_ppo_policy.zip')
            models['ppo'] = ppo_model
            print("  ✅ PPO model loaded")
        
        if Path('surgical_sac_policy.zip').exists():
            sac_model = SAC.load('surgical_sac_policy.zip')
            models['sac'] = sac_model
            print("  ✅ SAC model loaded")
            
    except Exception as e:
        print(f"  ⚠️  RL models not available: {e}")
    
    if not models:
        print("❌ No models available for analysis!")
        return
    
    # Initialize analyzer
    analyzer = SurgicalActionAnalyzer()
    
    # Collect predictions
    print("\n🔍 Collecting action predictions...")
    predictions = analyzer.collect_predictions(models, test_data, device)
    
    # Create visualizations
    print("\n🎨 Creating visualizations...")
    report = analyzer.create_all_visualizations()
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 ACTION ANALYSIS SUMMARY")
    print("=" * 60)
    
    for method, metrics in report['summary'].items():
        print(f"\n🤖 {method.replace('_', ' ').title()}:")
        print(f"   Hamming Loss: {metrics['hamming_loss']:.4f}")
        print(f"   Jaccard Score: {metrics['jaccard_score']:.4f}")
        print(f"   Avg Actions/Frame: {metrics['action_diversity']:.2f}")
    
    print("\n💡 Key Insights:")
    for insight in report['qualitative_insights']:
        print(f"   • {insight}")
    
    print(f"\n📁 All visualizations saved to: ./action_analysis/")
    print("🎯 Key files for analysis:")
    print("   • action_frequency_analysis.png - Overall action patterns")
    print("   • action_confusion_matrices.png - Prediction accuracy")
    print("   • phase_specific_analysis.png - Phase-wise performance") 
    print("   • action_timeline_*.html - Interactive timelines per video")
    print("   • action_analysis_report.json - Detailed metrics")
    
    return analyzer, report

if __name__ == "__main__":
    analyzer, report = run_complete_action_analysis()
    print("\n🎉 Action analysis completed!")
