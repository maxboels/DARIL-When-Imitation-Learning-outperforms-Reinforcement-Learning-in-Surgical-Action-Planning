# ===================================================================
# File: run_global_evaluation.py
# Main script for global video evaluation
# ===================================================================

import logging
from pathlib import Path
from global_video_evaluator import EnhancedActionAnalyzer


def run_global_video_evaluation(config_path: str = 'config_rl.yaml'):
    """
    Run global video evaluation with full temporal coverage
    """
    
    print("🌍 Starting Global Video Evaluation")
    print("=" * 60)
    
    # Setup
    import yaml
    from datasets.cholect50 import load_cholect50_data
    from models import WorldModel
    import logging
    import torch
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load test data
    print("📚 Loading test data...")
    test_data = load_cholect50_data(config, logger, split='test', max_videos=3)
    
    # Load models
    print("🤖 Loading models...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models = {}
    
    # Load world model
    try:
        world_model_path = config['experiment']['world_model']['best_model_path']
        checkpoint = torch.load(world_model_path, map_location=device)
        
        model_config = config['models']['world_model']
        world_model = WorldModel(**model_config).to(device)
        world_model.load_state_dict(checkpoint['model_state_dict'])
        world_model.eval()
        
        models['imitation_learning'] = world_model
        print("  ✅ World model loaded")
        
    except Exception as e:
        print(f"  ❌ Error loading world model: {e}")
        return
    
    # Load RL models
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
    
    # Initialize enhanced analyzer
    analyzer = EnhancedActionAnalyzer()
    
    # Environment config
    env_config = {
        'rl_horizon': 2000,  # Allow for long videos
        'context_length': config['data']['context_length'],
        'reward_weights': config.get('reward_weights', {})
    }
    
    # Collect global predictions
    print("\n🔄 Collecting global predictions...")
    global_predictions = analyzer.collect_global_predictions(
        models, test_data, world_model, env_config, device,
        max_frames_per_video=1500  # Evaluate up to 1500 frames per video
    )
    
    # Create enhanced visualizations
    print("\n🎨 Creating enhanced visualizations...")
    
    # Timeline visualizations for each video
    for video_id in global_predictions:
        print(f"  Creating enhanced timeline for {video_id}...")
        analyzer.create_enhanced_timeline_visualization(video_id)
    
    # Coverage analysis
    print("  Creating coverage analysis...")
    analyzer.create_coverage_analysis()
    
    print("\n" + "=" * 60)
    print("🎉 GLOBAL EVALUATION COMPLETE!")
    print("=" * 60)
    
    print(f"\n📁 Enhanced visualizations saved to: {analyzer.save_dir}")
    print("🎯 Key improvements:")
    print("   • Full video temporal coverage")
    print("   • Real RL policy evaluation") 
    print("   • Enhanced timeline visualizations")
    print("   • Coverage analysis across methods")
    
    print("\n📊 Files created:")
    for file in sorted(analyzer.save_dir.glob("*")):
        print(f"   - {file.name}")
    
    return analyzer

if __name__ == "__main__":
    analyzer = run_global_video_evaluation()
    print("\n🌍 Global video evaluation completed!")