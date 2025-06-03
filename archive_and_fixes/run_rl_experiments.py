# ===================================================================
# File: run_rl_experiments.py
# Main script to run the RL comparison experiments
# ===================================================================

import logging
from train import RLExperimentRunner
from datasets.cholect50 import load_cholect50_data


def run_rl_comparison_experiment(config_path: str = 'config.yaml'):
    """
    Main function to run the full RL comparison experiment
    """
    
    # Load config
    with open(config_path, 'r') as f:
        import yaml
        config = yaml.safe_load(f)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Add RL specific config
    config.update({
        'rl_horizon': 50,
        'rl_timesteps': 50000,
        'eval_horizon': 30,
        'reward_weights': {
            '_r_phase_completion': 1.0,
            '_r_phase_initiation': 0.5,
            '_r_phase_progression': 1.0,
            '_r_global_progression': 0.8,
            '_r_action_probability': 0.3,
            '_r_risk': -0.5,
        }
    })
    
    # Initialize experiment runner
    runner = RLExperimentRunner(config, logger)
    
    # Load data (using your existing function)
    from datasets.cholect50 import load_cholect50_data
    
    logger.info("Loading data...")
    train_data = load_cholect50_data(config, logger, split='train', max_videos=10)
    val_data = load_cholect50_data(config, logger, split='test', max_videos=5)
    
    # Load pre-trained world model
    world_model_path = config['experiment']['world_model']['best_model_path']
    runner.load_world_model(world_model_path)
    
    # Run experiments
    logger.info("=== Starting RL Comparison Experiment ===")
    
    # 1. Baseline Imitation Learning
    il_results = runner.run_baseline_imitation_learning(train_data, val_data)
    
    # 2. RL Algorithms
    rl_results = runner.run_rl_experiments(train_data, algorithms=['ppo', 'sac'])
    
    # 3. Compare results
    comparison = runner.compare_results()
    
    # 4. Save results
    runner.save_results('rl_comparison_results.json')
    
    # Print summary
    logger.info("=== EXPERIMENT SUMMARY ===")
    for method, metrics in comparison['comparison'].items():
        logger.info(f"{method}: {metrics}")
    
    for rec in comparison['recommendations']:
        logger.info(f"RECOMMENDATION: {rec}")
    
    return runner.results, comparison

if __name__ == "__main__":
    results, comparison = run_rl_comparison_experiment()
    print("RL Comparison Experiment Completed!")
    print(f"Summary: {comparison['recommendations']}")
