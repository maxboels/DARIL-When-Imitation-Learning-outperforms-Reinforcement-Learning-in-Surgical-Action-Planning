
# Integration script for your existing codebase
import sys
import os

# Add your project root to path if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import your existing modules
from models import WorldModel
from datasets import load_cholect50_data, NextFramePredictionDataset

# Import new RL components
from models import SurgicalWorldModelEnv
from train import RLExperimentRunner

def main():
    """Run the integrated RL experiment"""
    from run_rl_experiments import run_rl_comparison_experiment
    
    # Run the experiment with your config
    results, comparison = run_rl_comparison_experiment('config_rl.yaml')
    
    print("Experiment completed!")
    print("Check rl_comparison_results.json for detailed results")

if __name__ == "__main__":
    main()
