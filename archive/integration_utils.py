#!/usr/bin/env python3
"""
Integration Utilities and Validation Script
Helps with integrating and validating the enhanced evaluation framework
"""

import json
import yaml
import os
import sys
from pathlib import Path
import torch
import numpy as np
from typing import Dict, List, Any, Optional

class IntegrationValidator:
    """Validates that all components are properly integrated"""
    
    def __init__(self):
        self.issues = []
        self.warnings = []
        self.success = []
    
    def validate_file_structure(self, project_root: str = "."):
        """Validate that all required files are in place"""
        
        project_root = Path(project_root)
        
        required_files = {
            'integrated_evaluation_framework.py': 'evaluation/',
            'standalone_integrated_eval.py': '.',
            'updated_visualization.html': 'visualization/',
            'run_experiment_v2.py': '.',
        }
        
        print("🔍 Validating file structure...")
        
        for filename, expected_location in required_files.items():
            expected_path = project_root / expected_location / filename
            
            if expected_path.exists():
                self.success.append(f"✅ Found {filename} in {expected_location}")
            else:
                # Check if it exists elsewhere
                found_elsewhere = list(project_root.rglob(filename))
                if found_elsewhere:
                    self.warnings.append(f"⚠️ {filename} found at {found_elsewhere[0]} but expected in {expected_location}")
                else:
                    self.issues.append(f"❌ Missing {filename} - should be in {expected_location}")
        
        # Check for config file
        config_files = ['config.yaml', 'config_dgx_all.yaml', 'config_local_debug.yaml']
        config_found = False
        for config_file in config_files:
            if (project_root / config_file).exists():
                self.success.append(f"✅ Found config file: {config_file}")
                config_found = True
                break
        
        if not config_found:
            self.issues.append(f"❌ No config file found. Expected one of: {config_files}")
    
    def validate_dependencies(self):
        """Validate that all required Python packages are available"""
        
        print("🔍 Validating dependencies...")
        
        required_packages = {
            'torch': 'PyTorch for model loading',
            'numpy': 'NumPy for numerical operations',
            'pandas': 'Pandas for data handling',
            'matplotlib': 'Matplotlib for plotting',
            'seaborn': 'Seaborn for visualization',
            'sklearn': 'Scikit-learn for metrics',
            'scipy': 'SciPy for statistical tests',
            'stable_baselines3': 'Stable-Baselines3 for RL models'
        }
        
        for package, description in required_packages.items():
            try:
                __import__(package)
                self.success.append(f"✅ {package} available ({description})")
            except ImportError:
                self.issues.append(f"❌ Missing {package} - {description}")
    
    def validate_config_structure(self, config_path: str):
        """Validate that config file has required sections"""
        
        print(f"🔍 Validating config structure: {config_path}")
        
        if not Path(config_path).exists():
            self.issues.append(f"❌ Config file not found: {config_path}")
            return
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            required_sections = {
                'data': 'Data configuration',
                'models': 'Model configurations',
                'training': 'Training parameters',
                'experiment': 'Experiment settings'
            }
            
            for section, description in required_sections.items():
                if section in config:
                    self.success.append(f"✅ Config has {section} section ({description})")
                else:
                    self.warnings.append(f"⚠️ Config missing {section} section - {description}")
            
            # Check for evaluation section (new)
            if 'evaluation' in config:
                eval_config = config['evaluation']
                if 'horizon' in eval_config:
                    self.success.append(f"✅ Evaluation horizon configured: {eval_config['horizon']}")
                else:
                    self.warnings.append("⚠️ Evaluation horizon not configured - will use default")
            else:
                self.warnings.append("⚠️ No evaluation section in config - will use defaults")
                
        except Exception as e:
            self.issues.append(f"❌ Error reading config file: {e}")
    
    def validate_results_file(self, results_path: str):
        """Validate that results file has expected structure"""
        
        print(f"🔍 Validating results file: {results_path}")
        
        if not Path(results_path).exists():
            self.issues.append(f"❌ Results file not found: {results_path}")
            return None
        
        try:
            with open(results_path, 'r') as f:
                results = json.load(f)
            
            # Check for required method results
            expected_methods = [
                'method_1_il_baseline',
                'method_2_rl_world_model', 
                'method_3_rl_offline_videos'
            ]
            
            available_models = 0
            
            for method in expected_methods:
                if method in results:
                    method_result = results[method]
                    if method_result.get('status') == 'success':
                        self.success.append(f"✅ {method} completed successfully")
                        
                        # Check for model paths
                        if method == 'method_1_il_baseline':
                            if 'model_path' in method_result:
                                available_models += 1
                                self.success.append(f"✅ IL model path available")
                            else:
                                self.warnings.append(f"⚠️ IL model path missing")
                        else:
                            if 'rl_models' in method_result:
                                rl_models = method_result['rl_models']
                                for alg, alg_result in rl_models.items():
                                    if alg_result.get('status') == 'success' and 'model_path' in alg_result:
                                        available_models += 1
                                        self.success.append(f"✅ {method} {alg} model available")
                    else:
                        self.warnings.append(f"⚠️ {method} status: {method_result.get('status', 'unknown')}")
                else:
                    self.warnings.append(f"⚠️ {method} not found in results")
            
            if available_models >= 2:
                self.success.append(f"✅ {available_models} models available for comparison")
            else:
                self.issues.append(f"❌ Only {available_models} models available - need at least 2 for comparison")
            
            return results
            
        except Exception as e:
            self.issues.append(f"❌ Error reading results file: {e}")
            return None
    
    def print_validation_report(self):
        """Print validation report"""
        
        print("\n" + "="*60)
        print("🔍 INTEGRATION VALIDATION REPORT")
        print("="*60)
        
        if self.success:
            print(f"\n✅ SUCCESS ({len(self.success)} items):")
            for item in self.success:
                print(f"  {item}")
        
        if self.warnings:
            print(f"\n⚠️ WARNINGS ({len(self.warnings)} items):")
            for item in self.warnings:
                print(f"  {item}")
        
        if self.issues:
            print(f"\n❌ ISSUES ({len(self.issues)} items):")
            for item in self.issues:
                print(f"  {item}")
        
        print("\n" + "="*60)
        
        if self.issues:
            print("❌ VALIDATION FAILED - Fix issues above before proceeding")
            return False
        elif self.warnings:
            print("⚠️ VALIDATION PASSED WITH WARNINGS - Review warnings above")
            return True
        else:
            print("✅ VALIDATION PASSED - Ready for integrated evaluation!")
            return True

def create_sample_config():
    """Create a sample config with evaluation settings"""
    
    config = {
        'evaluation': {
            'horizon': 15,
            'max_videos': 10,
            'save_detailed_rollouts': True,
            'create_visualization_data': True,
            'significance_level': 0.05
        },
        'data': {
            'context_length': 20,
            'max_horizon': 15,
            'paths': {
                'data_dir': "/path/to/CholecT50",
                'metadata_file': "embeddings_f0_swin_bas_129_phase_complet_phase_transit_prog_prob_action_risk_glob_outcome.csv"
            }
        },
        'models': {
            'dual_world_model': {
                'hidden_dim': 768,
                'embedding_dim': 1024,
                'num_action_classes': 100
            }
        },
        'training': {
            'batch_size': 16,
            'learning_rate': 0.0001,
            'epochs': 1
        },
        'experiment': {
            'train': {'max_videos': 20},
            'test': {'max_videos': 10}
        }
    }
    
    return config

def create_sample_visualization_data():
    """Create sample visualization data for testing"""
    
    # Create minimal sample data
    sample_data = {
        'ground_truth': {
            'VID01': {
                'actions': [
                    [0] * 100 for _ in range(50)  # 50 timesteps, 100 actions each
                ],
                'phases': [
                    [1, 0, 0, 0, 0, 0, 0] for _ in range(50)  # 50 timesteps, 7 phases
                ]
            }
        },
        'predictions': {
            'IL_Baseline': {
                'VID01': {
                    'past_actions': [
                        [0] * 100 for _ in range(49)  # Past predictions
                    ],
                    'future_rollouts': {
                        '10': [  # Rollout from timestep 10
                            {
                                'step': 1,
                                'predicted_action': [0] * 100,
                                'confidence': 0.75,
                                'active_actions': 3
                            }
                        ]
                    },
                    'confidence_timeline': [0.7] * 49,
                    'thinking_process': {
                        '10': [
                            "Analyzing current surgical context",
                            "Identifying potential actions",
                            "Generating predictions"
                        ]
                    },
                    'metadata': {
                        'method_type': 'IL',
                        'avg_confidence': 0.75
                    }
                }
            }
        },
        'metadata': {
            'methods': ['IL_Baseline'],
            'videos': ['VID01'],
            'evaluation_timestamp': '2025-06-04T12:00:00',
            'per_method': {
                'IL_Baseline': {
                    'avg_confidence': 0.75,
                    'performance_rank': 1,
                    'method_type': 'IL'
                }
            }
        }
    }
    
    return sample_data

def setup_project_structure(project_root: str = "."):
    """Setup the project structure for integrated evaluation"""
    
    project_root = Path(project_root)
    
    print("🔧 Setting up project structure...")
    
    # Create directories
    directories = [
        'evaluation',
        'visualization',
        'logs',
        'data'
    ]
    
    for directory in directories:
        dir_path = project_root / directory
        dir_path.mkdir(exist_ok=True)
        print(f"✅ Created/verified directory: {directory}")
    
    # Create sample config if none exists
    config_files = ['config.yaml', 'config_dgx_all.yaml']
    config_exists = any((project_root / cf).exists() for cf in config_files)
    
    if not config_exists:
        sample_config = create_sample_config()
        config_path = project_root / 'config_sample.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(sample_config, f, indent=2)
        print(f"✅ Created sample config: {config_path}")
    
    # Create sample visualization data
    viz_dir = project_root / 'visualization'
    sample_viz_path = viz_dir / 'sample_visualization_data.json'
    
    if not sample_viz_path.exists():
        sample_data = create_sample_visualization_data()
        with open(sample_viz_path, 'w') as f:
            json.dump(sample_data, f, indent=2)
        print(f"✅ Created sample visualization data: {sample_viz_path}")
    
    print("✅ Project structure setup complete!")

def main():
    """Main function for utility script"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Integration Utilities for Enhanced Evaluation')
    parser.add_argument('--validate', action='store_true', help='Validate integration')
    parser.add_argument('--setup', action='store_true', help='Setup project structure')
    parser.add_argument('--config', default='config.yaml', help='Config file to validate')
    parser.add_argument('--results', help='Results file to validate')
    parser.add_argument('--project-root', default='.', help='Project root directory')
    
    args = parser.parse_args()
    
    if args.setup:
        setup_project_structure(args.project_root)
        return 0
    
    if args.validate:
        validator = IntegrationValidator()
        
        # Validate file structure
        validator.validate_file_structure(args.project_root)
        
        # Validate dependencies
        validator.validate_dependencies()
        
        # Validate config
        config_path = Path(args.project_root) / args.config
        if config_path.exists():
            validator.validate_config_structure(str(config_path))
        else:
            # Try to find any config file
            for config_name in ['config.yaml', 'config_dgx_all.yaml', 'config_local_debug.yaml']:
                config_path = Path(args.project_root) / config_name
                if config_path.exists():
                    validator.validate_config_structure(str(config_path))
                    break
        
        # Validate results file if provided
        if args.results:
            validator.validate_results_file(args.results)
        
        # Print report
        success = validator.print_validation_report()
        
        if success:
            print("\n🚀 NEXT STEPS:")
            print("1. Run your experiment: python run_experiment_v2.py")
            print("2. Or run standalone evaluation: python standalone_integrated_eval.py")
            print("3. Use the HTML visualization tool with the generated visualization_data.json")
            return 0
        else:
            print("\n🔧 FIX ISSUES THEN RE-RUN VALIDATION:")
            print(f"python {__file__} --validate")
            return 1
    
    # Default: show help
    parser.print_help()
    return 0

if __name__ == "__main__":
    exit(main())
