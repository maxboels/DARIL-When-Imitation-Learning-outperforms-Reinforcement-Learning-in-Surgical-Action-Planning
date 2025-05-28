#!/usr/bin/env python3
"""
System Validation Test
This script validates that your environment is ready for the IL vs RL comparison.
"""

import sys
import os
import importlib
import yaml
from pathlib import Path
import numpy as np
import torch


def test_python_version():
    """Test Python version compatibility."""
    print("🐍 Testing Python version...")
    
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.8+")
        return False


def test_dependencies():
    """Test required dependencies."""
    print("📦 Testing dependencies...")
    
    required_packages = [
        'torch',
        'numpy',
        'pandas', 
        'matplotlib',
        'seaborn',
        'sklearn',
        'tqdm',
        'yaml',
        'transformers'
    ]
    
    optional_packages = [
        'gymnasium',
        'stable_baselines3',
        'wandb',
        'plotly'
    ]
    
    results = {'required': 0, 'optional': 0}
    
    # Test required packages
    for package in required_packages:
        try:
            importlib.import_module(package if package != 'yaml' else 'yaml')
            print(f"  ✅ {package}")
            results['required'] += 1
        except ImportError:
            print(f"  ❌ {package} - REQUIRED")
    
    # Test optional packages  
    for package in optional_packages:
        try:
            importlib.import_module(package)
            print(f"  ✅ {package} (optional)")
            results['optional'] += 1
        except ImportError:
            print(f"  ⚠️ {package} - optional (needed for RL)")
    
    success = results['required'] == len(required_packages)
    print(f"📊 Dependencies: {results['required']}/{len(required_packages)} required, {results['optional']}/{len(optional_packages)} optional")
    
    return success


def test_cuda():
    """Test CUDA availability."""
    print("🔥 Testing CUDA...")
    
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0)
        memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        print(f"✅ CUDA available: {device_count} device(s)")
        print(f"  📱 Device 0: {device_name}")
        print(f"  💾 Memory: {memory:.1f} GB")
        
        # Test CUDA functionality
        try:
            x = torch.randn(100, 100).cuda()
            y = torch.randn(100, 100).cuda()
            z = torch.mm(x, y)
            print("  ✅ CUDA computation test passed")
            return True
        except Exception as e:
            print(f"  ❌ CUDA computation failed: {e}")
            return False
    else:
        print("⚠️ CUDA not available - will use CPU (slower)")
        return False


def test_config_files():
    """Test configuration files."""
    print("⚙️ Testing configuration files...")
    
    config_files = ['config.yaml', 'config_fixed.yaml']
    config_found = None
    
    for config_file in config_files:
        if os.path.exists(config_file):
            config_found = config_file
            print(f"✅ Found config: {config_file}")
            break
    
    if not config_found:
        print("❌ No configuration file found!")
        print("   Create either config.yaml or config_fixed.yaml")
        return False
    
    # Test config loading
    try:
        with open(config_found, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check essential sections
        essential_keys = ['data', 'models', 'training', 'experiment']
        missing_keys = [key for key in essential_keys if key not in config]
        
        if missing_keys:
            print(f"❌ Missing config sections: {missing_keys}")
            return False
        
        print("✅ Configuration file valid")
        
        # Check data directory
        data_dir = config.get('data', {}).get('paths', {}).get('data_dir', '')
        if data_dir and os.path.exists(data_dir):
            print(f"✅ Data directory found: {data_dir}")
        else:
            print(f"⚠️ Data directory not found: {data_dir}")
            print("   Update data_dir in config or ensure CholecT50 is available")
        
        return True
        
    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        return False


def test_project_structure():
    """Test project structure."""
    print("📁 Testing project structure...")
    
    required_files = [
        'datasets/cholect50.py',
        'utils/logger.py',
        'models/dual_world_model.py',
        'training/dual_trainer.py'
    ]
    
    optional_files = [
        'comprehensive_evaluation.py',
        'prediction_saver.py',
        'enhanced_interactive_viz.html'
    ]
    
    missing_required = []
    missing_optional = []
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path} - REQUIRED")
            missing_required.append(file_path)
    
    for file_path in optional_files:
        if os.path.exists(file_path):
            print(f"  ✅ {file_path}")
        else:
            print(f"  ⚠️ {file_path} - optional")
            missing_optional.append(file_path)
    
    if missing_required:
        print(f"❌ Missing required files: {len(missing_required)}")
        return False
    else:
        print("✅ All required files present")
        return True


def test_data_loading():
    """Test data loading functionality."""
    print("📊 Testing data loading...")
    
    try:
        # Import required modules
        from datasets.cholect50 import load_cholect50_data
        from utils.logger import SimpleLogger
        
        print("  ✅ Imports successful")
        
        # Load config
        config_file = 'config_fixed.yaml' if os.path.exists('config_fixed.yaml') else 'config.yaml'
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Create logger
        logger = SimpleLogger(log_dir="test_logs", name="validation_test")
        
        # Test data loading (just 1 video)
        print("  🔄 Testing data loading (1 video)...")
        test_data = load_cholect50_data(config, logger, split='train', max_videos=1)
        
        if test_data and len(test_data) > 0:
            video = test_data[0]
            print(f"  ✅ Data loading successful")
            print(f"    📹 Video ID: {video['video_id']}")
            print(f"    🎞️ Frames: {video['num_frames']}")
            print(f"    📐 Embedding shape: {video['frame_embeddings'].shape}")
            print(f"    🎯 Actions shape: {video['actions_binaries'].shape}")
            
            # Clean up test logs
            import shutil
            if os.path.exists("test_logs"):
                shutil.rmtree("test_logs")
            
            return True
        else:
            print("  ❌ No data loaded")
            return False
            
    except ImportError as e:
        print(f"  ❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"  ❌ Data loading failed: {e}")
        return False


def test_model_creation():
    """Test model creation."""
    print("🧠 Testing model creation...")
    
    try:
        from models.dual_world_model import DualWorldModel
        
        # Create a small test model
        model_config = {
            'hidden_dim': 256,
            'embedding_dim': 512,
            'action_embedding_dim': 64,
            'n_layer': 2,
            'num_action_classes': 100,
            'num_phase_classes': 7,
            'max_length': 512,
            'dropout': 0.1
        }
        
        model = DualWorldModel(**model_config)
        
        # Test forward pass
        batch_size, seq_len = 2, 10
        current_states = torch.randn(batch_size, seq_len, model_config['embedding_dim'])
        
        output = model(current_states=current_states, mode='supervised')
        
        print("  ✅ Model creation successful")
        print(f"    🔧 Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"    📤 Output keys: {list(output.keys())}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Model creation failed: {e}")
        return False


def run_full_validation():
    """Run all validation tests."""
    print("🧪 SYSTEM VALIDATION TEST")
    print("=" * 50)
    
    tests = [
        ("Python Version", test_python_version),
        ("Dependencies", test_dependencies), 
        ("CUDA Support", test_cuda),
        ("Configuration", test_config_files),
        ("Project Structure", test_project_structure),
        ("Data Loading", test_data_loading),
        ("Model Creation", test_model_creation)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 30)
        
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 VALIDATION SUMMARY")
    print("=" * 50)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<20} {status}")
    
    print("-" * 50)
    print(f"Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("Your system is ready for the IL vs RL comparison!")
        print("\nNext steps:")
        print("1. Run: python run_experiment_script.py")
        print("2. Or: python complete_comparison_script.py") 
    else:
        print(f"\n⚠️ {total - passed} tests failed!")
        print("Please fix the failing tests before proceeding.")
        print("Refer to the setup instructions for help.")
    
    return passed == total


def main():
    """Main validation function."""
    success = run_full_validation()
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
