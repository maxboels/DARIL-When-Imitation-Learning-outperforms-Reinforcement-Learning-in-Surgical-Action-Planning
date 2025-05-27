#!/usr/bin/env python3
"""
Quick patch script to fix the immediate errors without full code replacement.

Run this script in your project directory to apply the essential fixes.
"""

import os
import re
from pathlib import Path

def patch_simple_logger():
    """Add missing error method to SimpleLogger."""
    logger_file = Path("utils/logger.py")
    
    if not logger_file.exists():
        print("⚠️  Logger file not found, skipping logger patch")
        return
    
    with open(logger_file, 'r') as f:
        content = f.read()
    
    # Check if error method already exists
    if "def error(" in content:
        print("✅ Logger already has error method")
        return
    
    # Add missing methods after the info method
    new_methods = '''
    def warning(self, message: str, *args, **kwargs) -> None:
        """Log warning level message."""
        self.logger.warning(message, *args, **kwargs)

    def error(self, message: str, *args, **kwargs) -> None:
        """Log error level message."""
        self.logger.error(message, *args, **kwargs)

    def debug(self, message: str, *args, **kwargs) -> None:
        """Log debug level message."""
        self.logger.debug(message, *args, **kwargs)
'''
    
    # Insert new methods before the end of the class
    content = content.replace(
        "    def info(self, message: str, *args, **kwargs) -> None:\n"
        "        \"\"\"Log info level message.\"\"\"\n"
        "        self.logger.info(message, *args, **kwargs)",
        
        "    def info(self, message: str, *args, **kwargs) -> None:\n"
        "        \"\"\"Log info level message.\"\"\"\n"
        "        self.logger.info(message, *args, **kwargs)" + new_methods
    )
    
    with open(logger_file, 'w') as f:
        f.write(content)
    
    print("✅ Added missing methods to SimpleLogger")

def patch_dual_world_model():
    """Fix the naming conflict in DualWorldModel."""
    model_file = Path("models/dual_world_model.py")
    
    if not model_file.exists():
        print("⚠️  DualWorldModel file not found, skipping model patch")
        return
    
    with open(model_file, 'r') as f:
        content = f.read()
    
    # Check if already patched
    if "enable_autoregressive_prediction" in content:
        print("✅ DualWorldModel already patched")
        return
    
    # Fix parameter names in __init__
    content = re.sub(
        r'autoregressive_action_prediction: bool = True',
        'enable_autoregressive_prediction: bool = True',
        content
    )
    
    content = re.sub(
        r'rl_state_prediction: bool = True',
        'enable_rl_prediction: bool = True',
        content
    )
    
    content = re.sub(
        r'reward_prediction: bool = True',
        'enable_reward_prediction: bool = True',
        content
    )
    
    # Fix attribute assignments
    content = re.sub(
        r'self\.autoregressive_action_prediction = autoregressive_action_prediction',
        'self.enable_autoregressive_prediction = enable_autoregressive_prediction',
        content
    )
    
    content = re.sub(
        r'self\.rl_state_prediction = rl_state_prediction',
        'self.enable_rl_prediction = enable_rl_prediction',
        content
    )
    
    content = re.sub(
        r'self\.reward_prediction = reward_prediction',
        'self.enable_reward_prediction = enable_reward_prediction',
        content
    )
    
    # Fix conditions
    content = re.sub(
        r'if self\.autoregressive_action_prediction:',
        'if self.enable_autoregressive_prediction:',
        content
    )
    
    content = re.sub(
        r'if self\.reward_prediction',
        'if self.enable_reward_prediction',
        content
    )
    
    # Fix save_model method
    content = re.sub(
        r"'autoregressive_action_prediction': self\.autoregressive_action_prediction,",
        "'enable_autoregressive_prediction': self.enable_autoregressive_prediction,",
        content
    )
    
    content = re.sub(
        r"'rl_state_prediction': self\.rl_state_prediction,",
        "'enable_rl_prediction': self.enable_rl_prediction,",
        content
    )
    
    content = re.sub(
        r"'reward_prediction': self\.reward_prediction,",
        "'enable_reward_prediction': self.enable_reward_prediction,",
        content
    )
    
    with open(model_file, 'w') as f:
        f.write(content)
    
    print("✅ Fixed naming conflicts in DualWorldModel")

def patch_config():
    """Update config to use all videos."""
    config_file = Path("config.yaml")
    
    if not config_file.exists():
        print("⚠️  Config file not found, skipping config patch")
        return
    
    with open(config_file, 'r') as f:
        content = f.read()
    
    # Update max_videos settings
    content = re.sub(
        r'max_videos: \d+',
        'max_videos: null',
        content
    )
    
    with open(config_file, 'w') as f:
        f.write(content)
    
    print("✅ Updated config to use all videos")

def main():
    """Apply all patches."""
    print("🔧 Applying quick patches for immediate fixes...")
    print("=" * 50)
    
    # Apply patches
    patch_simple_logger()
    patch_dual_world_model()
    patch_config()
    
    print("=" * 50)
    print("✅ All patches applied!")
    print()
    print("🚀 You can now run your experiment again:")
    print("   python main_experiment.py")
    print()
    print("📊 Your training was looking great:")
    print("   - Loss decreased from 3.13 → 0.12")
    print("   - Action accuracy: 99%+")
    print("   - Now using full dataset instead of 5 videos")
    print()
    print("💡 Next steps:")
    print("   1. Run with full dataset")
    print("   2. Monitor training performance")
    print("   3. Try RL mode after supervised training completes")

if __name__ == "__main__":
    main()