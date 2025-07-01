#!/usr/bin/env python3
"""
Complete Prior Extraction Workflow
Run this script BEFORE your IRL training to extract all necessary priors
"""

import sys
import os
from pathlib import Path
import argparse

def main():
    """Complete workflow for prior extraction"""
    
    print("🔬 SURGICAL PRIOR EXTRACTION WORKFLOW")
    print("=" * 50)
    print("Extracting training data patterns to inform safety guardrails")
    print("Target: High-opportunity components (T: 52%, IVT: 33%)")
    print()
    
    parser = argparse.ArgumentParser(description="Extract surgical priors for IRL training")
    parser.add_argument('--config', type=str, default='config_dgx_all_v8.yaml', 
                       help="Configuration file path")
    parser.add_argument('--output', type=str, default='data/surgical_priors', 
                       help="Output directory for extracted priors")
    parser.add_argument('--max_videos', type=int, default=None, 
                       help="Maximum training videos to process (None for all)")
    parser.add_argument('--quick_test', action='store_true', 
                       help="Quick test with 2 videos only")
    
    args = parser.parse_args()
    
    # Quick test mode
    if args.quick_test:
        args.max_videos = 2
        args.output = 'surgical_priors_test'
        print("🧪 QUICK TEST MODE: Processing 2 videos only")
    
    # Check if config exists
    if not os.path.exists(args.config):
        print(f"❌ Configuration file not found: {args.config}")
        print("💡 Please ensure your config file exists")
        return False
    
    print(f"📋 Configuration: {args.config}")
    print(f"📁 Output directory: {args.output}")
    print(f"📹 Max videos: {args.max_videos or 'all'}")
    print()
    
    try:
        # Import and run extraction
        from datasets.negative_prior_extraction import extract_surgical_priors
        
        success = extract_surgical_priors(
            config_path=args.config,
            output_dir=args.output,
            max_videos=args.max_videos
        )
        
        if success:
            print("\n🎉 PRIOR EXTRACTION COMPLETED SUCCESSFULLY!")
            print("=" * 50)
            print(f"📁 Results saved to: {args.output}/")
            print()
            print("📋 Generated files:")
            output_path = Path(args.output)
            for file_path in output_path.glob('*.json'):
                print(f"   📄 {file_path.name}")
            for file_path in output_path.glob('*.png'):
                print(f"   📊 {file_path.name}")
            
            print()
            print("🚀 NEXT STEPS:")
            print("1. Review the extracted patterns in the output directory")
            print("2. Check the visualizations to understand your data")
            print("3. Use these priors in your IRL training:")
            print(f"   from prior_integration_example import PriorInformedNegativeGenerator")
            print(f"   generator = PriorInformedNegativeGenerator('{args.output}')")
            print("4. Integrate with your IRL trainer for targeted negative generation")
            print()
            print("✅ Ready for IRL training with safety guardrails!")
            
            return True
            
        else:
            print("\n❌ PRIOR EXTRACTION FAILED")
            print("💡 Check your data paths and configuration")
            return False
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Ensure all required modules are available")
        print("💡 Make sure negative_prior_extraction.py is in your path")
        return False
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
