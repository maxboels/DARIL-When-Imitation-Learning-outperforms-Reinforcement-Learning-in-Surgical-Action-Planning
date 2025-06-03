# ===================================================================
# File: run_video_animations.py
# Main script to create surgical video animations
# ===================================================================


from typing import Dict, List
import numpy as np
from surgical_video_animator import SurgicalActionAnimator


def create_surgical_animations(video_data: Dict, predictions: Dict, 
                              video_id: str, animation_types: List[str] = None):
    """
    Create various types of surgical action animations
    
    Args:
        video_data: Video data with ground truth
        predictions: Predictions from different methods  
        video_id: Video identifier
        animation_types: Types of animations to create
    """
    
    if animation_types is None:
        animation_types = ['realtime', 'comparative', 'trajectory_gif']
    
    animator = SurgicalActionAnimator()
    
    created_files = []
    
    try:
        if 'realtime' in animation_types:
            print("🎬 Creating real-time prediction animation...")
            file_path = animator.create_realtime_prediction_animation(
                video_data, predictions, video_id, max_frames=300
            )
            created_files.append(file_path)
        
        if 'comparative' in animation_types:
            print("🎭 Creating comparative animation...")
            file_path = animator.create_comparative_animation(
                video_data, predictions, video_id, max_frames=200
            )
            created_files.append(file_path)
        
        if 'trajectory_gif' in animation_types:
            print("🎪 Creating trajectory evolution GIF...")
            file_path = animator.create_trajectory_evolution_gif(
                video_data, predictions, video_id, max_frames=150
            )
            created_files.append(file_path)
            
    except Exception as e:
        print(f"❌ Error creating animations: {e}")
        return []
    
    return created_files

def run_video_animation_suite():
    """
    Run complete video animation suite
    """
    
    print("🎬 Starting Surgical Video Animation Suite")
    print("=" * 60)
    
    # Load your existing data and predictions
    try:
        # Load from your existing analysis results
        import json
        from pathlib import Path
        
        # Check if global evaluation results exist
        global_results_path = Path('enhanced_action_analysis/global_evaluation_results.json')
        if global_results_path.exists():
            print("📁 Loading global evaluation results...")
            with open(global_results_path, 'r') as f:
                results = json.load(f)
            
            global_predictions = results['global_predictions']
            ground_truth = results['ground_truth']
            
            # Create animations for each video
            for video_id in global_predictions:
                print(f"\n🎥 Creating animations for {video_id}")
                
                # Prepare data
                video_data = {'actions_binaries': np.array(ground_truth[video_id])}
                predictions = {k: np.array(v) for k, v in global_predictions[video_id].items()}
                
                # Create animations
                created_files = create_surgical_animations(
                    video_data, predictions, video_id,
                    animation_types=['realtime', 'comparative', 'trajectory_gif']
                )
                
                print(f"  ✅ Created {len(created_files)} animations for {video_id}")
                for file_path in created_files:
                    print(f"    - {Path(file_path).name}")
        
        else:
            print("❌ Global evaluation results not found!")
            print("Please run global evaluation first:")
            print("python run_global_evaluation.py")
            return
            
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    print("\n" + "=" * 60)
    print("🎉 VIDEO ANIMATION SUITE COMPLETE!")
    print("=" * 60)
    
    # List all created files
    animation_dir = Path('surgical_animations')
    if animation_dir.exists():
        print(f"\n📁 Animations saved to: {animation_dir}")
        print("📹 Created files:")
        for file_path in sorted(animation_dir.glob("*")):
            print(f"   - {file_path.name}")
        
        print("\n🎯 Animation types:")
        print("   • realtime_prediction_*.mp4 - Real-time prediction evolution")
        print("   • comparative_*.mp4 - Side-by-side method comparison")  
        print("   • trajectory_evolution_*.gif - Action trajectory evolution")
        
        print("\n💡 How to view:")
        print("   • MP4 files: Open with any video player")
        print("   • GIF files: Open in browser or image viewer")
        print("   • Use for presentations, papers, and demos!")

if __name__ == "__main__":
    # Install required packages
    print("📦 Checking required packages...")
    try:
        import matplotlib.animation
        import cv2
        from PIL import Image
        print("✅ All packages available")
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("Install with: pip install matplotlib opencv-python pillow")
        exit(1)
    
    run_video_animation_suite()