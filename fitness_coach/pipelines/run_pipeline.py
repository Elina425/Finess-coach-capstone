#!/usr/bin/env python3
"""
Non-interactive pipeline runner for QEVD-FIT-COACH
Runs the complete pipeline without prompting for input
"""

import sys
from pathlib import Path
from fitness_coach.pipelines.comprehensive_pipeline import ComprehensivePipeline


def main():
    """Main entry point - non-interactive mode"""
    
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║     QEVD-FIT-COACH POSE ESTIMATION PIPELINE                         ║
║                          AUTO MODE                                  ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    # Uses actual organized dataset location
    DATASET_ROOT = "./qevd-fit-coach-data"
    OUTPUT_DIR = "./results"
    
    # Configuration
    EXERCISE_TYPE = None  # All exercise types
    MAX_VIDEOS = 10  # Process first 10 videos for testing
    
    print(f"\nConfiguration:")
    print(f"  Dataset: {DATASET_ROOT}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Exercise type: {EXERCISE_TYPE or 'All'}")
    print(f"  Max videos: {MAX_VIDEOS or 'All'}")
    
    # Check dataset
    print("\n" + "="*70)
    print("CHECKING DATASET...")
    print("="*70)
    
    dataset_path = Path(DATASET_ROOT)
    if not dataset_path.exists():
        print(f"✗ Dataset not found at: {DATASET_ROOT}")
        return 1
    
    videos_dir = dataset_path / "videos"
    if not videos_dir.exists():
        print(f"✗ Videos directory not found at: {videos_dir}")
        return 1
    
    # Search recursively for videos in subdirectories (long_range, short_clips)
    video_count = len(list(videos_dir.rglob("*.mp4")))
    print(f"✓ Dataset found with {video_count} videos")
    
    # Initialize pipeline
    print("\n" + "="*70)
    print("INITIALIZING PIPELINE...")
    print("="*70)
    
    try:
        pipeline = ComprehensivePipeline(DATASET_ROOT, OUTPUT_DIR)
        print("✓ Pipeline initialized")
    except Exception as e:
        print(f"✗ Failed to initialize pipeline: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Run pipeline
    print("\n" + "="*70)
    print("RUNNING PIPELINE (pose stack selected in comprehensive_pipeline)")
    print("="*70)
    
    try:
        pipeline.run_complete_pipeline(
            exercise_type=EXERCISE_TYPE,
            max_videos=MAX_VIDEOS
        )
        
        print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║                    PIPELINE COMPLETE                                ║
╚══════════════════════════════════════════════════════════════════════╝

Results saved to: {OUTPUT_DIR}

Next steps:
  1. Check results/ folder for processed keypoints
  2. Review summary_report.json for statistics
  3. Examine visualizations in results/visualizations/
        """)
        return 0
        
    except Exception as e:
        print(f"\n✗ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
