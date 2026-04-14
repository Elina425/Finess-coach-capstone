"""
Quick-Start Guide: Running the QEVD-FIT-COACH Pipeline
Execute this file to run the complete pipeline with example configuration
"""

import sys
from pathlib import Path
from fitness_coach.pipelines.comprehensive_pipeline import ComprehensivePipeline


def check_dataset_exists(dataset_root: str) -> bool:
    """Check if dataset exists and is properly structured"""
    path = Path(dataset_root)
    
    if not path.exists():
        print(f"\n⚠ Dataset not found at: {dataset_root}")
        print("\nTo set up the dataset:")
        print("1. Download QEVD-FIT-COACH from Qualcomm website")
        print("2. Extract WITHOUT short clips")
        print("3. Organize as:")
        print("   qevd-fit-coach-data/")
        print("   ├── videos/")
        print("   ├── annotations/")
        print("   └── metadata.csv")
        return False
    
    if not (path / "videos").exists():
        print(f"✗ Missing 'videos' directory in {dataset_root}")
        return False
    
    return True


def main():
    """Main entry point for quick start"""
    
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║     QEVD-FIT-COACH POSE ESTIMATION PIPELINE - QUICK START           ║
║                                                                      ║
║  Complete workflow for extracting and normalizing fitness pose      ║
║  keypoints with model comparison and frame normalization            ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    # Configuration
    DATASET_ROOT = "./qevd-fit-coach-data"
    OUTPUT_DIR = "./results"
    
    print(f"""
CONFIGURATION:
  Dataset root: {DATASET_ROOT}
  Output directory: {OUTPUT_DIR}
  
To use different paths, edit the DATASET_ROOT and OUTPUT_DIR variables
in this file.
    """)
    
    # Check dataset
    print("\n" + "="*70)
    print("CHECKING DATASET...")
    print("="*70)
    
    if not check_dataset_exists(DATASET_ROOT):
        print("\nSetup failed. Please download and organize the QEVD-FIT-COACH dataset.")
        sys.exit(1)
    
    print(f"✓ Dataset found at: {DATASET_ROOT}")
    
    # Initialize pipeline
    print("\n" + "="*70)
    print("INITIALIZING PIPELINE...")
    print("="*70)
    
    try:
        pipeline = ComprehensivePipeline(DATASET_ROOT, OUTPUT_DIR)
    except Exception as e:
        print(f"✗ Failed to initialize pipeline: {e}")
        sys.exit(1)
    
    print("✓ Pipeline initialized")
    
    # Run pipeline
    print("\n" + "="*70)
    print("RUNNING COMPLETE PIPELINE...")
    print("="*70)
    
    print("""
This will execute in 6 steps:

Step 1: Dataset Validation & Loading
   - Verify structure
   - Load metadata
   - List available videos

Step 2: Model Comparison
   - Benchmark MediaPipe BlazePose
   - Benchmark YOLOv11
   - Benchmark OpenPose (if available)
   - Compare on speed, accuracy, efficiency

Step 3: Model Selection
   - Select best model based on comparison

Step 4: Preprocessing Demonstration
   - Show normalization techniques
   - Compare Center&Scale vs Skeleton-based vs Perspective
   - Demonstrate imputation and FPS sync

Step 5: Dataset Preprocessing
   - Process videos with selected model
   - Apply normalization
   - Handle missing joints
   - Synchronize frame rates
   - Save normalized keypoints

Step 6: Report Generation
   - Summary statistics
   - Exercise breakdown
   - Processing metrics

Output saved to: {OUTPUT_DIR}
    """)
    
    # Ask for confirmation
    response = input("\nProceed with pipeline execution? (y/n): ").strip().lower()
    if response != 'y':
        print("Pipeline cancelled.")
        sys.exit(0)
    
    # Run pipeline with options
    print("\n" + "="*70)
    print("SELECT PROCESSING OPTIONS")
    print("="*70)
    
    # Exercise type
    exercise_type = input("\nFilter by exercise type? (press Enter for all): ").strip() or None
    
    # Max videos
    max_videos_str = input("Maximum videos to process (press Enter for all): ").strip() or None
    max_videos = int(max_videos_str) if max_videos_str else None
    
    # Run
    try:
        pipeline.run_complete_pipeline(
            exercise_type=exercise_type,
            max_videos=max_videos
        )
        
        print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║ ✓ PIPELINE COMPLETE                                                 ║
╚══════════════════════════════════════════════════════════════════════╝

Results saved to: {OUTPUT_DIR}

Generated files:
  ✓ processed_keypoints/     - Normalized pose keypoints (NPZ format)
  ✓ processing_report.json   - Detailed statistics and metadata
  ✓ normalization_visualization.png - Comparison of normalization methods

Next steps:
  1. Review processing_report.json for quality metrics
  2. Load NPZ files: np.load('exercise_001_keypoints.npz')
  3. Extract features for your coaching model:
     - Joint angles (knee, elbow, shoulder)
     - Velocities and accelerations
     - Temporal smoothness
     - Symmetry metrics
  4. Train fitness correction models on normalized keypoints
        """)
        
    except Exception as e:
        print(f"\n✗ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
