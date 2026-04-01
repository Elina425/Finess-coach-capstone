"""
Complete QEVD-FIT-COACH Pose-Frame Normalization Pipeline
Ready-to-run implementation with annotation integration
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from pose_estimation_core import (
    MediaPipeDetector,
    YOLODetector,
    OpenPoseDetector,
    RTMPoseXLDetector,
    ViTPoseDetector,
    DETRPoseDetector,
    VideoProcessor,
    PosePreprocessor,
    COCO_KEYPOINTS,
)
from model_comparison import ModelComparison, PreprocessingComparison
from qevd_dataset_integration import QEVDDatasetLoader, DatasetPreprocessor


class ComprehensivePipeline:
    """End-to-end pipeline for QEVD dataset with pose estimation"""
    
    def __init__(self, dataset_root: str, output_dir: str):
        self.dataset_root = dataset_root
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.loader = None
        self.detector = None
        self.preprocessor = None
        self.comparison_results = {}
    
    def step_1_validate_dataset(self):
        """Step 1: Validate and load dataset"""
        print("\n" + "="*70)
        print("STEP 1: DATASET VALIDATION & LOADING")
        print("="*70)
        
        self.loader = QEVDDatasetLoader(self.dataset_root)
        
        if not self.loader.validate_structure():
            print("\n✗ Dataset structure invalid")
            return False
        
        print("✓ Dataset structure valid")
        
        # Load metadata
        metadata = self.loader.load_metadata()
        if metadata is not None:
            print(f"✓ Loaded metadata for {len(metadata)} videos")
            print(f"\nExercise types: {metadata['exercise_type'].unique()}")
            print(f"Difficulty levels: {metadata['difficulty'].unique()}")
        
        # List available videos
        videos = self.loader.list_videos()
        print(f"✓ Found {len(videos)} videos")
        
        return True
    
    def step_2_compare_models(self, test_video: str = None, max_frames: int = 50):
        """Step 2: Compare pose estimation models"""
        print("\n" + "="*70)
        print("STEP 2: MODEL COMPARISON & SELECTION")
        print("="*70)
        
        try:
            comparison = ModelComparison()
            comparison.initialize_models()
            
            if not comparison.detectors:
                print("✗ No detection models available")
                return None
            
            # Find test video
            if test_video is None:
                videos = self.loader.list_videos()
                if not videos:
                    print("✗ No videos found for testing")
                    return None
                test_video_path = self.loader.get_video_path(videos[0])
                if not test_video_path:
                    print(f"✗ Test video not found: {videos[0]}")
                    return None
                test_video = test_video_path
            else:
                test_video = Path(test_video)
            
            if not test_video.exists():
                print(f"✗ Test video not found: {test_video}")
                return None
            
            print(f"\nTesting on: {test_video.name}")
            results = comparison.benchmark_video(str(test_video), max_frames=max_frames)
            
            # Rank and compare
            comparison.compare_and_rank()
            
            self.comparison_results = comparison.metrics
            
            # Return best model
            if comparison.metrics:
                best_model = sorted(comparison.metrics.items(),
                                  key=lambda x: x[1].avg_fps,
                                  reverse=True)[0][0]
                print(f"\n✓ Recommended model for real-time: {best_model.upper()}")
                return best_model
            
            return None
            
        except Exception as e:
            print(f"⚠️  Model comparison failed: {e}")
            print("Continuing with default model (MediaPipe)...")
            return 'mediapipe'
        
        # Select best model for real-time
        if comparison.metrics:
            best_model = sorted(comparison.metrics.items(),
                              key=lambda x: x[1].avg_fps,
                              reverse=True)[0][0]
            print(f"\n✓ Recommended model for real-time: {best_model.upper()}")
            return best_model
        
        return None
    
    def step_3_select_model(self, model_name: str = 'mediapipe'):
        """Step 3: Initialize selected model"""
        print("\n" + "="*70)
        print("STEP 3: MODEL INITIALIZATION")
        print("="*70)
        
        print(f"Initializing {model_name.upper()}...")
        
        mn = model_name.lower()
        if mn == 'mediapipe':
            self.detector = MediaPipeDetector()
        elif mn == 'yolo':
            self.detector = YOLODetector()
        elif mn == 'openpose':
            self.detector = OpenPoseDetector()
        elif mn in ("rtmpose_x", "rtmpose-xl", "rtmpose_xl"):
            self.detector = RTMPoseXLDetector()
        elif mn in ('vitpose', 'vitpose_detector'):
            self.detector = ViTPoseDetector()
        elif mn in ('detrpose', 'detrpose_detector', 'deterpose'):
            self.detector = DETRPoseDetector()
        else:
            print(f"✗ Unknown model: {model_name}")
            return False
        
        if not getattr(self.detector, 'available', True):
            print(f"✗ {model_name} not available")
            return False
        
        print(f"✓ {model_name.upper()} ready")
        return True
    
    def step_4_demonstrate_preprocessing(self):
        """Step 4: Demonstrate preprocessing techniques"""
        print("\n" + "="*70)
        print("STEP 4: PREPROCESSING TECHNIQUE COMPARISON")
        print("="*70)
        
        # Use a sample video
        videos = self.loader.list_videos()
        if not videos:
            print("No videos available")
            return
        
        test_video_path = self.loader.get_video_path(videos[0])
        if not test_video_path:
            print(f"Video not found: {videos[0]}")
            return
        
        print(f"Processing: {videos[0]}")
        video_processor = VideoProcessor(str(test_video_path))
        pose_results = video_processor.process_with_detector(self.detector, max_frames=20)
        video_processor.close()
        
        if not pose_results:
            print("No poses detected")
            return
        
        keypoint_sequence = [r.keypoints for r in pose_results]
        PreprocessingComparison.compare_normalization_methods(keypoint_sequence)
        
        # Visualize normalization effects
        self._visualize_normalization(keypoint_sequence)
    
    def step_5_preprocess_dataset(self, exercise_type: str = None,
                                  difficulty: str = None,
                                  max_videos: int = None):
        """Step 5: Preprocess selected videos"""
        print("\n" + "="*70)
        print("STEP 5: DATASET PREPROCESSING")
        print("="*70)
        
        # Select videos
        if exercise_type:
            video_ids = self.loader.get_videos_by_exercise(exercise_type)
            print(f"Found {len(video_ids)} videos for exercise: {exercise_type}")
        elif difficulty:
            video_ids = self.loader.get_videos_by_difficulty(difficulty)
            print(f"Found {len(video_ids)} videos with difficulty: {difficulty}")
        else:
            video_ids = self.loader.list_videos()
            print(f"Processing all {len(video_ids)} videos")
        
        if max_videos:
            video_ids = video_ids[:max_videos]
        
        # Initialize preprocessor
        self.preprocessor = DatasetPreprocessor(self.loader)
        
        # Preprocess with recommended techniques
        # Available normalization: 'skeleton-based' (default), 'center-scale', 'dwt'
        results = self.preprocessor.preprocess_batch(
            video_ids,
            self.detector,
            preprocessing_techniques=[
                'normalization',      # Skeleton-based (recommended for fitness)
                'imputation',         # Handle missing joints
                'fps_sync'           # Synchronize to 30 FPS
                # Optional: Add 'dwt' for DWT normalization (frequency-based)
            ]
        )
        
        # Save results
        processed_dir = self.output_dir / "processed_keypoints"
        self.preprocessor.save_processed_data(results, str(processed_dir))
        
        return results
    
    def step_6_generate_report(self, results: Dict):
        """Step 6: Generate comprehensive report"""
        print("\n" + "="*70)
        print("STEP 6: PROCESSING REPORT")
        print("="*70)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'dataset_root': self.dataset_root,
            'model_used': self.detector.__class__.__name__ if self.detector else 'None',
            'videos_processed': len(results),
            'preprocessing_techniques': [
                'skeleton-based-normalization (torso-based scale)',
                'confidence-based-imputation (skeleton geometry)',
                'fps-synchronization-to-30fps',
                'optional-dwt-normalization (frequency decomposition)'
            ],
            'statistics': {}
        }
        
        # Aggregate statistics (use Python floats — numpy float32 is not JSON-serializable)
        total_frames = 0
        total_confidence = 0.0
        exercise_stats = {}
        
        for video_id, data in results.items():
            if not data:
                continue
            total_frames += int(data.get('original_frames', 0) or 0)
            stats = data.get('statistics') or {}
            mc = stats.get('mean_confidence', 0)
            total_confidence += float(mc) if mc is not None else 0.0

            ex_info = data.get('exercise_info')
            if isinstance(ex_info, dict):
                exercise = ex_info.get('exercise_type', 'unknown')
            else:
                exercise = 'unknown'
            if exercise not in exercise_stats:
                exercise_stats[exercise] = {'count': 0, 'total_frames': 0}
            exercise_stats[exercise]['count'] += 1
            exercise_stats[exercise]['total_frames'] += data.get('original_frames', 0)
        
        report['statistics'] = {
            'total_frames_processed': int(total_frames),
            'avg_confidence': float(total_confidence / len(results)) if results else 0.0,
            'exercise_breakdown': exercise_stats
        }
        
        # Save report
        report_path = self.output_dir / "processing_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n{'PROCESSING SUMMARY'}")
        print(f"{'-'*70}")
        print(f"Model: {report['model_used']}")
        print(f"Videos processed: {report['videos_processed']}")
        print(f"Total frames: {report['statistics']['total_frames_processed']}")
        print(f"Average confidence: {report['statistics']['avg_confidence']:.4f}")
        print(f"\nExercise breakdown:")
        for exercise, stats in report['statistics']['exercise_breakdown'].items():
            print(f"  {exercise}: {stats['count']} videos, {stats['total_frames']} total frames")
        
        print(f"\n✓ Report saved to: {report_path}")
        
        return report
    
    def _visualize_normalization(self, keypoint_sequence: List[np.ndarray]):
        """Create visualization of normalization techniques"""
        if not keypoint_sequence or keypoint_sequence[0] is None:
            return
        
        sample_kp = keypoint_sequence[0]
        
        # Apply normalization techniques
        norm_center_scale = PosePreprocessor.center_and_scale_normalize(sample_kp.copy())
        norm_skeleton = PosePreprocessor.skeleton_based_normalize(sample_kp.copy())
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        # Original
        axes[0].scatter(sample_kp[:, 0], sample_kp[:, 1], s=50, alpha=0.7, c='blue')
        axes[0].set_title('Original Keypoints', fontweight='bold')
        axes[0].set_xlabel('X (pixels)')
        axes[0].set_ylabel('Y (pixels)')
        axes[0].grid(True, alpha=0.3)
        
        # Center & Scale
        if norm_center_scale is not None:
            axes[1].scatter(norm_center_scale[:, 0], norm_center_scale[:, 1],
                          s=50, alpha=0.7, c='green')
            axes[1].set_title('Center & Scale Normalized', fontweight='bold')
            axes[1].set_xlabel('X (normalized)')
            axes[1].set_ylabel('Y (normalized)')
            axes[1].grid(True, alpha=0.3)
        
        # Skeleton-based
        if norm_skeleton is not None:
            axes[2].scatter(norm_skeleton[:, 0], norm_skeleton[:, 1],
                          s=50, alpha=0.7, c='orange')
            axes[2].set_title('Skeleton-based Normalized', fontweight='bold')
            axes[2].set_xlabel('X (normalized)')
            axes[2].set_ylabel('Y (normalized)')
            axes[2].grid(True, alpha=0.3)
        
        # DWT normalization
        try:
            norm_dwt_seq = PosePreprocessor.dwt_normalize(keypoint_sequence[:min(50, len(keypoint_sequence))])
            if norm_dwt_seq and norm_dwt_seq[0] is not None:
                norm_dwt = norm_dwt_seq[0]
                axes[3].scatter(norm_dwt[:, 0], norm_dwt[:, 1],
                              s=50, alpha=0.7, c='red')
                axes[3].set_title('DWT Normalized (Frequency-based)', fontweight='bold')
                axes[3].set_xlabel('X (frequency-normalized)')
                axes[3].set_ylabel('Y (frequency-normalized)')
                axes[3].grid(True, alpha=0.3)
        except Exception as e:
            axes[3].text(0.5, 0.5, f'DWT unavailable\n({str(e)[:40]}...)',
                        ha='center', va='center', transform=axes[3].transAxes)
            axes[3].set_title('DWT Normalization')
        
        # Add joint labels
        for idx, name in enumerate(COCO_KEYPOINTS.NAMES[:5]):  # Show first 5 for clarity
            axes[0].annotate(name, sample_kp[idx], fontsize=8)
        
        plt.tight_layout()
        viz_path = self.output_dir / "normalization_visualization.png"
        plt.savefig(viz_path, dpi=150)
        print(f"\n✓ Normalization techniques visualization saved")
        print(f"  Techniques demonstrated: Center&Scale, Skeleton-based, DWT")
        plt.close()
    
    def run_complete_pipeline(self, exercise_type: str = None, max_videos: int = 5):
        """Run complete pipeline end-to-end"""
        print("\n" + "="*70)
        print("QEVD-FIT-COACH COMPLETE PIPELINE")
        print("="*70)
        
        # Step 1: Validate
        if not self.step_1_validate_dataset():
            return
        
        # Step 2: Compare models (optional, but recommended)
        best_model = self.step_2_compare_models(max_frames=30)
        model_to_use = best_model if best_model else 'mediapipe'
        
        # Step 3: Initialize model
        if not self.step_3_select_model(model_to_use):
            return
        
        # Step 4: Demonstrate preprocessing
        self.step_4_demonstrate_preprocessing()
        
        # Step 5: Process dataset
        results = self.step_5_preprocess_dataset(
            exercise_type=exercise_type,
            max_videos=max_videos
        )
        
        # Step 6: Generate report
        if results:
            self.step_6_generate_report(results)
        
        print("\n" + "="*70)
        print("✓ PIPELINE COMPLETE")
        print("="*70)
        print(f"\nProcessed data saved to: {self.output_dir}")


def main():
    """Example usage"""
    print("""
COMPLETE QEVD-FIT-COACH POSE ESTIMATION PIPELINE
================================================

This comprehensive pipeline:
1. ✓ Validates QEVD dataset structure
2. ✓ Compares pose models (MediaPipe, YOLO, RTMPose-X, ViTPose/RTMPose, DETRPose/RTMO, OpenPose)
3. ✓ Selects best model based on comparison metrics
4. ✓ Demonstrates preprocessing techniques:
   - Skeleton-based normalization (recommended for fitness)
   - Missing joint imputation (handles occlusion)
   - FPS synchronization (consistent temporal data)
5. ✓ Processes selected videos with annotations integration
6. ✓ Generates comprehensive report with statistics

USAGE:

# Initialize pipeline
pipeline = ComprehensivePipeline(
    dataset_root="./qevd-fit-coach-data",
    output_dir="./results"
)

# Run complete pipeline
pipeline.run_complete_pipeline(
    exercise_type="squat",  # Optional filter
    max_videos=10           # Limit for testing
)

Output structure:
./results/
├── processed_keypoints/   (NPZ files with normalized keypoints)
├── processing_report.json (Detailed statistics)
├── normalization_visualization.png
└── model_comparison_results.json (from Step 2)
    """)


if __name__ == "__main__":
    main()
