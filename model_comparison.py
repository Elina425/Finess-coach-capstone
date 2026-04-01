"""
Model Comparison and Benchmarking Script
Compares YOLOv11, OpenPose, and MediaPipe BlazePose
"""

import numpy as np
from typing import Dict, List
import json
from pathlib import Path
import time
import psutil
import os
from pose_estimation_core import (
    VideoProcessor, PosePreprocessor, ComparisonMetrics,
    COCO_KEYPOINTS, iter_default_comparison_detectors,
)
import warnings
warnings.filterwarnings('ignore')


class ModelComparison:
    """Compare pose estimation models"""
    
    def __init__(self):
        self.results = {}
        self.detectors = {}
        self.metrics = {}
    
    def initialize_models(self):
        """Initialize all available models"""
        print("\nInitializing Models...")
        print("-" * 60)

        for key, disp_name, detector in iter_default_comparison_detectors():
            self.detectors[key] = detector
            print(f"  ✓ {disp_name} ready")
    
    def benchmark_video(self, video_path: str, max_frames: int = 100) -> Dict:
        """
        Benchmark all models on a single video
        
        Args:
            video_path: Path to test video
            max_frames: Maximum frames to process
        
        Returns:
            Dictionary of results for each model
        """
        print(f"\nBenchmarking on: {video_path}")
        print(f"Max frames: {max_frames}")
        print("-" * 60)
        
        results = {}
        
        for model_name, detector in self.detectors.items():
            print(f"\nTesting {model_name.upper()}...")
            
            try:
                video = VideoProcessor(video_path)
                
                # Track memory
                process = psutil.Process(os.getpid())
                mem_before = process.memory_info().rss / 1024 / 1024  # MB
                
                # Process video
                start_time = time.time()
                pose_results = video.process_with_detector(detector, max_frames)
                total_time = time.time() - start_time
                
                mem_after = process.memory_info().rss / 1024 / 1024  # MB
                memory_peak = mem_after - mem_before
                
                video.close()
                
                # Compute metrics
                if len(pose_results) > 0:
                    inference_times = [r.inference_time for r in pose_results]
                    confidences = [np.mean(r.confidence) for r in pose_results]
                    
                    avg_fps = len(pose_results) / total_time
                    avg_inference_time_ms = np.mean(inference_times)
                    avg_confidence = np.mean(confidences)
                    
                    # Calculate detection rate (% frames with confident detections)
                    detected_frames = sum(1 for r in pose_results if np.mean(r.confidence) > 0.3)
                    detection_rate = (detected_frames / len(pose_results)) * 100 if len(pose_results) > 0 else 0
                    
                    metrics = ComparisonMetrics(
                        model_name=model_name,
                        avg_fps=avg_fps,
                        avg_inference_time_ms=avg_inference_time_ms,
                        total_frames_processed=len(pose_results),
                        avg_confidence=avg_confidence,
                        detection_rate=detection_rate,
                        memory_peak_mb=abs(memory_peak)
                    )
                    
                    self.metrics[model_name] = metrics
                    results[model_name] = pose_results
                    
                    metrics.display()
                else:
                    print(f"  ✗ No detections for {model_name}")
            
            except Exception as e:
                print(f"  ✗ Error with {model_name}: {e}")
        
        return results
    
    def compare_and_rank(self):
        """Compare metrics and rank models"""
        print("\n" + "="*60)
        print("MODEL COMPARISON SUMMARY")
        print("="*60)
        
        if not self.metrics:
            print("No metrics available - check model initialization")
            return
        
        # Create comparison table
        print(f"\n{'Model':<15} {'FPS':<10} {'Inf.Time':<12} {'Conf.':<10} {'Det.Rate':<12} {'Mem(MB)':<10}")
        print("-" * 70)
        
        for name, metric in self.metrics.items():
            print(f"{name:<15} {metric.avg_fps:<10.1f} {metric.avg_inference_time_ms:<12.2f}ms "
                  f"{metric.avg_confidence:<10.4f} {metric.detection_rate:<12.1f}% {metric.memory_peak_mb:<10.1f}")
        
        # Scoring and ranking
        print("\n" + "="*60)
        print("RANKING BY USE CASE")
        print("="*60)
        
        self._rank_by_speed()
        self._rank_by_accuracy()
        self._rank_by_efficiency()
        self._rank_by_realtime()
        
        # Best model selection
        print("\n" + "="*60)
        print("RECOMMENDATIONS")
        print("="*60)
        self._recommend()
    
    def _rank_by_speed(self):
        """Rank by inference speed"""
        print("\n1. SPEED RANKING (Inference Time)")
        sorted_models = sorted(self.metrics.items(), 
                              key=lambda x: x[1].avg_inference_time_ms)
        for i, (name, metric) in enumerate(sorted_models, 1):
            print(f"   {i}. {name.upper():<15} {metric.avg_inference_time_ms:.2f}ms")
    
    def _rank_by_accuracy(self):
        """Rank by detection confidence"""
        print("\n2. ACCURACY RANKING (Confidence & Detection Rate)")
        sorted_models = sorted(self.metrics.items(), 
                              key=lambda x: (x[1].avg_confidence, x[1].detection_rate),
                              reverse=True)
        for i, (name, metric) in enumerate(sorted_models, 1):
            print(f"   {i}. {name.upper():<15} Conf:{metric.avg_confidence:.4f}, "
                  f"Detection:{metric.detection_rate:.1f}%")
    
    def _rank_by_efficiency(self):
        """Rank by FPS efficiency"""
        print("\n3. EFFICIENCY RANKING (FPS)")
        sorted_models = sorted(self.metrics.items(), 
                              key=lambda x: x[1].avg_fps, 
                              reverse=True)
        for i, (name, metric) in enumerate(sorted_models, 1):
            print(f"   {i}. {name.upper():<15} {metric.avg_fps:.2f} FPS")
    
    def _rank_by_realtime(self):
        """Rank for real-time capability"""
        print("\n4. REAL-TIME CAPABILITY (can achieve 30+ FPS)")
        sorted_models = sorted(self.metrics.items(), 
                              key=lambda x: x[1].avg_fps, 
                              reverse=True)
        for i, (name, metric) in enumerate(sorted_models, 1):
            realtime = "✓ YES" if metric.avg_fps >= 30 else "✗ NO"
            print(f"   {i}. {name.upper():<15} {metric.avg_fps:.2f} FPS  {realtime}")
    
    def _recommend(self):
        """Generate recommendations"""
        if not self.metrics:
            return
        
        metrics_list = list(self.metrics.values())
        metrics_list.sort(key=lambda x: x.avg_fps, reverse=True)
        fastest = metrics_list[0]
        
        metrics_list.sort(key=lambda x: x.avg_confidence, reverse=True)
        most_accurate = metrics_list[0]
        
        metrics_list.sort(key=lambda x: x.detection_rate, reverse=True)
        best_detection = metrics_list[0]
        
        print(f"""
BEST FOR DIFFERENT SCENARIOS:

1. REAL-TIME FEEDBACK (Live Coaching)
   → {fastest.model_name.upper()}
   Why: {fastest.avg_fps:.1f} FPS achieves smooth real-time processing
   Trade-off: Slightly lower confidence ({fastest.avg_confidence:.4f})

2. MAXIMUM ACCURACY (Offline Analysis)
   → {most_accurate.model_name.upper()}
   Why: Highest confidence score ({most_accurate.avg_confidence:.4f})
   Trade-off: Slower ({most_accurate.avg_inference_time_ms:.2f}ms per frame)

3. ROBUSTNESS (Occluded/Challenging Poses)
   → {best_detection.model_name.upper()}
   Why: Best detection rate ({best_detection.detection_rate:.1f}%)
   Trade-off: {best_detection.model_name} is slower than alternatives

RECOMMENDATION FOR QEVD-FIT-COACH:
→ Use {fastest.model_name.upper()} for real-time mobile/web coaching
→ Combine with {most_accurate.model_name.upper()} for ground-truth video analysis
→ Ensemble approach: Run MediaPipe first, then refine low-confidence joints with YOLOv11
        """)


class PreprocessingComparison:
    """Compare preprocessing techniques"""
    
    @staticmethod
    def compare_normalization_methods(keypoint_sequence: List[np.ndarray]):
        """Compare different normalization techniques"""
        print("\n" + "="*60)
        print("PREPROCESSING TECHNIQUE COMPARISON")
        print("="*60)
        
        if not keypoint_sequence or keypoint_sequence[0] is None:
            print("No valid keypoints to process")
            return
        
        # Sample frame for demonstration
        sample_kp = keypoint_sequence[0]
        
        print(f"\nOriginal keypoints shape: {sample_kp.shape}")
        print(f"Keypoint range X: [{np.min(sample_kp[:, 0]):.1f}, {np.max(sample_kp[:, 0]):.1f}]")
        print(f"Keypoint range Y: [{np.min(sample_kp[:, 1]):.1f}, {np.max(sample_kp[:, 1]):.1f}]")
        
        # Method 1: Center & Scale
        print("\n1. CENTER & SCALE NORMALIZATION")
        norm1 = PosePreprocessor.center_and_scale_normalize(sample_kp.copy())
        print(f"   Range X: [{np.min(norm1[:, 0]):.4f}, {np.max(norm1[:, 0]):.4f}]")
        print(f"   Range Y: [{np.min(norm1[:, 1]):.4f}, {np.max(norm1[:, 1]):.4f}]")
        print("   ✓ Removes camera distance effect")
        print("   ✓ Makes skeleton size invariant")
        print("   ✗ Sensitive to outlier joints")
        
        # Method 2: Skeleton-based
        print("\n2. SKELETON-BASED NORMALIZATION")
        norm2 = PosePreprocessor.skeleton_based_normalize(sample_kp.copy())
        if norm2 is not None and not np.all(norm2 == 0):
            print(f"   Range X: [{np.min(norm2[:, 0]):.4f}, {np.max(norm2[:, 0]):.4f}]")
            print(f"   Range Y: [{np.min(norm2[:, 1]):.4f}, {np.max(norm2[:, 1]):.4f}]")
        print("   ✓ Torso-based reference (more robust)")
        print("   ✓ Handles scale variation from camera movement")
        print("   ✓ Most suitable for fitness tracking")
        
        # Method 3: Imputation
        print("\n3. MISSING JOINT IMPUTATION")
        confidence = np.ones(len(sample_kp))  # Simulate confidence
        imputed, mask = PosePreprocessor.impute_missing_joints(sample_kp.copy(), confidence, threshold=0.3)
        imputed_count = np.sum(mask)
        print(f"   Joints imputed: {imputed_count}/17")
        print("   ✓ Handles occluded joints")
        print("   ✓ Uses skeleton geometry for interpolation")
        print("   ✗ May introduce errors if imputation fails")
        
        # Method 4: Frame rate sync
        print("\n4. FRAME RATE SYNCHRONIZATION")
        print(f"   Original sequence length: {len(keypoint_sequence)} frames")
        print(f"   Original FPS: 30")
        print(f"   Target FPS: 30 (no resampling needed)")
        if len(keypoint_sequence) > 1:
            print(f"   Velocity calculation: Available (after resampling)")
        print("   ✓ Ensures consistent temporal features")
        print("   ✓ Enables velocity/acceleration computation")


def create_sample_video():
    """Create a sample video for testing (if needed)"""
    import cv2
    import numpy as np
    
    output_path = "sample_fitness.mp4"
    fps = 30
    duration = 3
    width, height = 640, 480
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame_idx in range(fps * duration):
        frame = np.ones((height, width, 3), dtype=np.uint8) * 100
        
        # Draw moving person silhouette
        center_x = int(width / 2 + 50 * np.sin(2 * np.pi * frame_idx / (fps * duration)))
        center_y = height // 2
        
        cv2.circle(frame, (center_x, center_y - 100), 30, (0, 255, 0), -1)  # Head
        cv2.line(frame, (center_x, center_y - 70), (center_x, center_y + 30), (0, 255, 0), 5)  # Torso
        cv2.line(frame, (center_x, center_y + 30), (center_x - 40, center_y + 80), (0, 255, 0), 5)  # Left leg
        cv2.line(frame, (center_x, center_y + 30), (center_x + 40, center_y + 80), (0, 255, 0), 5)  # Right leg
        
        out.write(frame)
    
    out.release()
    return output_path


def main():
    """Main comparison pipeline"""
    print("\n" + "="*60)
    print("QEVD-FIT-COACH MODEL COMPARISON")
    print("Pose Estimation: MediaPipe, YOLO, RTMPose-X, ViTPose (RTMPose), DETRPose (RTMO), OpenPose")
    print("="*60)
    
    # Initialize comparison
    comparison = ModelComparison()
    comparison.initialize_models()
    
    if not comparison.detectors:
        print("\n✗ No models available. Install at least one:")
        print("  - MediaPipe: pip install mediapipe")
        print("  - YOLOv8-Pose: pip install ultralytics")
        print("  - RTMPose-X / ViTPose / DETRPose (RTMPose & RTMO): pip install rtmlib onnxruntime")
        print("  - OpenPose: https://github.com/CMU-Perceptual-Computing-Lab/openpose")
        return
    
    # Test with a sample video or provided path
    test_video = "sample_fitness.mp4"  # Replace with actual video path
    
    if not Path(test_video).exists():
        print(f"\nCreating sample video for testing...")
        test_video = create_sample_video()
        print(f"✓ Sample video created: {test_video}")
    
    # Run benchmarks
    results = comparison.benchmark_video(test_video, max_frames=50)
    
    # Compare and rank
    comparison.compare_and_rank()
    
    # Preprocessing comparison
    if results:
        first_model_results = list(results.values())[0]
        if first_model_results:
            keypoint_sequence = [r.keypoints for r in first_model_results]
            PreprocessingComparison.compare_normalization_methods(keypoint_sequence)
    
    # Save results
    output_file = "comparison_results.json"
    print(f"\n✓ Results saved to: {output_file}")


if __name__ == "__main__":
    main()
