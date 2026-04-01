"""
QEVD-FIT-COACH Dataset Integration
Handles dataset structure, annotations, and preprocessing pipeline
"""

import json
import os
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from dataclasses import dataclass
from pose_estimation_core import (
    VideoProcessor,
    apply_keypoint_preprocessing_pipeline,
    COCO_KEYPOINTS,
)


@dataclass
class FitExerciseAnnotation:
    """Store exercise annotation metadata"""
    video_id: str
    exercise_type: str
    difficulty: str
    correctness_labels: Dict
    body_parts_focused: List[str]
    duration_seconds: float
    frame_rate: float
    annotations_json: Optional[Dict] = None
    
    def to_dict(self):
        return {
            'video_id': self.video_id,
            'exercise_type': self.exercise_type,
            'difficulty': self.difficulty,
            'correctness_labels': self.correctness_labels,
            'body_parts_focused': self.body_parts_focused,
            'duration_seconds': self.duration_seconds,
            'frame_rate': self.frame_rate
        }


class QEVDDatasetLoader:
    """Load and organize QEVD-FIT-COACH dataset"""
    
    def __init__(self, dataset_root: str):
        """
        Initialize dataset loader
        
        Expected structure:
        dataset_root/
        ├── videos/
        │   ├── exercise_001.mp4
        │   └── ...
        ├── annotations/
        │   ├── exercise_001.json
        │   └── ...
        └── metadata.csv
        """
        self.dataset_root = Path(dataset_root)
        self.videos_dir = self.dataset_root / "videos"
        self.annotations_dir = self.dataset_root / "annotations"
        self.metadata_file = self.dataset_root / "metadata.csv"
        
        self.videos = {}
        self.annotations = {}
        self.metadata = None
    
    def validate_structure(self) -> bool:
        """Validate dataset directory structure"""
        if not self.dataset_root.exists():
            print(f"✗ Dataset root not found: {self.dataset_root}")
            return False
        
        if not self.videos_dir.exists():
            print(f"✗ Videos directory not found: {self.videos_dir}")
            return False
        
        if not self.annotations_dir.exists():
            print(f"⚠ Annotations directory not found: {self.annotations_dir}")
            print("  Creating annotations directory...")
            self.annotations_dir.mkdir(parents=True, exist_ok=True)
        
        return True
    
    def load_metadata(self) -> pd.DataFrame:
        """Load metadata CSV"""
        if not self.metadata_file.exists():
            print(f"⚠ Metadata file not found: {self.metadata_file}")
            return None
        
        try:
            self.metadata = pd.read_csv(self.metadata_file)
            print(f"✓ Loaded metadata: {len(self.metadata)} entries")
            return self.metadata
        except Exception as e:
            print(f"✗ Error loading metadata: {e}")
            return None
    
    def load_annotation(self, video_id: str) -> Optional[Dict]:
        """Load annotation for specific video"""
        annotation_file = self.annotations_dir / f"{video_id}.json"
        
        if not annotation_file.exists():
            return None
        
        try:
            with open(annotation_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"✗ Error loading annotation {video_id}: {e}")
            return None
    
    def get_exercise_info(self, video_id: str) -> Optional[FitExerciseAnnotation]:
        """
        Get exercise information from metadata and annotations
        """
        # Get from metadata
        if self.metadata is None:
            self.load_metadata()
        
        if self.metadata is None:
            return None
        
        # Find entry in metadata
        video_meta = self.metadata[self.metadata['video_id'] == video_id]
        if video_meta.empty:
            return None
        
        row = video_meta.iloc[0]
        
        # Get annotation
        anno = self.load_annotation(video_id)
        
        # Parse correctness labels
        correctness_labels = {}
        if anno and 'correctness' in anno:
            correctness_labels = anno['correctness']
        
        # Extract body parts
        body_parts = []
        if anno and 'body_parts' in anno:
            body_parts = anno['body_parts']
        
        return FitExerciseAnnotation(
            video_id=video_id,
            exercise_type=row.get('exercise_type', ''),
            difficulty=row.get('difficulty', 'unknown'),
            correctness_labels=correctness_labels,
            body_parts_focused=body_parts,
            duration_seconds=float(row.get('duration', 0)),
            frame_rate=float(row.get('fps', 30)),
            annotations_json=anno
        )
    
    def list_videos(self) -> List[str]:
        """List all video IDs in dataset"""
        if not self.videos_dir.exists():
            return []
        
        # Search recursively for videos in subdirectories (long_range, short_clips)
        videos = [f.stem for f in self.videos_dir.rglob("*.mp4")]
        return sorted(videos)

    def list_short_clip_videos(self) -> List[str]:
        """List stems of readable ``*.mp4`` files under ``videos/short_clips/`` only (for fine_grained / BiLSTM)."""
        d = self.videos_dir / "short_clips"
        if not d.is_dir():
            return []
        out: List[str] = []
        for f in sorted(d.glob("*.mp4")):
            try:
                if f.resolve().is_file():
                    out.append(f.stem)
            except OSError:
                continue
        return out
    
    def get_video_path(self, video_id: str) -> Optional[Path]:
        """Get full path to a video by its stem"""
        if not self.videos_dir.exists():
            return None
        
        # Search for video file in subdirectories
        for video_file in self.videos_dir.rglob(f"{video_id}.mp4"):
            return video_file
        
        return None
    
    def get_videos_by_exercise(self, exercise_type: str) -> List[str]:
        """Get all videos for specific exercise"""
        if self.metadata is None:
            self.load_metadata()
        
        if self.metadata is None:
            return []
        
        filtered = self.metadata[
            self.metadata['exercise_type'].str.lower() == exercise_type.lower()
        ]
        return filtered['video_id'].tolist()
    
    def get_videos_by_difficulty(self, difficulty: str) -> List[str]:
        """Get videos by difficulty level"""
        if self.metadata is None:
            self.load_metadata()
        
        if self.metadata is None:
            return []
        
        filtered = self.metadata[
            self.metadata['difficulty'].str.lower() == difficulty.lower()
        ]
        return filtered['video_id'].tolist()


class DatasetPreprocessor:
    """Preprocess dataset for pose estimation"""
    
    def __init__(self, dataset_loader: QEVDDatasetLoader):
        self.loader = dataset_loader
        self.processed_data = {}
        self.preprocessing_log = []

    def _postprocess_extracted_sequence(
        self,
        video_id: str,
        video_processor: VideoProcessor,
        pose_results: List,
        exercise_info: Optional[FitExerciseAnnotation],
        preprocessing_techniques: List[str],
        target_fps: float,
    ) -> Dict:
        """Spatial impute → torso normalization → temporal impute → FPS resample."""
        print(f"  ✓ Extracted {len(pose_results)} frames")

        measured_fps = (
            float(video_processor.fps)
            if video_processor.fps and video_processor.fps > 1e-6
            else 30.0
        )
        meta_fps = (
            float(exercise_info.frame_rate)
            if exercise_info and exercise_info.frame_rate
            else None
        )
        source_fps = measured_fps

        keypoint_sequence = [r.keypoints for r in pose_results]
        confidence_sequence = [r.confidence for r in pose_results]

        if "imputation" in preprocessing_techniques:
            if "laplacian_spatial" in preprocessing_techniques:
                print(
                    "  Applying per-frame spatial imputation (graph Laplacian harmonic, arXiv:2204.10312)..."
                )
            else:
                print("  Applying per-frame spatial imputation (low-confidence joints)...")
        if "normalization" in preprocessing_techniques:
            print("  Applying skeleton-based normalization (torso scale, hip-centered)...")
        if "bone_proportion" in preprocessing_techniques:
            print(
                "  Applying bone proportion scaling (BioPose-style limb/torso ratios, arXiv:2501.07800)..."
            )
        if "temporal_impute" in preprocessing_techniques or "imputation" in preprocessing_techniques:
            print("  Temporal imputation (linear along time for low-confidence joints)...")
        if "fps_sync" in preprocessing_techniques:
            tf = float(target_fps)
            if abs(source_fps - tf) >= 1e-3:
                print(
                    f"  Resampling keypoints: {source_fps:.2f} FPS (video) → {tf:.0f} FPS..."
                )
        if "dwt" in preprocessing_techniques:
            print("  Applying DWT (Discrete Wavelet Transform) normalization...")

        processed = apply_keypoint_preprocessing_pipeline(
            keypoint_sequence,
            confidence_sequence,
            preprocessing_techniques=preprocessing_techniques,
            target_fps=float(target_fps),
            source_fps=source_fps,
            original_frames=len(keypoint_sequence),
        )
        processed["video_id"] = video_id
        processed["exercise_info"] = exercise_info.to_dict() if exercise_info else None
        processed["metadata_fps"] = meta_fps

        if "imputation" in preprocessing_techniques and "imputation_rate" in processed:
            print(
                f"    Avg joints imputed per frame: {processed['imputation_rate']:.1f}% of joints"
            )
        if "normalization" in preprocessing_techniques:
            raw_range = float(np.max(np.abs(keypoint_sequence))) if keypoint_sequence else 0.0
            fk = processed["final_keypoints"]
            norm_range = (
                float(np.max(np.abs(np.stack([k for k in fk if k is not None], axis=0))))
                if fk
                else 0.0
            )
            print(
                f"    Pixel range before: {raw_range:.1f} | after (norm coords): {norm_range:.4f}"
            )
        if "fps_sync" in preprocessing_techniques:
            ta = processed.get("techniques_applied", {}).get("fps_sync", "")
            if ta and not str(ta).startswith("no_change") and "failed" not in str(ta):
                print(f"    Resampled to {len(processed['final_keypoints'])} frames")
            elif "failed" in str(ta):
                print(f"    ⚠ FPS sync failed: {ta}")
        if "dwt" in preprocessing_techniques:
            dwt_s = processed.get("techniques_applied", {}).get("dwt", "")
            if dwt_s and not str(dwt_s).startswith("failed"):
                print("    DWT decomposition applied (Daubechies-4 wavelet)")
            elif str(dwt_s).startswith("failed"):
                print(f"    ⚠ DWT normalization failed: {dwt_s}")
                print("    (Install PyWavelets: pip install PyWavelets)")

        video_processor.close()

        print("  ✓ Preprocessing complete\n")
        return processed

    def preprocess_video(
        self,
        video_id: str,
        detector,
        preprocessing_techniques: List[str] = None,
        target_fps: float = 30.0,
    ) -> Dict:
        """
        Preprocess single video with selected techniques
        
        Args:
            video_id: Video identifier
            detector: Pose estimation detector
            preprocessing_techniques: List of techniques to apply
                ['normalization', 'imputation', 'fps_sync']
        
        Returns:
            Dictionary with processed keypoints and metadata
        """
        if preprocessing_techniques is None:
            preprocessing_techniques = ['normalization', 'imputation', 'fps_sync']
        
        video_path = self.loader.get_video_path(video_id)
        
        if not video_path:
            print(f"✗ Video not found: {video_id}")
            return None
        
        # Get annotation info
        exercise_info = self.loader.get_exercise_info(video_id)
        
        print(f"\nProcessing {video_id}...")
        if exercise_info:
            print(f"  Exercise: {exercise_info.exercise_type}")
            print(f"  Difficulty: {exercise_info.difficulty}")
            print(f"  Body parts: {', '.join(exercise_info.body_parts_focused)}")
        
        # Extract poses
        print(f"  Extracting poses...")
        video_processor = VideoProcessor(str(video_path))
        pose_results = video_processor.process_with_detector(detector)

        if not pose_results:
            print("  ✗ No poses detected")
            video_processor.close()
            return None

        return self._postprocess_extracted_sequence(
            video_id,
            video_processor,
            pose_results,
            exercise_info,
            preprocessing_techniques,
            target_fps,
        )

    def preprocess_video_file(
        self,
        video_path: str,
        detector,
        video_id: Optional[str] = None,
        preprocessing_techniques: Optional[List[str]] = None,
        target_fps: float = 30.0,
    ) -> Optional[Dict]:
        """Same as preprocess_video but with an explicit filesystem path (any .mp4)."""
        if preprocessing_techniques is None:
            preprocessing_techniques = ["normalization", "imputation", "fps_sync"]
        p = Path(video_path)
        if not p.is_file():
            print(f"✗ Not a file: {video_path}")
            return None
        vid = video_id or p.stem
        exercise_info = self.loader.get_exercise_info(vid) if self.loader else None
        print(f"\nProcessing file {p.name} (id={vid})...")
        print("  Extracting poses (MediaPipe / given detector)...")
        video_processor = VideoProcessor(str(p))
        pose_results = video_processor.process_with_detector(detector)
        if not pose_results:
            print("  ✗ No poses detected")
            video_processor.close()
            return None
        return self._postprocess_extracted_sequence(
            vid,
            video_processor,
            pose_results,
            exercise_info,
            preprocessing_techniques,
            target_fps,
        )
    
    def preprocess_batch(
        self,
        video_ids: List[str],
        detector,
        preprocessing_techniques: List[str] = None,
        target_fps: float = 30.0,
    ) -> Dict[str, Dict]:
        """Preprocess batch of videos"""
        results = {}

        print(f"\n{'='*60}")
        print(f"Batch Preprocessing: {len(video_ids)} videos")
        print(f"{'='*60}")

        for i, video_id in enumerate(video_ids, 1):
            print(f"\n[{i}/{len(video_ids)}]")
            try:
                result = self.preprocess_video(
                    video_id,
                    detector,
                    preprocessing_techniques=preprocessing_techniques,
                    target_fps=target_fps,
                )
                if result:
                    results[video_id] = result
            except Exception as e:
                print(f"✗ Error processing {video_id}: {e}")
        
        print(f"\n{'='*60}")
        print(f"Batch complete: {len(results)}/{len(video_ids)} successful\n")
        
        return results
    
    def save_processed_data(self, data: Dict, output_dir: str):
        """Save processed keypoints and metadata"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving processed data to: {output_path}")
        
        for video_id, processed in data.items():
            # Save keypoints as NPZ
            keypoints_file = output_path / f"{video_id}_keypoints.npz"
            keypoints_array = np.array(processed["final_keypoints"])
            save_kw = {
                "keypoints": keypoints_array,
                "frames": processed["original_frames"],
                "original_frames": processed["original_frames"],
                "techniques_json": json.dumps(processed["techniques_applied"]),
                "source_fps": float(processed.get("source_fps", 0.0) or 0.0),
                "target_fps": float(processed.get("target_fps", 30)),
            }
            if processed.get("confidence_sequence") is not None:
                save_kw["confidence"] = np.array(processed["confidence_sequence"])
            np.savez_compressed(keypoints_file, **save_kw)
            
            # Save metadata as JSON
            metadata_file = output_path / f"{video_id}_metadata.json"
            metadata = {
                "video_id": video_id,
                "exercise_info": processed["exercise_info"],
                "techniques_applied": processed["techniques_applied"],
                "source_fps": processed.get("source_fps"),
                "target_fps": processed.get("target_fps"),
                "statistics": {
                    k: float(v) if isinstance(v, np.floating) else v
                    for k, v in processed.get("statistics", {}).items()
                },
                "imputation_rate": float(processed.get("imputation_rate", 0))
                if isinstance(processed.get("imputation_rate"), np.floating)
                else processed.get("imputation_rate", None),
            }
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        # Save summary
        summary_file = output_path / "processing_summary.json"
        summary = {
            "total_videos": len(data),
            "processed_successfully": sum(1 for d in data.values() if d is not None),
            "exercises": list(
                {
                    d["exercise_info"]["exercise_type"]
                    for d in data.values()
                    if d.get("exercise_info") and d["exercise_info"].get("exercise_type")
                }
            ),
        }
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✓ Saved {len(data)} processed videos")


def demonstrate_workflow():
    """Demonstrate complete workflow"""
    print("\n" + "="*70)
    print("QEVD-FIT-COACH Dataset Integration Workflow")
    print("="*70)
    
    # Example configuration
    dataset_root = "./qevd-fit-coach-data"
    output_dir = "./processed_poses"
    
    print(f"""
WORKFLOW STEPS:

1. INITIALIZE DATASET LOADER
   - Point to QEVD dataset root directory
   - Validate directory structure
   - Load metadata and annotations
   
   Example structure:
   {dataset_root}/
   ├── videos/ (downloaded videos)
   ├── annotations/ (JSON annotation files)
   └── metadata.csv (exercise metadata)

2. SELECT POSE ESTIMATION MODEL
   Based on comparison results:
   - MediaPipe: Fast real-time (30+ FPS)
   - YOLOv11: High accuracy (20-40ms per frame)
   - OpenPose: Most accurate (50-100ms per frame)

3. CONFIGURE PREPROCESSING
   Techniques to apply:
   - Skeleton-based normalization (recommended for fitness)
   - Missing joint imputation (handles occlusion)
   - FPS synchronization (consistent temporal data)

4. PROCESS VIDEOS
   - Extract keypoints for each frame
   - Apply selected preprocessing
   - Compute aggregate features
   - Save normalized data

5. OUTPUT FORMAT
   Both keypoints and confidence scores for each frame:
   - NPZ files: Compressed keypoints (17 x 2)
   - JSON files: Exercise metadata and statistics
   - Summary file: Processing report

EXAMPLE CODE:

```python
from qevd_dataset_integration import QEVDDatasetLoader, DatasetPreprocessor
from model_comparison import ModelComparison

# 1. Initialize dataset
loader = QEVDDatasetLoader("{dataset_root}")
if not loader.validate_structure():
    print("Invalid dataset structure")
    exit(1)

loader.load_metadata()

# 2. Initialize model (from comparison results)
from pose_estimation_core import MediaPipeDetector  # or YOLOv11Detector
detector = MediaPipeDetector()

# 3. Preprocess
preprocessor = DatasetPreprocessor(loader)

# Get videos for specific exercise
squat_videos = loader.get_videos_by_exercise("squat")

# Preprocess with recommended techniques
results = preprocessor.preprocess_batch(
    squat_videos,
    detector,
    preprocessing_techniques=['normalization', 'imputation', 'fps_sync']
)

# 4. Save results
preprocessor.save_processed_data(results, "{output_dir}")
```
    """)


if __name__ == "__main__":
    demonstrate_workflow()
