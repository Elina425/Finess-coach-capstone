"""
Pose Feature Extractor using MediaPipe

Extracts pose landmarks and computes joint angles from video frames.
Used for preprocessing videos into feature sequences for xLSTM training.

Features extracted:
- 33 MediaPipe pose landmarks (x, y, z coordinates)
- 13 joint angles (biomechanically relevant for exercise analysis)
- Optional: Normalized coordinates and velocity features

Output format:
- NPZ file with 'features' array: (num_frames, 13) for angles
- NPZ file with 'landmarks' array: (num_frames, 33, 3) for coordinates
"""

import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import json


class MediaPipePoseExtractor:
    """
    Extract pose landmarks and joint angles using MediaPipe Pose.

    MediaPipe Pose detects 33 landmarks on the human body:
    - Face: 0-10 (nose, eyes, ears, mouth)
    - Upper body: 11-22 (shoulders, elbows, wrists, hands)
    - Lower body: 23-32 (hips, knees, ankles, feet)

    We compute 13 joint angles relevant for fitness exercises.
    """

    def __init__(self,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 model_complexity: int = 1,
                 smooth_landmarks: bool = True):
        """
        Initialize MediaPipe Pose detector.

        Args:
            min_detection_confidence: Minimum confidence for detection
            min_tracking_confidence: Minimum confidence for tracking
            model_complexity: 0 (lite), 1 (full), 2 (heavy)
            smooth_landmarks: Temporal smoothing between frames
        """
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            smooth_landmarks=smooth_landmarks
        )

        # Landmark indices for angle calculations
        # See: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
        self.landmark_pairs = {
            'left_shoulder': (11, 12),
            'right_shoulder': (13, 14),
            'left_elbow': (13, 14),
            'right_elbow': (15, 16),
            'left_wrist': (15, 16),
            'right_wrist': (17, 18),
            'left_hip': (23, 24),
            'right_hip': (25, 26),
            'left_knee': (25, 26),
            'right_knee': (27, 28),
            'left_ankle': (27, 28),
            'right_ankle': (29, 30),
        }

    def extract_landmarks(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract 33 pose landmarks from a single frame.

        Args:
            frame: BGR image (H, W, 3)

        Returns:
            landmarks: (33, 3) array of (x, y, z) or None if no pose detected
        """
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect pose
        results = self.pose.process(rgb_frame)

        if results.pose_landmarks is None:
            return None

        # Extract landmarks as (33, 3) array
        landmarks = []
        for landmark in results.pose_landmarks.landmark:
            landmarks.append([landmark.x, landmark.y, landmark.z])

        return np.array(landmarks, dtype=np.float32)

    def compute_angle(self,
                      a: np.ndarray,
                      b: np.ndarray,
                      c: np.ndarray) -> float:
        """
        Compute angle at point b given three points a, b, c.

        Args:
            a: First point (e.g., shoulder)
            b: Vertex point (e.g., elbow)
            c: End point (e.g., wrist)

        Returns:
            Angle in degrees [0, 180]
        """
        # Convert to numpy arrays
        a = np.array(a, dtype=np.float32)
        b = np.array(b, dtype=np.float32)
        c = np.array(c, dtype=np.float32)

        # Vectors
        ba = a - b
        bc = c - b

        # Normalize
        ba_norm = ba / (np.linalg.norm(ba) + 1e-8)
        bc_norm = bc / (np.linalg.norm(bc) + 1e-8)

        # Dot product and angle
        cos_angle = np.dot(ba_norm, bc_norm)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Numerical stability
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)

        return angle_deg

    def compute_joint_angles(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Compute 13 biomechanically relevant joint angles from landmarks.

        Angles computed:
        1. Left shoulder angle (neck-shoulder-elbow)
        2. Right shoulder angle (neck-shoulder-elbow)
        3. Left elbow angle (shoulder-elbow-wrist)
        4. Right elbow angle (shoulder-elbow-wrist)
        5. Left hip angle (torso-hip-knee)
        6. Right hip angle (torso-hip-knee)
        7. Left knee angle (hip-knee-ankle)
        8. Right knee angle (hip-knee-ankle)
        9. Left ankle angle (knee-ankle-foot)
        10. Right ankle angle (knee-ankle-foot)
        11. Back angle (hip-shoulder-neck)
        12. Neck angle (shoulder-neck-ear)
        13. Wrist angle (elbow-wrist-hand)

        Args:
            landmarks: (33, 3) array of landmarks

        Returns:
            angles: (13,) array of joint angles in degrees
        """
        angles = np.zeros(13, dtype=np.float32)

        # Index mapping (MediaPipe landmark indices)
        NOSE = 0
        LEFT_EYE = 1
        RIGHT_EYE = 2
        LEFT_EAR = 3
        RIGHT_EAR = 4
        LEFT_SHOULDER = 11
        RIGHT_SHOULDER = 12
        LEFT_ELBOW = 13
        RIGHT_ELBOW = 14
        LEFT_WRIST = 15
        RIGHT_WRIST = 16
        LEFT_HIP = 23
        RIGHT_HIP = 24
        LEFT_KNEE = 25
        RIGHT_KNEE = 26
        LEFT_ANKLE = 27
        RIGHT_ANKLE = 28
        LEFT_HEEL = 29
        RIGHT_HEEL = 30
        LEFT_FOOT_INDEX = 31
        RIGHT_FOOT_INDEX = 32

        try:
            # 1-2. Shoulder angles (neck-shoulder-elbow)
            # Using nose as neck proxy
            angles[0] = self.compute_angle(
                landmarks[NOSE], landmarks[LEFT_SHOULDER], landmarks[LEFT_ELBOW]
            )
            angles[1] = self.compute_angle(
                landmarks[NOSE], landmarks[RIGHT_SHOULDER], landmarks[RIGHT_ELBOW]
            )

            # 3-4. Elbow angles (shoulder-elbow-wrist)
            angles[2] = self.compute_angle(
                landmarks[LEFT_SHOULDER], landmarks[LEFT_ELBOW], landmarks[LEFT_WRIST]
            )
            angles[3] = self.compute_angle(
                landmarks[RIGHT_SHOULDER], landmarks[RIGHT_ELBOW], landmarks[RIGHT_WRIST]
            )

            # 5-6. Hip angles (torso-hip-knee)
            # Using midpoint of shoulders as torso
            torso = (landmarks[LEFT_SHOULDER] + landmarks[RIGHT_SHOULDER]) / 2
            angles[4] = self.compute_angle(
                torso, landmarks[LEFT_HIP], landmarks[LEFT_KNEE]
            )
            angles[5] = self.compute_angle(
                torso, landmarks[RIGHT_HIP], landmarks[RIGHT_KNEE]
            )

            # 7-8. Knee angles (hip-knee-ankle)
            angles[6] = self.compute_angle(
                landmarks[LEFT_HIP], landmarks[LEFT_KNEE], landmarks[LEFT_ANKLE]
            )
            angles[7] = self.compute_angle(
                landmarks[RIGHT_HIP], landmarks[RIGHT_KNEE], landmarks[RIGHT_ANKLE]
            )

            # 9-10. Ankle angles (knee-ankle-foot)
            angles[8] = self.compute_angle(
                landmarks[LEFT_KNEE], landmarks[LEFT_ANKLE], landmarks[LEFT_HEEL]
            )
            angles[9] = self.compute_angle(
                landmarks[RIGHT_KNEE], landmarks[RIGHT_ANKLE], landmarks[RIGHT_HEEL]
            )

            # 11. Back angle (hip-shoulder-neck)
            # Using left side
            angles[10] = self.compute_angle(
                landmarks[LEFT_HIP], landmarks[LEFT_SHOULDER], landmarks[NOSE]
            )

            # 12. Neck angle (shoulder-neck-ear)
            angles[11] = self.compute_angle(
                landmarks[LEFT_SHOULDER], landmarks[NOSE], landmarks[LEFT_EAR]
            )

            # 13. Wrist angle (elbow-wrist-hand)
            # Using hand index as proxy
            angles[12] = self.compute_angle(
                landmarks[LEFT_ELBOW], landmarks[LEFT_WRIST], landmarks[LEFT_FOOT_INDEX]
            )

        except (IndexError, ValueError) as e:
            # Return zeros if computation fails
            print(f"Warning: Angle computation failed: {e}")
            pass

        return angles

    def process_video(self,
                     video_path: str,
                     target_frames: int = 60,
                     output_dir: Optional[str] = None) -> Dict:
        """
        Process entire video and extract features.

        Args:
            video_path: Path to video file
            target_frames: Target sequence length (for resampling)
            output_dir: Directory to save output (optional)

        Returns:
            Dictionary with:
                - features: (target_frames, 13) joint angles
                - landmarks: (target_frames, 33, 3) coordinates
                - metadata: video info
        """
        video_path = Path(video_path)

        # Open video
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        # Video metadata
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        print(f"Processing: {video_path}")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps:.1f}")
        print(f"  Total frames: {total_frames}")

        # Sample frames uniformly
        frame_indices = np.linspace(0, total_frames - 1, target_frames, dtype=int)

        all_landmarks = []
        frames_processed = 0

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()

            if not ret:
                print(f"Warning: Could not read frame {idx}")
                continue

            # Extract landmarks
            landmarks = self.extract_landmarks(frame)

            if landmarks is not None:
                all_landmarks.append(landmarks)
                frames_processed += 1
            else:
                # Use interpolation for missing frames
                all_landmarks.append(None)

        cap.release()

        # Interpolate missing landmarks
        all_landmarks = self._interpolate_missing(all_landmarks)
        landmarks_array = np.array(all_landmarks, dtype=np.float32)

        # Compute joint angles
        angles_list = []
        for frame_landmarks in landmarks_array:
            angles = self.compute_joint_angles(frame_landmarks)
            angles_list.append(angles)

        features_array = np.array(angles_list, dtype=np.float32)

        # Save to file if output_dir specified
        output_path = None
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            stem = video_path.stem

            # Save features
            features_file = output_dir / f"{stem}_pose.npz"
            np.savez(features_file, features=features_array, landmarks=landmarks_array)
            output_path = features_file

            print(f"Saved features to {features_file}")

        return {
            'features': features_array,
            'landmarks': landmarks_array,
            'metadata': {
                'video_path': str(video_path),
                'fps': fps,
                'total_frames': total_frames,
                'sampled_frames': target_frames,
                'frames_with_pose': frames_processed,
                'output_file': str(output_path) if output_path else None,
            }
        }

    def _interpolate_missing(self, landmarks_list: List[Optional[np.ndarray]]) -> List[np.ndarray]:
        """
        Interpolate missing landmarks using neighboring frames.

        Args:
            landmarks_list: List of landmarks (some may be None)

        Returns:
            List with all None values replaced by interpolated values
        """
        result = []

        for i, landmarks in enumerate(landsmarks_list):
            if landmarks is not None:
                result.append(landmarks)
            else:
                # Find nearest non-None neighbors
                prev_idx = None
                next_idx = None

                for j in range(i - 1, -1, -1):
                    if landmarks_list[j] is not None:
                        prev_idx = j
                        break

                for j in range(i + 1, len(landsmarks_list)):
                    if landmarks_list[j] is not None:
                        next_idx = j
                        break

                if prev_idx is not None and next_idx is not None:
                    # Linear interpolation
                    alpha = (i - prev_idx) / (next_idx - prev_idx)
                    result.append(
                        (1 - alpha) * landmarks_list[prev_idx] +
                        alpha * landmarks_list[next_idx]
                    )
                elif prev_idx is not None:
                    result.append(landmarks_list[prev_idx])
                elif next_idx is not None:
                    result.append(landmarks_list[next_idx])
                else:
                    # All frames missing - use default pose
                    result.append(np.zeros((33, 3), dtype=np.float32))

        return result

    def close(self):
        """Release MediaPipe resources."""
        self.pose.close()


def batch_extract_videos(video_list: List[str],
                        output_dir: str,
                        target_frames: int = 60) -> Dict:
    """
    Batch process multiple videos.

    Args:
        video_list: List of video paths
        output_dir: Directory to save features
        target_frames: Target sequence length

    Returns:
        Summary dictionary with processing results
    """
    extractor = MediaPipePoseExtractor()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'processed': [],
        'failed': [],
        'total_frames': 0,
        'frames_with_pose': 0,
    }

    for i, video_path in enumerate(video_list):
        print(f"\n[{i+1}/{len(video_list)}] Processing {video_path}...")

        try:
            result = extractor.process_video(
                video_path,
                target_frames=target_frames,
                output_dir=output_dir
            )

            results['processed'].append({
                'video': video_path,
                'output': result['metadata']['output_file'],
                'frames': result['metadata']['frames_with_pose'],
            })
            results['total_frames'] += result['metadata']['sampled_frames']
            results['frames_with_pose'] += result['metadata']['frames_with_pose']

        except Exception as e:
            print(f"Failed to process {video_path}: {e}")
            results['failed'].append({
                'video': video_path,
                'error': str(e),
            })

    extractor.close()

    # Save summary
    summary_file = output_dir / 'processing_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Batch processing complete!")
    print(f"   Processed: {len(results['processed'])}")
    print(f"   Failed: {len(results['failed'])}")
    print(f"   Pose detection rate: {results['frames_with_pose'] / max(1, results['total_frames']):.2%}")
    print(f"   Summary saved to {summary_file}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract pose features from videos")
    parser.add_argument("--video", type=str, required=True, help="Video file path")
    parser.add_argument("--output-dir", type=str, default="results/pose_features",
                       help="Output directory for features")
    parser.add_argument("--target-frames", type=int, default=60,
                       help="Target sequence length")

    args = parser.parse_args()

    extractor = MediaPipePoseExtractor()

    result = extractor.process_video(
        args.video,
        target_frames=args.target_frames,
        output_dir=args.output_dir
    )

    print("\nExtraction complete!")
    print(f"  Features shape: {result['features'].shape}")
    print(f"  Landmarks shape: {result['landmarks'].shape}")
    print(f"  Frames with pose: {result['metadata']['frames_with_pose']}/{result['metadata']['sampled_frames']}")

    extractor.close()
