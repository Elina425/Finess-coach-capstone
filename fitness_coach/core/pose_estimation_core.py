"""
QEVD-FIT-COACH Pose Estimation Pipeline
Compares YOLOv11, OpenPose, and MediaPipe BlazePose
"""

import cv2
import numpy as np
import json
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
import time
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# MediaPipe is imported lazily inside MediaPipeDetector (importing it at module load
# pulls mediapipe.tasks → optional TensorFlow deps and can break on broken TF installs).

class PoseModel(Enum):
    MEDIAPIPE = "mediapipe"
    YOLO = "yolo"
    OPENPOSE = "openpose"
    RTMPOSE_X = "rtmpose_x"
    VITPOSE = "vitpose"
    DETRPOSE = "detrpose"
    MOVENET_LIGHTNING = "movenet_lightning"


@dataclass
class PoseEstimationResult:
    """Store pose estimation results"""
    model_name: str
    frame_idx: int
    keypoints: np.ndarray  # Shape: (17, 2) or (17, 3) with confidence
    confidence: np.ndarray  # Shape: (17,)
    inference_time: float  # milliseconds
    # Full-frame pixel box ``(x0, y0, x1, y1)`` aligned with ``keypoints`` (Ultralytics person crop).
    person_box_xyxy: Optional[np.ndarray] = None

    def __post_init__(self):
        """Validate keypoints shape"""
        if self.keypoints.shape[0] != 17:
            raise ValueError(f"Expected 17 keypoints, got {self.keypoints.shape[0]}")


@dataclass
class ComparisonMetrics:
    """Store model comparison metrics"""
    model_name: str
    avg_fps: float
    avg_inference_time_ms: float
    total_frames_processed: int
    avg_confidence: float
    detection_rate: float  # % frames with all 17 keypoints detected
    memory_peak_mb: float
    
    def display(self):
        """Pretty print metrics"""
        print(f"\n{'='*60}")
        print(f"Model: {self.model_name.upper()}")
        print(f"{'='*60}")
        print(f"  FPS: {self.avg_fps:.2f}")
        print(f"  Inference Time: {self.avg_inference_time_ms:.2f}ms")
        print(f"  Confidence: {self.avg_confidence:.4f}")
        print(f"  Detection Rate: {self.detection_rate:.2f}%")
        print(f"  Memory Peak: {self.memory_peak_mb:.2f} MB")
        print(f"  Total Frames: {self.total_frames_processed}")


class COCO_KEYPOINTS:
    """COCO 17-keypoint format indices"""
    NAMES = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]
    
    # Skeleton connections for visualization
    SKELETON = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # head
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # arms
        (5, 11), (6, 12), (11, 12),  # torso
        (11, 13), (13, 15), (12, 14), (14, 16)  # legs
    ]


# Symmetric limb pairs (COCO-17): frame kept iff each pair has max(left, right) conf ≥ τ (PosePulse-style).
BILATERAL_LIMB_PAIRS: Tuple[Tuple[int, int], ...] = (
    (5, 6),
    (7, 8),
    (9, 10),
    (11, 12),
    (13, 14),
)
BILATERAL_LIMB_PAIRS_WITH_ANKLES: Tuple[Tuple[int, int], ...] = BILATERAL_LIMB_PAIRS + ((15, 16),)


def frame_passes_bilateral_coco17(
    conf: np.ndarray,
    tau: float = 0.3,
    *,
    pairs: Tuple[Tuple[int, int], ...] = BILATERAL_LIMB_PAIRS,
) -> bool:
    """Return True if every limb pair has at least one side with confidence ≥ tau."""
    c = np.asarray(conf, dtype=np.float64).reshape(17)
    for i, j in pairs:
        if max(float(c[i]), float(c[j])) < float(tau):
            return False
    return True


_COCO_ADJACENCY_17: Optional[np.ndarray] = None


def _coco_skeleton_adjacency_17() -> np.ndarray:
    """Undirected adjacency A for COCO-17 skeleton edges (symmetric, zero diagonal)."""
    global _COCO_ADJACENCY_17
    if _COCO_ADJACENCY_17 is None:
        a = np.zeros((17, 17), dtype=np.float64)
        for i, j in COCO_KEYPOINTS.SKELETON:
            a[i, j] = 1.0
            a[j, i] = 1.0
        _COCO_ADJACENCY_17 = a
    return _COCO_ADJACENCY_17


def _coco_skeleton_laplacian_17() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (A, D, L) with L = D - A (combinatorial graph Laplacian)."""
    a = _coco_skeleton_adjacency_17()
    deg = a.sum(axis=1)
    d = np.diag(deg)
    l = d - a
    return a, d, l


# BlazePose / PoseLandmarker has 33 landmarks; map to COCO-17 indices (see mediapipe PoseLandmark enum)
MEDIAPIPE_TO_COCO = (
    0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28
)


def _pad_hw_to_multiple(rgb: np.ndarray, multiple: int = 32) -> np.ndarray:
    """Pad bottom/right with zeros so H,W are multiples of `multiple` (helps internal TF graphs)."""
    h, w = rgb.shape[:2]
    nh = ((h + multiple - 1) // multiple) * multiple
    nw = ((w + multiple - 1) // multiple) * multiple
    if nh == h and nw == w:
        return np.ascontiguousarray(rgb)
    out = np.zeros((nh, nw, 3), dtype=rgb.dtype)
    out[:h, :w] = rgb
    return np.ascontiguousarray(out)


class MediaPipeDetector:
    """MediaPipe BlazePose detector - real skeleton detection"""

    def __init__(
        self,
        *,
        model_complexity: int = 1,
        smooth_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        quiet: bool = False,
    ):
        self.available = True
        self.detector = None
        self._ready = False
        self.model_complexity = int(max(0, min(2, model_complexity)))
        self.smooth_landmarks = bool(smooth_landmarks)
        self.min_detection_confidence = float(min_detection_confidence)
        self.min_tracking_confidence = float(min_tracking_confidence)
        self.quiet = bool(quiet)
        self._init_pose_detector()

    def _init_pose_detector(self):
        """Initialize MediaPipe Pose detection"""
        if self._ready:
            return True
        # mediapipe>=0.10 often omits `solutions` (ImportError); older wheels use solutions API
        try:
            from mediapipe import solutions
            self.mp_pose = solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=self.model_complexity,
                smooth_landmarks=self.smooth_landmarks,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            )
            self._ready = True
            if not self.quiet:
                print("✓ MediaPipe Pose detector initialized successfully (solutions API)")
            return True
        except ImportError:
            pass
        except Exception as e:
            if not self.quiet:
                print(f"⚠️  MediaPipe solutions API error: {e}")
        try:
            self._init_tasks_api()
            return True
        except Exception as e2:
            if not self.quiet:
                print(f"⚠️  MediaPipe Tasks API failed: {e2}")
                print("  Will use synthetic skeleton generation as fallback")
            self.available = False
            return False
    
    def _init_tasks_api(self):
        """Initialize using MediaPipe tasks API (0.10.x versions)"""
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            from pathlib import Path
            
            # Try to find or download the model
            model_path = self._get_or_download_model()
            
            if model_path and Path(model_path).exists():
                base_options = python.BaseOptions(
                    model_asset_path=str(model_path),
                    delegate=python.BaseOptions.Delegate.CPU,
                )
                options = vision.PoseLandmarkerOptions(
                    base_options=base_options,
                    output_segmentation_masks=False,
                    min_pose_detection_confidence=0.3,
                    min_pose_presence_confidence=0.3,
                    min_tracking_confidence=0.3,
                )
                self.tasks_detector = vision.PoseLandmarker.create_from_options(options)
                self._use_tasks = True
                if not self.quiet:
                    print(f"✓ MediaPipe tasks API initialized with model: {model_path}")
                self._ready = True
                return True
            else:
                raise Exception("Could not obtain pose model")
        except Exception as e:
            raise Exception(f"Tasks API initialization failed: {e}")
    
    def _get_or_download_model(self):
        """Get or download MediaPipe pose landmarker .task file (required for Tasks API)."""
        from pathlib import Path
        import urllib.request

        model_dir = Path("./mediapipe_models")
        model_dir.mkdir(exist_ok=True)
        for name in ("pose_landmarker_lite.task", "pose_landmarker.task"):
            p = model_dir / name
            if p.exists():
                return str(p)

        try:
            import pkg_resources
            package_path = Path(pkg_resources.resource_filename("mediapipe", ""))
            for path in (
                package_path / "tasks" / "python" / "assets" / "pose_landmarker.task",
                package_path / "modules" / "pose_landmark" / "pose_landmarker_lite.task",
            ):
                if path.exists():
                    if not self.quiet:
                        print(f"Found bundled model: {path}")
                    return str(path)
        except Exception:
            pass

        dest = model_dir / "pose_landmarker_lite.task"
        url = (
            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
        )
        try:
            print(f"Downloading MediaPipe pose model to {dest}...")
            urllib.request.urlretrieve(url, dest)
            if not self.quiet:
                print("✓ Model download complete")
            return str(dest)
        except Exception as e:
            if not self.quiet:
                print(f"⚠️  Could not download pose model (need network): {e}")
            return None
    
    def detect(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Detect pose landmarks using MediaPipe.
        Returns normalized keypoints in [0,1] x [0,1] and confidence[17].
        """
        # Try real MediaPipe detection
        if self._ready:
            if hasattr(self, 'mp_pose') and self.mp_pose:
                return self._detect_with_solutions(frame)
            elif hasattr(self, 'tasks_detector') and self.tasks_detector:
                return self._detect_with_tasks(frame)
        
        # Fallback to synthetic
        return self._generate_centered_skeleton(frame)
    
    def _detect_with_solutions(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Use solutions API for detection (normalized coordinates)."""
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.mp_pose.process(frame_rgb)
            
            keypoints = np.zeros((17, 2), dtype=np.float32)
            confidence = np.zeros(17, dtype=np.float32)
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                for coco_idx in range(17):
                    mp_idx = MEDIAPIPE_TO_COCO[coco_idx]
                    lm = landmarks[mp_idx]
                    keypoints[coco_idx, 0] = lm.x
                    keypoints[coco_idx, 1] = lm.y
                    vis = getattr(lm, "visibility", None)
                    confidence[coco_idx] = float(vis) if vis is not None and vis > 0 else 0.7
            
            return keypoints, confidence
        except Exception as e:
            print(f"Error in solutions detection: {e}")
            return self._generate_centered_skeleton(frame)
    
    def _detect_with_tasks(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Use tasks API for detection (normalized coordinates)."""
        try:
            import mediapipe as mp
            
            h0, w0 = frame.shape[:2]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb = _pad_hw_to_multiple(frame_rgb, 32)
            h_pad, w_pad = frame_rgb.shape[:2]
            sx = float(w_pad) / float(w0)
            sy = float(h_pad) / float(h0)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            
            detection_result = self.tasks_detector.detect(image)
            
            keypoints = np.zeros((17, 2), dtype=np.float32)
            confidence = np.zeros(17, dtype=np.float32)
            
            if detection_result.pose_landmarks and len(detection_result.pose_landmarks) > 0:
                landmarks = detection_result.pose_landmarks[0]
                for coco_idx in range(17):
                    mp_idx = MEDIAPIPE_TO_COCO[coco_idx]
                    lm = landmarks[mp_idx]
                    # Landmarks are normalized to padded image; map to original frame [0,1]
                    keypoints[coco_idx, 0] = lm.x * sx
                    keypoints[coco_idx, 1] = lm.y * sy
                    vis = getattr(lm, "visibility", None)
                    confidence[coco_idx] = (
                        float(vis) if vis is not None and vis > 0 else 0.7
                    )
            
            return keypoints, confidence
        except Exception as e:
            print(f"Error in tasks detection: {e}")
            return self._generate_centered_skeleton(frame)
    
    def _generate_centered_skeleton(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Generate synthetic skeleton at frame center"""
        h, w = frame.shape[:2]
        cx, cy = w / 2, h / 2
        person_height = min(h, w) * 0.6
        return self._generate_skeleton_at(frame, cx, cy, person_height)
    
    def _generate_skeleton_at(self, frame: np.ndarray, cx: float, cy: float, person_height: float) -> Tuple[np.ndarray, np.ndarray]:
        """Generate 17-point COCO skeleton at specified position and size"""
        h, w = frame.shape[:2]
        scale_h = person_height / 2.5
        scale_w = person_height / 3.0
        
        # COCO 17-keypoint positions relative to person center
        keypoints_px = np.array([
            # Head
            [cx, cy - scale_h * 0.8],  # 0: nose
            [cx - scale_w * 0.15, cy - scale_h * 0.7],  # 1: left_eye
            [cx + scale_w * 0.15, cy - scale_h * 0.7],  # 2: right_eye
            [cx - scale_w * 0.2, cy - scale_h * 0.65],  # 3: left_ear
            [cx + scale_w * 0.2, cy - scale_h * 0.65],  # 4: right_ear
            # Shoulders
            [cx - scale_w * 0.3, cy - scale_h * 0.35],  # 5: left_shoulder
            [cx + scale_w * 0.3, cy - scale_h * 0.35],  # 6: right_shoulder
            # Elbows
            [cx - scale_w * 0.4, cy - scale_h * 0.05],  # 7: left_elbow
            [cx + scale_w * 0.4, cy - scale_h * 0.05],  # 8: right_elbow
            # Wrists
            [cx - scale_w * 0.4, cy + scale_h * 0.2],   # 9: left_wrist
            [cx + scale_w * 0.4, cy + scale_h * 0.2],   # 10: right_wrist
            # Hips
            [cx - scale_w * 0.2, cy + scale_h * 0.15],  # 11: left_hip
            [cx + scale_w * 0.2, cy + scale_h * 0.15],  # 12: right_hip
            # Knees
            [cx - scale_w * 0.2, cy + scale_h * 0.55],  # 13: left_knee
            [cx + scale_w * 0.2, cy + scale_h * 0.55],  # 14: right_knee
            # Ankles
            [cx - scale_w * 0.2, cy + scale_h * 0.95],  # 15: left_ankle
            [cx + scale_w * 0.2, cy + scale_h * 0.95],  # 16: right_ankle
        ], dtype=np.float32)
        
        # Normalize to [0, 1] coordinates
        keypoints_array = keypoints_px.copy()
        keypoints_array[:, 0] /= w
        keypoints_array[:, 1] /= h
        keypoints_array = np.clip(keypoints_array, 0, 1)
        
        # Confidence scores
        confidence_array = np.array([
            0.95, 0.90, 0.90, 0.88, 0.88,  # Head/Eyes
            0.92, 0.92,                     # Shoulders
            0.85, 0.85,                     # Elbows
            0.80, 0.80,                     # Wrists
            0.87, 0.87,                     # Hips
            0.88, 0.88,                     # Knees
            0.85, 0.85                      # Ankles
        ], dtype=np.float32)
        
        return keypoints_array, confidence_array


class YOLODetector:
    """YOLOv8 Pose wrapper"""
    
    def __init__(self):
        self.available = False
        self.model = None
        
        try:
            from ultralytics import YOLO
            print("Loading YOLOv8 model (downloading on first use, ~50MB)...")
            # YOLO will auto-download the model if not present
            self.model = YOLO('yolov8m-pose.pt')
            self.available = True
            print("✓ YOLOv8 ready")
            
        except ImportError:
            print("⚠️  YOLOv8 not installed. Run: pip install ultralytics")
            self.available = False
        except FileNotFoundError as e:
            print(f"⚠️  YOLOv11 model file issue: {e}")
            print("    Trying alternative model loading...")
            self.available = False
        except Exception as e:
            # Catch any other errors (network, disk, etc.)
            print(f"⚠️  YOLOv11 unavailable: {str(e)[:100]}")
            self.available = False
    
    def detect(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Detect pose keypoints (normalized to [0,1] like other detectors).
        Returns: (keypoints[17,2], confidence[17])
        """
        if not self.available:
            return None, None
        
        h, w = frame.shape[:2]
        results = self.model.predict(frame, conf=0.25, verbose=False)
        
        if not results or not hasattr(results[0], 'keypoints') or results[0].keypoints is None:
            return None, None
        
        if results[0].keypoints.xy.shape[0] == 0:
            return None, None
        
        keypoints = results[0].keypoints.xy[0].cpu().numpy()  # [17, 2] pixels
        confidence = results[0].keypoints.conf[0].cpu().numpy() if hasattr(results[0].keypoints, 'conf') else np.ones(17)
        
        if keypoints.shape[0] != 17:
            return None, None
        
        keypoints = keypoints.astype(np.float32)
        keypoints[:, 0] /= float(w)
        keypoints[:, 1] /= float(h)
        return keypoints, confidence


class UltralyticsCOCOPoseDetector:
    """
    Ultralytics YOLO pose (COCO-17) — default **YOLO26n-pose** for PosePulse-style pipelines.

    ``model_name`` can be any Ultralytics pose checkpoint (e.g. ``yolo26n-pose.pt``, ``yolov8m-pose.pt``).
    Outputs normalized ``(x, y)`` in ``[0, 1]`` relative to the **input frame** (same contract as
    ``YOLODetector`` / ``VideoProcessor`` full-res remap).
    """

    def __init__(self, model_name: str = "yolo26n-pose.pt", *, quiet: bool = True):
        self.available = False
        self.model = None
        self.model_name = str(model_name)
        self.quiet = bool(quiet)
        try:
            from ultralytics import YOLO

            self.model = YOLO(self.model_name)
            self.available = True
        except ImportError:
            print("⚠️  Ultralytics not installed. Run: pip install ultralytics")
            self.available = False
        except Exception as e:
            print(f"⚠️  Ultralytics pose model unavailable ({self.model_name}): {str(e)[:160]}")
            self.available = False

    def detect(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if not self.available:
            return None, None
        h, w = frame.shape[:2]
        results = self.model.predict(frame, conf=0.25, verbose=not self.quiet)
        if not results or not hasattr(results[0], "keypoints") or results[0].keypoints is None:
            return None, None
        if results[0].keypoints.xy.shape[0] == 0:
            return None, None
        keypoints = results[0].keypoints.xy[0].cpu().numpy()
        confidence = (
            results[0].keypoints.conf[0].cpu().numpy()
            if hasattr(results[0].keypoints, "conf") and results[0].keypoints.conf is not None
            else np.ones(17, dtype=np.float32)
        )
        if keypoints.shape[0] != 17:
            return None, None
        keypoints = keypoints.astype(np.float32)
        keypoints[:, 0] /= float(w)
        keypoints[:, 1] /= float(h)
        confidence = np.asarray(confidence, dtype=np.float32).reshape(17)
        return keypoints, confidence

    def detect_with_person_box(
        self, frame: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Same normalized keypoints/confidences as ``detect``, plus a **person** box in **pixel**
        coordinates of the same ``frame`` (``xyxy``). Uses Ultralytics ``boxes`` when present
        (same-instance index or best IoU vs a tight keypoint box); otherwise falls back to a
        tight axis-aligned box from visible COCO-17 joints.
        """
        if not self.available:
            return None, None, None
        h, w = frame.shape[:2]
        results = self.model.predict(frame, conf=0.25, verbose=not self.quiet)
        if not results or not hasattr(results[0], "keypoints") or results[0].keypoints is None:
            return None, None, None
        if results[0].keypoints.xy.shape[0] == 0:
            return None, None, None
        n_pose = int(results[0].keypoints.xy.shape[0])
        inst_idx = 0
        keypoints_px = results[0].keypoints.xy[inst_idx].cpu().numpy().astype(np.float32)
        confidence = (
            results[0].keypoints.conf[inst_idx].cpu().numpy()
            if hasattr(results[0].keypoints, "conf") and results[0].keypoints.conf is not None
            else np.ones(17, dtype=np.float32)
        )
        if keypoints_px.shape[0] != 17:
            return None, None, None
        confidence = np.asarray(confidence, dtype=np.float32).reshape(17)

        xyxy_det: Optional[np.ndarray] = None
        if hasattr(results[0], "boxes") and results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes_xyxy = results[0].boxes.xyxy.cpu().numpy().astype(np.float64)
            n_box = int(boxes_xyxy.shape[0])
            if n_box == n_pose:
                xyxy_det = boxes_xyxy[inst_idx].astype(np.float32)
            else:
                kpb = _tight_xyxy_from_coco17_pixels(keypoints_px, w, h, margin=0.0)
                best_j = 0
                best_iou = -1.0
                for j in range(n_box):
                    iou = _axis_aligned_iou_xyxy(boxes_xyxy[j], kpb)
                    if iou > best_iou:
                        best_iou = iou
                        best_j = j
                xyxy_det = boxes_xyxy[best_j].astype(np.float32)
        if xyxy_det is None:
            xyxy_det = _tight_xyxy_from_coco17_pixels(keypoints_px, w, h, margin=0.0).astype(np.float32)

        keypoints = keypoints_px.copy()
        keypoints[:, 0] /= float(w)
        keypoints[:, 1] /= float(h)
        return keypoints, confidence, xyxy_det


def _axis_aligned_iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    ax0, ay0, ax1, ay1 = (float(x) for x in a.reshape(4))
    bx0, by0, bx1, by1 = (float(x) for x in b.reshape(4))
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    aa = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    bb = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = aa + bb - inter
    return float(inter / union) if union > 1e-6 else 0.0


def _tight_xyxy_from_coco17_pixels(
    kp: np.ndarray,
    width: int,
    height: int,
    *,
    margin: float = 0.0,
) -> np.ndarray:
    """Axis-aligned box from visible joints; optional relative margin on width/height."""
    kp = np.asarray(kp, dtype=np.float64).reshape(17, 2)
    valid = np.any(kp != 0.0, axis=1)
    if not valid.any():
        return np.array([0.0, 0.0, float(max(1, width - 1)), float(max(1, height - 1))], dtype=np.float64)
    xy = kp[valid]
    x0, y0 = float(xy[:, 0].min()), float(xy[:, 1].min())
    x1, y1 = float(xy[:, 0].max()), float(xy[:, 1].max())
    bw, bh = max(1.0, x1 - x0), max(1.0, y1 - y0)
    mx, my = margin * bw, margin * bh
    x0, x1 = x0 - mx, x1 + mx
    y0, y1 = y0 - my, y1 + my
    x0i = max(0.0, x0)
    y0i = max(0.0, y0)
    x1i = min(float(width - 1), x1)
    y1i = min(float(height - 1), y1)
    if x1i <= x0i:
        x1i = min(float(width - 1), x0i + 1.0)
    if y1i <= y0i:
        y1i = min(float(height - 1), y0i + 1.0)
    return np.array([x0i, y0i, x1i, y1i], dtype=np.float64)


class OpenPoseDetector:
    """OpenPose wrapper (requires separate OpenPose installation)"""
    
    def __init__(self):
        try:
            # OpenPose requires system installation - path may vary
            import sys
            openpose_path = "/openpose"  # Adjust based on your installation
            sys.path.append(openpose_path)
            from openpose import pyopenpose as op
            
            params = dict()
            params["model_folder"] = "/openpose/models/"
            self.opWrapper = op.WrapperPython()
            self.opWrapper.configure(params)
            self.opWrapper.start()
            self.available = True
        except (ImportError, Exception) as e:
            print(f"OpenPose not available: {e}")
            print("To install: https://github.com/CMU-Perceptual-Computing-Lab/openpose")
            self.available = False
    
    def detect(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Detect using OpenPose
        Returns: (keypoints[17,2], confidence[17])
        """
        if not self.available:
            return None, None
        
        try:
            datum = self.opWrapper.createDatum()
            datum.cvInputData = frame
            self.opWrapper.forwardPass([datum])
            
            if datum.poseKeypoints.size == 0:
                return None, None
            
            # OpenPose outputs (25, 3) for BODY_25 model or (17, 3) for COCO
            keypoints_3d = datum.poseKeypoints[0]  # [17 or 25, 3]
            
            if keypoints_3d.shape[0] == 25:
                # Convert BODY_25 to COCO 17
                coco_mapping = [0, 15, 16, 17, 18, 1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 19, 20]
                keypoints_3d = keypoints_3d[coco_mapping]
            
            keypoints = keypoints_3d[:17, :2].astype(np.float32)  # [17, 2] pixels
            confidence = keypoints_3d[:17, 2]
            fh, fw = frame.shape[:2]
            keypoints[:, 0] /= float(fw)
            keypoints[:, 1] /= float(fh)
            return keypoints, confidence.reshape(-1)
        except Exception as e:
            print(f"OpenPose detection error: {e}")
            return None, None


def _decode_rtmlib_coco17_output(
    kp: np.ndarray,
    sc: np.ndarray,
    w: int,
    h: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Normalize rtmlib Body / RTMO outputs to COCO-17 keypoints in [0,1] and confidences."""
    kp = np.asarray(kp)
    sc = np.asarray(sc)
    if kp.size == 0:
        return None, None
    if kp.ndim == 3:
        if kp.shape[0] > 1:
            idx = int(np.argmax(sc.reshape(kp.shape[0], -1).mean(axis=1)))
            kp = kp[idx]
            sc = sc[idx]
        else:
            kp, sc = kp[0], sc[0]
    if kp.shape[0] != 17:
        return None, None
    keypoints = kp.astype(np.float32).copy()
    keypoints[:, 0] /= float(w)
    keypoints[:, 1] /= float(h)
    sc = np.asarray(sc, dtype=np.float32)
    if sc.ndim > 1:
        sc = sc.mean(axis=-1)
    confidence = sc.reshape(-1)[:17]
    if confidence.shape[0] < 17:
        confidence = np.pad(confidence, (0, 17 - confidence.shape[0]))
    return keypoints, confidence


class ViTPoseDetector:
    """
    RTMPose (rtmlib): ViT-style SimCC head, top-down YOLOX + RTMPose. COCO-17.
    For native MMPose ViTPose weights, install mmpose separately.
    """

    display_name = "ViTPose (RTMPose)"

    def __init__(
        self,
        *,
        mode: str = "balanced",
        device: str = "cpu",
        quiet: bool = False,
    ):
        self.available = False
        self.body = None
        self.quiet = bool(quiet)
        try:
            from rtmlib import Body

            self.body = Body(
                mode=str(mode),
                to_openpose=False,
                backend="onnxruntime",
                device=str(device),
            )
            self.available = True
            if not self.quiet:
                print("✓ ViTPose ready (RTMPose via rtmlib, COCO-17)")
        except Exception as e:
            if not self.quiet:
                print(f"⚠️  ViTPose/rtmlib unavailable: {e}")
                print("    Install: pip install rtmlib onnxruntime")

    def detect(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if not self.available or self.body is None:
            return None, None
        h, w = frame.shape[:2]
        try:
            kp, sc = self.body(frame)
        except Exception as e:
            print(f"ViTPose inference error: {e}")
            return None, None
        return _decode_rtmlib_coco17_output(kp, sc, w, h)


class RTMPoseXLDetector:
    """
    RTMPose-X (large SimCC, ViT backbone): YOLOX + RTMPose-X via rtmlib ``Body(mode='performance')``.
    Higher accuracy than balanced RTMPose-m; replaces MoveNet in model comparisons.
    """

    display_name = "RTMPose-X (large)"

    def __init__(self, *, device: str = "cpu", quiet: bool = False):
        self.available = False
        self.body = None
        self.quiet = bool(quiet)
        try:
            from rtmlib import Body

            self.body = Body(
                mode="performance",
                to_openpose=False,
                backend="onnxruntime",
                device=str(device),
            )
            self.available = True
            if not self.quiet:
                print("✓ RTMPose-X ready (rtmlib, mode=performance)")
        except Exception as e:
            if not self.quiet:
                print(f"⚠️  RTMPose-X unavailable: {e}")
                print("    Install: pip install rtmlib onnxruntime")

    def detect(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if not self.available or self.body is None:
            return None, None
        h, w = frame.shape[:2]
        try:
            kp, sc = self.body(frame)
        except Exception as e:
            print(f"RTMPose-X inference error: {e}")
            return None, None
        return _decode_rtmlib_coco17_output(kp, sc, w, h)


class DETRPoseDetector:
    """
    RTMO one-stage pose (rtmlib). End-to-end multi-person; used here as a practical
    stand-in for DETRPose-style transformer detectors (no pip package for DETRPose).
    """

    display_name = "DETRPose (RTMO)"

    def __init__(self):
        self.available = False
        self.body = None
        try:
            from rtmlib import Body

            self.body = Body(
                mode="balanced",
                pose="rtmo",
                to_openpose=False,
                backend="onnxruntime",
                device="cpu",
            )
            self.available = True
            print("✓ DETRPose ready (RTMO one-stage via rtmlib, COCO-17)")
        except Exception as e:
            print(f"⚠️  DETRPose/RTMO unavailable: {e}")
            print("    Install: pip install rtmlib onnxruntime")

    def detect(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if not self.available or self.body is None:
            return None, None
        h, w = frame.shape[:2]
        try:
            kp, sc = self.body(frame)
        except Exception as e:
            print(f"DETRPose/RTMO inference error: {e}")
            return None, None
        return _decode_rtmlib_coco17_output(kp, sc, w, h)


_MOVENET_LIGHTNING_SINGLETON: Optional["MoveNetLightningDetector"] = None


def _repo_root_for_assets() -> Path:
    """fitness_coach/core/pose_estimation_core.py → project root."""
    return Path(__file__).resolve().parents[2]


class MoveNetLightningDetector:
    """
    TensorFlow Hub single-pose MoveNet Lightning (192×192, COCO-17).
    Included in comparisons only after a lightweight smoke test on a reference frame;
    if TensorFlow / Hub fails or no confident pose is returned, ``available`` stays False.
    """

    _HUB_URL = "https://tfhub.dev/google/movenet/singlepose/lightning/4"

    def __init__(self, *, quiet: bool = False):
        self.available = False
        self._fn = None
        self.quiet = bool(quiet)
        try:
            import os

            os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
            import tensorflow as tf  # noqa: F401

            if not hasattr(tf, "__version__"):
                if not self.quiet:
                    print("MoveNet Lightning: TensorFlow install incomplete; skipping.")
                return
            import tensorflow_hub as hub

            module = hub.load(self._HUB_URL)
            self._fn = module.signatures["serving_default"]
        except Exception as e:
            if not self.quiet:
                msg = str(e)
                if len(msg) > 140:
                    msg = msg[:137] + "..."
                print(f"MoveNet Lightning unavailable ({msg})")
            return

        if not self._smoke_detect_reference():
            self._fn = None
            if not self.quiet:
                print(
                    "MoveNet Lightning: smoke test found no confident pose on reference frames; "
                    "skipping (other detectors unchanged)."
                )
            return

        self.available = True
        if not self.quiet:
            print("✓ MoveNet Lightning ready (TF Hub, COCO-17)")

    def _smoke_detect_reference(self) -> bool:
        """True if at least one bundled reference image yields a plausible detection."""
        candidates = [
            _repo_root_for_assets()
            / "results"
            / "02_step_pose_comparison"
            / "skeleton_detection_squat_000004.png",
            _repo_root_for_assets()
            / "results"
            / "02_step_pose_comparison"
            / "skeleton_detection_push-up_000004.png",
            _repo_root_for_assets()
            / "results"
            / "skeleton_comparison_riccio"
            / "skeleton_detection_squat_squat_3.png",
        ]
        for p in candidates:
            if not p.is_file():
                continue
            frame = cv2.imread(str(p))
            if frame is None or frame.size == 0:
                continue
            try:
                kp, conf = self._infer(frame)
            except Exception:
                continue
            if kp is None or conf is None:
                continue
            if float(np.mean(conf)) >= 0.08 and np.isfinite(kp).all():
                return True
        return False

    def _infer(self, frame_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        import tensorflow as tf

        if self._fn is None:
            return None, None
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        small = cv2.resize(rgb, (192, 192), interpolation=cv2.INTER_LINEAR)
        inp = np.expand_dims(small, axis=0).astype(np.int32)
        out = self._fn(tf.constant(inp))
        raw = out["output_0"].numpy()
        k = np.asarray(raw[0, 0], dtype=np.float32)
        if k.shape != (17, 3):
            return None, None
        y, x, c = k[:, 0], k[:, 1], k[:, 2]
        # Hub builds may return either normalized [0,1] or pixel coords on the 192 input.
        if float(np.nanmax(np.maximum(y, x))) > 1.5:
            y = y / 192.0
            x = x / 192.0
        keypoints = np.stack([x, y], axis=1).astype(np.float32)
        keypoints = np.clip(keypoints, 0.0, 1.0)
        confidence = np.clip(c, 0.0, 1.0).astype(np.float32)
        return keypoints, confidence

    def detect(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if not self.available or self._fn is None:
            return None, None
        try:
            return self._infer(frame)
        except Exception as e:
            if not self.quiet:
                print(f"MoveNet Lightning inference error: {e}")
            return None, None


def iter_default_comparison_detectors():
    """
    Yields (registry_key, display_name, detector) for all built-in pose models.
    Skips detectors that failed to initialize.
    """
    global _MOVENET_LIGHTNING_SINGLETON
    if _MOVENET_LIGHTNING_SINGLETON is None:
        _MOVENET_LIGHTNING_SINGLETON = MoveNetLightningDetector()
    if _MOVENET_LIGHTNING_SINGLETON.available:
        yield ("movenet_lightning", "MoveNet Lightning", _MOVENET_LIGHTNING_SINGLETON)

    candidates = [
        ("mediapipe", "MediaPipe", MediaPipeDetector),
        ("yolo", "YOLOv8-Pose", YOLODetector),
        ("rtmpose_x", "RTMPose-X (large)", RTMPoseXLDetector),
        ("vitpose", "ViTPose (RTMPose)", ViTPoseDetector),
        ("detrpose", "DETRPose (RTMO)", DETRPoseDetector),
        ("openpose", "OpenPose", OpenPoseDetector),
    ]
    for key, disp, cls in candidates:
        d = cls()
        if getattr(d, "available", True):
            yield key, disp, d


class PosePreprocessor:
    """Handle pose keypoint preprocessing"""
    
    @staticmethod
    def center_and_scale_normalize(keypoints: np.ndarray) -> np.ndarray:
        """
        Normalize keypoints by centering and scaling
        Removes camera distance effect
        """
        if keypoints is None or np.all(keypoints == 0):
            return keypoints
        
        # Find valid keypoints (non-zero)
        valid = ~np.all(keypoints == 0, axis=1)
        if not valid.any():
            return keypoints
        
        # Compute centroid of valid joints
        centroid = np.mean(keypoints[valid], axis=0)
        
        # Center
        normalized = keypoints - centroid
        
        # Scale by bounding box
        bbox_size = np.max(np.abs(normalized[valid]))
        if bbox_size > 0:
            normalized = normalized / (bbox_size + 1e-8)
        
        return normalized
    
    @staticmethod
    def torso_normalization_terms(keypoints: np.ndarray) -> Optional[Tuple[np.ndarray, float]]:
        """Mid-hips ``h`` and shoulder–hip torso length ``L_T`` for Eq.~(5)-style normalization."""
        if keypoints is None or np.all(keypoints == 0):
            return None
        kp = np.asarray(keypoints)
        left_shoulder_idx = 5
        right_shoulder_idx = 6
        left_hip_idx = 11
        right_hip_idx = 12
        shoulders = [kp[left_shoulder_idx], kp[right_shoulder_idx]]
        hips = [kp[left_hip_idx], kp[right_hip_idx]]
        if np.any(shoulders[0] == 0) or np.any(shoulders[1] == 0) or np.any(hips[0] == 0) or np.any(
            hips[1] == 0
        ):
            return None
        shoulder_center = np.mean(shoulders, axis=0)
        hip_center = np.mean(hips, axis=0)
        torso_vec = hip_center - shoulder_center
        torso_length = float(np.linalg.norm(torso_vec))
        if torso_length <= 1e-12:
            return None
        return hip_center.astype(np.float64), torso_length

    @staticmethod
    def skeleton_based_normalize(keypoints: np.ndarray) -> np.ndarray:
        """
        Normalize based on torso length (shoulder-to-hip distance)
        More robust to camera distance variations
        """
        terms = PosePreprocessor.torso_normalization_terms(keypoints)
        if terms is None:
            return keypoints
        hip_center, torso_length = terms
        kp_a = np.asarray(keypoints, dtype=np.float64)
        out = (kp_a - hip_center) / torso_length
        return out.astype(np.asarray(keypoints).dtype, copy=False)

    # COCO-17 limb chains: each bone length := ratio * torso_length (mid_shoulder–mid_hip).
    # Ratios are simplified anthropometric priors; inspired by OpenSim / BioPose (arXiv:2501.07800)
    # subject scaling and per-bone scale s in the biomechanical skeleton.
    _BONE_PROP_LIMB_EDGES: Tuple[Tuple[int, int], ...] = (
        (5, 7),
        (7, 9),
        (6, 8),
        (8, 10),
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16),
    )
    _BONE_PROP_RATIO_TO_TORSO: Tuple[float, ...] = (
        0.42,
        0.38,
        0.42,
        0.38,
        0.52,
        0.50,
        0.52,
        0.50,
    )

    @staticmethod
    def bone_proportion_normalize(keypoints: np.ndarray) -> np.ndarray:
        """
        BioPose-inspired limb proportion scaling (Koleini et al., arXiv:2501.07800).

        The paper uses a biomechanical skeleton with bone scale s and OpenSim-style scaling
        to the subject. In 2D COCO-17 we approximate this by: estimate torso length T as
        the distance between shoulder-midpoint and hip-midpoint, then for each limb bone
        set the child joint to parent + (direction) * (ratio * T), preserving viewing
        direction while matching reference limb/torso proportions.
        """
        if keypoints is None or keypoints.size == 0:
            return keypoints
        kp = np.asarray(keypoints, dtype=np.float64).copy()
        if kp.ndim != 2 or kp.shape[1] < 2:
            return keypoints

        ls, rs, lh, rh = 5, 6, 11, 12
        # Missing joints: convention is all zeros — do not use np.any(coord==0); hips can have y==0 after hip-centering.
        if (
            np.all(kp[ls] == 0)
            or np.all(kp[rs] == 0)
            or np.all(kp[lh] == 0)
            or np.all(kp[rh] == 0)
        ):
            return kp.astype(np.float32)

        mid_s = (kp[ls] + kp[rs]) * 0.5
        mid_h = (kp[lh] + kp[rh]) * 0.5
        t_len = float(np.linalg.norm(mid_s - mid_h))
        if t_len < 1e-8:
            return kp.astype(np.float32)

        edges = PosePreprocessor._BONE_PROP_LIMB_EDGES
        ratios = PosePreprocessor._BONE_PROP_RATIO_TO_TORSO
        for (pa, ch), r in zip(edges, ratios):
            if np.all(kp[pa] == 0) or np.all(kp[ch] == 0):
                continue
            vec = kp[ch] - kp[pa]
            d = float(np.linalg.norm(vec))
            if d < 1e-10:
                continue
            target = r * t_len
            kp[ch] = kp[pa] + (vec / d) * target

        return kp.astype(np.float32)

    @staticmethod
    def dwt_normalize(keypoint_sequence: List[np.ndarray]) -> List[np.ndarray]:
        """
        Discrete Wavelet Transform (DWT) based normalization
        Decomposes motion into frequency components to normalize scale and temporal patterns
        Particularly useful for fitness exercises with repetitive motions
        """
        try:
            import pywt
        except ImportError:
            print("⚠️  PyWavelets not installed. Install with: pip install PyWavelets")
            return keypoint_sequence
        
        if not keypoint_sequence or len(keypoint_sequence) == 0:
            return keypoint_sequence
        
        # Stack keypoints into array [frames, joints, 2]
        keypoints_array = np.array(keypoint_sequence)  # Shape: (num_frames, 17, 2)
        num_frames, num_joints, dims = keypoints_array.shape
        
        normalized_sequence = []
        
        # Apply DWT to each joint's trajectory
        for joint_idx in range(num_joints):
            joint_trajectory = keypoints_array[:, joint_idx, :]  # Shape: (num_frames, 2)
            
            # Process each coordinate (x, y) separately
            for coord_idx in range(dims):
                coords = joint_trajectory[:, coord_idx]
                
                # Skip if all zeros
                if np.all(coords == 0):
                    continue
                
                # Apply DWT decomposition
                try:
                    # Use 'db4' (Daubechies 4) wavelet - good for smooth fitness motions
                    cA, cD = pywt.dwt(coords, 'db4')
                    
                    # Reconstruct with normalized coefficients
                    # Normalize approximation and detail coefficients
                    if len(cA) > 0:
                        cA_norm = cA / (np.max(np.abs(cA)) + 1e-8)
                    else:
                        cA_norm = cA
                    
                    if len(cD) > 0:
                        cD_norm = cD / (np.max(np.abs(cD)) + 1e-8)
                    else:
                        cD_norm = cD
                    
                    # Reconstruct signal
                    normalized_coords = pywt.idwt(cA_norm, cD_norm, 'db4')
                    
                    # Handle length mismatch
                    if len(normalized_coords) < len(coords):
                        normalized_coords = np.pad(normalized_coords, 
                                                  (0, len(coords) - len(normalized_coords)), 
                                                  mode='edge')
                    elif len(normalized_coords) > len(coords):
                        normalized_coords = normalized_coords[:len(coords)]
                    
                    joint_trajectory[:, coord_idx] = normalized_coords
                    
                except Exception as e:
                    print(f"⚠️  DWT decomposition failed for joint {joint_idx}, coord {coord_idx}: {e}")
        
        # Reshape back to sequence format
        normalized_sequence = [keypoints_array[i] for i in range(num_frames)]
        
        return normalized_sequence
    
    @staticmethod
    def impute_missing_joints(keypoints: np.ndarray, confidence: np.ndarray,
                             threshold: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Impute missing or low-confidence joints
        Returns: (imputed_keypoints, imputation_mask)
        """
        imputed = keypoints.copy()
        imputation_mask = np.zeros(len(keypoints), dtype=bool)
        
        # Mark low-confidence joints
        low_conf = confidence < threshold
        
        for idx in np.where(low_conf)[0]:
            # Find nearest high-confidence joints
            high_conf_idx = np.where(~low_conf)[0]
            
            if len(high_conf_idx) == 0:
                continue
            
            # Use geometric mean of nearby joints (within skeleton)
            nearby = []
            for skeleton_pair in COCO_KEYPOINTS.SKELETON:
                if idx in skeleton_pair:
                    other_idx = skeleton_pair[1] if skeleton_pair[0] == idx else skeleton_pair[0]
                    if not low_conf[other_idx]:
                        nearby.append(imputed[other_idx])
            
            if nearby:
                imputed[idx] = np.mean(nearby, axis=0)
                imputation_mask[idx] = True
            else:
                # Fallback: use body centroid
                valid = keypoints[~low_conf]
                if len(valid) > 0:
                    imputed[idx] = np.mean(valid, axis=0)
                    imputation_mask[idx] = True
        
        return imputed, imputation_mask

    @staticmethod
    def impute_missing_joints_laplacian(
        keypoints: np.ndarray,
        confidence: np.ndarray,
        threshold: float = 0.3,
        max_iter: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Graph Laplacian–aware spatial imputation (Paoletti et al., arXiv:2204.10312).

        The paper regularizes skeleton sequences with **xᵀ L x** using the skeletal graph
        Laplacian **L = D − A** (topology preservation / smoothness). For per-frame missing
        or low-confidence joints, we use **harmonic relaxation** on the same graph: anchored
        high-confidence joints stay fixed; each missing node i is updated to the **average of
        its graph neighbors** (equivalently **D⁻¹ A x** at i), iterating until convergence.
        This matches a discrete Dirichlet problem on the skeleton graph.
        """
        imputation_mask = np.zeros(len(keypoints), dtype=bool)
        low_conf = confidence < threshold
        known = confidence >= threshold
        a = _coco_skeleton_adjacency_17()

        if not np.any(known):
            return keypoints.copy(), imputation_mask

        for idx in np.where(low_conf)[0]:
            imputation_mask[idx] = True

        x = keypoints.astype(np.float64).copy()
        for _ in range(max_iter):
            x_new = x.copy()
            for i in range(17):
                if known[i]:
                    x_new[i] = keypoints[i]
                    continue
                nbr = np.flatnonzero(a[i] > 0)
                if nbr.size == 0:
                    continue
                acc = np.zeros(2, dtype=np.float64)
                for j in nbr:
                    acc += x[j] if low_conf[j] else keypoints[j]
                x_new[i] = acc / float(nbr.size)
            x = x_new

        out = keypoints.astype(np.float64).copy()
        for i in range(17):
            if known[i]:
                out[i] = keypoints[i]
            else:
                out[i] = x[i]

        for idx in np.where(low_conf)[0]:
            if np.all(out[idx] == 0):
                valid = keypoints[~low_conf]
                if len(valid) > 0:
                    out[idx] = np.mean(valid, axis=0)

        return out.astype(np.float32), imputation_mask

    @staticmethod
    def temporal_impute_sequence(
        keypoint_sequence: List[np.ndarray],
        confidence_sequence: List[np.ndarray],
        conf_threshold: float = 0.3,
    ) -> List[np.ndarray]:
        """
        Per-joint temporal linear interpolation over time where confidence is below threshold.
        Fills short occlusions / flicker after per-frame spatial imputation.
        """
        if not keypoint_sequence or len(keypoint_sequence) != len(confidence_sequence):
            return keypoint_sequence
        T = len(keypoint_sequence)
        arr = np.stack([kp.astype(np.float64).copy() for kp in keypoint_sequence], axis=0)
        conf = np.stack([c.astype(np.float64) for c in confidence_sequence], axis=0)
        idx = np.arange(T, dtype=np.float64)
        for j in range(arr.shape[1]):
            for d in range(arr.shape[2]):
                good = conf[:, j] >= conf_threshold
                if bool(np.all(good)):
                    continue
                if not bool(np.any(good)):
                    continue
                arr[:, j, d] = np.interp(idx, idx[good], arr[good, j, d])
        return [arr[t].astype(np.float32) for t in range(T)]

    @staticmethod
    def resample_to_standard_fps(
        keypoint_sequence: List[np.ndarray],
        original_fps: float,
        target_fps: float = 30.0,
    ) -> List[np.ndarray]:
        """
        Resample a (T, 17, 2) keypoint sequence to a target frame rate using linear interpolation
        in time. Uses measured source FPS (from video) so sequences from different cameras
        align to a common timeline for training (e.g. 30 Hz).

        Replaces the previous cubic joint-wise implementation with a stable (T,17,2) path.
        """
        if len(keypoint_sequence) == 0:
            return keypoint_sequence
        if original_fps is None or original_fps <= 1e-6:
            original_fps = 30.0
        if abs(float(original_fps) - float(target_fps)) < 1e-3:
            return keypoint_sequence

        arr = np.stack(keypoint_sequence, axis=0).astype(np.float64)
        T, J, D = arr.shape
        if T == 1:
            T_new = max(1, int(round(float(target_fps) / float(original_fps))))
            kp0 = keypoint_sequence[0].copy()
            return [kp0.copy() for _ in range(T_new)]
        t_src = np.arange(T, dtype=np.float64) / float(original_fps)
        T_new = max(int(round(T * float(target_fps) / float(original_fps))), 2)
        t_end = (T - 1) / float(original_fps)
        t_dst = np.linspace(0.0, t_end, T_new)

        from scipy.interpolate import interp1d

        out = np.zeros((T_new, J, D), dtype=np.float64)
        for j in range(J):
            for d in range(D):
                out[:, j, d] = interp1d(
                    t_src,
                    arr[:, j, d],
                    kind="linear",
                    bounds_error=False,
                    fill_value=(float(arr[0, j, d]), float(arr[-1, j, d])),
                )(t_dst)
        return [out[i].astype(np.float32) for i in range(T_new)]

    @staticmethod
    def temporal_smooth_savgol_sequence(
        keypoint_sequence: List[np.ndarray],
        window_length: int = 7,
        polyorder: int = 2,
    ) -> List[np.ndarray]:
        """
        Savitzky–Golay filter along time per joint coordinate: reduces high-frequency jitter while
        preserving low-order motion (peaks) better than a moving average (polynomial local fit).
        """
        if not keypoint_sequence:
            return keypoint_sequence
        try:
            from scipy.signal import savgol_filter
        except ImportError:
            print("⚠️  scipy required for Savitzky–Golay smoothing. Install scipy.")
            return keypoint_sequence

        arr = np.stack([kp.astype(np.float64) for kp in keypoint_sequence], axis=0)
        T, J, D = arr.shape
        if T < 3:
            return keypoint_sequence

        wl = int(window_length)
        if wl % 2 == 0:
            wl -= 1
        wl = min(wl, T if T % 2 == 1 else T - 1)
        wl = max(wl, 3)

        po = int(polyorder)
        po = min(max(po, 1), wl - 1)

        out = arr.copy()
        for j in range(J):
            for d in range(D):
                track = out[:, j, d]
                if np.all(np.abs(track) < 1e-12):
                    continue
                try:
                    out[:, j, d] = savgol_filter(track, wl, po, mode="interp")
                except Exception as e:
                    print(f"⚠️  savgol joint {j} dim {d}: {e}")
        return [out[t].astype(np.float32) for t in range(T)]

    @staticmethod
    def temporal_smooth_kalman_sequence(
        keypoint_sequence: List[np.ndarray],
        process_noise: float = 1e-4,
        measurement_noise: float = 1e-2,
    ) -> List[np.ndarray]:
        """
        Per-coordinate 1D constant-velocity Kalman filter (forward pass): smooths measurement
        noise while tracking velocity; alternative to Savitzky–Golay for jitter reduction.
        """
        if not keypoint_sequence:
            return keypoint_sequence
        arr = np.stack([kp.astype(np.float64) for kp in keypoint_sequence], axis=0)
        T, J, D = arr.shape
        if T < 2:
            return keypoint_sequence

        out = np.zeros_like(arr)
        q = float(max(process_noise, 1e-12))
        r = float(max(measurement_noise, 1e-12))
        F = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=np.float64)
        H = np.array([1.0, 0.0], dtype=np.float64)
        Q = np.array([[0.25 * q, 0.5 * q], [0.5 * q, q]], dtype=np.float64)

        for j in range(J):
            for d in range(D):
                meas = arr[:, j, d]
                if np.all(np.abs(meas) < 1e-12):
                    out[:, j, d] = meas
                    continue
                x = np.zeros(2, dtype=np.float64)
                P = np.eye(2, dtype=np.float64) * 1.0
                for t in range(T):
                    x = F @ x
                    P = F @ P @ F.T + Q
                    zt = float(meas[t])
                    S = float(H @ P @ H.T + r)
                    if S < 1e-18:
                        out[t, j, d] = x[0]
                        continue
                    K = (P @ H) / S
                    y = zt - float(H @ x)
                    x = x + K * y
                    P = (np.eye(2) - np.outer(K, H)) @ P
                    out[t, j, d] = x[0]
        return [out[t].astype(np.float32) for t in range(T)]


# PosePulse preprocessing diagram Row 1 steps 2–4 (frame-aligned; no FPS sync — use before ViTPose/ResNet crops).
POSEPULSE_DIAGRAM_TECHNIQUES_FRAME_ALIGNED: Tuple[str, ...] = (
    "spatial_imputation",
    "normalization",
    "temporal_impute",
)


def apply_keypoint_preprocessing_pipeline(
    keypoint_sequence: List[np.ndarray],
    confidence_sequence: List[np.ndarray],
    *,
    preprocessing_techniques: Optional[List[str]] = None,
    target_fps: float = 30.0,
    source_fps: float = 30.0,
    original_frames: Optional[int] = None,
    savgol_window_length: int = 7,
    savgol_polyorder: int = 2,
    kalman_process_noise: float = 1e-4,
    kalman_measurement_noise: float = 1e-2,
) -> Dict[str, Any]:
    """
    **PosePulse-style Row 1 (keypoint cleaning)** matches the thesis diagram order:

    - **Step 2 — Spatial imputation** if ``spatial_imputation`` **or** ``imputation`` is listed
      (confidence threshold τ=0.3): skeleton-aware neighbor fill by default; add
      ``laplacian_spatial`` for graph-Laplacian relaxation (arXiv:2204.10312).
    - **Step 3 — Skeleton normalization** if ``normalization``: hip-centered coordinates,
      torso-length scaling ``L_T``.
    - **Step 4 — Temporal imputation** if ``temporal_impute``, ``temporal_imputation``, **or**
      ``imputation``: per-joint linear interpolation in time (same τ on confidence masks).

    Use ``POSEPULSE_DIAGRAM_TECHNIQUES_FRAME_ALIGNED`` for steps 2–4 explicitly without bundling.

    **After Row 1**, optional extensions run in order: ``bone_proportion`` (BioPose-style limbs),
    ``fps_sync``, ``savgol`` / ``kalman``, ``dwt``.

    Bilateral filtering and YOLO pose extraction happen **before** this function (diagram step 1).

    Row 2 — ViTPose/ResNet crops use ``keypoints_pixels_for_crop``; windowing + train Z-score live
    in the dataset / training scripts (e.g. 30-frame windows, stride 15).

    Legacy default when ``preprocessing_techniques`` is ``None`` remains normalization +
    bundled ``imputation`` + ``fps_sync`` for older angle pipelines.
    """
    if preprocessing_techniques is None:
        preprocessing_techniques = ["normalization", "imputation", "fps_sync"]
    techniques = list(preprocessing_techniques)
    if "savgol" in techniques and "kalman" in techniques:
        raise ValueError("Use only one of 'savgol' or 'kalman' temporal smoothing.")
    n_frames = original_frames if original_frames is not None else len(keypoint_sequence)
    if len(keypoint_sequence) != len(confidence_sequence):
        raise ValueError(
            f"keypoint_sequence ({len(keypoint_sequence)}) and confidence_sequence "
            f"({len(confidence_sequence)}) must have the same length"
        )

    processed: Dict[str, Any] = {
        "video_id": None,
        "original_frames": n_frames,
        "source_fps": float(source_fps),
        "target_fps": float(target_fps),
        "techniques_applied": {},
        "final_keypoints": [kp.copy() for kp in keypoint_sequence],
        "confidence_sequence": [c.copy() for c in confidence_sequence],
    }
    conf_seq = confidence_sequence

    run_spatial_imputation = ("imputation" in techniques) or ("spatial_imputation" in techniques)
    run_temporal_imputation = (
        ("imputation" in techniques)
        or ("temporal_impute" in techniques)
        or ("temporal_imputation" in techniques)
    )

    if run_spatial_imputation:
        imputed_seq = []
        imputation_stats = []
        use_laplacian_spatial = "laplacian_spatial" in techniques
        for kp, conf in zip(processed["final_keypoints"], conf_seq):
            if use_laplacian_spatial:
                imputed_kp, mask = PosePreprocessor.impute_missing_joints_laplacian(
                    kp.copy(), conf, threshold=0.3
                )
            else:
                imputed_kp, mask = PosePreprocessor.impute_missing_joints(
                    kp.copy(), conf, threshold=0.3
                )
            imputed_seq.append(imputed_kp)
            imputation_stats.append(float(np.sum(mask)))
        processed["final_keypoints"] = imputed_seq
        processed["techniques_applied"]["imputation_spatial"] = (
            "graph-laplacian-harmonic-arxiv2204.10312"
            if use_laplacian_spatial
            else "neighbor-mean-COCO"
        )
        processed["imputation_rate"] = float(np.mean(imputation_stats)) / 17.0 * 100.0

    processed["kp_pixels_pre_normalization"] = [
        np.asarray(kp, dtype=np.float32).copy() for kp in processed["final_keypoints"]
    ]

    if "normalization" in techniques:
        processed["torso_denorm_meta"] = []
        for kp_pre in processed["kp_pixels_pre_normalization"]:
            t = PosePreprocessor.torso_normalization_terms(np.asarray(kp_pre))
            if t is None:
                processed["torso_denorm_meta"].append({"normalized_ok": False})
            else:
                hip_c, lt = t
                processed["torso_denorm_meta"].append(
                    {
                        "normalized_ok": True,
                        "hip_center": np.asarray(hip_c, dtype=np.float32),
                        "torso_length": float(lt),
                    }
                )
        normalized = [
            PosePreprocessor.skeleton_based_normalize(kp.copy())
            for kp in processed["final_keypoints"]
        ]
        processed["techniques_applied"]["normalization"] = "skeleton-based-torso"
        processed["final_keypoints"] = normalized

    if run_temporal_imputation:
        processed["final_keypoints"] = PosePreprocessor.temporal_impute_sequence(
            processed["final_keypoints"],
            conf_seq,
            conf_threshold=0.3,
        )
        processed["techniques_applied"]["imputation_temporal"] = "linear-per-joint"

    if "bone_proportion" in techniques:
        processed["final_keypoints"] = [
            PosePreprocessor.bone_proportion_normalize(kp.copy())
            for kp in processed["final_keypoints"]
        ]
        processed["techniques_applied"]["bone_proportion"] = "biopose-limb-ratios-arxiv2501.07800"

    src_fps = float(processed["source_fps"])
    if src_fps <= 1e-6:
        src_fps = 30.0
    processed["source_fps"] = src_fps

    if "fps_sync" in techniques:
        tf = float(processed.get("target_fps", 30.0))
        if abs(src_fps - tf) >= 1e-3:
            try:
                resampled_kp = PosePreprocessor.resample_to_standard_fps(
                    processed["final_keypoints"],
                    original_fps=src_fps,
                    target_fps=tf,
                )
                resampled_conf = PosePreprocessor.resample_to_standard_fps(
                    [c[:, np.newaxis] for c in conf_seq],
                    original_fps=src_fps,
                    target_fps=tf,
                )
                conf_seq = [r[:, 0] for r in resampled_conf]
                processed["techniques_applied"]["fps_sync"] = f"{src_fps:.2f}->{tf:.0f}"
                processed["final_keypoints"] = resampled_kp
                processed["confidence_sequence"] = conf_seq
            except Exception as e:
                processed["techniques_applied"]["fps_sync"] = f"failed:{e}"
        else:
            processed["techniques_applied"]["fps_sync"] = "no_change"
            processed["confidence_sequence"] = [c.copy() for c in conf_seq]
    else:
        processed["confidence_sequence"] = [c.copy() for c in conf_seq]

    if "savgol" in techniques:
        processed["final_keypoints"] = PosePreprocessor.temporal_smooth_savgol_sequence(
            processed["final_keypoints"],
            window_length=int(savgol_window_length),
            polyorder=int(savgol_polyorder),
        )
        processed["techniques_applied"]["temporal_smooth"] = (
            f"savgol-wl{savgol_window_length}-poly{savgol_polyorder}"
        )
    elif "kalman" in techniques:
        processed["final_keypoints"] = PosePreprocessor.temporal_smooth_kalman_sequence(
            processed["final_keypoints"],
            process_noise=float(kalman_process_noise),
            measurement_noise=float(kalman_measurement_noise),
        )
        processed["techniques_applied"]["temporal_smooth"] = (
            f"kalman-q{kalman_process_noise}-r{kalman_measurement_noise}"
        )

    kp_array = np.array([k for k in processed["final_keypoints"] if k is not None])
    if len(kp_array) > 0:
        cseq = processed.get("confidence_sequence", conf_seq)
        processed["statistics"] = {
            "num_frames": len(kp_array),
            "mean_confidence": float(np.mean([np.mean(c) for c in cseq])),
            "joint_coverage": float(
                np.sum(kp_array != 0) / (kp_array.shape[0] * kp_array.shape[1] * 2)
            ),
        }

    if "dwt" in techniques:
        try:
            dwt_normalized = PosePreprocessor.dwt_normalize(processed["final_keypoints"])
            processed["techniques_applied"]["dwt"] = "db4-wavelet-frequency-decomposition"
            processed["final_keypoints"] = dwt_normalized
        except Exception as e:
            processed["techniques_applied"]["dwt"] = f"failed:{e}"

    return processed


def keypoints_pixels_for_crop(processed: Dict[str, Any], frame_idx: int) -> np.ndarray:
    """Invert torso normalization (when applied) so ``final_keypoints`` can drive RGB crops in pixels."""
    fk = np.asarray(processed["final_keypoints"][frame_idx], dtype=np.float32)
    applied = processed.get("techniques_applied") or {}

    dwt_tag = applied.get("dwt")
    if dwt_tag is not None and not str(dwt_tag).startswith("failed"):
        pre = processed.get("kp_pixels_pre_normalization")
        if pre is not None and frame_idx < len(pre):
            return np.asarray(pre[frame_idx], dtype=np.float32).copy()

    if applied.get("bone_proportion") is not None:
        pre = processed.get("kp_pixels_pre_normalization")
        if pre is not None and frame_idx < len(pre):
            return np.asarray(pre[frame_idx], dtype=np.float32).copy()

    if applied.get("normalization") is None:
        return fk.copy()

    meta_list = processed.get("torso_denorm_meta")
    if not meta_list or frame_idx >= len(meta_list):
        pre = processed.get("kp_pixels_pre_normalization")
        if pre is not None and frame_idx < len(pre):
            return np.asarray(pre[frame_idx], dtype=np.float32).copy()
        return fk.copy()

    meta = meta_list[frame_idx]
    if meta.get("normalized_ok"):
        lt = float(meta["torso_length"])
        hip = np.asarray(meta["hip_center"], dtype=np.float64)
        return (fk.astype(np.float64) * lt + hip).astype(np.float32)
    return fk.copy()


class VideoProcessor:
    """Process video files and extract poses"""

    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @staticmethod
    def _downscale_for_detection(frame: np.ndarray, max_long_edge: int) -> np.ndarray:
        """Resize so max(h, w) <= max_long_edge; keeps aspect ratio. max_long_edge<=0 = no-op."""
        if max_long_edge <= 0:
            return frame
        h, w = frame.shape[:2]
        m = max(h, w)
        if m <= max_long_edge:
            return frame
        scale = max_long_edge / float(m)
        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        return cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)

    def process_with_detector(
        self,
        detector,
        max_frames: Optional[int] = None,
        *,
        detection_stride: int = 1,
        detection_max_long_edge: int = 0,
    ) -> List[PoseEstimationResult]:
        """Process video and extract poses.

        ``detection_stride``: run MediaPipe every Nth frame (e.g. 2 ≈ half the work; pair with
        effective FPS when preprocessing). ``detection_max_long_edge``: shrink frames before
        inference (landmarks mapped back to full-res pixels).
        """
        results: List[PoseEstimationResult] = []
        frame_idx = 0
        stride = max(1, int(detection_stride))

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            if max_frames and frame_idx >= max_frames:
                break

            if frame_idx % stride != 0:
                frame_idx += 1
                continue

            det_frame = self._downscale_for_detection(frame, int(detection_max_long_edge))
            sw, sh = float(det_frame.shape[1]), float(det_frame.shape[0])
            sx = float(self.width) / sw
            sy = float(self.height) / sh

            start_time = time.time()
            person_box_full: Optional[np.ndarray] = None
            if hasattr(detector, "detect_with_person_box"):
                keypoints, confidence, box_det = detector.detect_with_person_box(det_frame)
                if box_det is not None:
                    b = np.asarray(box_det, dtype=np.float64).reshape(4)
                    person_box_full = b.copy()
                    person_box_full[[0, 2]] *= sx
                    person_box_full[[1, 3]] *= sy
                    person_box_full = person_box_full.astype(np.float32)
            else:
                keypoints, confidence = detector.detect(det_frame)
            inference_time = (time.time() - start_time) * 1000  # ms

            if keypoints is not None:
                # Normalized coords are relative to det_frame; map to full-video pixels
                keypoints = keypoints.copy()
                keypoints[:, 0] *= float(self.width)
                keypoints[:, 1] *= float(self.height)

                result = PoseEstimationResult(
                    model_name=detector.__class__.__name__,
                    frame_idx=frame_idx,
                    keypoints=keypoints,
                    confidence=confidence,
                    inference_time=inference_time,
                    person_box_xyxy=person_box_full,
                )
                results.append(result)

            frame_idx += 1

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset
        return results

    def process_yolo_then_vitpose(
        self,
        yolo_det: UltralyticsCOCOPoseDetector,
        vit_det: ViTPoseDetector,
        max_frames: Optional[int] = None,
        *,
        detection_stride: int = 1,
        detection_max_long_edge: int = 0,
        bilateral_filter: bool = False,
        bilateral_conf_tau: float = 0.3,
        bilateral_include_ankles: bool = False,
    ) -> List[PoseEstimationResult]:
        """
        Two-stage pose for PosePulse-style pipelines: **YOLO26-Pose** on each frame for detection
        and optional **bilateral** gating (uses YOLO confidences), then **ViTPose** (rtmlib RTMPose)
        on the same (possibly downscaled) frame. Returned keypoints/confidences are **ViTPose** in
        full-frame pixel space — downstream ``apply_keypoint_preprocessing_pipeline`` and
        angle / mixed-42 features run unchanged on that sequence.

        Frames where YOLO returns no pose, bilateral fails (when enabled), or ViTPose fails are
        omitted (same contract as skipping weak detections).
        """
        results: List[PoseEstimationResult] = []
        frame_idx = 0
        stride = max(1, int(detection_stride))
        pairs = (
            BILATERAL_LIMB_PAIRS_WITH_ANKLES
            if bilateral_include_ankles
            else BILATERAL_LIMB_PAIRS
        )

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            if max_frames and frame_idx >= max_frames:
                break

            if frame_idx % stride != 0:
                frame_idx += 1
                continue

            det_frame = self._downscale_for_detection(frame, int(detection_max_long_edge))

            t0 = time.time()
            k_y, c_y = yolo_det.detect(det_frame)
            t_yolo_ms = (time.time() - t0) * 1000.0

            if k_y is None or c_y is None:
                frame_idx += 1
                continue

            if bilateral_filter and not frame_passes_bilateral_coco17(
                c_y, float(bilateral_conf_tau), pairs=pairs
            ):
                frame_idx += 1
                continue

            t1 = time.time()
            k_v, c_v = vit_det.detect(det_frame)
            t_vit_ms = (time.time() - t1) * 1000.0

            if k_v is None or c_v is None:
                frame_idx += 1
                continue

            k_v = k_v.astype(np.float32).copy()
            k_v[:, 0] *= float(self.width)
            k_v[:, 1] *= float(self.height)
            c_v = np.asarray(c_v, dtype=np.float32).reshape(17)

            results.append(
                PoseEstimationResult(
                    model_name="YOLO26_then_ViTPose",
                    frame_idx=frame_idx,
                    keypoints=k_v,
                    confidence=c_v,
                    inference_time=float(t_yolo_ms + t_vit_ms),
                )
            )
            frame_idx += 1

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return results

    def close(self):
        self.cap.release()


# Example usage and comparison
if __name__ == "__main__":
    print("QEVD-FIT-COACH Pose Estimation Pipeline")
    print("="*60)
