"""
Deterministic feedback templates and pose measurements for exercise coaching.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from fitness_coach.core.biomechanical_features import ANGLE_FEATURE_NAMES, compute_sequence_angles


ANGLE_INDEX = {name: idx for idx, name in enumerate(ANGLE_FEATURE_NAMES)}

EXERCISE_ALIASES = {
    "barbell_squat": "squat",
    "squats": "squat",
    "barbell squat": "squat",
    "pushup": "push_up",
    "push-ups": "push_up",
    "push ups": "push_up",
    "barbell_biceps_curl": "biceps_curl",
    "hammer_curl": "hammer_curl",
    "shoulder_press": "shoulder_press",
}

EXERCISE_TAG_PRIORITY: Dict[str, Tuple[str, ...]] = {
    "squat": (
        "knees_too_far_forward",
        "insufficient_depth",
        "back_not_straight",
        "alignment",
    ),
    "push_up": (
        "hip_position",
        "elbows_flared",
        "incomplete_extension",
        "alignment",
    ),
    "biceps_curl": (
        "range_of_motion",
        "elbows_flared",
        "shoulder_instability",
        "tempo_control",
    ),
    "hammer_curl": (
        "range_of_motion",
        "elbows_flared",
        "shoulder_instability",
        "tempo_control",
    ),
    "shoulder_press": (
        "incomplete_extension",
        "back_not_straight",
        "shoulder_instability",
        "alignment",
    ),
}


def canonicalize_exercise_name(name: str) -> str:
    key = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    return EXERCISE_ALIASES.get(key, key)


def quality_label(score_01: float) -> str:
    score_01 = float(score_01)
    if score_01 < 0.35:
        return "POOR FORM"
    if score_01 < 0.65:
        return "FAIR FORM"
    if score_01 < 0.85:
        return "GOOD FORM"
    return "EXCELLENT FORM"


def _midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (np.asarray(a, dtype=np.float64) + np.asarray(b, dtype=np.float64)) / 2.0


def _bottom_frame_idx(angles: np.ndarray) -> int:
    knee = angles[:, [ANGLE_INDEX["left_knee"], ANGLE_INDEX["right_knee"]]]
    knee = np.nanmean(knee, axis=1)
    return int(np.nanargmin(knee)) if np.isfinite(knee).any() else int(len(angles) // 2)


def _torso_lean_deg(frame_kp: np.ndarray) -> float:
    hip_mid = _midpoint(frame_kp[11], frame_kp[12])
    sh_mid = _midpoint(frame_kp[5], frame_kp[6])
    vec = sh_mid - hip_mid
    if np.linalg.norm(vec) < 1e-6:
        return 0.0
    # 0 deg = upright vertical line
    return abs(math.degrees(math.atan2(float(vec[0]), float(-vec[1]) + 1e-6)))


def _body_line_break_deg(frame_kp: np.ndarray) -> float:
    sh_mid = _midpoint(frame_kp[5], frame_kp[6])
    hip_mid = _midpoint(frame_kp[11], frame_kp[12])
    ank_mid = _midpoint(frame_kp[15], frame_kp[16])
    upper = sh_mid - hip_mid
    lower = ank_mid - hip_mid
    n1 = np.linalg.norm(upper)
    n2 = np.linalg.norm(lower)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cosang = float(np.clip(np.dot(upper, lower) / (n1 * n2), -1.0, 1.0))
    return abs(180.0 - math.degrees(math.acos(cosang)))


def _knee_forward_proxy_deg(frame_kp: np.ndarray) -> float:
    left = abs(float(frame_kp[13, 0] - frame_kp[15, 0]))
    right = abs(float(frame_kp[14, 0] - frame_kp[16, 0]))
    left_depth = abs(float(frame_kp[13, 1] - frame_kp[15, 1])) + 1e-6
    right_depth = abs(float(frame_kp[14, 1] - frame_kp[16, 1])) + 1e-6
    left_angle = math.degrees(math.atan2(left, left_depth))
    right_angle = math.degrees(math.atan2(right, right_depth))
    return float((left_angle + right_angle) / 2.0)


def _mean_angle(angles: np.ndarray, left_name: str, right_name: str, frame_idx: Optional[int] = None) -> float:
    if frame_idx is None:
        values = angles[:, [ANGLE_INDEX[left_name], ANGLE_INDEX[right_name]]]
        return float(np.nanmean(values))
    values = angles[frame_idx, [ANGLE_INDEX[left_name], ANGLE_INDEX[right_name]]]
    return float(np.nanmean(values))


def compute_rule_measurements(exercise: str, keypoints: np.ndarray, angles: Optional[np.ndarray] = None) -> Dict[str, Dict[str, float]]:
    exercise = canonicalize_exercise_name(exercise)
    keypoints = np.asarray(keypoints, dtype=np.float64)
    if angles is None:
        angles, _ = compute_sequence_angles(keypoints)
    bottom_idx = _bottom_frame_idx(angles)
    measurements: Dict[str, Dict[str, float]] = {}

    if exercise == "squat":
        knee_bottom = _mean_angle(angles, "left_knee", "right_knee", bottom_idx)
        hip_bottom = _mean_angle(angles, "left_hip", "right_hip", bottom_idx)
        torso_lean = _torso_lean_deg(keypoints[bottom_idx])
        knee_forward = _knee_forward_proxy_deg(keypoints[bottom_idx])
        measurements["insufficient_depth"] = {
            "measured": knee_bottom,
            "target_min": 80.0,
            "target_max": 110.0,
            "error_deg": max(0.0, knee_bottom - 110.0),
        }
        measurements["knees_too_far_forward"] = {
            "measured": knee_forward,
            "target_min": 0.0,
            "target_max": 18.0,
            "error_deg": max(0.0, knee_forward - 18.0),
        }
        measurements["back_not_straight"] = {
            "measured": torso_lean,
            "target_min": 0.0,
            "target_max": 25.0,
            "error_deg": max(0.0, torso_lean - 25.0),
        }
        measurements["alignment"] = {
            "measured": hip_bottom,
            "target_min": 55.0,
            "target_max": 120.0,
            "error_deg": max(0.0, 55.0 - hip_bottom),
        }
        return measurements

    if exercise == "push_up":
        elbow_bottom = _mean_angle(angles, "left_elbow", "right_elbow", bottom_idx)
        line_break = _body_line_break_deg(keypoints[bottom_idx])
        extension = _mean_angle(angles, "left_elbow", "right_elbow", None)
        shoulder = _mean_angle(angles, "left_shoulder", "right_shoulder", bottom_idx)
        measurements["hip_position"] = {
            "measured": line_break,
            "target_min": 0.0,
            "target_max": 20.0,
            "error_deg": max(0.0, line_break - 20.0),
        }
        measurements["elbows_flared"] = {
            "measured": shoulder,
            "target_min": 35.0,
            "target_max": 80.0,
            "error_deg": max(0.0, shoulder - 80.0),
        }
        measurements["incomplete_extension"] = {
            "measured": extension,
            "target_min": 150.0,
            "target_max": 180.0,
            "error_deg": max(0.0, 150.0 - extension),
        }
        measurements["alignment"] = measurements["hip_position"]
        return measurements

    if exercise in {"biceps_curl", "hammer_curl"}:
        elbow_series = np.nanmean(angles[:, [ANGLE_INDEX["left_elbow"], ANGLE_INDEX["right_elbow"]]], axis=1)
        rom = float(np.nanmax(elbow_series) - np.nanmin(elbow_series))
        shoulder_var = float(np.nanstd(np.nanmean(angles[:, [ANGLE_INDEX["left_shoulder"], ANGLE_INDEX["right_shoulder"]]], axis=1)))
        measurements["range_of_motion"] = {
            "measured": rom,
            "target_min": 70.0,
            "target_max": 180.0,
            "error_deg": max(0.0, 70.0 - rom),
        }
        measurements["elbows_flared"] = {
            "measured": shoulder_var,
            "target_min": 0.0,
            "target_max": 12.0,
            "error_deg": max(0.0, shoulder_var - 12.0),
        }
        measurements["shoulder_instability"] = measurements["elbows_flared"]
        measurements["tempo_control"] = {
            "measured": float(len(elbow_series)),
            "target_min": 40.0,
            "target_max": 160.0,
            "error_deg": 0.0,
        }
        return measurements

    if exercise == "shoulder_press":
        elbow_peak = float(
            np.nanmax(np.nanmean(angles[:, [ANGLE_INDEX["left_elbow"], ANGLE_INDEX["right_elbow"]]], axis=1))
        )
        torso_lean = float(np.nanmean([_torso_lean_deg(frame) for frame in keypoints]))
        shoulder_peak = float(
            np.nanmax(np.nanmean(angles[:, [ANGLE_INDEX["left_shoulder"], ANGLE_INDEX["right_shoulder"]]], axis=1))
        )
        measurements["incomplete_extension"] = {
            "measured": elbow_peak,
            "target_min": 160.0,
            "target_max": 180.0,
            "error_deg": max(0.0, 160.0 - elbow_peak),
        }
        measurements["back_not_straight"] = {
            "measured": torso_lean,
            "target_min": 0.0,
            "target_max": 18.0,
            "error_deg": max(0.0, torso_lean - 18.0),
        }
        measurements["shoulder_instability"] = {
            "measured": shoulder_peak,
            "target_min": 140.0,
            "target_max": 180.0,
            "error_deg": max(0.0, 140.0 - shoulder_peak),
        }
        measurements["alignment"] = measurements["back_not_straight"]
        return measurements

    return {
        "alignment": {
            "measured": 0.0,
            "target_min": 0.0,
            "target_max": 0.0,
            "error_deg": 0.0,
        }
    }


def _message_for_tag(exercise: str, tag: str, measurement: Dict[str, float]) -> str:
    err = measurement.get("error_deg", 0.0)
    err_text = f"{err:.0f}\N{DEGREE SIGN} error"
    if tag == "knees_too_far_forward":
        return f"Knees too far forward ({err_text}). Keep knees over toes."
    if tag == "insufficient_depth":
        return f"Depth is too shallow ({err_text}). Sit lower while keeping control."
    if tag == "back_not_straight":
        return f"Back angle is off ({err_text}). Brace the core and keep the torso more neutral."
    if tag == "hip_position":
        return f"Hip line is breaking ({err_text}). Keep shoulders, hips, and ankles in one line."
    if tag == "elbows_flared":
        return f"Elbows are drifting out ({err_text}). Keep them closer to the intended path."
    if tag == "incomplete_extension":
        return f"Extension is incomplete ({err_text}). Finish the rep with full lockout."
    if tag == "range_of_motion":
        return f"Range of motion is limited ({err_text}). Move through a fuller controlled rep."
    if tag == "shoulder_instability":
        return f"Shoulders are unstable ({err_text}). Keep the shoulders packed and controlled."
    return f"Form needs adjustment ({err_text}). Focus on controlled alignment."


def select_feedback(
    exercise: str,
    quality_score: float,
    keypoints: np.ndarray,
    angles: Optional[np.ndarray] = None,
    error_tag_names: Optional[Sequence[str]] = None,
    error_probabilities: Optional[Sequence[float]] = None,
) -> Dict[str, object]:
    canonical_exercise = canonicalize_exercise_name(exercise)
    measurements = compute_rule_measurements(canonical_exercise, keypoints, angles=angles)
    priorities = EXERCISE_TAG_PRIORITY.get(canonical_exercise, ("alignment",))

    tag_scores = {tag: 0.0 for tag in measurements.keys()}
    if error_tag_names is not None and error_probabilities is not None:
        for tag, prob in zip(error_tag_names, error_probabilities):
            if tag in tag_scores:
                tag_scores[tag] = float(prob)

    selected_tag = None
    best_score = -1.0
    for tag in priorities:
        if tag not in measurements:
            continue
        score = tag_scores.get(tag, 0.0) + 0.01 * measurements[tag].get("error_deg", 0.0)
        if score > best_score:
            best_score = score
            selected_tag = tag
    if selected_tag is None:
        selected_tag = next(iter(measurements.keys()))

    measurement = measurements[selected_tag]
    return {
        "exercise": canonical_exercise,
        "quality_label": quality_label(float(quality_score)),
        "selected_tag": selected_tag,
        "measurement": measurement,
        "feedback": _message_for_tag(canonical_exercise, selected_tag, measurement),
    }
