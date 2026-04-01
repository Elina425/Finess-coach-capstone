"""
Map QEVD fine-grained labels to exercise family names and weak quality targets.

fine_grained_labels.json uses strings like:
  "elbow plank - stopping early"
  "alternating v ups - rom=5"

We take the **base exercise** as the substring before the first \" - \" (trimmed, lowercased for keys).
Quality is a continuous proxy in [0, 1] derived from form cues + coaching feedback count
(not human ratings — replace with real scores when available).
"""

from __future__ import annotations

import re
from typing import List, Optional

import numpy as np

# Phrases that usually indicate form issues (reduce score)
_NEGATIVE_CUES = (
    "stopping early",
    "too wide",
    "too narrow",
    "shaking",
    "wrong",
    "too low",
    "too high",
    "not reaching",
    "touching wrong",
    "off the ground",
    "on the floor",
    "arms facing inward",
    "head on floor",
    "butt off",
    "knees close",
    "low knees",
    "arms too narrow",
    "arms still",
    "no reach",
)

# Mild positives in the label text (small bump)
_POSITIVE_CUES = (
    "as fast as possible",
    "straight",
    "rom=5",
    "rom=4",
    "clockwise",
    "counterclockwise",
)


def parse_base_exercise(label: str) -> str:
    """First segment before ' - ' is the exercise family name."""
    s = label.strip()
    if " - " in s:
        s = s.split(" - ", 1)[0].strip()
    return s.strip()


def exercise_class_from_labels(labels: List[str]) -> str:
    """
    Primary exercise for the clip: use the base exercise from the **first** label.
    (Multi-label clips usually describe the same exercise with multiple form tags.)
    """
    if not labels:
        return "unknown"
    return parse_base_exercise(labels[0])


def normalize_class_name(name: str) -> str:
    """Stable key for classification: lower, collapse spaces."""
    n = name.lower().strip()
    n = re.sub(r"\s+", " ", n)
    return n


def heuristic_quality_score(
    labels: List[str],
    feedbacks: Optional[List[str]] = None,
) -> float:
    """
    Continuous proxy in [0.05, 0.99] for \"how well performed\".
    Uses label text heuristics + number of coaching feedback lines (more feedback → more issues).
    """
    text = " ".join(labels).lower()
    score = 0.72
    for phrase in _NEGATIVE_CUES:
        if phrase in text:
            score -= 0.09
    for phrase in _POSITIVE_CUES:
        if phrase in text:
            score += 0.04
    # ROM hints embedded like rom=1 (low) vs rom=5 — soft nudge
    roms = re.findall(r"rom=(\d+)", text)
    for r in roms:
        try:
            v = int(r)
            score += 0.02 * (v - 3) / 2.0  # mid ROM neutral
        except ValueError:
            pass
    if feedbacks:
        score -= 0.04 * min(len(feedbacks), 8)
    return float(np.clip(score, 0.05, 0.99))
