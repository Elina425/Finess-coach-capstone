#!/usr/bin/env python3
"""
Download Riccardo Riccio's real-time exercise recognition dataset (Kaggle) via kagglehub.

After download, run preprocessing + angles for BiLSTM classification::

  ./venv/bin/python kaggle_exercise_recognition_pipeline.py --download --riccio
"""
from __future__ import annotations

import sys


def main() -> int:
    try:
        import kagglehub
    except ImportError as e:
        print("pip install kagglehub", file=sys.stderr)
        raise SystemExit(1) from e

    path = kagglehub.dataset_download("riccardoriccio/real-time-exercise-recognition-dataset")
    print("Path to dataset files:", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
