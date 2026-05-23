#!/usr/bin/env python3
"""Shim — implementation: fitness_coach.training.train_exercise_stgcn"""
import sys
from pathlib import Path

# Add workspace root to sys.path for editable install compatibility
workspace_root = Path(__file__).parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

from fitness_coach.training.train_exercise_stgcn import main
if __name__ == "__main__":
    raise SystemExit(main())
