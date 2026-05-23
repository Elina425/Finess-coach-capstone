#!/usr/bin/env python3
"""
Export a trained ``exercise_bilstm_best.pt`` checkpoint to **ONNX** for visualization in
`Netron <https://netron.app>`_ (or the Netron desktop app).

Netron does not reliably graph arbitrary PyTorch ``state_dict`` pickles; ONNX is the supported path.
Install once: ``./venv/bin/python -m pip install onnx``.

Example::

  PYTHONPATH=. ./venv/bin/python export_bilstm_onnx_for_netron.py \\
    --checkpoint results/exercise_bilstm/exercise_bilstm_best.pt \\
    --output results/exercise_bilstm/exercise_bilstm_netron.onnx

Or use the end-to-end script::

  ./scripts/train_posepulse_bilstm_export_netron.sh
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn


def _load_ckpt(path: Path) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise ValueError(f"Expected dict with 'model' state_dict, got keys: {list(ckpt) if isinstance(ckpt, dict) else type(ckpt)}")
    return ckpt


class _BiLSTMOnnxWrapper(nn.Module):
    """ONNX: (B,T,F) + (B,text_dim) → logits [+ quality if checkpoint has regression head]."""

    def __init__(self, inner: nn.Module, text_dim: int, *, regression: bool):
        super().__init__()
        self.inner = inner
        self.text_dim = int(text_dim)
        self.regression = bool(regression)

    def forward(self, x: torch.Tensor, text: torch.Tensor):
        o = self.inner(x, text)
        if isinstance(o, tuple) and len(o) == 1:
            return o[0]
        if self.regression and isinstance(o, tuple) and len(o) >= 2:
            return o[0], o[1]
        return o[0] if isinstance(o, tuple) else o


class _BiLSTMOnnxNoText(nn.Module):
    """ONNX when text_dim==0: (B,T,F) → logits [+ quality if model has regression head]."""

    def __init__(self, inner: nn.Module, *, regression: bool):
        super().__init__()
        self.inner = inner
        self.regression = bool(regression)

    def forward(self, x: torch.Tensor):
        o = self.inner(x, None)
        if isinstance(o, tuple) and len(o) == 1:
            return o[0]
        if self.regression and isinstance(o, tuple) and len(o) >= 2:
            return o[0], o[1]
        return o[0] if isinstance(o, tuple) else o


def main() -> int:
    ap = argparse.ArgumentParser(description="Export BiLSTM checkpoint to ONNX for Netron")
    ap.add_argument("--checkpoint", type=Path, required=True, help="exercise_bilstm_best.pt")
    ap.add_argument("--output", type=Path, default=None, help="Output .onnx path (default: next to ckpt)")
    ap.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    args = ap.parse_args()

    ckpt_path = args.checkpoint.resolve()
    if not ckpt_path.is_file():
        print(f"Missing checkpoint: {ckpt_path}", file=sys.stderr)
        return 1

    out_path = args.output.resolve() if args.output else ckpt_path.with_suffix(".onnx")

    from fitness_coach.models.exercise_bilstm_model import build_exercise_bilstm_from_checkpoint

    ckpt = _load_ckpt(ckpt_path)
    text_dim = int(ckpt.get("text_dim", 0))
    model = build_exercise_bilstm_from_checkpoint(ckpt)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    feat_dim = int(ckpt.get("feat_dim", 8))
    window = int(ckpt.get("window", 30))
    batch = 1
    arch = str(ckpt.get("architecture", "plain")).strip().lower()
    # PosePulse-style heads use a fixed T (must match checkpoint ``window``); only batch is dynamic.
    posepulse_like = arch in (
        "posepulse_bilstm_cnn",
        "posepulse",
        "bilstm_cnn_posepulse",
        "posepulse_diagram",
        "posepulse_diagram_cnn",
        "paper_riccio_bilstm_cnn",
        "paper_riccio",
        "riccio_paper_bilstm",
    )
    regression = bool(ckpt.get("has_regression_head", True))

    if text_dim > 0:
        wrapped = _BiLSTMOnnxWrapper(model, text_dim, regression=regression)
        dummy_x = torch.randn(batch, window, feat_dim, dtype=torch.float32)
        dummy_text = torch.randn(batch, text_dim, dtype=torch.float32)
        in_args = (dummy_x, dummy_text)
        input_names = ["window_features", "text_embedding"]
        if posepulse_like:
            dynamic_axes = {
                "window_features": {0: "batch"},
                "text_embedding": {0: "batch"},
            }
        else:
            dynamic_axes = {
                "window_features": {0: "batch", 1: "time"},
                "text_embedding": {0: "batch"},
            }
    else:
        wrapped = _BiLSTMOnnxNoText(model, regression=regression)
        dummy_x = torch.randn(batch, window, feat_dim, dtype=torch.float32)
        in_args = (dummy_x,)
        input_names = ["window_features"]
        if posepulse_like:
            dynamic_axes = {"window_features": {0: "batch"}}
        else:
            dynamic_axes = {"window_features": {0: "batch", 1: "time"}}
    out_names = ["class_logits", "quality"] if regression else ["class_logits"]

    torch.onnx.export(
        wrapped,
        in_args,
        str(out_path),
        input_names=input_names,
        output_names=out_names,
        opset_version=int(args.opset),
        do_constant_folding=True,
        dynamic_axes=dynamic_axes,
        dynamo=False,
    )
    print(f"Wrote ONNX for Netron: {out_path}")
    print("  Open in: https://netron.app  (File → Open) or the Netron desktop app.")
    if text_dim > 0:
        if posepulse_like:
            print(
                f"  Inputs: {input_names}  shapes: (batch, {window}, {feat_dim}) [T fixed], (batch, {text_dim})"
            )
        else:
            print(
                f"  Inputs: {input_names}  shapes: (batch, {window}, {feat_dim}), (batch, {text_dim})"
            )
    else:
        if posepulse_like:
            print(
                f"  Inputs: {input_names}  shape: (batch, {window}, {feat_dim})  (T={window} fixed for posepulse)"
            )
        else:
            print(f"  Inputs: {input_names}  shape: (batch, {window}, {feat_dim})  (dynamic batch and time)")
    print(f"  Architecture: {ckpt.get('architecture', 'plain')!r}  feat_dim={feat_dim}  num_classes={ckpt.get('num_classes')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
