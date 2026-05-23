#!/usr/bin/env python3
"""
Keras (TensorFlow) clone of ``ExerciseBiLSTMPosePulseDiagramNet`` for **architecture diagrams**.

This does **not** load PyTorch weights; it mirrors layer types and order so ``keras.utils.model_to_dot``
/ ``plot_model`` match the training model in ``fitness_coach.models.exercise_bilstm_model``.

Dependencies::

  pip install tensorflow pydot
  # For PNG/SVG from the same script (optional):
  # macOS: brew install graphviz
  # Ubuntu: sudo apt install graphviz

Examples::

  ./venv/bin/python scripts/visualize_posepulse_diagram_keras.py \\
    --output results/diagrams/posepulse_diagram

  # Writes ``posepulse_diagram.dot``; if Graphviz ``dot`` is found (PATH or Homebrew), also ``.png`` / ``.svg``.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def _find_graphviz_dot() -> str | None:
    """``shutil.which('dot')`` plus common Homebrew locations (Apple Silicon / Intel)."""
    w = shutil.which("dot")
    if w:
        return w
    for p in ("/opt/homebrew/bin/dot", "/usr/local/bin/dot"):
        if Path(p).is_file():
            return p
    return None


def build_posepulse_diagram_keras(
    *,
    sequence_len: int = 30,
    feat_dim: int = 42,
    num_classes: int = 10,
    lstm_hidden: int = 128,
    lstm_dropout: float = 0.2,
):
    """Functional Keras model aligned with ``ExerciseBiLSTMPosePulseDiagramNet`` (channels_last)."""
    try:
        import tensorflow as tf
        from tensorflow.keras import Input, Model, layers
    except ImportError as e:
        raise SystemExit(
            "TensorFlow is required. Install with: pip install tensorflow"
        ) from e

    inp = Input(shape=(sequence_len, feat_dim), name="window_features")
    x = layers.Bidirectional(
        layers.LSTM(lstm_hidden, return_sequences=True),
        name="bilstm_layer1",
    )(inp)
    x = layers.Dropout(lstm_dropout, name="lstm_dropout")(x)
    x = layers.Bidirectional(
        layers.LSTM(lstm_hidden, return_sequences=True),
        name="bilstm_layer2",
    )(x)
    # (B, T, 256) -> (B, T, 1, 256) for Conv2D; matches PyTorch (B, C, T, 1) after transpose.
    x = layers.Reshape((sequence_len, 1, 2 * lstm_hidden), name="reshape_T_1_C")(x)
    x = layers.Conv2D(
        128,
        (3, 1),
        strides=(1, 1),
        padding="same",
        use_bias=True,
        name="conv2d_256_to_128",
    )(x)
    x = layers.BatchNormalization(name="bn_conv1")(x)
    x = layers.Activation("relu", name="relu_conv1")(x)
    x = layers.Conv2D(
        256,
        (3, 1),
        strides=(2, 1),
        padding="same",
        use_bias=True,
        name="conv2d_stride2_time",
    )(x)
    x = layers.BatchNormalization(name="bn_conv2")(x)
    x = layers.Activation("relu", name="relu_conv2")(x)
    x = layers.Conv2D(
        64,
        (3, 1),
        strides=(1, 1),
        padding="same",
        use_bias=True,
        name="conv2d_to_64",
    )(x)
    x = layers.BatchNormalization(name="bn_conv3")(x)
    x = layers.Activation("relu", name="relu_conv3")(x)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    out = layers.Dense(num_classes, name="class_logits")(x)
    return Model(inp, out, name="PosePulseDiagramNet_KerasMirror")


def main() -> int:
    ap = argparse.ArgumentParser(description="Keras diagram for PosePulse diagram BiLSTM–CNN (mirror of PyTorch).")
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("results/diagrams/posepulse_diagram"),
        help="Base path without extension (writes .dot and optionally .png/.svg).",
    )
    ap.add_argument("--sequence-len", type=int, default=30)
    ap.add_argument("--feat-dim", type=int, default=42)
    ap.add_argument("--num-classes", type=int, default=10)
    ap.add_argument("--lstm-hidden", type=int, default=128)
    ap.add_argument("--lstm-dropout", type=float, default=0.2)
    ap.add_argument(
        "--summary",
        action="store_true",
        help="Also print Keras model.summary() to stdout.",
    )
    args = ap.parse_args()

    out_base: Path = args.output.resolve()
    out_base.parent.mkdir(parents=True, exist_ok=True)

    model = build_posepulse_diagram_keras(
        sequence_len=args.sequence_len,
        feat_dim=args.feat_dim,
        num_classes=args.num_classes,
        lstm_hidden=args.lstm_hidden,
        lstm_dropout=args.lstm_dropout,
    )

    if args.summary:
        model.summary(line_length=100)

    import tensorflow as tf

    dot_path = out_base.with_suffix(".dot")
    dot = tf.keras.utils.model_to_dot(
        model,
        show_shapes=True,
        show_layer_names=True,
        rankdir="TB",
        dpi=96,
    )
    dot.write(str(dot_path))
    print(f"Wrote Graphviz DOT (open in a .dot viewer, or render with the graphviz CLI): {dot_path}")

    dot_bin = _find_graphviz_dot()
    if dot_bin:
        png_path = out_base.with_suffix(".png")
        r = subprocess.run(
            [dot_bin, "-Tpng", str(dot_path), "-o", str(png_path)],
            capture_output=True,
            text=True,
        )
        if r.returncode == 0:
            print(f"Wrote PNG: {png_path}")
        else:
            print("graphviz ``dot`` failed:", r.stderr or r.stdout, file=sys.stderr)
    else:
        print(
            "Graphviz ``dot`` not found — skipped PNG/SVG. Install: ``brew install graphviz`` (macOS) "
            "or ``apt install graphviz`` (Linux), then run:\n"
            f"  dot -Tpng {dot_path} -o {out_base.with_suffix('.png')}",
            file=sys.stderr,
        )

    if dot_bin:
        svg_path = out_base.with_suffix(".svg")
        r2 = subprocess.run(
            [dot_bin, "-Tsvg", str(dot_path), "-o", str(svg_path)],
            capture_output=True,
            text=True,
        )
        if r2.returncode == 0:
            print(f"Wrote SVG: {svg_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
