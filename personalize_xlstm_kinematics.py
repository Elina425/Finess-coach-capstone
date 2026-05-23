#!/usr/bin/env python3
"""
Personalize an xLSTM exercise classifier to individual kinematics without expert re-labeling.

Uses **LoRA** or **DoRA-style** low-rank adapters (or the legacy **bottleneck** head adapter) on top
of a frozen population checkpoint. Training labels can come from the dataset (e.g. Riccio pose
strings) or from a **teacher** copy of the base model (self-distillation) when you pass
``--distill-teacher``, so new sessions can refine the profile without manual annotation.

Example (Riccio mixed, LoRA, teacher targets):

  python personalize_xlstm_kinematics.py \\
    --base-checkpoint results/xlstm_riccio_mixed/xlstm_keypoints_best.pt \\
    --riccio-data-dir results/riccio_realtime_exercise_recognition \\
    --method lora \\
    --user-id user01 \\
    --distill-teacher \\
    --epochs 12
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from fitness_coach.datasets.exercise_bilstm_dataset import build_kaggle_angle_datasets, build_kaggle_mixed_datasets
from fitness_coach.models.personalization import (
    build_personalization_checkpoint,
    prepare_model_for_personalization,
    trainable_personalization_params,
)
from fitness_coach.models.xlstm_model import xLSTMExerciseClassifier


def _load_base_model(ckpt: dict, device: torch.device) -> xLSTMExerciseClassifier:
    classes = list(ckpt["classes"])
    error_tags = list(ckpt.get("error_tags", []))
    model = xLSTMExerciseClassifier(
        input_size=int(ckpt["input_size"]),
        hidden_size=int(ckpt.get("hidden", 128)),
        num_layers=int(ckpt.get("layers", 2)),
        num_classes=len(classes),
        dropout=float(ckpt.get("dropout", 0.1)),
        bidirectional=bool(ckpt.get("bidirectional", True)),
        num_heads=int(ckpt.get("num_heads", 4)),
        conv_kernel_size=int(ckpt.get("conv_kernel_size", 4)),
        projection_factor=float(ckpt.get("projection_factor", 4.0 / 3.0)),
        num_error_tags=len(error_tags),
        quality_scale=float(ckpt.get("quality_scale", 1.0)),
        linear_classifier=bool(ckpt.get("linear_classifier", False)),
        block_pattern=ckpt.get("block_pattern") if isinstance(ckpt.get("block_pattern"), str) else None,
    )
    model.load_state_dict(ckpt["model"], strict=True)
    return model.to(device)


def _build_riccio_loader(
    data_dir: Path,
    stem: str,
    window: int,
    stride: int,
    feature_mode: str,
    batch_size: int,
) -> DataLoader:
    if feature_mode == "mixed":
        train_ds, _, _, _, _, _, _ = build_kaggle_mixed_datasets(
            data_dir,
            stem=stem,
            window=window,
            stride=stride,
            standardize=True,
        )
    elif feature_mode == "angles":
        train_ds, _, _, _, _, _, _ = build_kaggle_angle_datasets(
            data_dir,
            stem=stem,
            window=window,
            stride=stride,
            standardize=True,
        )
    else:
        raise ValueError(f"Unsupported feature_mode for this script: {feature_mode!r} (use angles or mixed)")

    return DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)


def main() -> int:
    ap = argparse.ArgumentParser(description="Personalize xLSTM with LoRA / DoRA / bottleneck (Riccio kinematics)")
    ap.add_argument("--base-checkpoint", type=Path, required=True)
    ap.add_argument("--riccio-data-dir", type=Path, required=True)
    ap.add_argument("--riccio-stem", type=str, default="riccio_realtime_exercise_recognition")
    ap.add_argument("--user-id", type=str, required=True)
    ap.add_argument("--output-dir", type=Path, default=Path("results/personalization"))
    ap.add_argument(
        "--method",
        choices=("lora", "dora", "bottleneck"),
        default="lora",
        help="Low-rank (LoRA / DoRA-style) or legacy bottleneck adapter on pooled features.",
    )
    ap.add_argument("--lora-rank", type=int, default=8)
    ap.add_argument("--lora-alpha", type=float, default=16.0)
    ap.add_argument(
        "--lora-targets",
        type=str,
        default="input_proj,class_in",
        help="Comma-separated: input_proj, class_in",
    )
    ap.add_argument("--adapter-dim", type=int, default=32, help="Bottleneck dim when --method bottleneck")
    ap.add_argument("--distill-teacher", action="store_true", help="Use frozen teacher argmax as class targets")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--reg-weight", type=float, default=0.5, help="MSE weight on quality head")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    if not args.base_checkpoint.is_file():
        print(f"Missing {args.base_checkpoint}", file=sys.stderr)
        return 1

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    ckpt = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
    feature_mode = str(ckpt.get("feature_mode", "angles"))
    window = int(ckpt.get("window", 30))
    stride = int(ckpt.get("stride", 15))

    teacher = None
    model = _load_base_model(ckpt, device)
    if args.distill_teacher:
        teacher = copy.deepcopy(model)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False

    targets = tuple(s.strip() for s in args.lora_targets.split(",") if s.strip())
    replaced = prepare_model_for_personalization(
        model,
        method=args.method,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_targets=targets,
        adapter_dim=args.adapter_dim if args.method == "bottleneck" else 0,
    )
    print("Personalization modules:", replaced)

    loader = _build_riccio_loader(
        args.riccio_data_dir,
        args.riccio_stem,
        window,
        stride,
        feature_mode,
        args.batch_size,
    )

    opt = torch.optim.AdamW(trainable_personalization_params(model), lr=args.lr, weight_decay=1e-4)
    model.train()

    for epoch in range(1, args.epochs + 1):
        total = 0.0
        n = 0
        for xb, y_cls, y_q in loader:
            xb = xb.to(device)
            y_cls = y_cls.to(device)
            y_q = y_q.to(device)
            if teacher is not None:
                with torch.no_grad():
                    t_out = teacher(xb)
                    t_logits = t_out[0]
                y_cls = t_logits.argmax(dim=1)

            opt.zero_grad()
            out = model(xb)
            logits, pred_q = out[0], out[1]
            pred_q = pred_q.squeeze(-1)
            loss = F.cross_entropy(logits, y_cls) + args.reg_weight * F.mse_loss(pred_q, y_q)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_personalization_params(model), 2.0)
            opt.step()
            total += float(loss.item()) * xb.size(0)
            n += xb.size(0)
        print(f"epoch {epoch:03d} personalization_loss={total / max(n, 1):.4f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"{args.user_id}_kinematics_{args.method}.pt"
    extra = {"replaced_modules": replaced, "distill_teacher": bool(args.distill_teacher)}
    if args.method == "bottleneck":
        extra["adapter_dim"] = int(args.adapter_dim)

    payload = build_personalization_checkpoint(
        user_id=args.user_id,
        base_checkpoint=str(args.base_checkpoint.resolve()),
        method=args.method,
        model=model,
        lora_rank=args.lora_rank if args.method in ("lora", "dora") else None,
        lora_alpha=args.lora_alpha if args.method in ("lora", "dora") else None,
        targets=list(targets) if args.method in ("lora", "dora") else None,
        extra_meta=extra,
    )
    torch.save(payload, out_path)

    meta_path = args.output_dir / f"{args.user_id}_kinematics_{args.method}.json"
    meta_path.write_text(
        json.dumps(
            {
                "user_id": args.user_id,
                "base_checkpoint": str(args.base_checkpoint.resolve()),
                "personalization_checkpoint": str(out_path.resolve()),
                "method": args.method,
                "epochs": args.epochs,
                "distill_teacher": args.distill_teacher,
            },
            indent=2,
        )
    )
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
