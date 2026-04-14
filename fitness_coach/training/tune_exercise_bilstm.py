#!/usr/bin/env python3
"""
Optuna (TPE) hyperparameter search for the exercise BiLSTM trainer.

Search ranges are centered on Riccio et al. (arXiv:2411.11548) Table 4 BiLSTM defaults
(hidden 73, dropout ~0.22, lr 4e-4, batch 54) but explore nearby values like a paper-style
grid/random search, using validation classification accuracy as the objective.

Requires: pip install optuna

Example:
  ./venv/bin/python tune_exercise_bilstm.py --standardize --n-trials 40 --tune-epochs 12 \\
    --kaggle-angles-dir results/riccio_realtime_exercise_recognition \\
    --kaggle-stem riccio_realtime_exercise_recognition

  ./venv/bin/python tune_exercise_bilstm.py --narrow-space --n-trials 25 --tune-epochs 15 \\
    --output-dir results/exercise_bilstm_tune

  # Table 3-style search + CNN/Attention BiLSTM + classification-only (longer trials):
  ./venv/bin/python tune_exercise_bilstm.py --search-table3 --n-trials 20 \\
    --kaggle-angles-dir results/riccio_realtime_exercise_recognition \\
    --kaggle-stem riccio_realtime_exercise_recognition \\
    --architecture cnn_attn --classification-only --output-dir results/exercise_bilstm_tune_t3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from fitness_coach.training.train_exercise_bilstm import (
    add_bilstm_train_args,
    apply_text_coaching_supervision,
    build_bilstm_train_context,
    compute_inverse_frequency_class_weights,
    evaluate,
    run_bilstm_training,
)


def suggest_hyperparams(trial, narrow: bool) -> dict:
    """Riccio Table 4–style search: LSTM width, dropout, lr, batch, depth, AdamW decay."""
    if narrow:
        hidden = trial.suggest_int("hidden", 56, 96, step=8)
        dropout = trial.suggest_float("dropout", 0.15, 0.32)
        lr = trial.suggest_float("lr", 8e-5, 8e-4, log=True)
        batch_size = trial.suggest_categorical("batch_size", [40, 48, 54, 64])
    else:
        hidden = trial.suggest_int("hidden", 32, 128, step=8)
        dropout = trial.suggest_float("dropout", 0.1, 0.45)
        lr = trial.suggest_float("lr", 1e-5, 3e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [24, 32, 48, 54, 64, 96])
    layers = trial.suggest_int("layers", 1, 2)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    return {
        "hidden": hidden,
        "dropout": dropout,
        "lr": lr,
        "batch_size": batch_size,
        "layers": layers,
        "weight_decay": weight_decay,
    }


def suggest_hyperparams_table3(trial) -> dict:
    """Table 3-style ranges: units 50–150, dropout 0.2–0.5, lr 1e-4–1e-3 (uniform), batch 32–64, epochs 50–100."""
    return {
        "hidden": trial.suggest_int("hidden", 50, 150),
        "dropout": trial.suggest_float("dropout", 0.2, 0.5),
        "lr": trial.suggest_float("lr", 1e-4, 1e-3),
        "batch_size": trial.suggest_int("batch_size", 32, 64),
        "epochs": trial.suggest_int("epochs", 50, 100),
        "layers": trial.suggest_int("layers", 1, 2),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
    }


def main() -> int:
    try:
        import optuna
        from optuna.exceptions import TrialPruned
    except ImportError:
        print(
            "Install Optuna: pip install optuna",
            file=sys.stderr,
        )
        return 1

    ap = argparse.ArgumentParser(
        description="Optuna hyperparameter search for BiLSTM (Riccio-style ranges, val acc objective)"
    )
    add_bilstm_train_args(ap)
    ap.add_argument(
        "--tune-epochs",
        type=int,
        default=15,
        help="Training epochs per Optuna trial (keep small for fast search).",
    )
    ap.add_argument("--epochs", type=int, default=30, help="Epochs for final retrain after search.")
    ap.add_argument("--n-trials", type=int, default=30)
    ap.add_argument("--tune-seed", type=int, default=42, help="Optuna + PyTorch sampler seed.")
    ap.add_argument(
        "--narrow-space",
        action="store_true",
        help="Search in a band around Table 4 (73 units, ~0.22 dropout, ~4e-4 lr, batch ~54).",
    )
    ap.add_argument(
        "--storage",
        default="",
        help="Optuna storage URL, e.g. sqlite:///results/bilstm_study.db (empty = in-memory).",
    )
    ap.add_argument("--study-name", default="bilstm_exercise")
    ap.add_argument(
        "--no-pruning",
        action="store_true",
        help="Disable median pruning (use if val set is tiny).",
    )
    ap.add_argument(
        "--skip-retrain",
        action="store_true",
        help="Do not train a final model with best hyperparameters after the study.",
    )
    ap.add_argument(
        "--eval-test",
        action="store_true",
        help="After final retrain, evaluate on Kaggle test split (same as train_exercise_bilstm.py).",
    )
    ap.add_argument(
        "--search-table3",
        action="store_true",
        help="Use Table 3 hyperparameter ranges (units50–150, dropout 0.2–0.5, lr uniform 1e-4–1e-3, batch 32–64, epochs 50–100 per trial).",
    )
    args = ap.parse_args()

    # Do not apply --preset here: search explores hyperparameters (use --narrow-space for Table 4–local search).
    if getattr(args, "classification_only", False):
        args.reg_weight = 0.0

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    ctx = build_bilstm_train_context(args)
    if ctx is None:
        return 1
    text_dim, text_enc = apply_text_coaching_supervision(ctx, args)
    ce_w = None
    if getattr(args, "balanced_class_weights", False):
        ce_w = compute_inverse_frequency_class_weights(ctx)
        print(
            "Balanced CE weights: "
            + ", ".join(f"{ctx.classes[i]}={ce_w[i].item():.3f}" for i in range(len(ctx.classes)))
        )
    if len(ctx.val_ds) == 0:
        print(
            "Hyperparameter search needs a non-empty validation split. "
            "Use Kaggle mode with val_ratio > 0 or an index with split=val.",
            file=sys.stderr,
        )
        return 1

    sampler = optuna.samplers.TPESampler(seed=args.tune_seed)
    pruner = (
        optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
        if not args.no_pruning
        else optuna.pruners.NopPruner()
    )
    storage = args.storage.strip()
    if storage:
        study = optuna.create_study(
            study_name=args.study_name,
            storage=storage,
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
            load_if_exists=True,
        )
    else:
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
        )

    arch = getattr(args, "architecture", "plain")
    cnn_h = getattr(args, "cnn_hidden", 64)
    attn_h = getattr(args, "attn_heads", 4)

    def objective(trial: optuna.Trial) -> float:
        if args.search_table3:
            hp = suggest_hyperparams_table3(trial)
            trial_epochs = int(hp["epochs"])
        else:
            hp = suggest_hyperparams(trial, narrow=args.narrow_space)
            trial_epochs = args.tune_epochs
        try:
            res = run_bilstm_training(
                ctx,
                hidden=hp["hidden"],
                layers=hp["layers"],
                dropout=hp["dropout"],
                lr=hp["lr"],
                batch_size=hp["batch_size"],
                weight_decay=hp["weight_decay"],
                epochs=trial_epochs,
                cls_weight=args.cls_weight,
                reg_weight=args.reg_weight,
                device=device,
                out_dir=None,
                save_checkpoint=False,
                verbose=False,
                trial=trial,
                text_dim=text_dim,
                text_coaching_encoder=None,
                architecture=arch,
                cnn_hidden=cnn_h,
                attn_heads=attn_h,
                ce_class_weights=ce_w,
            )
        except TrialPruned:
            raise
        return float(res["best_val_acc"])

    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    best = study.best_trial
    best_params = dict(best.params)
    print("\nBest trial:")
    print(f"  value (val acc): {best.value:.6f}")
    print(f"  params: {json.dumps(best_params, indent=2)}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "best_val_acc": best.value,
        "best_params": best_params,
        "n_trials": args.n_trials,
        "tune_epochs": args.tune_epochs,
        "narrow_space": args.narrow_space,
        "search_table3": args.search_table3,
        "architecture": arch,
        "study_name": args.study_name,
    }
    with open(out_dir / "bilstm_tune_best.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {out_dir / 'bilstm_tune_best.json'}")

    if args.skip_retrain:
        return 0

    p = best_params
    final_epochs = int(p["epochs"]) if args.search_table3 and "epochs" in p else args.epochs
    res = run_bilstm_training(
        ctx,
        hidden=int(p["hidden"]),
        layers=int(p["layers"]),
        dropout=float(p["dropout"]),
        lr=float(p["lr"]),
        batch_size=int(p["batch_size"]),
        weight_decay=float(p["weight_decay"]),
        epochs=final_epochs,
        cls_weight=args.cls_weight,
        reg_weight=args.reg_weight,
        device=device,
        out_dir=out_dir,
        save_checkpoint=True,
        verbose=True,
        trial=None,
        text_dim=text_dim,
        text_coaching_encoder=text_enc,
        architecture=arch,
        cnn_hidden=cnn_h,
        attn_heads=attn_h,
        ce_class_weights=ce_w,
    )
    ckpt_path = out_dir / "exercise_bilstm_best.pt"
    print(f"Retrained best model: val acc ≈ {res['best_val_acc']:.4f}; checkpoint: {ckpt_path}")

    if args.eval_test and ctx.kaggle_dir and res.get("test_loader") is not None and ckpt_path.is_file():
        model = res["model"]
        try:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        ta, trmse, treg = evaluate(model, res["test_loader"], device, len(ctx.classes))
        print(
            f"Test acc={ta:.4f}  quality RMSE={trmse:.4f}  MAE={treg['mae']:.4f}  R²={treg['r2']:.4f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
