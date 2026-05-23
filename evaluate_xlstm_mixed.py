#!/usr/bin/env python3
"""
Evaluate trained XLSTM model on Riccio mixed features (angles + normalized keypoints).
"""

import argparse
import torch
import json
import numpy as np
from pathlib import Path
from fitness_coach.models.xlstm_model import xLSTMExerciseClassifier
from fitness_coach.datasets.exercise_bilstm_dataset import build_kaggle_mixed_datasets
from torch.utils.data import DataLoader

def evaluate_model(model, loader, device, class_names):
    model.eval()
    correct = 0
    total = 0
    y_true = []
    y_pred = []
    prob_list = []
    mse = torch.nn.MSELoss()
    reg_loss = 0.0
    abs_err = 0.0
    with torch.no_grad():
        for xb, y_cls, y_q in loader:
            xb = xb.to(device)
            y_cls = y_cls.to(device)
            y_q = y_q.to(device)
            logits, q_pred = model(xb)
            q_pred = q_pred.squeeze(-1)
            preds = logits.argmax(dim=1)
            correct += int((preds == y_cls).sum().item())
            total += xb.size(0)
            reg_loss += float(mse(q_pred, y_q).item()) * xb.size(0)
            abs_err += float(torch.abs(q_pred - y_q).sum().item())
            y_true.extend(y_cls.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
            prob_list.append(torch.softmax(logits, dim=1).cpu().numpy())

    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
    y_true = np.array(y_true, dtype=np.int64)
    y_pred = np.array(y_pred, dtype=np.int64)
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'f1_macro': float(f1_score(y_true, y_pred, average='macro')),
        'f1_weighted': float(f1_score(y_true, y_pred, average='weighted')),
        'precision_macro': float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
        'recall_macro': float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
        'class_names': class_names,
    }
    reg_metrics = {
        'rmse': float((reg_loss / max(total, 1)) ** 0.5),
        'mae': float(abs_err / max(total, 1)),
    }
    return metrics, reg_metrics, y_true, y_pred, np.vstack(prob_list) if prob_list else np.zeros((0, len(class_names)), dtype=np.float32)

def main():
    parser = argparse.ArgumentParser(description="Evaluate xLSTM model on Riccio mixed features")
    parser.add_argument("--model-path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--kaggle-keypoints-dir", type=str, required=True, help="Directory with keypoints")
    parser.add_argument("--kaggle-stem", type=str, default="riccio_realtime_exercise_recognition", help="Dataset stem")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for metrics")
    parser.add_argument("--window", type=int, default=30, help="Window size")
    parser.add_argument("--stride", type=int, default=15, help="Stride")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--hidden-size", type=int, default=512, help="Hidden size")
    parser.add_argument("--num-layers", type=int, default=6, help="Number of layers")
    parser.add_argument("--dropout", type=float, default=0.25, help="Dropout rate (fallback if not in checkpoint)")
    parser.add_argument(
        "--window-label",
        choices=("first", "last"),
        default="last",
        help="Must match training (posepulse uses first).",
    )
    parser.add_argument(
        "--exclude-classes",
        default=None,
        metavar="NAMES",
        help="Comma-separated exclusions; default from checkpoint or none.",
    )
    args = parser.parse_args()

    def _parse_exc(raw: str):
        s = (raw or "").strip()
        if not s:
            return None
        return [p.strip() for p in s.split(",") if p.strip()]

    exc = _parse_exc(args.exclude_classes or "")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    data_dir = Path(args.kaggle_keypoints_dir)
    ckpt_path = Path(args.model_path)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    wl = str(ckpt.get("window_label", args.window_label))
    exc_final = exc
    if exc_final is None and ckpt.get("excluded_coarse_classes"):
        exc_final = list(ckpt["excluded_coarse_classes"])

    train_ds, val_ds, test_ds, class_to_idx, idx_to_class, mean, std = build_kaggle_mixed_datasets(
        data_dir,
        stem=args.kaggle_stem,
        window=args.window,
        stride=args.stride,
        standardize=True,
        window_label=wl,
        exclude_coarse_classes=exc_final,
    )

    print(f'Classes: {list(class_to_idx.keys())}')
    print(f'Test samples: {len(test_ds)}')
    print(f'Input feature dim: {test_ds[0][0].shape[-1]}')

    input_size = int(ckpt.get("input_size", test_ds[0][0].shape[-1]))
    num_classes = int(ckpt.get("num_classes", len(class_to_idx)))
    hidden_size = int(ckpt.get("hidden", args.hidden_size))
    num_layers = int(ckpt.get("layers", args.num_layers))
    dropout = float(ckpt.get("dropout", args.dropout))
    linear_classifier = bool(ckpt.get("linear_classifier", False))
    block_pattern = ckpt.get("block_pattern") if isinstance(ckpt.get("block_pattern"), str) else None

    model = xLSTMExerciseClassifier(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout,
        bidirectional=True,
        linear_classifier=linear_classifier,
        block_pattern=block_pattern,
    )

    if 'model' in ckpt:
        state_dict = ckpt['model']
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Evaluate
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    metrics, reg_metrics, y_true, y_pred, y_prob = evaluate_model(
        model, test_loader, device, list(class_to_idx.keys())
    )

    print('Test Results:')
    print(f'Accuracy: {metrics["accuracy"]:.4f}')
    print(f'F1 weighted: {metrics["f1_weighted"]:.4f}')
    print(f'Precision macro: {metrics["precision_macro"]:.4f}')
    print(f'Recall macro: {metrics["recall_macro"]:.4f}')
    print(f'Quality RMSE: {reg_metrics["rmse"]:.4f}')

    # Save metrics
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from sklearn.metrics import classification_report
    report = classification_report(y_true, y_pred, target_names=list(class_to_idx.keys()), zero_division=0, output_dict=True)

    results = {
        'classification_metrics': metrics,
        'regression_metrics': reg_metrics,
        'classification_report': report,
        'num_samples': len(y_true),
        'config': {
            'input_size': input_size,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'dropout': dropout,
            'num_classes': num_classes,
            'block_pattern': block_pattern,
        }
    }

    metrics_path = out_dir / 'mixed_xlstm_evaluation_results.json'
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Save probabilities
    prob_path = out_dir / 'mixed_xlstm_test_probabilities.npy'
    np.save(prob_path, y_prob)

    print(f'\nMetrics saved to {metrics_path}')
    print(f'Probabilities saved to {prob_path}')

if __name__ == "__main__":
    main()