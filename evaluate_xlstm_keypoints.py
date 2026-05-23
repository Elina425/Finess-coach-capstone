#!/usr/bin/env python3
"""
Evaluate trained XLSTM model on Riccio keypoints test set.
"""

import argparse
import torch
import json
import numpy as np
from pathlib import Path
from fitness_coach.models.xlstm_model import xLSTMExerciseClassifier
from fitness_coach.datasets.exercise_stgcn_dataset import build_kaggle_stgcn_datasets
from fitness_coach.training.train_exercise_stgcn import evaluate
from torch.utils.data import DataLoader, Dataset

class KeypointXLSTMDataset(Dataset):
    """Dataset for xLSTM: flatten keypoints to (T, 34) per window."""

    def __init__(
        self,
        samples: list,
        mean=None,
        std=None,
    ):
        self.samples = samples
        self.mean = mean
        self.std = std

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        x, y, q = self.samples[idx]
        x = np.asarray(x, dtype=np.float32)  # (T, 17, 2)
        if self.mean is not None and self.std is not None:
            x = (x - self.mean) / self.std
        # Flatten to (T, 34)
        x = x.reshape(x.shape[0], -1)
        return (
            torch.from_numpy(x),
            torch.tensor(y, dtype=torch.long),
            torch.tensor(q, dtype=torch.float32),
        )

def build_kaggle_xlstm_datasets(
    data_dir: Path,
    *,
    stem: str = "riccio_realtime_exercise_recognition",
    window: int = 30,
    stride: int = 15,
    test_ratio: float = 0.15,
    val_ratio: float = 0.15,
    seed: int = 42,
    standardize: bool = True,
    quality_default: float = 0.75,
):
    # Use the same loading as STGCN
    train_ds, val_ds, test_ds, class_to_idx, idx_to_class, mean, std = build_kaggle_stgcn_datasets(
        data_dir,
        stem=stem,
        window=window,
        stride=stride,
        test_ratio=test_ratio,
        val_ratio=val_ratio,
        seed=seed,
        standardize=standardize,
        quality_default=quality_default,
    )

    # Convert to XLSTM datasets
    train_xlstm = KeypointXLSTMDataset(train_ds.samples, mean, std)
    val_xlstm = KeypointXLSTMDataset(val_ds.samples, mean, std)
    test_xlstm = KeypointXLSTMDataset(test_ds.samples, mean, std)

    return train_xlstm, val_xlstm, test_xlstm, class_to_idx, idx_to_class, mean, std

def main():
    parser = argparse.ArgumentParser(description="Evaluate xLSTM model on Riccio keypoints")
    parser.add_argument("--model-path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--kaggle-keypoints-dir", type=str, required=True, help="Directory with keypoints")
    parser.add_argument("--kaggle-stem", type=str, default="riccio_realtime_exercise_recognition", help="Dataset stem")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for metrics")
    args = parser.parse_args()

    # Load data
    data_dir = Path(args.kaggle_keypoints_dir)
    train_ds, val_ds, test_ds, class_to_idx, idx_to_class, mean, std = build_kaggle_xlstm_datasets(
        data_dir, stem=args.kaggle_stem
    )

    # Load model
    ckpt_path = Path(args.model_path)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model = xLSTMExerciseClassifier(
        input_size=34,
        hidden_size=128,
        num_layers=2,
        num_classes=5,
        dropout=0.2174,
        bidirectional=True,
    )
    model.load_state_dict(ckpt['model'])
    model.eval()

    # Evaluate
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    ta, trmse, test_cls, y_true_t, y_prob_t = evaluate(
        model, test_loader, torch.device('cpu'), 5, detailed=True, class_names=list(class_to_idx.keys())
    )

    print('Test Results:')
    print(f'Accuracy: {ta:.4f}')
    print(f'F1 (weighted): {test_cls["f1_weighted"]:.4f}')
    print(f'Precision (macro): {test_cls["precision_macro"]:.4f}')
    print(f'Recall (macro): {test_cls["recall_macro"]:.4f}')
    print(f'Quality RMSE: {trmse:.4f}')

    print('\nPer-class F1:')
    for name, f1 in test_cls['f1_per_class'].items():
        print(f'  {name}: {f1:.4f}')

    print('\nConfusion Matrix (rows=true, cols=predicted):')
    from fitness_coach.evaluation.classification_metrics import format_confusion_matrix_text
    print(format_confusion_matrix_text(
        test_cls['confusion_matrix'],
        test_cls['confusion_matrix_row_labels'],
    ))

    # Save metrics
    out_dir = Path(args.output_dir)
    metrics_path = out_dir / 'test_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump({
            'accuracy': test_cls['accuracy'],
            'f1_macro': test_cls['f1_macro'],
            'f1_weighted': test_cls['f1_weighted'],
            'precision_macro': test_cls['precision_macro'],
            'precision_weighted': test_cls['precision_macro'],
            'recall_macro': test_cls['recall_macro'],
            'recall_weighted': test_cls['recall_macro'],
            'f1_per_class': list(test_cls['f1_per_class'].values()),
            'confusion_matrix': test_cls['confusion_matrix'],
            'class_names': list(class_to_idx.keys()),
            'quality_rmse': trmse,
            'quality_mae': 0.0,  # Not computed
            'quality_r2': 0.0,  # Not computed
        }, f, indent=2)
    print(f'\nMetrics saved to {metrics_path}')

if __name__ == "__main__":
    main()