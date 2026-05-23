"""Window-level classification metrics: F1, recall, precision, confusion matrix, ROC-OvR."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def detailed_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
) -> Dict[str, Any]:
    """
    Full report for exercise classification. Confusion matrix: **rows = true class**,
    **columns = predicted class** (sklearn convention).
    """
    n = len(class_names)
    labels = list(range(n))
    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()
    cm = confusion_matrix(yt, yp, labels=labels)
    f1_each = f1_score(yt, yp, average=None, labels=labels, zero_division=0)
    rec_each = recall_score(yt, yp, average=None, labels=labels, zero_division=0)
    prec_each = precision_score(yt, yp, average=None, labels=labels, zero_division=0)
    return {
        "accuracy": float(accuracy_score(yt, yp)),
        "f1_macro": float(f1_score(yt, yp, average="macro", labels=labels, zero_division=0)),
        "f1_weighted": float(f1_score(yt, yp, average="weighted", labels=labels, zero_division=0)),
        "f1_per_class": {class_names[i]: float(f1_each[i]) for i in range(n)},
        "recall_macro": float(recall_score(yt, yp, average="macro", labels=labels, zero_division=0)),
        "recall_per_class": {class_names[i]: float(rec_each[i]) for i in range(n)},
        "precision_macro": float(
            precision_score(yt, yp, average="macro", labels=labels, zero_division=0)
        ),
        "precision_per_class": {class_names[i]: float(prec_each[i]) for i in range(n)},
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_row_labels": class_names,
        "confusion_matrix_col_labels": class_names,
        "confusion_matrix_note": "rows=true class, columns=predicted class",
    }


def confusion_matrix_per_true_class(cm: List[List[int]], class_names: List[str]) -> Dict[str, Dict[str, int]]:
    """For each true exercise, counts of predictions by predicted label (human-readable rows of ``cm``)."""
    out: Dict[str, Dict[str, int]] = {}
    for i, name in enumerate(class_names):
        out[name] = {class_names[j]: int(cm[i][j]) for j in range(len(class_names))}
    return out


def confusion_matrix_normalized_by_true(
    cm: List[List[int]], class_names: List[str], eps: float = 1e-12
) -> Dict[str, Dict[str, float]]:
    """Row-normalised confusion matrix: each row sums to ~1 (recall decomposition)."""
    out: Dict[str, Dict[str, float]] = {}
    for i, name in enumerate(class_names):
        row = np.asarray(cm[i], dtype=np.float64)
        s = float(row.sum())
        denom = max(s, eps)
        out[name] = {class_names[j]: float(row[j] / denom) for j in range(len(class_names))}
    return out


def multiclass_roc_ovr_payload(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    class_names: List[str],
    *,
    max_points_per_curve: Optional[int] = 400,
) -> Dict[str, Any]:
    """One-vs-rest ROC per class plus macro / weighted ROC-AUC (sklearn OvR convention).

    ``y_proba`` shape ``(n_samples, n_classes)`` — use calibrated probabilities or softmax.

    ROC curves optional subsampling: keep endpoints + uniform stride when ``max_points_per_curve``.
    Full AUC computed on raw scores before downsampling curves.
    """
    n_classes = len(class_names)
    labels = np.arange(n_classes, dtype=int)
    yt = np.asarray(y_true).ravel().astype(int)
    prob = np.asarray(y_proba, dtype=np.float64)
    if prob.ndim != 2 or prob.shape[1] != n_classes:
        raise ValueError(f"y_proba must be (N, C) with C={n_classes}; got {prob.shape}")
    if yt.min() < 0 or yt.max() >= n_classes:
        raise ValueError("y_true labels must be within 0 .. n_classes-1")

    try:
        auc_per = roc_auc_score(yt, prob, multi_class="ovr", average=None, labels=labels)
        auc_macro = float(
            roc_auc_score(yt, prob, multi_class="ovr", average="macro", labels=labels)
        )
        auc_weighted = float(
            roc_auc_score(yt, prob, multi_class="ovr", average="weighted", labels=labels)
        )
    except ValueError:
        auc_per = np.full(n_classes, np.nan)
        auc_macro = float("nan")
        auc_weighted = float("nan")

    curves: Dict[str, Dict[str, List[float]]] = {}
    for i, name in enumerate(class_names):
        y_bin = (yt == i).astype(int)
        fpr, tpr, thr = roc_curve(y_bin, prob[:, i])
        if max_points_per_curve is not None and len(fpr) > max_points_per_curve:
            ix = np.linspace(0, len(fpr) - 1, num=max_points_per_curve, dtype=int)
            fpr_f = np.asarray(fpr)[ix].tolist()
            tpr_f = np.asarray(tpr)[ix].tolist()
            curves[name] = {"fpr": fpr_f, "tpr": tpr_f}
        else:
            thr_list = thr.tolist()
            curves[name] = {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "thresholds_preview": thr_list[: min(64, len(thr_list))],
            }

    return {
        "roc_auc_ovr_macro": float(auc_macro),
        "roc_auc_ovr_weighted": float(auc_weighted),
        "roc_auc_ovr_per_class": {class_names[k]: float(auc_per[k]) for k in range(n_classes)},
        "roc_ovr_note": "per-class ROC: positive class = that exercise, negative = all others",
        "roc_ovr_curves": curves,
    }


def classification_report_with_roc_proba(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    class_names: List[str],
    *,
    split: str = "test",
    max_curve_points: Optional[int] = 400,
) -> Dict[str, Any]:
    """Single JSON-serializable dict: detailed metrics + per-class confusion view + ROC-OvR."""
    base = detailed_classification_metrics(y_true, y_pred, class_names)
    cm = base["confusion_matrix"]
    roc_part = multiclass_roc_ovr_payload(
        y_true, y_proba, class_names, max_points_per_curve=max_curve_points
    )
    merged: Dict[str, Any] = {
        "split": split,
        "class_names": list(class_names),
        **base,
        "prediction_counts_per_true_class": confusion_matrix_per_true_class(cm, class_names),
        "confusion_matrix_normalized_by_true": confusion_matrix_normalized_by_true(cm, class_names),
        **roc_part,
    }
    return merged


def format_confusion_matrix_text(cm: List[List[int]], row_labels: List[str]) -> str:
    """ASCII table for terminal / logs."""
    if not cm:
        return "(empty)"
    n = len(cm)
    w = max(len(str(row_labels[i])) for i in range(n)) if row_labels else 6
    w = max(w, 8)
    header = " " * (w + 2) + "".join(f"{row_labels[j][:12]:>14}" for j in range(n))
    lines = [header, " " * (w + 2) + "(predicted →)"]
    for i in range(n):
        lab = (row_labels[i][: w + 2] if row_labels else f"c{i}")[: w + 2].ljust(w + 2)
        row = "".join(f"{int(cm[i][j]):>14}" for j in range(n))
        prefix = "true " if i == 0 else "     "
        lines.append(f"{prefix}{lab}{row}")
    return "\n".join(lines)
