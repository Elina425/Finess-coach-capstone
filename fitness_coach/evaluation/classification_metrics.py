"""Window-level classification metrics: F1, recall, precision, confusion matrix."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
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
