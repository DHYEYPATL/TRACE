"""
Evaluation Metrics module.
Computes Wilson 95% confidence intervals, confusion matrices, latency percentiles (P50, P95, max),
and Rule-of-Three false alarm rates matching Section 3 and Section 11.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple
import numpy as np


def wilson_interval(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float, float]:
    """
    Computes Wilson score interval for binomial proportions.
    Formula:
      p_tilde = (k + z^2/2) / (n + z^2)
      margin = z * sqrt( (p*(1-p)/n) + (z^2 / (4*n^2)) ) / (1 + z^2/n)
    Returns: (point_estimate, lower_bound, upper_bound)
    """
    if n <= 0:
        return 0.0, 0.0, 1.0

    p = float(k) / float(n)
    # Standard normal critical value for 95% confidence
    z = 1.959963984540054

    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    spread = (z / denom) * math.sqrt((p * (1.0 - p) / n) + (z2 / (4.0 * n * n)))

    lower = max(0.0, center - spread)
    upper = min(1.0, center + spread)

    return round(p, 4), round(lower, 4), round(upper, 4)


def compute_confusion_matrix(
    y_true: List[str],
    y_pred: List[str],
    classes: List[str]
) -> Dict[str, Any]:
    """
    Computes confusion matrix, per-class precision and recall with Wilson intervals.
    """
    c_map = {c: i for i, c in enumerate(classes)}
    size = len(classes)
    matrix = np.zeros((size, size), dtype=int)

    for yt, yp in zip(y_true, y_pred):
        if yt in c_map and yp in c_map:
            matrix[c_map[yt], c_map[yp]] += 1

    per_class = {}
    for c in classes:
        idx = c_map[c]
        tp = int(matrix[idx, idx])
        fn = int(np.sum(matrix[idx, :]) - tp)
        fp = int(np.sum(matrix[:, idx]) - tp)

        prec_p, prec_low, prec_high = wilson_interval(tp, tp + fp)
        rec_p, rec_low, rec_high = wilson_interval(tp, tp + fn)

        per_class[c] = {
            "tp": tp, "fn": fn, "fp": fp,
            "precision": prec_p,
            "precision_ci": (prec_low, prec_high),
            "recall": rec_p,
            "recall_ci": (rec_low, rec_high),
        }

    return {
        "classes": classes,
        "matrix": matrix.tolist(),
        "per_class": per_class
    }


def compute_latency_stats(latencies_ms: List[float]) -> Dict[str, float]:
    """Computes P50, P95, max, and mean latency in milliseconds or seconds."""
    if not latencies_ms:
        return {"p50": 0.0, "p95": 0.0, "max": 0.0, "mean": 0.0}

    arr = np.array(latencies_ms)
    return {
        "p50": round(float(np.percentile(arr, 50)), 2),
        "p95": round(float(np.percentile(arr, 95)), 2),
        "max": round(float(np.max(arr)), 2),
        "mean": round(float(np.mean(arr)), 2),
    }


def rule_of_three_rate(trials_camera_days: float) -> float:
    """Upper 95% bound on error rate when zero errors occur in T camera-days."""
    if trials_camera_days <= 0:
        return 3.0
    return round(3.0 / trials_camera_days, 3)
