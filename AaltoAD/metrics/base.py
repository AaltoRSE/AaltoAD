"""Detection metrics at a given threshold.

`common_metrics` computes the metric block stored in results.json from
prediction errors, ground-truth labels and an already chosen threshold. It is
the same for every threshold method (POT, conformal, oracle).
"""

import numpy as np
from sklearn import metrics as sk_metrics

from AaltoAD.thresholds import point_adjust

# Added to every rate denominator so an empty class gives a near-zero rate
# instead of a division by zero (as in `point_adjust.calc_point2point`).
EPSILON = 1e-5


def _predictions(scores, labels, threshold, expand_segments):
    """Raw alarms ``scores > threshold`` and the alarms the metrics count.

    Returns ``(pred, adj)``: `adj` is `pred` expanded over labelled segments
    (``adjust_predicts``) with `expand_segments`, otherwise `pred` itself.
    """
    pred = (scores > threshold).astype(int)
    if expand_segments:
        adj = np.asarray(point_adjust.adjust_predicts(scores, labels, pred=pred.copy()))
    else:
        adj = pred
    return pred, adj


def _f1_from_predictions(adj, labels, epsilon):
    """Confusion counts, precision, recall and F1 for binary alarms `adj` vs `labels`."""
    adj = np.asarray(adj, dtype=float)
    tp = int(np.sum(adj * labels))
    fp = int(np.sum(adj * (1 - labels)))
    fn = int(np.sum((1 - adj) * labels))
    tn = int(np.sum((1 - adj) * (1 - labels)))
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1 = 2 * precision * recall / (precision + recall + epsilon)
    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "precision": float(precision),
        "recall": float(recall),
        "F1": float(f1),
    }


def f1_metrics(scores, labels, threshold, expand_segments, epsilon=EPSILON):
    """Confusion counts, precision, recall and F1 for alarms ``scores > threshold`` against `labels`.

    With `expand_segments`, alarms are expanded over labelled segments first,
    as in `common_metrics`. Every rate denominator has `epsilon` added.
    Returns a dict with keys ``TP, FP, FN, TN, precision, recall, F1``.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    _, adj = _predictions(scores, labels, threshold, expand_segments)
    return _f1_from_predictions(adj, labels, epsilon)


def common_metrics(scores, labels, threshold, expand_segments, epsilon=EPSILON):
    """Metric block for alarms ``scores > threshold`` against `labels`.

    With `expand_segments`, an alarm anywhere in a labelled segment is
    expanded over the whole segment (``adjust_predicts``) before the counts
    and rates are computed. ``P_latency`` is always taken from the raw,
    unexpanded alarms. Every rate denominator has `epsilon` added, and
    ``ROC/AUC`` is 0.0 when it is undefined (only one class in `labels`).

    Returns a dict with keys ``threshold, F1, precision, recall, FPR, TP, FP,
    FN, TN, P_latency, ROC/AUC``. If `threshold` is None or NaN (e.g. a
    failed threshold fit), every value is NaN and ``P_latency`` is None.
    """
    if threshold is None or np.isnan(threshold):
        return {
            "threshold": np.nan,
            "F1": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "FPR": np.nan,
            "TP": np.nan,
            "FP": np.nan,
            "FN": np.nan,
            "TN": np.nan,
            "P_latency": None,
            "ROC/AUC": np.nan,
        }
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    pred, adj = _predictions(scores, labels, threshold, expand_segments)
    m = _f1_from_predictions(adj, labels, epsilon)
    fpr = m["FP"] / (m["FP"] + m["TN"] + epsilon)
    try:
        roc_auc = float(sk_metrics.roc_auc_score(labels, adj))
    except ValueError:
        roc_auc = 0.0
    return {
        "threshold": float(threshold),
        "F1": m["F1"],
        "precision": m["precision"],
        "recall": m["recall"],
        "FPR": float(fpr),
        "TP": m["TP"],
        "FP": m["FP"],
        "FN": m["FN"],
        "TN": m["TN"],
        "P_latency": point_adjust.segment_latency(pred, labels),
        "ROC/AUC": roc_auc,
    }
