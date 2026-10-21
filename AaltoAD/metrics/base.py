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


def common_metrics(scores, labels, threshold, expand_segments):
    """Metric block for alarms ``scores > threshold`` against `labels`.

    With `expand_segments`, an alarm anywhere in a labelled segment is
    expanded over the whole segment (``adjust_predicts``) before the counts
    and rates are computed. ``p_latency`` is always taken from the raw,
    unexpanded alarms. Every rate denominator has `EPSILON` added, and
    ``ROC/AUC`` is 0.0 when it is undefined (only one class in `labels`).

    Returns a dict with keys ``threshold, f1, precision, recall, fpr, tp, fp,
    fn, tn, p_latency, ROC/AUC``.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    pred = (scores > threshold).astype(int)
    if expand_segments:
        adj = np.asarray(point_adjust.adjust_predicts(scores, labels, pred=pred.copy()))
    else:
        adj = pred
    tp = int(np.sum(adj * labels))
    fp = int(np.sum(adj * (1 - labels)))
    fn = int(np.sum((1 - adj) * labels))
    tn = int(np.sum((1 - adj) * (1 - labels)))
    precision = tp / (tp + fp + EPSILON)
    recall = tp / (tp + fn + EPSILON)
    f1 = 2 * precision * recall / (precision + recall + EPSILON)
    fpr = fp / (fp + tn + EPSILON)
    try:
        roc_auc = float(sk_metrics.roc_auc_score(labels, adj))
    except ValueError:
        roc_auc = 0.0
    return {
        "threshold": float(threshold),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "fpr": float(fpr),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "p_latency": point_adjust.segment_latency(pred, labels),
        "ROC/AUC": roc_auc,
    }
