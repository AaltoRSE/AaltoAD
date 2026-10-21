"""Shared helpers for threshold-based detection metrics.

Public API: prediction_metrics.
"""

import numpy as np


def prediction_metrics(pred, labels):
    """Confusion counts and detection rates for binarized predictions vs labels.

    Returns a tuple `(f1, prec, rec, fpr, tp, fp, fn, tn)` — counts as ints,
    rates as floats, with zero denominators yielding 0.0.
    """
    pred = np.asarray(pred)
    labels = np.asarray(labels)
    tp = int(np.sum(pred * labels))
    fp = int(np.sum(pred * (1 - labels)))
    fn = int(np.sum((1 - pred) * labels))
    tn = int(np.sum((1 - pred) * (1 - labels)))
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return f1, prec, rec, fpr, tp, fp, fn, tn
