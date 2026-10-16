"""Shared helpers for threshold-based detection metrics.

Public API: adjusted_predictions, prediction_metrics.
"""

import numpy as np

from AaltoAD.thresholds import point_adjust


def adjusted_predictions(scores, labels, threshold, expand_segments):
    """Point predictions at `threshold` and their segment-adjusted version.

    Returns a tuple `(pred, adj)`: `pred` is the raw point prediction
    (`scores >= threshold`); `adj` is `pred` expanded along labelled segments
    with `adjust_predicts` when `expand_segments` is true, else `pred` itself.
    `adjust_predicts` mutates the array passed as `pred`, so the raw
    predictions are copied before adjustment — both returned arrays are safe
    to use independently (e.g. latency on raw `pred`, metrics on `adj`).
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    pred = (scores >= threshold).astype(int)
    if expand_segments:
        adj = np.asarray(point_adjust.adjust_predicts(scores, labels, pred=pred.copy()))
    else:
        adj = pred
    return pred, adj


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
