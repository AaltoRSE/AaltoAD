"""Metric values checked against independent reference implementations.

Each test draws a synthetic series, picks a random threshold between its lowest
and highest prediction error, and compares `metrics.common_metrics` against a
straightforward calculation written here, not shared with the code under test.
"""

import numpy as np
import pytest

from AaltoAD import metrics

SEEDS = [1, 2, 3, 4, 5]

# Rate regulator passed to `common_metrics` and used by the reference.
EPSILON = 1e-5


def _random_threshold(scores, seed):
    """A uniformly random threshold between the lowest and highest score."""
    rng = np.random.default_rng(seed + 1000)
    return float(rng.uniform(scores.min(), scores.max()))


def _segments(labels):
    """(start, end) pairs, end exclusive, of each contiguous run of label 1."""
    segments = []
    start = None
    for i, label in enumerate(labels):
        if label == 1 and start is None:
            start = i
        elif label != 1 and start is not None:
            segments.append((start, i))
            start = None
    if start is not None:
        segments.append((start, len(labels)))
    return segments


def _reference_metrics(scores, labels, threshold, expand_segments):
    """The metric block `common_metrics` should return, computed directly.

    Alarms are ``scores > threshold``. With expansion, a segment with any
    alarm counts as fully alarmed. Latency is the mean, over segments, of the
    offset to the first raw alarm (the segment length if there is none). For
    binary alarms the ROC AUC is the mean of the true positive and true
    negative rates.
    """
    alarms = scores > threshold
    positive = labels == 1
    adjusted = alarms.copy()
    latencies = []
    for start, end in _segments(labels):
        hits = np.flatnonzero(alarms[start:end])
        latencies.append(hits[0] if hits.size else end - start)
        if expand_segments and hits.size:
            adjusted[start:end] = True

    tp = int(np.sum(adjusted & positive))
    fp = int(np.sum(adjusted & ~positive))
    fn = int(np.sum(~adjusted & positive))
    tn = int(np.sum(~adjusted & ~positive))
    precision = tp / (tp + fp + EPSILON)
    recall = tp / (tp + fn + EPSILON)
    return {
        "threshold": threshold,
        "f1": 2 * precision * recall / (precision + recall + EPSILON),
        "precision": precision,
        "recall": recall,
        "fpr": fp / (fp + tn + EPSILON),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "p_latency": float(np.mean(latencies)),
        "ROC/AUC": (tp / (tp + fn) + tn / (tn + fp)) / 2,
    }


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("expand_segments", [False, True])
def test_common_metrics_match_reference(synthetic_series, expand_segments, seed):
    scores, labels = synthetic_series(seed)
    threshold = _random_threshold(scores, seed)

    block = metrics.common_metrics(
        scores, labels, threshold, expand_segments, epsilon=EPSILON
    )

    assert block == pytest.approx(
        _reference_metrics(scores, labels, threshold, expand_segments)
    )
