"""Metric values checked against independent reference implementations.

Each test draws a synthetic series, picks a random threshold between its lowest
and highest prediction error, and compares `metrics.common_metrics` against a
straightforward calculation written here, not shared with the code under test.
"""

import numpy as np
import pytest

from AaltoAD import metrics

SEEDS = [1, 2, 3, 4, 5]


def _random_threshold(scores, seed):
    """A uniformly random threshold between the lowest and highest score."""
    rng = np.random.default_rng(seed + 1000)
    return float(rng.uniform(scores.min(), scores.max()))


def _reference_fpr(scores, labels, threshold):
    """False positive rate: alarms on normal points over all normal points.

    Segment expansion only extends alarms inside labelled segments, so the
    same value holds with and without it.
    """
    normal = labels == 0
    alarms = scores > threshold
    return np.sum(alarms & normal) / np.sum(normal)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("expand_segments", [False, True])
def test_fpr_matches_reference(synthetic_series, expand_segments, seed):
    scores, labels = synthetic_series(seed)
    threshold = _random_threshold(scores, seed)

    block = metrics.common_metrics(scores, labels, threshold, expand_segments)

    assert block["fpr"] == pytest.approx(_reference_fpr(scores, labels, threshold))
