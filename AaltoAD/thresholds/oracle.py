"""Oracle-threshold metrics and a vectorised multi-dataset oracle threshold search."""

import numpy as np

from AaltoAD.metrics.base import common_metrics



def _label_segments(labels):
    """Return a list of (start, end) index pairs for contiguous label==1 runs.

    ``end`` is exclusive, matching ``segment_latency``'s segment finder.
    """
    labels = np.asarray(labels).astype(bool)
    padded = np.concatenate(([False], labels, [False]))
    diff = np.diff(padded.astype(np.int8))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return list(zip(starts, ends))


def _raw_counts(scores, labels, candidates):
    """Vectorised (tp, fp) point-level counts for every threshold in `candidates`.

    ``tp``/``fp`` at index i are the counts of label==1/label==0 points with
    ``score > candidates[i]``, computed via a sorted cumulative-sum trick
    (no explicit sweep over thresholds).
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    order = np.argsort(scores)
    s_asc = scores[order]
    l_asc = labels[order]
    n = len(s_asc)
    if n == 0:
        zeros = np.zeros(len(candidates))
        return zeros, zeros
    # suffix_tp[i] = sum(l_asc[i:]), with suffix_tp[n] = 0
    suffix_tp = np.concatenate([np.cumsum(l_asc[::-1])[::-1], [0.0]])
    idx = np.searchsorted(s_asc, candidates, side="right")
    tp = suffix_tp[idx]
    total = n - idx
    fp = total - tp
    return tp, fp


def _expanded_counts(scores, labels, candidates):
    """Vectorised (tp, fp, fn) segment-level counts for every threshold in `candidates`.

    A ground-truth segment (contiguous run of ``label==1``) counts as fully
    detected, contributing its whole length to ``tp``, iff the maximum score
    inside it is ``> threshold``; otherwise its whole length counts toward
    ``fn``. This approximates the true segment-expansion behaviour of
    ``adjust_predicts`` for the purpose of *searching* a threshold: it
    ignores the exact bidirectional expansion around the first detected
    point, so counts can differ from ``adjust_predicts`` by up to a
    segment's worth of points near ambiguous cases (notably
    ``adjust_predicts``'s backward-expansion loop never reaches index 0, an
    off-by-one quirk that does not exist here). ``fp`` is computed
    point-wise on ``label==0`` points exactly as in the raw (unexpanded)
    case, since expansion never turns a true negative into a positive
    prediction outside a labelled segment. The final, authoritative metrics
    for the chosen threshold are always recomputed with ``common_metrics``.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    segments = _label_segments(labels)
    total_pos = float(labels.sum())
    if segments:
        seg_max = np.array([scores[s:e].max() for s, e in segments])
        seg_len = np.array([e - s for s, e in segments], dtype=float)
        order = np.argsort(seg_max)
        m_asc = seg_max[order]
        l_asc = seg_len[order]
        suffix_len = np.concatenate([np.cumsum(l_asc[::-1])[::-1], [0.0]])
        idx = np.searchsorted(m_asc, candidates, side="right")
        tp = suffix_len[idx]
    else:
        tp = np.zeros(len(candidates))
    fn = total_pos - tp
    _, fp = _raw_counts(scores, labels, candidates)
    return tp, fp, fn


def shared_oracle_threshold(scores_list, labels_list, expand_segments):
    """Search one oracle threshold maximising pooled adjusted-F1 over several datasets.

    Candidate thresholds are the sorted unique values of all scores
    concatenated across datasets. For each candidate, per-dataset confusion
    counts are computed vectorised (raw case: point-wise via sorted
    cumulative sums; expanded case: per-segment, a labelled segment counts
    as detected iff its max score is > threshold — see ``_expanded_counts``
    for the approximation this implies relative to exact
    ``adjust_predicts`` semantics) and summed over datasets (no segment
    crosses a dataset boundary). The threshold with the highest pooled F1 is
    returned; ties are broken toward the *higher* threshold (fewer alarms).
    This function only searches; call ``common_metrics`` with the returned
    threshold for the exact, authoritative metrics.
    """
    candidates = np.unique(
        np.concatenate([np.asarray(s, dtype=float) for s in scores_list])
    )
    total_tp = np.zeros(len(candidates))
    total_fp = np.zeros(len(candidates))
    total_fn = np.zeros(len(candidates))
    for scores, labels in zip(scores_list, labels_list):
        if expand_segments:
            tp, fp, fn = _expanded_counts(scores, labels, candidates)
        else:
            tp, fp = _raw_counts(scores, labels, candidates)
            fn = float(np.asarray(labels, dtype=float).sum()) - tp
        total_tp += tp
        total_fp += fp
        total_fn += fn

    denom = 2 * total_tp + total_fp + total_fn
    f1 = np.where(denom > 0, 2 * total_tp / np.where(denom > 0, denom, 1.0), 0.0)
    best_idx = np.flatnonzero(f1 == f1.max())[-1]
    return float(candidates[best_idx])
