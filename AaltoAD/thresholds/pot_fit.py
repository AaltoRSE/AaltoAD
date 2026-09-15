"""POT (SPOT) threshold fitting and metric computation, extracted from pot.pot_eval.

``SPOT`` is a module-level name here so tests can monkeypatch
``AaltoAD.thresholds.pot_fit.SPOT``.
"""

import numpy as np

from AaltoAD.spot import SPOT
from AaltoAD.thresholds.point_adjust import calc_point2point, adjust_predicts, segment_latency


def fit_pot_threshold(init_score, score, q, level):
    """Fit a POT/SPOT threshold on calibration data ``init_score`` and test data ``score``.

    Runs ``SPOT(q).fit(init_score, score)`` then ``initialize(level=...)``;
    on failure, retries with ``level`` lowered by a factor of 0.95, up to 100
    times. Returns the fitted ``extreme_quantile`` as a float, or
    ``float('nan')`` if ``init_score``/``score`` contain NaNs, or if the fit
    keeps failing after all retries.
    """
    if np.any(np.isnan(init_score)) or np.any(np.isnan(score)):
        return float('nan')

    retries = 0
    while True:
        try:
            s = SPOT(q)
            s.fit(init_score, score)
            s.initialize(level=level, min_extrema=False, verbose=False)
        except Exception as e:
            retries += 1
            if retries > 100:
                print(f'SPOT: giving up after {retries} retries: {e}')
                return float('nan')
            level = level * 0.95
        else:
            break
    return float(s.extreme_quantile)


def pot_metrics(score, label, threshold, expand_segments):
    """Compute POT-threshold detection metrics for an already-fitted threshold.

    Mirrors the tail of ``pot.pot_eval``: raw predictions are ``score >
    threshold``; if ``expand_segments`` is true, predictions are expanded to
    cover full ground-truth segments via ``adjust_predicts``. Returns a
    dict with keys ``f1, precision, recall, fpr, TP, TN, FP, FN, ROC/AUC,
    threshold, p_latency``. If ``threshold`` is NaN (a failed fit), returns
    the same all-NaN dict `pot.pot_eval` used for a failed fit, with
    ``p_latency=None``.
    """
    if threshold is None or np.isnan(threshold):
        return {
            'f1': np.nan, 'precision': np.nan, 'recall': np.nan, 'fpr': np.nan,
            'TP': np.nan, 'TN': np.nan, 'FP': np.nan, 'FN': np.nan,
            'ROC/AUC': np.nan, 'threshold': np.nan, 'p_latency': None,
        }

    raw_pred = score > threshold
    p_latency = segment_latency(raw_pred, label)
    if expand_segments:
        pred = adjust_predicts(score, label, threshold)
    else:
        pred = raw_pred
    p_t = calc_point2point(pred, label)
    fp, tn = float(p_t[5]), float(p_t[4])
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return {
        'f1': p_t[0],
        'precision': p_t[1],
        'recall': p_t[2],
        'fpr': fpr,
        'TP': p_t[3],
        'TN': p_t[4],
        'FP': p_t[5],
        'FN': p_t[6],
        'ROC/AUC': p_t[7],
        'threshold': threshold,
        'p_latency': p_latency,
    }
