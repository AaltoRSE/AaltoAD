"""Split-conformal thresholding: a calibration quantile with a finite-sample correction.

The threshold is the calibration score of rank ``ceil((n + 1) * (1 - q))``, so
at most a fraction `q` of the calibration set sits above it. Calling a test
point anomalous when ``score > threshold`` then holds the false alarm rate at
or below `q` for exchangeable calibration and normal test data. It targets the
same risk `q` POT does, reached by counting ranks rather than fitting a tail
distribution -- so the two give similar thresholds, and conformal needs no fit
that can fail to converge.

Because the guarantee comes from the calibration sample, the threshold is fit
on all of it at once: `AaltoAD.thresholds.shared` pools the calibration scores
of every dataset in a report and fits one threshold for them all.
"""

import math

import numpy as np

# Metrics at a given threshold are method-agnostic: predictions are score >
# threshold either way, so the POT block's implementation is reused verbatim.
from AaltoAD.thresholds.pot_fit import pot_metrics as threshold_metrics


def conformal_threshold(calib, q):
    """Conformal threshold for calibration scores `calib` at false alarm rate `q`.

    NaNs in `calib` are dropped. Returns ``float('nan')`` when no usable
    calibration score remains.
    """
    calib = np.asarray(calib, dtype=float)
    calib = calib[~np.isnan(calib)]
    if calib.size == 0:
        return float("nan")
    # A `q` below 1/(n+1) asks for a guarantee this calibration set cannot
    # give; the largest calibration score is the most conservative it offers.
    rank = min(max(math.ceil((calib.size + 1) * (1 - q)), 1), calib.size)
    return float(np.sort(calib)[rank - 1])


def conformal_metrics(scores, labels, threshold, expand_segments):
    """Detection metrics at a conformal `threshold`, in the same shape as the POT blocks."""
    return threshold_metrics(scores, labels, threshold, expand_segments)
