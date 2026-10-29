"""POT (SPOT) threshold fitting and metric computation, extracted from pot.pot_eval.

``SPOT`` is a module-level name here so tests can monkeypatch
``AaltoAD.thresholds.pot_fit.SPOT``.
"""

import numpy as np

from AaltoAD.spot import SPOT


def fit_pot_threshold(init_score, score, q, level):
    """Fit a POT/SPOT threshold on calibration data ``init_score`` and test data ``score``.

    Runs ``SPOT(q).fit(init_score, score)`` then ``initialize(level=...)``;
    on failure, retries with ``level`` lowered by a factor of 0.95, up to 100
    times. Returns the fitted ``extreme_quantile`` as a float, or
    ``float('nan')`` if ``init_score``/``score`` contain NaNs, or if the fit
    keeps failing after all retries.
    """
    if np.any(np.isnan(init_score)) or np.any(np.isnan(score)):
        return float("nan")

    retries = 0
    while True:
        try:
            s = SPOT(q)
            s.fit(init_score, score)
            s.initialize(level=level, min_extrema=False, verbose=False)
        except Exception as e:
            retries += 1
            if retries > 100:
                print(f"SPOT: giving up after {retries} retries: {e}")
                return float("nan")
            level = level * 0.95
        else:
            break
    return float(s.extreme_quantile)
