"""Synthetic labelled prediction errors for tests.

Random score series with contiguous anomaly segments, and on-disk run results
(results JSON, labels CSV, calibration CSV) built from them.
"""

import json

import numpy as np
import pandas as pd


def _make_series(seed, n=60):
    """Random scores/labels with 2-3 contiguous segments, none starting at row 0."""
    rng = np.random.default_rng(seed)
    scores = rng.random(n)
    labels = np.zeros(n)
    n_segments = rng.integers(2, 4)
    starts = rng.choice(range(5, n - 10), size=n_segments, replace=False)
    for s in sorted(starts):
        length = rng.integers(2, 5)
        labels[s:s + length] = 1
    return scores, labels


def _write_dataset(tmp_path, name, seed, hp_suffix="M_exp1.0"):
    ds_dir = tmp_path / name
    ds_dir.mkdir()

    scores, labels = _make_series(seed)
    calib_rng = np.random.default_rng(seed + 100)
    calib = calib_rng.random(30)

    results_path = ds_dir / f"{hp_suffix}_results.json"
    labels_path = ds_dir / f"{hp_suffix}_labels.csv"
    calib_path = ds_dir / f"{hp_suffix}_calib_scores.csv"

    with open(results_path, "w") as f:
        json.dump({"model": "M", "applied_hyperparameters": {}}, f)

    pd.DataFrame({
        "timestamp": np.arange(len(scores)),
        "ground_truth": labels,
        "prediction_error": scores,
    }).to_csv(labels_path, index=False)

    pd.DataFrame({"prediction_error": calib}).to_csv(calib_path, index=False)

    with open(results_path) as f:
        result = json.load(f)
    result["_source_path"] = str(results_path)
    return result, scores, labels, calib
