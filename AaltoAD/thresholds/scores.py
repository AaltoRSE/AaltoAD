"""Load per-run calibration/test scores and labels from a result's sidecar CSVs."""

import pandas as pd


def load_run_scores(result):
    """Load (calib, test, labels) arrays for one saved result dict.

    Reads ``<...>_labels.csv`` (columns include ``ground_truth`` and
    ``prediction_error``) and ``<...>_calib_scores.csv`` (column
    ``prediction_error``), located next to ``result["_source_path"]`` by
    replacing the ``_results.json`` suffix (same convention as
    ``report.py``'s ``_generate_prediction_error_plot``).

    Returns ``(calib, test, labels)`` as float numpy arrays (via
    ``pandas.Series.to_numpy``), or ``None`` if the source path is missing,
    either CSV cannot be read, or a required column is absent. The
    calibration CSV may have a different row count than the test CSV; that
    is expected and not treated as an error.
    """
    src = result.get("_source_path")
    if not src:
        return None
    labels_path = src.replace("_results.json", "_labels.csv")
    calib_path = src.replace("_results.json", "_calib_scores.csv")
    try:
        labels_df = pd.read_csv(labels_path)
        calib_df = pd.read_csv(calib_path)
    except (OSError, ValueError):
        return None
    if "prediction_error" not in labels_df.columns or "ground_truth" not in labels_df.columns:
        return None
    if "prediction_error" not in calib_df.columns:
        return None
    test = labels_df["prediction_error"].to_numpy(dtype=float)
    labels = labels_df["ground_truth"].to_numpy(dtype=float)
    calib = calib_df["prediction_error"].to_numpy(dtype=float)
    return calib, test, labels
