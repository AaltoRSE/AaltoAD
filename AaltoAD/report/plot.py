"""Prediction-error plot generation for dataset reports."""

import json
import math
import os

import matplotlib

matplotlib.use("Agg")

import pandas as pd

from AaltoAD import constants
from AaltoAD.report import cli, metrics
from AaltoAD.report.report_figures import downsample, style
from AaltoAD.thresholds import shared


# How many models the prediction-error overlay shows. Kept small so the overlay
# stays readable; --plot-models overrides it.
DEFAULT_PLOT_MODELS = constants.PLOT_MODELS


PLOT_VALUE_CLIP = 2.0


PLOT_Y_TOP = 2.2


# Default downsampling of plotted series (--downsample, --downsample-window).
DOWNSAMPLE = constants.DOWNSAMPLE


DOWNSAMPLE_WINDOWS = dict(constants.DOWNSAMPLE_WINDOWS)


DOWNSAMPLERS = {
    "min": downsample.min_in_window,
    "max": downsample.max_in_window,
    "mean": downsample.mean_in_window,
    "nth": downsample.every_nth_step,
}


def _downsample(series, mode=DOWNSAMPLE, window=None):
    """Reduce a plotted series, returning ``(low, high)``; `low` is None for a plain line.

    `mode` is one of `DOWNSAMPLERS` or "range" (see
    `AaltoAD.report_figures.downsample`). Clipping happens here too, so every
    plot is drawn on the same scale.
    """
    window = DOWNSAMPLE_WINDOWS.get(mode, 10) if window is None else window
    if mode == "range":
        low, high = downsample.min_max_in_window(series, window)
        return _clip_for_plot(low), _clip_for_plot(high)
    reduce = DOWNSAMPLERS.get(mode, downsample.max_in_window)
    return None, _clip_for_plot(reduce(series, window))


# Methods a per-model plot draws a reference line for, when the result has them.
REFERENCE_METHODS = ("conformal", "pot", "oracle")


def _thresholds(result):
    """``{method: threshold}`` for every `REFERENCE_METHODS` block with a usable threshold."""
    found = {}
    for method in REFERENCE_METHODS:
        value = _plot_threshold(result, method)
        if value is not None:
            found[method] = value
    return found


def _plot_threshold(result, method):
    """Threshold used to scale a result's errors in plots, or None if missing/non-positive."""
    block = result.get(method) if isinstance(result, dict) else None
    if not isinstance(block, dict):
        return None
    try:
        value = float(block.get("threshold"))
    except (TypeError, ValueError):
        return None
    return value if value > 0 and not math.isnan(value) else None


def _clip_for_plot(values):
    """Clip threshold-scaled values to `PLOT_VALUE_CLIP` so every plot shares one y scale."""
    return values.clip(upper=PLOT_VALUE_CLIP)


def _generate_prediction_error_plot(dataset, metric, by_model, output_path, models=None, blocks_key=None,
                                    n_models=DEFAULT_PLOT_MODELS, threshold_method=cli.THRESHOLD_METHOD,
                                    model_order=None, downsample_mode=DOWNSAMPLE,
                                    downsample_window=None):
    """Overlay each model's best-result prediction_error for a dataset.

    `models`, when given, fixes which models are plotted (and their order)
    instead of the top `n_models` by `model_order`; the combined report of a
    multi-dataset run passes its pooled ranking that way.
    `blocks_key` (e.g. ``"shared"``) scales by the thresholds stored under that
    key of each result instead of the local ones; models without it are skipped.

    For each model, take its best result (by `metric`), read the matching
    ``*_labels.csv``, and plot the ``prediction_error`` column scaled by that
    model's `threshold_method` threshold, so the threshold is 1 for every
    model. It is drawn once (dashed) and anomaly regions are shaded once. The
    series are downsampled to every 10th time step before plotting. The
    figure is saved as a PNG at ``output_path``.
    """
    model_order = cli.metric_list(metric if model_order is None else model_order)
    series = {}
    ground_truth = None
    best_by_model = {}
    for model, results in by_model.items():
        best = metrics._best_result(results, metric)
        if not best:
            continue
        src = best.get("_source_path")
        if not src:
            continue
        csv_path = src.replace("_results.json", "_labels.csv")
        if not os.path.exists(csv_path):
            continue
        try:
            df = pd.read_csv(csv_path)
        except (ValueError, OSError):
            continue
        if "prediction_error" not in df.columns:
            continue
        # Scale by the report's threshold, so it maps to 1 for every model.
        thr_source = shared.with_blocks(best, blocks_key) if blocks_key else best
        threshold = _plot_threshold(thr_source, threshold_method) if thr_source else None
        if not threshold:
            print(f"No usable {threshold_method} threshold for {model}; skipping in plot.")
            continue
        test_scores = df["prediction_error"].reset_index(drop=True) / threshold
        # Prepend calibration scores (negative steps) when the sidecar CSV
        # exists, so the plot shows the data the threshold was fitted on.
        calib_path = src.replace("_results.json", "_calib_scores.csv")
        if os.path.exists(calib_path):
            try:
                calib = pd.read_csv(calib_path)["prediction_error"] / threshold
                test_scores = pd.concat([
                    pd.Series(calib.values, index=range(-len(calib), 0)),
                    test_scores,
                ])
            except (ValueError, OSError, KeyError):
                pass
        series[model] = test_scores
        # Rank on the same blocks the series is scaled by, so a shared-threshold
        # plot ranks by the shared metrics.
        best_by_model[model] = thr_source
        # Ground truth is shared across models for a dataset; capture it once.
        if ground_truth is None and "ground_truth" in df.columns:
            ground_truth = df["ground_truth"].reset_index(drop=True)

    if not series:
        print(f"No prediction_error data found for {dataset}; skipping plot.")
        return

    # Keep only the `n_models` best models (see `metrics._rank_key`) so the overlay
    # stays readable.
    if models is not None:
        series = {m: series[m] for m in models if m in series}
    elif len(series) > n_models:
        keep = sorted(
            series,
            key=lambda m: metrics._rank_key(best_by_model[m], model_order, threshold_method),
        )[:n_models]
        series = {m: series[m] for m in keep}

    # Models have different calibration lengths, so the outer join leaves the
    # union index unsorted; sort it or the lines wrap back to the start.
    combined = pd.concat(series, axis=1).sort_index()

    low, combined = _downsample(combined, downsample_mode, downsample_window)

    fig, ax = style.new_figure()
    if low is None:
        style.plot_series(ax, combined)
    else:
        style.plot_bands(ax, low, combined)
    ax.set_ylim(0, PLOT_Y_TOP)
    # Every series is scaled by its own threshold, so one line at 1 is the
    # threshold for all models.
    style.draw_threshold_line(ax, 1.0, f"{threshold_method} threshold")
    if ground_truth is not None:
        style.shade_anomalies(ax, ground_truth)
    if combined.index.min() < 0:
        style.mark_calibration_end(ax)
    ax.set_xlabel("time step")
    ax.set_ylabel(f"prediction error / {threshold_method} threshold")
    style.legend_below(ax)
    style.save_png(fig, output_path)


def _generate_model_plots(dataset, metric, by_model, output_dir, blocks_key=None,
                          threshold_method=cli.THRESHOLD_METHOD, downsample_mode=DOWNSAMPLE,
                          downsample_window=None):
    """Plot the best result per model as prediction error vs. threshold.

    For each model, take its best result (by `metric`), read the matching
    ``*_labels.csv``, and plot the ``prediction_error`` series (downsampled
    to every 10th time step) against the POT and oracle thresholds (scaled
    by the metric's own threshold) with ground-truth anomaly regions shaded.
    Each model is written to ``output_dir`` (typically
    ``reports/<dataset>/plots/``) as a PNG.
    """
    os.makedirs(output_dir, exist_ok=True)
    for model, results in by_model.items():
        best = metrics._best_result(results, metric)
        if not best:
            continue
        src = best.get("_source_path")
        if not src:
            continue
        csv_path = src.replace("_results.json", "_labels.csv")
        if not os.path.exists(csv_path):
            print(f"No labels CSV for {model} best result; skipping plot.")
            continue
        try:
            df = pd.read_csv(csv_path)
        except (ValueError, OSError):
            continue
        if "prediction_error" not in df.columns:
            continue

        thr_source = shared.with_blocks(best, blocks_key) if blocks_key else best
        if thr_source is None:
            continue
        reference = _thresholds(thr_source)
        threshold = _plot_threshold(thr_source, threshold_method)
        if not threshold:
            print(f"No usable {threshold_method} threshold for {model}; skipping plot.")
            continue

        # Scale by the threshold of the method the metric names so it maps to 1.
        series = df["prediction_error"].reset_index(drop=True) / threshold

        # Prepend calibration scores at negative steps when the sidecar exists.
        calib_path = src.replace("_results.json", "_calib_scores.csv")
        n_calib = 0
        if os.path.exists(calib_path):
            try:
                calib = pd.read_csv(calib_path)["prediction_error"] / threshold
                n_calib = len(calib)
                series = pd.concat([
                    pd.Series(calib.values, index=range(-n_calib, 0)),
                    series,
                ])
            except (ValueError, OSError, KeyError):
                n_calib = 0

        low, series = _downsample(series, downsample_mode, downsample_window)
        series = series.rename("prediction_error")

        fig, ax = style.new_figure()
        if low is None:
            style.plot_series(ax, series)
        else:
            style.plot_bands(ax, low.rename("prediction_error"), series)
        ax.set_ylim(0, PLOT_Y_TOP)
        # Every method the result carries is drawn for comparison; the one the
        # report is generated for is the red line the series is scaled by.
        for name, value in reference.items():
            color = "tab:red" if name == threshold_method else "grey"
            style.draw_threshold_line(ax, value / threshold, f"{name} threshold", color=color)
        if "ground_truth" in df.columns:
            style.shade_anomalies(ax, df["ground_truth"])
        if n_calib:
            style.mark_calibration_end(ax)
        ax.set_xlabel("time step")
        ax.set_ylabel(f"prediction error / {threshold_method} threshold")
        ax.set_title(f"{model} — {dataset}" + (" (shared threshold)" if blocks_key else ""))
        style.legend_below(ax)
        out_path = os.path.join(output_dir, f"{model}.png")
        style.save_png(fig, out_path)


def _model_order_path(dataset):
    """Where `dataset`'s hand-picked plot-model file lives, next to its report files."""
    return os.path.join("reports", dataset, "model-order.json")


def _read_model_order_file(path):
    """Load a hand-picked model list from `path`, or None when there is no file.

    The file holds a JSON array of model names, e.g. ["TranAD", "LSTM_AE",
    "USAD"]. Anything else raises ValueError: a malformed hand-picked list
    should stop the report rather than silently fall back.
    """
    if not os.path.exists(path):
        return None
    with open(path) as f:
        names = json.load(f)
    if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
        raise ValueError(f"{path} must contain a JSON array of model names")
    print(f"Using hand-picked plot models from {path}: {', '.join(names)}")
    return names


def _plot_model_selection(dataset, by_model):
    """Hand-picked models for `dataset`'s overlay plots, or None to rank as usual.

    Reads ``reports/<dataset>/model-order.json`` (see `_read_model_order_file`);
    the combined report reads ``reports/combined/model-order.json``. When the
    file exists, the prediction-error overlays show exactly the models it
    names, in its order, instead of the top `n_plot_models` by ranking — and
    nothing else changes: tables, orderings and the overview are untouched. A
    name with no results is skipped with a warning, so a typo is visible
    instead of silently thinning the plot. No file means the usual top-`n`
    selection.
    """
    names = _read_model_order_file(_model_order_path(dataset))
    if names is None:
        return None
    kept = []
    for name in names:
        if name in by_model:
            kept.append(name)
        else:
            print(f'Warning: model "{name}" in {_model_order_path(dataset)} has no results; skipping.')
    return kept
