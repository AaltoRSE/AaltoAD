"""Generate HTML and PDF reports from hyperparameter sweep result JSON files."""

import csv
import json
import math
import os

from tqdm import tqdm
from glob import glob

import matplotlib
from AaltoAD import constants
from AaltoAD.report_figures import downsample, style
from AaltoAD.thresholds import shared

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from jinja2 import Template
from matplotlib.backends.backend_pdf import PdfPages


METHODS = ["pot", "oracle"]
METHOD_METRICS = ["f1", "precision", "recall", "fpr", "threshold", "p_latency"]
# Column display order: each metric paired with its point-adjusted (PA) counterpart.
DISPLAY_COLUMNS = []
for _m in METHOD_METRICS:
    DISPLAY_COLUMNS.append(_m)
    DISPLAY_COLUMNS.append(f"{_m} (PA)")
DISPLAY_COLUMNS.append("eval_time")
DISPLAY_COLUMNS.append("calibration_loss")

# Used by --metric flag; default points at pot.f1 (raw, non-PA).
DISPLAY_METRICS = [f"{method}.{m}" for method in METHODS for m in METHOD_METRICS]

# Metrics where a smaller value is the better one.
LOWER_IS_BETTER = {"p_latency", "fpr", "threshold", "calibration_loss"}

# Metrics that live at the top level of a result rather than inside a threshold
# method's block.
TOP_LEVEL_METRICS = {"calibration_loss", "eval_time"}

# Spellings accepted on the command line for a metric's real name.
METRIC_ALIASES = {"latency": "p_latency", "precision": "precision", "recall": "recall"}

SUMMARY_SORT_METRIC = "pot.f1"


def metric_list(metric):
    """Normalize a `--metric` argument into a tuple of bare metric names.

    Accepts a comma-separated string, a sequence, or a single name, and takes
    the metric out of a dotted path, so 'latency,fpr,f1', ('p_latency', 'fpr',
    'f1') and 'conformal.f1' all work. The threshold method comes from
    `--threshold`, not from here.
    """
    if isinstance(metric, str):
        parts = [p.strip() for p in metric.replace(" ", ",").split(",")]
    else:
        parts = [str(p).strip() for p in metric]
    names = []
    for part in parts:
        if not part:
            continue
        name = part.split(".")[-1] if "." in part else part
        names.append(METRIC_ALIASES.get(name, name))
    return tuple(names)


def metric_display(metric):
    """Human-readable form of a metric argument, for table headers and titles."""
    return ",".join(metric_list(metric))


def _metric_path(name, method):
    """Full lookup path for a bare metric name: 'f1' -> '<method>.f1'."""
    if "." in name or name in TOP_LEVEL_METRICS:
        return name
    return f"{method}.{name}"


def _no_detection(result, method):
    """True when `result`'s `method` block has no true positives.

    `segment_latency` charges an undetected segment its full length, so such a
    latency records how long the segment was, not how long detection took. It
    must not be ranked or printed as a detection time — and its value varies
    with the model's warm-up, which makes it look faster the more rows a model
    drops before scoring.
    """
    counts = _confusion_counts(result, method) if result else None
    return counts is not None and counts[0] == 0


def _fmt_metric(result, path, method):
    """Format one metric cell, as an em dash for a latency that is really a non-detection."""
    if isinstance(path, str) and path.endswith(".p_latency") and _no_detection(result, method):
        return "---"
    return _fmt(_col_value(result, path))


def _rank_key(result, metric, method):
    """Sort key ranking one result by `metric`, best first.

    `metric` is a tuple of bare names (see `metric_list`) read from the `method`
    block, each in its natural direction. A missing value sorts last within its
    own term, and so does the latency of a model that never fired (see
    `_no_detection`), so a model that never fires cannot outrank one that does.
    """
    key = []
    for name in metric_list(metric):
        value = _metric_value(result, _metric_path(name, method)) if result else None
        if name == "p_latency" and _no_detection(result, method):
            value = None
        if value is None:
            key.append(float("inf"))
        else:
            key.append(value if name in LOWER_IS_BETTER else -value)
    return tuple(key)

# How many models the prediction-error overlay shows. Kept small so the overlay
# stays readable; --plot-models overrides it.
DEFAULT_PLOT_MODELS = constants.PLOT_MODELS

# Plots are drawn on a fixed scale: every series is divided by its own
# threshold, so 1 is the threshold for every model and a common scale makes
# plots comparable across models and cases. Values above the clip are "far over
# threshold" either way, and clipping them keeps the interesting range legible.
POOL_BASELINES = constants.POOL_BASELINES

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


def _table_sort_metric(metric):
    """Fallback single metric for the unlabeled tables, which cannot rank by detections."""
    metric = metric if isinstance(metric, str) else ",".join(metric)
    return metric if "." in metric else SUMMARY_SORT_METRIC


def _get(d, dotted_key):
    """Look up a possibly-dotted key like 'pot.f1' in a nested dict, returning None if absent."""
    cur = d
    for part in dotted_key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _load_results(dataset, results_folder="results"):
    """Load all result JSONs for a dataset, grouped by model."""
    pattern = os.path.join(results_folder, dataset, "*_results.json")
    files = glob(pattern)
    by_model = {}
    for path in files:
        with open(path) as f:
            data = json.load(f)
        # Remember where this result came from so per-model plots can locate the
        # matching *_labels.csv (same path with _results.json -> _labels.csv).
        data["_source_path"] = path
        model = data.get("model", "unknown")
        by_model.setdefault(model, []).append(data)
    return by_model


def _metric_value(result, metric):
    """Return the metric as a float, or None if absent/non-numeric/NaN."""
    try:
        v = float(_get(result, metric))
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def _hp_key(result):
    """Canonical key identifying a hyperparameter configuration."""
    return json.dumps(result.get("applied_hyperparameters", {}), sort_keys=True)


def _confusion_counts(result, method):
    """Return (TP, FP, FN) for a method block, or None if any is absent.

    POT blocks store upper-case keys, oracle blocks lower-case ones.
    """
    block = result.get(method)
    if not isinstance(block, dict):
        return None
    out = []
    for name in ("TP", "FP", "FN"):
        v = block.get(name, block.get(name.lower()))
        if v is None:
            return None
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            return None
    return tuple(out)


def _pooled_f1(results, method):
    """F1 recomputed from confusion counts summed over `results`.

    Returns None if any result lacks counts for `method`.
    """
    tp = fp = fn = 0.0
    for r in results:
        counts = _confusion_counts(r, method)
        if counts is None:
            return None
        tp += counts[0]
        fp += counts[1]
        fn += counts[2]
    denom = 2 * tp + fp + fn
    return 0.0 if denom == 0 else 2 * tp / denom


def _select_shared_best(by_dataset, metric, method=None):
    """Select one hyperparameter set per model across all datasets.

    For each model, group results by hyperparameter configuration (keeping the
    best result per dataset within a configuration) and pick the configuration
    with the best score over the listed datasets. Each term of `metric` is
    aggregated across the datasets: an F1 is recomputed from pooled confusion
    counts, since averaging F1 values is not meaningful, and everything else is
    averaged. Only configurations with a run in every dataset are eligible; if a
    model has none, it falls back to the best score over the available runs and
    warns.

    Returns {dataset: {model: [result]}} with at most one result per model; an
    empty list (with a warning) marks a dataset where the selected
    configuration has no run.
    """
    method = method or THRESHOLD_METHOD
    metric = metric_list(metric)
    datasets = list(by_dataset)
    per_model = {}  # model -> hp_key -> {dataset: result}
    for ds, by_model in by_dataset.items():
        for model, results in by_model.items():
            for r in results:
                slot = per_model.setdefault(model, {}).setdefault(_hp_key(r), {})
                if ds not in slot or _rank_key(r, metric, method) < _rank_key(slot[ds], metric, method):
                    slot[ds] = r

    selected = {ds: {} for ds in datasets}
    for model, configs in per_model.items():
        complete = {k: v for k, v in configs.items() if len(v) == len(datasets)}
        pool = complete or configs
        if not complete:
            print(
                f"Warning: {model}: no hyperparameter set has runs in all "
                f"datasets; selecting by score over available runs."
            )

        def _score(key):
            """Aggregate each metric term over the datasets, best first."""
            results = list(pool[key].values())
            score = []
            for name in metric:
                if name == "f1":
                    # An average of F1 values is not meaningful; recompute it
                    # from the confusion counts pooled over the datasets.
                    pooled = _pooled_f1(results, method)
                    score.append(float("inf") if pooled is None else -pooled)
                    continue
                values = [_metric_value(r, _metric_path(name, method)) for r in results]
                values = [v for v in values if v is not None]
                if not values:
                    score.append(float("inf"))
                else:
                    mean = sum(values) / len(values)
                    score.append(mean if name in LOWER_IS_BETTER else -mean)
            return tuple(score)

        best_key = min(pool, key=_score)
        for ds in datasets:
            entry = configs[best_key].get(ds)
            if entry is None:
                print(
                    f"Warning: {model}: selected hyperparameters have no run "
                    f"for dataset {ds}."
                )
                selected[ds][model] = []
            else:
                selected[ds][model] = [entry]
    return selected


def _best_result(results, metric, method=None):
    """Return the best of `results` by `metric`, read from the `method` block.

    Ranking is `_rank_key`, so a multi-part metric such as ('p_latency', 'fpr',
    'f1') is resolved term by term. A result missing every term ranks last but
    is still returned when it is all there is, so its other columns render
    instead of blanking the row to N/A.
    """
    if not results:
        return None
    method = method or THRESHOLD_METHOD
    return min(results, key=lambda r: _rank_key(r, metric, method))


def _fmt(value):
    """Format a numeric value for display (scientific notation below 1e-3)."""
    if value is None:
        return "N/A"
    try:
        v = float(value)
        if math.isnan(v):
            return "NaN"
        if v == int(v) and abs(v) < 1e15:
            return str(int(v))
        if abs(v) < 1e-3:
            return f"{v:.3e}"
        return f"{v:.4f}"
    except (TypeError, ValueError):
        return str(value)


def _is_unlabeled(result):
    """Return True when a result has no positive labels (pot.TP + pot.FN == 0).

    Returns False if the pot section or the required keys are absent.
    """
    pot = result.get("pot")
    if not isinstance(pot, dict):
        return False
    tp = pot.get("TP")
    fn = pot.get("FN")
    if tp is None or fn is None:
        return False
    try:
        return float(tp) + float(fn) == 0
    except (TypeError, ValueError):
        return False


def _detected(result):
    """Return detected count (pot.TP + pot.FP), or None if keys absent."""
    pot = result.get("pot")
    if not isinstance(pot, dict):
        return None
    tp = pot.get("TP")
    fp = pot.get("FP")
    if tp is None or fp is None:
        return None
    try:
        return float(tp) + float(fp)
    except (TypeError, ValueError):
        return None


def _detection_rate(result):
    """Return detected / total (TP+FP+TN+FN), or None if keys absent or denominator zero."""
    pot = result.get("pot")
    if not isinstance(pot, dict):
        return None
    tp = pot.get("TP")
    fp = pot.get("FP")
    tn = pot.get("TN")
    fn = pot.get("FN")
    if any(v is None for v in (tp, fp, tn, fn)):
        return None
    try:
        total = float(tp) + float(fp) + float(tn) + float(fn)
        if total == 0:
            return None
        return (float(tp) + float(fp)) / total
    except (TypeError, ValueError):
        return None


def _col_value(result, path_or_fn):
    """Return a column value from a result dict.

    `path_or_fn` may be a dotted path string or a callable taking the result dict.
    """
    if callable(path_or_fn):
        return path_or_fn(result)
    return _get(result, path_or_fn)


def _sort_rows(rows, order, lower_is_better, key="model"):
    """Sort ``(sort_value, row, ...)`` tuples in place.

    With `order` (``{model: position}``) the report's one model order wins;
    without it, rows fall back to sorting by their own metric value.
    """
    if order is not None:
        rows.sort(key=lambda t: order.get(t[1][key], len(order)))
    else:
        rows.sort(key=lambda t: t[0], reverse=not lower_is_better)


def _sort_grouped(grouped, order, lower_is_better):
    """`_sort_rows` for `_build_summary`'s ``(sort_value, method_rows, hp_row)`` tuples."""
    if order is not None:
        grouped.sort(key=lambda t: order.get(t[2]["row"], len(order)))
    else:
        grouped.sort(key=lambda t: t[0], reverse=not lower_is_better)


def _build_summary(by_model, metric, unlabeled=False, order=None):
    """Build metric rows and hyperparameter rows for best result per model.

    In labeled mode (unlabeled=False):
        Each model produces two metric rows — one for `pot`, one for `oracle` —
        where each numeric column appears twice: raw (point-wise) and PA
        (point-adjusted, i.e. segment-expanded). The row label is "<model> / <method>".

    In unlabeled mode (unlabeled=True):
        Each model produces a single row with columns: detected, detection rate,
        threshold, calibration_loss, eval_time.

    Sort order is by the row whose metric path matches the --metric argument.
    Returns (metric_rows, hp_rows, all_hp_keys, display_columns).
    """
    metric_rows = []
    hp_rows = []
    all_hp_keys = []
    seen_hp = set()

    # Collect hyperparameter keys from *every* result, not just the best. This
    # ensures any column that's ever been swept (e.g. `epochs` added later)
    # appears in the table even when a given model's best result predates it.
    for results in by_model.values():
        for r in results:
            for k in r.get("applied_hyperparameters", {}).keys():
                if k not in seen_hp:
                    all_hp_keys.append(k)
                    seen_hp.add(k)

    # Single fallback metric for the unlabeled path, which has no detections to
    # rank by; everywhere else `order` decides.
    sort_metric = _table_sort_metric(metric)
    lower_is_better = sort_metric in LOWER_IS_BETTER
    missing_sort_val = float("inf") if lower_is_better else float("-inf")

    if unlabeled:
        display_columns = [
            "detected",
            "detection rate",
            "threshold",
            "calibration_loss",
            "eval_time",
        ]
        grouped = []
        for model, results in by_model.items():
            best = _best_result(results, metric)
            row = {"row": model}
            if best:
                row["detected"] = _fmt(_detected(best))
                row["detection rate"] = _fmt(_detection_rate(best))
                row["threshold"] = _fmt(_get(best, "pot.threshold"))
                row["calibration_loss"] = _fmt(_get(best, "calibration_loss"))
                row["eval_time"] = _fmt(_get(best, "eval_time"))
            else:
                for c in display_columns:
                    row[c] = "N/A"
            try:
                sort_val = float(_get(best, sort_metric)) if best else missing_sort_val
                if math.isnan(sort_val):
                    sort_val = missing_sort_val
            except (TypeError, ValueError):
                sort_val = missing_sort_val
            h_row = {"row": model}
            hp = best.get("applied_hyperparameters", {}) if best else {}
            for k in all_hp_keys:
                h_row[k] = _fmt(hp[k]) if k in hp else ""
            grouped.append((sort_val, [row], h_row))
        _sort_grouped(grouped, order, lower_is_better)
        for _, rows, h_row in grouped:
            metric_rows.extend(rows)
            hp_rows.append(h_row)
        return metric_rows, hp_rows, all_hp_keys, display_columns

    # Labeled mode
    display_columns = DISPLAY_COLUMNS

    # Order by the selection metric when comparable across models (dotted
    # metrics like oracle.f1); otherwise fall back to SUMMARY_SORT_METRIC.
    table_sort = _table_sort_metric(metric)
    lower_is_better = table_sort in LOWER_IS_BETTER
    missing_sort_val = float("inf") if lower_is_better else float("-inf")
    sort_method, sort_sep, sort_metric = table_sort.partition(".")

    grouped = []  # list of (sort_value, [method_rows], hp_row)
    for model, results in by_model.items():
        best = _best_result(results, metric)
        method_rows = []
        sort_val = missing_sort_val
        for method in METHODS:
            row = {"row": f"{model} / {method}"}
            for m in METHOD_METRICS:
                row[m] = _fmt(_get(best, f"{method}.{m}")) if best else "N/A"
                row[f"{m} (PA)"] = (
                    _fmt(_get(best, f"{method}_expanded.{m}")) if best else "N/A"
                )
            row["eval_time"] = _fmt(_get(best, "eval_time")) if best else "N/A"
            row["calibration_loss"] = (
                _fmt(_get(best, "calibration_loss")) if best else "N/A"
            )
            if sort_sep and method == sort_method:
                try:
                    sv = (
                        float(row[sort_metric])
                        if row.get(sort_metric) not in ("N/A", "NaN", None)
                        else missing_sort_val
                    )
                    if math.isnan(sv):
                        sv = missing_sort_val
                    sort_val = sv
                except (TypeError, ValueError, KeyError):
                    sort_val = missing_sort_val
            method_rows.append(row)
        # Handle dotless metric (e.g. 'calibration_loss')
        if not sort_sep and best:
            try:
                sv = (
                    float(_get(best, sort_metric))
                    if _get(best, sort_metric) is not None
                    else missing_sort_val
                )
                if math.isnan(sv):
                    sv = missing_sort_val
                sort_val = sv
            except (TypeError, ValueError):
                sort_val = missing_sort_val
        h_row = {"row": model}
        hp = best.get("applied_hyperparameters", {}) if best else {}
        for k in all_hp_keys:
            h_row[k] = _fmt(hp[k]) if k in hp else ""
        grouped.append((sort_val, method_rows, h_row))

    _sort_grouped(grouped, order, lower_is_better)
    for _, method_rows, h_row in grouped:
        metric_rows.extend(method_rows)
        hp_rows.append(h_row)

    return metric_rows, hp_rows, all_hp_keys, display_columns


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Report: {{ dataset }}</title>
<style>
  body { font-family: Arial, sans-serif; margin: 2em; background: #f9f9f9; color: #222; }
  h1 { border-bottom: 2px solid #444; padding-bottom: 0.3em; }
  h2 { margin-top: 2em; color: #333; }
  table { border-collapse: collapse; margin-bottom: 1em; background: white; }
  th { background: #3a5a8a; color: white; padding: 6px 10px; text-align: left; }
  td { padding: 5px 10px; border-bottom: 1px solid #ddd; }
  tr:hover td { background: #f0f4ff; }
  .section { margin-bottom: 3em; }
</style>
</head>
<body>
<h1>Experiment Report &mdash; Dataset: {{ dataset }}</h1>
<p>Best result per model by: <strong>{{ metric }}</strong></p>

<div class="section">
<h2>Metrics</h2>
<table>
  <thead>
    <tr>
      <th>Model / method</th>
      {% for m in display_columns %}<th>{{ m }}</th>{% endfor %}
    </tr>
  </thead>
  <tbody>
    {% for row in metric_rows %}
    <tr>
      <td>{{ row.row }}</td>
      {% for m in display_columns %}<td>{{ row[m] }}</td>{% endfor %}
    </tr>
    {% endfor %}
  </tbody>
</table>
</div>

<div class="section">
<h2>Hyperparameters</h2>
<table>
  <thead>
    <tr>
      <th>Model</th>
      {% for k in hp_keys %}<th>{{ k }}</th>{% endfor %}
    </tr>
  </thead>
  <tbody>
    {% for row in hp_rows %}
    <tr>
      <td>{{ row.row }}</td>
      {% for k in hp_keys %}<td>{{ row[k] }}</td>{% endfor %}
    </tr>
    {% endfor %}
  </tbody>
</table>
</div>

</body>
</html>
"""


def _generate_html(dataset, metric, by_model, output_path, unlabeled=False, order=None):
    metric_rows, hp_rows, hp_keys, display_columns = _build_summary(
        by_model, metric, unlabeled=unlabeled, order=order
    )
    template = Template(_HTML_TEMPLATE)
    html = template.render(
        dataset=dataset,
        metric=metric_display(metric),
        display_columns=display_columns,
        metric_rows=metric_rows,
        hp_rows=hp_rows,
        hp_keys=hp_keys,
    )
    with open(output_path, "w") as f:
        f.write(html)
    print(f"HTML report written to {output_path}")


# ---------------------------------------------------------------------------
# PDF generation
# ---------------------------------------------------------------------------


def _draw_table_page(pdf, title, columns, rows):
    """Draw a single table as a matplotlib figure and save to the PDF."""
    n_rows = len(rows)
    n_cols = len(columns)

    fig_height = max(2.5, 0.35 * (n_rows + 2))
    fig, ax = plt.subplots(figsize=(max(10, 1.5 * n_cols), fig_height))
    ax.axis("off")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8, loc="left")

    cell_data = [[str(row.get(c, "")) for c in columns] for row in rows]
    table = ax.table(
        cellText=cell_data,
        colLabels=columns,
        colWidths=[1.0 / n_cols] * n_cols,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.3)

    for j in range(n_cols):
        cell = table[0, j]
        cell.set_facecolor("#3a5a8a")
        cell.set_text_props(color="white", fontweight="bold")

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _generate_pdf(dataset, metric, by_model, output_path, unlabeled=False, order=None):
    metric_rows, hp_rows, hp_keys, display_columns = _build_summary(
        by_model, metric, unlabeled=unlabeled, order=order
    )

    # Rename internal 'row' key to a display label
    for row in metric_rows:
        row["Model / method"] = row.pop("row")
    for row in hp_rows:
        row["Model"] = row.pop("row")

    with PdfPages(output_path) as pdf:
        _draw_table_page(
            pdf,
            title=f"Metrics — Dataset: {dataset}  (best by {metric_display(metric)})",
            columns=["Model / method"] + display_columns,
            rows=metric_rows,
        )
        _draw_table_page(
            pdf,
            title=f"Hyperparameters — Dataset: {dataset}  (best by {metric_display(metric)})",
            columns=["Model"] + hp_keys,
            rows=hp_rows,
        )

    print(f"PDF report written to {output_path}")


# ---------------------------------------------------------------------------
# Slide-ready summary (CSV metrics + per-model hyperparameter markdown)
# ---------------------------------------------------------------------------

# Slim metric set for presentation tables. Each entry is (column_label, dotted_path_or_callable).
SLIDE_COLUMNS = [
    ("POT F1", "pot.f1"),
    ("POT F1 (PA)", "pot_expanded.f1"),
    ("POT FPR", "pot.fpr"),
    ("POT latency", "pot.p_latency"),
    ("Oracle F1", "oracle.f1"),
    ("Eval time (s)", "eval_time"),
]

# Default threshold method the report quotes and scales by (--threshold-method
# overrides). Table headers carry no method label, so every generated table and
# plot in one report refers to one and the same method.
THRESHOLD_METHOD = constants.THRESHOLD_METHOD

# Default metric: selects each model's hyperparameters and, unless
# --model-order says otherwise, orders the models in every table and plot.
METRIC = tuple(constants.METRIC)

# Ordering metrics where a smaller value is the better one.
LOWER_IS_BETTER = {"p_latency", "fpr", "threshold", "calibration_loss"}


def method_name(value):
    """Method named by `value`: 'conformal' and 'conformal.f1' both give 'conformal'."""
    name = str(value).split(".")[0]
    return name[: -len("_expanded")] if name.endswith("_expanded") else name


# Header and lookup for every column --table-columns can name. An "_expanded"
# path reaches the point-adjusted block of the same method.
TABLE_COLUMN_SPECS = {
    "f1": ("F1", "{method}.f1"),
    "adjusted_f1": ("Adjusted F1", "{method}_expanded.f1"),
    "precision": ("Precision", "{method}.precision"),
    "recall": ("Recall", "{method}.recall"),
    "fpr": ("FPR", "{method}.fpr"),
    "p_latency": ("Latency", "{method}.p_latency"),
    "threshold": ("Threshold", "{method}.threshold"),
    "calibration_loss": ("Calib. loss", "calibration_loss"),
    "eval_time": ("Eval time (s)", "eval_time"),
}

TABLE_COLUMNS = tuple(constants.TABLE_COLUMNS)
TABLE_BLOCKS = constants.TABLE_BLOCKS


def table_column_list(columns):
    """Normalize a `--table-columns` argument into a tuple of column names."""
    if isinstance(columns, str):
        parts = [p.strip() for p in columns.replace(" ", ",").split(",")]
    else:
        parts = [str(p).strip() for p in columns]
    return tuple(METRIC_ALIASES.get(p, p) for p in parts if p)


def _latex_columns(method=THRESHOLD_METHOD, columns=TABLE_COLUMNS):
    """Summary-table columns reading the `method` threshold block.

    `columns` names entries of `TABLE_COLUMN_SPECS`; an unknown name is read as
    a metric of the method's block so a column can be asked for without being
    listed there. Raises nothing — an absent metric simply renders as N/A.
    """
    specs = []
    for name in table_column_list(columns):
        header, path = TABLE_COLUMN_SPECS.get(name, (name, "{method}." + name))
        specs.append((header, path.format(method=method)))
    return specs

# Cross-case overview: one table per metric, as (file suffix, caption phrase,
# metric name under THRESHOLD_METHOD). Split by metric because a single table of
# model x case x three metrics is unreadable at this width.
OVERVIEW_TABLES = [
    ("f1", "F1", "f1"),
    ("fpr", "false positive rate", "fpr"),
    ("latency", "detection latency in time steps", "p_latency"),
]

# Unlabeled mode: alternate slim column spec.
UNLABELED_SLIDE_COLUMNS = [
    ("Detected", _detected),
    ("Detection rate", _detection_rate),
    ("Threshold", "pot.threshold"),
    ("Eval time (s)", "eval_time"),
]


def _generate_csv(dataset, metric, by_model, output_path, unlabeled=False, order=None):
    """One row per model, slim metric set; selected and ordered by `metric`
    (falling back to F1 ordering for non-comparable metrics)."""
    sort_metric = _table_sort_metric(metric)
    lower_is_better = sort_metric in LOWER_IS_BETTER
    missing_sort_val = float("inf") if lower_is_better else float("-inf")

    slide_cols = UNLABELED_SLIDE_COLUMNS if unlabeled else SLIDE_COLUMNS
    rows = []
    for model, results in by_model.items():
        best = _best_result(results, metric)
        row = {"model": model}
        for label, path_or_fn in slide_cols:
            if best:
                row[label] = _fmt(_col_value(best, path_or_fn))
            else:
                row[label] = "N/A"
        try:
            sort_val = float(_get(best, sort_metric)) if best else missing_sort_val
        except (TypeError, ValueError):
            sort_val = missing_sort_val
        if sort_val != sort_val:  # NaN
            sort_val = missing_sort_val
        rows.append((sort_val, row))
    _sort_rows(rows, order, lower_is_better, key="model")

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model"] + [c for c, _ in slide_cols])
        writer.writeheader()
        for _, row in rows:
            writer.writerow(row)
    print(f"CSV summary written to {output_path}")


def _generate_hp_markdown(dataset, metric, by_model, output_path, order=None):
    """One line per model listing its best-result hyperparameters."""
    sort_metric = _table_sort_metric(metric)
    lower_is_better = sort_metric in LOWER_IS_BETTER
    missing_sort_val = float("inf") if lower_is_better else float("-inf")

    lines = [
        f"# Hyperparameters — {dataset}",
        f"_Best result per model by {metric_display(metric)}_",
        "",
    ]
    entries = []
    for model, results in by_model.items():
        best = _best_result(results, metric)
        hp = best.get("applied_hyperparameters", {}) if best else {}
        try:
            sort_val = float(_get(best, sort_metric)) if best else missing_sort_val
        except (TypeError, ValueError):
            sort_val = missing_sort_val
        if sort_val != sort_val:
            sort_val = missing_sort_val
        parts = ", ".join(f"{k}={_fmt(v)}" for k, v in sorted(hp.items()))
        entries.append((sort_val, {"model": model}, f"- **{model}**: {parts if parts else '(defaults)'}"))
    _sort_rows(entries, order, lower_is_better, key="model")
    lines.extend(line for _, _row, line in entries)

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Hyperparameter summary written to {output_path}")


# ---------------------------------------------------------------------------
# LaTeX generation
# ---------------------------------------------------------------------------


def _latex_escape(s):
    """Escape LaTeX special characters in a string cell."""
    if s is None:
        return ""
    s = str(s)
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    out = []
    for ch in s:
        out.append(repl.get(ch, ch))
    return "".join(out)


def _blocked_rows(rows, blocks):
    """Split `rows` into `blocks` side-by-side columns, filled top to bottom.

    Returns a list of tuples, one per printed line, each holding the row of
    every block (None where a block has run out). Column-major, so reading down
    the first block and continuing down the second follows the row order.
    """
    if blocks < 2 or not rows:
        return [(row,) for row in rows]
    height = -(-len(rows) // blocks)  # ceiling division
    columns = [rows[i * height:(i + 1) * height] for i in range(blocks)]
    return [
        tuple(column[i] if i < len(column) else None for column in columns)
        for i in range(height)
    ]


def _latex_tabular(columns, rows, caption=None, label=None, blocks=1):
    """Render a tabular block (no surrounding table float).

    With `blocks` > 1 the column set is repeated that many times side by side,
    separated by a wide gap, and the rows are dealt into them by
    `_blocked_rows` — a narrow table set two-up is half as tall.
    """
    blocks = max(1, blocks)
    one = "l" + "r" * (len(columns) - 1)
    align = r"@{\qquad}".join([one] * blocks)
    header = " & ".join([" & ".join(_latex_escape(c) for c in columns)] * blocks)

    lines = []
    if caption or label:
        lines.append("% " + (caption or "") + (f"  [{label}]" if label else ""))
    lines.append(r"\begin{tabular}{" + align + r"}")
    lines.append(r"\hline")
    lines.append(header + r" \\")
    lines.append(r"\hline")
    for line in _blocked_rows(rows, blocks):
        cells = []
        for row in line:
            cells.extend(_latex_escape(row.get(c, "")) if row else "" for c in columns)
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def _case_id(dataset):
    """Short test-case id for a dataset name: TOL_DNV_1_1 -> '1.1', TOL_2_2_1 -> '2.2.1'."""
    name = dataset
    for prefix in ("TOL_", "DNV_"):
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name.replace("_", ".")


def _case_sort_key(case_id):
    """Natural order for case ids, so 1.2 < 2.1.1 < 2.2.1 < 3.3.2 rather than string order."""
    return [(0, int(part), "") if part.isdigit() else (1, 0, part) for part in case_id.split(".")]


def _generate_overview_latex(by_dataset, metric, output_dir, threshold_method=THRESHOLD_METHOD,
                             model_order=None):
    """Write one model-by-test-case table per entry in `OVERVIEW_TABLES`.

    Rows are models and columns test cases (there are far more models than
    cases). Every table shares one row order — `model_order` again, each of its
    metrics averaged over the cases — so the three can be read side by side.
    Values come from the `threshold_method` block of each model's selected run,
    the same numbers the per-case summary tables show.

    A latency cell is an em dash when the model produced no true positives on
    that case: `segment_latency` charges an undetected segment its full length,
    which would otherwise read as a real (very slow) detection.
    """
    model_order = metric_list(metric if model_order is None else model_order)
    cases = sorted({_case_id(ds) for ds in by_dataset}, key=_case_sort_key)
    best = {}
    for dataset, by_model in by_dataset.items():
        case = _case_id(dataset)
        for model, results in by_model.items():
            result = _best_result(results, metric)
            if result is not None:
                best[(model, case)] = result
    models = {model for model, _ in best}

    def mean_rank_key(model):
        """`_rank_key` with each metric averaged over the cases the model ran on."""
        keys = [
            _rank_key(best[(model, case)], model_order, threshold_method)
            for case in cases
            if (model, case) in best
        ]
        if not keys:
            return tuple(float("inf") for _ in model_order)
        return tuple(sum(values) / len(values) for values in zip(*keys))

    order = sorted(models, key=lambda model: (mean_rank_key(model), model))

    def cell(model, case, name):
        result = best.get((model, case))
        if result is None:
            return "N/A"
        return _fmt_metric(result, f"{threshold_method}.{name}", threshold_method)

    os.makedirs(output_dir, exist_ok=True)
    for suffix, phrase, name in OVERVIEW_TABLES:
        rows = [
            {"model": model, **{case: cell(model, case, name) for case in cases}}
            for model in order
        ]
        output_path = os.path.join(output_dir, f"overview_{suffix}.tex")
        with open(output_path, "w") as f:
            f.write("\n".join([
                f"% Auto-generated by AaltoAD.report — {phrase} per model and test case, "
                f"threshold_method={threshold_method}, rows ordered by mean F1",
                _latex_tabular(["model"] + cases, rows),
                "",
            ]))
        print(f"Overview table written to {output_path}")


def _generate_latex(dataset, metric, by_model, output_path, unlabeled=False,
                    threshold_method=THRESHOLD_METHOD, order=None, columns=TABLE_COLUMNS,
                    blocks=TABLE_BLOCKS):
    """Write a LaTeX file with two tabular blocks (metrics + hyperparameters).

    `threshold_method` names the threshold block the columns are read from and
    `columns` which ones appear (see `_latex_columns`) and `blocks` how many
    models are set side by side per row.
    Designed to be ``\\input``-ed inside a user-provided ``table`` float —
    no float wrapper is emitted.
    """
    sort_metric = _table_sort_metric(metric)
    lower_is_better = sort_metric in LOWER_IS_BETTER
    missing_sort_val = float("inf") if lower_is_better else float("-inf")

    slide_cols = UNLABELED_SLIDE_COLUMNS if unlabeled else _latex_columns(threshold_method, columns)

    # Slim metrics table (oracle-threshold metrics, see LATEX_COLUMNS).
    metric_rows = []
    for model, results in by_model.items():
        best = _best_result(results, metric)
        row = {"model": model}
        for label, path_or_fn in slide_cols:
            if best:
                row[label] = _fmt_metric(best, path_or_fn, threshold_method)
            else:
                row[label] = "N/A"
        try:
            sort_val = float(_get(best, sort_metric)) if best else missing_sort_val
        except (TypeError, ValueError):
            sort_val = missing_sort_val
        if sort_val != sort_val:
            sort_val = missing_sort_val
        metric_rows.append((sort_val, row))
    _sort_rows(metric_rows, order, lower_is_better, key="model")
    metric_rows = [r for _, r in metric_rows]
    metric_cols = ["model"] + [c for c, _ in slide_cols]

    parts = [
        f"% Auto-generated by AaltoAD.report — dataset={dataset}, metric={metric_display(metric)}, "
        f"threshold_method={threshold_method}",
        _latex_tabular(metric_cols, metric_rows, blocks=blocks),
        "",
    ]
    with open(output_path, "w") as f:
        f.write("\n".join(parts))
    print(f"LaTeX summary written to {output_path}")


# ---------------------------------------------------------------------------
# Prediction-error overlay plot
# ---------------------------------------------------------------------------


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
                                    n_models=DEFAULT_PLOT_MODELS, threshold_method=THRESHOLD_METHOD,
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
    model_order = metric_list(metric if model_order is None else model_order)
    series = {}
    ground_truth = None
    best_by_model = {}
    for model, results in by_model.items():
        best = _best_result(results, metric)
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

    # Keep only the `n_models` best models (see `_rank_key`) so the overlay
    # stays readable.
    if models is not None:
        series = {m: series[m] for m in models if m in series}
    elif len(series) > n_models:
        keep = sorted(
            series,
            key=lambda m: _rank_key(best_by_model[m], model_order, threshold_method),
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


# ---------------------------------------------------------------------------
# Per-model plots (best result per model)
# ---------------------------------------------------------------------------


def _generate_model_plots(dataset, metric, by_model, output_dir, blocks_key=None,
                          threshold_method=THRESHOLD_METHOD, downsample_mode=DOWNSAMPLE,
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
        best = _best_result(results, metric)
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


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _apply_shared_thresholds(by_dataset, results_folder, conformal_q=constants.CONFORMAL_Q,
                             pool_baselines=POOL_BASELINES):
    """Attach conformal/pot/oracle blocks to every result, in place.

    With `pool_baselines`, one threshold per configuration is fit on the
    calibration data of every dataset pooled together — which assumes the
    baselines are alike, and lets a contaminated one set the threshold for all
    the rest. By default each dataset is fit on its own baseline instead, by
    running the same machinery once per dataset; the blocks still land where
    the pooled ones would, so the combined report and the shared plots keep
    working on separately fitted thresholds.
    """
    if not pool_baselines and len(by_dataset) > 1:
        for dataset, by_model in by_dataset.items():
            _fit_threshold_blocks({dataset: by_model}, results_folder, conformal_q)
        return
    _fit_threshold_blocks(by_dataset, results_folder, conformal_q)


def _fit_threshold_blocks(by_dataset, results_folder, conformal_q=constants.CONFORMAL_Q):
    """Fit and attach threshold blocks over the given datasets, pooling their calibration.

    Only configurations (model + hyperparameters) with a run in every dataset
    are processed; per dataset the lowest-``calibration_loss`` run represents
    the configuration. Each processed result gets its shared blocks under
    ``result["shared"]`` and ``shared_threshold = True``; the local pot/oracle
    blocks stay untouched and remain what per-dataset tables show, while the
    conformal blocks, which have no local counterpart, are written to the top
    level as well (see ``shared.apply_blocks``). POT uses each configuration's
    swept ``q``; the conformal threshold uses `conformal_q`. Fits are cached
    under ``results_folder/_shared_thresholds/`` keyed by the CSV modification
    times. With a single dataset the "pool" is that dataset's own calibration,
    which is how `_apply_shared_thresholds` fits separate baselines.
    """

    datasets = list(by_dataset)
    groups = shared.group_configurations(
        by_dataset, _hp_key, lambda rs: _best_result(rs, "calibration_loss"))
    complete = {k: v for k, v in groups.items() if set(v) == set(datasets)}
    print(f"Fitting thresholds for {len(complete)} configurations on the calibration of {datasets}")

    cache_file = shared.cache_path(results_folder, datasets)
    cache = shared.load_cache(cache_file)
    shared_count, skipped_count = {}, {}
    for (model, _), members in groups.items():
        if set(members) != set(datasets):
            skipped_count[model] = skipped_count.get(model, 0) + 1

    for (model, hp_key), results_by_dataset in tqdm(complete.items(), desc="shared thresholds", unit="config"):
        q = next(iter(results_by_dataset.values())).get("applied_hyperparameters", {}).get("q", 1e-5)
        constants.initialize(datasets[0], model)
        blocks = shared.shared_blocks_cached(cache, model, hp_key, results_by_dataset, q,
                                             constants.level, conformal_q)
        if blocks is None:
            skipped_count[model] = skipped_count.get(model, 0) + 1
            continue
        shared.apply_blocks(results_by_dataset, blocks)
        shared_count[model] = shared_count.get(model, 0) + 1

    shared.save_cache(cache_file, cache)
    for model in sorted(set(shared_count) | set(skipped_count)):
        print(f"  {model}: {shared_count.get(model, 0)} configuration(s) shared, "
              f"{skipped_count.get(model, 0)} skipped (missing runs or CSVs)")


def generate_report(dataset, metric=METRIC, results_folder="results",
                    n_plot_models=DEFAULT_PLOT_MODELS, threshold_method=THRESHOLD_METHOD,
                    model_order=None, conformal_q=constants.CONFORMAL_Q, pool_baselines=POOL_BASELINES,
                    downsample_mode=DOWNSAMPLE, downsample_window=None,
                    table_columns=TABLE_COLUMNS, table_blocks=TABLE_BLOCKS):
    """Generate HTML, PDF, CSV, and hyperparameter-markdown reports.

    `dataset` may be a single name, a comma-separated string, or a list of
    names. With several datasets, one hyperparameter set per model is selected
    by the best *sum* of `metric` over all of them, and each dataset's report
    shows that shared configuration with its local (per-dataset) thresholds.
    POT and oracle thresholds are also fit jointly across the datasets (see
    `_apply_shared_thresholds`); those shared thresholds drive the
    reports/combined/ summary (confusion counts pooled over the datasets) and a
    second set of plots per dataset. Files are saved in reports/{dataset}/.
    `n_plot_models` is how many models each prediction-error overlay shows; the
    models are the best `n_plot_models` by `metric` on that dataset.
    A multi-dataset run also writes the cross-case overview tables to
    reports/overview_*.tex (see `_generate_overview_latex`).

    `threshold_method` is the threshold every table quotes and every plot is
    scaled by. `metric` is a tuple of metric names inside it (see
    `metric_list`): it selects each model's run — aggregated over the datasets,
    an F1 recomputed from pooled counts rather than averaged — and orders the
    models everywhere, unless `model_order` gives a separate ordering.
    `conformal_q` is the false alarm rate the conformal threshold targets,
    `pool_baselines` whether one threshold is fit across all the datasets'
    calibration data instead of one per dataset, and
    `downsample_mode`/`downsample_window` decide how plotted series are reduced,
    `table_columns` which columns the LaTeX summary shows and `table_blocks`
    how many models it sets side by side.
    """
    # One metric does both jobs unless the caller separates them.
    metric = metric_list(metric)
    model_order = metric if model_order is None else metric_list(model_order)

    if isinstance(dataset, str):
        datasets = [d for d in dataset.split(",") if d]
    else:
        datasets = list(dataset)

    by_dataset = {}
    for ds in datasets:
        by_model = _load_results(ds, results_folder)
        if not by_model:
            print(f'No results found for dataset "{ds}" in {results_folder}/')
            continue
        by_dataset[ds] = by_model
    if not by_dataset:
        return

    # Always fit the shared thresholds: the conformal threshold is defined by
    # the calibration sample it is fitted on, and this is where that fit happens
    # (with one dataset listed, "shared across the datasets" is just that one).
    _apply_shared_thresholds(by_dataset, results_folder, conformal_q, pool_baselines)

    selected = _select_shared_best(by_dataset, metric, threshold_method)
    # Each dataset's overlay picks its own best `n_plot_models` (plot_models=None),
    # so a model that only wins on one case still shows up there. The combined
    # report is a single pooled ranking, so it gets the pooled top list.
    for ds in by_dataset:
        _generate_dataset_report(ds, metric, selected[ds],
                                 shared_plots=len(by_dataset) > 1, n_plot_models=n_plot_models,
                                 threshold_method=threshold_method, model_order=model_order,
                                 downsample_mode=downsample_mode, downsample_window=downsample_window,
                                 table_columns=table_columns, table_blocks=table_blocks)
    if len(by_dataset) > 1:
        _generate_overview_latex(selected, metric, "reports", threshold_method=threshold_method,
                                 model_order=model_order)
        pooled = _pooled_by_model(selected)
        _generate_dataset_report("combined", metric, pooled,
                                 plot_models=_top_models(pooled, metric, n=n_plot_models,
                                                         threshold_method=threshold_method,
                                                         model_order=model_order),
                                 n_plot_models=n_plot_models, threshold_method=threshold_method,
                                 model_order=model_order, downsample_mode=downsample_mode,
                                 downsample_window=downsample_window, table_columns=table_columns,
                                 table_blocks=table_blocks)


def _order_models(by_model, metric, threshold_method=THRESHOLD_METHOD, model_order=None):
    """Model names in the report's display order, best first.

    `metric` selects each model's best run (its hyperparameters); the ranking
    across models reads `model_order` from the `threshold_method` block, or
    `metric` itself when no separate ordering is given (see `_rank_key`). Every
    table and plot in a report uses this one order.
    """
    model_order = metric_list(metric if model_order is None else model_order)
    return sorted(
        by_model,
        key=lambda m: _rank_key(_best_result(by_model[m], metric, threshold_method), model_order, threshold_method),
    )


def _order_rank(by_model, metric, threshold_method=THRESHOLD_METHOD, model_order=None):
    """``{model: position}`` in the display order, for sorting already-built table rows."""
    return {m: i for i, m in enumerate(_order_models(by_model, metric, threshold_method, model_order))}


def _top_models(by_model, metric, n=DEFAULT_PLOT_MODELS, threshold_method=THRESHOLD_METHOD,
                model_order=None):
    """Names of the `n` models a plot should show, in `_order_models` order."""
    return _order_models(by_model, metric, threshold_method, model_order)[:n]


def _pooled_by_model(selected):
    """Pool each model's selected per-dataset results into one combined result, using the shared-threshold blocks."""
    from AaltoAD.thresholds.pooled import pooled_result

    models = {m for by_model in selected.values() for m in by_model}
    pooled = {}
    for model in sorted(models):
        combined = pooled_result([r for by_model in selected.values() for r in by_model.get(model, [])], blocks_key="shared")
        if combined:
            pooled[model] = [combined]
    return pooled


def _generate_dataset_report(dataset, metric, by_model, plot_models=None, shared_plots=False,
                             n_plot_models=DEFAULT_PLOT_MODELS, threshold_method=THRESHOLD_METHOD,
                             model_order=None, downsample_mode=DOWNSAMPLE, downsample_window=None,
                             table_columns=TABLE_COLUMNS, table_blocks=TABLE_BLOCKS):
    """Write all report files for one dataset from its (pre-selected) results.

    `plot_models` fixes the models shown in the overlay plot (see
    `_generate_prediction_error_plot`). With `shared_plots`, a second set of
    plots scaled by the shared thresholds is written next to the local ones
    (``prediction_errors_shared.png`` and ``plots_shared/``).
    """
    # Determine unlabeled flag: True when at least one result has a pot confusion
    # matrix AND all results that have one satisfy _is_unlabeled.
    results_with_pot = [
        r
        for results in by_model.values()
        for r in results
        if isinstance(r.get("pot"), dict) and r["pot"].get("TP") is not None
    ]
    unlabeled = bool(results_with_pot) and all(
        _is_unlabeled(r) for r in results_with_pot
    )

    dataset_dir = os.path.join("reports", dataset)

    os.makedirs(dataset_dir, exist_ok=True)
    html_path = os.path.join(dataset_dir, f"report.html")
    pdf_path = os.path.join(dataset_dir, f"report.pdf")
    csv_path = os.path.join(dataset_dir, f"summary.csv")
    hp_path = os.path.join(dataset_dir, f"hyperparams.md")
    tex_path = os.path.join(dataset_dir, f"summary.tex")
    plot_path = os.path.join(dataset_dir, f"prediction_errors.png")
    plots_dir = os.path.join(dataset_dir, "plots")

    # One model order for every table and plot of this dataset. Unlabeled runs
    # have no detections to rank, so they keep their own metric sort.
    order = None if unlabeled else _order_rank(by_model, metric, threshold_method, model_order)

    _generate_html(dataset, metric, by_model, html_path, unlabeled=unlabeled, order=order)
    _generate_pdf(dataset, metric, by_model, pdf_path, unlabeled=unlabeled, order=order)
    _generate_csv(dataset, metric, by_model, csv_path, unlabeled=unlabeled, order=order)
    _generate_hp_markdown(dataset, metric, by_model, hp_path, order=order)
    _generate_latex(dataset, metric, by_model, tex_path, unlabeled=unlabeled,
                    threshold_method=threshold_method, order=order, columns=table_columns,
                    blocks=table_blocks)
    _generate_prediction_error_plot(dataset, metric, by_model, plot_path, models=plot_models,
                                    n_models=n_plot_models, threshold_method=threshold_method,
                                    model_order=model_order, downsample_mode=downsample_mode,
                                    downsample_window=downsample_window)
    _generate_model_plots(dataset, metric, by_model, plots_dir, threshold_method=threshold_method,
                          downsample_mode=downsample_mode, downsample_window=downsample_window)
    if shared_plots:
        shared_plot_path = os.path.join(dataset_dir, "prediction_errors_shared.png")
        _generate_prediction_error_plot(dataset, metric, by_model, shared_plot_path, models=plot_models,
                                        blocks_key="shared", n_models=n_plot_models,
                                        threshold_method=threshold_method, model_order=model_order,
                                        downsample_mode=downsample_mode,
                                        downsample_window=downsample_window)
        _generate_model_plots(dataset, metric, by_model, os.path.join(dataset_dir, "plots_shared"),
                              blocks_key="shared", threshold_method=threshold_method,
                              downsample_mode=downsample_mode, downsample_window=downsample_window)
