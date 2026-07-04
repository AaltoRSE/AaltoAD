"""Generate HTML and PDF reports from hyperparameter sweep result JSON files."""

import csv
import json
import math
import os
from glob import glob

import matplotlib

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

# Metrics where lower is better (affects _best_result and sort order).
LOWER_IS_BETTER = {"calibration_loss"}

# Models are *selected* (best hyperparameters) by the `metric` argument
# (default calibration_loss), but the summary tables are *ordered* by F1.
# Calibration loss is not comparable across models, so it makes a poor ranking;
# F1 reflects detection quality and is comparable.
SUMMARY_SORT_METRIC = "pot.f1"


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


def _best_result(results, metric):
    """Return the result dict with the best non-NaN metric value.

    For metrics in LOWER_IS_BETTER, returns the result with the minimum value;
    otherwise returns the result with the maximum value.
    `metric` may be a dotted path like 'pot.f1' to reach into the nested schema.
    """

    def _val(r):
        v = _get(r, metric)
        if v is None:
            return float("nan")
        try:
            return float(v)
        except (TypeError, ValueError):
            return float("nan")

    valid = [r for r in results if not math.isnan(_val(r))]
    if not valid:
        # No result has this metric (e.g. calibration_loss missing on
        # old-style results that used the training threshold). Fall back to
        # the first result so its available metrics still render instead of
        # blanking the whole row to N/A.
        return results[0] if results else None
    if metric in LOWER_IS_BETTER:
        return min(valid, key=_val)
    return max(valid, key=_val)


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


def _build_summary(by_model, metric, unlabeled=False):
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

    # Determine lower-is-better for missing-value sentinel in sort.
    lower_is_better = metric in LOWER_IS_BETTER
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
                sort_val = float(_get(best, metric)) if best else missing_sort_val
                if math.isnan(sort_val):
                    sort_val = missing_sort_val
            except (TypeError, ValueError):
                sort_val = missing_sort_val
            h_row = {"row": model}
            hp = best.get("applied_hyperparameters", {}) if best else {}
            for k in all_hp_keys:
                h_row[k] = _fmt(hp[k]) if k in hp else ""
            grouped.append((sort_val, [row], h_row))
        grouped.sort(key=lambda t: t[0], reverse=not lower_is_better)
        for _, rows, h_row in grouped:
            metric_rows.extend(rows)
            hp_rows.append(h_row)
        return metric_rows, hp_rows, all_hp_keys, display_columns

    # Labeled mode
    display_columns = DISPLAY_COLUMNS

    # Order by F1 (detection quality), independent of the selection metric.
    lower_is_better = SUMMARY_SORT_METRIC in LOWER_IS_BETTER
    missing_sort_val = float("inf") if lower_is_better else float("-inf")
    sort_method, sort_sep, sort_metric = SUMMARY_SORT_METRIC.partition(".")

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
                    float(_get(best, metric))
                    if _get(best, metric) is not None
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

    grouped.sort(key=lambda t: t[0], reverse=not lower_is_better)
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


def _generate_html(dataset, metric, by_model, output_path, unlabeled=False):
    metric_rows, hp_rows, hp_keys, display_columns = _build_summary(
        by_model, metric, unlabeled=unlabeled
    )
    template = Template(_HTML_TEMPLATE)
    html = template.render(
        dataset=dataset,
        metric=metric,
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


def _generate_pdf(dataset, metric, by_model, output_path, unlabeled=False):
    metric_rows, hp_rows, hp_keys, display_columns = _build_summary(
        by_model, metric, unlabeled=unlabeled
    )

    # Rename internal 'row' key to a display label
    for row in metric_rows:
        row["Model / method"] = row.pop("row")
    for row in hp_rows:
        row["Model"] = row.pop("row")

    with PdfPages(output_path) as pdf:
        _draw_table_page(
            pdf,
            title=f"Metrics — Dataset: {dataset}  (best by {metric})",
            columns=["Model / method"] + display_columns,
            rows=metric_rows,
        )
        _draw_table_page(
            pdf,
            title=f"Hyperparameters — Dataset: {dataset}  (best by {metric})",
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
    ("Calib. loss", "calibration_loss"),
    ("Eval time (s)", "eval_time"),
]

# Unlabeled mode: alternate slim column spec.
UNLABELED_SLIDE_COLUMNS = [
    ("Detected", _detected),
    ("Detection rate", _detection_rate),
    ("Threshold", "pot.threshold"),
    ("Calib. loss", "calibration_loss"),
    ("Eval time (s)", "eval_time"),
]


def _generate_csv(dataset, metric, by_model, output_path, unlabeled=False):
    """One row per model, slim metric set; selected by `metric`, ordered by F1."""
    sort_metric = metric if unlabeled else SUMMARY_SORT_METRIC
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
    rows.sort(key=lambda t: t[0], reverse=not lower_is_better)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model"] + [c for c, _ in slide_cols])
        writer.writeheader()
        for _, row in rows:
            writer.writerow(row)
    print(f"CSV summary written to {output_path}")


def _generate_hp_markdown(dataset, metric, by_model, output_path):
    """One line per model listing its best-result hyperparameters."""
    lower_is_better = metric in LOWER_IS_BETTER
    missing_sort_val = float("inf") if lower_is_better else float("-inf")

    lines = [
        f"# Hyperparameters — {dataset}",
        f"_Best result per model by {metric}_",
        "",
    ]
    entries = []
    for model, results in by_model.items():
        best = _best_result(results, metric)
        hp = best.get("applied_hyperparameters", {}) if best else {}
        try:
            sort_val = float(_get(best, metric)) if best else missing_sort_val
        except (TypeError, ValueError):
            sort_val = missing_sort_val
        if sort_val != sort_val:
            sort_val = missing_sort_val
        parts = ", ".join(f"{k}={_fmt(v)}" for k, v in sorted(hp.items()))
        entries.append((sort_val, f"- **{model}**: {parts if parts else '(defaults)'}"))
    entries.sort(key=lambda t: t[0], reverse=not lower_is_better)
    lines.extend(line for _, line in entries)

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


def _latex_tabular(columns, rows, caption=None, label=None):
    """Render a tabular block (no surrounding table float)."""
    align = "l" + "r" * (len(columns) - 1)
    lines = []
    if caption or label:
        lines.append("% " + (caption or "") + (f"  [{label}]" if label else ""))
    lines.append(r"\begin{tabular}{" + align + r"}")
    lines.append(r"\hline")
    lines.append(" & ".join(_latex_escape(c) for c in columns) + r" \\")
    lines.append(r"\hline")
    for row in rows:
        lines.append(
            " & ".join(_latex_escape(row.get(c, "")) for c in columns) + r" \\"
        )
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def _generate_latex(dataset, metric, by_model, output_path, unlabeled=False):
    """Write a LaTeX file with two tabular blocks (metrics + hyperparameters).

    Designed to be ``\\input``-ed inside a user-provided ``table`` float —
    no float wrapper is emitted.
    """
    sort_metric = metric if unlabeled else SUMMARY_SORT_METRIC
    lower_is_better = sort_metric in LOWER_IS_BETTER
    missing_sort_val = float("inf") if lower_is_better else float("-inf")

    slide_cols = UNLABELED_SLIDE_COLUMNS if unlabeled else SLIDE_COLUMNS

    # Slim metrics table (same columns as CSV).
    metric_rows = []
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
        if sort_val != sort_val:
            sort_val = missing_sort_val
        metric_rows.append((sort_val, row))
    metric_rows.sort(key=lambda t: t[0], reverse=not lower_is_better)
    metric_rows = [r for _, r in metric_rows]
    metric_cols = ["model"] + [c for c, _ in slide_cols]

    parts = [
        f"% Auto-generated by TranAD.report — dataset={dataset}, metric={metric}",
        _latex_tabular(metric_cols, metric_rows),
        "",
    ]
    with open(output_path, "w") as f:
        f.write("\n".join(parts))
    print(f"LaTeX summary written to {output_path}")


# ---------------------------------------------------------------------------
# PDF prediction-error plot
# ---------------------------------------------------------------------------


def _generate_prediction_error_plot(dataset, metric, by_model, output_path):
    """Overlay each model's best-result prediction_error for a dataset.

    For each model, take its best result (by `metric`), read the matching
    ``*_labels.csv``, and plot the ``prediction_error`` column scaled by that
    model's POT threshold so the threshold is 1 — any peak above 1 is flagged
    anomalous. All models are drawn on one axis with a dashed line at y=1 and
    ground-truth anomaly regions shaded once. The figure is saved as both a
    vector PDF (for ``\\includegraphics`` in a LaTeX/Overleaf document) and an
    SVG, sharing the basename of ``output_path``.
    """
    series = {}
    ground_truth = None
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
        # Scale by the POT threshold so the threshold maps to 1.
        threshold = _get(best, "pot.threshold")
        try:
            threshold = float(threshold)
        except (TypeError, ValueError):
            threshold = None
        if not threshold or threshold <= 0:
            print(f"No usable POT threshold for {model}; skipping in PDF plot.")
            continue
        series[model] = df["prediction_error"].reset_index(drop=True) / threshold
        # Ground truth is shared across models for a dataset; capture it once.
        if ground_truth is None and "ground_truth" in df.columns:
            ground_truth = df["ground_truth"].reset_index(drop=True)

    if not series:
        print(f"No prediction_error data found for {dataset}; skipping PDF plot.")
        return

    combined = pd.concat(series, axis=1)

    fig, ax = plt.subplots(figsize=(10, 4))
    combined.plot(ax=ax, ylim=(-0.2, 3.0), linewidth=0.8)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, label="threshold")
    if ground_truth is not None:
        mask = ground_truth.astype(bool)
        ax.fill_between(
            ground_truth.index,
            0,
            1,
            where=mask,
            color="tomato",
            alpha=0.15,
            transform=ax.get_xaxis_transform(),
            label="ground_truth",
        )
    ax.set_xlabel("test step")
    ax.set_ylabel("prediction error / threshold")
    ax.set_title(f"Prediction error — {dataset}")
    ax.legend(loc="best", fontsize=8, ncol=2)
    plt.tight_layout()
    base = os.path.splitext(output_path)[0]
    for fmt in ("pdf", "svg"):
        out = f"{base}.{fmt}"
        fig.savefig(out, format=fmt, bbox_inches="tight")
        print(f"Prediction-error plot written to {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-model SVG plots (best result per model)
# ---------------------------------------------------------------------------


def _generate_model_plots(dataset, metric, by_model, output_dir):
    """Plot the best result per model as prediction error vs. threshold.

    For each model, take its best result (by `metric`), read the matching
    ``*_labels.csv``, and plot the ``prediction_error`` series against the POT
    threshold with ground-truth anomaly regions shaded. Each model is written to
    ``output_dir`` (typically ``reports/<dataset>/plots/``) as both a PDF (for
    ``\\includegraphics`` in a LaTeX/Overleaf document) and an SVG.
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

        cols = ["prediction_error"]
        threshold = _get(best, "pot.threshold")
        if threshold is not None:
            df["threshold"] = threshold
            cols.append("threshold")

        ax = df[cols].plot(figsize=(10, 4), linewidth=0.8)
        if "ground_truth" in df.columns:
            mask = df["ground_truth"].astype(bool)
            ax.fill_between(
                df.index,
                0,
                1,
                where=mask,
                color="tomato",
                alpha=0.15,
                transform=ax.get_xaxis_transform(),
                label="ground_truth",
            )
        ax.set_xlabel("test step")
        ax.set_ylabel("prediction error")
        ax.set_title(f"{model} — {dataset}")
        ax.legend(loc="best", fontsize=8)
        plt.tight_layout()
        for fmt in ("pdf", "svg"):
            out_path = os.path.join(output_dir, f"{model}.{fmt}")
            ax.figure.savefig(out_path, format=fmt, bbox_inches="tight")
            print(f"Model plot written to {out_path}")
        plt.close(ax.figure)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def generate_report(dataset, metric="calibration_loss", results_folder="results"):
    """Generate HTML, PDF, CSV, and hyperparameter-markdown reports for a dataset.

    Files are saved in reports/{dataset}/ (e.g., reports/TOL_1_1/).
    """
    by_model = _load_results(dataset, results_folder)
    if not by_model:
        print(f'No results found for dataset "{dataset}" in {results_folder}/')
        return

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
    plot_path = os.path.join(dataset_dir, f"prediction_errors.pdf")
    plots_dir = os.path.join(dataset_dir, "plots")

    _generate_html(dataset, metric, by_model, html_path, unlabeled=unlabeled)
    _generate_pdf(dataset, metric, by_model, pdf_path, unlabeled=unlabeled)
    _generate_csv(dataset, metric, by_model, csv_path, unlabeled=unlabeled)
    _generate_hp_markdown(dataset, metric, by_model, hp_path)
    _generate_latex(dataset, metric, by_model, tex_path, unlabeled=unlabeled)
    _generate_prediction_error_plot(dataset, metric, by_model, plot_path)
    _generate_model_plots(dataset, metric, by_model, plots_dir)
