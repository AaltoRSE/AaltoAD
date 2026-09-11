"""Generate HTML and PDF reports from hyperparameter sweep result JSON files."""

import csv
import json
import math
import os

from tqdm import tqdm
from glob import glob

import matplotlib
import matplotlib.ticker
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

# Metrics where lower is better (affects _best_result and sort order).
LOWER_IS_BETTER = {"calibration_loss"}

# Models are *selected* (best hyperparameters) by the `metric` argument
# (default calibration_loss). Tables are *ordered* by the selection metric when
# it is comparable across models (a dotted method metric like 'oracle.f1');
# calibration loss is not comparable across models, so it makes a poor ranking
# and tables fall back to F1 instead.
SUMMARY_SORT_METRIC = "pot.f1"


def _table_sort_metric(metric):
    """Metric used to order table rows for a given selection metric."""
    return metric if "." in str(metric) else SUMMARY_SORT_METRIC


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


def _select_shared_best(by_dataset, metric):
    """Select one hyperparameter set per model across all datasets.

    For each model, group results by hyperparameter configuration (keeping the
    best result per dataset within a configuration) and pick the configuration
    with the best score over the listed datasets. When `metric` is an F1
    (e.g. 'pot.f1'), the score is the F1 recomputed from confusion counts
    pooled over the datasets; otherwise it is the sum of `metric`. Only
    configurations with a run in every dataset are eligible; if a model has
    none, it falls back to the best score over the available runs and warns.

    Returns {dataset: {model: [result]}} with at most one result per model; an
    empty list (with a warning) marks a dataset where the selected
    configuration has no run.
    """
    lower = metric in LOWER_IS_BETTER
    datasets = list(by_dataset)
    per_model = {}  # model -> hp_key -> {dataset: (value, result)}
    for ds, by_model in by_dataset.items():
        for model, results in by_model.items():
            for r in results:
                v = _metric_value(r, metric)
                if v is None:
                    continue
                slot = per_model.setdefault(model, {}).setdefault(_hp_key(r), {})
                if ds not in slot or (v < slot[ds][0]) == lower:
                    slot[ds] = (v, r)

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
            values = pool[key].values()
            if metric.endswith(".f1"):
                pooled = _pooled_f1([r for _, r in values], metric[: -len(".f1")])
                if pooled is not None:
                    return pooled
            return sum(v for v, _ in values)

        best_key = min(pool, key=_score) if lower else max(pool, key=_score)
        for ds in datasets:
            entry = configs[best_key].get(ds)
            if entry is None:
                print(
                    f"Warning: {model}: selected hyperparameters have no run "
                    f"for dataset {ds}."
                )
                selected[ds][model] = []
            else:
                selected[ds][model] = [entry[1]]
    return selected


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

# LaTeX summary table: oracle-threshold metrics only, without method labels.
LATEX_COLUMNS = [
    ("F1", "oracle.f1"),
    ("Adjusted F1", "oracle_expanded.f1"),
    ("FPR", "oracle.fpr"),
    ("Latency", "oracle.p_latency"),
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
    """One row per model, slim metric set; selected and ordered by `metric`
    (falling back to F1 ordering for non-comparable metrics)."""
    sort_metric = metric if unlabeled else _table_sort_metric(metric)
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
    sort_metric = metric if unlabeled else _table_sort_metric(metric)
    lower_is_better = sort_metric in LOWER_IS_BETTER
    missing_sort_val = float("inf") if lower_is_better else float("-inf")

    slide_cols = UNLABELED_SLIDE_COLUMNS if unlabeled else LATEX_COLUMNS

    # Slim metrics table (oracle-threshold metrics, see LATEX_COLUMNS).
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
# Prediction-error overlay plot
# ---------------------------------------------------------------------------


def _thresholds(result):
    """Return (pot, oracle) thresholds as floats, None where missing/invalid."""
    parsed = []
    for key in ("pot.threshold", "oracle.threshold"):
        value = _get(result, key)
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = None
        if value is not None and value <= 0:
            value = None
        parsed.append(value)
    return tuple(parsed)


def _plot_threshold_method(metric):
    """Method whose threshold the plots scale by: 'oracle' for oracle.* metrics, else 'pot'."""
    return "oracle" if str(metric).startswith("oracle") else "pot"


def _plot_threshold(result, metric):
    """Threshold used to scale a result's errors in plots, or None if missing/non-positive."""
    pot_thr, oracle_thr = _thresholds(result)
    return oracle_thr if _plot_threshold_method(metric) == "oracle" else pot_thr


def _plot_y_top(values, ground_truth=None):
    """Upper y limit: 99.9th percentile of the range-setting values with headroom, never below 2.

    `values` is a Series (or DataFrame) indexed by time step, calibration at
    negative steps. With `ground_truth` (Series indexed 0..N-1), only the
    calibration rows and the labelled-anomaly rows set the range, so large
    spikes in the recovery phase clip instead of flattening the anomaly.
    """
    if ground_truth is not None:
        anomaly_steps = ground_truth.index[ground_truth.astype(bool)]
        keep = (values.index < 0) | values.index.isin(anomaly_steps)
        if keep.any():
            values = values[keep]
    flat = values.stack() if isinstance(values, pd.DataFrame) else values
    return max(2.0, float(flat.quantile(0.999))) * 1.15


def _format_y_ticks(ax, y_top):
    """Write y ticks in scientific notation on each tick (no corner multiplier) when the range exceeds 1000."""
    if y_top > 1000:
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.1e}"))


def _generate_prediction_error_plot(dataset, metric, by_model, output_path, models=None, blocks_key=None):
    """Overlay each model's best-result prediction_error for a dataset.

    `models`, when given, fixes which models are plotted (and their order)
    instead of the per-dataset top 5 by `metric`; multi-dataset reports pass
    the same list to every dataset so the overlays are comparable.
    `blocks_key` (e.g. ``"shared"``) scales by the thresholds stored under that
    key of each result instead of the local ones; models without it are skipped.

    For each model, take its best result (by `metric`), read the matching
    ``*_labels.csv``, and plot the ``prediction_error`` column scaled by that
    model's oracle threshold, so the threshold is 1. The oracle threshold is
    drawn once (dashed) and ground-truth anomaly regions are shaded once. The
    series are downsampled to every 10th time step before plotting. The
    figure is saved as a PNG at ``output_path``.
    """
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
        # Scale by the threshold of the method the metric names, so it maps to 1.
        thr_source = shared.with_blocks(best, blocks_key) if blocks_key else best
        threshold = _plot_threshold(thr_source, metric) if thr_source else None
        if not threshold:
            print(f"No usable {_plot_threshold_method(metric)} threshold for {model}; skipping in plot.")
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
        best_by_model[model] = best
        # Ground truth is shared across models for a dataset; capture it once.
        if ground_truth is None and "ground_truth" in df.columns:
            ground_truth = df["ground_truth"].reset_index(drop=True)

    if not series:
        print(f"No prediction_error data found for {dataset}; skipping plot.")
        return

    # Keep only the 5 best models (by `metric`) so the overlay stays readable;
    # models without a usable metric value rank last.
    if models is not None:
        series = {m: series[m] for m in models if m in series}
    elif len(series) > 5:
        def _metric_val(model):
            try:
                return float(_get(best_by_model[model], metric))
            except (TypeError, ValueError):
                return float("nan")

        def _sort_key(model):
            v = _metric_val(model)
            if math.isnan(v):
                return (1, 0.0)
            return (0, v if metric in LOWER_IS_BETTER else -v)

        keep = sorted(series, key=_sort_key)[:5]
        series = {m: series[m] for m in keep}

    # Models have different calibration lengths, so the outer join leaves the
    # union index unsorted; sort it or the lines wrap back to the start.
    combined = pd.concat(series, axis=1).sort_index()

    # Show most of the mass rather than the peaks: cap the y-axis at the
    # 99.9th percentile of all plotted values, with a little headroom, and
    # never below 2.
    y_top = _plot_y_top(combined, ground_truth)
    combined = downsample.every_nth_step(combined, 10)

    fig, ax = style.new_figure()
    style.plot_series(ax, combined)
    ax.set_ylim(0, y_top)
    _format_y_ticks(ax, y_top)
    # Every series is scaled by its own threshold, so one line at 1 is the
    # threshold for all models.
    method = _plot_threshold_method(metric)
    style.draw_threshold_line(ax, 1.0, f"{method} threshold")
    if ground_truth is not None:
        style.shade_anomalies(ax, ground_truth)
    if combined.index.min() < 0:
        style.mark_calibration_end(ax)
    ax.set_xlabel("time step")
    ax.set_ylabel(f"prediction error / {method} threshold")
    style.legend_below(ax)
    style.save_png(fig, output_path)


# ---------------------------------------------------------------------------
# Per-model plots (best result per model)
# ---------------------------------------------------------------------------


def _generate_model_plots(dataset, metric, by_model, output_dir, blocks_key=None):
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
        pot_thr, oracle_thr = _thresholds(thr_source)
        threshold = _plot_threshold(thr_source, metric)
        if not threshold:
            print(f"No usable {_plot_threshold_method(metric)} threshold for {model}; skipping plot.")
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

        y_top = _plot_y_top(series, df["ground_truth"] if "ground_truth" in df.columns else None)
        series = downsample.every_nth_step(series, 10)

        method = _plot_threshold_method(metric)
        fig, ax = style.new_figure()
        style.plot_series(ax, series.rename("prediction_error"))
        ax.set_ylim(0, y_top)
        _format_y_ticks(ax, y_top)
        if pot_thr:
            color = "tab:red" if method == "pot" else "grey"
            style.draw_threshold_line(ax, pot_thr / threshold, "POT threshold", color=color)
        if oracle_thr:
            color = "tab:red" if method == "oracle" else "grey"
            style.draw_threshold_line(ax, oracle_thr / threshold, "oracle threshold", color=color)
        if "ground_truth" in df.columns:
            style.shade_anomalies(ax, df["ground_truth"])
        if n_calib:
            style.mark_calibration_end(ax)
        ax.set_xlabel("time step")
        ax.set_ylabel(f"prediction error / {method} threshold")
        ax.set_title(f"{model} — {dataset}" + (" (shared threshold)" if blocks_key else ""))
        style.legend_below(ax)
        out_path = os.path.join(output_dir, f"{model}.png")
        style.save_png(fig, out_path)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _apply_shared_thresholds(by_dataset, results_folder):
    """Attach pot/oracle blocks fit with thresholds shared across datasets, in place.

    Only configurations (model + hyperparameters) with a run in every dataset
    are processed; per dataset the lowest-``calibration_loss`` run represents
    the configuration. Each processed result gets its shared blocks under
    ``result["shared"]`` and ``shared_threshold = True``; the local blocks stay
    untouched and remain what per-dataset tables show. Fits are cached under
    ``results_folder/_shared_thresholds/`` keyed by the CSV modification times.
    """

    datasets = list(by_dataset)
    groups = shared.group_configurations(
        by_dataset, _hp_key, lambda rs: _best_result(rs, "calibration_loss"))
    complete = {k: v for k, v in groups.items() if set(v) == set(datasets)}
    print(f"Fitting shared thresholds for {len(complete)} configurations across datasets {datasets}")

    cache_file = shared.cache_path(results_folder, datasets)
    cache = shared.load_cache(cache_file)
    shared_count, skipped_count = {}, {}
    for (model, _), members in groups.items():
        if set(members) != set(datasets):
            skipped_count[model] = skipped_count.get(model, 0) + 1

    for (model, hp_key), results_by_dataset in tqdm(complete.items(), desc="shared thresholds", unit="config"):
        q = next(iter(results_by_dataset.values())).get("applied_hyperparameters", {}).get("q", 1e-5)
        constants.initialize(datasets[0], model)
        blocks = shared.shared_blocks_cached(cache, model, hp_key, results_by_dataset, q, constants.level)
        if blocks is None:
            skipped_count[model] = skipped_count.get(model, 0) + 1
            continue
        shared.apply_blocks(results_by_dataset, blocks)
        shared_count[model] = shared_count.get(model, 0) + 1

    shared.save_cache(cache_file, cache)
    for model in sorted(set(shared_count) | set(skipped_count)):
        print(f"  {model}: {shared_count.get(model, 0)} configuration(s) shared, "
              f"{skipped_count.get(model, 0)} skipped (missing runs or CSVs)")


def generate_report(dataset, metric="calibration_loss", results_folder="results"):
    """Generate HTML, PDF, CSV, and hyperparameter-markdown reports.

    `dataset` may be a single name, a comma-separated string, or a list of
    names. With several datasets, one hyperparameter set per model is selected
    by the best *sum* of `metric` over all of them, and each dataset's report
    shows that shared configuration with its local (per-dataset) thresholds.
    POT and oracle thresholds are also fit jointly across the datasets (see
    `_apply_shared_thresholds`); those shared thresholds drive the
    reports/combined/ summary (confusion counts pooled over the datasets) and a
    second set of plots per dataset. Files are saved in reports/{dataset}/.
    """
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

    if len(by_dataset) > 1:
        _apply_shared_thresholds(by_dataset, results_folder)

    selected = _select_shared_best(by_dataset, metric)
    plot_models = None
    if len(by_dataset) > 1:
        pooled = _pooled_by_model(selected)
        plot_models = _top_models(pooled, metric)
    for ds in by_dataset:
        _generate_dataset_report(ds, metric, selected[ds], plot_models=plot_models, shared_plots=len(by_dataset) > 1)
    if len(by_dataset) > 1:
        _generate_dataset_report("combined", metric, pooled, plot_models=plot_models)


def _top_models(by_model, metric, n=5):
    """Names of the `n` best models by `metric` over `by_model`; models without a value rank last."""
    lower = metric in LOWER_IS_BETTER
    def key(model):
        v = _metric_value(_best_result(by_model[model], metric), metric)
        return (1, 0.0) if v is None else (0, v if lower else -v)
    return sorted(by_model, key=key)[:n]


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


def _generate_dataset_report(dataset, metric, by_model, plot_models=None, shared_plots=False):
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

    _generate_html(dataset, metric, by_model, html_path, unlabeled=unlabeled)
    _generate_pdf(dataset, metric, by_model, pdf_path, unlabeled=unlabeled)
    _generate_csv(dataset, metric, by_model, csv_path, unlabeled=unlabeled)
    _generate_hp_markdown(dataset, metric, by_model, hp_path)
    _generate_latex(dataset, metric, by_model, tex_path, unlabeled=unlabeled)
    _generate_prediction_error_plot(dataset, metric, by_model, plot_path, models=plot_models)
    _generate_model_plots(dataset, metric, by_model, plots_dir)
    if shared_plots:
        shared_plot_path = os.path.join(dataset_dir, "prediction_errors_shared.png")
        _generate_prediction_error_plot(dataset, metric, by_model, shared_plot_path, models=plot_models, blocks_key="shared")
        _generate_model_plots(dataset, metric, by_model, os.path.join(dataset_dir, "plots_shared"), blocks_key="shared")
