"""Generate HTML and PDF reports from hyperparameter sweep result JSON files."""
from glob import glob
import os
import csv
import json
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from jinja2 import Template

METHODS = ['pot', 'oracle']
METHOD_METRICS = ['f1', 'precision', 'recall', 'fpr', 'threshold', 'p_latency']
# Column display order: each metric paired with its point-adjusted (PA) counterpart.
DISPLAY_COLUMNS = []
for _m in METHOD_METRICS:
    DISPLAY_COLUMNS.append(_m)
    DISPLAY_COLUMNS.append(f'{_m} (PA)')

# Used by --metric flag; default points at pot.f1 (raw, non-PA).
DISPLAY_METRICS = [f'{method}.{m}' for method in METHODS for m in METHOD_METRICS]


def _get(d, dotted_key):
    """Look up a possibly-dotted key like 'pot.f1' in a nested dict, returning None if absent."""
    cur = d
    for part in dotted_key.split('.'):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _load_results(dataset, results_folder='results'):
    """Load all result JSONs for a dataset, grouped by model."""
    pattern = os.path.join(results_folder, dataset, '*_results.json')
    files = glob(pattern)
    by_model = {}
    for path in files:
        with open(path) as f:
            data = json.load(f)
        model = data.get('model', 'unknown')
        by_model.setdefault(model, []).append(data)
    return by_model


def _best_result(results, metric):
    """Return the result dict with the highest non-NaN metric value.

    `metric` may be a dotted path like 'pot.f1' to reach into the nested schema.
    """
    def _val(r):
        v = _get(r, metric)
        if v is None:
            return float('nan')
        try:
            return float(v)
        except (TypeError, ValueError):
            return float('nan')
    valid = [r for r in results if not math.isnan(_val(r))]
    if not valid:
        return None
    return max(valid, key=_val)


def _fmt(value):
    """Format a numeric value for display."""
    if value is None:
        return 'N/A'
    try:
        v = float(value)
        if math.isnan(v):
            return 'NaN'
        if v == int(v) and abs(v) < 1e15:
            return str(int(v))
        return f'{v:.4f}'
    except (TypeError, ValueError):
        return str(value)


def _build_summary(by_model, metric):
    """Build metric rows and hyperparameter rows for best result per model.

    Each model produces two metric rows — one for `pot`, one for `oracle` —
    where each numeric column appears twice: raw (point-wise) and PA
    (point-adjusted, i.e. segment-expanded). The row label is "<model> / <method>".
    Sort order is by the row whose metric path matches the --metric argument.
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
            for k in r.get('applied_hyperparameters', {}).keys():
                if k not in seen_hp:
                    all_hp_keys.append(k)
                    seen_hp.add(k)

    sort_method, _, sort_metric = metric.partition('.')

    grouped = []  # list of (sort_value, [method_rows], hp_row)
    for model, results in by_model.items():
        best = _best_result(results, metric)
        method_rows = []
        sort_val = -1.0
        for method in METHODS:
            row = {'row': f'{model} / {method}'}
            for m in METHOD_METRICS:
                row[m] = _fmt(_get(best, f'{method}.{m}')) if best else 'N/A'
                row[f'{m} (PA)'] = _fmt(_get(best, f'{method}_expanded.{m}')) if best else 'N/A'
            if method == sort_method:
                try:
                    sort_val = float(row[sort_metric]) if row[sort_metric] not in ('N/A', 'NaN') else -1.0
                except (TypeError, ValueError):
                    sort_val = -1.0
            method_rows.append(row)
        h_row = {'row': model}
        hp = best.get('applied_hyperparameters', {}) if best else {}
        for k in all_hp_keys:
            h_row[k] = _fmt(hp[k]) if k in hp else ''
        grouped.append((sort_val, method_rows, h_row))

    grouped.sort(key=lambda t: t[0], reverse=True)
    for _, method_rows, h_row in grouped:
        metric_rows.extend(method_rows)
        hp_rows.append(h_row)

    return metric_rows, hp_rows, all_hp_keys


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


def _generate_html(dataset, metric, by_model, output_path):
    metric_rows, hp_rows, hp_keys = _build_summary(by_model, metric)
    template = Template(_HTML_TEMPLATE)
    html = template.render(
        dataset=dataset,
        metric=metric,
        display_columns=DISPLAY_COLUMNS,
        metric_rows=metric_rows,
        hp_rows=hp_rows,
        hp_keys=hp_keys,
    )
    with open(output_path, 'w') as f:
        f.write(html)
    print(f'HTML report written to {output_path}')


# ---------------------------------------------------------------------------
# PDF generation
# ---------------------------------------------------------------------------

def _draw_table_page(pdf, title, columns, rows):
    """Draw a single table as a matplotlib figure and save to the PDF."""
    n_rows = len(rows)
    n_cols = len(columns)

    fig_height = max(2.5, 0.35 * (n_rows + 2))
    fig, ax = plt.subplots(figsize=(max(10, 1.5 * n_cols), fig_height))
    ax.axis('off')
    ax.set_title(title, fontsize=11, fontweight='bold', pad=8, loc='left')

    cell_data = [[str(row.get(c, '')) for c in columns] for row in rows]
    table = ax.table(
        cellText=cell_data,
        colLabels=columns,
        colWidths=[1.0 / n_cols] * n_cols,
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.3)

    for j in range(n_cols):
        cell = table[0, j]
        cell.set_facecolor('#3a5a8a')
        cell.set_text_props(color='white', fontweight='bold')

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def _generate_pdf(dataset, metric, by_model, output_path):
    metric_rows, hp_rows, hp_keys = _build_summary(by_model, metric)

    # Rename internal 'row' key to a display label
    for row in metric_rows:
        row['Model / method'] = row.pop('row')
    for row in hp_rows:
        row['Model'] = row.pop('row')

    with PdfPages(output_path) as pdf:
        _draw_table_page(
            pdf,
            title=f'Metrics — Dataset: {dataset}  (best by {metric})',
            columns=['Model / method'] + DISPLAY_COLUMNS,
            rows=metric_rows,
        )
        _draw_table_page(
            pdf,
            title=f'Hyperparameters — Dataset: {dataset}  (best by {metric})',
            columns=['Model'] + hp_keys,
            rows=hp_rows,
        )

    print(f'PDF report written to {output_path}')


# ---------------------------------------------------------------------------
# Slide-ready summary (CSV metrics + per-model hyperparameter markdown)
# ---------------------------------------------------------------------------

# Slim metric set for presentation tables. Each entry is (column_label, dotted_path).
SLIDE_COLUMNS = [
    ('POT F1',       'pot.f1'),
    ('POT F1 (PA)',  'pot_expanded.f1'),
    ('POT FPR',      'pot.fpr'),
    ('POT latency',  'pot.p_latency'),
    ('Oracle F1',    'oracle.f1'),
]


def _generate_csv(dataset, metric, by_model, output_path):
    """One row per model, slim metric set, sorted by `metric` desc."""
    rows = []
    for model, results in by_model.items():
        best = _best_result(results, metric)
        row = {'model': model}
        for label, path in SLIDE_COLUMNS:
            row[label] = _fmt(_get(best, path)) if best else 'N/A'
        try:
            sort_val = float(_get(best, metric)) if best else float('-inf')
        except (TypeError, ValueError):
            sort_val = float('-inf')
        if sort_val != sort_val:  # NaN
            sort_val = float('-inf')
        rows.append((sort_val, row))
    rows.sort(key=lambda t: t[0], reverse=True)

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['model'] + [c for c, _ in SLIDE_COLUMNS])
        writer.writeheader()
        for _, row in rows:
            writer.writerow(row)
    print(f'CSV summary written to {output_path}')


def _generate_hp_markdown(dataset, metric, by_model, output_path):
    """One line per model listing its best-result hyperparameters."""
    lines = [f'# Hyperparameters — {dataset}', f'_Best result per model by {metric}_', '']
    entries = []
    for model, results in by_model.items():
        best = _best_result(results, metric)
        hp = best.get('applied_hyperparameters', {}) if best else {}
        try:
            sort_val = float(_get(best, metric)) if best else float('-inf')
        except (TypeError, ValueError):
            sort_val = float('-inf')
        if sort_val != sort_val:
            sort_val = float('-inf')
        parts = ', '.join(f'{k}={_fmt(v)}' for k, v in sorted(hp.items()))
        entries.append((sort_val, f'- **{model}**: {parts if parts else "(defaults)"}'))
    entries.sort(key=lambda t: t[0], reverse=True)
    lines.extend(line for _, line in entries)

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'Hyperparameter summary written to {output_path}')


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_report(dataset, metric='f1', results_folder='results'):
    """Generate HTML, PDF, CSV, and hyperparameter-markdown reports for a dataset."""
    by_model = _load_results(dataset, results_folder)
    if not by_model:
        print(f'No results found for dataset "{dataset}" in {results_folder}/')
        return

    os.makedirs('reports', exist_ok=True)
    html_path = os.path.join('reports', f'{dataset}_report.html')
    pdf_path = os.path.join('reports', f'{dataset}_report.pdf')
    csv_path = os.path.join('reports', f'{dataset}_summary.csv')
    hp_path = os.path.join('reports', f'{dataset}_hyperparams.md')

    _generate_html(dataset, metric, by_model, html_path)
    _generate_pdf(dataset, metric, by_model, pdf_path)
    _generate_csv(dataset, metric, by_model, csv_path)
    _generate_hp_markdown(dataset, metric, by_model, hp_path)
