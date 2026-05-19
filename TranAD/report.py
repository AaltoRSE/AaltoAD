"""Generate HTML and PDF reports from hyperparameter sweep result JSON files."""
from glob import glob
import os
import json
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from jinja2 import Template

DISPLAY_METRICS = [
    'pot.f1', 'pot.precision', 'pot.recall', 'pot.threshold',
    'oracle.f1', 'oracle.precision', 'oracle.recall', 'oracle.threshold',
]


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
    """Build metric rows and hyperparameter rows for best result per model."""
    metric_rows = []
    hp_rows = []
    all_hp_keys = []
    seen_hp = set()

    for model, results in by_model.items():
        best = _best_result(results, metric)
        if best:
            for k in best.get('applied_hyperparameters', {}).keys():
                if k not in seen_hp:
                    all_hp_keys.append(k)
                    seen_hp.add(k)

    for model, results in by_model.items():
        best = _best_result(results, metric)
        m_row = {'model': model}
        h_row = {'model': model}
        for m in DISPLAY_METRICS:
            m_row[m] = _fmt(_get(best, m)) if best else 'N/A'
        hp = best.get('applied_hyperparameters', {}) if best else {}
        for k in all_hp_keys:
            h_row[k] = _fmt(hp[k]) if k in hp else ''
        metric_rows.append(m_row)
        hp_rows.append(h_row)

    # Sort by the metric we optimized for (which is one of DISPLAY_METRICS).
    sort_key = lambda r: float(r[metric]) if metric in r and r[metric] not in ('N/A', 'NaN') else -1
    order = sorted(range(len(metric_rows)), key=lambda i: sort_key(metric_rows[i]), reverse=True)
    metric_rows = [metric_rows[i] for i in order]
    hp_rows = [hp_rows[i] for i in order]

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
      <th>Model</th>
      {% for m in display_metrics %}<th>{{ m }}</th>{% endfor %}
    </tr>
  </thead>
  <tbody>
    {% for row in metric_rows %}
    <tr>
      <td>{{ row.model }}</td>
      {% for m in display_metrics %}<td>{{ row[m] }}</td>{% endfor %}
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
      <td>{{ row.model }}</td>
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
        display_metrics=DISPLAY_METRICS,
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

    # Rename 'model' to 'Model' for display
    for row in metric_rows + hp_rows:
        row['Model'] = row.pop('model')

    with PdfPages(output_path) as pdf:
        _draw_table_page(
            pdf,
            title=f'Metrics — Dataset: {dataset}  (best by {metric})',
            columns=['Model'] + DISPLAY_METRICS,
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
# Public entry point
# ---------------------------------------------------------------------------

def generate_report(dataset, metric='f1', results_folder='results'):
    """Generate HTML and PDF reports for a dataset."""
    by_model = _load_results(dataset, results_folder)
    if not by_model:
        print(f'No results found for dataset "{dataset}" in {results_folder}/')
        return

    os.makedirs('reports', exist_ok=True)
    html_path = os.path.join('reports', f'{dataset}_report.html')
    pdf_path = os.path.join('reports', f'{dataset}_report.pdf')

    _generate_html(dataset, metric, by_model, html_path)
    _generate_pdf(dataset, metric, by_model, pdf_path)
