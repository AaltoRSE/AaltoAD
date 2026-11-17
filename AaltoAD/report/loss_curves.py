"""Per-epoch loss curve plots for dataset reports.

Reads the ``training_loss_curve`` and ``calibration_loss_curve`` lists that
run_experiment stores in each result JSON and draws, per model, every run's
curves in one figure (training and calibration loss side by side).
"""

import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from AaltoAD.report.report_figures import style

# Result JSON keys holding one loss value per epoch, with their panel titles.
CURVE_KEYS = (
    ("training_loss_curve", "training loss"),
    ("calibration_loss_curve", "calibration loss"),
)

# Line style for the many sweep runs; the selected run is drawn on top in red.
RUN_ALPHA = 0.35
RUN_LINEWIDTH = 0.8


def _draw_panel(ax, curves, selected_curve, title):
    """Draw every curve in `curves` thinly and `selected_curve` highlighted.

    The y axis is logarithmic when all values are positive, since a sweep over
    learning rates spreads losses across orders of magnitude.
    """
    for curve in curves:
        ax.plot(
            range(1, len(curve) + 1),
            curve,
            color="tab:blue",
            alpha=RUN_ALPHA,
            linewidth=RUN_LINEWIDTH,
        )
    if selected_curve:
        ax.plot(
            range(1, len(selected_curve) + 1),
            selected_curve,
            color="tab:red",
            linewidth=style.LINEWIDTH,
            label="selected run",
        )
        ax.legend(frameon=False)
    values = [v for c in curves for v in c]
    if values and min(values) > 0:
        ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_title(f"{title} ({len(curves)} runs)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def generate_loss_curve_plots(dataset, by_model, output_dir, selected=None):
    """Write one PNG per model with the loss curves of all its runs.

    `by_model` is ``{model: [results]}`` with every run to draw (typically all
    of the dataset's runs, not just the selected ones). `selected` is the
    report's ``{model: [result]}`` selection; its run is highlighted. Runs
    without curves (failed, or not trained in that run) are skipped, and a
    model with no curves at all gets no file. Files are written to
    ``output_dir/<model>.png`` (typically ``reports/<dataset>/loss_curves/``).
    """
    selected = selected or {}
    for model, results in by_model.items():
        chosen = (selected.get(model) or [None])[0] or {}
        panels = [
            ([r[key] for r in results if r.get(key)], chosen.get(key), title)
            for key, title in CURVE_KEYS
        ]
        panels = [p for p in panels if p[0]]
        if not panels:
            continue
        os.makedirs(output_dir, exist_ok=True)
        fig, axes = plt.subplots(
            1, len(panels), figsize=style.FIGSIZE, squeeze=False
        )
        for ax, (curves, selected_curve, title) in zip(axes[0], panels):
            _draw_panel(ax, curves, selected_curve, title)
        fig.suptitle(f"{model} — {dataset}")
        style.save_png(fig, os.path.join(output_dir, f"{model}.png"))
