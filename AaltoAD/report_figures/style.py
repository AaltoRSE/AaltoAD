"""Shared matplotlib styling for the prediction-error report figures."""

import matplotlib.pyplot as plt

PALETTE = [f"tab:{name}" for name in (
    "blue", "orange", "green", "red", "purple",
    "brown", "pink", "gray", "olive", "cyan",
)]
FIGSIZE = (14, 5.4)
DPI = 300
LINEWIDTH = 1.5


def new_figure():
    """Create a figure/axes pair at the report size with top/right spines hidden."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig, ax


def plot_series(ax, data):
    """Plot each column of `data` (or the Series itself) as a solid line, cycling PALETTE."""
    if hasattr(data, "columns"):
        columns = [(name, data[name]) for name in data.columns]
    else:
        columns = [(data.name, data)]
    for i, (name, values) in enumerate(columns):
        ax.plot(
            values.index,
            values.values,
            color=PALETTE[i % len(PALETTE)],
            linestyle="-",
            linewidth=LINEWIDTH,
            label=name,
        )


def draw_threshold_line(ax, y, label, color="tab:red", linestyle="--"):
    """Draw a horizontal threshold line at `y` with the given label, color, and linestyle."""
    ax.axhline(y, color=color, linestyle=linestyle, linewidth=LINEWIDTH, label=label)


def shade_anomalies(ax, ground_truth):
    """Shade ground-truth anomaly regions (red, alpha 0.15) with dashed lines at their edges."""
    mask = ground_truth.astype(bool)
    ax.fill_between(
        ground_truth.index,
        0,
        1,
        where=mask,
        color="red",
        alpha=0.15,
        transform=ax.get_xaxis_transform(),
        label="Anomaly",
    )
    in_region = False
    for step, flagged in zip(ground_truth.index, mask):
        if flagged and not in_region:
            ax.axvline(step, color="red", linestyle="--", linewidth=1.0)
            in_region = True
        elif not flagged and in_region:
            ax.axvline(step, color="red", linestyle="--", linewidth=1.0)
            in_region = False
    if in_region:
        ax.axvline(ground_truth.index[-1], color="red", linestyle="--", linewidth=1.0)


def mark_calibration_end(ax):
    """Draw a dashed vertical line at x=0 marking the end of the calibration period."""
    ax.axvline(0, color="grey", linestyle="--", linewidth=1.0)


def legend_below(ax):
    """Place a single-row, frameless legend below the axes."""
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=len(handles),
        fontsize=11,
        frameon=False,
    )


def save_png(fig, path):
    """Save `fig` as a PNG at DPI with tight bounding box, close it, and print the path."""
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot written to {path}")
