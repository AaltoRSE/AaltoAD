"""Downsampling of time-indexed prediction-error data for plotting.

A full series is 7200 points per model, too many to draw honestly at report
width, so it is reduced first. How it is reduced decides what the reader can
see:

- `every_nth_step` keeps one point in `step` and drops the rest, so a spike
  narrower than `step` is invisible unless it happens to land on a kept step.
- `mean_in_window` averages each window, which flattens a narrow spike towards
  the surrounding level — the same blind spot, softened.
- `max_in_window` keeps every peak, so a threshold crossing is always drawn,
  but discards the dips and makes a series look consistently higher than it is.
- `min_in_window` is its mirror: dips survive, peaks do not.
- `min_max_in_window` keeps both extremes as a band, at the cost of drawing a
  range rather than a line. Nothing is lost, so the window can be wider.

All of them take a pandas Series or DataFrame indexed by time step, with
calibration at negative steps, and label each window by its first step (floor
division puts -3595 in the window starting at -3600).
"""


def _windows(data, window):
    """Group `data` by window-start label."""
    return data.groupby((data.index // window) * window)


def every_nth_step(data, step=10):
    """Return the rows of `data` whose integer index is divisible by `step`."""
    return data[data.index % step == 0]


def min_in_window(data, window=10):
    """Return the per-window minimum of `data`."""
    return _windows(data, window).min()


def max_in_window(data, window=10):
    """Return the per-window maximum of `data`."""
    return _windows(data, window).max()


def mean_in_window(data, window=10):
    """Return the per-window mean of `data`."""
    return _windows(data, window).mean()


def min_max_in_window(data, window=20):
    """Return ``(low, high)``, the per-window minimum and maximum of `data`."""
    grouped = _windows(data, window)
    return grouped.min(), grouped.max()
