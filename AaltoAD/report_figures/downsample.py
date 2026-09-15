"""Downsampling of time-indexed prediction-error data for plotting."""


def every_nth_step(data, step=10):
    """Return the rows of `data` whose integer index is divisible by `step`.

    `data` is a pandas Series or DataFrame indexed by time step (calibration
    steps are negative; Python's `%` still gives 0 for multiples of `step`).
    """
    return data[data.index % step == 0]
