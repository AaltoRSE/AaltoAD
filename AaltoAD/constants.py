# Data folders
DEFAULT_OUTPUT_FOLDER = "processed"
DEFAULT_DATA_FOLDER = "data"
output_folder = DEFAULT_OUTPUT_FOLDER
data_folder = DEFAULT_DATA_FOLDER

# These dictionaries define dataset-specific configuration values
POT_INIT = {
    "SMD": 0.99995,
    "synthetic": 0.999,
    "SWaT": 0.993,
    "UCR": 0.993,
    "NAB": 0.991,
    "SMAP": 0.98,
    "MSL": 0.97,
    "WADI": 0.99,
    "MSDS": 0.91,
    "MBA": 0.87,
    "default": 0.99,
}



MERLIN_PERCENTILES = {
    "SMD": (98, 2000),
    "synthetic": (95, 10),
    "SWaT": (95, 10),
    "SMAP": (97, 5000),
    "MSL": (97, 150),
    "WADI": (99, 1200),
    "MSDS": (96, 30),
    "UCR": (98, 2),
    "NAB": (98, 2),
    "MBA": (99, 2),
}

# How many models the report's prediction-error overlay plots show (best first).
PLOT_MODELS = 3

# Threshold method the report quotes in tables and scales plots by
# (--threshold-method).
THRESHOLD_METHOD = "conformal"

# How plotted series are reduced from 7200 points to something drawable
# (--downsample): "min", "max" or "mean" of each window, "range" for a min-max
# band, or "nth" to keep every Nth step. A range shows both extremes, so it
# affords a wider window.
DOWNSAMPLE = "range"
DOWNSAMPLE_WINDOWS = {"nth": 10, "min": 10, "max": 10, "mean": 10, "range": 20}

# Columns of the per-case LaTeX summary table (--table-columns). Names resolve
# inside the threshold method's block, except the top-level ones.
TABLE_COLUMNS = ("f1", "fpr", "latency")

# Models per row in the per-case LaTeX summary table (--table-blocks): with few
# columns the table is narrow enough to set two side by side.
TABLE_BLOCKS = 2

# Whether one threshold is fit on the calibration data of every dataset in a
# report pooled together (--pool-baselines), or each dataset is thresholded on
# its own baseline. Pooling assumes the baselines are alike; when one of them
# is contaminated it sets the threshold for all the others.
POOL_BASELINES = False

# Whether one threshold is fit on the calibration data of every dataset in a
# report pooled together (--pool-baselines), or each dataset is thresholded on
# its own baseline. Pooling assumes the baselines are alike; when one of them
# is contaminated it sets the threshold for all the others.
POOL_BASELINES = False

# Target false alarm rate for the conformal threshold (--conformal-q): at most
# this fraction of the calibration scores sit above it. A reporting choice, not
# a swept hyperparameter, so it is set here rather than read from a run.
CONFORMAL_Q = 1e-3

# The report's metric (--metric): metric names inside the threshold method's
# block, most significant first. It selects each model's hyperparameters and
# orders the models in every table and plot, unless --model-order gives the
# ordering its own metric. Each name is ranked in its natural direction, so
# this reads "fastest detection first, then fewest false positives, then best
# F1 to break a tie".
METRIC = ("p_latency", "fpr", "f1")

# These will be initialized by main.py after parsing arguments
level = None
percentile_merlin = None
cvp = None
preds = []
debug = 9


def initialize(dataset: str, model: str):
    """Initialize configuration based on dataset and model.

    This should be called from main.py after parsing command-line arguments.

    Args:
        dataset (str): Dataset name (e.g., 'SMD', 'SMAP', 'synthetic')
        model (str): Model name (e.g., 'TranAD', 'LSTM_Univariate')
    """
    global level, percentile_merlin, cvp

    # Resolve the POT_INIT entry: exact key match, else the longest key
    # that is a prefix of the dataset name, else "default".
    if dataset in POT_INIT:
        tail_key = dataset
    else:
        prefix_matches = [key for key in POT_INIT if key != "default" and dataset.startswith(key)]
        tail_key = max(prefix_matches, key=len) if prefix_matches else "default"
    level = POT_INIT[tail_key]

    # Set percentile values for debugging based on dataset
    percentile_pair = MERLIN_PERCENTILES.get(dataset, MERLIN_PERCENTILES["synthetic"])
    percentile_merlin = percentile_pair[0]
    cvp = percentile_pair[1]


def reset():
    """Reset configuration to uninitialized state."""
    global level, percentile_merlin, cvp, preds
    level = None
    percentile_merlin = None
    cvp = None
    preds = []
