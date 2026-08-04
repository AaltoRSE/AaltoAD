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
