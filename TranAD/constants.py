# Data folders
DEFAULT_OUTPUT_FOLDER = 'processed'
DEFAULT_DATA_FOLDER = 'data'
output_folder = DEFAULT_OUTPUT_FOLDER
data_folder = DEFAULT_DATA_FOLDER

# These dictionaries define dataset-specific configuration values
POT_INIT = {
    'SMD': [(0.99995, 1.04), (0.99995, 1.06)],
    'synthetic': [(0.999, 1), (0.999, 1)],
    'SWaT': [(0.993, 1), (0.993, 1)],
    'UCR': [(0.993, 1), (0.99935, 1)],
    'NAB': [(0.991, 1), (0.99, 1)],
    'SMAP': [(0.98, 1), (0.98, 1)],
    'MSL': [(0.97, 1), (0.999, 1.04)],
    'WADI': [(0.99, 1), (0.999, 1)],
    'MSDS': [(0.91, 1), (0.9, 1.04)],
    'MBA': [(0.87, 1), (0.93, 1.04)],
    'default': [(0.993, 1), (0.993, 1)],
}

_LR_D = {
    'SMD': 0.0001,
    'synthetic': 0.0001,
    'SWaT': 0.008,
    'SMAP': 0.001,
    'MSL': 0.002,
    'WADI': 0.0001,
    'MSDS': 0.001,
    'UCR': 0.006,
    'NAB': 0.009,
    'MBA': 0.001,
}

_PERCENTILES = {
    'SMD': (98, 2000),
    'synthetic': (95, 10),
    'SWaT': (95, 10),
    'SMAP': (97, 5000),
    'MSL': (97, 150),
    'WADI': (99, 1200),
    'MSDS': (96, 30),
    'UCR': (98, 2),
    'NAB': (98, 2),
    'MBA': (99, 2),
}

# These will be initialized by main.py after parsing arguments
lm = None
lr = None
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
    global lm, lr, percentile_merlin, cvp
    
    # Set learning rate threshold based on dataset
    lm_pair = POT_INIT.get(dataset, POT_INIT['default'])
    # Use second value for TranAD models, first for others
    lm = lm_pair[1 if 'TranAD' in model else 0]
    
    # Set learning rate based on dataset
    lr = _LR_D.get(dataset, 0.0001)
    
    # Set percentile values for debugging based on dataset
    percentile_pair = _PERCENTILES.get(dataset, _PERCENTILES['synthetic'])
    percentile_merlin = percentile_pair[0]
    cvp = percentile_pair[1]


def reset():
    """Reset configuration to uninitialized state."""
    global lm, lr, percentile_merlin, cvp, preds
    lm = None
    lr = None
    percentile_merlin = None
    cvp = None
    preds = []