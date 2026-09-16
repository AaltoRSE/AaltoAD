import argparse

from AaltoAD import constants

parser = argparse.ArgumentParser(
	description='Time-Series Anomaly Detection - Unified Runner',
	formatter_class=argparse.RawDescriptionHelpFormatter,
	epilog="""
Examples:
  # Single run with command-line arguments
  python main.py --model TranAD --dataset TOL
  
  # Run all models or datasets
  python main.py --model ALL --dataset TOL
  
  # Run experiment from file by index
  python main.py --experiment 5
  python main.py --experiment 5 --experiment-file experiments_TOL.json
  
  # List experiments and their sub-experiments
  python main.py --experiment-file experiments_TOL.json --list
  
  # Check status of experiments
  python main.py --experiment-file experiments_TOL.json --status
  
  # Array job mode (for SLURM/PBS array jobs)
  python main.py --array-index $SLURM_ARRAY_TASK_ID
  sbatch --array=0-99 run_array.sh  # Runs 100 incomplete sub-experiments
	"""
)

# Single run arguments
parser.add_argument('-d', '--dataset',
					type=str,
					required=False,
					default='synthetic',
                    help="Dataset name (or 'ALL' for all datasets; with --report, a comma-separated list selects shared hyperparameters by summed metric and refits shared POT/oracle thresholds jointly across the datasets)")
parser.add_argument('-m', '--model',
					type=str,
					required=False,
					default='LSTM_Univariate',
                    help="Model name (or 'ALL' for all models)")
parser.add_argument('--test', 
					action='store_true', 
					help="test the model")
parser.add_argument('--retrain', 
					action='store_true', 
					help="retrain the model")
parser.add_argument('--less', 
					action='store_true', 
					help="train using less data")
parser.add_argument('--experiment-index',
					type=int,
					default=None,
					help="experiment index for tracking/caching")
parser.add_argument('--hyperparams',
					type=str,
					default=None,
					help="JSON string or file with hyperparameters to apply")

# Experiment file arguments
parser.add_argument('--experiment', 
					type=int,
					default=None,
					help="Run specific experiment by index from experiment file")
parser.add_argument('--experiment-file', 
					type=str, 
					default=None,
					help="Path to experiment configuration JSON file (default: experiments.json)")
parser.add_argument('--array-index',
					type=int,
					default=None,
					help="Array job index for SLURM/PBS array jobs (maps to incomplete sub-experiments)")
parser.add_argument('--list', 
					action='store_true',
					help="List experiments in experiment file")
parser.add_argument('--status', 
					action='store_true',
					help="Show status of experiments in file")
parser.add_argument('--worker',
					action='store_true',
					help="Run as a worker that loops claiming and running tasks until none remain")

parser.add_argument('--plot',
                    action="store_true",
                    help="Generate plots after training")
parser.add_argument('--report',
                    action="store_true",
                    help="Generate tables of top results for each model and dataset")
parser.add_argument('--metric',
                    type=str,
                    default=','.join(constants.METRIC),
                    help="Comma-separated metrics inside the --threshold block, most significant first "
                         f"(default: {','.join(constants.METRIC)}). Selects each model's hyperparameters "
                         "and orders the models in every table and plot. Each is ranked in its natural "
                         "direction, so latency and fpr sort ascending and f1 descending; 'latency' is "
                         "accepted for p_latency.")
parser.add_argument('--plot-models',
                    type=int,
                    default=constants.PLOT_MODELS,
                    help="How many models the report's prediction-error overlay plots show, best first "
                         f"(default: {constants.PLOT_MODELS})")
parser.add_argument('--threshold', '--threshold-method',
                    type=str,
                    dest='threshold',
                    default=constants.THRESHOLD_METHOD,
                    help="Threshold method the report's metrics are read from: conformal, pot or oracle "
                         f"(default: {constants.THRESHOLD_METHOD}). Tables quote it and plots are scaled "
                         "by it.")
parser.add_argument('--conformal-q',
                    type=float,
                    default=constants.CONFORMAL_Q,
                    help="False alarm rate the conformal threshold targets: at most this fraction of the "
                         f"calibration scores sit above it (default: {constants.CONFORMAL_Q:g}). POT keeps "
                         "using each run's own swept q.")
parser.add_argument('--model-order',
                    type=str,
                    default=None,
                    help="Comma-separated metrics that order the models, if they should be ordered by "
                         "something other than --metric. Same spelling rules as --metric.")

def parse_arguments():
	"""Parse command-line arguments and return the args object."""
	global args
	args = parser.parse_args()
	return args

# Initialize with default values (will be overwritten when parse_arguments() is called)
args = parser.parse_args([])