import argparse

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
parser.add_argument('--dataset', 
					metavar='-d', 
					type=str, 
					required=False,
					default='synthetic',
                    help="Dataset name (or 'ALL' for all datasets)")
parser.add_argument('--model', 
					metavar='-m', 
					type=str, 
					required=False,
					default='LSTM_Multivariate',
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

def parse_arguments():
	"""Parse command-line arguments and return the args object."""
	global args
	args = parser.parse_args()
	return args

# Initialize with default values (will be overwritten when parse_arguments() is called)
args = parser.parse_args([])