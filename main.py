import os
import sys
import json
from pathlib import Path
from typing import Dict, Optional

import TranAD
from TranAD.parser import parser
from TranAD import constants
from TranAD.run_experiment import run_experiment, run_all

# Suppress matplotlib font warnings
import warnings
warnings.filterwarnings('ignore', message='.*findfont.*')


def load_experiments(config_path: str) -> Dict:
	"""Load experiment config from JSON file."""
	with open(config_path, 'r') as f:
		return json.load(f)


def get_result_path(dataset: str, model: str, index: int) -> str:
	"""Get result file path for experiment."""
	results_dir = Path('results') / dataset
	results_dir.mkdir(parents=True, exist_ok=True)
	return str(results_dir / f"{model}_exp{index}_results.json")


def has_matching_result(dataset: str, model: str, index: int) -> Optional[str]:
	"""Check if experiment result exists with matching index."""
	result_path = get_result_path(dataset, model, index)
	if os.path.exists(result_path):
		try:
			with open(result_path, 'r') as f:
				data = json.load(f)
				stored_index = data.get('experiment_index', -1)
				if stored_index == index:
					return result_path
		except:
			pass
	return None


def run_single_experiment(dataset: str, model: str, index: int, hyperparams: Dict) -> bool:
	"""Run one experiment from experiment file.
	
	Returns:
		True if run successfully, False if skipped or failed
	"""
	# Check if already run with same index
	existing_result = has_matching_result(dataset, model, index)
	if existing_result:
		print(f"✓ SKIP   Exp {index}: {model} (index {index} already cached)")
		return False
	
	print(f"→ RUN    Exp {index}: {model}", end="")
	# Print hyperparams
	hp_str = ", ".join([f"{k}={v}" for k, v in hyperparams.items() if k not in ['notes']])
	if hp_str:
		print(f" ({hp_str})", end="")
	print()
	
	try:
		# Initialize config for this dataset/model combination
		constants.initialize(dataset, model)
		
		result = run_experiment(
			model_name=model,
			dataset_name=dataset,
			experiment_index=index,
			hyperparams_str=json.dumps(hyperparams),
			less=False,
			test=False,
			retrain=True
		)
		
		result_path = get_result_path(dataset, model, index)
		if result and os.path.exists(result_path):
			print(f"✓ DONE   Exp {index}: {model}")
			return True
		else:
			print(f"✗ ERROR  Exp {index}: Result file not created")
			return False
	except Exception as e:
		print(f"✗ ERROR  Exp {index}: {str(e)[:200]}")
		return False


def handle_single_run(args):
	"""Handle single experiment run with command-line arguments."""
	# If either model or dataset is 'ALL', run the full sweep
	if args.model == 'ALL' or args.dataset == 'ALL':
		models_list = None if args.model == 'ALL' else [args.model]
		datasets_list = None if args.dataset == 'ALL' else [args.dataset]
		run_all(models_list=models_list, datasets_list=datasets_list)
	else:
		constants.initialize(args.dataset, args.model)
		run_experiment(
			model_name=args.model,
			dataset_name=args.dataset,
			hyperparams_str=args.hyperparams,
			experiment_index=None,
			less=args.less,
			test=args.test,
			retrain=args.retrain
		)


def handle_experiment_file(args):
	"""Handle experiment file operations."""
	if not os.path.exists(args.experiment_file):
		print(f"Error: Experiment file '{args.experiment_file}' not found")
		return 1
	
	config = load_experiments(args.experiment_file)
	experiments = config.get('experiments', [])
	
	if not experiments:
		print("Error: No experiments defined in file")
		return 1
	
	# Filter by index if specified
	if args.experiment:
		experiments = [e for e in experiments if e.get('index') == args.experiment]
		if not experiments:
			print(f"Error: No experiment with index {args.experiment}")
			return 1
	
	# List mode
	if args.list:
		print(f"\nExperiments in {args.experiment_file}:")
		print(f"{'Idx':<4} {'Dataset':<12} {'Model':<15} {'LR':<10} {'Notes'}")
		print("-" * 70)
		for exp in experiments:
			idx = exp.get('index')
			dataset = exp.get('dataset', '?')
			model = exp.get('model')
			lr = exp.get('lr', 'default')
			notes = exp.get('notes', '')
			print(f"{idx:<4} {dataset:<12} {model:<15} {lr:<10} {notes}")
		return 0
	
	# Status mode
	if args.status:
		print(f"\nExperiment Status in {args.experiment_file}:")
		print(f"{'Idx':<4} {'Dataset':<12} {'Model':<15} {'Status':<12}")
		print("-" * 55)
		for exp in experiments:
			idx = exp.get('index')
			dataset = exp.get('dataset', '?')
			model = exp.get('model')
			result_path = get_result_path(dataset, model, idx)
			if os.path.exists(result_path):
				try:
					with open(result_path, 'r') as f:
						data = json.load(f)
						stored_idx = data.get('experiment_index', -1)
						status = "✓ CACHED" if stored_idx == idx else f"✗ STALE"
				except:
					status = "✗ ERROR"
			else:
				status = "○ TODO"
			print(f"{idx:<4} {dataset:<12} {model:<15} {status:<12}")
		return 0
	
	# Run mode (experiment is specified or run all in file)
	print(f"\nRunning experiments...")
	print(f"Config: {args.experiment_file}")
	print(f"Total: {len(experiments)} experiment(s)\n")
	
	completed = 0
	skipped = 0
	failed = 0
	
	for exp in experiments:
		idx = exp.get('index')
		dataset = exp.get('dataset')
		model = exp.get('model')
		
		if not dataset or not model:
			print(f"✗ SKIP   Exp {idx}: Missing dataset or model")
			failed += 1
			continue
		
		# Extract hyperparams (everything except index, model, notes, dataset)
		hyperparams = {k: v for k, v in exp.items() 
					  if k not in ['index', 'model', 'notes', 'dataset']}
		
		# Run experiment
		success = run_single_experiment(dataset, model, idx, hyperparams)
		if success:
			completed += 1
		else:
			if has_matching_result(dataset, model, idx):
				skipped += 1
			else:
				failed += 1
	
	print(f"\n{'='*50}")
	print(f"Summary:")
	print(f"  Completed: {completed}")
	print(f"  Skipped:   {skipped}")
	print(f"  Failed:    {failed}")
	print(f"  Total:     {len(experiments)}")
	print(f"\nResults organized by dataset in results/{{dataset}}/")
	
	return 0 if failed == 0 else 1


def main():
	"""Main entry point."""
	args = parser.parse_args()
	
	# Determine which mode to run
	if args.experiment is not None or args.list or args.status:
		# Experiment file mode
		return handle_experiment_file(args)
	else:
		# Single run mode
		try:
			constants.initialize(args.dataset, args.model)
			handle_single_run(args)
			return 0
		except Exception as e:
			print(f"Error: {e}")
			return 1


if __name__ == '__main__':
	sys.exit(main())
