import os
import sys
import json
import traceback
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from itertools import product
import re
import time

from AaltoAD.parser import parser
from AaltoAD import constants
from AaltoAD.run_experiment import run_experiment, run_all
from AaltoAD.report import generate_report

# Suppress matplotlib font warnings
import warnings
warnings.filterwarnings('ignore', message='.*findfont.*')


def load_experiments(config_path: str) -> Dict:
	"""Load experiment config from JSON file."""
	with open(config_path, 'r') as f:
		return json.load(f)


def generate_parameter_grid(params: Dict) -> List[Dict]:
	"""Generate cross-product of parameter lists.

	Args:
		params: Dictionary where values can be lists or scalars

	Returns:
		List of dictionaries, each representing one parameter combination

	Example:
		>>> generate_parameter_grid({'lr': [0.1, 0.01], 'batch': [32, 64]})
		[
			{'lr': 0.1, 'batch': 32},
			{'lr': 0.1, 'batch': 64},
			{'lr': 0.01, 'batch': 32},
			{'lr': 0.01, 'batch': 64}
		]
	"""
	# Separate list and non-list parameters
	list_params = {}
	fixed_params = {}

	for key, value in params.items():
		if isinstance(value, list):
			list_params[key] = value
		else:
			fixed_params[key] = value

	# If no list parameters, return single config
	if not list_params:
		return [params.copy()]

	# Generate cross product
	keys = list(list_params.keys())
	values = [list_params[k] for k in keys]

	result = []
	for combination in product(*values):
		config = fixed_params.copy()
		config.update(dict(zip(keys, combination)))
		result.append(config)

	return result


def expand_experiment_to_subexperiments(exp: Dict) -> List[Tuple[str, Dict]]:
	"""Expand experiment with parameter lists into sub-experiments.

	Args:
		exp: Experiment dictionary from experiments.json

	Returns:
		List of (sub_id, hyperparams) tuples where:
			- sub_id is "index.subindex" (e.g., "5.0", "5.1", "5.2")
			- hyperparams is the parameter combination for that sub-experiment
	"""
	exp_index = exp.get('index')
	dataset = exp.get('dataset')
	model = exp.get('model')

	# Extract hyperparameters (exclude metadata fields)
	hyperparams = {k: v for k, v in exp.items()
				  if k not in ['index', 'model', 'notes', 'dataset']}

	# Generate all parameter combinations
	param_grids = generate_parameter_grid(hyperparams)

	# Create sub-experiment IDs and combine with metadata
	sub_experiments = []
	for sub_idx, params in enumerate(param_grids):
		sub_id = f"{exp_index}.{sub_idx}"
		sub_exp_params = params.copy()
		sub_experiments.append((sub_id, sub_exp_params))

	return sub_experiments


def collect_all_subexperiments(experiments: List[Dict]) -> List[Tuple[str, str, str, Dict]]:
	"""Collect all sub-experiments from experiment list.

	Args:
		experiments: List of experiment configurations from JSON file

	Returns:
		List of (exp_id, dataset, model, hyperparams) tuples for all sub-experiments
	"""
	all_subexps = []

	for exp in experiments:
		datasets = exp.get('dataset')
		model = exp.get('model')
		if not datasets or not model:
			continue
		# Allow `dataset` to be either a string or a list. A list means: run this
		# experiment (and all its hyperparam combinations) against each dataset
		# independently. Each (dataset, model, exp_id) gets its own result file.
		if isinstance(datasets, str):
			datasets = [datasets]

		sub_experiments = expand_experiment_to_subexperiments(exp)
		for ds in datasets:
			for exp_id, hyperparams in sub_experiments:
				all_subexps.append((exp_id, ds, model, hyperparams))

	return all_subexps


def get_incomplete_subexperiments(all_subexps: List[Tuple[str, str, str, Dict]]) -> List[Tuple[str, str, str, Dict]]:
	"""Filter to only incomplete sub-experiments.

	Args:
		all_subexps: List of (exp_id, dataset, model, hyperparams) tuples

	Returns:
		List of incomplete sub-experiments (those without matching results)
	"""
	incomplete = []

	for exp_id, dataset, model, hyperparams in all_subexps:
		if not has_matching_result(dataset, model, exp_id):
			incomplete.append((exp_id, dataset, model, hyperparams))

	return incomplete


def map_array_index_to_subexperiment(
	experiments: List[Dict],
	array_index: int,
	retrain: bool = False,
) -> Optional[Tuple[str, str, str, Dict]]:
	"""Map array job index to a sub-experiment.

	By default the mapping targets only incomplete sub-experiments so that
	array jobs are assigned remaining work. If `retrain=True` the mapping
	covers the full expanded list (useful when you want to re-run everything).

	Args:
		experiments: List of experiment configurations from JSON file
		array_index: Array job index (0-based)
		retrain: If True, map into the full set instead of only incomplete ones

	Returns:
		Tuple of (exp_id, dataset, model, hyperparams) or None if index is out of range
	"""
	# Collect all sub-experiments
	all_subexps = collect_all_subexperiments(experiments)

	if not all_subexps:
		return None

	# When retrain is requested, map directly into the full expanded list
	if retrain:
		if 0 <= array_index < len(all_subexps):
			print(f"Array job mapping (retrain=True): using all {len(all_subexps)} sub-experiments; assigned index {array_index}")
			return all_subexps[array_index]
		else:
			print(f"Array index {array_index} out of range for {len(all_subexps)} total sub-experiments (retrain)")
			return None

	# For normal runs, map into the list of incomplete sub-experiments
	incomplete = get_incomplete_subexperiments(all_subexps)
	if not incomplete:
		return None

	if 0 <= array_index < len(incomplete):
		print(f"Array job mapping: {len(incomplete)} incomplete sub-experiments; assigned index {array_index}")
		return incomplete[array_index]
	else:
		print(f"Array index {array_index} out of range for {len(incomplete)} incomplete sub-experiments")
		return None


def get_result_path(dataset: str, model: str, exp_id: str) -> str:
	"""Get result file path for experiment or sub-experiment.

	Args:
		dataset: Dataset name
		model: Model name
		exp_id: Experiment ID (e.g., "5" or "5.2" for sub-experiments)
	"""
	results_dir = Path('results') / dataset
	results_dir.mkdir(parents=True, exist_ok=True)
	return str(results_dir / f"{model}_exp{exp_id}_results.json")


def has_matching_result(dataset: str, model: str, exp_id: str) -> Optional[str]:
	"""Check if experiment result exists with matching ID.

	Args:
		dataset: Dataset name
		model: Model name
		exp_id: Experiment ID (e.g., "5" or "5.2" for sub-experiments)

	Returns:
		Path to result file if it exists and matches, None otherwise
	"""
	result_path = get_result_path(dataset, model, exp_id)
	if os.path.exists(result_path):
		try:
			with open(result_path, 'r') as f:
				data = json.load(f)
				stored_id = data.get('experiment_id')
				# If experiment_id present in JSON, compare
				if stored_id is not None and str(stored_id) == str(exp_id):
					return result_path
				# Fallback: try to extract id from filename if JSON lacks it
				fname = Path(result_path).name
				m = re.search(r'_exp(?P<id>[^_]+)_results\.json$', fname)
				if m and m.group('id') == str(exp_id):
					return result_path
		except Exception as e:
			# If JSON is corrupted/truncated, remove it so it can be overwritten by new runs
			try:
				os.remove(result_path)
			except Exception:
				pass
			print(f"Warning: removed corrupted result file {result_path} ({e})")
	return None


def get_claim_path(dataset: str, model: str, exp_id: str) -> str:
	"""Get claim file path for a sub-experiment."""
	results_dir = Path('results') / dataset
	results_dir.mkdir(parents=True, exist_ok=True)
	return str(results_dir / f".claim_{model}_exp{exp_id}")


def try_claim_task(dataset: str, model: str, exp_id: str) -> bool:
	"""Atomically claim a task using exclusive file creation.

	Returns True if successfully claimed, False if already claimed.
	"""
	claim_path = get_claim_path(dataset, model, exp_id)
	try:
		fd = os.open(claim_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
		worker_id = os.environ.get('SLURM_ARRAY_TASK_ID', str(os.getpid()))
		os.write(fd, f"{worker_id}\n{time.time()}\n".encode())
		os.close(fd)
		return True
	except FileExistsError:
		return False


def release_claim(dataset: str, model: str, exp_id: str):
	"""Remove claim file for a task."""
	claim_path = get_claim_path(dataset, model, exp_id)
	try:
		os.remove(claim_path)
	except FileNotFoundError:
		pass


def is_stale_claim(dataset: str, model: str, exp_id: str, max_age_hours: float = 13.0) -> bool:
	"""Check if a claim file is stale (older than max_age_hours) and remove it if so."""
	claim_path = get_claim_path(dataset, model, exp_id)
	try:
		mtime = os.path.getmtime(claim_path)
		if time.time() - mtime > max_age_hours * 3600:
			os.remove(claim_path)
			print(f"Removed stale claim: {claim_path}")
			return True
	except FileNotFoundError:
		pass
	return False


def run_single_experiment(dataset: str, model: str, exp_id: str, hyperparams: Dict, test: bool = False) -> bool:
	"""Run one experiment or sub-experiment from experiment file.

	Args:
		dataset: Dataset name
		model: Model name
		exp_id: Experiment ID (e.g., "5" or "5.2")
		hyperparams: Dictionary of hyperparameters
		test: When True, evaluate only — load the existing checkpoint, skip
			training, and rewrite results/labels/calib outputs. Experiments
			without a checkpoint are skipped instead of evaluated untrained.

	Returns:
		True if run successfully, False if skipped or failed
	"""
	# Check if already run with same ID
	existing_result = has_matching_result(dataset, model, exp_id)
	if existing_result:
		print(f"✓ SKIP   Exp {exp_id}: {model} (already cached)")
		return False

	hp_str = ", ".join([f"{k}={v}" for k, v in hyperparams.items() if k not in ['notes']])
	# flush=True so this line lands in the SLURM log before any
	# long-running / memory-hungry load, even when stdout is block-buffered.
	print(f"→ RUN    Exp {exp_id}: {model} on {dataset} ({hp_str})", flush=True)

	try:
		# Initialize config for this dataset/model combination
		constants.initialize(dataset, model)

		if test:
			# Eval-only refresh: require an existing checkpoint.
			from AaltoAD.run_experiment import _hyperparams_hash, _data_hash
			ckpt = os.path.join(
				'checkpoints', f'{model}_{dataset}',
				f'{_hyperparams_hash(hyperparams)}_{_data_hash(dataset)}', 'model.ckpt')
			if not os.path.exists(ckpt):
				print(f"✗ SKIP   Exp {exp_id}: {model} (no checkpoint for eval-only refresh)")
				return False

		result = run_experiment(
			model_name=model,
			dataset_name=dataset,
			experiment_id=exp_id,
			hyperparams_str=json.dumps(hyperparams),
			less=False,
			test=test,
			retrain=not test
		)

		result_path = get_result_path(dataset, model, exp_id)
		if result and os.path.exists(result_path):
			print(f"✓ DONE   Exp {exp_id}: {model}")
			return True
		else:
			print(f"✗ ERROR  Exp {exp_id}: Result file not created")
			_write_failure_marker(dataset, model, exp_id, hyperparams,
			                     "run_experiment returned without writing a result file")
			return False
	except Exception as e:
		print(f"✗ ERROR  Exp {exp_id}: {e}")
		traceback.print_exc()
		tb_frames = traceback.extract_tb(e.__traceback__)
		location = f" ({tb_frames[-1].filename}:{tb_frames[-1].lineno})" if tb_frames else ""
		_write_failure_marker(dataset, model, exp_id, hyperparams,
		                     f"{type(e).__name__}: {e}{location}",
		                     tb_text=traceback.format_exc())
		return False


def _write_failure_marker(dataset: str, model: str, exp_id: str,
                          hyperparams: dict, error: str, tb_text: str = None):
	"""Persist a minimal result file recording the failure so the worker
	loop's has_matching_result() check skips this sub-experiment on rescan
	instead of re-running it forever. The report's metric lookup returns
	None for these and the row is dropped from best-result selection.
	"""
	result_path = get_result_path(dataset, model, exp_id)
	os.makedirs(os.path.dirname(result_path), exist_ok=True)
	marker = {
		'experiment_id': exp_id,
		'model': model,
		'dataset': dataset,
		'applied_hyperparameters': hyperparams,
		'status': 'failed',
		'error': error,
	}
	if tb_text:
		marker['traceback'] = tb_text
	try:
		with open(result_path, 'w') as f:
			json.dump(marker, f, indent=2, default=str)
	except Exception as write_err:
		print(f"  (could not write failure marker: {write_err})")


def run_worker_loop(experiments: list, retrain: bool = False, test: bool = False):
	"""Run a worker loop that claims and executes tasks until none remain.

	Each iteration scans all sub-experiments, skips completed and claimed ones,
	and tries to claim and run the next available task.
	"""
	all_subexps = collect_all_subexperiments(experiments)
	worker_id = os.environ.get('SLURM_ARRAY_TASK_ID', str(os.getpid()))
	print(f"Worker {worker_id} starting. Total sub-experiments: {len(all_subexps)}")

	completed = 0
	failed = 0

	while True:
		claimed_any = False

		for exp_id, dataset, model, hyperparams in all_subexps:
			# Skip if already has results
			if not retrain and has_matching_result(dataset, model, exp_id):
				continue

			# Clean up stale claims
			is_stale_claim(dataset, model, exp_id)

			# Try to claim this task
			if not try_claim_task(dataset, model, exp_id):
				continue

			claimed_any = True
			print(f"Worker {worker_id} claimed Exp {exp_id}: {model} on {dataset}")

			try:
				success = run_single_experiment(dataset, model, exp_id, hyperparams, test=test)
				if success:
					completed += 1
				else:
					failed += 1
			finally:
				release_claim(dataset, model, exp_id)

			# Break to rescan from the beginning
			break

		if not claimed_any:
			break

	print(f"\nWorker {worker_id} finished. Completed: {completed}, Failed: {failed}")


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
			experiment_id=None,
			less=args.less,
			test=args.test,
			retrain=args.retrain,
			plot=args.plot
		)


def handle_experiment_file(args):
	"""Handle experiment file operations with sub-experiment support."""
	# Use default experiment file if not specified
	experiment_file = args.experiment_file if args.experiment_file else 'experiments.json'

	if not os.path.exists(experiment_file):
		print(f"Error: Experiment file '{experiment_file}' not found")
		return 1

	config = load_experiments(experiment_file)
	experiments = config.get('experiments', [])

	if not experiments:
		print("Error: No experiments defined in file")
		return 1

	# Filter by experiment index if specified
	if args.experiment:
		experiments = [e for e in experiments if e.get('index') == args.experiment]
		if not experiments:
			print(f"Error: No experiment with index {args.experiment}")
			return 1

	# List mode - show all sub-experiments
	if args.list:
		print(f"\nExperiments in {experiment_file}:")
		print(f"{'Exp ID':<12} {'Dataset':<12} {'Model':<20} {'Params'}")
		print("-" * 100)

		for exp in experiments:
			exp_idx = exp.get('index')
			dataset = exp.get('dataset', '?')
			model = exp.get('model', '?')
			sub_exps = expand_experiment_to_subexperiments(exp)

			if len(sub_exps) == 1:
				# Single configuration
				exp_id, params = sub_exps[0]
				param_str = ", ".join([f"{k}={v}" for k, v in params.items() if k != 'notes'])
				print(f"{exp_id:<12} {dataset:<12} {model:<20} {param_str}")
			else:
				# Multiple sub-experiments
				print(f"{exp_idx}.*      {dataset:<12} {model:<20} [{len(sub_exps)} sub-experiments]")
				for exp_id, params in sub_exps[:3]:  # Show first 3
					param_str = ", ".join([f"{k}={v}" for k, v in params.items() if k != 'notes'])
					print(f"  {exp_id:<10} {'':<12} {'':<20} {param_str}")
				if len(sub_exps) > 3:
					print(f"  ... ({len(sub_exps) - 3} more)")

		# Show summary with task counts
		all_subexps = collect_all_subexperiments(experiments)
		incomplete_subexps = get_incomplete_subexperiments(all_subexps)
		total_tasks = len(all_subexps)
		incomplete_tasks = len(incomplete_subexps)

		print("\n" + "=" * 100)
		print(f"Summary:")
		print(f"  Total sub-experiments:      {total_tasks}")
		print(f"  Incomplete sub-experiments: {incomplete_tasks}")
		print(f"  Completed sub-experiments:  {total_tasks - incomplete_tasks}")
		print(f"\nFor SLURM array jobs, use: --array=0-{max(0, incomplete_tasks - 1)}")
		if incomplete_tasks > 20:
			print(f"  Or limit concurrency:    --array=0-{max(0, incomplete_tasks - 1)}%20")

		return 0

	# Status mode - show completion status of sub-experiments
	if args.status:
		print(f"\nExperiment Status in {experiment_file}:")
		print(f"{'Exp ID':<12} {'Dataset':<12} {'Model':<20} {'Status':<12} {'Complete/Total'}")
		print("-" * 80)

		for exp in experiments:
			exp_idx = exp.get('index')
			dataset = exp.get('dataset', '?')
			model = exp.get('model', '?')
			sub_exps = expand_experiment_to_subexperiments(exp)

			# Count completed sub-experiments
			completed_count = 0
			for exp_id, _ in sub_exps:
				if has_matching_result(dataset, model, exp_id):
					completed_count += 1

			total_count = len(sub_exps)

			if completed_count == total_count:
				status = "✓ COMPLETE"
			elif completed_count == 0:
				status = "○ TODO"
			else:
				status = "◐ PARTIAL"

			print(f"{exp_idx}.*      {dataset:<12} {model:<20} {status:<12} {completed_count}/{total_count}")

		return 0

	# Worker loop mode
	if args.worker:
		run_worker_loop(experiments, retrain=args.retrain, test=args.test)
		return 0

	# Array job mode
	if args.array_index is not None:
		result = map_array_index_to_subexperiment(experiments, args.array_index, retrain=args.retrain)

		if result is None:
			print(f"Array index {args.array_index} out of range (no incomplete sub-experiments at this index)")
			return 0  # Not an error - just no work to do

		exp_id, dataset, model, hyperparams = result
		print(f"Array job {args.array_index} → Exp {exp_id}: {model} on {dataset}")

		success = run_single_experiment(dataset, model, exp_id, hyperparams, test=args.test)
		return 0 if success else 1

	# Run mode - expand to sub-experiments and run all
	all_subexps = collect_all_subexperiments(experiments)

	print(f"\nRunning experiments...")
	print(f"Config: {experiment_file}")
	print(f"Total: {len(all_subexps)} sub-experiment(s) from {len(experiments)} experiment(s)\n")

	completed = 0
	skipped = 0
	failed = 0

	for exp_id, dataset, model, hyperparams in all_subexps:
		# Run sub-experiment
		if has_matching_result(dataset, model, exp_id):
			skipped += 1
			continue

		success = run_single_experiment(dataset, model, exp_id, hyperparams, test=args.test)
		if success:
			completed += 1
		else:
			failed += 1

	print(f"\n{'='*50}")
	print(f"Summary:")
	print(f"  Completed: {completed}")
	print(f"  Skipped:   {skipped}")
	print(f"  Failed:    {failed}")
	print(f"  Total:     {len(all_subexps)}")
	print(f"\nResults organized by dataset in results/{{dataset}}/")

	return 0 if failed == 0 else 1


def main():
	"""Main entry point."""
	args = parser.parse_args()

	if args.report:
		generate_report(args.dataset, metric=args.metric)
		return 0

	# Determine which mode to run
	# Use experiment file mode if:
	# 1. experiment_file is explicitly provided, OR
	# 2. any experiment-related flag is set (--experiment, --list, --status, --array-index)
	if (args.experiment_file is not None or
	    args.experiment is not None or
	    args.list or
	    args.status or
	    args.array_index is not None):
		# Experiment file mode (handles --experiment, --list, --status, --array-index, or running all)
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
