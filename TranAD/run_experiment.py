"""
Experiment running functions.

This module contains functions for running individual and multiple experiments.
"""

import os
import hashlib
import pandas as pd
import csv
import json
import importlib
from typing import Dict, Tuple
from time import time
from pprint import pprint

import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_recall_curve, confusion_matrix

import TranAD
from TranAD import models
from TranAD import constants
from TranAD import plotting
from TranAD import pot
from TranAD import merlin
from TranAD import diagnosis
from TranAD import utils

import tempfile
import shutil
import re


def loss_stats(name, x):
	"""Print summary statistics (mean, asymmetric stds, percentiles, min/max) for a loss array."""
	x = np.asarray(x)
	mean = float(x.mean())
	above = x[x > mean]
	below = x[x < mean]
	std_pos = float(above.std()) if above.size else 0.0
	std_neg = float(below.std()) if below.size else 0.0
	pcts = np.percentile(x, [1, 50, 90, 99, 99.9])
	print(
		f'{name}: n={x.size} mean={mean:.6g} '
		f'std+={std_pos:.6g} std-={std_neg:.6g} '
		f'p1={pcts[0]:.6g} p50={pcts[1]:.6g} '
		f'p90={pcts[2]:.6g} p99={pcts[3]:.6g} p99.9={pcts[4]:.6g} '
		f'min={float(x.min()):.6g} max={float(x.max()):.6g}'
	)


def oracle_f1(scores, labels, calib_fraction=0.5, seed=0):
	"""Pick the F1-optimal threshold on a calibration split, score the held-out split.

	Splits `(scores, labels)` into a stratified-random calibration set (size `calib_fraction`)
	and a held-out evaluation set (remainder). The threshold is chosen to maximise F1 on the
	calibration set; the returned metrics are computed on the evaluation set. This makes the
	oracle an honest upper bound (uses labels, but not the labels it's evaluated on).

	Stratified rather than sequential because attack timestamps in real datasets are clustered
	in time — a sequential split can leave one half with a wildly different attack base rate
	than the other, producing thresholds calibrated to the wrong distribution.

	Returns a dict with threshold, f1, precision, recall, fpr, and confusion-matrix counts
	on the evaluation set. Returns None if calibration has no usable thresholds.
	"""
	scores = np.asarray(scores)
	labels = np.asarray(labels)
	n = len(scores)

	rng = np.random.default_rng(seed)
	pos_idx = np.where(labels == 1)[0]
	neg_idx = np.where(labels == 0)[0]
	rng.shuffle(pos_idx)
	rng.shuffle(neg_idx)
	n_calib_pos = int(len(pos_idx) * calib_fraction)
	n_calib_neg = int(len(neg_idx) * calib_fraction)
	calib_idx = np.concatenate([pos_idx[:n_calib_pos], neg_idx[:n_calib_neg]])
	eval_idx  = np.concatenate([pos_idx[n_calib_pos:], neg_idx[n_calib_neg:]])
	calib_scores, calib_labels = scores[calib_idx], labels[calib_idx]
	eval_scores,  eval_labels  = scores[eval_idx],  labels[eval_idx]
	n_calib = len(calib_idx)

	precision, recall, thresholds = precision_recall_curve(calib_labels, calib_scores)
	if len(thresholds) == 0:
		return None
	f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
	best = int(np.nanargmax(f1))
	thr = float(thresholds[best])

	pred = (eval_scores >= thr).astype(int)
	tn, fp, fn, tp = confusion_matrix(eval_labels, pred, labels=[0, 1]).ravel()
	denom_p = tp + fp
	denom_r = tp + fn
	prec = tp / denom_p if denom_p else 0.0
	rec  = tp / denom_r if denom_r else 0.0
	f1_eval = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
	fpr = fp / (fp + tn) if (fp + tn) else 0.0
	return {
		'threshold': thr,
		'f1': float(f1_eval),
		'precision': float(prec),
		'recall': float(rec),
		'fpr': float(fpr),
		'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
		'calib_fraction': calib_fraction,
		'n_calib': int(n_calib),
		'n_eval': int(n - n_calib),
	}


def _convert_json_types(obj):
	"""Convert numpy/pandas types to native Python types for JSON serialisation."""
	if isinstance(obj, (np.integer, np.floating)):
		return obj.item()
	if isinstance(obj, np.ndarray):
		return obj.tolist()
	if isinstance(obj, pd.Timestamp):
		return obj.isoformat()
	if isinstance(obj, (pd.Series, pd.DataFrame)):
		return obj.to_dict(orient='records')
	if isinstance(obj, (int, float, str, bool)):
		return obj
	return str(obj)


def _safe_write_json(path, data):
	"""Atomically write `data` as JSON to `path` via a tempfile + replace."""
	tmp_fd, tmp_path = tempfile.mkstemp(suffix='.tmp', prefix='tmp_result_', dir=os.path.dirname(path))
	os.close(tmp_fd)
	try:
		with open(tmp_path, 'w') as tf:
			json.dump(data, tf, indent=2, default=_convert_json_types)
			tf.flush()
			os.fsync(tf.fileno())
		shutil.move(tmp_path, path)
	finally:
		if os.path.exists(tmp_path):
			try:
				os.remove(tmp_path)
			except Exception:
				pass


def _hyperparams_hash(hyperparams: dict) -> str:
	"""Return a short hash (first 8 chars of SHA256) of the sorted JSON representation of hyperparams.

	Args:
		hyperparams (dict): Hyperparameter dictionary to hash.

	Returns:
		str: 8-character hex string, or "default" if hyperparams is empty or None.
	"""
	if not hyperparams:
		return 'default'
	serialized = json.dumps(hyperparams, sort_keys=True)
	return hashlib.sha256(serialized.encode()).hexdigest()[:8]


def save_model(model, optimizer, scheduler, epoch, accuracy_list, model_name: str, dataset_name: str, root_path: str = '', hyperparams: dict = None, metadata: dict = None):
	"""Save model checkpoint including state dicts and training metadata.

	Args:
		model (torch.nn.Module): The neural network model to save.
		optimizer (torch.optim.Optimizer): The optimizer state to save (e.g., AdamW).
		scheduler (torch.optim.lr_scheduler): Learning rate scheduler state to save.
		epoch (int): Current epoch number for checkpoint tracking.
		accuracy_list (list): List of tuples containing (loss, learning_rate) for each epoch.
		model_name (str): Name of the model for checkpoint path.
		dataset_name (str): Name of the dataset for checkpoint path.
		root_path (str): Root path prefix for the checkpoints directory.
		hyperparams (dict): Hyperparameters used to train the model; included in the hash subdir.
		metadata (dict): Additional metadata to save (e.g., data shapes, dataset info).

	Returns:
		None. Saves checkpoint to checkpoints/{model_name}_{dataset_name}/{hash}/model.ckpt
		with hyperparams.json and metadata.json alongside it.
	"""
	hp_hash = _hyperparams_hash(hyperparams)
	folder = f'{root_path}checkpoints/{model_name}_{dataset_name}/{hp_hash}/'
	os.makedirs(folder, exist_ok=True)
	file_path = f'{folder}/model.ckpt'
	torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'accuracy_list': accuracy_list}, file_path)
	with open(f'{folder}/hyperparams.json', 'w') as hf:
		json.dump(hyperparams if hyperparams else {}, hf, indent=2)
	if metadata:
		with open(f'{folder}/metadata.json', 'w') as mf:
			json.dump(metadata, mf, indent=2)


def load_model(
		modelname: str,
		dims: int,
		dataset_name: str,
        hyperparams_str: str = None,
		retrain: bool = False,
		test: bool = False,
		root_path: str = ''
	):
	"""Load or create a model with optimizer and scheduler, resuming from checkpoint if available.
	
	Args:
		modelname (str): Name of the model class to instantiate (e.g., 'TranAD', 'USAD', 'DAGMM').
		dims (int): Number of features/dimensions in the input data.
		dataset_name (str): Dataset name for loading hyperparameters and checkpoint path.
		hyperparams_str (str): JSON string or file path with hyperparameters to apply.
		retrain (bool): Force retrain (don't load checkpoint).
		test (bool): Test mode flag.
		root_path (str): Root path for loading checkpoints.
	
	Returns:
		tuple: A tuple containing:
			- model (torch.nn.Module): The loaded or newly created model in double precision.
			- optimizer (torch.optim.AdamW): AdamW optimizer configured for the model.
			- scheduler (torch.optim.lr_scheduler.StepLR): Learning rate scheduler (step every 5 epochs).
			- epoch (int): Starting epoch (-1 if new model, otherwise loaded epoch from checkpoint).
			- accuracy_list (list): List of training metrics from checkpoint or empty list for new model.
			- applied_hyperparams (dict): Hyperparameters that were applied to the model.
			- hp_hash (str): Short hash of the applied hyperparameters, used in checkpoint paths.
	"""
	model_class = getattr(models, modelname)

	hyperparams = utils.load_hyperparams_from_string(hyperparams_str) if hyperparams_str else {}
	model = model_class(dims, **hyperparams).double()
	applied_hyperparams = hyperparams
	hp_hash = _hyperparams_hash(applied_hyperparams)

	optimizer = torch.optim.AdamW(model.parameters(), lr=model.lr, weight_decay=1e-5)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)
	fname = f'{root_path}checkpoints/{modelname}_{dataset_name}/{hp_hash}/model.ckpt'
	if os.path.exists(fname) and (not retrain or test):
		print(f"{utils.color.GREEN}Loading pre-trained model: {model.name}{utils.color.ENDC}")
		try:
			checkpoint = torch.load(fname, weights_only=False)
		except TypeError:
			checkpoint = torch.load(fname)
		model.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
		epoch = checkpoint['epoch']
		accuracy_list = checkpoint['accuracy_list']
	else:
		print(f"{utils.color.GREEN}Creating new model: {model.name}{utils.color.ENDC}")
		epoch = -1
		accuracy_list = []
	return model, optimizer, scheduler, epoch, accuracy_list, applied_hyperparams, hp_hash


def append_benchmark_row(model_name, dataset_name, result_dict, bench_path=os.path.join('results', 'benchmarks.csv')):
	"""Append a single benchmark row to CSV, creating file/header if needed.

	Writes both POT and oracle metrics so downstream tracking can compare the
	threshold-method result against the F1-optimal upper bound.
	"""
	pot = result_dict.get('pot') or {}
	oracle = result_dict.get('oracle') or {}
	try:
		os.makedirs(os.path.dirname(bench_path) or '.', exist_ok=True)
		write_header = (not os.path.exists(bench_path)) or os.path.getsize(bench_path) == 0
		with open(bench_path, 'a', newline='') as csvfile:
			writer = csv.writer(csvfile)
			if write_header:
				writer.writerow([
					'model', 'dataset',
					'pot_precision', 'pot_recall', 'pot_AUC', 'pot_f1',
					'oracle_precision', 'oracle_recall', 'oracle_f1', 'oracle_threshold',
				])
			writer.writerow([
				model_name, dataset_name,
				pot.get('precision'), pot.get('recall'), pot.get('ROC/AUC'), pot.get('f1'),
				oracle.get('precision'), oracle.get('recall'), oracle.get('f1'), oracle.get('threshold'),
			])
	except Exception as e:
		print(f"Could not write benchmark CSV: {e}")


def run_experiment(model_name: str, dataset_name: str, model_name_full: str = None,
                   hyperparams_str: str = None, experiment_id: str = None,
                   less: bool = False, test: bool = False, retrain: bool = False,
                   plot: bool = False) -> Dict:
	"""Run a single experiment with specified parameters.

	Args:
		model_name (str): Name of the model (e.g., 'TranAD')
		dataset_name (str): Name of the dataset (e.g., 'TOL')
		model_name_full (str): Full model name for checkpoints (e.g., 'TranAD_TOL'), defaults to f'{model_name}_{dataset_name}'
		hyperparams_str (str): JSON string or file path with hyperparameters
		experiment_id (str): Optional experiment identifier for tracking
		less (bool): Whether to train using less data
		test (bool): Test mode (skip training)
		retrain (bool): Force retrain

	Returns:
		dict: result dictionary produced at the end of the experiment.
	"""
	if model_name_full is None:
		model_name_full = f'{model_name}_{dataset_name}'

	preds = []
	train_loader, test_loader, labels = utils.load_dataset(dataset_name, less=less, output_folder=constants.output_folder)

	calib_fraction = 0.2
	trainD = next(iter(train_loader))
	split_index = int(len(trainD) * (1 - calib_fraction))
	train_data = trainD[:split_index]
	calib_data = trainD[split_index:]
	trainD, trainO = train_data, train_data
	calibD, calibO = calib_data, calib_data
	

	if model_name in ['MERLIN']:
		# Call MERLIN's runner and append its result to benchmarks CSV
		res = merlin.run_merlin(test_loader, labels, dataset_name)
		append_benchmark_row(model_name, dataset_name, res)
		return res
	model, optimizer, scheduler, epoch, accuracy_list, applied_hyperparams, hp_hash = load_model(
		model_name, labels.shape[1], dataset_name,
		hyperparams_str=hyperparams_str, retrain=retrain, test=test
	)

	## Prepare data
	testD = next(iter(test_loader))
	testO = testD
	if hasattr(model, 'n_window'):
		trainD, testD = utils.convert_to_windows(trainD, model), utils.convert_to_windows(testD, model)
		calibD = utils.convert_to_windows(calibD, model)

	### Training phase
	if not test:
		print(f'{utils.color.HEADER}Training {model_name} on {utils.color.ENDC}')
		if hasattr(model, 'epochs'):
			num_epochs = model.epochs
		else:
			num_epochs = 5
		e = epoch + 1
		start = time()
		for e in tqdm(list(range(epoch+1, epoch+num_epochs+1))):
			feats = trainO.shape[1]
			lossT, lr = model.train_step(e, trainD, optimizer, scheduler, feats)
			accuracy_list.append((lossT, lr))
		print(utils.color.BOLD+'Training time: '+"{:10.4f}".format(time()-start)+ utils.color.ENDC)
		data_metadata = {
			'dataset': dataset_name,
			'train_shape': list(trainO.shape),
			'test_shape': list(testO.shape),
			'labels_shape': list(labels.shape),
			'num_features': int(trainO.shape[1]),
		}
		save_model(model, optimizer, scheduler, e, accuracy_list, model_name, dataset_name, hyperparams=applied_hyperparams, metadata=data_metadata)
		if plot:
			utils.plot_accuracies(accuracy_list, f'{model_name}_{dataset_name}')

	### Testing phase
	model.eval()
	print(f'{utils.color.HEADER}Testing {model_name} on {utils.color.ENDC}')
	feats = testO.shape[1]
	start = time()
	with torch.no_grad():
		loss, y_pred = model.eval_step(0, testD, optimizer, scheduler, feats)
		lossT, _ = model.eval_step(0, trainD, optimizer, scheduler, feats)
		lossC, _ = model.eval_step(0, calibD, optimizer, scheduler, feats)
	print(utils.color.BOLD+'Testing time: '+"{:10.4f}".format(time()-start)+ utils.color.ENDC)

	### Plot curves
	if plot:
		if 'TranAD' in model.name:
			testO = torch.roll(testO, 1, 0)
		plotting.plotter(f'{model_name}_{dataset_name}', testO, y_pred, loss, labels)

	### Scores
	df = pd.DataFrame()
	feats = trainO.shape[1]
	for i in tqdm(range(loss.shape[1]), desc='Evaluating features'):
		lt, l, ls = lossC[:, i], loss[:, i], labels[:, i]
		result, pred = pot.pot_eval(lt, l, ls)
		preds.append(pred)
		df = pd.concat([df, pd.DataFrame([result])], ignore_index=True)

	lossTfinal, lossFinal = np.mean(lossT, axis=1), np.mean(loss, axis=1)
	lossCfinal = np.mean(lossC, axis=1)
	labelsFinal = (np.sum(labels, axis=1) >= 1) + 0

	# Sanity checks: are inputs and predictions actually in the range we expect?
	_trainD_arr = trainD.detach().cpu().numpy() if hasattr(trainD, 'detach') else np.asarray(trainD)
	_testD_arr  = testD.detach().cpu().numpy()  if hasattr(testD,  'detach') else np.asarray(testD)
	print(f'trainD range: min={_trainD_arr.min():.4g} max={_trainD_arr.max():.4g} '
	      f'<0: {(_trainD_arr < 0).sum()}/{_trainD_arr.size} '
	      f'>1: {(_trainD_arr > 1).sum()}/{_trainD_arr.size}')
	print(f'calibD range: min={float(lossC.min()):.4g} max={float(lossC.max()):.4g}'
	   	  f'<0: {(lossC < 0).sum()}/{lossC.size} '
		  f'>1: {(lossC > 1).sum()}/{lossC.size}')
	print(f'testD  range: min={_testD_arr.min():.4g} max={_testD_arr.max():.4g} '
	      f'<0: {(_testD_arr < 0).sum()}/{_testD_arr.size} '
	      f'>1: {(_testD_arr > 1).sum()}/{_testD_arr.size}')
	print(f'y_pred range: min={float(y_pred.min()):.4g} max={float(y_pred.max()):.4g}')
	print(f'raw loss [N, F] shape={loss.shape} min={float(loss.min()):.4g} max={float(loss.max()):.4g}')
	print(f'NaN/Inf in loss? nan={int(np.isnan(loss).sum())} inf={int(np.isinf(loss).sum())}')

	loss_test_normal = lossFinal[labelsFinal == 0]
	loss_test_attack = lossFinal[labelsFinal == 1]

	loss_stats('train loss (mean over feats)', lossTfinal)
	loss_stats('calib  loss (mean over feats)', lossCfinal)
	loss_stats('test  loss | label=0 (normal)', loss_test_normal)
	if loss_test_attack.size:
		loss_stats('test  loss | label=1 (attack)', loss_test_attack)
	loss_stats('test  loss (mean over feats)', lossFinal)

	oracle = oracle_f1(lossFinal, labelsFinal)
	if oracle is not None:
		print(
			f'oracle (F1-max on test): threshold={oracle["threshold"]:.6g} '
			f'f1={oracle["f1"]:.4f} precision={oracle["precision"]:.4f} '
			f'recall={oracle["recall"]:.4f} fpr={oracle["fpr"]:.6f} '
			f'TP={oracle["tp"]} FP={oracle["fp"]} FN={oracle["fn"]} TN={oracle["tn"]}'
		)
		oracle_pred = (lossFinal >= oracle['threshold']).astype(int)
	else:
		oracle_pred = np.zeros_like(labelsFinal)

	pot_result, pot_pred = pot.pot_eval(lossCfinal, lossFinal, labelsFinal)

	result = {
		'pot': pot_result,
		'oracle': oracle,
	}
	result.update(diagnosis.hit_att(loss, labels))
	result.update(diagnosis.ndcg(loss, labels))

	# Write labels with timestamps to a csv
	timestamps = utils.load_timestamps(dataset_name, constants.output_folder)
	labels_df = pd.DataFrame({
		'timestamp': timestamps,
		'pot_label': pot_pred,
		'oracle_label': oracle_pred,
		'ground_truth': labelsFinal,
	})
	labels_csv_path = os.path.join('results', dataset_name, f'{model_name}_exp{experiment_id}_labels.csv')
	os.makedirs(os.path.dirname(labels_csv_path), exist_ok=True)
	labels_df.to_csv(labels_csv_path, index=False)

	# Add metadata to results
	result['git_hash'] = utils.get_git_hash()
	result['model'] = model_name
	result['dataset'] = dataset_name
	result['applied_hyperparameters'] = applied_hyperparams
	if experiment_id is not None:
		result['experiment_id'] = experiment_id
	
	results_dir = os.path.join('results', dataset_name)
	os.makedirs(results_dir, exist_ok=True)

	# Include experiment id in filename if provided
	if experiment_id is not None:
		results_file = os.path.join(results_dir, f'{model_name}_exp{experiment_id}_results.json')
	else:
		results_file = os.path.join(results_dir, f'{model_name}_results.json')

	try:
		_safe_write_json(results_file, result)
	except Exception as e:
		# If write fails, ensure no truncated file left and raise
		print(f"Warning: failed to write results file {results_file}: {e}")
		# Attempt to remove incomplete file
		try:
			if os.path.exists(results_file):
				os.remove(results_file)
		except Exception:
			pass
		raise
	
	print(df)
	pprint(result)
	# Append benchmark to CSV: model, dataset, precision, recall, AUC, f1
	append_benchmark_row(model_name, dataset_name, result)
	return result


def run_all(models_list=None, datasets_list=None, args_obj=None):
	"""Run all models on all datasets and print a summary report.

	Args:
		models_list (list): List of model names to run. If None, discovers from TranAD.models.
		datasets_list (list): List of dataset names to run. If None, discovers from processed folder.
		args_obj: Optional args object with retrain attribute.
	"""
	# discover datasets from processed output folder if not provided
	if datasets_list is None:
		folder = os.path.join(TranAD.constants.output_folder)
		if not os.path.exists(folder):
			raise Exception(f'Processed data folder not found: {folder}')
		datasets_list = sorted([d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))])

	# discover model class names if not provided
	if models_list is None:
		importlib.reload(TranAD.models)
		models_list = [name for name in dir(TranAD.models) if name[0].isupper() and callable(getattr(TranAD.models, name))] + ["MERLIN"]

	# Load existing benchmark entries to optionally skip already-run experiments
	bench_file = os.path.join('results', 'benchmarks.csv')
	existing_runs = set()
	if os.path.exists(bench_file):
		try:
			with open(bench_file, 'r', newline='') as csvfile:
				reader = csv.DictReader(csvfile)
				for row in reader:
					m = row.get('model')
					d = row.get('dataset')
					if m and d:
						existing_runs.add((m, d))
		except Exception as e:
			print(f"Warning: could not read benchmark CSV: {e}")

	summary = {}
	for dataset in datasets_list:
		# Initialize constants for this dataset
		for modelname in models_list:
			constants.initialize(dataset, modelname)
			# reload models so they pick up new dataset
			importlib.reload(TranAD.models)
			
			# Skip if this run is already present in the benchmarks CSV and retrain not requested
			should_retrain = getattr(args_obj, 'retrain', False) if args_obj else False
			if (modelname, dataset) in existing_runs and not should_retrain:
				print(f"Skipping {modelname} on {dataset}: already in results/benchmarks.csv (use --retrain to override)")
				continue
			
			print(f'Running {modelname} on {dataset}')
			# run and collect result
			try:
				res = run_experiment(modelname, dataset)
			except Exception as e:
				res = {'error': str(e)}
			summary[(modelname, dataset)] = res

	# Print concise report
	# Build summary DataFrame
	rows = []
	for (m, d), r in summary.items():
		if isinstance(r, dict) and 'precision' in r:
			rows.append({'model': m, 'dataset': d, 'precision': r.get('precision'), 'recall': r.get('recall'), 'AUC': r.get('ROC/AUC'), 'F1': r.get('f1')})
		else:
			rows.append({'model': m, 'dataset': d, 'precision': None, 'recall': None, 'AUC': None, 'F1': None, 'error': str(r)})
	report = pd.DataFrame(rows)
	print('\n=== Summary Report ===')
	print(report.to_string(index=False))
	return summary, report
