"""
Experiment running functions.

This module contains functions for running individual and multiple experiments.
These functions can be imported and called programmatically without requiring
command-line argument parsing.
"""

import os
import pandas as pd
import csv
import json
import importlib
from typing import Dict, Tuple
from time import time
from pprint import pprint

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np

import TranAD
from TranAD import models
from TranAD import constants
from TranAD import plotting
from TranAD import pot
from TranAD.utils import *
from TranAD.diagnosis import *
from TranAD.merlin import *
from TranAD.experiment_utils import (
	get_git_hash, load_hyperparameters, apply_hyperparameters,
	load_hyperparams_from_string, convert_to_windows, load_dataset
)


def save_model(model, optimizer, scheduler, epoch, accuracy_list, model_name: str, dataset_name: str):
	"""Save model checkpoint including state dicts and training metadata.
	
	Args:
		model (torch.nn.Module): The neural network model to save.
		optimizer (torch.optim.Optimizer): The optimizer state to save (e.g., AdamW).
		scheduler (torch.optim.lr_scheduler): Learning rate scheduler state to save.
		epoch (int): Current epoch number for checkpoint tracking.
		accuracy_list (list): List of tuples containing (loss, learning_rate) for each epoch.
		model_name (str): Name of the model for checkpoint path.
		dataset_name (str): Name of the dataset for checkpoint path.
	
	Returns:
		None. Saves checkpoint to checkpoints/{model_name}_{dataset_name}/model.ckpt
	"""
	folder = f'checkpoints/{model_name}_{dataset_name}/'
	os.makedirs(folder, exist_ok=True)
	file_path = f'{folder}/model.ckpt'
	torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'accuracy_list': accuracy_list}, file_path)


def load_model(modelname: str, dims: int, dataset_name: str, model_name_full: str, 
               hyperparams_str: str = None, retrain: bool = False, test: bool = False):
	"""Load or create a model with optimizer and scheduler, resuming from checkpoint if available.
	
	Args:
		modelname (str): Name of the model class to instantiate (e.g., 'TranAD', 'USAD', 'DAGMM').
		dims (int): Number of features/dimensions in the input data.
		dataset_name (str): Dataset name for loading hyperparameters and checkpoint path.
		model_name_full (str): Full model name including variant (e.g., 'TranAD_MBA').
		hyperparams_str (str): JSON string or file path with hyperparameters to apply.
		retrain (bool): Force retrain (don't load checkpoint).
		test (bool): Test mode flag.
	
	Returns:
		tuple: A tuple containing:
			- model (torch.nn.Module): The loaded or newly created model in double precision.
			- optimizer (torch.optim.AdamW): AdamW optimizer configured for the model.
			- scheduler (torch.optim.lr_scheduler.StepLR): Learning rate scheduler (step every 5 epochs).
			- epoch (int): Starting epoch (-1 if new model, otherwise loaded epoch from checkpoint).
			- accuracy_list (list): List of training metrics from checkpoint or empty list for new model.
			- applied_hyperparams (dict): Hyperparameters that were applied to the model.
	"""
	model_class = getattr(models, modelname)
	model = model_class(dims).double()
	
	# Load and apply hyperparameters
	# Priority: command-line args > config file > model defaults
	hyperparams = load_hyperparameters(dataset_name, modelname)
	cmd_hyperparams = load_hyperparams_from_string(hyperparams_str) if hyperparams_str else {}
	hyperparams.update(cmd_hyperparams)  # Command-line overrides config
	applied_hyperparams = apply_hyperparameters(model, hyperparams)
	
	optimizer = torch.optim.AdamW(model.parameters(), lr=model.lr, weight_decay=1e-5)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)
	fname = f'checkpoints/{model_name_full}/model.ckpt'
	if os.path.exists(fname) and (not retrain or test):
		print(f"{color.GREEN}Loading pre-trained model: {model.name}{color.ENDC}")
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
		print(f"{color.GREEN}Creating new model: {model.name}{color.ENDC}")
		epoch = -1
		accuracy_list = []
	return model, optimizer, scheduler, epoch, accuracy_list, applied_hyperparams


def append_benchmark_row(model_name, dataset_name, result_dict, bench_path=os.path.join('results', 'benchmarks.csv')):
	"""Append a single benchmark row to CSV, creating file/header if needed.

	Args:
		model_name (str), dataset_name (str), result_dict (dict), bench_path (str)
	"""
	try:
		os.makedirs(os.path.dirname(bench_path) or '.', exist_ok=True)
		write_header = (not os.path.exists(bench_path)) or os.path.getsize(bench_path) == 0
		with open(bench_path, 'a', newline='') as csvfile:
			writer = csv.writer(csvfile)
			if write_header:
				writer.writerow(['model', 'dataset', 'precision', 'recall', 'AUC', 'f1'])
			writer.writerow([model_name, dataset_name, result_dict.get('precision'), result_dict.get('recall'), result_dict.get('ROC/AUC'), result_dict.get('f1')])
	except Exception as e:
		print(f"Could not write benchmark CSV: {e}")


def backprop(epoch, model, data, dataO, optimizer, scheduler, training=True):
	"""Dispatcher for model-specific backprop/eval routines.

	This function delegates to smaller helper functions implemented below.
	"""
	feats = dataO.shape[1]
	if 'DAGMM' in model.name:
		return _backprop_dagmm(epoch, model, data, dataO, optimizer, scheduler, training, feats)
	if 'Attention' in model.name:
		return _backprop_attention(epoch, model, data, dataO, optimizer, scheduler, training, feats)
	if 'OmniAnomaly' in model.name:
		return _backprop_omni(epoch, model, data, dataO, optimizer, scheduler, training, feats)
	if 'USAD' in model.name:
		return _backprop_usad(epoch, model, data, dataO, optimizer, scheduler, training, feats)
	if model.name in ['GDN', 'MTAD_GAT', 'MSCRED', 'CAE_M']:
		return _backprop_simple(epoch, model, data, dataO, optimizer, scheduler, training, feats)
	if 'GAN' in model.name:
		return _backprop_gan(epoch, model, data, dataO, optimizer, scheduler, training, feats)
	if 'TranAD' in model.name:
		return _backprop_tranad(epoch, model, data, dataO, optimizer, scheduler, training, feats)
	return _backprop_default(epoch, model, data, dataO, optimizer, scheduler, training, feats)


def _backprop_dagmm(epoch, model, data, dataO, optimizer, scheduler, training, feats):
	l = nn.MSELoss(reduction='none')
	compute = models.ComputeLoss(model, 0.1, 0.005, 'cpu', model.n_gmm)
	n = epoch + 1
	l1s = []
	l2s = []
	if training:
		for d in data:
			_, x_hat, z, gamma = model(d)
			l1, l2 = l(x_hat, d), l(gamma, d)
			l1s.append(torch.mean(l1).item())
			l2s.append(torch.mean(l2).item())
			loss = torch.mean(l1) + torch.mean(l2)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
		scheduler.step()
		tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)},\tL2 = {np.mean(l2s)}')
		return np.mean(l1s)+np.mean(l2s), optimizer.param_groups[0]['lr']
	else:
		ae1s = []
		for d in data:
			_, x_hat, _, _ = model(d)
			ae1s.append(x_hat)
		ae1s = torch.stack(ae1s)
		y_pred = ae1s[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
		loss = l(ae1s, data)[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
		return loss.detach().numpy(), y_pred.detach().numpy()


def _backprop_attention(epoch, model, data, dataO, optimizer, scheduler, training, feats):
	l = nn.MSELoss(reduction='none')
	n = epoch + 1
	l1s = []
	if training:
		for d in data:
			ae, ats = model(d)
			l1 = l(ae, d)
			l1s.append(torch.mean(l1).item())
			loss = torch.mean(l1)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
		scheduler.step()
		tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)}')
		return np.mean(l1s), optimizer.param_groups[0]['lr']
	else:
		ae1s, y_pred = [], []
		for d in data:
			ae1, ats = model(d)
			y_pred.append(ae1[-1])
			ae1s.append(ae1)
		ae1s, y_pred = torch.stack(ae1s), torch.stack(y_pred)
		loss = torch.mean(l(ae1s, data), axis=1)
		return loss.detach().numpy(), y_pred.detach().numpy()


def _backprop_omni(epoch, model, data, dataO, optimizer, scheduler, training, feats):
	l = nn.MSELoss(reduction='none')
	if training:
		mses, klds = [], []
		for i, d in enumerate(data):
			y_pred, mu, logvar, hidden = model(d, hidden if i else None)
			MSE = l(y_pred, d)
			KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=0)
			loss = MSE + model.beta * KLD
			mses.append(torch.mean(MSE).item())
			klds.append(model.beta * torch.mean(KLD).item())
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
		tqdm.write(f'Epoch {epoch},\tMSE = {np.mean(mses)},\tKLD = {np.mean(klds)}')
		scheduler.step()
		return loss.item(), optimizer.param_groups[0]['lr']
	else:
		y_preds = []
		for i, d in enumerate(data):
			y_pred, _, _, hidden = model(d, hidden if i else None)
			y_preds.append(y_pred)
		y_pred = torch.stack(y_preds)
		MSE = l(y_pred, data)
		return MSE.detach().numpy(), y_pred.detach().numpy()


def _backprop_usad(epoch, model, data, dataO, optimizer, scheduler, training, feats):
	l = nn.MSELoss(reduction='none')
	n = epoch + 1
	l1s, l2s = [], []
	if training:
		for d in data:
			ae1s, ae2s, ae2ae1s = model(d)
			l1 = (1 / n) * l(ae1s, d) + (1 - 1/n) * l(ae2ae1s, d)
			l2 = (1 / n) * l(ae2s, d) - (1 - 1/n) * l(ae2ae1s, d)
			l1s.append(torch.mean(l1).item())
			l2s.append(torch.mean(l2).item())
			loss = torch.mean(l1 + l2)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
		scheduler.step()
		tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)},\tL2 = {np.mean(l2s)}')
		return np.mean(l1s)+np.mean(l2s), optimizer.param_groups[0]['lr']
	else:
		ae1s, ae2s, ae2ae1s = [], [], []
		for d in data:
			ae1, ae2, ae2ae1 = model(d)
			ae1s.append(ae1)
			ae2s.append(ae2)
			ae2ae1s.append(ae2ae1)
		ae1s, ae2s, ae2ae1s = torch.stack(ae1s), torch.stack(ae2s), torch.stack(ae2ae1s)
		y_pred = ae1s[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
		loss = 0.1 * l(ae1s, data) + 0.9 * l(ae2ae1s, data)
		loss = loss[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
		return loss.detach().numpy(), y_pred.detach().numpy()


def _backprop_simple(epoch, model, data, dataO, optimizer, scheduler, training, feats):
	l = nn.MSELoss(reduction='none')
	n = epoch + 1
	l1s = []
	if training:
		for i, d in enumerate(data):
			if 'MTAD_GAT' in model.name:
				x, h = model(d, h if i else None)
			else:
				x = model(d)
			loss = torch.mean(l(x, d))
			l1s.append(torch.mean(loss).item())
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
		tqdm.write(f'Epoch {epoch},\tMSE = {np.mean(l1s)}')
		return np.mean(l1s), optimizer.param_groups[0]['lr']
	else:
		xs = []
		for d in data:
			if 'MTAD_GAT' in model.name:
				x, h = model(d, None)
			else:
				x = model(d)
			xs.append(x)
		xs = torch.stack(xs)
		y_pred = xs[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
		loss = l(xs, data)
		loss = loss[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
		return loss.detach().numpy(), y_pred.detach().numpy()


def _backprop_gan(epoch, model, data, dataO, optimizer, scheduler, training, feats):
	l = nn.MSELoss(reduction='none')
	bcel = nn.BCELoss(reduction='mean')
	msel = nn.MSELoss(reduction='mean')
	real_label, fake_label = torch.tensor([0.9]), torch.tensor([0.1])
	real_label, fake_label = real_label.type(torch.DoubleTensor), fake_label.type(torch.DoubleTensor)
	n = epoch + 1
	mses, gls, dls = [], [], []
	if training:
		for d in data:
			model.discriminator.zero_grad()
			_, real, fake = model(d)
			dl = bcel(real, real_label) + bcel(fake, fake_label)
			dl.backward()
			model.generator.zero_grad()
			optimizer.step()
			z, _, fake = model(d)
			mse = msel(z, d)
			gl = bcel(fake, real_label)
			tl = gl + mse
			tl.backward()
			model.discriminator.zero_grad()
			optimizer.step()
			mses.append(mse.item())
			gls.append(gl.item())
			dls.append(dl.item())
		tqdm.write(f'Epoch {epoch},\tMSE = {np.mean(mses)},\tG = {np.mean(gls)},\tD = {np.mean(dls)}')
		return np.mean(gls)+np.mean(dls), optimizer.param_groups[0]['lr']
	else:
		outputs = []
		for d in data:
			z, _, _ = model(d)
			outputs.append(z)
		outputs = torch.stack(outputs)
		y_pred = outputs[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
		loss = l(outputs, data)
		loss = loss[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
		return loss.detach().numpy(), y_pred.detach().numpy()


def _backprop_tranad(epoch, model, data, dataO, optimizer, scheduler, training, feats):
	l = nn.MSELoss(reduction='none')
	data_x = torch.DoubleTensor(data)
	dataset = TensorDataset(data_x, data_x)
	bs = model.batch if training else len(data)
	dataloader = DataLoader(dataset, batch_size=bs)
	n = epoch + 1
	l1s, l2s = [], []
	if training:
		for d, _ in dataloader:
			local_bs = d.shape[0]
			window = d.permute(1, 0, 2)
			elem = window[-1, :, :].view(1, local_bs, feats)
			z = model(window, elem)
			l1 = l(z, elem) if not isinstance(z, tuple) else (1 / n) * l(z[0], elem) + (1 - 1/n) * l(z[1], elem)
			if isinstance(z, tuple):
				z = z[1]
			l1s.append(torch.mean(l1).item())
			loss = torch.mean(l1)
			optimizer.zero_grad()
			loss.backward(retain_graph=True)
			optimizer.step()
		scheduler.step()
		tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)}')
		return np.mean(l1s), optimizer.param_groups[0]['lr']
	else:
		for d, _ in dataloader:
			window = d.permute(1, 0, 2)
			elem = window[-1, :, :].view(1, bs, feats)
			z = model(window, elem)
			if isinstance(z, tuple):
				z = z[1]
		loss = l(z, elem)[0]
		return loss.detach().numpy(), z.detach().numpy()[0]


def _backprop_default(epoch, model, data, dataO, optimizer, scheduler, training, feats):
	l = nn.MSELoss(reduction='mean' if training else 'none')
	y_pred = model(data)
	loss = l(y_pred, data)
	if training:
		tqdm.write(f'Epoch {epoch},\tMSE = {loss}')
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		scheduler.step()
		return loss.item(), optimizer.param_groups[0]['lr']
	else:
		return loss.detach().numpy(), y_pred.detach().numpy()


def run_experiment(model_name: str, dataset_name: str, model_name_full: str = None,
                   hyperparams_str: str = None, experiment_index: int = None,
                   less: bool = False, test: bool = False, retrain: bool = False) -> Dict:
	"""Run a single experiment with specified parameters.

	Args:
		model_name (str): Name of the model (e.g., 'TranAD')
		dataset_name (str): Name of the dataset (e.g., 'TOL')
		model_name_full (str): Full model name for checkpoints (e.g., 'TranAD_TOL'), defaults to f'{model_name}_{dataset_name}'
		hyperparams_str (str): JSON string or file path with hyperparameters
		experiment_index (int): Optional experiment index for tracking
		less (bool): Whether to train using less data
		test (bool): Test mode (skip training)
		retrain (bool): Force retrain

	Returns:
		dict: result dictionary produced at the end of the experiment.
	"""
	if model_name_full is None:
		model_name_full = f'{model_name}_{dataset_name}'

	preds = []
	train_loader, test_loader, labels = load_dataset(dataset_name, less=less, output_folder=constants.output_folder)
	if model_name in ['MERLIN']:
		# Call MERLIN's runner and append its result to benchmarks CSV
		res = run_merlin(test_loader, labels, dataset_name)
		append_benchmark_row(model_name, dataset_name, res)
		return res
	model, optimizer, scheduler, epoch, accuracy_list, applied_hyperparams = load_model(
		model_name, labels.shape[1], dataset_name, model_name_full,
		hyperparams_str=hyperparams_str, retrain=retrain, test=test
	)

	## Prepare data
	trainD, testD = next(iter(train_loader)), next(iter(test_loader))
	trainO, testO = trainD, testD
	if model.name in ['Attention', 'DAGMM', 'USAD', 'MSCRED', 'CAE_M', 'GDN', 'MTAD_GAT', 'MAD_GAN', 'MERLIN'] or 'TranAD' in model.name:
		trainD, testD = convert_to_windows(trainD, model, model_name), convert_to_windows(testD, model, model_name)

	### Training phase
	if not test:
		print(f'{color.HEADER}Training {model_name} on {dataset_name}{color.ENDC}')
		num_epochs = 5
		e = epoch + 1
		start = time()
		for e in tqdm(list(range(epoch+1, epoch+num_epochs+1))):
			lossT, lr = backprop(e, model, trainD, trainO, optimizer, scheduler)
			accuracy_list.append((lossT, lr))
		print(color.BOLD+'Training time: '+"{:10.4f}".format(time()-start)+' s'+color.ENDC)
		save_model(model, optimizer, scheduler, e, accuracy_list, model_name, dataset_name)
		plot_accuracies(accuracy_list, f'{model_name}_{dataset_name}')

	### Testing phase
	torch.zero_grad = True
	model.eval()
	print(f'{color.HEADER}Testing {model_name} on {dataset_name}{color.ENDC}')
	loss, y_pred = backprop(0, model, testD, testO, optimizer, scheduler, training=False)

	### Plot curves
	if not test:
		if 'TranAD' in model.name:
			testO = torch.roll(testO, 1, 0)
		plotting.plotter(f'{model_name}_{dataset_name}', testO, y_pred, loss, labels)

	### Scores
	df = pd.DataFrame()
	lossT, _ = backprop(0, model, trainD, trainO, optimizer, scheduler, training=False)
	for i in range(loss.shape[1]):
		lt, l, ls = lossT[:, i], loss[:, i], labels[:, i]
		result, pred = pot.pot_eval(lt, l, ls)
		preds.append(pred)
		df = pd.concat([df, pd.DataFrame([result])], ignore_index=True)

	lossTfinal, lossFinal = np.mean(lossT, axis=1), np.mean(loss, axis=1)
	labelsFinal = (np.sum(labels, axis=1) >= 1) + 0
	result, _ = pot.pot_eval(lossTfinal, lossFinal, labelsFinal)
	result.update(hit_att(loss, labels))
	result.update(ndcg(loss, labels))
	
	# Add metadata to results
	result['git_hash'] = get_git_hash()
	result['model'] = model_name
	result['dataset'] = dataset_name
	result['applied_hyperparameters'] = applied_hyperparams
	if experiment_index is not None:
		result['experiment_index'] = experiment_index
	
	# Save detailed results with metadata to JSON
	results_dir = os.path.join('results', dataset_name)
	os.makedirs(results_dir, exist_ok=True)
	
	# Include experiment index in filename if provided
	if experiment_index is not None:
		results_file = os.path.join(results_dir, f'{model_name}_exp{experiment_index}_results.json')
	else:
		results_file = os.path.join(results_dir, f'{model_name}_results.json')
	
	with open(results_file, 'w') as f:
		json.dump(result, f, indent=2)
	
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
		folder = os.path.join(TranAD.folderconstants.output_folder)
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
