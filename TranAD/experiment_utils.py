"""
Utility functions for running experiments.
"""

import os
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, Tuple


def get_git_hash():
	"""Get current git commit hash."""
	import subprocess
	try:
		result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
		                       capture_output=True, text=True, cwd=os.path.dirname(__file__))
		return result.stdout.strip() if result.returncode == 0 else 'unknown'
	except:
		return 'unknown'


def load_hyperparameters(dataset: str, model_name: str, config_path: str = 'hyperparameters.json') -> Dict:
	"""Load hyperparameters from config file.
	
	Args:
		dataset (str): Dataset name
		model_name (str): Model name
		config_path (str): Path to hyperparameter config file
		
	Returns:
		dict: Hyperparameters for the model/dataset combination
	"""
	try:
		if not os.path.isabs(config_path):
			config_path = os.path.join(os.path.dirname(__file__), '..', config_path)
		
		with open(config_path, 'r') as f:
			config = json.load(f)
		
		# Check dataset-specific sweep first, then fall back to defaults
		if 'sweeps' in config and dataset in config['sweeps'] and model_name in config['sweeps'][dataset]:
			return config['sweeps'][dataset][model_name]
		elif 'defaults' in config and model_name in config['defaults']:
			return config['defaults'][model_name]
		else:
			return {}
	except Exception as e:
		print(f'Warning: Could not load hyperparameters from {config_path}: {e}')
		return {}


def apply_hyperparameters(model: torch.nn.Module, hyperparams: Dict) -> Dict:
	"""Apply hyperparameters to model instance.
	
	Args:
		model (torch.nn.Module): Model instance
		hyperparams (dict): Hyperparameters to apply
		
	Returns:
		dict: The hyperparameters that were actually applied
	"""
	applied = {}
	for key, value in hyperparams.items():
		if hasattr(model, key):
			setattr(model, key, value)
			applied[key] = value
	return applied


def load_hyperparams_from_string(hyperparams_str: str) -> Dict:
	"""Load hyperparameters from a JSON string or file path.
	
	Args:
		hyperparams_str (str): JSON string or path to hyperparameter file
		
	Returns:
		dict: Hyperparameters to apply
	"""
	if not hyperparams_str:
		return {}
	
	# Try to parse as JSON string first
	try:
		return json.loads(hyperparams_str)
	except json.JSONDecodeError:
		pass
	
	# Try to load as file
	if os.path.exists(hyperparams_str):
		try:
			with open(hyperparams_str) as f:
				return json.load(f)
		except:
			pass
	
	return {}


def convert_to_windows(data: torch.Tensor, model_obj, model_name: str) -> torch.Tensor:
	"""Convert time series data into sliding windows for model input.
	
	Args:
		data (torch.Tensor): Input time series data with shape (num_samples, num_features).
		model_obj: Model object that contains the n_window attribute specifying window size.
		model_name (str): Name of the model (to determine window format).
	
	Returns:
		torch.Tensor: Stacked windows with shape (num_samples, window_size, num_features) for TranAD/Attention
			models or (num_samples, window_size * num_features) for other models.
	"""
	windows = []
	w_size = model_obj.n_window
	for i, g in enumerate(data):
		if i >= w_size:
			w = data[i - w_size:i]
		else:
			w = torch.cat([data[0].repeat(w_size - i, 1), data[0:i]])
		# TranAD and Attention models use 3D windows, others use flattened
		windows.append(w if 'TranAD' in model_name or 'Attention' in model_name else w.view(-1))
	return torch.stack(windows)


def load_dataset(dataset: str, less: bool = False, output_folder: str = 'processed') -> Tuple:
	"""Load pre-processed training, testing, and label data for a given dataset.
	
	Args:
		dataset (str): Name of the dataset (e.g., 'SMD', 'SMAP', 'MSL', 'UCR', 'NAB').
			Must have corresponding processed .npy files in the output folder.
		less (bool): Whether to train using less data (by cutting to 20% of training data).
		output_folder (str): Path to the processed data folder (default: 'processed').
	
	Returns:
		tuple: A tuple containing:
			- train_loader (DataLoader): PyTorch DataLoader for training data.
			- test_loader (DataLoader): PyTorch DataLoader for test data.
			- labels (np.ndarray): Ground truth anomaly labels with shape (num_test_samples, num_features).
	
	Raises:
		Exception: If processed data folder does not exist for the given dataset.
	"""
	from TranAD.utils import cut_array
	
	folder = os.path.join(output_folder, dataset)
	if not os.path.exists(folder):
		raise Exception('Processed Data not found.')
	
	loader = []
	for file in ['train', 'test', 'labels']:
		if dataset == 'SMD':
			file = 'machine-1-1_' + file
		if dataset == 'SMAP':
			file = 'P-1_' + file
		if dataset == 'MSL':
			file = 'C-1_' + file
		if dataset == 'UCR':
			file = '136_' + file
		if dataset == 'NAB':
			file = 'ec2_request_latency_system_failure_' + file
		loader.append(np.load(os.path.join(folder, f'{file}.npy')))
	
	if less:
		loader[0] = cut_array(0.2, loader[0])
	
	train_loader = DataLoader(loader[0], batch_size=loader[0].shape[0])
	test_loader = DataLoader(loader[1], batch_size=loader[1].shape[0])
	labels = loader[2]
	
	return train_loader, test_loader, labels
