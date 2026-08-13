"""
Utility functions for running experiments.
"""

import os
import json
from unittest import loader
import torch
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import os
import pandas as pd


class color:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    RED = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def plot_accuracies(accuracy_list, folder):
	os.makedirs(f'plots/{folder}/', exist_ok=True)
	trainAcc = [i[0] for i in accuracy_list]
	lrs = [i[1] for i in accuracy_list]
	plt.xlabel('Epochs')
	plt.ylabel('Average Training Loss')
	plt.plot(range(len(trainAcc)), trainAcc, label='Average Training Loss', linewidth=1, linestyle='-', marker='.')
	plt.twinx()
	plt.plot(range(len(lrs)), lrs, label='Learning Rate', color='r', linewidth=1, linestyle='--', marker='.')
	plt.savefig(f'plots/{folder}/training-graph.pdf')
	plt.clf()

def cut_array(percentage, arr):
	print(f'{color.BOLD}Slicing dataset to {int(percentage*100)}%{color.ENDC}')
	mid = round(arr.shape[0] / 2)
	window = round(arr.shape[0] * percentage * 0.5)
	return arr[mid - window : mid + window, :]

def getresults2(df, result):
	results2, df1, df2 = {}, df.sum(), df.mean()
	for a in ['FN', 'FP', 'TP', 'TN']:
		results2[a] = df1[a]
	for a in ['precision', 'recall']:
		results2[a] = df2[a]
	results2['f1*'] = 2 * results2['precision'] * results2['recall'] / (results2['precision'] + results2['recall'])
	return results2

def get_git_hash():
	"""Get current git commit hash."""
	import subprocess
	try:
		result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
		                       capture_output=True, text=True, cwd=os.path.dirname(__file__))
		return result.stdout.strip() if result.returncode == 0 else 'unknown'
	except:
		return 'unknown'


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
		with open(hyperparams_str) as f:
			return json.load(f)

	raise ValueError(f"Could not parse hyperparameters: not valid JSON and not an existing file path: {hyperparams_str!r}")


def convert_to_windows(data: torch.Tensor, model_obj) -> torch.Tensor:
	"""Convert time series data into sliding windows for model input.
	
	Args:
		data (torch.Tensor): Input time series data with shape (num_samples, num_features).
		model_obj: Model object that contains the n_window and flat_window attributes.
	
	Returns:
		torch.Tensor: Stacked windows with shape (num_samples - n_window, window_size, num_features)
			when flat_window is False, or (num_samples - n_window, window_size * num_features)
			when flat_window is True. The first n_window positions have no complete
			history and are dropped (no padding), so callers must trim labels and
			timestamps by n_window to stay aligned.
	"""
	windows = []
	w_size = model_obj.n_window
	flatten = getattr(model_obj, 'flat_window', False)
	for i in range(w_size, len(data)):
		w = data[i - w_size:i]
		windows.append(w.reshape(-1) if flatten else w)
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
			- calib_loader (DataLoader or None): DataLoader for a held-out calibration partition
			  if `calib.npy` exists in the processed folder, else None. Callers that need calibration
			  data should fall back to training data when this is None.

	Raises:
		Exception: If processed data folder does not exist for the given dataset.
	"""
	from AaltoAD.utils import cut_array

	folder = os.path.join(output_folder, dataset)
	if not os.path.exists(folder):
		raise Exception('Processed Data not found.')

	arrays = []
	for file in ['train', 'test', 'labels', 'calib']:
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
		filepath = os.path.join(folder, f'{file}.npy')
		if not os.path.exists(filepath):
			arrays.append(None)
		else:
			mat = np.load(filepath)
			if mat.shape[0] == 0:
				arrays.append(None)
				continue
			if file == 'labels':
				arrays.append(mat)
			else:
				arrays.append(torch.from_numpy(mat))

	return arrays[0], arrays[1], arrays[2], arrays[3]

def load_timestamps(dataset: str, output_folder: str = 'processed') -> np.ndarray:
	"""Load timestamps for a given dataset.
	

		Args:
			dataset (str): Name of the dataset (e.g., 'SMD', 'SMAP', 'MSL', 'UCR', 'NAB').
				Must have corresponding processed .npy files in the output folder.
			output_folder (str): Path to the processed data folder (default: 'processed').
		Returns:
			pd.Series: Timestamps corresponding to the test data samples.
	"""

	folder = os.path.join(output_folder, dataset)
	if not os.path.exists(folder):
		raise Exception('Processed Data not found.')
	
	timestamps_file = os.path.join(folder, 'timestamps.csv')
	if not os.path.exists(timestamps_file):
		# If no timestamps, create a simple numerical index
		test_file = os.path.join(folder, 'test.npy')
		n = np.load(test_file, mmap_mode='r').shape[0]
		return pd.Series(np.arange(n), name='index')
	timestamps_df = pd.read_csv(timestamps_file, header=None).iloc[:, 0]
	return timestamps_df

	
