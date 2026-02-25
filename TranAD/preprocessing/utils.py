import os
import numpy as np
from TranAD.constants import  DEFAULT_OUTPUT_FOLDER


def load_and_save(category, filename, dataset, dataset_folder, output_folder=DEFAULT_OUTPUT_FOLDER):
	"""Load a CSV file and save as normalized numpy array.
	
	Args:
		category (str): Subdirectory name ('train', 'test', etc.).
		filename (str): Name of the file to load.
		dataset (str): Dataset identifier for output filename.
		dataset_folder (str): Root path to dataset folder.
	
	Returns:
		tuple: Shape of loaded array as (num_samples, num_features).
	"""
	temp = np.genfromtxt(os.path.join(dataset_folder, category, filename),
	                         dtype=np.float64,
	                         delimiter=',')
	print(dataset, category, filename, temp.shape)
	np.save(os.path.join(output_folder, f"SMD/{dataset}_{category}.npy"), temp)
	return temp.shape

def load_and_save2(category, filename, dataset, dataset_folder, shape, output_folder=DEFAULT_OUTPUT_FOLDER):
	"""Load anomaly labels from interpretation label file and save as binary numpy array.
	
	Args:
		category (str): Label category name.
		filename (str): Name of the label file to parse.
		dataset (str): Dataset identifier for output filename.
		dataset_folder (str): Root path to dataset folder.
		shape (tuple): Target shape (num_samples, num_features) for labels array.
	
	Returns:
		None. Saves binary label array as '{dataset}_{category}.npy'.
		
	File Format Expected:
		Lines formatted as 'start-end:feature1,feature2,...' where start and end are indices
		and features are 1-indexed column numbers to mark as anomalous.
	"""
	temp = np.zeros(shape)
	with open(os.path.join(dataset_folder, 'interpretation_label', filename), "r") as f:
		ls = f.readlines()
	for line in ls:
		pos, values = line.split(':')[0], line.split(':')[1].split(',')
		start, end, indx = int(pos.split('-')[0]), int(pos.split('-')[1]), [int(i)-1 for i in values]
		temp[start-1:end-1, indx] = 1
	print(dataset, category, filename, temp.shape)
	np.save(os.path.join(output_folder, f"SMD/{dataset}_{category}.npy"), temp)

def normalize(a):
	"""Normalize array to range [0.25, 0.75] using symmetric scaling.
	
	Args:
		a (np.ndarray): Input array of any shape.
	
	Returns:
		np.ndarray: Normalized array with same shape, values in approximately [0.25, 0.75].
	"""
	a = a / np.maximum(np.absolute(a.max(axis=0)), np.absolute(a.min(axis=0)))
	return (a / 2 + 0.5)

def normalize2(a, min_a = None, max_a = None):
	"""Normalize array to range [0, 1] using min-max scaling.
	
	Args:
		a (np.ndarray): Input array to normalize.
		min_a (float, optional): Minimum value for scaling. If None, computed from data.
		max_a (float, optional): Maximum value for scaling. If None, computed from data.
	
	Returns:
		tuple: A tuple containing:
			- normalized_array (np.ndarray): Normalized values in range [0, 1].
			- min_a (float): Minimum value used for normalization.
			- max_a (float): Maximum value used for normalization.
	"""
	if min_a is None: min_a, max_a = min(a), max(a)
	return (a - min_a) / (max_a - min_a), min_a, max_a

def normalize3(a, min_a = None, max_a = None):
	"""Normalize array to range [0, 1] using per-feature min-max scaling.
	
	Args:
		a (np.ndarray): Input array of shape (num_samples, num_features).
		min_a (np.ndarray, optional): Per-feature minimum values. If None, computed from data.
		max_a (np.ndarray, optional): Per-feature maximum values. If None, computed from data.
	
	Returns:
		tuple: A tuple containing:
			- normalized_array (np.ndarray): Normalized values in range [0, 1].
			- min_a (np.ndarray): Per-feature minimum values used for normalization.
			- max_a (np.ndarray): Per-feature maximum values used for normalization.
	"""
	if min_a is None: min_a, max_a = np.min(a, axis = 0), np.max(a, axis = 0)
	return (a - min_a) / (max_a - min_a + 0.0001), min_a, max_a

def convertNumpy(df):
	"""Convert pandas DataFrame to normalized numpy array, skipping first 3 columns and downsampling.
	
	Args:
		df (pd.DataFrame): Input DataFrame with timestamp/metadata in first 3 columns.
	
	Returns:
		np.ndarray: Normalized feature data with shape (num_samples//10, num_features-3),
				downsampled by factor of 10 and scaled to [0, 1].
	"""
	x = df[df.columns[3:]].values[::10, :]
	return (x - x.min(0)) / (x.ptp(0) + 1e-4)


