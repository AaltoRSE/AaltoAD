import os
import sys
import pandas as pd
import numpy as np
import json
import argparse
from TranAD.folderconstants import  DEFAULT_OUTPUT_FOLDER, DEFAULT_DATA_FOLDER


datasets = ['synthetic', 'SMD', 'SWaT', 'SMAP', 'MSL', 'WADI', 'MSDS', 'UCR', 'MBA', 'NAB', 'TOL']

wadi_drop = ['2_LS_001_AL', '2_LS_002_AL','2_P_001_STATUS','2_P_002_STATUS']

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

def load_TOL(folder, csv_path=None, data_folder=DEFAULT_DATA_FOLDER):
	"""Load and preprocess TOL dataset (network traffic aggregated by timestamp).
	
	Args:
		folder (str): Output folder path where .npy files will be saved.
		csv_path (str, optional): Path to a CSV file to preprocess. If None, uses default data/sample_data.csv.
	
	Returns:
		None. Saves train.npy, test.npy, and labels.npy in the output folder.
	"""
	if csv_path:
		df = pd.read_csv(csv_path)
	else:
		df = pd.read_csv(os.path.join(data_folder, 'sample_data.csv'))

	# Determine timestamp and protocol column names
	col_name = df.columns[0]
	protocol_col = df.columns[1] if len(df.columns) > 1 else 'protocol'

	# Normalize protocol values (treat missing as 'unknown') and compute top-5 protocols
	df[protocol_col] = df[protocol_col].fillna('unknown').astype(str)
	top_protocols = df[protocol_col].value_counts().index.tolist()[:5]

	# Build per-second, per-protocol counts (rows: timestamp, cols: protocol)
	grp = df.groupby([col_name, protocol_col]).size().unstack(fill_value=0)
	grp = grp.sort_index()

	# Create feature matrix with 5 columns for top protocols and 1 column for the rest
	# Ensure deterministic ordering: top_protocols may be fewer than 5
	features = []
	for i in range(5):
		if i < len(top_protocols):
			p = top_protocols[i]
			features.append(grp.get(p, pd.Series(0, index=grp.index)))
		else:
			features.append(pd.Series(0, index=grp.index))

	# other = total per-timestamp minus sum of selected top protocol counts
	total_per_ts = grp.sum(axis=1)
	selected_sum = pd.concat(features, axis=1).sum(axis=1)
	other = total_per_ts - selected_sum
	features.append(other)

	# Combine into a single numpy array (rows sorted by timestamp)
	features_df = pd.concat(features, axis=1)
	features_df.columns = [f'proto_top_{i+1}' for i in range(5)] + ['proto_other']
	connection_counts = features_df.values.astype(float)
	
	# Create train/test split (70/30)
	split_idx = int(len(connection_counts) * 0.7)
	train = connection_counts[:split_idx, :]
	test = connection_counts[split_idx:, :]
	
	# Normalize using per-feature scaling (preserve columns/features)
	train, min_a, max_a = normalize3(train)
	test, _, _ = normalize3(test, min_a, max_a)
	
	# Create dummy labels (all zeros - no ground truth for this dataset)
	labels = np.zeros_like(test)
	
	# Save files
	for file in ['train', 'test', 'labels']:
		np.save(os.path.join(folder, f'{file}.npy'), eval(file).astype('float64'))
	
	if csv_path:
		print(f"Processed {csv_path} as TOL -> {folder}/")
		print(f"  train.npy: {train.shape}, test.npy: {test.shape}, labels.npy: {labels.shape}")

def load_synthetic(folder, data_folder=DEFAULT_DATA_FOLDER):
	"""Load and preprocess synthetic dataset.
	
	Args:
		folder (str): Output folder path where .npy files will be saved.
	
	Returns:
		None. Saves train.npy, test.npy, and labels.npy in the output folder.
	"""
	train_file = os.path.join(data_folder, 'synthetic', 'synthetic_data_with_anomaly-s-1.csv')
	test_labels = os.path.join(data_folder, 'synthetic', 'test_anomaly.csv')
	dat = pd.read_csv(train_file, header=None)
	split = 10000
	train = normalize(dat.values[:, :split].reshape(split, -1))
	test = normalize(dat.values[:, split:].reshape(split, -1))
	lab = pd.read_csv(test_labels, header=None)
	lab[0] -= split
	labels = np.zeros(test.shape)
	for i in range(lab.shape[0]):
		point = lab.values[i][0]
		labels[point-30:point+30, lab.values[i][1:]] = 1
	test += labels * np.random.normal(0.75, 0.1, test.shape)
	for file in ['train', 'test', 'labels']:
		np.save(os.path.join(folder, f'{file}.npy'), eval(file))

def load_UCR(folder, data_folder=DEFAULT_DATA_FOLDER):
	"""Load and preprocess UCR dataset files.

	Reads all .txt files in data/UCR, parses the header to determine train/test split
	and creates normalized train/test/labels files named {dnum}_train.npy, etc.
	"""
	dataset_folder = os.path.join(data_folder, 'UCR')
	file_list = os.listdir(dataset_folder)
	for filename in file_list:
		if not filename.endswith('.txt'):
			continue
		vals = filename.split('.')[0].split('_')
		dnum, vals = int(vals[0]), vals[-3:]
		vals = [int(i) for i in vals]
		temp = np.genfromtxt(os.path.join(dataset_folder, filename),
							dtype=np.float64,
							delimiter=',')
		min_temp, max_temp = np.min(temp), np.max(temp)
		temp = (temp - min_temp) / (max_temp - min_temp)
		train, test = temp[:vals[0]], temp[vals[0]:]
		labels = np.zeros_like(test)
		labels[vals[1]-vals[0]:vals[2]-vals[0]] = 1
		train, test, labels = train.reshape(-1, 1), test.reshape(-1, 1), labels.reshape(-1, 1)
		for file in ['train', 'test', 'labels']:
			np.save(os.path.join(folder, f'{dnum}_{file}.npy'), eval(file))

def load_SMD(folder):
	"""
	Preprocess the SMD dataset and save train/test/labels arrays.

	SMD has per-machine files under `data/SMD/train` and `data/SMD/test`.
	For each file ending with `.txt`, this function calls `load_and_save`
	to create `{filename}_train.npy` and `{filename}_test.npy` and
	`load_and_save2` to create `{filename}_labels.npy` using the saved
	split index returned from the test processing.

	Args:
		folder (str): Base output folder where processed/* datasets are saved.
	"""
	dataset_folder = os.path.join(DEFAULT_DATA_FOLDER, 'SMD')
	train_dir = os.path.join(dataset_folder, 'train')
	test_dir = os.path.join(dataset_folder, 'test')

	if not os.path.isdir(train_dir):
		raise FileNotFoundError(f'Expected SMD train directory not found: {train_dir}')

	file_list = os.listdir(train_dir)
	for filename in file_list:
		if filename.endswith('.txt'):
			name = filename[:-4]
			load_and_save('train', filename, name, dataset_folder)
			s = load_and_save('test', filename, name, dataset_folder)
			load_and_save2('labels', filename, name, dataset_folder, s)



def load_NAB(folder, data_folder=DEFAULT_DATA_FOLDER):
	dataset_folder = os.path.join(data_folder, 'NAB')
	file_list = os.listdir(dataset_folder)
	with open(dataset_folder + '/labels.json') as f:
		labeldict = json.load(f)
	for filename in file_list:
		if not filename.endswith('.csv'): continue
		df = pd.read_csv(dataset_folder+'/'+filename)
		vals = df.values[:,1]
		labels = np.zeros_like(vals, dtype=np.float64)
		for timestamp in labeldict['realKnownCause/'+filename]:
			tstamp = timestamp.replace('.000000', '')
			index = np.where(((df['timestamp'] == tstamp).values + 0) == 1)[0][0]
			labels[index-4:index+4] = 1
		min_temp, max_temp = np.min(vals), np.max(vals)
		vals = (vals - min_temp) / (max_temp - min_temp)
		train, test = vals.astype(float), vals.astype(float)
		train, test, labels = train.reshape(-1, 1), test.reshape(-1, 1), labels.reshape(-1, 1)
		fn = filename.replace('.csv', '')
		for file in ['train', 'test', 'labels']:
			np.save(os.path.join(folder, f'{fn}_{file}.npy'), eval(file))



def load_MSDS(folder, data_folder=DEFAULT_DATA_FOLDER):
	dataset_folder = os.path.join(data_folder, 'MSDS')
	df_train = pd.read_csv(os.path.join(dataset_folder, 'train.csv'))
	df_test  = pd.read_csv(os.path.join(dataset_folder, 'test.csv'))
	df_train, df_test = df_train.values[::5, 1:], df_test.values[::5, 1:]
	_, min_a, max_a = normalize3(np.concatenate((df_train, df_test), axis=0))
	train, _, _ = normalize3(df_train, min_a, max_a)
	test, _, _ = normalize3(df_test, min_a, max_a)
	labels = pd.read_csv(os.path.join(dataset_folder, 'labels.csv'))
	labels = labels.values[::1, 1:]
	for file in ['train', 'test', 'labels']:
		np.save(os.path.join(folder, f'{file}.npy'), eval(file).astype('float64'))



def load_SWaT(folder, data_folder=DEFAULT_DATA_FOLDER):
	dataset_folder = os.path.join(data_folder, 'SWaT')
	file = os.path.join(dataset_folder, 'series.json')
	df_train = pd.read_json(file, lines=True)[['val']][3000:6000]
	df_test  = pd.read_json(file, lines=True)[['val']][7000:12000]
	train, min_a, max_a = normalize2(df_train.values)
	test, _, _ = normalize2(df_test.values, min_a, max_a)
	labels = pd.read_json(file, lines=True)[['noti']][7000:12000] + 0
	for file in ['train', 'test', 'labels']:
		np.save(os.path.join(folder, f'{file}.npy'), eval(file))



def load_SMAP_MSL(folder, dataset, data_folder=DEFAULT_DATA_FOLDER):
	dataset_folder = os.path.join(data_folder, 'SMAP_MSL')
	file = os.path.join(dataset_folder, 'labeled_anomalies.csv')
	values = pd.read_csv(file)
	values = values[values['spacecraft'] == dataset]
	filenames = values['chan_id'].values.tolist()
	for fn in filenames:
		train = np.load(f'{dataset_folder}/train/{fn}.npy')
		test = np.load(f'{dataset_folder}/test/{fn}.npy')
		train, min_a, max_a = normalize3(train)
		test, _, _ = normalize3(test, min_a, max_a)
		np.save(f'{folder}/{fn}_train.npy', train)
		np.save(f'{folder}/{fn}_test.npy', test)
		labels = np.zeros(test.shape)
		indices = values[values['chan_id'] == fn]['anomaly_sequences'].values[0]
		indices = indices.replace(']', '').replace('[', '').split(', ')
		indices = [int(i) for i in indices]
		for i in range(0, len(indices), 2):
			labels[indices[i]:indices[i+1], :] = 1
		np.save(f'{folder}/{fn}_labels.npy', labels)



def load_WADI(folder, data_folder=DEFAULT_DATA_FOLDER):
	dataset_folder = os.path.join(data_folder, 'WADI')
	ls = pd.read_csv(os.path.join(dataset_folder, 'WADI_attacklabels.csv'))
	train = pd.read_csv(os.path.join(dataset_folder, 'WADI_14days.csv'), skiprows=1000, nrows=2e5)
	test = pd.read_csv(os.path.join(dataset_folder, 'WADI_attackdata.csv'))
	train.dropna(how='all', inplace=True); test.dropna(how='all', inplace=True)
	train.fillna(0, inplace=True); test.fillna(0, inplace=True)
	test['Time'] = test['Time'].astype(str)
	test['Time'] = pd.to_datetime(test['Date'] + ' ' + test['Time'])
	labels = test.copy(deep = True)
	for i in test.columns.tolist()[3:]: labels[i] = 0
	for i in ['Start Time', 'End Time']:
		ls[i] = ls[i].astype(str)
		ls[i] = pd.to_datetime(ls['Date'] + ' ' + ls[i])
	for index, row in ls.iterrows():
		to_match = row['Affected'].split(', ')
		matched = []
		for i in test.columns.tolist()[3:]:
			for tm in to_match:
				if tm in i:
					matched.append(i); break
		st, et = str(row['Start Time']), str(row['End Time'])
		labels.loc[(labels['Time'] >= st) & (labels['Time'] <= et), matched] = 1
	train, test, labels = convertNumpy(train), convertNumpy(test), convertNumpy(labels)
	print(train.shape, test.shape, labels.shape)
	for file in ['train', 'test', 'labels']:
		np.save(os.path.join(folder, f'{file}.npy'), eval(file))



def load_MBA(folder, data_folder=DEFAULT_DATA_FOLDER):
	dataset_folder = os.path.join(data_folder, 'MBA')
	ls = pd.read_excel(os.path.join(dataset_folder, 'labels.xlsx'))
	train = pd.read_excel(os.path.join(dataset_folder, 'train.xlsx'))
	test = pd.read_excel(os.path.join(dataset_folder, 'test.xlsx'))
	train, test = train.values[1:,1:].astype(float), test.values[1:,1:].astype(float)
	train, min_a, max_a = normalize3(train)
	test, _, _ = normalize3(test, min_a, max_a)
	ls = ls.values[:,1].astype(int)
	labels = np.zeros_like(test)
	for i in range(-20, 20):
		labels[ls + i, :] = 1
	for file in ['train', 'test', 'labels']:
		np.save(os.path.join(folder, f'{file}.npy'), eval(file))



def load_data(dataset, csv_path=None, output_folder=DEFAULT_OUTPUT_FOLDER, data_folder=DEFAULT_DATA_FOLDER):
	folder = os.path.join(output_folder, dataset)
	os.makedirs(folder, exist_ok=True)

	# If a CSV path was provided, ensure the file exists before proceeding
	if csv_path and not os.path.exists(csv_path):
		raise Exception(f'CSV file not found: {csv_path}')
	
	if dataset == 'TOL':
		load_TOL(folder, csv_path=csv_path, data_folder=data_folder)
	elif csv_path:
		raise Exception(f'CSV processing not implemented for dataset type {dataset}. Currently only TOL is supported.')
	elif dataset == 'UCR':
		load_UCR(folder, data_folder=data_folder)
	elif dataset == 'synthetic':
		load_synthetic(folder, data_folder=data_folder)
	elif dataset == 'SMD':
		load_SMD(folder)
	elif dataset == 'NAB':
		load_NAB(folder, data_folder=data_folder)
	elif dataset == 'MSDS':
		load_MSDS(folder, data_folder=data_folder)
	elif dataset == 'SWaT':
		load_SWaT(folder, data_folder=data_folder)
	elif dataset in ['SMAP', 'MSL']:
		load_SMAP_MSL(folder, dataset, data_folder=data_folder)
	elif dataset == 'WADI':
		load_WADI(folder, data_folder=data_folder)
	elif dataset == 'MBA':
		load_MBA(folder, data_folder=data_folder)
	else:
		raise Exception(f'Not Implemented. Check one of {datasets}')

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Preprocess datasets for anomaly detection')
	parser.add_argument('dataset', nargs='?', help='Dataset type to preprocess (required)')
	parser.add_argument('--csv', type=str, help='Path to CSV file to preprocess (optional, uses dataset type for processing)')
	args = parser.parse_args()
	
	if not args.dataset:
		print("Usage: python preprocess.py <dataset_type>")
		print(f"       where <dataset_type> is one of {datasets}")
		print()
		print("Or:    python preprocess.py <dataset_type> --csv <path_to_csv>")
		print("       where <path_to_csv> is path to a CSV file, processed using <dataset_type> logic")
		sys.exit(1)
	
	# Call load_data with optional csv_path
	load_data(args.dataset, csv_path=args.csv)