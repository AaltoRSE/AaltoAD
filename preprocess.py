import os
import pandas as pd
import numpy as np
import json
import argparse
from TranAD.constants import  DEFAULT_OUTPUT_FOLDER, DEFAULT_DATA_FOLDER


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


def list_unique_ips(df, src_col='src_ip', dst_col='dst_ip'):
	"""Return sorted list of unique IPs found in DataFrame across src/dst columns.

	If neither column exists, tries common column names ('ip', 'address').

	Args:
		df (pd.DataFrame): Input dataframe.
		src_col (str): Source IP column name to look for.
		dst_col (str): Destination IP column name to look for.

	Returns:
		list: Sorted list of unique IP strings.
	"""
	ips = set()
	for col in (src_col, dst_col, 'ip', 'address'):
		if col in df.columns:
			vals = df[col].dropna().unique().tolist()
			# convert non-string ip-like entries to str for consistency
			vals = [str(v) for v in vals]
			ips.update(vals)
	# return deterministic ordering
	return sorted(ips)


def count_unique_ips(df, src_col='src_ip', dst_col='dst_ip'):
	"""Return the number of unique IPs found in the DataFrame.

	This is a small wrapper around `list_unique_ips` kept for compatibility
	with the previous function name.
	"""
	return len(list_unique_ips(df, src_col=src_col, dst_col=dst_col))


def event_count_per_ip(df, timestamp_col, ip_col, start=None, end=None):
	"""Compute per-second event counts for each IP.

	This bins events to 1-second resolution (flooring timestamps to seconds)
	and returns a DataFrame indexed by second timestamps with one column per IP.

	Args:
		df (pd.DataFrame): Input dataframe containing timestamp and IP columns.
		timestamp_col (str): Name of the timestamp column.
		ip_col (str): Name of the column containing the IP for which to count events.
		start (pd.Timestamp or None): Optional start time for full index range.
		end (pd.Timestamp or None): Optional end time for full index range.

	Returns:
		pd.DataFrame: Index is per-second timestamps (pd.DatetimeIndex), columns are IPs,
		values are integer event counts per second.
	"""
	if timestamp_col not in df.columns:
		raise KeyError(f"timestamp column '{timestamp_col}' not found in dataframe")
	if ip_col not in df.columns:
		raise KeyError(f"ip column '{ip_col}' not found in dataframe")

	# Parse timestamps robustly
	ts = pd.to_datetime(df[timestamp_col], errors='coerce')
	if ts.isna().all():
		# try treating numeric values as epoch seconds
		try:
			ts = pd.to_datetime(df[timestamp_col].astype(float), unit='s', errors='coerce')
		except Exception:
			pass
	if ts.isna().all():
		raise ValueError('Unable to parse any timestamps from column: ' + timestamp_col)

	# floor to seconds
	ts = ts.dt.floor('S')
	working = df[[ip_col]].copy()
	working['__ts'] = ts.values

	# group by second + ip and unstack into wide format
	grp = working.groupby(['__ts', ip_col]).size().unstack(fill_value=0).sort_index()

	# ensure full continuous second-range index
	_start = pd.to_datetime(start) if start is not None else grp.index.min()
	_end = pd.to_datetime(end) if end is not None else grp.index.max()
	full_idx = pd.date_range(_start, _end, freq='S')
	grp = grp.reindex(full_idx, fill_value=0)

	# cast to integer counts
	grp = grp.astype(int)
	grp.index.name = 'timestamp'
	return grp

def load_TOL(folder, csv_path=None, data_folder=DEFAULT_DATA_FOLDER, top_k=10):
	"""Load and preprocess TOL dataset (network traffic aggregated by timestamp).

	This function reads a CSV (either the provided ``csv_path`` or
	``data/sample_data.csv``) and produces per-second, per-IP
	connection-count features suitable for model training. The function
	assumes the timestamp is in the first column and that source and
	destination IPs are present (commonly named ``src_ip`` and
	``dst_ip``). For each unique IP observed (across source and
	destination columns) the function produces two features: incoming
	(counts where the IP is destination) and outgoing (counts where the
	IP is source). Missing IP values are ignored.

	Processing steps and outputs:
	- Parse timestamps (first column) and floor to 1-second resolution
	  to create per-second bins.
	- Compute per-second event counts for each IP appearing as source
	  (outgoing) and as destination (incoming) using
	  ``event_count_per_ip``. The full time range is aligned across
	  incoming/outgoing counts so both matrices share the same index.
	- For a deterministic column order, the IPs are sorted and the
	  feature columns are created in the order: ``[ip1_in, ip1_out,
	  ip2_in, ip2_out, ...]``.
	- Split the time-ordered rows into a 70% train / 30% test split.
	- Normalize features per-column (per-feature min/max scaling) using
	  ``normalize3``: min/max are computed from the training partition
	  and applied to the test partition. Normalized values are in [0, 1].
	- Create a dummy labels array for the test partition (all zeros;
	  there is no ground-truth anomaly labeling in the sample data).
	- Save three files into ``folder``: ``train.npy``, ``test.npy``, and
	  ``labels.npy``. Files are float64 numpy arrays; columns correspond
	  to interleaved incoming/outgoing counts per sorted IP as described
	  above.

	Args:
		folder (str): Output folder path where .npy files will be saved.
		csv_path (str, optional): Path to a CSV file to preprocess. If
			None, the default ``data/sample_data.csv`` is used.

	Returns:
		None. Side effects: saves ``train.npy``, ``test.npy``, and
		``labels.npy`` to ``folder``. The training min/max used for
		normalization are computed from the training partition and
		reused for the test partition.
	"""
	if csv_path:
		df = pd.read_csv(csv_path)
	else:
		df = pd.read_csv(os.path.join(data_folder, 'sample_data.csv'))

	# Determine timestamp and IP column names
	col_name = df.columns[0]
	src_col = 'src_ip' if 'src_ip' in df.columns else None
	dst_col = 'dst_ip' if 'dst_ip' in df.columns else None
	if src_col is None or dst_col is None:
		# fall back to any column names containing 'src'/'dst' or 'ip'
		for c in df.columns:
			if src_col is None and 'src' in c.lower(): src_col = c
			if dst_col is None and 'dst' in c.lower(): dst_col = c
		if src_col is None or dst_col is None:
			for c in df.columns:
				if src_col is None and 'ip' in c.lower(): src_col = c
				if dst_col is None and 'ip' in c.lower(): dst_col = c
	# Parse timestamps to establish full time range for consistent bins
	ts = pd.to_datetime(df[col_name], errors='coerce')
	if ts.isna().all():
		try:
			ts = pd.to_datetime(df[col_name].astype(float), unit='s', errors='coerce')
		except Exception:
			pass
	if ts.isna().all():
		raise ValueError('Unable to parse timestamps from first column for TOL preprocessing')
	start, end = ts.dt.floor('S').min(), ts.dt.floor('S').max()

	# Compute per-second outgoing (source) and incoming (destination) counts
	grp_out = event_count_per_ip(df, col_name, src_col, start=start, end=end)
	grp_in = event_count_per_ip(df, col_name, dst_col, start=start, end=end)

	# Compute per-IP mention counts (clean strings) to select top_k
	# Coerce to strings, strip whitespace, drop empties
	s = pd.concat([
		df[src_col].dropna().astype(str),
		df[dst_col].dropna().astype(str)
	])
	s = s.str.strip()
	s = s[s != '']
	counts = s.value_counts()
	if counts.empty:
		raise ValueError('No IP addresses found in source/destination columns')
	# determine internal prefix from the most common IP (first octet)
	most_common_ip = counts.idxmax()
	internal_prefix = most_common_ip.split('.')[0]
	print('TOL: most common IP', most_common_ip, '=> internal prefix', internal_prefix)

	# select the top_k IPs by mentions; if fewer than top_k available, take all
	selected_ips = counts.index.tolist()[:top_k]
	print(f'TOL: selecting top_{top_k} IPs (keeps {len(selected_ips)})')

	# Build features: for each selected IP include in/out (two columns)
	# and aggregate all other IPs into two buckets: other_internal and other_external
	features = []
	feature_names = []
	index = grp_out.index
	for ip in selected_ips:
		# incoming
		if ip in grp_in.columns:
			features.append(grp_in[ip])
		else:
			features.append(pd.Series(0, index=index))
		feature_names.append(f'{ip}_in')
		# outgoing
		if ip in grp_out.columns:
			features.append(grp_out[ip])
		else:
			features.append(pd.Series(0, index=index))
		feature_names.append(f'{ip}_out')

	# aggregate remaining IPs into internal/external (sum in+out)
	remaining_ips = [ip for ip in counts.index if ip not in selected_ips]
	# aggregate remaining IPs into internal/external, keeping separate in/out
	other_internal_in = pd.Series(0, index=index, dtype=float)
	other_internal_out = pd.Series(0, index=index, dtype=float)
	other_external_in = pd.Series(0, index=index, dtype=float)
	other_external_out = pd.Series(0, index=index, dtype=float)
	for ip in remaining_ips:
		if ip in grp_in.columns:
			if str(ip).startswith(internal_prefix + '.'):
				other_internal_in = other_internal_in + grp_in[ip]
			else:
				other_external_in = other_external_in + grp_in[ip]
		if ip in grp_out.columns:
			if str(ip).startswith(internal_prefix + '.'):
				other_internal_out = other_internal_out + grp_out[ip]
			else:
				other_external_out = other_external_out + grp_out[ip]

	# append aggregated buckets (internal_in, internal_out, external_in, external_out)
	features.append(other_internal_in)
	feature_names.append('other_internal_in')
	features.append(other_internal_out)
	feature_names.append('other_internal_out')
	features.append(other_external_in)
	feature_names.append('other_external_in')
	features.append(other_external_out)
	feature_names.append('other_external_out')

	features_df = pd.concat(features, axis=1)
	features_df.columns = feature_names
	connection_counts = features_df.values.astype(float)
	print('TOL: feature columns', feature_names)
	
	# Create train/test split (70/30) with guards for small datasets
	n_rows = connection_counts.shape[0]
	split_idx = int(n_rows * 0.7)
	if split_idx == 0 and n_rows > 0:
		# ensure at least one row in training when possible
		split_idx = 1
	train = connection_counts[:split_idx, :]
	test = connection_counts[split_idx:, :]
	print(f'TOL: rows={n_rows}, split_idx={split_idx}, train={train.shape}, test={test.shape}')

	# Normalize using per-feature scaling (preserve columns/features)
	if train.size == 0:
		raise ValueError('Training partition is empty after split; cannot normalize')
	try:
		train, min_a, max_a = normalize3(train)
		# normalize test with same min/max (handle empty test)
		if test.size != 0:
			test, _, _ = normalize3(test, min_a, max_a)
		else:
			test = test.astype(float)
	except Exception as e:
		raise RuntimeError('Normalization failed: ' + str(e))
	
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
	parser.add_argument('datasets', nargs='+', help='One or more dataset types to preprocess (e.g. SMAP SMD SWaT)')
	parser.add_argument('--csv', type=str, help='Path to CSV file to preprocess (optional, only used for TOL)')
	args = parser.parse_args()

	# Iterate over provided dataset names and process each
	for ds in args.datasets:
		if ds not in datasets:
			print(f"Skipping unknown dataset '{ds}'. Supported: {datasets}")
			continue
		try:
			load_data(ds, csv_path=args.csv)
		except Exception as e:
			print(f"Error processing {ds}: {e}")