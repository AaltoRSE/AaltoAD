import os
import pandas as pd
import numpy as np

from TranAD.constants import DEFAULT_DATA_FOLDER
from TranAD.preprocessing.utils import normalize3


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
	ts = ts.dt.floor('s')
	working = df[[ip_col]].copy()
	working['__ts'] = ts.values

	# group by second + ip and unstack into wide format
	grp = working.groupby(['__ts', ip_col]).size().unstack(fill_value=0).sort_index()

	# ensure full continuous second-range index
	_start = pd.to_datetime(start) if start is not None else grp.index.min()
	_end = pd.to_datetime(end) if end is not None else grp.index.max()
	full_idx = pd.date_range(_start, _end, freq='s')
	grp = grp.reindex(full_idx, fill_value=0)

	# cast to integer counts
	grp = grp.astype(int)
	grp.index.name = 'timestamp'
	return grp





def load_TOL(folder, csv_path=None, data_folder=DEFAULT_DATA_FOLDER, top_k=10):
	"""Load and preprocess TOL dataset (network traffic aggregated by timestamp).
	
	This function is extracted from the original `preprocess.py` and mirrors
	the behavior expected by the rest of the codebase. It creates per-second,
	per-IP incoming/outgoing features and aggregates non-top IPs into
	internal/external buckets (separate in/out columns).
	"""
	if csv_path:
		df = pd.read_csv(csv_path)
	else:
		df = pd.read_csv(data_folder + '/sample_data.csv')

	col_name = df.columns[0]
	src_col = 'src_ip' if 'src_ip' in df.columns else None
	dst_col = 'dst_ip' if 'dst_ip' in df.columns else None
	if src_col is None or dst_col is None:
		for c in df.columns:
			if src_col is None and 'src' in c.lower(): src_col = c
			if dst_col is None and 'dst' in c.lower(): dst_col = c
		if src_col is None or dst_col is None:
			for c in df.columns:
				if src_col is None and 'ip' in c.lower(): src_col = c
				if dst_col is None and 'ip' in c.lower(): dst_col = c

	ts = pd.to_datetime(df[col_name], errors='coerce')
	if ts.isna().all():
		try:
			ts = pd.to_datetime(df[col_name].astype(float), unit='s', errors='coerce')
		except Exception:
			pass
	if ts.isna().all():
		raise ValueError('Unable to parse timestamps from first column for TOL preprocessing')
	start, end = ts.dt.floor('s').min(), ts.dt.floor('s').max()

	grp_out = event_count_per_ip(df, col_name, src_col, start=start, end=end)
	grp_in = event_count_per_ip(df, col_name, dst_col, start=start, end=end)

	# Compute per-IP mention counts (clean strings) to select top_k
	s = pd.concat([
		df[src_col].dropna().astype(str),
		df[dst_col].dropna().astype(str)
	])
	s = s.str.strip()
	s = s[s != '']
	counts = s.value_counts()
	if counts.empty:
		raise ValueError('No IP addresses found in source/destination columns')
	most_common_ip = counts.idxmax()
	internal_prefix = most_common_ip.split('.')[0]
	print('TOL: most common IP', most_common_ip, '=> internal prefix', internal_prefix)

	selected_ips = counts.index.tolist()[:top_k]
	print(f'TOL: selecting top_{top_k} IPs (keeps {len(selected_ips)})')

	features = []
	feature_names = []
	index = grp_out.index
	for ip in selected_ips:
		if ip in grp_in.columns:
			features.append(grp_in[ip])
		else:
			features.append(pd.Series(0, index=index))
		feature_names.append(f'{ip}_in')
		if ip in grp_out.columns:
			features.append(grp_out[ip])
		else:
			features.append(pd.Series(0, index=index))
		feature_names.append(f'{ip}_out')

	# aggregate remaining IPs into internal/external, keeping separate in/out
	remaining_ips = [ip for ip in counts.index if ip not in selected_ips]
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

	# train/test split and normalization preserved for backwards compatibility
	n_rows = connection_counts.shape[0]
	split_idx = int(n_rows * 0.7)
	if split_idx == 0 and n_rows > 0:
		split_idx = 1
	train = connection_counts[:split_idx, :]
	test = connection_counts[split_idx:, :]
	if train.size == 0:
		raise ValueError('Training partition is empty after split; cannot normalize')
	train, min_a, max_a = normalize3(train)
	if test.size != 0:
		test, _, _ = normalize3(test, min_a, max_a)
	labels = np.zeros_like(test)
	for file in ['train', 'test', 'labels']:
		np.save(os.path.join(folder, f'{file}.npy'), eval(file).astype('float64'))
	if csv_path:
		print(f"Processed {csv_path} as TOL -> {folder}/")
		print(f"  train.npy: {train.shape}, test.npy: {test.shape}, labels.npy: {labels.shape}")
