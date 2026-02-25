import os
import pandas as pd
import numpy as np
import re

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


def event_count_per_ip(df, ip_col, timestamp_col="timestamp"):
	"""Compute per-second event counts for each IP.

	This bins events to 1-second resolution (flooring timestamps to seconds)
	and returns a DataFrame indexed by second timestamps with one column per IP.

	Args:
		df (pd.DataFrame): Input dataframe containing timestamp and IP columns.
		timestamp_col (str): Name of the timestamp column.
		ip_col (str): Name of the column containing the IP for which to count events.

	Returns:
		pd.DataFrame: Index is per-second timestamps (pd.DatetimeIndex), columns are IPs,
		values are integer event counts per second.
	"""
	if timestamp_col not in df.columns:
		raise KeyError(f"timestamp column '{timestamp_col}' not found in dataframe")
	if ip_col not in df.columns:
		raise KeyError(f"ip column '{ip_col}' not found in dataframe")

	# Expect the dataframe to already contain numeric unix-seconds in
	# `timestamp_col`. Convert those numeric seconds to datetimes for
	# grouping/indexing. Do not attempt to parse arbitrary datetime strings here.
	series = df[timestamp_col]
	numeric = pd.to_numeric(series, errors='coerce')
	if numeric.isna().all():
		raise ValueError('timestamp column must contain numeric unix-seconds (float/int): ' + timestamp_col)
	# convert numeric seconds to datetimes
	ts = pd.to_datetime(numeric, unit='s', errors='coerce')
	if ts.isna().all():
		raise ValueError('Unable to convert numeric timestamp column to datetimes: ' + timestamp_col)

	# floor to seconds
	ts = ts.dt.floor('s')
	working = df[[ip_col]].copy()
	working['__ts'] = ts.values

	# group by second + ip and unstack into wide format
	grp = working.groupby(['__ts', ip_col]).size().unstack(fill_value=0).sort_index()

	# ensure full continuous second-range index
	_start = grp.index.min()
	_end = grp.index.max()
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

	# Try to detect a timestamp-like column instead of assuming first column
	timestamp_candidates = ['timestamp', 'time', 'ts', 'date', 'datetime']
	timestamp_col = None
	for name in timestamp_candidates:
		for c in df.columns:
			if c.lower() == name:
				timestamp_col = c
				break
		if timestamp_col is not None:
			break
	# If not found, pick the column that best parses to datetimes
	if timestamp_col is None:
		best_col = None
		best_non_na = 0
		for c in df.columns:
			try:
				ts_try = pd.to_datetime(df[c], errors='coerce')
				non_na = ts_try.notna().sum()
				# prefer columns that parse and have multiple unique times
				if non_na > best_non_na:
					best_non_na = non_na
					best_col = c
			except Exception:
				continue
		if best_col is not None and best_non_na > 0:
			timestamp_col = best_col
	# fallback to first column if nothing else
	if timestamp_col is None:
		timestamp_col = df.columns[0]
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

	# Robust parsing for timestamp column: inspect the first non-null value and
	# decide whether to treat the column as epoch-seconds or datetime strings.
	series = df[timestamp_col]
	non_na = series.dropna()
	if non_na.empty:
		raise ValueError('Timestamp column contains only nulls: ' + timestamp_col)
	first = non_na.iloc[0]
	try:
		is_numeric = isinstance(first, (int, float, np.integer, np.floating))
		if not is_numeric:
			fs = str(first).strip()
			is_numeric = bool(re.match(r'^\d+(\.\d+)?$', fs))
	except Exception:
		is_numeric = False
	if is_numeric:
		# interpret as epoch timestamps in numeric form. Detect scale
		# (seconds, milliseconds, microseconds, nanoseconds) by magnitude
		numeric = pd.to_numeric(series, errors='coerce').astype(float)
		if numeric.dropna().empty:
			# no numeric values present -> produce nullable Int64 series of NAs
			numeric_seconds = pd.Series([pd.NA] * len(numeric), index=numeric.index, dtype='Int64')
		else:
			maxv = numeric.dropna().abs().max()
			# thresholds based on approximate epoch magnitudes
			if maxv > 1e17:
				# values are likely in nanoseconds
				div = 1e9
			elif maxv > 1e14:
				# values are likely in microseconds
				div = 1e6
			elif maxv > 1e11:
				# values are likely in milliseconds
				div = 1e3
			else:
				# already in seconds
				div = 1.0
			secs = np.floor(numeric / div)
			# floor to integer seconds and keep NA via nullable Int64
			numeric_seconds = pd.Series(secs, index=numeric.index).where(~pd.isna(secs)).astype('Int64')
	else:
		# interpret as datetime strings and convert to numeric unix seconds
		dt = pd.to_datetime(series, errors='coerce')
		if dt.isna().all():
			raise ValueError('Unable to parse timestamps from first column for TOL preprocessing')
		# dt is in ns resolution; convert to integer seconds (nullable Int64)
		numeric_seconds = pd.Series((dt.view('int64') // 1_000_000_000), index=dt.index).astype('Int64')
	if pd.Series(numeric_seconds).isna().all():
		raise ValueError('Unable to obtain numeric unix-seconds from timestamp column')
	# store numeric unix-seconds in dataframe for downstream processing
	df['timestamp'] = numeric_seconds

	grp_out = event_count_per_ip(df, src_col, timestamp_col="timestamp")
	grp_in = event_count_per_ip(df, dst_col, timestamp_col="timestamp")

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
	# print number of unique IPs found
	print('TOL: total unique IPs', len(counts))
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
	
	# Diagnostic: print counts for the first second to allow direct comparison
	if len(features_df) > 0:
		first_dt = features_df.index[0]
		# datetime -> unix seconds
		try:
			first_unix = int(first_dt.value // 1_000_000_000)
		except Exception:
			first_unix = None
		print('TOL: first second datetime', first_dt, 'unix', first_unix)
		print('TOL: first second counts:')
		row = features_df.iloc[0]
		for n in feature_names:
			print(' ', n, ':', int(row.get(n, 0)))
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
