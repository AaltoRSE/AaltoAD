import os
import pandas as pd
import numpy as np
import re

from TranAD.constants import DEFAULT_DATA_FOLDER
from TranAD.preprocessing.utils import normalize


def parse_timestamp_column(series):
	"""Parse a timestamp column and return integer unix-seconds (nullable Int64).

	Rules:
	- If values are numeric, detect scale (s, ms, us, ns) by magnitude and
	  normalize to seconds, floored to integers.
	- If values are datetime strings, parse to datetimes and convert to integer seconds.
	- Returns a pandas Series of dtype `Int64` with unix seconds or NA.
	"""
	non_na = series.dropna()
	if non_na.empty:
		raise ValueError('Timestamp column contains only nulls')
	first = non_na.iloc[0]
	try:
		is_numeric = isinstance(first, (int, float, np.integer, np.floating))
		if not is_numeric:
			fs = str(first).strip()
			is_numeric = bool(re.match(r'^\d+(\.\d+)?$', fs))
	except Exception:
		is_numeric = False
	if is_numeric:
		numeric = pd.to_numeric(series, errors='coerce').astype(float)
		if numeric.dropna().empty:
			# produce nullable Int64 of NAs
			numeric_seconds = pd.Series([pd.NA] * len(numeric), index=numeric.index, dtype='Int64')
		else:
			maxv = numeric.dropna().abs().max()
			if maxv > 1e17:
				div = 1e9
			elif maxv > 1e14:
				div = 1e6
			elif maxv > 1e11:
				div = 1e3
			else:
				div = 1.0
			secs = np.floor(numeric / div)
			numeric_seconds = pd.Series(secs, index=numeric.index).where(~pd.isna(secs)).astype('Int64')
	else:
		dt = pd.to_datetime(series, errors='coerce')
		if dt.isna().all():
			raise ValueError('Unable to parse timestamps from column')
		# convert ns -> seconds integer
		numeric_seconds = pd.Series((dt.view('int64') // 1_000_000_000), index=dt.index).astype('Int64')
	return numeric_seconds


def find_timestamp_column(df, timestamp_candidates=None):
	"""Find the most likely timestamp column name in `df`.

	Strategy:
	- If any common candidate names are present (case-insensitive), return it.
	- Otherwise, pick the column that yields the most non-null datetimes
	  when parsed with `pd.to_datetime`.
	- Fallback: return the first column name.
	"""
	if timestamp_candidates is None:
		timestamp_candidates = ['timestamp', 'time', 'ts', 'date', 'datetime']
	# exact name match (case-insensitive)
	for name in timestamp_candidates:
		for c in df.columns:
			if c.lower() == name:
				return c
	# otherwise, pick the column that best parses to datetimes
	best_col = None
	best_non_na = 0
	for c in df.columns:
		try:
			ts_try = pd.to_datetime(df[c], errors='coerce')
			non_na = ts_try.notna().sum()
			if non_na > best_non_na:
				best_non_na = non_na
				best_col = c
		except Exception:
			continue
	if best_col is not None and best_non_na > 0:
		return best_col
	# fallback to first column
	return df.columns[0]


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


def _detect_column(df, candidates):
	"""Return first column whose lower-cased name is in candidates, else None."""
	for c in df.columns:
		if c.lower() in candidates:
			return c
	return None


def _shannon_entropy(series):
	"""Shannon entropy (bits) of a categorical Series."""
	p = series.value_counts(normalize=True)
	return -(p * np.log2(p + 1e-10)).sum()



DEFAULT_SEGMENTS = [
	('train', False, 300),
	('calib', False, 60),
	('test',  False, 120),
	('test',  True,  240),
	('test',  False, 120),
	('valid', False, 120),
]


def _validate_segments(segments):
	"""Check segments form a valid layout: each partition appears as one contiguous block."""
	valid_partitions = {'train', 'calib', 'test', 'valid'}
	seen_blocks = {}  # partition -> (first_idx, last_idx)
	for i, (partition, anomalous, seconds) in enumerate(segments):
		if partition not in valid_partitions:
			raise ValueError(f'Unknown partition {partition!r}; must be one of {sorted(valid_partitions)}')
		if not isinstance(anomalous, bool):
			raise ValueError(f'Anomalous flag for segment {i} must be bool, got {type(anomalous).__name__}')
		if not isinstance(seconds, int) or seconds <= 0:
			raise ValueError(f'Seconds for segment {i} must be a positive int, got {seconds!r}')
		if partition in seen_blocks:
			first, last = seen_blocks[partition]
			if i != last + 1:
				raise ValueError(
					f'Partition {partition!r} is not contiguous: appears at index {first} and again at {i} '
					f'after a gap. Group all segments of the same partition together.'
				)
			seen_blocks[partition] = (first, i)
		else:
			seen_blocks[partition] = (i, i)


def load_TOL(folder, csv_path=None, data_folder=DEFAULT_DATA_FOLDER, top_k=10,
             segments=None, extended_features=False):
	"""Load and preprocess TOL dataset (network traffic aggregated by timestamp).

	Groups by integer unix-second and IP to produce per-second per-IP features.
	By default only num unique sources and destinations per IP are included.
	Set extended_features=True to also include in/out event counts, bytes
	in/out, rw ratio, port entropy, rows_per_second, and empty_rows.

	Non-top-k IPs are replaced by 'other_internal' or 'other_external' before
	grouping, so they naturally aggregate in a single groupby pass.

	`segments` is an ordered list of (partition, anomalous, seconds) tuples
	describing the data layout. Partitions are 'train', 'calib', 'test', 'valid'.
	Each partition's segments must be contiguous in the ordering (a partition
	cannot be split by another). Labels are produced for 'test' and 'valid'
	partitions from the anomalous flags.
	"""
	if segments is None:
		segments = DEFAULT_SEGMENTS
	_validate_segments(segments)
	if csv_path:
		df = pd.read_csv(csv_path)
	else:
		df = pd.read_csv(data_folder + '/sample_data.csv')

	# Detect columns
	timestamp_col = find_timestamp_column(df)
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
	bytes_col    = _detect_column(df, ['bytes', 'length', 'len', 'pkt_size', 'size', 'octets', 'framelen'])
	src_port_col = _detect_column(df, ['src_port', 'sport', 'source_port'])

	# Parse timestamp → integer unix seconds
	df['ts_sec'] = parse_timestamp_column(df[timestamp_col])
	df = df.dropna(subset=['ts_sec'])
	df['ts_sec'] = df['ts_sec'].astype(int)

	# Find top-k IPs by combined occurrence count
	all_ips = pd.concat([
		df[src_col].dropna().astype(str).str.strip(),
		df[dst_col].dropna().astype(str).str.strip(),
	])
	all_ips = all_ips[all_ips != '']
	counts = all_ips.value_counts()
	if counts.empty:
		raise ValueError('No IP addresses found in source/destination columns')
	most_common_ip = counts.idxmax()
	internal_prefix = most_common_ip.split('.')[0]
	top_ips = set(counts.head(top_k).index)
	print('TOL: total unique IPs', len(counts))
	print('TOL: most common IP', most_common_ip, '=> internal prefix', internal_prefix)
	print(f'TOL: selecting top_{top_k} IPs (keeps {len(top_ips)})')

	# Replace non-top-k IPs with 'other_internal' / 'other_external'
	def _classify(ip_str):
		ip_str = str(ip_str).strip()
		if ip_str in top_ips:
			return ip_str
		return 'other_internal' if ip_str.startswith(internal_prefix + '.') else 'other_external'

	df[src_col] = df[src_col].astype(str).str.strip().map(_classify)
	df[dst_col] = df[dst_col].astype(str).str.strip().map(_classify)

	if extended_features:
		# Global rows-per-second
		rows_per_second = df.groupby('ts_sec').size().rename('rows_per_second')

		# Empty-row count (rows with no src/dst IP)
		empty_mask = (
			(df[src_col].isna() | (df[src_col].astype(str).str.strip() == '')) &
			(df[dst_col].isna() | (df[dst_col].astype(str).str.strip() == ''))
		)
		empty_rows = df[empty_mask].groupby('ts_sec').size().rename('empty_rows')

	# Outgoing aggregation: group by (ts_sec, src_ip)
	out_grp = df.groupby(['ts_sec', src_col])
	out_agg = pd.DataFrame({'num_dsts': out_grp[dst_col].nunique()})
	if extended_features:
		out_agg['out_count'] = out_grp.size()
		if bytes_col:
			out_agg['bytes_out'] = out_grp[bytes_col].sum()
		if src_port_col:
			out_agg['port_entropy'] = out_grp[src_port_col].apply(_shannon_entropy)

	# Incoming aggregation: group by (ts_sec, dst_ip)
	in_grp = df.groupby(['ts_sec', dst_col])
	in_agg = pd.DataFrame({'num_srcs': in_grp[src_col].nunique()})
	if extended_features:
		in_agg['in_count'] = in_grp.size()
		if bytes_col:
			in_agg['bytes_in'] = in_grp[bytes_col].sum()

	# Unstack IP level → wide format; flatten MultiIndex columns to "{ip}_{metric}"
	out_wide = out_agg.unstack(level=src_col, fill_value=0)
	out_wide.columns = [f'{ip}_{metric}' for metric, ip in out_wide.columns]
	in_wide = in_agg.unstack(level=dst_col, fill_value=0)
	in_wide.columns = [f'{ip}_{metric}' for metric, ip in in_wide.columns]

	# Build full contiguous second-range index and join everything
	ts_min, ts_max = df['ts_sec'].min(), df['ts_sec'].max()
	full_idx = np.arange(ts_min, ts_max + 1)

	features_df = (
		out_wide.reindex(full_idx, fill_value=0)
		.join(in_wide.reindex(full_idx, fill_value=0), how='outer')
		.fillna(0)
	)
	if extended_features:
		features_df = (
			features_df
			.join(rows_per_second.reindex(full_idx, fill_value=0), how='outer')
			.join(empty_rows.reindex(full_idx, fill_value=0), how='outer')
			.fillna(0)
		)
	features_df.index.name = 'ts_sec'

	# Derive rw_ratio per IP
	if extended_features and bytes_col:
		for ip in list(top_ips) + ['other_internal', 'other_external']:
			b_in_col  = f'{ip}_bytes_in'
			b_out_col = f'{ip}_bytes_out'
			if b_in_col in features_df.columns and b_out_col in features_df.columns:
				b_in  = features_df[b_in_col]
				b_out = features_df[b_out_col]
				features_df[f'{ip}_rw_ratio'] = b_in / (b_in + b_out + 1e-10)

	# Diagnostic: first-second summary
	feature_names = features_df.columns.tolist()
	print('TOL: feature columns', feature_names)
	if len(features_df) > 0:
		print('TOL: first second ts_sec', features_df.index[0])
		print('TOL: first second counts:')
		row = features_df.iloc[0]
		for n in feature_names:
			print(' ', n, ':', row[n])

	# Walk segments in order, slicing features and accumulating per-partition
	# label arrays. Each segment claims `seconds` rows from the feature stream.
	features = features_df.values.astype(float)
	total_needed = sum(s[2] for s in segments)
	if features.shape[0] < total_needed:
		raise ValueError(
			f'Segments require {total_needed} rows but only {features.shape[0]} available after aggregation'
		)

	partitions = {}  # name -> list of feature slices, in order
	label_segs = {}  # name -> list of (seconds, anomalous) for label construction
	offset = 0
	for partition, anomalous, seconds in segments:
		partitions.setdefault(partition, []).append(features[offset:offset + seconds])
		label_segs.setdefault(partition, []).append((seconds, anomalous))
		offset += seconds

	# Concatenate per-partition slices into single arrays. Missing partitions
	# become zero-row arrays of the right width so downstream save/load works.
	n_feat = features.shape[1]
	train = np.concatenate(partitions['train'], axis=0) if 'train' in partitions else np.empty((0, n_feat))
	calib = np.concatenate(partitions['calib'], axis=0) if 'calib' in partitions else np.empty((0, n_feat))
	test  = np.concatenate(partitions['test'],  axis=0) if 'test'  in partitions else np.empty((0, n_feat))
	valid = np.concatenate(partitions['valid'], axis=0) if 'valid' in partitions else np.empty((0, n_feat))

	if train.size == 0:
		raise ValueError('Training partition is empty after split; cannot normalize')
	train, min_a, max_a = normalize(train)
	if calib.size != 0:
		calib, _, _ = normalize(calib, min_a, max_a)
	if test.size != 0:
		test, _, _ = normalize(test, min_a, max_a)
	if valid.size != 0:
		valid, _, _ = normalize(valid, min_a, max_a)

	def _build_labels(arr, name):
		out = np.zeros_like(arr)
		row = 0
		for seconds, anomalous in label_segs.get(name, []):
			if anomalous:
				out[row:row + seconds] = 1
			row += seconds
		return out

	labels = _build_labels(test, 'test')
	valid_labels = _build_labels(valid, 'valid')
	n_feat_lbl = max(labels.shape[1] if labels.size else 1, 1)
	print(f"  Anomaly labels: {int(labels.sum() / n_feat_lbl)}/{labels.shape[0]} test rows, "
	      f"{int(valid_labels.sum() / n_feat_lbl) if valid_labels.size else 0}/{valid_labels.shape[0]} valid rows")

	for name, arr in [('train', train), ('test', test), ('calib', calib), ('valid', valid), ('labels', labels), ('valid_labels', valid_labels)]:
		np.save(os.path.join(folder, f'{name}.npy'), arr.astype('float64'))
	if csv_path:
		print(f"Processed {csv_path} as TOL -> {folder}/")
		print(f"  train.npy: {train.shape}, test.npy: {test.shape}, calib.npy: {calib.shape}, valid.npy: {valid.shape}, labels.npy: {labels.shape}")
