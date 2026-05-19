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
	"""Check segments form a valid layout: each non-skip partition appears as one contiguous block.

	'skip' segments are unrestricted — they consume seconds from the CSV but produce
	no output, so multiple skip blocks in arbitrary positions are fine.
	"""
	valid_partitions = {'train', 'calib', 'test', 'valid', 'skip'}
	seen_blocks = {}  # partition -> (first_idx, last_idx)
	for i, (partition, anomalous, seconds) in enumerate(segments):
		if partition not in valid_partitions:
			raise ValueError(f'Unknown partition {partition!r}; must be one of {sorted(valid_partitions)}')
		if not isinstance(anomalous, bool):
			raise ValueError(f'Anomalous flag for segment {i} must be bool, got {type(anomalous).__name__}')
		if anomalous and partition == 'skip':
			raise ValueError(f"Segment {i}: 'skip' segments cannot be anomalous")
		if not isinstance(seconds, int) or seconds <= 0:
			raise ValueError(f'Seconds for segment {i} must be a positive int, got {seconds!r}')
		if partition == 'skip':
			continue  # skip is exempt from contiguity tracking
		if partition in seen_blocks:
			first, last = seen_blocks[partition]
			if i != last + 1:
				# Allow contiguity to be broken only by skip segments between blocks.
				for j in range(last + 1, i):
					if segments[j][0] != 'skip':
						raise ValueError(
							f'Partition {partition!r} is not contiguous: appears at index {first} and again at {i} '
							f'after a gap. Group all segments of the same partition together '
							f'(skip-only gaps allowed).'
						)
				seen_blocks[partition] = (first, i)
			else:
				seen_blocks[partition] = (first, i)
		else:
			seen_blocks[partition] = (i, i)


def _detect_ip_columns(df):
	"""Find src/dst IP columns in a TOL CSV using the same fallback order as before."""
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
	return src_col, dst_col


def _load_and_prepare(csv_path):
	"""Read CSV, detect needed columns, parse timestamps. Returns a parsed-CSV record."""
	df = pd.read_csv(csv_path)
	timestamp_col = find_timestamp_column(df)
	src_col, dst_col = _detect_ip_columns(df)
	bytes_col    = _detect_column(df, ['bytes', 'length', 'len', 'pkt_size', 'size', 'octets', 'framelen'])
	src_port_col = _detect_column(df, ['src_port', 'sport', 'source_port'])
	df['ts_sec'] = parse_timestamp_column(df[timestamp_col])
	df = df.dropna(subset=['ts_sec'])
	df['ts_sec'] = df['ts_sec'].astype(int)
	return {
		'path': csv_path, 'df': df,
		'src_col': src_col, 'dst_col': dst_col,
		'bytes_col': bytes_col, 'port_col': src_port_col,
	}


def _baseline_mask(df, segments):
	"""Mask selecting rows whose ts_sec falls in any train segment of the segment list."""
	ts_min = df['ts_sec'].min()
	ranges = []
	offset = 0
	for partition, _, seconds in segments:
		if partition == 'train':
			ranges.append((ts_min + offset, ts_min + offset + seconds))
		offset += seconds
	if not ranges:
		# No train segments — caller (multi-CSV) may still want IP stats from this
		# CSV if it's a pure-test source; return empty mask so it contributes nothing.
		return pd.Series(False, index=df.index)
	mask = pd.Series(False, index=df.index)
	for lo, hi in ranges:
		mask |= (df['ts_sec'] >= lo) & (df['ts_sec'] < hi)
	return mask


def _aggregate_csv(parsed, top_ips, internal_prefix, extended_features):
	"""Aggregate one parsed CSV into a per-second features dataframe.

	Uses the supplied `top_ips` and `internal_prefix` so the column space is
	consistent across CSVs that share a baseline. Non-top-k IPs collapse into
	'other_internal' or 'other_external'.
	"""
	df = parsed['df'].copy()
	src_col, dst_col = parsed['src_col'], parsed['dst_col']
	bytes_col, src_port_col = parsed['bytes_col'], parsed['port_col']

	def _classify(ip_str):
		ip_str = str(ip_str).strip()
		if ip_str in top_ips:
			return ip_str
		return 'other_internal' if ip_str.startswith(internal_prefix + '.') else 'other_external'

	df[src_col] = df[src_col].astype(str).str.strip().map(_classify)
	df[dst_col] = df[dst_col].astype(str).str.strip().map(_classify)

	rows_per_second = None
	empty_rows = None
	if extended_features:
		rows_per_second = df.groupby('ts_sec').size().rename('rows_per_second')
		empty_mask = (
			(df[src_col].isna() | (df[src_col].astype(str).str.strip() == '')) &
			(df[dst_col].isna() | (df[dst_col].astype(str).str.strip() == ''))
		)
		empty_rows = df[empty_mask].groupby('ts_sec').size().rename('empty_rows')

	out_grp = df.groupby(['ts_sec', src_col])
	out_agg = pd.DataFrame({'num_dsts': out_grp[dst_col].nunique()})
	if extended_features:
		out_agg['out_count'] = out_grp.size()
		if bytes_col:
			out_agg['bytes_out'] = out_grp[bytes_col].sum()
		if src_port_col:
			out_agg['port_entropy'] = out_grp[src_port_col].apply(_shannon_entropy)

	in_grp = df.groupby(['ts_sec', dst_col])
	in_agg = pd.DataFrame({'num_srcs': in_grp[src_col].nunique()})
	if extended_features:
		in_agg['in_count'] = in_grp.size()
		if bytes_col:
			in_agg['bytes_in'] = in_grp[bytes_col].sum()

	out_wide = out_agg.unstack(level=src_col, fill_value=0)
	out_wide.columns = [f'{ip}_{metric}' for metric, ip in out_wide.columns]
	in_wide = in_agg.unstack(level=dst_col, fill_value=0)
	in_wide.columns = [f'{ip}_{metric}' for metric, ip in in_wide.columns]

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

	if extended_features and bytes_col:
		for ip in list(top_ips) + ['other_internal', 'other_external']:
			b_in_col  = f'{ip}_bytes_in'
			b_out_col = f'{ip}_bytes_out'
			if b_in_col in features_df.columns and b_out_col in features_df.columns:
				features_df[f'{ip}_rw_ratio'] = features_df[b_in_col] / (features_df[b_in_col] + features_df[b_out_col] + 1e-10)

	return features_df


def _canonical_columns(top_ips, extended_features, has_bytes, has_ports):
	"""Build the canonical column list shared across all CSVs in a combined dataset."""
	ip_keys = sorted(top_ips) + ['other_internal', 'other_external']
	cols = []
	# Outgoing block (per-IP num_dsts + extended)
	for ip in ip_keys:
		cols.append(f'{ip}_num_dsts')
	if extended_features:
		for ip in ip_keys:
			cols.append(f'{ip}_out_count')
		if has_bytes:
			for ip in ip_keys:
				cols.append(f'{ip}_bytes_out')
		if has_ports:
			for ip in ip_keys:
				cols.append(f'{ip}_port_entropy')
	# Incoming block (per-IP num_srcs + extended)
	for ip in ip_keys:
		cols.append(f'{ip}_num_srcs')
	if extended_features:
		for ip in ip_keys:
			cols.append(f'{ip}_in_count')
		if has_bytes:
			for ip in ip_keys:
				cols.append(f'{ip}_bytes_in')
	# Derived
	if extended_features and has_bytes:
		for ip in ip_keys:
			cols.append(f'{ip}_rw_ratio')
	# Global
	if extended_features:
		cols.append('rows_per_second')
		cols.append('empty_rows')
	return cols


def load_TOL(folder, csv_groups=None, csv_path=None, segments=None,
             data_folder=DEFAULT_DATA_FOLDER, top_k=10, extended_features=False):
	"""Load and preprocess TOL dataset (network traffic aggregated by timestamp).

	Two calling modes:

	- Single-CSV (legacy): pass `csv_path` and (optionally) `segments`. One CSV is
	  aggregated, sliced into partitions, and saved.
	- Combined / multi-CSV: pass `csv_groups`, an ordered list of
	  `(csv_paths, segments)` tuples. Each CSV in each group is aggregated
	  independently, sliced by its group's segments, and per-partition outputs
	  are concatenated across all CSVs of all groups. Top-k IPs and the
	  internal/external prefix are computed once from baseline (train) rows
	  across ALL CSVs in ALL groups so the feature column space is consistent.

	Partitions are 'train', 'calib', 'test', 'valid'. Within each group's
	segment list, each partition must be contiguous. Labels are produced for
	'test' and 'valid' partitions from the anomalous flags.
	"""
	# Normalize input modes into a uniform csv_groups list.
	if csv_groups is None:
		if csv_path is None:
			csv_path = data_folder + '/sample_data.csv'
		if segments is None:
			segments = DEFAULT_SEGMENTS
		csv_groups = [([csv_path], segments)]
	if not csv_groups:
		raise ValueError('At least one (csv_paths, segments) group must be provided')

	# Validate each group's segments.
	for paths, segs in csv_groups:
		if not paths:
			raise ValueError('Every group must contain at least one CSV path')
		if not segs:
			raise ValueError('Every group must contain at least one segment spec')
		_validate_segments(segs)

	# Pass 1: load each CSV, parse columns + timestamps. Also collect baseline IPs.
	parsed_records = []  # list of (parsed_csv_dict, group_segments)
	all_baseline_ips = []
	for paths, segs in csv_groups:
		for p in paths:
			parsed = _load_and_prepare(p)
			parsed_records.append((parsed, segs))
			mask = _baseline_mask(parsed['df'], segs)
			if mask.any():
				all_baseline_ips.append(parsed['df'].loc[mask, parsed['src_col']].dropna().astype(str).str.strip())
				all_baseline_ips.append(parsed['df'].loc[mask, parsed['dst_col']].dropna().astype(str).str.strip())

	if not all_baseline_ips:
		raise ValueError('No train segments found across any CSV — cannot determine baseline IP set')

	all_ips_concat = pd.concat(all_baseline_ips)
	all_ips_concat = all_ips_concat[all_ips_concat != '']
	counts = all_ips_concat.value_counts()
	if counts.empty:
		raise ValueError('No IP addresses found in baseline source/destination columns')
	most_common_ip = counts.idxmax()
	internal_prefix = most_common_ip.split('.')[0]
	top_ips = set(counts.head(top_k).index)
	total_baseline_rows = sum(len(s) for s in all_baseline_ips) // 2  # we appended src + dst
	print(f'TOL: unique IPs across all baselines: {len(counts)} '
	      f'(total baseline rows aggregated: {total_baseline_rows})')
	print(f'TOL: most common IP {most_common_ip} => internal prefix {internal_prefix}')
	print(f'TOL: selecting top_{top_k} IPs from combined baseline (keeps {len(top_ips)})')

	# Determine canonical column set. has_bytes / has_ports are True if ANY CSV has them.
	has_bytes = any(rec['bytes_col'] for rec, _ in parsed_records)
	has_ports = any(rec['port_col'] for rec, _ in parsed_records)
	canonical_cols = _canonical_columns(top_ips, extended_features, has_bytes, has_ports)

	# Pass 2: aggregate each CSV, reindex to canonical columns, slice, accumulate.
	partitions = {'train': [], 'calib': [], 'test': [], 'valid': []}
	label_segs = {'train': [], 'calib': [], 'test': [], 'valid': []}
	for parsed, segs in parsed_records:
		features_df = _aggregate_csv(parsed, top_ips, internal_prefix, extended_features)
		features_df = features_df.reindex(columns=canonical_cols, fill_value=0)
		features = features_df.values.astype(float)
		total_needed = sum(s[2] for s in segs)
		if features.shape[0] < total_needed:
			raise ValueError(
				f"CSV {parsed['path']}: segments require {total_needed} rows but only "
				f"{features.shape[0]} available after aggregation"
			)
		offset = 0
		for partition, anomalous, seconds in segs:
			if partition == 'skip':
				# Consume seconds without writing them to any partition.
				offset += seconds
				continue
			partitions[partition].append(features[offset:offset + seconds])
			label_segs[partition].append((seconds, anomalous))
			offset += seconds

	# Concatenate per-partition. Missing partitions become zero-row arrays.
	n_feat = len(canonical_cols)
	def _stack(parts):
		return np.concatenate(parts, axis=0) if parts else np.empty((0, n_feat))
	train = _stack(partitions['train'])
	calib = _stack(partitions['calib'])
	test  = _stack(partitions['test'])
	valid = _stack(partitions['valid'])

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
	print(f"  Anomaly labels: {int(labels.sum() / n_feat_lbl) if labels.size else 0}/{labels.shape[0]} test rows, "
	      f"{int(valid_labels.sum() / n_feat_lbl) if valid_labels.size else 0}/{valid_labels.shape[0]} valid rows")
	print(f'  Feature columns ({len(canonical_cols)}): {canonical_cols}')

	for name, arr in [('train', train), ('test', test), ('calib', calib), ('valid', valid), ('labels', labels), ('valid_labels', valid_labels)]:
		np.save(os.path.join(folder, f'{name}.npy'), arr.astype('float64'))
	all_paths = [p for paths, _ in csv_groups for p in paths]
	print(f"Processed {len(all_paths)} CSV(s) as TOL -> {folder}/")
	print(f"  train.npy: {train.shape}, test.npy: {test.shape}, calib.npy: {calib.shape}, valid.npy: {valid.shape}, labels.npy: {labels.shape}")
