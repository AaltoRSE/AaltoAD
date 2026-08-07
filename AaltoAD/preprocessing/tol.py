import os
import pandas as pd
import numpy as np
import re

from AaltoAD.constants import DEFAULT_DATA_FOLDER
from AaltoAD.preprocessing.utils import normalize


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
		numeric_seconds = pd.Series((dt.view('int64') // 1_000_000_000), index=dt.index).astype('Int64')
	print(f"TT Timestamp parsed: original='{first}' -> parsed={numeric_seconds.iloc[0]}")
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


# Per-IP features expand to one column per IP key (sorted top_ips + 'Other').
PER_IP_FEATURES = {
	'sent_row_count', 'sent_bytes', 'sent_port_entropy', 'num_dsts',
	'received_row_count', 'received_frameln', 'received_port_entropy', 'num_srcs',
	'rw_ratio',
}

# Global (single-column) features.
GLOBAL_FEATURES = {'rows_per_window', 'empty_rows'}

# Full ordered registry of valid TOL feature names, used for argparse choices
# and validation.
TOL_FEATURES = [
	'sent_row_count', 'sent_bytes', 'sent_port_entropy', 'num_dsts',
	'received_row_count', 'received_frameln', 'received_port_entropy', 'num_srcs',
	'rw_ratio',
	'rows_per_window', 'empty_rows',
]

DEFAULT_FEATURES = [
	'sent_row_count', 'sent_bytes', 'sent_port_entropy',
	'received_row_count', 'received_frameln', 'received_port_entropy',
]


def _validate_features(features):
	"""Raise ValueError if any requested feature name is not in TOL_FEATURES."""
	unknown = [f for f in features if f not in TOL_FEATURES]
	if unknown:
		raise ValueError(f'Unknown TOL feature(s) {unknown}; must be one of {TOL_FEATURES}')


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
	"""Find src/dst IP columns in a TOL CSV, preferring IP columns over MAC columns."""
	ip_src_names = ('src_ip', 'ip.src', 'ip_src')
	ip_dst_names = ('dst_ip', 'ip.dst', 'ip_dst')
	src_col = next((c for c in df.columns if c.lower() in ip_src_names), None)
	dst_col = next((c for c in df.columns if c.lower() in ip_dst_names), None)
	if src_col is None or dst_col is None:
		for c in df.columns:
			cl = c.lower()
			if 'eth' in cl or 'mac' in cl:
				continue
			if src_col is None and 'src' in cl: src_col = c
			if dst_col is None and 'dst' in cl: dst_col = c
		if src_col is None or dst_col is None:
			for c in df.columns:
				if src_col is None and 'ip' in c.lower(): src_col = c
				if dst_col is None and 'ip' in c.lower(): dst_col = c
	return src_col, dst_col


def _load_and_prepare(csv_path):
	"""Read CSV, detect needed columns, parse timestamps. Returns a parsed-CSV record."""
	df = pd.read_csv(csv_path, on_bad_lines='warn')
	timestamp_col = find_timestamp_column(df)
	src_col, dst_col = _detect_ip_columns(df)
	bytes_col    = _detect_column(df, ['bytes', 'length', 'len', 'pkt_size', 'size', 'octets', 'framelen', 'frame.len'])
	src_port_col = _detect_column(df, ['src_port', 'sport', 'source_port', 'srcport', 'tcp.srcport', 'udp.srcport'])
	dst_port_col = _detect_column(df, ['dst_port', 'dport', 'destination_port', 'dstport', 'tcp.dstport', 'udp.dstport'])
	df['ts_sec'] = parse_timestamp_column(df[timestamp_col])
	df = df.dropna(subset=['ts_sec'])
	df['ts_sec'] = df['ts_sec'].astype(int)
	return {
		'path': csv_path, 'df': df,
		'src_col': src_col, 'dst_col': dst_col,
		'bytes_col': bytes_col, 'port_col': src_port_col, 'dst_port_col': dst_port_col,
	}


def _baseline_mask(df, segments, interval_col = 'ts_sec'):
	"""Mask selecting rows whose ts_sec falls in any train segment of the segment list."""
	ts_min = df[interval_col].min()
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
		mask |= (df[interval_col] >= lo) & (df[interval_col] < hi)
	return mask


def _aggregate_csv(parsed, top_ips, features, window):
	"""Aggregate one parsed CSV into a per-window features dataframe.

	Uses the supplied `top_ips` so the column space is consistent across CSVs
	that share a baseline. Non-top-k IPs collapse into a single 'Other' bucket.
	The aggregation window is `window` seconds, indexed relative to this CSV's
	own data start (`ts_win = (ts_sec - ts_min) // window`).
	"""
	df = parsed['df'].copy()
	src_col, dst_col = parsed['src_col'], parsed['dst_col']
	bytes_col = parsed['bytes_col']
	src_port_col = parsed['port_col']
	dst_port_col = parsed.get('dst_port_col')
	received_port_col = dst_port_col or src_port_col

	# Compute the empty-rows mask on the ORIGINAL IP columns, before mapping
	# them through the top-k classifier.
	empty_mask = (
		(df[src_col].isna() | (df[src_col].astype(str).str.strip() == '')) &
		(df[dst_col].isna() | (df[dst_col].astype(str).str.strip() == ''))
	)

	ts_min = df['ts_sec'].min()
	df['ts_win'] = (df['ts_sec'] - ts_min) // window

	def _classify(ip_str):
		ip_str = str(ip_str).strip()
		return ip_str if ip_str in top_ips else 'Other'

	df[src_col] = df[src_col].astype(str).str.strip().map(_classify)
	df[dst_col] = df[dst_col].astype(str).str.strip().map(_classify)

	# rw_ratio needs the underlying sent_bytes/received_frameln aggregations
	# even if they are not themselves requested.
	need_sent_bytes = 'sent_bytes' in features or 'rw_ratio' in features
	need_received_frameln = 'received_frameln' in features or 'rw_ratio' in features

	out_grp = df.groupby(['ts_win', src_col])
	out_cols = {}
	if 'sent_row_count' in features:
		out_cols['sent_row_count'] = out_grp.size()
	if need_sent_bytes and bytes_col:
		out_cols['sent_bytes'] = out_grp[bytes_col].sum()
	if 'sent_port_entropy' in features and src_port_col:
		out_cols['sent_port_entropy'] = out_grp[src_port_col].apply(_shannon_entropy)
	if 'num_dsts' in features:
		out_cols['num_dsts'] = out_grp[dst_col].nunique()

	in_grp = df.groupby(['ts_win', dst_col])
	in_cols = {}
	if 'received_row_count' in features:
		in_cols['received_row_count'] = in_grp.size()
	if need_received_frameln and bytes_col:
		in_cols['received_frameln'] = in_grp[bytes_col].sum()
	if 'received_port_entropy' in features and received_port_col:
		in_cols['received_port_entropy'] = in_grp[received_port_col].apply(_shannon_entropy)
	if 'num_srcs' in features:
		in_cols['num_srcs'] = in_grp[src_col].nunique()

	max_win = df['ts_win'].max()
	full_idx = np.arange(0, max_win + 1)
	features_df = pd.DataFrame(index=pd.Index(full_idx, name='ts_win'))

	if out_cols:
		out_agg = pd.DataFrame(out_cols)
		out_wide = out_agg.unstack(level=src_col, fill_value=0)
		out_wide.columns = [f'{ip}_{metric}' for metric, ip in out_wide.columns]
		features_df = features_df.join(out_wide.reindex(full_idx, fill_value=0), how='left')

	if in_cols:
		in_agg = pd.DataFrame(in_cols)
		in_wide = in_agg.unstack(level=dst_col, fill_value=0)
		in_wide.columns = [f'{ip}_{metric}' for metric, ip in in_wide.columns]
		features_df = features_df.join(in_wide.reindex(full_idx, fill_value=0), how='left')

	features_df = features_df.fillna(0)

	if 'rows_per_window' in features:
		rows_per_window = df.groupby('ts_win').size().reindex(full_idx, fill_value=0)
		features_df['rows_per_window'] = rows_per_window

	if 'empty_rows' in features:
		empty_rows = df[empty_mask].groupby('ts_win').size().reindex(full_idx, fill_value=0)
		features_df['empty_rows'] = empty_rows

	if 'rw_ratio' in features and bytes_col:
		for ip in list(top_ips) + ['Other']:
			recv_col = f'{ip}_received_frameln'
			sent_col = f'{ip}_sent_bytes'
			if recv_col in features_df.columns and sent_col in features_df.columns:
				features_df[f'{ip}_rw_ratio'] = features_df[recv_col] / (features_df[recv_col] + features_df[sent_col] + 1e-10)

	features_df.index.name = 'ts_win'
	return features_df


def _filter_available_features(features, has_bytes, has_src_ports, has_recv_ports):
	"""Drop requested features whose required column is absent from every CSV, warning."""
	requires = {
		'sent_bytes': has_bytes,
		'received_frameln': has_bytes,
		'rw_ratio': has_bytes,
		'sent_port_entropy': has_src_ports,
		'received_port_entropy': has_recv_ports,
	}
	selected = []
	for f in features:
		if f in requires and not requires[f]:
			print(f"TOL: warning: feature '{f}' requires a column not present in any CSV; dropping it.")
			continue
		selected.append(f)
	return selected


def _canonical_columns(top_ips, features):
	"""Build the canonical column list shared across all CSVs in a combined dataset."""
	ip_keys = sorted(top_ips) + ['Other']
	cols = []
	for f in features:
		if f in PER_IP_FEATURES:
			for ip in ip_keys:
				cols.append(f'{ip}_{f}')
		else:
			cols.append(f)
	return cols


def load_TOL(folder, csv_groups=None, csv_path=None, segments=None,
             data_folder=DEFAULT_DATA_FOLDER, top_k=10, features=None, window=1):
	"""Load and preprocess TOL dataset (network traffic aggregated by timestamp).

	Two calling modes:

	- Single-CSV (the common case): pass `csv_path` and (optionally) `segments`.
	  One CSV is aggregated, sliced into partitions, and saved.
	- Combined / multi-CSV: pass `csv_groups`, an ordered list of
	  `(csv_paths, segments)` tuples. Each CSV in each group is aggregated
	  independently, sliced by its group's segments, and per-partition outputs
	  are concatenated across all CSVs of all groups. Top-k IPs are computed
	  once from baseline (train) rows across ALL CSVs in ALL groups so the
	  feature column space is consistent.

	Partitions are 'train', 'calib', 'test', 'valid'. Within each group's
	segment list, each partition must be contiguous. Labels are produced for
	'test' and 'valid' partitions from the anomalous flags.

	`features` selects which feature columns to compute (see `TOL_FEATURES`
	for the full registry); defaults to `DEFAULT_FEATURES` when None.
	`window` is the aggregation window in seconds (default 1); every
	segment's seconds must be evenly divisible by `window`.
	"""
	if features is None:
		features = list(DEFAULT_FEATURES)
	_validate_features(features)

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
		for partition, anomalous, seconds in segs:
			if seconds % window != 0:
				raise ValueError(
					f"Segment ({partition}, {seconds}s) is not divisible by window ({window}s)"
				)

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

	if not all_baseline_ips:
		raise ValueError('No train segments found across any CSV — cannot determine baseline IP set')

	all_ips_concat = pd.concat(all_baseline_ips)
	all_ips_concat = all_ips_concat[all_ips_concat != '']
	counts = all_ips_concat.value_counts()
	if counts.empty:
		raise ValueError('No IP addresses found in baseline source/destination columns')
	top_ips = set(counts.head(top_k).index)
	total_baseline_rows = sum(len(s) for s in all_baseline_ips) // 2  # we appended src + dst
	print(f'TOL: unique IPs across all baselines: {len(counts)} '
	      f'(total baseline rows aggregated: {total_baseline_rows})')
	print(f'TOL: selecting top_{top_k} IPs from combined baseline (keeps {len(top_ips)})')

	# Determine canonical column set. has_bytes / has_*_ports are True if ANY CSV has them.
	has_bytes = any(rec['bytes_col'] for rec, _ in parsed_records)
	has_src_ports = any(rec['port_col'] for rec, _ in parsed_records)
	has_recv_ports = any(rec.get('dst_port_col') or rec['port_col'] for rec, _ in parsed_records)
	features = _filter_available_features(features, has_bytes, has_src_ports, has_recv_ports)
	canonical_cols = _canonical_columns(top_ips, features)
	print(f'TOL: features={features}, window={window}s')

	# Pass 2: aggregate each CSV, reindex to canonical columns, slice, accumulate.
	partitions = {'train': [], 'calib': [], 'test': [], 'valid': []}
	label_segs = {'train': [], 'calib': [], 'test': [], 'valid': []}
	for parsed, segs in parsed_records:
		features_df = _aggregate_csv(parsed, top_ips, features, window)
		features_df = features_df.reindex(columns=canonical_cols, fill_value=0)
		feat_arr = features_df.values.astype(float)
		total_needed = sum(seconds // window for _, _, seconds in segs)
		if feat_arr.shape[0] < total_needed:
			raise ValueError(
				f"CSV {parsed['path']}: segments require {total_needed} rows (window={window}s) but only "
				f"{feat_arr.shape[0]} available after aggregation"
			)
		offset = 0
		for partition, anomalous, seconds in segs:
			rows = seconds // window
			if partition == 'skip':
				# Consume rows without writing them to any partition.
				offset += rows
				continue
			partitions[partition].append(feat_arr[offset:offset + rows])
			label_segs[partition].append((rows, anomalous))
			offset += rows

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

	# Zero out columns that are constant in the training (baseline) data across
	# every partition. Without this, the eps-regularised divisor in normalize()
	# amplifies any calib/test variation in such columns by 1/eps (~10000x),
	# drowning out signal from the well-behaved features.
	keep = train.max(axis=0) != train.min(axis=0)
	if not keep.all():
		constant_cols = [c for c, k in zip(canonical_cols, keep) if not k]
		print(f'  zeroing {len(constant_cols)} baseline-constant columns: {constant_cols}')
		for part in (train, calib, test, valid):
			if part.size:
				part[:, ~keep] = 0.0

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
		for rows, anomalous in label_segs.get(name, []):
			if anomalous:
				out[row:row + rows] = 1
			row += rows
		return out

	labels = _build_labels(test, 'test')
	valid_labels = _build_labels(valid, 'valid')
	n_feat_lbl = max(labels.shape[1] if labels.size else 1, 1)
	print(f"  Anomaly labels: {int(labels.sum() / n_feat_lbl) if labels.size else 0}/{labels.shape[0]} test rows, "
	      f"{int(valid_labels.sum() / n_feat_lbl) if valid_labels.size else 0}/{valid_labels.shape[0]} valid rows")
	print(f'  Feature columns ({len(canonical_cols)}): {canonical_cols}')

	for name, arr in [('train', train), ('test', test), ('calib', calib), ('valid', valid), ('labels', labels), ('valid_labels', valid_labels)]:
		np.save(os.path.join(folder, f'{name}.npy'), np.ascontiguousarray(arr.astype('float64')))
	all_paths = [p for paths, _ in csv_groups for p in paths]
	print(f"Processed {len(all_paths)} CSV(s) as TOL -> {folder}/")
	print(f"  train.npy: {train.shape}, test.npy: {test.shape}, calib.npy: {calib.shape}, valid.npy: {valid.shape}, labels.npy: {labels.shape}")
