import os
import TranAD
import argparse

from TranAD.constants import  DEFAULT_OUTPUT_FOLDER, DEFAULT_DATA_FOLDER

# dataset-specific implementations live under TranAD.preprocessing
datasets = ['synthetic', 'SMD', 'SWaT', 'SMAP', 'MSL', 'WADI', 'MSDS', 'UCR', 'MBA', 'NAB', 'TOL', 'SWaT_physical', 'SWaT_netflow', 'SWaT_payload_netflow', 'nettraffic']

wadi_drop = ['2_LS_001_AL', '2_LS_002_AL','2_P_001_STATUS','2_P_002_STATUS']


def parse_segment(spec):
	"""Parse a single segment spec 'TYPE[-]:SECONDS' into (partition, anomalous, seconds)."""
	if ':' not in spec:
		raise argparse.ArgumentTypeError(f"Segment must look like TYPE:SECONDS or TYPE-:SECONDS, got {spec!r}")
	type_part, sec_part = spec.split(':', 1)
	anomalous = type_part.endswith('-')
	partition = type_part[:-1] if anomalous else type_part
	if partition not in {'train', 'calib', 'test', 'valid', 'skip'}:
		raise argparse.ArgumentTypeError(
			f"Segment partition must be one of train/calib/test/valid/skip (with optional '-' suffix for anomalous), got {type_part!r}"
		)
	if anomalous and partition == 'skip':
		raise argparse.ArgumentTypeError(f"'skip' segments cannot be marked anomalous")
	try:
		seconds = int(sec_part)
	except ValueError:
		raise argparse.ArgumentTypeError(f"Segment seconds must be an integer, got {sec_part!r}")
	if seconds <= 0:
		raise argparse.ArgumentTypeError(f"Segment seconds must be positive, got {seconds}")
	return (partition, anomalous, seconds)


class _GroupBuilder(argparse.Action):
	"""Pair interleaved --csv and --segment flags into ordered (csvs, segments) groups.

	Each --csv opens a new group with the listed CSV paths. Each --segment appends
	its specs to the most recently opened group. The pairing is by occurrence
	order on the command line, not by index. Result is stored as
	`args.csv_groups: List[(List[str], List[(partition, anomalous, seconds)])]`.
	"""
	def __init__(self, option_strings, dest, role, **kwargs):
		self.role = role
		super().__init__(option_strings, dest, **kwargs)

	def __call__(self, parser, namespace, values, option_string=None):
		groups = getattr(namespace, 'csv_groups', None)
		if groups is None:
			groups = []
			setattr(namespace, 'csv_groups', groups)
		if self.role == 'csv':
			groups.append([list(values), []])
		else:  # 'segment'
			if not groups:
				raise argparse.ArgumentError(self, '--segment must follow a --csv')
			groups[-1][1].extend(values)


def load_data(
		dataset, csv_groups=None,
		output_folder=DEFAULT_OUTPUT_FOLDER,
		data_folder=DEFAULT_DATA_FOLDER,
        top_k=100, extended_features=False,
		interval=10,
	):
	folder = os.path.join(output_folder, dataset)
	os.makedirs(folder, exist_ok=True)

	# Validate all referenced CSV files exist before kicking off any processing.
	if csv_groups:
		for paths, _ in csv_groups:
			for p in paths:
				if not os.path.exists(p):
					raise Exception(f'CSV file not found: {p}')

	if dataset.startswith('TOL'):
		# Convert mutable inner lists to tuples for clarity, but load_TOL accepts either.
		groups_arg = [(paths, segs) for paths, segs in csv_groups] if csv_groups else None
		TranAD.preprocessing.load_TOL(
			folder, csv_groups=groups_arg, data_folder=data_folder,
			top_k=top_k,
			extended_features=extended_features,
		)
	elif dataset.startswith('nettraffic'):
		groups_arg = [(paths, segs) for paths, segs in csv_groups] if csv_groups else None
		TranAD.preprocessing.load_nettraffic(
			folder, csv_groups=groups_arg, data_folder=data_folder,
			top_k=top_k, interval=interval,
		)
	elif csv_groups:
		raise Exception(f'CSV processing not implemented for dataset type {dataset}. Currently only TOL is supported.')
	elif dataset == 'UCR':
		TranAD.preprocessing.load_UCR(folder, data_folder=data_folder)
	elif dataset == 'synthetic':
		TranAD.preprocessing.load_synthetic(folder, data_folder=data_folder)
	elif dataset == 'SMD':
		TranAD.preprocessing.load_SMD(folder)
	elif dataset == 'NAB':
		TranAD.preprocessing.load_NAB(folder, data_folder=data_folder)
	elif dataset == 'MSDS':
		TranAD.preprocessing.load_MSDS(folder, data_folder=data_folder)
	elif dataset == 'SWaT':
		TranAD.preprocessing.load_SWaT(folder, data_folder=data_folder)
	elif dataset in ['SMAP', 'MSL']:
		TranAD.preprocessing.load_SMAP_MSL(folder, dataset, data_folder=data_folder)
	elif dataset == 'WADI':
		TranAD.preprocessing.load_WADI(folder, data_folder=data_folder)
	elif dataset == 'MBA':
		TranAD.preprocessing.load_MBA(folder, data_folder=data_folder)
	elif dataset == 'SWaT_physical':
		TranAD.preprocessing.load_SWaT_physical(folder, data_folder=data_folder)
	elif dataset == 'SWaT_netflow':
		TranAD.preprocessing.load_SWaT_netflow(folder, data_folder=data_folder)
	elif dataset == 'SWaT_payload_netflow':
		TranAD.preprocessing.load_SWaT_payload_netflow(folder, data_folder=data_folder)
	else:
		raise Exception(f'Not Implemented. Check one of {datasets}')

if __name__ == '__main__':
	parser = argparse.ArgumentParser(
		description='Preprocess datasets for anomaly detection',
		epilog=(
			"TOL examples:\n"
			"  Single CSV: --csv data.csv --segment train:300 calib:60 test:120 test-:240 test:120 valid:120\n"
			"  Combined dataset (pooled baselines + multiple cases):\n"
			"    --csv baseline_a.csv baseline_b.csv --segment train:300 calib:60\n"
			"    --csv case_1.csv case_2.csv case_3.csv --segment test:120 test-:240 test:120\n"
			"  Each --csv opens a new group; the following --segment(s) apply per-CSV in that group.\n"
			"  Top-k IPs are computed once from all train segments across all groups."
		),
		formatter_class=argparse.RawDescriptionHelpFormatter,
	)
	parser.add_argument('datasets', nargs='+', help='One or more dataset types to preprocess (e.g. SMAP SMD SWaT). Names starting with TOL are dispatched to the TOL loader.')
	# TOL group flags: --csv and --segment are paired by occurrence order via _GroupBuilder.
	parser.add_argument(
		'--csv', nargs='+', action=_GroupBuilder, role='csv', dest='_csv',
		help='[TOL] One or more CSV paths starting a new group. Subsequent --segment flag(s) apply per-CSV in this group.',
	)
	parser.add_argument(
		'--segment', nargs='+', type=parse_segment, action=_GroupBuilder, role='segment', dest='_segment',
		help='[TOL] Segment spec(s) TYPE[-]:SECONDS for the current group. TYPE is train|calib|test|valid|skip; trailing "-" marks the segment as anomalous (e.g. test-:240). Use skip:N to discard N seconds (e.g. lead-in before an attack).',
	)
	parser.add_argument('--top-k', type=int, default=100, help='[TOL] Keep the top-k most frequent IPs from baseline; the rest collapse into other_internal/other_external (default: 100)')
	parser.add_argument('--interval', type=int, default=10, help='Number of seconds to aggregate to one row of data in nettraffic data.')
	parser.add_argument('--extended-features', action='store_true', help='[TOL] Include bytes/ports/entropy features in addition to src/dst counts')
	args = parser.parse_args()
	csv_groups = getattr(args, 'csv_groups', None)

	# Iterate over provided dataset names and process each
	for ds in args.datasets:
		if ds not in datasets and not ds.startswith('TOL'):
			print(f"Skipping unknown dataset '{ds}'. Supported: {datasets} (or any name starting with 'TOL')")
			continue
		load_data(
			ds,
			csv_groups=csv_groups,
			top_k=args.top_k,
			extended_features=args.extended_features,
			interval=args.interval,
		)
