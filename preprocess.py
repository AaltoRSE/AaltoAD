import os
import TranAD
import argparse

from TranAD.constants import  DEFAULT_OUTPUT_FOLDER, DEFAULT_DATA_FOLDER

# dataset-specific implementations live under TranAD.preprocessing
datasets = ['synthetic', 'SMD', 'SWaT', 'SMAP', 'MSL', 'WADI', 'MSDS', 'UCR', 'MBA', 'NAB', 'TOL', 'SWaT_physical', 'SWaT_netflow', 'SWaT_payload_netflow']

wadi_drop = ['2_LS_001_AL', '2_LS_002_AL','2_P_001_STATUS','2_P_002_STATUS']


def parse_segment(spec):
	"""Parse a single segment spec 'TYPE[-]:SECONDS' into (partition, anomalous, seconds)."""
	if ':' not in spec:
		raise argparse.ArgumentTypeError(f"Segment must look like TYPE:SECONDS or TYPE-:SECONDS, got {spec!r}")
	type_part, sec_part = spec.split(':', 1)
	anomalous = type_part.endswith('-')
	partition = type_part[:-1] if anomalous else type_part
	if partition not in {'train', 'calib', 'test', 'valid'}:
		raise argparse.ArgumentTypeError(
			f"Segment partition must be one of train/calib/test/valid (with optional '-' suffix for anomalous), got {type_part!r}"
		)
	try:
		seconds = int(sec_part)
	except ValueError:
		raise argparse.ArgumentTypeError(f"Segment seconds must be an integer, got {sec_part!r}")
	if seconds <= 0:
		raise argparse.ArgumentTypeError(f"Segment seconds must be positive, got {seconds}")
	return (partition, anomalous, seconds)


def load_data(dataset, csv_path=None, output_folder=DEFAULT_OUTPUT_FOLDER, data_folder=DEFAULT_DATA_FOLDER,
              top_k=100, segments=None, extended_features=False):
	folder = os.path.join(output_folder, dataset)
	os.makedirs(folder, exist_ok=True)

	# If a CSV path was provided, ensure the file exists before proceeding
	if csv_path and not os.path.exists(csv_path):
		raise Exception(f'CSV file not found: {csv_path}')

	if dataset.startswith('TOL'):
		TranAD.preprocessing.load_TOL(
			folder, csv_path=csv_path, data_folder=data_folder,
			top_k=top_k,
			segments=segments,
			extended_features=extended_features,
		)
	elif csv_path:
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
		epilog="TOL segment example: --segment train:300 --segment calib:60 --segment test:120 --segment test-:240 --segment test:120 --segment valid:120",
	)
	parser.add_argument('datasets', nargs='+', help='One or more dataset types to preprocess (e.g. SMAP SMD SWaT). Names starting with TOL are dispatched to the TOL loader.')
	parser.add_argument('--csv', type=str, help='Path to CSV file to preprocess (only used for TOL)')
	# TOL-specific knobs (ignored by other loaders)
	parser.add_argument('--top-k', type=int, default=100, help='[TOL] Keep the top-k most frequent IPs; the rest collapse into other_internal/other_external (default: 100)')
	parser.add_argument('--extended-features', action='store_true', help='[TOL] Include bytes/ports/entropy features in addition to src/dst counts')
	parser.add_argument(
		'--segment', dest='segments', action='append', type=parse_segment, default=None,
		help='[TOL] Ordered segment spec TYPE[-]:SECONDS. TYPE is train|calib|test|valid; trailing "-" marks the segment as anomalous (e.g. test-:240). Repeat in order. If omitted, a default 300/60/120+240+120/120 layout is used.',
	)
	args = parser.parse_args()

	# Iterate over provided dataset names and process each
	for ds in args.datasets:
		if ds not in datasets and not ds.startswith('TOL'):
			print(f"Skipping unknown dataset '{ds}'. Supported: {datasets} (or any name starting with 'TOL')")
			continue
		load_data(
			ds,
			csv_path=args.csv,
			top_k=args.top_k,
			segments=args.segments,
			extended_features=args.extended_features,
		)
