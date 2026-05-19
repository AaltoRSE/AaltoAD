import os
import TranAD
import argparse

from TranAD.constants import  DEFAULT_OUTPUT_FOLDER, DEFAULT_DATA_FOLDER

# dataset-specific implementations live under TranAD.preprocessing
datasets = ['synthetic', 'SMD', 'SWaT', 'SMAP', 'MSL', 'WADI', 'MSDS', 'UCR', 'MBA', 'NAB', 'TOL', 'SWaT_physical', 'SWaT_netflow', 'SWaT_payload_netflow']

wadi_drop = ['2_LS_001_AL', '2_LS_002_AL','2_P_001_STATUS','2_P_002_STATUS']


def load_data(dataset, csv_path=None, output_folder=DEFAULT_OUTPUT_FOLDER, data_folder=DEFAULT_DATA_FOLDER, anomaly_start_sec=480, anomaly_duration_sec=None):
	folder = os.path.join(output_folder, dataset)
	os.makedirs(folder, exist_ok=True)

	# If a CSV path was provided, ensure the file exists before proceeding
	if csv_path and not os.path.exists(csv_path):
		raise Exception(f'CSV file not found: {csv_path}')
	
	if dataset == 'TOL':
		TranAD.preprocessing.load_TOL(folder, top_k=100, csv_path=csv_path, data_folder=data_folder, extended_features=False, anomaly_start_sec=anomaly_start_sec, anomaly_duration_sec=anomaly_duration_sec)
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
	parser = argparse.ArgumentParser(description='Preprocess datasets for anomaly detection')
	parser.add_argument('datasets', nargs='+', help='One or more dataset types to preprocess (e.g. SMAP SMD SWaT)')
	parser.add_argument('--csv', type=str, help='Path to CSV file to preprocess (optional, only used for TOL)')
	parser.add_argument('--anomaly-start', type=int, default=480, help='Absolute second when anomaly begins (default: 480, i.e. minute 8)')
	parser.add_argument('--anomaly-duration', type=int, default=None, help='Anomaly duration in seconds (default: None = until end of data)')
	args = parser.parse_args()

	# Iterate over provided dataset names and process each
	for ds in args.datasets:
		if ds not in datasets:
			print(f"Skipping unknown dataset '{ds}'. Supported: {datasets}")
			continue
		load_data(ds, csv_path=args.csv, anomaly_start_sec=args.anomaly_start, anomaly_duration_sec=args.anomaly_duration)
