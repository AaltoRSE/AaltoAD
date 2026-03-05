import os
import TranAD
import argparse

from TranAD.constants import  DEFAULT_OUTPUT_FOLDER, DEFAULT_DATA_FOLDER

# dataset-specific implementations live under TranAD.preprocessing
datasets = ['synthetic', 'SMD', 'SWaT', 'SMAP', 'MSL', 'WADI', 'MSDS', 'UCR', 'MBA', 'NAB', 'TOL']

wadi_drop = ['2_LS_001_AL', '2_LS_002_AL','2_P_001_STATUS','2_P_002_STATUS']


def load_data(dataset, csv_path=None, output_folder=DEFAULT_OUTPUT_FOLDER, data_folder=DEFAULT_DATA_FOLDER):
	folder = os.path.join(output_folder, dataset)
	os.makedirs(folder, exist_ok=True)

	# If a CSV path was provided, ensure the file exists before proceeding
	if csv_path and not os.path.exists(csv_path):
		raise Exception(f'CSV file not found: {csv_path}')
	
	if dataset == 'TOL':
		TranAD.preprocessing.load_TOL(folder, top_k=100, csv_path=csv_path, data_folder=data_folder, extended_features=False)
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