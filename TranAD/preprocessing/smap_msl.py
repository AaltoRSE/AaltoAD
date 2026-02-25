import os
import numpy as np
import pandas as pd
from TranAD.constants import DEFAULT_DATA_FOLDER
from TranAD.preprocessing.utils import normalize3


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
