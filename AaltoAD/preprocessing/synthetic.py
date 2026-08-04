import os
import pandas as pd
import numpy as np
from AaltoAD.constants import DEFAULT_DATA_FOLDER


def load_synthetic(folder, data_folder=DEFAULT_DATA_FOLDER):
	train_file = os.path.join(data_folder, 'synthetic', 'synthetic_data_with_anomaly-s-1.csv')
	test_labels = os.path.join(data_folder, 'synthetic', 'test_anomaly.csv')
	dat = pd.read_csv(train_file, header=None)
	split = 10000
	train = dat.values[:, :split].reshape(split, -1)
	train = (train - train.min()) / (train.max() - train.min())
	test = dat.values[:, split:].reshape(split, -1)
	test = (test - test.min()) / (test.max() - test.min())
	lab = pd.read_csv(test_labels, header=None)
	lab[0] -= split
	labels = np.zeros(test.shape)
	for i in range(lab.shape[0]):
		point = lab.values[i][0]
		labels[point-30:point+30, lab.values[i][1:]] = 1
	test += labels * np.random.normal(0.75, 0.1, test.shape)
	for file in ['train', 'test', 'labels']:
		np.save(os.path.join(folder, f'{file}.npy'), eval(file))
