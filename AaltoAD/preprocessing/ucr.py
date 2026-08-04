import os
import numpy as np
import pandas as pd
from AaltoAD.constants import DEFAULT_DATA_FOLDER


def load_UCR(folder, data_folder=DEFAULT_DATA_FOLDER):
	dataset_folder = os.path.join(data_folder, 'UCR')
	file_list = os.listdir(dataset_folder)
	for filename in file_list:
		if not filename.endswith('.txt'):
			continue
		vals = filename.split('.')[0].split('_')
		dnum, vals = int(vals[0]), vals[-3:]
		vals = [int(i) for i in vals]
		temp = np.genfromtxt(os.path.join(dataset_folder, filename),
				dtype=np.float64,
				delimiter=',')
		min_temp, max_temp = np.min(temp), np.max(temp)
		temp = (temp - min_temp) / (max_temp - min_temp)
		train, test = temp[:vals[0]], temp[vals[0]:]
		labels = np.zeros_like(test)
		labels[vals[1]-vals[0]:vals[2]-vals[0]] = 1
		train, test, labels = train.reshape(-1, 1), test.reshape(-1, 1), labels.reshape(-1, 1)
		for file in ['train', 'test', 'labels']:
			np.save(os.path.join(folder, f'{dnum}_{file}.npy'), eval(file))
