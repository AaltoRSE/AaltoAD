import os
import numpy as np
import pandas as pd
from TranAD.constants import DEFAULT_DATA_FOLDER
from TranAD.preprocessing.utils import normalize


def load_MBA(folder, data_folder=DEFAULT_DATA_FOLDER):
	dataset_folder = os.path.join(data_folder, 'MBA')
	ls = pd.read_excel(os.path.join(dataset_folder, 'labels.xlsx'))
	train = pd.read_excel(os.path.join(dataset_folder, 'train.xlsx'))
	test = pd.read_excel(os.path.join(dataset_folder, 'test.xlsx'))
	train, test = train.values[1:,1:].astype(float), test.values[1:,1:].astype(float)
	train, min_a, max_a = normalize(train)
	test, _, _ = normalize(test, min_a, max_a)
	ls = ls.values[:,1].astype(int)
	labels = np.zeros_like(test)
	for i in range(-20, 20):
		labels[ls + i, :] = 1
	for file in ['train', 'test', 'labels']:
		np.save(os.path.join(folder, f'{file}.npy'), eval(file))
