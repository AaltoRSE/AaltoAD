import os
import numpy as np
import pandas as pd
from TranAD.constants import DEFAULT_DATA_FOLDER
from TranAD.preprocessing.utils import normalize3


def load_MSDS(folder, data_folder=DEFAULT_DATA_FOLDER):
	dataset_folder = os.path.join(data_folder, 'MSDS')
	df_train = pd.read_csv(os.path.join(dataset_folder, 'train.csv'))
	df_test  = pd.read_csv(os.path.join(dataset_folder, 'test.csv'))
	df_train, df_test = df_train.values[::5, 1:], df_test.values[::5, 1:]
	_, min_a, max_a = normalize3(np.concatenate((df_train, df_test), axis=0))
	train, _, _ = normalize3(df_train, min_a, max_a)
	test, _, _ = normalize3(df_test, min_a, max_a)
	labels = pd.read_csv(os.path.join(dataset_folder, 'labels.csv'))
	labels = labels.values[::1, 1:]
	for file in ['train', 'test', 'labels']:
		np.save(os.path.join(folder, f'{file}.npy'), eval(file).astype('float64'))
