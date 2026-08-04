import os
import numpy as np
import pandas as pd
from AaltoAD.preprocessing.utils import convertNumpy
from AaltoAD.constants import DEFAULT_DATA_FOLDER


def load_WADI(folder, data_folder=DEFAULT_DATA_FOLDER):
	dataset_folder = os.path.join(data_folder, 'WADI')
	ls = pd.read_csv(os.path.join(dataset_folder, 'WADI_attacklabels.csv'))
	train = pd.read_csv(os.path.join(dataset_folder, 'WADI_14days.csv'), skiprows=1000, nrows=2e5)
	test = pd.read_csv(os.path.join(dataset_folder, 'WADI_attackdata.csv'))
	train.dropna(how='all', inplace=True); test.dropna(how='all', inplace=True)
	train.fillna(0, inplace=True); test.fillna(0, inplace=True)
	test['Time'] = test['Time'].astype(str)
	test['Time'] = pd.to_datetime(test['Date'] + ' ' + test['Time'])
	labels = test.copy(deep = True)
	for i in test.columns.tolist()[3:]: labels[i] = 0
	for i in ['Start Time', 'End Time']:
		ls[i] = ls[i].astype(str)
		ls[i] = pd.to_datetime(ls['Date'] + ' ' + ls[i])
	for index, row in ls.iterrows():
		to_match = row['Affected'].split(', ')
		matched = []
		for i in test.columns.tolist()[3:]:
			for tm in to_match:
				if tm in i:
					matched.append(i); break
		st, et = str(row['Start Time']), str(row['End Time'])
		labels.loc[(labels['Time'] >= st) & (labels['Time'] <= et), matched] = 1
	train, test, labels = convertNumpy(train), convertNumpy(test), convertNumpy(labels)
	print(train.shape, test.shape, labels.shape)
	for file in ['train', 'test', 'labels']:
		np.save(os.path.join(folder, f'{file}.npy'), eval(file))
