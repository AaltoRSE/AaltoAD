import os
import json
import numpy as np
import pandas as pd
from TranAD.constants import DEFAULT_DATA_FOLDER


def load_NAB(folder, data_folder=DEFAULT_DATA_FOLDER):
	dataset_folder = os.path.join(data_folder, 'NAB')
	file_list = os.listdir(dataset_folder)
	with open(dataset_folder + '/labels.json') as f:
		labeldict = json.load(f)
	for filename in file_list:
		if not filename.endswith('.csv'): continue
		df = pd.read_csv(dataset_folder+'/'+filename)
		vals = df.values[:,1]
		labels = np.zeros_like(vals, dtype=np.float64)
		for timestamp in labeldict['realKnownCause/'+filename]:
			tstamp = timestamp.replace('.000000', '')
			index = np.where(((df['timestamp'] == tstamp).values + 0) == 1)[0][0]
			labels[index-4:index+4] = 1
		min_temp, max_temp = np.min(vals), np.max(vals)
		vals = (vals - min_temp) / (max_temp - min_temp)
		train, test = vals.astype(float), vals.astype(float)
		train, test, labels = train.reshape(-1, 1), test.reshape(-1, 1), labels.reshape(-1, 1)
		fn = filename.replace('.csv', '')
		for file in ['train', 'test', 'labels']:
			np.save(os.path.join(folder, f'{fn}_{file}.npy'), eval(file))
