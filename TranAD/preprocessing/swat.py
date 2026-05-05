import os
import numpy as np
import pandas as pd
from TranAD.preprocessing.utils import normalize
from TranAD.constants import DEFAULT_DATA_FOLDER


def load_SWaT(folder, data_folder=DEFAULT_DATA_FOLDER):
	dataset_folder = os.path.join(data_folder, 'SWaT')
	file = os.path.join(dataset_folder, 'series.json')
	df_train = pd.read_json(file, lines=True)[['val']][3000:6000]
	df_test  = pd.read_json(file, lines=True)[['val']][7000:12000]
	train, min_a, max_a = normalize(df_train.values)
	test, _, _ = normalize(df_test.values, min_a, max_a)
	labels = pd.read_json(file, lines=True)[['noti']][7000:12000] + 0
	for file in ['train', 'test', 'labels']:
		np.save(os.path.join(folder, f'{file}.npy'), eval(file))
