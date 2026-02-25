import os
from TranAD.constants import DEFAULT_DATA_FOLDER
from TranAD.preprocessing.utils import load_and_save, load_and_save2


def load_SMD(folder):
	dataset_folder = os.path.join(DEFAULT_DATA_FOLDER, 'SMD')
	train_dir = os.path.join(dataset_folder, 'train')
	test_dir = os.path.join(dataset_folder, 'test')

	if not os.path.isdir(train_dir):
		raise FileNotFoundError(f'Expected SMD train directory not found: {train_dir}')

	file_list = os.listdir(train_dir)
	for filename in file_list:
		if filename.endswith('.txt'):
			name = filename[:-4]
			load_and_save('train', filename, name, dataset_folder)
			s = load_and_save('test', filename, name, dataset_folder)
			load_and_save2('labels', filename, name, dataset_folder, s)
