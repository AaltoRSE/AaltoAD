import os
import pandas as pd
import csv
from tqdm import tqdm
import warnings
import TranAD
from TranAD import models
from TranAD import constants
from TranAD import plotting
from TranAD import pot
from TranAD.utils import *
from TranAD.diagnosis import *
from TranAD.merlin import *
from torch.utils.data import DataLoader, TensorDataset
import torch
from torch import nn
from time import time
from pprint import pprint
# from beepy import beep
import importlib

# Suppress matplotlib font warnings
warnings.filterwarnings('ignore', message='.*findfont.*')


def convert_to_windows(data, model):
	"""Convert time series data into sliding windows for model input.
	
	Args:
		data (torch.Tensor): Input time series data with shape (num_samples, num_features).
		model: Model object that contains the n_window attribute specifying window size.
	
	Returns:
		torch.Tensor: Stacked windows with shape (num_samples, window_size, num_features) for TranAD/Attention
			models or (num_samples, window_size * num_features) for other models.
	"""
	windows = []; w_size = model.n_window
	for i, g in enumerate(data): 
		if i >= w_size: w = data[i-w_size:i]
		else: w = torch.cat([data[0].repeat(w_size-i, 1), data[0:i]])
		windows.append(w if 'TranAD' in args.model or 'Attention' in args.model else w.view(-1))
	return torch.stack(windows)

def load_dataset(dataset):
	"""Load pre-processed training, testing, and label data for a given dataset.
	
	Args:
		dataset (str): Name of the dataset (e.g., 'SMD', 'SMAP', 'MSL', 'UCR', 'NAB').
			Must have corresponding processed .npy files in the output folder.
	
	Returns:
		tuple: A tuple containing:
			- train_loader (DataLoader): PyTorch DataLoader for training data.
			- test_loader (DataLoader): PyTorch DataLoader for test data.
			- labels (np.ndarray): Ground truth anomaly labels with shape (num_test_samples, num_features).
	
	Raises:
		Exception: If processed data folder does not exist for the given dataset.
	"""
	folder = os.path.join(constants.output_folder, dataset)
	if not os.path.exists(folder):
		raise Exception('Processed Data not found.')
	loader = []
	for file in ['train', 'test', 'labels']:
		if dataset == 'SMD': file = 'machine-1-1_' + file
		if dataset == 'SMAP': file = 'P-1_' + file
		if dataset == 'MSL': file = 'C-1_' + file
		if dataset == 'UCR': file = '136_' + file
		if dataset == 'NAB': file = 'ec2_request_latency_system_failure_' + file
		loader.append(np.load(os.path.join(folder, f'{file}.npy')))
	# loader = [i[:, debug:debug+1] for i in loader]
	if args.less: loader[0] = cut_array(0.2, loader[0])
	train_loader = DataLoader(loader[0], batch_size=loader[0].shape[0])
	test_loader = DataLoader(loader[1], batch_size=loader[1].shape[0])
	labels = loader[2]
	return train_loader, test_loader, labels

def save_model(model, optimizer, scheduler, epoch, accuracy_list):
	"""Save model checkpoint including state dicts and training metadata.
	
	Args:
		model (torch.nn.Module): The neural network model to save.
		optimizer (torch.optim.Optimizer): The optimizer state to save (e.g., AdamW).
		scheduler (torch.optim.lr_scheduler): Learning rate scheduler state to save.
		epoch (int): Current epoch number for checkpoint tracking.
		accuracy_list (list): List of tuples containing (loss, learning_rate) for each epoch.
	
	Returns:
		None. Saves checkpoint to checkpoints/{model_name}_{dataset_name}/model.ckpt
	"""
	folder = f'checkpoints/{args.model}_{args.dataset}/'
	os.makedirs(folder, exist_ok=True)
	file_path = f'{folder}/model.ckpt'
	torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'accuracy_list': accuracy_list}, file_path)

def load_model(modelname, dims):
	"""Load or create a model with optimizer and scheduler, resuming from checkpoint if available.
	
	Args:
		modelname (str): Name of the model class to instantiate (e.g., 'TranAD', 'USAD', 'DAGMM').
		dims (int): Number of features/dimensions in the input data.
	
	Returns:
		tuple: A tuple containing:
			- model (torch.nn.Module): The loaded or newly created model in double precision.
			- optimizer (torch.optim.AdamW): AdamW optimizer configured for the model.
			- scheduler (torch.optim.lr_scheduler.StepLR): Learning rate scheduler (step every 5 epochs).
			- epoch (int): Starting epoch (-1 if new model, otherwise loaded epoch from checkpoint).
			- accuracy_list (list): List of training metrics from checkpoint or empty list for new model.
			
	Note:
		Attempts to load from checkpoints/{modelname}_{dataset}/model.ckpt if it exists,
		unless args.retrain is True and args.test is False.
	"""
	model_class = getattr(models, modelname)
	model = model_class(dims).double()
	optimizer = torch.optim.AdamW(model.parameters() , lr=model.lr, weight_decay=1e-5)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)
	fname = f'checkpoints/{args.model}_{args.dataset}/model.ckpt'
	if os.path.exists(fname) and (not args.retrain or args.test):
		print(f"{color.GREEN}Loading pre-trained model: {model.name}{color.ENDC}")
		try:
			checkpoint = torch.load(fname, weights_only=False)
		except TypeError:
			checkpoint = torch.load(fname)
		model.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
		epoch = checkpoint['epoch']
		accuracy_list = checkpoint['accuracy_list']
	else:
		print(f"{color.GREEN}Creating new model: {model.name}{color.ENDC}")
		epoch = -1; accuracy_list = []
	return model, optimizer, scheduler, epoch, accuracy_list


def append_benchmark_row(model_name, dataset_name, result_dict, bench_path=os.path.join('results', 'benchmarks.csv')):
	"""Append a single benchmark row to CSV, creating file/header if needed.

	Args:
		model_name (str), dataset_name (str), result_dict (dict), bench_path (str)
	"""
	try:
		os.makedirs(os.path.dirname(bench_path) or '.', exist_ok=True)
		write_header = (not os.path.exists(bench_path)) or os.path.getsize(bench_path) == 0
		with open(bench_path, 'a', newline='') as csvfile:
			writer = csv.writer(csvfile)
			if write_header:
				writer.writerow(['model', 'dataset', 'precision', 'recall', 'AUC', 'f1'])
			writer.writerow([model_name, dataset_name, result_dict.get('precision'), result_dict.get('recall'), result_dict.get('ROC/AUC'), result_dict.get('f1')])
	except Exception as e:
		print(f"Could not write benchmark CSV: {e}")

def backprop(epoch, model, data, dataO, optimizer, scheduler, training = True):
	"""Dispatcher for model-specific backprop/eval routines.

	This function delegates to smaller helper functions implemented below.
	"""
	feats = dataO.shape[1]
	if 'DAGMM' in model.name:
		return _backprop_dagmm(epoch, model, data, dataO, optimizer, scheduler, training, feats)
	if 'Attention' in model.name:
		return _backprop_attention(epoch, model, data, dataO, optimizer, scheduler, training, feats)
	if 'OmniAnomaly' in model.name:
		return _backprop_omni(epoch, model, data, dataO, optimizer, scheduler, training, feats)
	if 'USAD' in model.name:
		return _backprop_usad(epoch, model, data, dataO, optimizer, scheduler, training, feats)
	if model.name in ['GDN', 'MTAD_GAT', 'MSCRED', 'CAE_M']:
		return _backprop_simple(epoch, model, data, dataO, optimizer, scheduler, training, feats)
	if 'GAN' in model.name:
		return _backprop_gan(epoch, model, data, dataO, optimizer, scheduler, training, feats)
	if 'TranAD' in model.name:
		return _backprop_tranad(epoch, model, data, dataO, optimizer, scheduler, training, feats)
	return _backprop_default(epoch, model, data, dataO, optimizer, scheduler, training, feats)


def _backprop_dagmm(epoch, model, data, dataO, optimizer, scheduler, training, feats):
	l = nn.MSELoss(reduction = 'none')
	compute = models.ComputeLoss(model, 0.1, 0.005, 'cpu', model.n_gmm)
	n = epoch + 1; w_size = model.n_window
	l1s = []; l2s = []
	if training:
		for d in data:
			_, x_hat, z, gamma = model(d)
			l1, l2 = l(x_hat, d), l(gamma, d)
			l1s.append(torch.mean(l1).item()); l2s.append(torch.mean(l2).item())
			loss = torch.mean(l1) + torch.mean(l2)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
		scheduler.step()
		tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)},\tL2 = {np.mean(l2s)}')
		return np.mean(l1s)+np.mean(l2s), optimizer.param_groups[0]['lr']
	else:
		ae1s = []
		for d in data:
			_, x_hat, _, _ = model(d)
			ae1s.append(x_hat)
		ae1s = torch.stack(ae1s)
		y_pred = ae1s[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
		loss = l(ae1s, data)[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
		return loss.detach().numpy(), y_pred.detach().numpy()


def _backprop_attention(epoch, model, data, dataO, optimizer, scheduler, training, feats):
	l = nn.MSELoss(reduction = 'none')
	n = epoch + 1; w_size = model.n_window
	l1s = []
	if training:
		for d in data:
			ae, ats = model(d)
			l1 = l(ae, d)
			l1s.append(torch.mean(l1).item())
			loss = torch.mean(l1)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
		scheduler.step()
		tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)}')
		return np.mean(l1s), optimizer.param_groups[0]['lr']
	else:
		ae1s, y_pred = [], []
		for d in data:
			ae1, ats = model(d)
			y_pred.append(ae1[-1])
			ae1s.append(ae1)
		ae1s, y_pred = torch.stack(ae1s), torch.stack(y_pred)
		loss = torch.mean(l(ae1s, data), axis=1)
		return loss.detach().numpy(), y_pred.detach().numpy()


def _backprop_omni(epoch, model, data, dataO, optimizer, scheduler, training, feats):
	l = nn.MSELoss(reduction = 'none')
	if training:
		mses, klds = [], []
		for i, d in enumerate(data):
			y_pred, mu, logvar, hidden = model(d, hidden if i else None)
			MSE = l(y_pred, d)
			KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=0)
			loss = MSE + model.beta * KLD
			mses.append(torch.mean(MSE).item()); klds.append(model.beta * torch.mean(KLD).item())
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
		tqdm.write(f'Epoch {epoch},\tMSE = {np.mean(mses)},\tKLD = {np.mean(klds)}')
		scheduler.step()
		return loss.item(), optimizer.param_groups[0]['lr']
	else:
		y_preds = []
		for i, d in enumerate(data):
			y_pred, _, _, hidden = model(d, hidden if i else None)
			y_preds.append(y_pred)
		y_pred = torch.stack(y_preds)
		MSE = l(y_pred, data)
		return MSE.detach().numpy(), y_pred.detach().numpy()


def _backprop_usad(epoch, model, data, dataO, optimizer, scheduler, training, feats):
	l = nn.MSELoss(reduction = 'none')
	n = epoch + 1; w_size = model.n_window
	l1s, l2s = [], []
	if training:
		for d in data:
			ae1s, ae2s, ae2ae1s = model(d)
			l1 = (1 / n) * l(ae1s, d) + (1 - 1/n) * l(ae2ae1s, d)
			l2 = (1 / n) * l(ae2s, d) - (1 - 1/n) * l(ae2ae1s, d)
			l1s.append(torch.mean(l1).item()); l2s.append(torch.mean(l2).item())
			loss = torch.mean(l1 + l2)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
		scheduler.step()
		tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)},\tL2 = {np.mean(l2s)}')
		return np.mean(l1s)+np.mean(l2s), optimizer.param_groups[0]['lr']
	else:
		ae1s, ae2s, ae2ae1s = [], [], []
		for d in data:
			ae1, ae2, ae2ae1 = model(d)
			ae1s.append(ae1); ae2s.append(ae2); ae2ae1s.append(ae2ae1)
		ae1s, ae2s, ae2ae1s = torch.stack(ae1s), torch.stack(ae2s), torch.stack(ae2ae1s)
		y_pred = ae1s[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
		loss = 0.1 * l(ae1s, data) + 0.9 * l(ae2ae1s, data)
		loss = loss[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
		return loss.detach().numpy(), y_pred.detach().numpy()


def _backprop_simple(epoch, model, data, dataO, optimizer, scheduler, training, feats):
	l = nn.MSELoss(reduction = 'none')
	n = epoch + 1; w_size = model.n_window
	l1s = []
	if training:
		for i, d in enumerate(data):
			if 'MTAD_GAT' in model.name:
				x, h = model(d, h if i else None)
			else:
				x = model(d)
			loss = torch.mean(l(x, d))
			l1s.append(torch.mean(loss).item())
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
		tqdm.write(f'Epoch {epoch},\tMSE = {np.mean(l1s)}')
		return np.mean(l1s), optimizer.param_groups[0]['lr']
	else:
		xs = []
		for d in data:
			if 'MTAD_GAT' in model.name:
				x, h = model(d, None)
			else:
				x = model(d)
			xs.append(x)
		xs = torch.stack(xs)
		y_pred = xs[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
		loss = l(xs, data)
		loss = loss[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
		return loss.detach().numpy(), y_pred.detach().numpy()


def _backprop_gan(epoch, model, data, dataO, optimizer, scheduler, training, feats):
	l = nn.MSELoss(reduction = 'none')
	bcel = nn.BCELoss(reduction = 'mean')
	msel = nn.MSELoss(reduction = 'mean')
	real_label, fake_label = torch.tensor([0.9]), torch.tensor([0.1])
	real_label, fake_label = real_label.type(torch.DoubleTensor), fake_label.type(torch.DoubleTensor)
	n = epoch + 1; w_size = model.n_window
	mses, gls, dls = [], [], []
	if training:
		for d in data:
			model.discriminator.zero_grad()
			_, real, fake = model(d)
			dl = bcel(real, real_label) + bcel(fake, fake_label)
			dl.backward()
			model.generator.zero_grad()
			optimizer.step()
			z, _, fake = model(d)
			mse = msel(z, d)
			gl = bcel(fake, real_label)
			tl = gl + mse
			tl.backward()
			model.discriminator.zero_grad()
			optimizer.step()
			mses.append(mse.item()); gls.append(gl.item()); dls.append(dl.item())
		tqdm.write(f'Epoch {epoch},\tMSE = {np.mean(mses)},\tG = {np.mean(gls)},\tD = {np.mean(dls)}')
		return np.mean(gls)+np.mean(dls), optimizer.param_groups[0]['lr']
	else:
		outputs = []
		for d in data:
			z, _, _ = model(d)
			outputs.append(z)
		outputs = torch.stack(outputs)
		y_pred = outputs[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
		loss = l(outputs, data)
		loss = loss[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
		return loss.detach().numpy(), y_pred.detach().numpy()


def _backprop_tranad(epoch, model, data, dataO, optimizer, scheduler, training, feats):
	l = nn.MSELoss(reduction = 'none')
	data_x = torch.DoubleTensor(data); dataset = TensorDataset(data_x, data_x)
	bs = model.batch if training else len(data)
	dataloader = DataLoader(dataset, batch_size = bs)
	n = epoch + 1; w_size = model.n_window
	l1s, l2s = [], []
	if training:
		for d, _ in dataloader:
			local_bs = d.shape[0]
			window = d.permute(1, 0, 2)
			elem = window[-1, :, :].view(1, local_bs, feats)
			z = model(window, elem)
			l1 = l(z, elem) if not isinstance(z, tuple) else (1 / n) * l(z[0], elem) + (1 - 1/n) * l(z[1], elem)
			if isinstance(z, tuple): z = z[1]
			l1s.append(torch.mean(l1).item())
			loss = torch.mean(l1)
			optimizer.zero_grad()
			loss.backward(retain_graph=True)
			optimizer.step()
		scheduler.step()
		tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)}')
		return np.mean(l1s), optimizer.param_groups[0]['lr']
	else:
		for d, _ in dataloader:
			window = d.permute(1, 0, 2)
			elem = window[-1, :, :].view(1, bs, feats)
			z = model(window, elem)
			if isinstance(z, tuple): z = z[1]
		loss = l(z, elem)[0]
		return loss.detach().numpy(), z.detach().numpy()[0]


def _backprop_default(epoch, model, data, dataO, optimizer, scheduler, training, feats):
	l = nn.MSELoss(reduction = 'mean' if training else 'none')
	y_pred = model(data)
	loss = l(y_pred, data)
	if training:
		tqdm.write(f'Epoch {epoch},\tMSE = {loss}')
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		scheduler.step()
		return loss.item(), optimizer.param_groups[0]['lr']
	else:
		return loss.detach().numpy(), y_pred.detach().numpy()



def run_experiment():
	"""Run a single experiment using the current module-level `args`.

	Returns:
		dict: result dictionary produced at the end of the experiment.
	"""
	preds = []
	train_loader, test_loader, labels = load_dataset(args.dataset)
	if args.model in ['MERLIN']:
		# Call MERLIN's runner and append its result to benchmarks CSV
		res = run_merlin(test_loader, labels, args.dataset)
		append_benchmark_row(args.model, args.dataset, res)
		return res
	model, optimizer, scheduler, epoch, accuracy_list = load_model(args.model, labels.shape[1])

	## Prepare data
	trainD, testD = next(iter(train_loader)), next(iter(test_loader))
	trainO, testO = trainD, testD
	if model.name in ['Attention', 'DAGMM', 'USAD', 'MSCRED', 'CAE_M', 'GDN', 'MTAD_GAT', 'MAD_GAN', 'MERLIN'] or 'TranAD' in model.name:
		trainD, testD = convert_to_windows(trainD, model), convert_to_windows(testD, model)

	### Training phase
	if not args.test:
		print(f'{color.HEADER}Training {args.model} on {args.dataset}{color.ENDC}')
		num_epochs = 5; e = epoch + 1; start = time()
		for e in tqdm(list(range(epoch+1, epoch+num_epochs+1))):
			lossT, lr = backprop(e, model, trainD, trainO, optimizer, scheduler)
			accuracy_list.append((lossT, lr))
		print(color.BOLD+'Training time: '+"{:10.4f}".format(time()-start)+' s'+color.ENDC)
		save_model(model, optimizer, scheduler, e, accuracy_list)
		plot_accuracies(accuracy_list, f'{args.model}_{args.dataset}')

	### Testing phase
	torch.zero_grad = True
	model.eval()
	print(f'{color.HEADER}Testing {args.model} on {args.dataset}{color.ENDC}')
	loss, y_pred = backprop(0, model, testD, testO, optimizer, scheduler, training=False)

	### Plot curves
	if not args.test:
		if 'TranAD' in model.name: testO = torch.roll(testO, 1, 0)
		plotting.plotter(f'{args.model}_{args.dataset}', testO, y_pred, loss, labels)

	### Scores
	df = pd.DataFrame()
	lossT, _ = backprop(0, model, trainD, trainO, optimizer, scheduler, training=False)
	for i in range(loss.shape[1]):
		lt, l, ls = lossT[:, i], loss[:, i], labels[:, i]
		result, pred = pot.pot_eval(lt, l, ls); preds.append(pred)
		df = pd.concat([df, pd.DataFrame([result])], ignore_index=True)

	lossTfinal, lossFinal = np.mean(lossT, axis=1), np.mean(loss, axis=1)
	labelsFinal = (np.sum(labels, axis=1) >= 1) + 0
	result, _ = pot.pot_eval(lossTfinal, lossFinal, labelsFinal)
	result.update(hit_att(loss, labels))
	result.update(ndcg(loss, labels))
	print(df)
	pprint(result)
	# Append benchmark to CSV: model, dataset, precision, recall, AUC, f1
	append_benchmark_row(args.model, args.dataset, result)
	return result


def run_all(models_list=None, datasets_list=None):
	"""Run all models on all datasets and print a summary report.

	The function updates `TranAD.parser.args` and reloads dependent modules so
	per-dataset/model constants are applied.
	"""
	# discover datasets from processed output folder if not provided
	if datasets_list is None:
		folder = os.path.join(TranAD.folderconstants.output_folder)
		if not os.path.exists(folder):
			raise Exception(f'Processed data folder not found: {folder}')
		datasets_list = sorted([d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))])

	# discover model class names if not provided
	if models_list is None:
		importlib.reload(TranAD.models)
		models_list = [name for name in dir(TranAD.models) if name[0].isupper() and callable(getattr(TranAD.models, name))] + ["MERLIN"]

	# Load existing benchmark entries to optionally skip already-run experiments
	bench_file = os.path.join('results', 'benchmarks.csv')
	existing_runs = set()
	if os.path.exists(bench_file):
		try:
			with open(bench_file, 'r', newline='') as csvfile:
				reader = csv.DictReader(csvfile)
				for row in reader:
					m = row.get('model')
					d = row.get('dataset')
					if m and d:
						existing_runs.add((m, d))
		except Exception as e:
			print(f"Warning: could not read benchmark CSV: {e}")

	summary = {}
	for dataset in datasets_list:
		TranAD.parser.args.dataset = dataset
		# reflect new dataset in module-level args
		global args
		args = TranAD.parser.args
		# reload constants and models so they pick up new args
		importlib.reload(TranAD.constants)
		importlib.reload(TranAD.models)
		for modelname in models_list:
			TranAD.parser.args.model = modelname
			args = TranAD.parser.args
			# Skip if this run is already present in the benchmarks CSV and retrain not requested
			if (modelname, dataset) in existing_runs and not getattr(args, 'retrain', False):
				print(f"Skipping {modelname} on {dataset}: already in results/benchmarks.csv (use --retrain to override)")
				continue
			print(f'Running {modelname} on {dataset}')
			# reload models to pick any dataset-dependent hyperparams
			importlib.reload(TranAD.models)
			# run and collect result
			try:
				res = run_experiment()
			except Exception as e:
				res = {'error': str(e)}
			summary[(modelname, dataset)] = res

	# Print concise report
	# Build summary DataFrame
	rows = []
	for (m, d), r in summary.items():
		if isinstance(r, dict) and 'precision' in r:
			rows.append({'model': m, 'dataset': d, 'precision': r.get('precision'), 'recall': r.get('recall'), 'AUC': r.get('ROC/AUC'), 'F1': r.get('f1')})
		else:
			rows.append({'model': m, 'dataset': d, 'precision': None, 'recall': None, 'AUC': None, 'F1': None, 'error': str(r)})
	report = pd.DataFrame(rows)
	print('\n=== Summary Report ===')
	print(report.to_string(index=False))
	return summary, report


if __name__ == '__main__':
	# Parse command-line arguments at runtime (not on import)
	parse_arguments()
	args = TranAD.parser.args
	# If either model or dataset is 'ALL', run the full sweep
	if args.model == 'ALL' or args.dataset == 'ALL':
		models_list = None if args.model == 'ALL' else [args.model]
		datasets_list = None if args.dataset == 'ALL' else [args.dataset]
		run_all(models_list=models_list, datasets_list=datasets_list)
	else:
		run_experiment()
