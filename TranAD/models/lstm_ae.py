import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from .base import BaseModel


class LSTM_AE(BaseModel):
	def __init__(self, feats, n_hidden=64, n_layers=1, sequence_length=30,
	             learning_rate=1e-3, epochs=10, batch_size=256, weight_decay=1e-5):
		super().__init__(feats, learning_rate=learning_rate, epochs=epochs, weight_decay=weight_decay)
		self.name = 'LSTM_AE'
		self.n_hidden = n_hidden
		self.n_layers = n_layers
		self.n_window = sequence_length
		self.flat_window = False
		self.batch_size = batch_size
		self.lstm = nn.LSTM(
			input_size=feats,
			hidden_size=n_hidden,
			num_layers=n_layers,
			batch_first=True,
		)
		self.output = nn.Linear(n_hidden, feats)
		self.lstm = self.lstm.double()
		self.output = self.output.double()

	def forward(self, x):
		encoded, _ = self.lstm(x)
		return self.output(encoded)

	def train_step(self, epoch, data, optimizer, scheduler, feats):
		data_x = torch.as_tensor(data, dtype=torch.float64)
		if data_x.ndim != 3:
			print(data_x.shape)
			raise ValueError("LSTM_AE expects windowed input [N, W, F]")

		dataset = TensorDataset(data_x, data_x)
		dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)
		device = next(self.parameters()).device

		self.train()
		criterion = nn.MSELoss()
		batch_losses = []
		for xb, yb in dataloader:
			xb = xb.to(device)
			yb = yb.to(device)
			optimizer.zero_grad(set_to_none=True)
			pred = self(xb)
			loss = criterion(pred, yb)
			loss.backward()
			optimizer.step()
			batch_losses.append(loss.item())

		scheduler.step()
		mean_loss = float(np.mean(batch_losses))
		tqdm.write(f'Epoch {epoch},\tMSE = {mean_loss}')
		return mean_loss, optimizer.param_groups[0]['lr']

	def eval_step(self, epoch, data, optimizer, scheduler, feats):
		data_x = torch.as_tensor(data, dtype=torch.float64)
		if data_x.ndim != 3:
			print(data_x.shape)
			raise ValueError("LSTM_AE expects windowed input [N, W, F]")

		dataset = TensorDataset(data_x, data_x)
		bs = len(dataset)
		dataloader = DataLoader(dataset, batch_size=bs, shuffle=False, drop_last=False)
		device = next(self.parameters()).device

		self.eval()
		criterion = nn.MSELoss(reduction='none')
		all_loss = []
		all_pred = []
		with torch.no_grad():
			for xb, yb in dataloader:
				xb = xb.to(device)
				yb = yb.to(device)
				pred = self(xb)
				loss = criterion(pred, yb)
				all_loss.append(loss.cpu())
				all_pred.append(pred.cpu())

		pred = torch.cat(all_pred, dim=0)    # [N, W, F]
		losses = torch.cat(all_loss, dim=0)  # [N, W, F]

		losses_np = losses.numpy()
		N, W, F = losses_np.shape
		starts = np.arange(N) - W + 1
		positions = starts[:, None] + np.arange(W)[None, :]   # [N, W]
		mask = (positions >= 0) & (positions < N)

		valid_positions = positions[mask]
		valid_losses    = losses_np[mask]                     # [count, F]

		score_sum   = np.zeros((N, F), dtype=np.float64)
		score_count = np.zeros(N,       dtype=np.float64)
		np.add.at(score_sum,   valid_positions, valid_losses)
		np.add.at(score_count, valid_positions, 1)
		losses_per_t = score_sum / np.maximum(score_count, 1)[:, None]   # [N, F]

		pred = pred[:, -1, :]                                 # [N, F] - kept for plotting only
		return losses_per_t, pred.detach().numpy()
