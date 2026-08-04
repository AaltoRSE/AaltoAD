import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from .base import BaseModel


## Separate LSTM for each variable
class LSTM_Univariate(BaseModel):
	def __init__(self, feats, n_hidden=1, n_layers=1, learning_rate=0.002, batch_size=512, epochs=5, weight_decay=1e-5):
		super().__init__(feats, learning_rate=learning_rate, epochs=epochs, weight_decay=weight_decay)
		self.name = 'LSTM_Univariate'
		self.n_hidden = n_hidden
		self.n_layers = n_layers
		self.batch_size = batch_size
		self.lstm = nn.ModuleList([nn.LSTM(1, self.n_hidden, self.n_layers) for i in range(feats)])
		self.lstm = self.lstm.double()

	def forward(self, x):
		T = x.shape[0]
		x = x.to(torch.float64)
		outputs = []
		for j in range(self.n_feats):
			seq = x[:, j].view(T, 1, 1)  # (seq_len=T, batch=1, input=1)
			h0 = torch.rand(self.n_layers, 1, self.n_hidden, dtype=torch.float64)
			c0 = torch.randn(self.n_layers, 1, self.n_hidden, dtype=torch.float64)
			out, _ = self.lstm[j](seq, (h0, c0))
			outputs.append(2 * out[:, 0, -1])  # (T,) — last hidden unit, *2 matches original scaling
		return torch.stack(outputs, dim=1)  # (T, F)

	def train_step(self, epoch, data, optimizer, scheduler, feats):
		l = nn.MSELoss(reduction='mean')
		total = 0.0
		n_samples = 0
		for b in range(0, data.shape[0], self.batch_size):
			batch_data = data[b:b+self.batch_size]
			batch_pred = self(batch_data)
			loss = l(batch_pred, batch_data)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			total += loss.item() * batch_data.shape[0]
			n_samples += batch_data.shape[0]
		scheduler.step()
		mean_loss = total / max(n_samples, 1)
		tqdm.write(f'Epoch {epoch},\tMSE = {mean_loss}')
		return mean_loss, optimizer.param_groups[0]['lr']

	def eval_step(self, epoch, data, optimizer, scheduler, feats):
		l = nn.MSELoss(reduction='none')
		y_pred = self(data)
		loss = l(y_pred, data)
		return loss.detach().numpy(), y_pred.detach().numpy()
