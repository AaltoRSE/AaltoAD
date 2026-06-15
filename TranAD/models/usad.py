import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from .base import BaseModel


## USAD Model (KDD 20)
class USAD(BaseModel):
	def __init__(self, feats, n_hidden=16, n_latent=5, n_window=5, learning_rate=0.0001, epochs=5, weight_decay=1e-5):
		super().__init__(feats, learning_rate=learning_rate, epochs=epochs, weight_decay=weight_decay)
		self.name = 'USAD'
		self.n_hidden = n_hidden
		self.n_latent = n_latent
		self.n_window = n_window
		self.flat_window = True
		self.n = self.n_feats * self.n_window
		self.encoder = nn.Sequential(
			nn.Flatten(),
			nn.Linear(self.n, self.n_hidden), nn.ReLU(True),
			nn.Linear(self.n_hidden, self.n_hidden), nn.ReLU(True),
			nn.Linear(self.n_hidden, self.n_latent), nn.ReLU(True),
		)
		self.decoder1 = nn.Sequential(
			nn.Linear(self.n_latent, self.n_hidden), nn.ReLU(True),
			nn.Linear(self.n_hidden, self.n_hidden), nn.ReLU(True),
			nn.Linear(self.n_hidden, self.n), nn.Sigmoid(),
		)
		self.decoder2 = nn.Sequential(
			nn.Linear(self.n_latent, self.n_hidden), nn.ReLU(True),
			nn.Linear(self.n_hidden, self.n_hidden), nn.ReLU(True),
			nn.Linear(self.n_hidden, self.n), nn.Sigmoid(),
		)
		self.encoder = self.encoder.double()
		self.decoder1 = self.decoder1.double()
		self.decoder2 = self.decoder2.double()

	def forward(self, g):
		z = self.encoder(g.view(1,-1))
		ae1 = self.decoder1(z)
		ae2 = self.decoder2(z)
		ae2ae1 = self.decoder2(self.encoder(ae1))
		return ae1.view(-1), ae2.view(-1), ae2ae1.view(-1)

	def train_step(self, epoch, data, optimizer, scheduler, feats):
		l = nn.MSELoss(reduction='none')
		n = epoch + 1
		l1s, l2s = [], []
		for d in data:
			ae1s, ae2s, ae2ae1s = self(d)
			l1 = (1 / n) * l(ae1s, d) + (1 - 1/n) * l(ae2ae1s, d)
			l2 = (1 / n) * l(ae2s, d) - (1 - 1/n) * l(ae2ae1s, d)
			l1s.append(torch.mean(l1).item())
			l2s.append(torch.mean(l2).item())
			loss = torch.mean(l1 + l2)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
		scheduler.step()
		tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)},\tL2 = {np.mean(l2s)}')
		return np.mean(l1s)+np.mean(l2s), optimizer.param_groups[0]['lr']

	def eval_step(self, epoch, data, optimizer, scheduler, feats):
		l = nn.MSELoss(reduction='none')
		ae1s, ae2s, ae2ae1s = [], [], []
		for d in data:
			ae1, ae2, ae2ae1 = self(d)
			ae1s.append(ae1)
			ae2s.append(ae2)
			ae2ae1s.append(ae2ae1)
		ae1s, ae2s, ae2ae1s = torch.stack(ae1s), torch.stack(ae2s), torch.stack(ae2ae1s)
		y_pred = ae1s[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
		loss = 0.1 * l(ae1s, data) + 0.9 * l(ae2ae1s, data)
		loss = loss[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
		return loss.detach().numpy(), y_pred.detach().numpy()
