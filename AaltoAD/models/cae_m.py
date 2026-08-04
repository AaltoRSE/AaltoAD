import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from .base import BaseModel


## CAE-M Model (TKDE 21)
class CAE_M(BaseModel):
	def __init__(self, feats, n_window=None, learning_rate=0.0001, epochs=5, weight_decay=1e-5):
		super().__init__(feats, learning_rate=learning_rate, epochs=epochs, weight_decay=weight_decay)
		self.name = 'CAE_M'
		self.n_window = n_window or feats
		self.flat_window = True
		self.encoder = nn.Sequential(
			nn.Conv2d(1, 8, (3, 3), 1, 1), nn.Sigmoid(),
			nn.Conv2d(8, 16, (3, 3), 1, 1), nn.Sigmoid(),
			nn.Conv2d(16, 32, (3, 3), 1, 1), nn.Sigmoid(),
		)
		self.decoder = nn.Sequential(
			nn.ConvTranspose2d(32, 4, (3, 3), 1, 1), nn.Sigmoid(),
			nn.ConvTranspose2d(4, 4, (3, 3), 1, 1), nn.Sigmoid(),
			nn.ConvTranspose2d(4, 1, (3, 3), 1, 1), nn.Sigmoid(),
		)
		self.encoder = self.encoder.double()
		self.decoder = self.decoder.double()

	def forward(self, g):
		z = g.view(1, 1, self.n_feats, self.n_window)
		z = self.encoder(z)
		x = self.decoder(z)
		return x.view(-1)

	def train_step(self, epoch, data, optimizer, scheduler, feats):
		l = nn.MSELoss(reduction='none')
		l1s = []
		for i, d in enumerate(data):
			x = self(d)
			loss = torch.mean(l(x, d))
			l1s.append(torch.mean(loss).item())
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
		tqdm.write(f'Epoch {epoch},\tMSE = {np.mean(l1s)}')
		return np.mean(l1s), optimizer.param_groups[0]['lr']

	def eval_step(self, epoch, data, optimizer, scheduler, feats):
		l = nn.MSELoss(reduction='none')
		xs = []
		for d in data:
			x = self(d)
			xs.append(x)
		xs = torch.stack(xs)
		y_pred = xs[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
		loss = l(xs, data)
		loss = loss[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
		return loss.detach().numpy(), y_pred.detach().numpy()
