import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from TranAD import dlutils
from .base import BaseModel


## MSCRED Model (AAAI 19)
class MSCRED(BaseModel):
	def __init__(self, feats, n_window=None, learning_rate=0.0001, epochs=5, weight_decay=1e-5):
		super().__init__(feats, learning_rate=learning_rate, epochs=epochs, weight_decay=weight_decay)
		self.name = 'MSCRED'
		self.n_window = n_window or feats
		self.flat_window = True
		self.encoder = nn.ModuleList([
			dlutils.ConvLSTM(1, 32, (3, 3), 1, True, True, False),
			dlutils.ConvLSTM(32, 64, (3, 3), 1, True, True, False),
			dlutils.ConvLSTM(64, 128, (3, 3), 1, True, True, False),
		])
		self.decoder = nn.Sequential(
			nn.ConvTranspose2d(128, 64, (3, 3), 1, 1), nn.ReLU(True),
			nn.ConvTranspose2d(64, 32, (3, 3), 1, 1), nn.ReLU(True),
			nn.ConvTranspose2d(32, 1, (3, 3), 1, 1), nn.Sigmoid(),
		)
		self.encoder = self.encoder.double()
		self.decoder = self.decoder.double()

	def forward(self, g):
		z = g.view(1, 1, self.n_feats, self.n_window)
		for cell in self.encoder:
			_, z = cell(z.view(1, *z.shape))
			z = z[0][0]
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
