import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from .base import BaseModel


## DAGMM Model (ICLR 18)
class DAGMM(BaseModel):
	def __init__(self, feats, n_hidden=16, n_latent=8, n_window=5, learning_rate=0.0001, epochs=5, weight_decay=1e-5):
		super().__init__(feats, learning_rate=learning_rate, epochs=epochs, weight_decay=weight_decay)
		self.name = 'DAGMM'
		self.n_hidden = n_hidden
		self.n_latent = n_latent
		self.n_window = n_window
		self.flat_window = True
		self.n = self.n_feats * self.n_window
		self.n_gmm = self.n_feats * self.n_window
		self.encoder = nn.Sequential(
			nn.Linear(self.n, self.n_hidden), nn.Tanh(),
			nn.Linear(self.n_hidden, self.n_hidden), nn.Tanh(),
			nn.Linear(self.n_hidden, self.n_latent)
		)
		self.decoder = nn.Sequential(
			nn.Linear(self.n_latent, self.n_hidden), nn.Tanh(),
			nn.Linear(self.n_hidden, self.n_hidden), nn.Tanh(),
			nn.Linear(self.n_hidden, self.n), nn.Sigmoid(),
		)
		self.estimate = nn.Sequential(
			nn.Linear(self.n_latent+2, self.n_hidden), nn.Tanh(), nn.Dropout(0.5),
			nn.Linear(self.n_hidden, self.n_gmm), nn.Softmax(dim=1),
		)
		self.encoder = self.encoder.double()
		self.decoder = self.decoder.double()
		self.estimate = self.estimate.double()

	def compute_reconstruction(self, x, x_hat):
		relative_euclidean_distance = (x-x_hat).norm(2, dim=1) / (x.norm(2, dim=1) + 1e-10)
		cosine_similarity = F.cosine_similarity(x, x_hat, dim=1)
		return relative_euclidean_distance, cosine_similarity

	def forward(self, x):
		x = x.view(1, -1)
		z_c = self.encoder(x)
		x_hat = self.decoder(z_c)
		rec_1, rec_2 = self.compute_reconstruction(x, x_hat)
		z = torch.cat([z_c, rec_1.unsqueeze(-1), rec_2.unsqueeze(-1)], dim=1)
		gamma = self.estimate(z)
		return z_c, x_hat.view(-1), z, gamma.view(-1)

	def train_step(self, epoch, data, optimizer, scheduler, feats):
		l = nn.MSELoss(reduction='none')
		l1s = []
		l2s = []
		for d in data:
			_, x_hat, z, gamma = self(d)
			l1, l2 = l(x_hat, d), l(gamma, d)
			l1s.append(torch.mean(l1).item())
			l2s.append(torch.mean(l2).item())
			loss = torch.mean(l1) + torch.mean(l2)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
		scheduler.step()
		tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)},\tL2 = {np.mean(l2s)}')
		return np.mean(l1s)+np.mean(l2s), optimizer.param_groups[0]['lr']

	def eval_step(self, epoch, data, optimizer, scheduler, feats):
		l = nn.MSELoss(reduction='none')
		ae1s = []
		for d in data:
			_, x_hat, _, _ = self(d)
			ae1s.append(x_hat)
		ae1s = torch.stack(ae1s)
		y_pred = ae1s[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
		loss = l(ae1s, data)[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
		return loss.detach().numpy(), y_pred.detach().numpy()
