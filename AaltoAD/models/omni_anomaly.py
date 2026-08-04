import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from .base import BaseModel


## OmniAnomaly Model (KDD 19)
class OmniAnomaly(BaseModel):
	def __init__(self, feats, n_hidden=32, n_latent=8, n_layers=2, beta=0.01, learning_rate=0.002, epochs=5, weight_decay=1e-5):
		super().__init__(feats, learning_rate=learning_rate, epochs=epochs, weight_decay=weight_decay)
		self.name = 'OmniAnomaly'
		self.beta = beta
		self.n_hidden = n_hidden
		self.n_latent = n_latent
		self.n_layers = n_layers
		self.lstm = nn.GRU(feats, self.n_hidden, self.n_layers)
		self.encoder = nn.Sequential(
			nn.Linear(self.n_hidden, self.n_hidden), nn.PReLU(),
			nn.Linear(self.n_hidden, self.n_hidden), nn.PReLU(),
			nn.Flatten(),
			nn.Linear(self.n_hidden, 2*self.n_latent)
		)
		self.decoder = nn.Sequential(
			nn.Linear(self.n_latent, self.n_hidden), nn.PReLU(),
			nn.Linear(self.n_hidden, self.n_hidden), nn.PReLU(),
			nn.Linear(self.n_hidden, self.n_feats), nn.Sigmoid(),
		)
		self.lstm = self.lstm.double()
		self.encoder = self.encoder.double()
		self.decoder = self.decoder.double()

	def forward(self, x, hidden=None):
		hidden = torch.rand(self.n_layers, 1, self.n_hidden, dtype=torch.float64) if hidden is not None else hidden
		out, hidden = self.lstm(x.view(1, 1, -1), hidden)
		x = self.encoder(out)
		mu, logvar = torch.split(x, [self.n_latent, self.n_latent], dim=-1)
		std = torch.exp(0.5*logvar)
		eps = torch.randn_like(std)
		x = mu + eps*std
		x = self.decoder(x)
		return x.view(-1), mu.view(-1), logvar.view(-1), hidden

	def train_step(self, epoch, data, optimizer, scheduler, feats):
		l = nn.MSELoss(reduction='none')
		mses, klds = [], []
		for i, d in enumerate(data):
			y_pred, mu, logvar, hidden = self(d, hidden if i else None)
			MSE = l(y_pred, d)
			KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=0)
			loss = torch.mean(MSE) + self.beta * torch.mean(KLD)
			mses.append(torch.mean(MSE).item())
			klds.append(self.beta * torch.mean(KLD).item())
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
		tqdm.write(f'Epoch {epoch},\tMSE = {np.mean(mses)},\tKLD = {np.mean(klds)}')
		scheduler.step()
		return loss.item(), optimizer.param_groups[0]['lr']

	def eval_step(self, epoch, data, optimizer, scheduler, feats):
		l = nn.MSELoss(reduction='none')
		y_preds = []
		for i, d in enumerate(data):
			y_pred, _, _, hidden = self(d, hidden if i else None)
			y_preds.append(y_pred)
		y_pred = torch.stack(y_preds)
		MSE = l(y_pred, data)
		return MSE.detach().numpy(), y_pred.detach().numpy()
