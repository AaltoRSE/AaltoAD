import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from .base import BaseModel


## Simple Multi-Head Self-Attention Model
class Attention(BaseModel):
	def __init__(self, feats, n_window=5, learning_rate=0.0001, epochs=5, weight_decay=1e-5):
		super().__init__(feats, learning_rate=learning_rate, epochs=epochs, weight_decay=weight_decay)
		self.name = 'Attention'
		self.n_window = n_window
		self.flat_window = False
		self.n = self.n_feats * self.n_window
		self.atts = [nn.Sequential(nn.Linear(self.n, feats * feats), nn.ReLU(True)) for i in range(1)]
		self.atts = nn.ModuleList(self.atts)
		self.atts = self.atts.double()

	def forward(self, g):
		for at in self.atts:
			ats = at(g.view(-1)).reshape(self.n_feats, self.n_feats)
			g = torch.matmul(g, ats)
		return g, ats

	def train_step(self, epoch, data, optimizer, scheduler, feats):
		l = nn.MSELoss(reduction='none')
		l1s = []
		for d in data:
			ae, ats = self(d)
			l1 = l(ae, d)
			l1s.append(torch.mean(l1).item())
			loss = torch.mean(l1)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
		scheduler.step()
		tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)}')
		return np.mean(l1s), optimizer.param_groups[0]['lr']

	def eval_step(self, epoch, data, optimizer, scheduler, feats):
		l = nn.MSELoss(reduction='none')
		ae1s, y_pred = [], []
		for d in data:
			ae1, ats = self(d)
			y_pred.append(ae1[-1])
			ae1s.append(ae1)
		ae1s, y_pred = torch.stack(ae1s), torch.stack(y_pred)
		loss = torch.mean(l(ae1s, data), axis=1)
		return loss.detach().numpy(), y_pred.detach().numpy()
