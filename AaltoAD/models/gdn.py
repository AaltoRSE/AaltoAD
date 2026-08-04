import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import GATConv
from .base import BaseModel


## GDN Model (AAAI 21)
class GDN(BaseModel):
	def __init__(self, feats, n_window=5, n_hidden=16, learning_rate=0.0001, epochs=5, weight_decay=1e-5):
		super().__init__(feats, learning_rate=learning_rate, epochs=epochs, weight_decay=weight_decay)
		self.name = 'GDN'
		self.n_window = n_window
		self.flat_window = True
		self.n_hidden = n_hidden
		self.n = self.n_window * self.n_feats
		src_ids = np.repeat(np.array(list(range(feats))), feats)
		dst_ids = np.array(list(range(feats))*feats)
		edge_index = torch.tensor(np.array([src_ids, dst_ids]), dtype=torch.long)
		edge_index, _ = add_self_loops(edge_index)
		self.g = Data(edge_index=edge_index)
		self.feature_gat = GATConv(1, 1, feats)
		self.attention = nn.Sequential(
			nn.Linear(self.n, self.n_hidden), nn.LeakyReLU(True),
			nn.Linear(self.n_hidden, self.n_hidden), nn.LeakyReLU(True),
			nn.Linear(self.n_hidden, self.n_window), nn.Softmax(dim=0),
		)
		self.fcn = nn.Sequential(
			nn.Linear(self.n_feats, self.n_hidden), nn.LeakyReLU(True),
			nn.Linear(self.n_hidden, self.n_window), nn.Sigmoid(),
		)
		self.feature_gat = self.feature_gat.double()
		self.attention = self.attention.double()
		self.fcn = self.fcn.double()

	def forward(self, data):
		att_score = self.attention(data).view(self.n_window, 1)
		data = data.view(self.n_window, self.n_feats)
		data_r = torch.matmul(data.permute(1, 0), att_score)
		feat_r = self.feature_gat(data_r, self.g.edge_index)
		feat_r = feat_r.view(self.n_feats, self.n_feats)
		x = self.fcn(feat_r)
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
