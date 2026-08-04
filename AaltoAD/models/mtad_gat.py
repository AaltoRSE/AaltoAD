import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import GATConv
from .base import BaseModel


## MTAD_GAT Model (ICDM 20)
class MTAD_GAT(BaseModel):
	def __init__(self, feats, n_window=None, learning_rate=0.0001, epochs=5, weight_decay=1e-5):
		super().__init__(feats, learning_rate=learning_rate, epochs=epochs, weight_decay=weight_decay)
		self.name = 'MTAD_GAT'
		self.n_window = n_window or feats
		self.flat_window = True
		edge_index = torch.tensor([list(range(1, feats+1)), [0]*feats], dtype=torch.long)
		edge_index, _ = add_self_loops(edge_index)
		self.g = Data(edge_index=edge_index)
		self.feature_gat = GATConv(feats, feats, heads=1)
		self.time_gat = GATConv(feats, feats, heads=1)
		self.gru = nn.GRU((feats+1)*feats*3, feats*feats, 1)
		self.feature_gat = self.feature_gat.double()
		self.time_gat = self.time_gat.double()
		self.gru = self.gru.double()

	def forward(self, data, hidden=None):
		if hidden is None:
			hidden = torch.rand(1, 1, self.n_feats * self.n_feats, dtype=torch.float64)
		data = data.view(self.n_window, self.n_feats)
		data_r = torch.cat((torch.zeros(1, self.n_feats), data))
		feat_r = self.feature_gat(data_r, self.g.edge_index)
		data_t = torch.cat((torch.zeros(1, self.n_feats), data.t()))
		time_r = self.time_gat(data_t, self.g.edge_index)
		data = torch.cat((torch.zeros(1, self.n_feats), data))
		data = data.view(self.n_window+1, self.n_feats, 1)
		feat_r = feat_r.unsqueeze(2)
		time_r = time_r.unsqueeze(2)
		x = torch.cat((data, feat_r, time_r), dim=2).view(1, 1, -1)
		x, h = self.gru(x, hidden)
		return x.view(-1), h

	def train_step(self, epoch, data, optimizer, scheduler, feats):
		l = nn.MSELoss(reduction='none')
		l1s = []
		for i, d in enumerate(data):
			x, h = self(d, h.detach() if i else None)
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
			x, h = self(d, None)
			xs.append(x)
		xs = torch.stack(xs)
		y_pred = xs[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
		loss = l(xs, data)
		loss = loss[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
		return loss.detach().numpy(), y_pred.detach().numpy()
