import torch.nn as nn
import numpy as np


class BaseModel(nn.Module):
	def __init__(self, feats, learning_rate=0.002, epochs=5, weight_decay=1e-5):
		super(BaseModel, self).__init__()
		self.name = 'BaseModel'
		self.lr = learning_rate
		self.epochs = epochs
		self.weight_decay = weight_decay
		self.n_feats = feats

	def forward(self, x):
		return x

	def train_step(self, epoch, data, optimizer, scheduler, feats):
		return 0.0, optimizer.param_groups[0]['lr']

	def eval_step(self, epoch, data, optimizer, scheduler, feats):
		return np.zeros_like(data), np.zeros_like(data)
