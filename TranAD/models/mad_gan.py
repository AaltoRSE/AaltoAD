import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from .base import BaseModel


# MAD_GAN (ICANN 19)
class MAD_GAN(BaseModel):
	def __init__(self, feats, n_window=5, n_hidden=16, learning_rate=0.0001, epochs=5, weight_decay=1e-5):
		super().__init__(feats, learning_rate=learning_rate, epochs=epochs, weight_decay=weight_decay)
		self.name = 'MAD_GAN'
		self.n_hidden = n_hidden
		self.n_window = n_window
		self.flat_window = True
		self.n = self.n_feats * self.n_window
		self.generator = nn.Sequential(
			nn.Flatten(),
			nn.Linear(self.n, self.n_hidden), nn.LeakyReLU(True),
			nn.Linear(self.n_hidden, self.n_hidden), nn.LeakyReLU(True),
			nn.Linear(self.n_hidden, self.n), nn.Sigmoid(),
		)
		self.discriminator = nn.Sequential(
			nn.Flatten(),
			nn.Linear(self.n, self.n_hidden), nn.LeakyReLU(True),
			nn.Linear(self.n_hidden, self.n_hidden), nn.LeakyReLU(True),
			nn.Linear(self.n_hidden, 1), nn.Sigmoid(),
		)
		self.generator = self.generator.double()
		self.discriminator = self.discriminator.double()

	def forward(self, g):
		z = self.generator(g.view(1,-1))
		real_score = self.discriminator(g.view(1,-1))
		fake_score = self.discriminator(z.view(1,-1))
		return z.view(-1), real_score.view(-1), fake_score.view(-1)

	def train_step(self, epoch, data, optimizer, scheduler, feats):
		l = nn.MSELoss(reduction='none')
		bcel = nn.BCELoss(reduction='mean')
		msel = nn.MSELoss(reduction='mean')
		real_label, fake_label = torch.tensor([0.9]), torch.tensor([0.1])
		real_label, fake_label = real_label.type(torch.DoubleTensor), fake_label.type(torch.DoubleTensor)
		mses, gls, dls = [], [], []
		for d in data:
			self.discriminator.zero_grad()
			_, real, fake = self(d)
			dl = bcel(real, real_label) + bcel(fake, fake_label)
			dl.backward()
			self.generator.zero_grad()
			optimizer.step()
			z, _, fake = self(d)
			mse = msel(z, d)
			gl = bcel(fake, real_label)
			tl = gl + mse
			tl.backward()
			self.discriminator.zero_grad()
			optimizer.step()
			mses.append(mse.item())
			gls.append(gl.item())
			dls.append(dl.item())
		tqdm.write(f'Epoch {epoch},\tMSE = {np.mean(mses)},\tG = {np.mean(gls)},\tD = {np.mean(dls)}')
		return np.mean(gls)+np.mean(dls), optimizer.param_groups[0]['lr']

	def eval_step(self, epoch, data, optimizer, scheduler, feats):
		l = nn.MSELoss(reduction='none')
		outputs = []
		for d in data:
			z, _, _ = self(d)
			outputs.append(z)
		outputs = torch.stack(outputs)
		y_pred = outputs[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
		loss = l(outputs, data)
		loss = loss[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
		return loss.detach().numpy(), y_pred.detach().numpy()
