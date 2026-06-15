import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset


class TranADBase:
	def train_step(self, epoch, data, optimizer, scheduler, feats):
		l = nn.MSELoss(reduction='none')
		data_x = torch.DoubleTensor(data)
		dataset = TensorDataset(data_x, data_x)
		bs = self.batch
		dataloader = DataLoader(dataset, batch_size=bs)
		n = epoch + 1
		l1s = []
		for d, _ in dataloader:
			local_bs = d.shape[0]
			window = d.permute(1, 0, 2)
			elem = window[-1, :, :].view(1, local_bs, feats)
			z = self(window, elem)
			l1 = l(z, elem) if not isinstance(z, tuple) else (1 / n) * l(z[0], elem) + (1 - 1/n) * l(z[1], elem)
			if isinstance(z, tuple):
				z = z[1]
			l1s.append(torch.mean(l1).item())
			loss = torch.mean(l1)
			optimizer.zero_grad()
			loss.backward(retain_graph=True)
			optimizer.step()
		scheduler.step()
		tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)}')
		return np.mean(l1s), optimizer.param_groups[0]['lr']

	def eval_step(self, epoch, data, optimizer, scheduler, feats):
		l = nn.MSELoss(reduction='none')
		data_x = torch.DoubleTensor(data)
		dataset = TensorDataset(data_x, data_x)
		bs = len(data)
		dataloader = DataLoader(dataset, batch_size=bs)
		for d, _ in dataloader:
			window = d.permute(1, 0, 2)
			elem = window[-1, :, :].view(1, bs, feats)
			z = self(window, elem)
			if isinstance(z, tuple):
				z = z[1]
		loss = l(z, elem)[0]
		return loss.detach().numpy(), z.detach().numpy()[0]
