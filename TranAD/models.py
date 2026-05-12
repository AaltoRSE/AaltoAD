import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import GATConv
from torch.nn import TransformerEncoder
from torch.nn import TransformerDecoder
from torch.nn import TransformerEncoderLayer, TransformerDecoderLayer
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import math
from tqdm import tqdm
from TranAD import dlutils
from TranAD import constants
from TranAD.stagnn_core import STAGNN_Core
torch.manual_seed(1)


## Separate LSTM for each variable
class LSTM_Univariate(nn.Module):
	def __init__(self, feats, n_hidden=1, n_layers=1, learning_rate=0.002, batch_size = 512):
		super(LSTM_Univariate, self).__init__()
		self.name = 'LSTM_Univariate'
		self.lr = learning_rate
		self.n_feats = feats
		self.n_hidden = n_hidden
		self.n_layers = n_layers
		self.batch_size = batch_size
		self.lstm = nn.ModuleList([nn.LSTM(1, self.n_hidden, self.n_layers) for i in range(feats)])
		self.lstm = self.lstm.double()

	def forward(self, x):
		hidden = [(torch.rand(self.n_layers, 1, self.n_hidden, dtype=torch.float64), 
			torch.randn(self.n_layers, 1, self.n_hidden, dtype=torch.float64)) for i in range(self.n_feats)]
		outputs = []
		for i, g in enumerate(x):
			multivariate_output = []
			for j in range(self.n_feats):
				univariate_input = g.view(-1)[j].view(1, 1, -1).to(torch.float64)
				out, hidden[j] = self.lstm[j](univariate_input, hidden[j])
				multivariate_output.append(2 * out[:, :, -1].view(-1))
			output = torch.cat(multivariate_output)
			outputs.append(output)
		return torch.stack(outputs)

	def train_step(self, epoch, data, optimizer, scheduler, feats):
		l = nn.MSELoss(reduction='mean')
		total = 0.0
		n_samples = 0
		for b in range(0, data.shape[0], self.batch_size):
			batch_data = data[b:b+self.batch_size]
			batch_pred = self(batch_data)
			loss = l(batch_pred, batch_data)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			total += loss.item() * batch_data.shape[0]
			n_samples += batch_data.shape[0]
		scheduler.step()
		mean_loss = total / max(n_samples, 1)
		tqdm.write(f'Epoch {epoch},\tMSE = {mean_loss}')
		return mean_loss, optimizer.param_groups[0]['lr']

	def eval_step(self, epoch, data, optimizer, scheduler, feats):
		l = nn.MSELoss(reduction='none')
		y_pred = self(data)
		loss = l(y_pred, data)
		return loss.detach().numpy(), y_pred.detach().numpy()



## Simple Multi-Head Self-Attention Model
class Attention(nn.Module):
	def __init__(self, feats, n_window=5, learning_rate=0.0001):
		super(Attention, self).__init__()
		self.name = 'Attention'
		self.lr = learning_rate
		self.n_feats = feats
		self.n_window = n_window
		self.flat_window = False
		self.n = self.n_feats * self.n_window
		self.atts = [ nn.Sequential( nn.Linear(self.n, feats * feats), 
				nn.ReLU(True))	for i in range(1)]
		self.atts = nn.ModuleList(self.atts)
		self.atts = self.atts.double()

	def forward(self, g):
		for at in self.atts:
			ats = at(g.view(-1)).reshape(self.n_feats, self.n_feats)
			g = torch.matmul(g, ats)
		return g, ats

	def train_step(self, epoch, data, optimizer, scheduler, feats):
		l = nn.MSELoss(reduction='none')
		#n = epoch + 1
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


## LSTM_AD Model
class LSTM_AD(nn.Module):
	def __init__(self, feats, n_hidden=64, n_layers=1, sequence_length = 30, learning_rate=0.002, epochs=10, batch_size = 256):
		super(LSTM_AD, self).__init__()
		self.name = 'LSTM_AD'
		self.lr = learning_rate
		self.n_feats = feats
		self.n_hidden = n_hidden
		self.n_layers = n_layers
		self.n_window = sequence_length
		self.flat_window = False
		self.epochs = epochs
		self.batch_size = batch_size
		self.lstm = nn.LSTM(feats, self.n_hidden, n_layers, batch_first=True)
		self.fcn = nn.Linear(self.n_hidden, self.n_feats)
		self.lstm = self.lstm.double()
		self.fcn = self.fcn.double()

	def forward(self, x):
		B = x.size(0)
		h0 = torch.zeros(self.n_layers, B, self.n_hidden, dtype=x.dtype, device=x.device)
		c0 = torch.zeros(self.n_layers, B, self.n_hidden, dtype=x.dtype, device=x.device)
		out, _ = self.lstm(x, (h0, c0))
		out = self.fcn(out)
		return out

	def train_step(self, epoch, data, optimizer, scheduler, feats):
		data_x = torch.as_tensor(data, dtype=torch.float64)
		if data_x.ndim != 3:
			print(data_x.shape)
			raise ValueError("LSTM_AD expects windowed input [N, W, F]")
		
		dataset = TensorDataset(data_x, data_x)
		bs = self.batch_size
		dataloader = DataLoader(dataset, batch_size=bs, shuffle=True, drop_last=False)

		device = next(self.parameters()).device

		self.train()
		criterion = nn.MSELoss(reduction='mean')
		total_loss = 0.0
		n_samples = 0
		for xb, yb in dataloader:
			xb = xb.to(device)
			yb = yb.to(device)

			pred = self(xb)
			loss = criterion(pred, yb)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			total_loss += loss.item() * xb.size(0)
			n_samples += xb.size(0)

		scheduler.step()
		mean_loss = total_loss / max(n_samples, 1)
		tqdm.write(f'Epoch {epoch},\tMSE = {mean_loss}')
		
		return mean_loss, optimizer.param_groups[0]['lr']

	def eval_step(self, epoch, data, optimizer, scheduler, feats):
		data_x = torch.as_tensor(data, dtype=torch.float64)
		if data_x.ndim != 3:
			print(data_x.shape)
			raise ValueError("LSTM_AD expects windowed input [N, W, F]")
		
		dataset = TensorDataset(data_x, data_x)
		bs = len(dataset)
		dataloader = DataLoader(dataset, batch_size=bs, shuffle=False, drop_last=False)

		device = next(self.parameters()).device

		self.eval()
		criterion = nn.MSELoss(reduction='none')
		all_loss = []
		all_pred = []
		with torch.no_grad():
			for xb, yb in dataloader:
				xb = xb.to(device)
				yb = yb.to(device)

				pred = self(xb)
				loss = criterion(pred, yb)

				all_loss.append(loss.cpu())
				all_pred.append(pred.cpu())

		pred = torch.cat(all_pred, dim=0)    # [N, W, F]
		losses = torch.cat(all_loss, dim=0)  # [N, W, F]

		losses_np = losses.numpy()                       # [N, W, F]
		N, W, F = losses_np.shape
		# Window i (right-aligned by convert_to_windows) covers original times [i-W+1, i].
		starts = np.arange(N) - W + 1
		positions = starts[:, None] + np.arange(W)[None, :]   # [N, W], original-time index
		mask = (positions >= 0) & (positions < N)             # padded prefix is masked out

		valid_positions = positions[mask]                     # [count]
		valid_losses    = losses_np[mask]                     # [count, F]

		score_sum   = np.zeros((N, F), dtype=np.float64)
		score_count = np.zeros(N,       dtype=np.float64)
		np.add.at(score_sum,   valid_positions, valid_losses)
		np.add.at(score_count, valid_positions, 1)
		losses_per_t = score_sum / np.maximum(score_count, 1)[:, None]   # [N, F]

		pred = pred[:, -1, :]                                 # [N, F] - kept for plotting only
		return losses_per_t, pred.detach().numpy()

## DAGMM Model (ICLR 18)
class DAGMM(nn.Module):
	def __init__(self, feats, n_hidden=16, n_latent=8, n_window=5, learning_rate=0.0001):
		super(DAGMM, self).__init__()
		self.name = 'DAGMM'
		self.lr = learning_rate
		self.n_feats = feats
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
		## Encode Decoder
		x = x.view(1, -1)
		z_c = self.encoder(x)
		x_hat = self.decoder(z_c)
		## Compute Reconstructoin
		rec_1, rec_2 = self.compute_reconstruction(x, x_hat)
		z = torch.cat([z_c, rec_1.unsqueeze(-1), rec_2.unsqueeze(-1)], dim=1)
		## Estimate
		gamma = self.estimate(z)
		return z_c, x_hat.view(-1), z, gamma.view(-1)

	def train_step(self, epoch, data, optimizer, scheduler, feats):
		l = nn.MSELoss(reduction='none')
		#compute = models.ComputeLoss(self, 0.1, 0.005, 'cpu', self.n_gmm)
		#n = epoch + 1
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

## OmniAnomaly Model (KDD 19)
class OmniAnomaly(nn.Module):
	def __init__(self, feats, n_hidden=32, n_latent=8, n_layers=2, beta=0.01, learning_rate=0.002):
		super(OmniAnomaly, self).__init__()
		self.name = 'OmniAnomaly'
		self.lr = learning_rate
		self.beta = beta
		self.n_feats = feats
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

	def forward(self, x, hidden = None):
		hidden = torch.rand(self.n_layers, 1, self.n_hidden, dtype=torch.float64) if hidden is not None else hidden
		out, hidden = self.lstm(x.view(1, 1, -1), hidden)
		## Encode
		x = self.encoder(out)
		mu, logvar = torch.split(x, [self.n_latent, self.n_latent], dim=-1)
		## Reparameterization trick
		std = torch.exp(0.5*logvar)
		eps = torch.randn_like(std)
		x = mu + eps*std
		## Decoder
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

## USAD Model (KDD 20)
class USAD(nn.Module):
	def __init__(self, feats, n_hidden=16, n_latent=5, n_window=5, learning_rate=0.0001):
		super(USAD, self).__init__()
		self.name = 'USAD'
		self.lr = learning_rate
		self.n_feats = feats
		self.n_hidden = n_hidden
		self.n_latent = n_latent
		self.n_window = n_window
		self.flat_window = True
		self.n = self.n_feats * self.n_window
		self.encoder = nn.Sequential(
			nn.Flatten(),
			nn.Linear(self.n, self.n_hidden), nn.ReLU(True),
			nn.Linear(self.n_hidden, self.n_hidden), nn.ReLU(True),
			nn.Linear(self.n_hidden, self.n_latent), nn.ReLU(True),
		)
		self.decoder1 = nn.Sequential(
			nn.Linear(self.n_latent,self.n_hidden), nn.ReLU(True),
			nn.Linear(self.n_hidden, self.n_hidden), nn.ReLU(True),
			nn.Linear(self.n_hidden, self.n), nn.Sigmoid(),
		)
		self.decoder2 = nn.Sequential(
			nn.Linear(self.n_latent,self.n_hidden), nn.ReLU(True),
			nn.Linear(self.n_hidden, self.n_hidden), nn.ReLU(True),
			nn.Linear(self.n_hidden, self.n), nn.Sigmoid(),
		)
		self.encoder = self.encoder.double()
		self.decoder1 = self.decoder1.double()
		self.decoder2 = self.decoder2.double()

	def forward(self, g):
		## Encode
		z = self.encoder(g.view(1,-1))
		## Decoders (Phase 1)
		ae1 = self.decoder1(z)
		ae2 = self.decoder2(z)
		## Encode-Decode (Phase 2)
		ae2ae1 = self.decoder2(self.encoder(ae1))
		return ae1.view(-1), ae2.view(-1), ae2ae1.view(-1)

	def train_step(self, epoch, data, optimizer, scheduler, feats):
		l = nn.MSELoss(reduction='none')
		n = epoch + 1
		l1s, l2s = [], []
		for d in data:
			ae1s, ae2s, ae2ae1s = self(d)
			l1 = (1 / n) * l(ae1s, d) + (1 - 1/n) * l(ae2ae1s, d)
			l2 = (1 / n) * l(ae2s, d) - (1 - 1/n) * l(ae2ae1s, d)
			l1s.append(torch.mean(l1).item())
			l2s.append(torch.mean(l2).item())
			loss = torch.mean(l1 + l2)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
		scheduler.step()
		tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)},\tL2 = {np.mean(l2s)}')
		return np.mean(l1s)+np.mean(l2s), optimizer.param_groups[0]['lr']

	def eval_step(self, epoch, data, optimizer, scheduler, feats):
		l = nn.MSELoss(reduction='none')
		ae1s, ae2s, ae2ae1s = [], [], []
		for d in data:
			ae1, ae2, ae2ae1 = self(d)
			ae1s.append(ae1)
			ae2s.append(ae2)
			ae2ae1s.append(ae2ae1)
		ae1s, ae2s, ae2ae1s = torch.stack(ae1s), torch.stack(ae2s), torch.stack(ae2ae1s)
		y_pred = ae1s[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
		loss = 0.1 * l(ae1s, data) + 0.9 * l(ae2ae1s, data)
		loss = loss[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
		return loss.detach().numpy(), y_pred.detach().numpy()

## MSCRED Model (AAAI 19)
class MSCRED(nn.Module):
	def __init__(self, feats, n_window=None, learning_rate=0.0001):
		super(MSCRED, self).__init__()
		self.name = 'MSCRED'
		self.lr = learning_rate
		self.n_feats = feats
		self.n_window = n_window or feats
		self.flat_window = True
		self.encoder = nn.ModuleList([
			dlutils.ConvLSTM(1, 32, (3, 3), 1, True, True, False),
			dlutils.ConvLSTM(32, 64, (3, 3), 1, True, True, False),
			dlutils.ConvLSTM(64, 128, (3, 3), 1, True, True, False),
			]
		)
		self.decoder = nn.Sequential(
			nn.ConvTranspose2d(128, 64, (3, 3), 1, 1), nn.ReLU(True),
			nn.ConvTranspose2d(64, 32, (3, 3), 1, 1), nn.ReLU(True),
			nn.ConvTranspose2d(32, 1, (3, 3), 1, 1), nn.Sigmoid(),
		)
		self.encoder = self.encoder.double()
		self.decoder = self.decoder.double()

	def forward(self, g):
		## Encode
		z = g.view(1, 1, self.n_feats, self.n_window)
		for cell in self.encoder:
			_, z = cell(z.view(1, *z.shape))
			z = z[0][0]
		## Decode
		x = self.decoder(z)
		return x.view(-1)

	def train_step(self, epoch, data, optimizer, scheduler, feats):
		l = nn.MSELoss(reduction='none')
		n = epoch + 1
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

## CAE-M Model (TKDE 21)
class CAE_M(nn.Module):
	def __init__(self, feats, n_window=None, learning_rate=0.0001):
		super(CAE_M, self).__init__()
		self.name = 'CAE_M'
		self.lr = learning_rate
		self.n_feats = feats
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
		## Encode
		z = g.view(1, 1, self.n_feats, self.n_window)
		z = self.encoder(z)
		## Decode
		x = self.decoder(z)
		return x.view(-1)

	def train_step(self, epoch, data, optimizer, scheduler, feats):
		l = nn.MSELoss(reduction='none')
		n = epoch + 1
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

## MTAD_GAT Model (ICDM 20)
class MTAD_GAT(nn.Module):
	def __init__(self, feats, n_window=None, learning_rate=0.0001):
		super(MTAD_GAT, self).__init__()
		self.name = 'MTAD_GAT'
		self.lr = learning_rate
		self.n_feats = feats
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
		n = epoch + 1
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

## GDN Model (AAAI 21)
class GDN(nn.Module):
	def __init__(self, feats, n_window=5, n_hidden=16, learning_rate=0.0001):
		super(GDN, self).__init__()
		self.name = 'GDN'
		self.lr = learning_rate
		self.n_feats = feats
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
		# Bahdanau style attention
		att_score = self.attention(data).view(self.n_window, 1)
		data = data.view(self.n_window, self.n_feats)
		data_r = torch.matmul(data.permute(1, 0), att_score)
		# GAT convolution on complete graph
		feat_r = self.feature_gat(data_r, self.g.edge_index)
		feat_r = feat_r.view(self.n_feats, self.n_feats)
		# Pass through a FCN
		x = self.fcn(feat_r)
		return x.view(-1)

	def train_step(self, epoch, data, optimizer, scheduler, feats):
		l = nn.MSELoss(reduction='none')
		n = epoch + 1
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

# MAD_GAN (ICANN 19)
class MAD_GAN(nn.Module):
	def __init__(self, feats, n_window=5, n_hidden=16, learning_rate=0.0001):
		super(MAD_GAN, self).__init__()
		self.name = 'MAD_GAN'
		self.lr = learning_rate
		self.n_feats = feats
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
		## Generate
		z = self.generator(g.view(1,-1))
		## Discriminator
		real_score = self.discriminator(g.view(1,-1))
		fake_score = self.discriminator(z.view(1,-1))
		return z.view(-1), real_score.view(-1), fake_score.view(-1)

	def train_step(self, epoch, data, optimizer, scheduler, feats):
		l = nn.MSELoss(reduction='none')
		bcel = nn.BCELoss(reduction='mean')
		msel = nn.MSELoss(reduction='mean')
		real_label, fake_label = torch.tensor([0.9]), torch.tensor([0.1])
		real_label, fake_label = real_label.type(torch.DoubleTensor), fake_label.type(torch.DoubleTensor)
		n = epoch + 1
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

## Shared backprop logic for all TranAD variants
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


# Proposed Model (VLDB 22)
class TranAD_Basic(TranADBase, nn.Module):
	def __init__(self, feats, n_window=10, batch_size=128, dim_feedforward=16, dropout=0.1, nheads=None, learning_rate=None):
		super(TranAD_Basic, self).__init__()
		self.name = 'TranAD_Basic'
		self.lr = constants.lr if learning_rate is None else learning_rate
		self.batch = batch_size
		self.n_feats = feats
		self.n_window = n_window
		self.flat_window = False
		self.nheads = nheads if nheads is not None else feats
		self.n = self.n_feats * self.n_window
		self.pos_encoder = dlutils.PositionalEncoding(feats, 0.1, self.n_window)
		encoder_layers = TransformerEncoderLayer(d_model=feats, nhead=self.nheads, dim_feedforward=dim_feedforward, dropout=dropout)
		self.transformer_encoder = TransformerEncoder(encoder_layers, 1)
		decoder_layers = TransformerDecoderLayer(d_model=feats, nhead=self.nheads, dim_feedforward=dim_feedforward, dropout=dropout)
		self.transformer_decoder = TransformerDecoder(decoder_layers, 1)
		self.fcn = nn.Sigmoid()
		self.pos_encoder = self.pos_encoder.double()
		self.transformer_encoder = self.transformer_encoder.double()
		self.transformer_decoder = self.transformer_decoder.double()

	def forward(self, src, tgt):
		src = src * math.sqrt(self.n_feats)
		src = self.pos_encoder(src)
		memory = self.transformer_encoder(src)
		x = self.transformer_decoder(tgt, memory)
		x = self.fcn(x)
		return x

# Proposed Model (FCN) + Self Conditioning + Adversarial + MAML (VLDB 22)
class TranAD_Transformer(TranADBase, nn.Module):
	def __init__(self, feats, n_window=10, batch_size = 128, n_hidden=8, learning_rate=None):
		super(TranAD_Transformer, self).__init__()
		self.name = 'TranAD_Transformer'
		self.lr = constants.lr if learning_rate is None else learning_rate
		self.batch = batch_size
		self.n_feats = feats
		self.n_hidden = n_hidden
		self.n_window = n_window
		self.flat_window = False
		self.n = 2 * self.n_feats * self.n_window
		self.transformer_encoder = nn.Sequential(
			nn.Linear(self.n, self.n_hidden), nn.ReLU(True),
			nn.Linear(self.n_hidden, self.n), nn.ReLU(True))
		self.transformer_decoder1 = nn.Sequential(
			nn.Linear(self.n, self.n_hidden), nn.ReLU(True),
			nn.Linear(self.n_hidden, 2 * feats), nn.ReLU(True))
		self.transformer_decoder2 = nn.Sequential(
			nn.Linear(self.n, self.n_hidden), nn.ReLU(True),
			nn.Linear(self.n_hidden, 2 * feats), nn.ReLU(True))
		self.fcn = nn.Sequential(nn.Linear(2 * feats, feats), nn.Sigmoid())
		self.transformer_encoder = self.transformer_encoder.double()
		self.transformer_decoder1 = self.transformer_decoder1.double()
		self.transformer_decoder2 = self.transformer_decoder2.double()
		self.fcn = self.fcn.double()

	def encode(self, src, c, tgt):
		src = torch.cat((src, c), dim=2)
		src = src.permute(1, 0, 2).flatten(start_dim=1)
		tgt = self.transformer_encoder(src)
		return tgt

	def forward(self, src, tgt):
		# Phase 1 - Without anomaly scores
		c = torch.zeros_like(src)
		x1 = self.transformer_decoder1(self.encode(src, c, tgt))
		x1 = x1.reshape(-1, 1, 2*self.n_feats).permute(1, 0, 2)
		x1 = self.fcn(x1)
		# Phase 2 - With anomaly scores
		c = (x1 - src) ** 2
		x2 = self.transformer_decoder2(self.encode(src, c, tgt))
		x2 = x2.reshape(-1, 1, 2*self.n_feats).permute(1, 0, 2)
		x2 = self.fcn(x2)
		return x1, x2

# Proposed Model + Self Conditioning + MAML (VLDB 22)
class TranAD_Adversarial(TranADBase, nn.Module):
	def __init__(self, feats, n_window=10, batch_size = 128, dim_feedforward=16, dropout=0.1, nheads=None, learning_rate=None):
		super(TranAD_Adversarial, self).__init__()
		self.name = 'TranAD_Adversarial'
		self.lr = constants.lr if learning_rate is None else learning_rate
		self.batch = batch_size
		self.n_feats = feats
		self.n_window = n_window
		self.flat_window = False
		self.n = self.n_feats * self.n_window
		self.nheads = nheads if nheads is not None else feats
		self.pos_encoder = dlutils.PositionalEncoding(2 * feats, 0.1, self.n_window)
		encoder_layers = TransformerEncoderLayer(d_model=2 * feats, nhead=self.nheads, dim_feedforward=dim_feedforward, dropout=dropout)
		self.transformer_encoder = TransformerEncoder(encoder_layers, 1)
		decoder_layers = TransformerDecoderLayer(d_model=2 * feats, nhead=self.nheads, dim_feedforward=dim_feedforward, dropout=dropout)
		self.transformer_decoder = TransformerDecoder(decoder_layers, 1)
		self.fcn = nn.Sequential(nn.Linear(2 * feats, feats), nn.Sigmoid())
		self.pos_encoder = self.pos_encoder.double()
		self.transformer_encoder = self.transformer_encoder.double()
		self.transformer_decoder = self.transformer_decoder.double()
		self.fcn = self.fcn.double()

	def encode_decode(self, src, c, tgt):
		src = torch.cat((src, c), dim=2)
		src = src * math.sqrt(self.n_feats)
		src = self.pos_encoder(src)
		memory = self.transformer_encoder(src)
		tgt = tgt.repeat(1, 1, 2)
		x = self.transformer_decoder(tgt, memory)
		x = self.fcn(x)
		return x

	def forward(self, src, tgt):
		# Phase 1 - Without anomaly scores
		c = torch.zeros_like(src)
		x = self.encode_decode(src, c, tgt)
		# Phase 2 - With anomaly scores
		c = (x - src) ** 2
		x = self.encode_decode(src, c, tgt)
		return x

# Proposed Model + Adversarial + MAML (VLDB 22)
class TranAD_SelfConditioning(TranADBase, nn.Module):
	def __init__(self, feats, n_window=10, batch_size = 128, dim_feedforward=16, dropout=0.1, nheads=None, learning_rate=None):
		super(TranAD_SelfConditioning, self).__init__()
		self.name = 'TranAD_SelfConditioning'
		self.lr = constants.lr if learning_rate is None else learning_rate
		self.batch = batch_size
		self.n_feats = feats
		self.n_window = n_window
		self.flat_window = False
		self.n = self.n_feats * self.n_window
		self.nheads = nheads if nheads is not None else feats
		self.pos_encoder = dlutils.PositionalEncoding(2 * feats, 0.1, self.n_window)
		encoder_layers = TransformerEncoderLayer(d_model=2 * feats, nhead=self.nheads, dim_feedforward=dim_feedforward, dropout=dropout)
		self.transformer_encoder = TransformerEncoder(encoder_layers, 1)
		decoder_layers1 = TransformerDecoderLayer(d_model=2 * feats, nhead=self.nheads, dim_feedforward=dim_feedforward, dropout=dropout)
		self.transformer_decoder1 = TransformerDecoder(decoder_layers1, 1)
		decoder_layers2 = TransformerDecoderLayer(d_model=2 * feats, nhead=self.nheads, dim_feedforward=dim_feedforward, dropout=dropout)
		self.transformer_decoder2 = TransformerDecoder(decoder_layers2, 1)
		self.fcn = nn.Sequential(nn.Linear(2 * feats, feats), nn.Sigmoid())
		self.pos_encoder = self.pos_encoder.double()
		self.transformer_encoder = self.transformer_encoder.double()
		self.transformer_decoder1 = self.transformer_decoder1.double()
		self.transformer_decoder2 = self.transformer_decoder2.double()
		self.fcn = self.fcn.double()

	def encode(self, src, c, tgt):
		src = torch.cat((src, c), dim=2)
		src = src * math.sqrt(self.n_feats)
		src = self.pos_encoder(src)
		memory = self.transformer_encoder(src)
		tgt = tgt.repeat(1, 1, 2)
		return tgt, memory

	def forward(self, src, tgt):
		# Phase 1 - Without anomaly scores
		c = torch.zeros_like(src)
		x1 = self.fcn(self.transformer_decoder1(*self.encode(src, c, tgt)))
		# Phase 2 - With anomaly scores
		x2 = self.fcn(self.transformer_decoder2(*self.encode(src, c, tgt)))
		return x1, x2

# Proposed Model + Self Conditioning + Adversarial + MAML (VLDB 22)
class TranAD(TranADBase, nn.Module):
	def __init__(self, feats, n_window=10, batch_size = 128, dim_feedforward=16, dropout=0.1, nheads=None, learning_rate=None):
		super(TranAD, self).__init__()
		self.name = 'TranAD'
		self.lr = constants.lr if learning_rate is None else learning_rate
		self.batch = batch_size
		self.n_feats = feats
		self.n_window = n_window
		self.flat_window = False
		self.n = self.n_feats * self.n_window
		self.nheads = nheads if nheads is not None else feats
		self.pos_encoder = dlutils.PositionalEncoding(2 * feats, 0.1, self.n_window)
		encoder_layers = TransformerEncoderLayer(d_model=2 * feats, nhead=self.nheads, dim_feedforward=dim_feedforward, dropout=dropout)
		self.transformer_encoder = TransformerEncoder(encoder_layers, 1)
		decoder_layers1 = TransformerDecoderLayer(d_model=2 * feats, nhead=self.nheads, dim_feedforward=dim_feedforward, dropout=dropout)
		self.transformer_decoder1 = TransformerDecoder(decoder_layers1, 1)
		decoder_layers2 = TransformerDecoderLayer(d_model=2 * feats, nhead=self.nheads, dim_feedforward=dim_feedforward, dropout=dropout)
		self.transformer_decoder2 = TransformerDecoder(decoder_layers2, 1)
		self.fcn = nn.Sequential(nn.Linear(2 * feats, feats), nn.Sigmoid())
		self.pos_encoder = self.pos_encoder.double()
		self.transformer_encoder = self.transformer_encoder.double()
		self.transformer_decoder1 = self.transformer_decoder1.double()
		self.transformer_decoder2 = self.transformer_decoder2.double()
		self.fcn = self.fcn.double()

	def encode(self, src, c, tgt):
		src = torch.cat((src, c), dim=2)
		src = src * math.sqrt(self.n_feats)
		src = self.pos_encoder(src)
		memory = self.transformer_encoder(src)
		tgt = tgt.repeat(1, 1, 2)
		return tgt, memory

	def forward(self, src, tgt):
		# Phase 1 - Without anomaly scores
		c = torch.zeros_like(src)
		x1 = self.fcn(self.transformer_decoder1(*self.encode(src, c, tgt)))
		# Phase 2 - With anomaly scores
		c = (x1 - src) ** 2
		x2 = self.fcn(self.transformer_decoder2(*self.encode(src, c, tgt)))
		return x1, x2



class STAGNN(nn.Module):
	def __init__(self, feats, n_window=10, batch_size=128, embed_dim=64, topk=5, dropout=0.1, graph_num_heads=4, learning_rate=0.001):
		super(STAGNN, self).__init__()
		self.name = 'STAGNN'
		self.lr = learning_rate
		self.n_feats = feats
		self.n_window = n_window
		self.flat_window = False
		self.batch = batch_size
		self.model = STAGNN_Core(
			num_sensors=feats,
			embed_dim=embed_dim,
			window_size=n_window,
			topk=topk,
			dropout=dropout,
			feature_dim=1,
			graph_num_heads=graph_num_heads,
		)
		self.model = self.model.double()

	def forward(self, x):
		out, alpha = self.model(x)
		out = out.squeeze(-1)
		return out, alpha

	def train_step(self, epoch, data, optimizer, scheduler, feats):
		l = nn.MSELoss(reduction='none')
		data_x = torch.DoubleTensor(data)
		dataset = TensorDataset(data_x, data_x)
		bs = self.batch
		dataloader = DataLoader(dataset, batch_size=bs)
		l1s = []
		for d, _ in dataloader:
			target = d[:, -1, :]
			out, alpha = self(d)
			loss = torch.mean(l(out, target))
			l1s.append(loss.item())
			optimizer.zero_grad()
			loss.backward()
			torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
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
			target = d[:, -1, :]
			out, alpha = self(d)
		loss = l(out, target)
		return loss.detach().numpy(), out.detach().numpy()

