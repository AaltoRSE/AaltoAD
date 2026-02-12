import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import GATConv
from torch.nn import TransformerEncoder
from torch.nn import TransformerDecoder
from torch.nn import TransformerEncoderLayer, TransformerDecoderLayer
import numpy as np
import math
from TranAD import dlutils
from TranAD import constants
torch.manual_seed(1)

## Separate LSTM for each variable
class LSTM_Univariate(nn.Module):
	def __init__(self, feats, n_hidden=1, n_layers=1, learning_rate=0.002):
		super(LSTM_Univariate, self).__init__()
		self.name = 'LSTM_Univariate'
		self.lr = learning_rate
		self.n_feats = feats
		self.n_hidden = n_hidden
		self.n_layers = n_layers
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
				multivariate_output.append(2 * out.view(-1))
			output = torch.cat(multivariate_output)
			outputs.append(output)
		return torch.stack(outputs)

## Simple Multi-Head Self-Attention Model
class Attention(nn.Module):
	def __init__(self, feats, n_window=5, learning_rate=0.0001):
		super(Attention, self).__init__()
		self.name = 'Attention'
		self.lr = learning_rate
		self.n_feats = feats
		self.n_window = n_window
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

## LSTM_AD Model
class LSTM_AD(nn.Module):
	def __init__(self, feats, n_hidden=64, n_layers=1, learning_rate=0.002):
		super(LSTM_AD, self).__init__()
		self.name = 'LSTM_AD'
		self.lr = learning_rate
		self.n_feats = feats
		self.n_hidden = n_hidden
		self.n_layers = n_layers
		self.lstm = nn.LSTM(feats, self.n_hidden, n_layers)
		self.fcn = nn.Sequential(nn.Linear(self.n_hidden, self.n_feats), nn.Sigmoid())
		self.lstm = self.lstm.double()
		self.fcn = self.fcn.double()

	def forward(self, x):
		hidden = (torch.rand(self.n_layers, 1, self.n_hidden, dtype=torch.float64), torch.randn(self.n_layers, 1, self.n_hidden, dtype=torch.float64))
		outputs = []
		for i, g in enumerate(x):
			out, hidden = self.lstm(g.view(1, 1, -1), hidden)
			out = self.fcn(out.view(-1))
			outputs.append(2 * out.view(-1))
		return torch.stack(outputs)

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
		relative_euclidean_distance = (x-x_hat).norm(2, dim=1) / x.norm(2, dim=1)
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

## MSCRED Model (AAAI 19)
class MSCRED(nn.Module):
	def __init__(self, feats, n_window=None, learning_rate=0.0001):
		super(MSCRED, self).__init__()
		self.name = 'MSCRED'
		self.lr = learning_rate
		self.n_feats = feats
		self.n_window = feats if n_window is None else n_window
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

## CAE-M Model (TKDE 21)
class CAE_M(nn.Module):
	def __init__(self, feats, n_window=None, learning_rate=0.0001):
		super(CAE_M, self).__init__()
		self.name = 'CAE_M'
		self.lr = learning_rate
		self.n_feats = feats
		self.n_window = feats if n_window is None else n_window
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

## MTAD_GAT Model (ICDM 20)
class MTAD_GAT(nn.Module):
	def __init__(self, feats, n_window=None, n_hidden=None, learning_rate=0.0001):
		super(MTAD_GAT, self).__init__()
		self.name = 'MTAD_GAT'
		self.lr = learning_rate
		self.n_feats = feats
		self.n_window = feats if n_window is None else n_window
		self.n_hidden = feats * feats if n_hidden is None else n_hidden
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
		hidden = torch.rand(1, 1, self.n_hidden, dtype=torch.float64) if hidden is not None else hidden
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

## GDN Model (AAAI 21)
class GDN(nn.Module):
	def __init__(self, feats, n_window=5, n_hidden=16, learning_rate=0.0001):
		super(GDN, self).__init__()
		self.name = 'GDN'
		self.lr = learning_rate
		self.n_feats = feats
		self.n_window = n_window
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

# MAD_GAN (ICANN 19)
class MAD_GAN(nn.Module):
	def __init__(self, feats, n_window=5, n_hidden=16, learning_rate=0.0001):
		super(MAD_GAN, self).__init__()
		self.name = 'MAD_GAN'
		self.lr = learning_rate
		self.n_feats = feats
		self.n_hidden = n_hidden
		self.n_window = n_window
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

# Proposed Model (VLDB 22)
class TranAD_Basic(nn.Module):
	def __init__(self, feats, n_window=10, batch_size=128, dim_feedforward=16, dropout=0.1, nheads=None, learning_rate=None):
		super(TranAD_Basic, self).__init__()
		self.name = 'TranAD_Basic'
		self.lr = constants.lr if learning_rate is None else learning_rate
		self.batch = batch_size
		self.n_feats = feats
		self.n_window = n_window
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
class TranAD_Transformer(nn.Module):
	def __init__(self, feats, n_window=10, batch_size = 128, n_hidden=8, learning_rate=None):
		super(TranAD_Transformer, self).__init__()
		self.name = 'TranAD_Transformer'
		self.lr = constants.lr if learning_rate is None else learning_rate
		self.batch = batch_size
		self.n_feats = feats
		self.n_hidden = n_hidden
		self.n_window = n_window
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
class TranAD_Adversarial(nn.Module):
	def __init__(self, feats, n_window=10, batch_size = 128, dim_feedforward=16, dropout=0.1, nheads=None, learning_rate=None):
		super(TranAD_Adversarial, self).__init__()
		self.name = 'TranAD_Adversarial'
		self.lr = constants.lr if learning_rate is None else learning_rate
		self.batch = batch_size
		self.n_feats = feats
		self.n_window = n_window
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
class TranAD_SelfConditioning(nn.Module):
	def __init__(self, feats, n_window=10, batch_size = 128, dim_feedforward=16, dropout=0.1, nheads=None, learning_rate=None):
		super(TranAD_SelfConditioning, self).__init__()
		self.name = 'TranAD_SelfConditioning'
		self.lr = constants.lr if learning_rate is None else learning_rate
		self.batch = batch_size
		self.n_feats = feats
		self.n_window = n_window
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
class TranAD(nn.Module):
	def __init__(self, feats, n_window=10, batch_size = 128, dim_feedforward=16, dropout=0.1, nheads=None, learning_rate=None):
		super(TranAD, self).__init__()
		self.name = 'TranAD'
		self.lr = constants.lr if learning_rate is None else learning_rate
		self.batch = batch_size
		self.n_feats = feats
		self.n_window = n_window
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
