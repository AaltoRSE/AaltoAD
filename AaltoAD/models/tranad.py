import torch
import torch.nn as nn
import math
from torch.nn import TransformerEncoder, TransformerDecoder
from torch.nn import TransformerEncoderLayer, TransformerDecoderLayer
from AaltoAD import dlutils
from .base import BaseModel
from .tranad_base import TranADBase


# Proposed Model + Self Conditioning + Adversarial + MAML (VLDB 22)
class TranAD(TranADBase, BaseModel):
	def __init__(self, feats, n_window=10, batch_size=128, dim_feedforward=16, dropout=0.1, nheads=None, learning_rate=0.0001, epochs=5, weight_decay=1e-5):
		super().__init__(feats, learning_rate=learning_rate, epochs=epochs, weight_decay=weight_decay)
		self.name = 'TranAD'
		self.batch = batch_size
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
		c = torch.zeros_like(src)
		x1 = self.fcn(self.transformer_decoder1(*self.encode(src, c, tgt)))
		c = (x1 - src) ** 2
		x2 = self.fcn(self.transformer_decoder2(*self.encode(src, c, tgt)))
		return x1, x2
