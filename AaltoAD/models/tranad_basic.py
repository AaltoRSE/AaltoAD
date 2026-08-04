import torch.nn as nn
import math
from torch.nn import TransformerEncoder, TransformerDecoder
from torch.nn import TransformerEncoderLayer, TransformerDecoderLayer
from AaltoAD import dlutils
from .base import BaseModel
from .tranad_base import TranADBase


# Proposed Model (VLDB 22)
class TranAD_Basic(TranADBase, BaseModel):
	def __init__(self, feats, n_window=10, batch_size=128, dim_feedforward=16, dropout=0.1, nheads=None, learning_rate=0.0001, epochs=5, weight_decay=1e-5):
		super().__init__(feats, learning_rate=learning_rate, epochs=epochs, weight_decay=weight_decay)
		self.name = 'TranAD_Basic'
		self.batch = batch_size
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
