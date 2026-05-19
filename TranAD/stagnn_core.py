import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class STAGNN_Core(nn.Module):
	def __init__(
		self,
		num_sensors: int,
		embed_dim: int = 512,
		window_size: int = 5,
		topk: int = 10,
		dropout: float = 0.1,
		use_soft_topk: bool = True,
		feature_dim: int = 1,
		use_learned_static_if_none: bool = True,
		normalize_external_static: bool = True,
		graph_num_heads: int = 8,
		hard_mask_from_static: bool = True,
		topk_mask_fill: float = -5.0,
		static_prior_scale: float = 0.693147,
	):
		super().__init__()
		self.N = num_sensors
		self.d = embed_dim
		self.w = window_size
		self.k = topk
		self.use_soft_topk = use_soft_topk
		self.F = feature_dim
		self.use_learned_static_if_none = use_learned_static_if_none
		self.normalize_external_static = normalize_external_static
		self.hard_mask_from_static = hard_mask_from_static
		self.topk_mask_fill = float(topk_mask_fill)
		self.sim_scale_raw = nn.Parameter( torch.tensor(math.log(math.expm1(static_prior_scale))) )
		h = min(graph_num_heads, self.d)
		while h > 1 and (self.d % h) != 0:
			h -= 1
		self.graph_num_heads = max(1, h)
		self.d_head = self.d // self.graph_num_heads

		self.sensor_emb = nn.Parameter(torch.randn(self.N, self.d))
		nn.init.xavier_uniform_(self.sensor_emb)

		self.temp = nn.Parameter(torch.tensor(1.0))

		self.pos_emb = nn.Parameter(torch.randn(self.w, self.d))
		self.temporal_in = nn.Linear(self.F, self.d, bias=False)

		self.temporal_attn = nn.MultiheadAttention(
			embed_dim=self.d,
			num_heads=min(8, self.d),
			batch_first=False,
			dropout=dropout
		)
		self.temporal_ln = nn.LayerNorm(self.d)

		self.q_proj = nn.Linear(self.d, self.d, bias=False)
		self.k_proj = nn.Linear(self.d, self.d, bias=False)
		self.v_proj = nn.Linear(self.d, self.d, bias=False)

		self.post_ln = nn.LayerNorm(self.d)
		self.post_dropout = nn.Dropout(dropout)

		self.mlp = nn.Sequential(
			nn.Linear(self.d, self.d),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(self.d, self.F)
		)

	def _normalize_static(self, A: torch.Tensor) -> torch.Tensor:
		A = A.clone()
		A.fill_diagonal_(0.0)

		if not self.normalize_external_static:
			return A

		nonzero = (A != 0)
		if nonzero.any():
			mean = A[nonzero].mean()
			std = A[nonzero].std()
		else:
			mean = A.mean()
			std = A.std()

		std = std.clamp_min(1e-6)
		return (A - mean) / std

	def _forward_internal(self, X: torch.Tensor, A_static: torch.Tensor | None = None):
		if X.dim() == 3:
			X = X.unsqueeze(-1)

		B, w, N, Fdim = X.shape
		if not (w == self.w and N == self.N and Fdim == self.F):
			raise ValueError(
				f"Bad input shape. Expected (B,{self.w},{self.N},{self.F}) "
				f"or (B,{self.w},{self.N}), got {tuple(X.shape)}"
			)

		X_ = X.permute(0, 2, 1, 3).contiguous().view(B * N, w, self.F)
		tokens = self.temporal_in(X_) + self.pos_emb.unsqueeze(0)
		tokens = tokens.permute(1, 0, 2)

		attn_out, _ = self.temporal_attn(tokens, tokens, tokens, need_weights=False)
		H = self.temporal_ln(attn_out.mean(dim=0)).view(B, N, self.d)

		H_flat = F.normalize(H.reshape(B * N, self.d), dim=1).view(B, N, self.d)
		ctx_sim = torch.matmul(H_flat, H_flat.transpose(1, 2))

		if A_static is not None:
			static_sim = A_static.to(H.device).float()
			if static_sim.shape != (self.N, self.N):
				raise ValueError(f"A_static must be shape ({self.N},{self.N}), got {tuple(static_sim.shape)}")
			static_sim = self._normalize_static(static_sim)
		else:
			if self.use_learned_static_if_none:
				emb_norm = F.normalize(self.sensor_emb, dim=1)
				static_sim = torch.matmul(emb_norm, emb_norm.t())
			else:
				static_sim = torch.zeros(self.N, self.N, device=H.device)

		static_sim = static_sim - torch.eye(self.N, device=H.device) * 1e9

		sim_scale = F.softplus(self.sim_scale_raw)

		prior = ctx_sim + sim_scale * static_sim.unsqueeze(0)

		h = self.graph_num_heads
		d_head = self.d_head

		Q = self.q_proj(H).view(B, N, h, d_head).transpose(1, 2)
		K = self.k_proj(H).view(B, N, h, d_head).transpose(1, 2)
		V = self.v_proj(H).view(B, N, h, d_head).transpose(1, 2)

		scores = torch.matmul(Q, K.transpose(-2, -1)) / (
			math.sqrt(d_head) * self.temp.clamp_min(1e-3)
		)

		scores = scores + prior.unsqueeze(1)

		eye = torch.eye(N, device=scores.device).view(1, 1, N, N)
		scores = scores.masked_fill(eye.bool(), float("-inf"))

		if self.hard_mask_from_static and (A_static is not None):
			allowed = (A_static.to(scores.device).float() != 0).view(1, 1, N, N)
			scores = scores.masked_fill(~allowed, float("-inf"))

		if self.k is not None:
			k_eff = min(self.k, N - 1)
			topk_val, topk_idx = torch.topk(scores, k_eff, dim=-1)
			mask = torch.full_like(scores, fill_value=self.topk_mask_fill)
			mask.scatter_(dim=-1, index=topk_idx, src=torch.zeros_like(topk_val))
			scores = scores + mask

		alpha_heads = F.softmax(scores, dim=-1)
		head_out = torch.matmul(alpha_heads, V)
		neighbor_values = head_out.transpose(1, 2).reshape(B, N, self.d)

		neighbor_values = neighbor_values + 0.1 * H
		neighbor_values = self.post_ln(neighbor_values)
		neighbor_values = self.post_dropout(neighbor_values)

		out = self.mlp(neighbor_values)

		alpha = alpha_heads.mean(dim=1)
		ctx = neighbor_values
		return out, alpha, ctx

	def forward(self, X: torch.Tensor, A_static: torch.Tensor | None = None):
		out, alpha, _ = self._forward_internal(X, A_static=A_static)
		return out, alpha

	def forward_with_ctx(self, X: torch.Tensor, A_static: torch.Tensor | None = None):
		return self._forward_internal(X, A_static=A_static)
