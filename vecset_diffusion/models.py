# adapted from https://github.com/1zb/3DShape2VecSet/blob/master/models_class_cond.py

from vecset_diffusion.model_utils import get_attention_class

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

import numpy as np

CrossAttention = get_attention_class()

def zero_module(module):
	"""
	Zero out the parameters of a module and return it.
	"""
	for p in module.parameters():
		p.detach().zero_()
	return module

class PositionalEmbedding(torch.nn.Module):
	def __init__(self, num_channels, max_positions=10000, endpoint=False):
		super().__init__()
		self.num_channels = num_channels
		self.max_positions = max_positions
		self.endpoint = endpoint

	def forward(self, x):
		freqs = torch.arange(start=0, end=self.num_channels//2, dtype=torch.float32, device=x.device)
		freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
		freqs = (1 / self.max_positions) ** freqs
		x = x.ger(freqs.to(x.dtype))
		x = torch.cat([x.cos(), x.sin()], dim=1)
		return x

class FixedPositionalEmbedding(torch.nn.Module):
	def __init__(self, dim):
		super().__init__()
		inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
		self.register_buffer('inv_freq', inv_freq)

	def forward(self, x, seq_dim=1, offset=0, indices=None, sequence_length=None):
		if sequence_length is None:
			sequence_length = x.shape[seq_dim]
		t = torch.arange(sequence_length, device=x.device).type_as(self.inv_freq) + offset
		if indices is not None:
			t = t[indices]
		sinusoid_inp = torch.einsum('i , j -> i j', t, self.inv_freq)
		emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
		return emb[None, :, :]

class LayerScale(nn.Module):
	def __init__(self, dim, init_values=1e-5, inplace=False):
		super().__init__()
		self.inplace = inplace
		self.gamma = nn.Parameter(init_values * torch.ones(dim))

	def forward(self, x):
		return x.mul_(self.gamma) if self.inplace else x * self.gamma

class GEGLU(nn.Module):
	def __init__(self, dim_in, dim_out, with_bias=True):
		super().__init__()
		self.proj = nn.Linear(dim_in, dim_out * 2, bias=with_bias)

	def forward(self, x):
		x, gate = self.proj(x).chunk(2, dim=-1)
		return x * F.gelu(gate)


class FeedForward(nn.Module):
	def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0., with_bias=True):
		super().__init__()
		inner_dim = int(dim * mult)
		if dim_out is None:
			dim_out = dim

		project_in = nn.Sequential(
			nn.Linear(dim, inner_dim, with_bias=with_bias),
			nn.GELU()
		) if not glu else GEGLU(dim, inner_dim, with_bias=with_bias)

		self.net = nn.Sequential(
			project_in,
			nn.Dropout(dropout),
			nn.Linear(inner_dim, dim_out, bias=with_bias)
		)

	def forward(self, x):
		return self.net(x)

class AdaLayerNorm(nn.Module):
	def __init__(self, n_embd, with_bias=True):
		super().__init__()

		self.with_bias = with_bias
		self.silu = nn.SiLU()
		self.linear = nn.Linear(
			n_embd,
			n_embd*2 if with_bias else n_embd,
			bias=with_bias,
		)
		self.layernorm = nn.LayerNorm(n_embd, elementwise_affine=False)

	def forward(self, x, timestep):
		emb = self.linear(timestep)
		if self.with_bias:
			scale, shift = torch.chunk(emb, 2, dim=2)
			x = self.layernorm(x) * (1 + scale) + shift
		else:
			x = self.layernorm(x) * (1 + emb)
		return x

class BasicTransformerBlock(nn.Module):
	def __init__(
			self,
			dim,
			n_heads,
			d_head,
			dropout=0.,
			context_dim=None,
			gated_ff=True,
			with_cross_attention=True,
			with_bias=True,
		):
		super().__init__()
		self.with_cross_attention = with_cross_attention

		self.attn1 = CrossAttention(
			query_dim=dim,
			heads=n_heads,
			dim_head=d_head,
			with_bias=with_bias,
		)  # self-attention
		self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff, with_bias=with_bias)
		self.norm1 = AdaLayerNorm(dim, with_bias=with_bias)
		self.norm3 = AdaLayerNorm(dim, with_bias=with_bias)

		if self.with_cross_attention:
			 # cross attention with class label
			self.norm2 = AdaLayerNorm(dim, with_bias=with_bias)
			self.attn2 = CrossAttention(
				query_dim=dim,
				context_dim=context_dim,
				heads=n_heads,
				dim_head=d_head,
				with_bias=with_bias,
			)

	def forward(self, x, t, context=None):
		x = self.attn1(self.norm1(x, t)) + x
		if self.with_cross_attention:
			x = self.attn2(self.norm2(x, t), context=context) + x
		x = self.ff(self.norm3(x, t)) + x
		return x

class LatentArrayTransformer(nn.Module):
	"""
	Transformer block for image-like data.
	First, project the input (aka embedding)
	and reshape to b, t, d.
	Then apply standard transformer action.
	Finally, reshape to image
	"""

	def __init__(
			self,
			in_channels,
			t_channels,
			n_heads,
			d_head,
			depth=1,
			dropout=0.,
			context_dim=None,
			out_channels=None,
			use_pos_embedding=False,
			with_bias=True,
			output_norm=True,
		):
		super().__init__()
		self.in_channels = in_channels
		inner_dim = n_heads * d_head

		self.t_channels = t_channels

		self.proj_in = nn.Linear(in_channels, inner_dim, bias=False)

		self.transformer_blocks = nn.ModuleList([
			BasicTransformerBlock(
				dim=inner_dim,
				n_heads=n_heads,
				d_head=d_head,
				dropout=dropout,
				context_dim=context_dim,
				with_bias=with_bias,
			)
			for _ in range(depth)
		])

		self.norm = nn.LayerNorm(inner_dim, bias=with_bias) if output_norm else nn.Identity()

		if out_channels is None:
			self.proj_out = zero_module(nn.Linear(inner_dim, in_channels, bias=with_bias))
		else:
			self.num_cls = out_channels
			self.proj_out = zero_module(nn.Linear(inner_dim, out_channels, bias=with_bias))

		self.context_dim = context_dim

		self.map_noise = PositionalEmbedding(t_channels)

		self.map_layer0 = nn.Linear(in_features=t_channels, out_features=inner_dim, bias=with_bias)
		self.map_layer1 = nn.Linear(in_features=inner_dim, out_features=inner_dim, bias=with_bias)


		# ###
		self.pos_emb = FixedPositionalEmbedding(inner_dim) if use_pos_embedding else None
		# ###

	def forward(self, x, t, indices=None, sequence_length=None, cond=None):

		t_emb = self.map_noise(t)[:, None]
		t_emb = F.silu(self.map_layer0(t_emb))
		t_emb = F.silu(self.map_layer1(t_emb))
		x = self.proj_in(x)

		###
		if self.pos_emb is not None:
			x = x + self.pos_emb(x, indices=indices, sequence_length=sequence_length)
		###

		for block in self.transformer_blocks:
			x = block(x, t_emb, context=cond)

		x = self.norm(x)

		x = self.proj_out(x)
		return x

def edm_sampler(
	net, latents, class_labels=None, randn_like=torch.randn_like,
	num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
	# S_churn=40, S_min=0.05, S_max=50, S_noise=1.003,
	S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
):
	# Time step discretization.
	step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
	t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
	t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

	# Main sampling loop.
	x_next = latents.to(torch.float64) * t_steps[0]
	for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
		x_cur = x_next

		# Increase noise temporarily.
		gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
		t_hat = net.round_sigma(t_cur + gamma * t_cur)
		x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

		# Euler step.
		denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
		d_cur = (x_hat - denoised) / t_hat
		x_next = x_hat + (t_next - t_hat) * d_cur

		# Apply 2nd order correction.
		if i < num_steps - 1:
			denoised = net(x_next, t_next, class_labels).to(torch.float64)
			d_prime = (x_next - denoised) / t_next
			x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

	return x_next

@dataclass
class ModelConfig:
	in_channels: int = 8
	sigma_data: float = 1
	n_heads: int = 8
	d_head: int = 64
	depth: int = 12
	use_pos_embedding: bool = True
	with_cross_attention: bool = True
	with_bias: bool = True
	output_norm: bool = True
	dropout: float = 0.

class EDMPrecond(torch.nn.Module):
	def __init__(self, config: ModelConfig):
		super().__init__()
		self.config = config

		self.model = LatentArrayTransformer(
			in_channels=self.config.in_channels,
			t_channels=256,
			n_heads=self.config.n_heads,
			d_head=self.config.d_head,
			depth=self.config.depth,
			use_pos_embedding=self.config.use_pos_embedding,
			with_bias=self.config.with_bias,
			output_norm=config.output_norm,
			dropout=self.config.dropout,
		)

		if self.config.with_cross_attention:
			self.category_emb = nn.Embedding(55, self.config.n_heads * self.config.d_head)

	def emb_category(self, class_labels):
		if not self.config.with_cross_attention:
			return None

		if class_labels.dtype == torch.float32:
			return class_labels
		else:
			return self.category_emb(class_labels).unsqueeze(1)


	def forward(self, x, sigma, class_labels=None, sigma_data=1.0, **model_kwargs):
		cond_emb = self.emb_category(class_labels)

		sigma = sigma.to(x.dtype).reshape(-1, 1, 1)

		c_skip = sigma_data ** 2 / (sigma ** 2 + sigma_data ** 2)
		c_out = sigma * sigma_data / (sigma ** 2 + sigma_data ** 2).sqrt()
		c_in = 1 / (sigma_data ** 2 + sigma ** 2).sqrt()
		c_noise = sigma.log() / 4

		F_x = self.model((c_in * x), c_noise.flatten(), cond=cond_emb, **model_kwargs)
		D_x = c_skip * x + c_out * F_x
		return D_x, F_x

	def round_sigma(self, sigma):
		return torch.as_tensor(sigma)



class NoisePredictor(EDMPrecond):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)


	def forward(self, x, sigma, class_labels=None, sigma_data=1.0, **model_kwargs):
		cond_emb = self.emb_category(class_labels)

		sigma = sigma.to(x.dtype).reshape(-1, 1, 1)

		c_skip = sigma / (1 + sigma ** 2)
		c_in = 1 / (sigma_data ** 2 + sigma ** 2).sqrt()
		c_out = 1 / (sigma_data ** 2 + sigma ** 2).sqrt()
		c_noise = sigma.log() / 4

		model_output = self.model(c_in * x, c_noise.flatten(), cond=cond_emb, **model_kwargs)
		F_x = c_out * model_output + c_skip * x
		return F_x, model_output
