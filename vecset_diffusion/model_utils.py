import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from torch_cluster import fps
import numpy as np

try:
	from flash_attn import flash_attn_kvpacked_func
	FLASH_ATTN_AVAILABLE = True
except ImportError:
	FLASH_ATTN_AVAILABLE = False
	flash_attn_kvpacked_func = None

def get_attention_class():
	sm_family = torch.cuda.get_device_capability('cuda')[0]
	if sm_family >= 8 and FLASH_ATTN_AVAILABLE:
		return FlashAttention
	else:
		if sm_family >= 8 and not FLASH_ATTN_AVAILABLE:
			print('Flash Attention not installed. Install with: pip install flash-attn --no-build-isolation')
		elif sm_family < 8:
			print('Compute capability < 8, disabling Flash Attention.')
		return SimpleAttention

def exists(val):
	return val is not None

def default(val, d):
	return val if exists(val) else d




class SimpleAttention(nn.Module):
	def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, drop_path_rate=0.0, with_bias=True):
		super().__init__()
		inner_dim = dim_head * heads
		context_dim = default(context_dim, query_dim)
		self.scale = dim_head ** -0.5
		self.heads = heads

		self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
		self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
		self.to_out = nn.Linear(inner_dim, query_dim, bias=with_bias)

	def forward(self, x, context=None, mask=None):
		h = self.heads

		q = self.to_q(x)
		context = default(context, x)
		k, v = self.to_kv(context).chunk(2, dim = -1)

		q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

		sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

		if exists(mask):
			mask = rearrange(mask, 'b ... -> b (...)')
			max_neg_value = -torch.finfo(sim.dtype).max
			mask = repeat(mask, 'b j -> (b h) () j', h = h)
			sim.masked_fill_(~mask, max_neg_value)

		# attention, what we cannot get enough of
		attn = sim.softmax(dim=-1)

		out = einsum('b i j, b j d -> b i d', attn, v)
		out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
		return self.to_out(out)

class FlashAttention(nn.Module):
	def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, with_bias=True):
		super().__init__()
		inner_dim = dim_head * heads
		context_dim = default(context_dim, query_dim)
		self.scale = dim_head ** -0.5
		self.heads = heads

		self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
		self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
		self.to_out = nn.Linear(inner_dim, query_dim, bias=with_bias)

	def forward(self, x, context=None, mask= None, window_size=-1):
		h = self.heads

		q = self.to_q(x)
		context = default(context, x)
		kv = self.to_kv(context)

		q = rearrange(q, 'b n (h d) -> b n h d', h = h)
		kv = rearrange(kv, 'b n (p h d) -> b n p h d', h = h, p=2)

		out = flash_attn_kvpacked_func(q.bfloat16(), kv.bfloat16(), window_size=(window_size, window_size))
		out = out.to(x.dtype)

		return self.to_out(rearrange(out, 'b n h d -> b n (h d)'))

class PointEmbed(nn.Module):
    def __init__(self, hidden_dim=48, dim=128):
        super().__init__()

        assert hidden_dim % 6 == 0

        self.embedding_dim = hidden_dim
        e = torch.pow(2, torch.arange(self.embedding_dim // 6)).float() * np.pi
        e = torch.stack([
            torch.cat([e, torch.zeros(self.embedding_dim // 6),
                        torch.zeros(self.embedding_dim // 6)]),
            torch.cat([torch.zeros(self.embedding_dim // 6), e,
                        torch.zeros(self.embedding_dim // 6)]),
            torch.cat([torch.zeros(self.embedding_dim // 6),
                        torch.zeros(self.embedding_dim // 6), e]),
        ])
        self.register_buffer('basis', e)  # 3 x 16

        self.mlp = nn.Linear(self.embedding_dim+3, dim)

    @staticmethod
    def embed(input, basis):
        projections = torch.einsum(
            'bnd,de->bne', input, basis)
        embeddings = torch.cat([projections.sin(), projections.cos()], dim=2)
        return embeddings

    def forward(self, input):
        # input: B x N x 3
        embed = self.mlp(torch.cat([self.embed(input, self.basis), input], dim=2)) # B x N x C
        return embed

def subsample(pc, N, M):
    # pc: B x N x 3
    B, N0, D = pc.shape
    assert N == N0

    ###### fps
    flattened = pc.view(B*N, D)

    batch = torch.arange(B).to(pc.device)
    batch = torch.repeat_interleave(batch, N)

    pos = flattened

    ratio = 1.0 * M / N

    idx = fps(pos, batch, ratio=ratio)

    sampled_pc = pos[idx]
    sampled_pc = sampled_pc.view(B, -1, 3)
    ######

    return sampled_pc
