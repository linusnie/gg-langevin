# Adapted from https://github.com/1zb/VecSetX/blob/master/vecset/models/autoencoder.py

import torch
import torch.nn as nn

from einops import repeat
import torch.nn.functional as F
import math
from dataclasses import dataclass

from vecset_diffusion import model_utils

def exists(val):
    return val is not None

class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim=None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context=normed_context)

        return self.fn(x, **kwargs)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)

class VecSetAutoEncoder(nn.Module):
    def __init__(
        self,
        *,
        depth=24,
        dim=768,
        output_dim=1,
        num_inputs=2048,
        num_latents=1280,
        dim_head=64,
        query_type='point',
        encoder_mode='encode',
        latent_scale=1.0,
        latent_offset=None,
        bottleneck=None,
        final_layer_norm=True, # for backwards compatibility
        context_norm=False, # for backwards compatibility
        single_cross_attend_head=False,
        bottleneck_layer=0, # number of self attention layers to apply before bottleneck
        bottleneck_args={},
        attention_class=None,
    ):
        super().__init__()
        if attention_class is None:
            Attention = model_utils.get_attention_class()
        elif attention_class == 'flash':
            Attention = model_utils.FlashAttention
        elif attention_class == 'simple':
            Attention = model_utils.SimpleAttention
        else:
            raise ValueError(f'Attention class {attention_class} not recognized.')
        print(f'{Attention=}')

        queries_dim = dim

        self.depth = depth
        self.bottleneck_layer = bottleneck_layer

        self.num_inputs = num_inputs
        self.num_latents = num_latents

        self.query_type = query_type
        if query_type == 'point':
            pass
        elif query_type == 'learnable':
            self.latents = nn.Embedding(num_latents, dim)
        else:
            raise NotImplementedError(f'Query type {query_type} not implemented')

        self.encoder_mode = encoder_mode
        if isinstance(encoder_mode, int) or encoder_mode.isdigit():
            self.encoder_layers = int(encoder_mode)
        elif encoder_mode == 'encode':
            self.encoder_layers = 0
        elif encoder_mode == 'encode_learn':
            self.encoder_layers = depth
        else:
            raise NotImplementedError(f'Encoder mode {encoder_mode} not implemented')

        self.latent_scale = latent_scale
        self.latent_offset = latent_offset if latent_offset is not None else 0.

        self.cross_attend_blocks = nn.ModuleList([
            PreNorm(
                dim,
                Attention(
                    dim, dim,
                    heads=1 if single_cross_attend_head else dim // dim_head,
                    dim_head=dim if single_cross_attend_head else dim_head,
                ),
                context_dim=dim if context_norm else None,
            ),
            PreNorm(
                dim,
                FeedForward(dim)
            )
        ])

        self.point_embed = model_utils.PointEmbed(dim=dim)

        self.layers = nn.ModuleList([])

        for i in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(
                    dim,
                    Attention(dim, heads = dim // dim_head, dim_head = dim_head)
                ),
                PreNorm(
                    dim,
                    FeedForward(dim)
                )
            ]))

         # Always use SimpleAttention for final decoder layer to support eikonal loss gradients
        self.decoder_cross_attn = PreNorm(
            queries_dim,
            model_utils.SimpleAttention(
                queries_dim, dim,
                heads=1 if single_cross_attend_head else dim // dim_head,
                dim_head=dim if single_cross_attend_head else dim_head,
            ),
            context_dim=queries_dim if context_norm else None,
        )

        if final_layer_norm:
            self.to_outputs = nn.Sequential(
                nn.LayerNorm(queries_dim),
                nn.Linear(queries_dim, output_dim)
            )
            nn.init.zeros_(self.to_outputs[1].weight)
            nn.init.zeros_(self.to_outputs[1].bias)
        else:
            self.to_outputs = nn.Linear(queries_dim, output_dim)
            nn.init.zeros_(self.to_outputs.weight)
            nn.init.zeros_(self.to_outputs.bias)

        self.bottleneck = bottleneck(**bottleneck_args)


    def encode(self, pc, num_latents=None, deterministic=False):
        B, N, _ = pc.shape
        if num_latents is None:
            num_latents = self.num_latents

        if self.query_type == 'point':
            sampled_pc = model_utils.subsample(pc, N, num_latents)
            x = self.point_embed(sampled_pc)
        elif self.query_type == 'learnable':
            x = repeat(self.latents.weight, 'n d -> b n d', b = B)

        pc_embeddings = self.point_embed(pc)

        cross_attn, cross_ff = self.cross_attend_blocks

        x = cross_attn(x, context = pc_embeddings, mask = None) + x
        x = cross_ff(x) + x

        for self_attn, self_ff in self.layers[:self.bottleneck_layer]:
            x = self_attn(x) + x
            x = self_ff(x) + x

        bottleneck = self.bottleneck.pre(x, deterministic=deterministic)
        bottleneck = {
            'x': self.learn(bottleneck['x'], 0, self.encoder_layers),
            'kl': bottleneck['kl'],
        }
        bottleneck['x'] = (bottleneck['x'] - self.latent_offset) / self.latent_scale
        return bottleneck

    def learn(self, x, start_layer=0, end_layer=None):
        if end_layer is None:
            end_layer = len(self.layers)
        elif end_layer == 0:
            return x

        # if start_layer == self.bottleneck_layer:
        x = self.bottleneck.post(x)
        if self.query_type == 'learnable':
            x = x + self.latents.weight[None]

        for self_attn, self_ff in self.layers[self.bottleneck_layer:]: # TODO: support encode/decode split that's not the bottleneck
            x = self_attn(x) + x
            x = self_ff(x) + x

        return x

    def decode(self, x, queries):
        x = x * self.latent_scale + self.latent_offset
        x = self.learn(x, self.encoder_layers, None)
        queries_embeddings = self.point_embed(queries)
        latents = self.decoder_cross_attn(queries_embeddings, context = x)

        return self.to_outputs(latents)

    def decode_final(self, x, queries):
        queries_embeddings = self.point_embed(queries)
        latents = self.decoder_cross_attn(queries_embeddings, context=x)
        return self.to_outputs(latents)

    def forward(self, pc, queries):
        bottleneck = self.encode(pc)
        return {
            'o': self.decode(bottleneck['x'], queries),
            'kl': bottleneck['kl']
        }

    # MY
    def decode_latent(self, x, queries):
        # do batching
        if queries.shape[1] > 65536:
            N = 65536
            os = []
            for block_idx in range(math.ceil(queries.shape[1] / N)):
                o = self.decode(x, queries[:, block_idx*N:(block_idx+1)*N, :]).squeeze(-1)
                os.append(o)
            o = torch.cat(os, dim=1)
        else:
            o = self.decode(x, queries).squeeze(-1)
        return o

class Bottleneck(nn.Module):
    def __init__(self):
        super().__init__()

    def pre(self, x):
        return {'x': x}

    def post(self, x):
        return x


class DiagonalGaussianDistribution(object):
    def __init__(self, mean, logvar, deterministic=False):
        self.mean = mean
        self.logvar = logvar
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.mean.device)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.mean.device)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.mean(torch.pow(self.mean, 2)
                                       + self.var - 1.0 - self.logvar,
                                       dim=[1, 2])
            else:
                return 0.5 * torch.mean(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=[1, 2, 3])

    def nll(self, sample, dims=[1,2,3]):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims)

    def mode(self):
        return self.mean


class KLBottleneck(nn.Module):
    def __init__(self, dim, latent_dim, kl_weight):
        super().__init__()

        self.kl_weight = kl_weight

        self.proj = nn.Linear(latent_dim, dim)

        self.mean_fc = nn.Linear(dim, latent_dim)
        self.logvar_fc = nn.Linear(dim, latent_dim)

    def pre(self, x, deterministic=False):

        mean = self.mean_fc(x)
        logvar = self.logvar_fc(x)

        posterior = DiagonalGaussianDistribution(mean, logvar, deterministic)
        x = posterior.sample()
        kl = posterior.kl()

        return {'x': x, 'kl': self.kl_weight * kl}

    def post(self, x):

        x = self.proj(x)
        return x

## LaGeM: A Large Geometry Model for 3D Representation Learning and Diffusion
## https://openreview.net/forum?id=72OSO38a2z
class NormalizedBottleneck(nn.Module):
    def __init__(self, dim, latent_dim):
        super().__init__()

        self.post_bottleneck_proj = nn.Linear(latent_dim, dim)

        self.pre_bottleneck_proj = nn.Linear(dim, latent_dim)
        self.pre_bottleneck_norm = nn.LayerNorm(latent_dim, elementwise_affine=False, eps=1e-6)

        self.gamma = nn.Parameter(torch.ones(latent_dim))
        self.beta = nn.Parameter(torch.zeros(latent_dim))

    def pre(self, x):
        x = self.pre_bottleneck_norm(self.pre_bottleneck_proj(x))
        return {'x': x}

    def post(self, x):
        x = x * self.gamma + self.beta

        x = self.post_bottleneck_proj(x)

        return x


@dataclass
class AEConfig:
    depth: int = 24

    dim: int = 512

    num_latents: int = 512

    latent_dim: int = 8

    query_type: str = 'learnable' # learnable or point

    # number of self-attention blocks before bottleneck
    bottleneck_layer: int = 0

    # set to use original 3DShape2VecSet architecture (some minor differernces)
    original_vecset: bool = False


def create_autoencoder_from_config(config: AEConfig, **kwargs):
    model = VecSetAutoEncoder(
        depth=config.depth,
        dim=config.dim,
        output_dim=1,
        num_latents=config.num_latents,
        query_type=config.query_type,
        bottleneck_layer=config.bottleneck_layer,
        bottleneck=KLBottleneck,
        attention_class='simple',
        bottleneck_args={
            'dim': config.dim,
            'latent_dim': config.latent_dim,
            'kl_weight': 1.
        },
        **kwargs,
    )
    return model


# legacy version of AE
def create_autoencoder(depth=24, dim=512, M=512, N=2048, query_type='point', bottleneck=None, bottleneck_args={}, **kwargs):
    model = VecSetAutoEncoder(
        depth=depth, dim=dim, output_dim=1, num_inputs=N,
        num_latents=M, query_type=query_type,
        bottleneck=bottleneck,
        bottleneck_args=bottleneck_args,
        **kwargs
    )
    return model

def learnable_vec1024x16_dim1024_depth24_nb(pc_size=8192, **kwargs):
    return create_autoencoder(
        depth=24, dim=1024, M=1024,
        N=pc_size, query_type='learnable',
        bottleneck=NormalizedBottleneck,
        bottleneck_args={'dim': 1024, 'latent_dim': 16},
        **kwargs,
    )

def learnable_vec1024x32_dim1024_depth24_nb(pc_size=8192, **kwargs):
    return create_autoencoder(
        depth=24, dim=1024, M=1024,
        N=pc_size, query_type='learnable',
        bottleneck=NormalizedBottleneck,
        bottleneck_args={'dim': 1024, 'latent_dim': 32},
        **kwargs,
    )

def point_vec1024x16_dim1024_depth24_nb(pc_size=8192, **kwargs):
    return create_autoencoder(
        depth=24, dim=1024, M=1024,
        N=pc_size, query_type='point',
        bottleneck=NormalizedBottleneck,
        bottleneck_args={'dim': 1024, 'latent_dim': 16},
        **kwargs,
    )

def point_vec1024x32_dim1024_depth24_nb(pc_size=8192, **kwargs):
    return create_autoencoder(
        depth=24, dim=1024, M=1024,
        N=pc_size, query_type='point',
        bottleneck=NormalizedBottleneck,
        bottleneck_args={'dim': 1024, 'latent_dim': 32},
        **kwargs,
    )

def learnable_vec1024_dim1024_depth24(pc_size=8192, **kwargs):
    return create_autoencoder(
        depth=24, dim=1024, M=1024,
        N=pc_size, query_type='learnable',
        bottleneck=Bottleneck,
        bottleneck_args={},
        **kwargs,
    )

def point_vec1024_dim1024_depth24(pc_size=8192, **kwargs):
    return create_autoencoder(
        depth=24, dim=1024, M=1024,
        N=pc_size, query_type='point',
        bottleneck=Bottleneck,
        bottleneck_args={},
        **kwargs,
    )