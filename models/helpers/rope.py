# --------------------------------------------------------
# Lightly modified from https://github.com/baaivision/EVA/EVA02
# EVA-02: A Visual Representation for Neon Genesis
# Github source: https://github.com/baaivision/EVA/EVA02
# Copyright (c) 2023 Beijing Academy of Artificial Intelligence (BAAI)
# Licensed under The MIT License [see LICENSE for details]
# By Yuxin Fang
#
# Based on https://github.com/lucidrains/rotary-embedding-torch
# --------------------------------------------------------'

import math
from math import pi

import torch
from torch import nn

from einops import rearrange, repeat


def to_tuple(x):
    if isinstance(x, tuple):
        return x
    return (x,x)


def broadcat(tensors, dim = -1):
    num_tensors = len(tensors)
    shape_lens = set(list(map(lambda t: len(t.shape), tensors)))
    assert len(shape_lens) == 1, 'tensors must all have the same number of dimensions'
    shape_len = list(shape_lens)[0]
    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), tensors)))
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all([*map(lambda t: len(set(t[1])) <= 2, expandable_dims)]), 'invalid dimensions for broadcastable concatentation'
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_tensors), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    tensors = list(map(lambda t: t[0].expand(*t[1]), zip(tensors, expandable_shapes)))
    return torch.cat(tensors, dim = dim)


def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r = 2)
    x1, x2 = x.unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    return rearrange(x, '... d r -> ... (d r)')


class VisionRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        pt_seq_len,
        ft_seq_len=None,
        custom_freqs = None,
        freqs_for = 'lang',
        theta = 10000,
        max_freq = 10,
        num_freqs = 1,
    ):
        super().__init__()
        if custom_freqs:
            freqs = custom_freqs
        elif freqs_for == 'lang':
            freqs = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        elif freqs_for == 'pixel':
            freqs = torch.linspace(1., max_freq / 2, dim // 2) * pi
        elif freqs_for == 'constant':
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f'unknown modality {freqs_for}')

        if ft_seq_len is None: ft_seq_len = pt_seq_len
        t = torch.arange(ft_seq_len) / ft_seq_len * pt_seq_len

        freqs_h = torch.einsum('..., f -> ... f', t, freqs)
        freqs_h = repeat(freqs_h, '... n -> ... (n r)', r = 2)

        freqs_w = torch.einsum('..., f -> ... f', t, freqs)
        freqs_w = repeat(freqs_w, '... n -> ... (n r)', r = 2)

        freqs = broadcat((freqs_h[:, None, :], freqs_w[None, :, :]), dim = -1)

        self.register_buffer("freqs_cos", freqs.cos())
        self.register_buffer("freqs_sin", freqs.sin())

        print('======== shape of rope freq', self.freqs_cos.shape, '========')

    def forward(self, t, start_index = 0):
        rot_dim = self.freqs_cos.shape[-1]
        end_index = start_index + rot_dim
        assert rot_dim <= t.shape[-1], f'feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}'
        t_left, t, t_right = t[..., :start_index], t[..., start_index:end_index], t[..., end_index:]
        t = (t * self.freqs_cos) + (rotate_half(t) * self.freqs_sin)
        return torch.cat((t_left, t, t_right), dim = -1)


class VisionRotaryEmbeddingFast(nn.Module):
    def __init__(
        self,
        dim,
        pt_seq_len=16,
        ft_seq_len=None,
        custom_freqs = None,
        freqs_for = 'lang',
        theta = 10000,
        max_freq = 10,
        num_freqs = 1,
    ):
        """
        dim: dimension of the feature
        pt_seq_len: sequence length for the positional encoding
        ft_seq_len: sequence length for the feature, this is used for interpolation of the frequency when taking different input in fine-tuning/evaluation
        custom_freqs: custom frequencies
        freqs_for: frequency type
        theta: theta for the frequency
        """
        super().__init__()
        if custom_freqs:
            freqs = custom_freqs
        elif freqs_for == 'lang':
            freqs = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        elif freqs_for == 'pixel':
            freqs = torch.linspace(1., max_freq / 2, dim // 2) * pi
        elif freqs_for == 'constant':
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f'unknown modality {freqs_for}')

        pt_seq_len = to_tuple(pt_seq_len)
        if ft_seq_len is None: 
            ft_seq_len = pt_seq_len
        else:
            ft_seq_len = to_tuple(ft_seq_len)
        
        t_h = torch.arange(ft_seq_len[0]) / ft_seq_len[0] * pt_seq_len[0]
        t_w = torch.arange(ft_seq_len[1]) / ft_seq_len[1] * pt_seq_len[1]
        freqs_h = torch.einsum('..., f -> ... f', t_h, freqs)
        freqs_w = torch.einsum('..., f -> ... f', t_w, freqs)
        freqs_h = repeat(freqs_h, '... n -> ... (n r)', r = 2)
        freqs_w = repeat(freqs_w, '... n -> ... (n r)', r = 2)
        freqs = broadcat((freqs_h[:, None, :], freqs_w[None, :, :]), dim = -1)

        freqs_cos = freqs.cos().view(-1, freqs.shape[-1])
        freqs_sin = freqs.sin().view(-1, freqs.shape[-1])

        self.register_buffer("freqs_cos", freqs_cos)
        self.register_buffer("freqs_sin", freqs_sin)

        print('======== shape of rope freq', self.freqs_cos.shape, '========')

    def forward(self, t):
        N = t.shape[2]
        rope_N = self.freqs_cos.shape[0]
        if N != rope_N:
            freqs_cos = rearrange(self.freqs_cos, '(h w) d -> h w d', h = int(math.sqrt(rope_N)), w = int(math.sqrt(rope_N)))
            freqs_sin = rearrange(self.freqs_sin, '(h w) d -> h w d', h = int(math.sqrt(rope_N)), w = int(math.sqrt(rope_N)))
            freqs_cos = freqs_cos[:int(math.sqrt(N)), :int(math.sqrt(N)), :]
            freqs_sin = freqs_sin[:int(math.sqrt(N)), :int(math.sqrt(N)), :]
            freqs_cos = rearrange(freqs_cos, 'h w d -> (h w) d')
            freqs_sin = rearrange(freqs_sin, 'h w d -> (h w) d')
        else:
            freqs_cos = self.freqs_cos
            freqs_sin = self.freqs_sin
        return  t * freqs_cos + rotate_half(t) * freqs_sin
