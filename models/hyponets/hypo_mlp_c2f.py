### inspired by and adapted from https://github.com/yinboc/trans-inr/blob/f4bdc013286e2be00f9117e4e53913d6692fa49d/models/hyponets/hypo_mlp.py
### see the TransINR license at documentation/TransINR_usage.md

import einops
import torch
import torch.nn as nn
import numpy as np

from .layers import batched_linear_mm


class HypoMLPC2F(nn.Module):

    def __init__(self, depth, in_dim, out_dim, hidden_dim, use_pe, pe_dim, strides, out_bias=0, pe_sigma=1024, force_act=False):
        super().__init__()
        self.use_pe = use_pe
        self.pe_dim = pe_dim
        self.pe_sigma = pe_sigma
        self.depth = depth
        self.force_act = force_act
        self.param_shapes = dict()
        self.strides = [int(x) for x in strides.split('_')]
        assert self.strides[-1] == 1
        hidden_dim = [int(x) for x in hidden_dim.split('_')]
        hidden_dim.append(out_dim)
        if use_pe:
            last_dim = in_dim * pe_dim
        else:
            last_dim = in_dim
        for i in range(depth):
            cur_dim = hidden_dim[i]
            self.param_shapes[f'wb{i}'] = (last_dim + 1, cur_dim)
            last_dim = cur_dim
        self.relu = nn.ReLU()
        self.params = None
        self.out_bias = out_bias

    def set_params(self, params):
        self.params = params

    def convert_posenc(self, x):
        w = torch.exp(torch.linspace(0, np.log(self.pe_sigma), self.pe_dim // 2, device=x.device))
        x = torch.matmul(x.unsqueeze(-1), w.unsqueeze(0)).view(*x.shape[:-1], -1)
        x = torch.cat([torch.cos(np.pi * x), torch.sin(np.pi * x)], dim=-1)
        return x

    def OutImg(self, x):
        if self.out_bias == 'sigmoid':
            return (torch.sigmoid(x) * 2) - 1.0
        elif self.out_bias == 'tanh':
            return torch.tanh(x)
        else:
            return x + float(self.out_bias)

    def forward(self, x):
        B, h, w = x.shape[0], x.shape[1], x.shape[2]
        x = x.view(B, -1, x.shape[-1]) # B x (h * w) x in_dim, in_dim is 2
        if self.use_pe:
            x = self.convert_posenc(x) # B x (h * w) x pe_dim
        for i in range(self.depth):
            x = batched_linear_mm(x, self.params[f'wb{i}'])
            if i < self.depth - 1 or self.force_act:
                x = self.relu(x)
            if i == self.depth - 1:
                x = self.OutImg(x)
            if self.strides[i] > 1:
                x =  einops.repeat(x, 'b (h1 w1) d -> b (h1 h2 w1 w2) d', h1=h, w1=w, h2=self.strides[i], w2=self.strides[i])
                h *= self.strides[i]
                w *= self.strides[i]
        x = x.view(B, h, w, -1) # B x h x w x out_dim, out_dim is 3
        return x
