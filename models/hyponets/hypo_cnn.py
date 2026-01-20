### inspired by https://github.com/yinboc/trans-inr/blob/f4bdc013286e2be00f9117e4e53913d6692fa49d/models/hyponets/hypo_mlp.py
### see the TransINR license at documentation/TransINR_usage.md

import torch
import torch.nn as nn
import numpy as np
import einops

from .layers import batched_conv


class HypoCNN(nn.Module):

    def __init__(self, in_dim, out_dim, hid_dim, strds, ks, dilation='1', use_pe=False, pe_dim=None,
        act='relu', out_bias=0, pe_sigma=1024):
        super().__init__()
        from math import ceil
        self.use_pe = use_pe
        self.pe_dim = pe_dim
        self.pe_sigma = pe_sigma
        if use_pe:
            last_dim = in_dim * pe_dim
        else:
            last_dim = in_dim
        strds_list = [int(x) for x in strds.split('_' if '_' in strds else ' ')]
        depth = len(strds_list)
        ks_list = [int(x) for x in ks.split('_')]
        ks_list += ks_list[-1:] * (depth - len(ks_list))
        dilation_list = [int(x) for x in dilation.split('_')]
        dilation_list += dilation_list[-1:] * (depth - len(dilation_list))
        ch = hid_dim

        self.ps_layers = nn.ModuleList()
        self.param_shapes = dict()
        self.conv_shape_list = []
        for i, cur_strd in enumerate(strds_list):
            cur_dim = ch if i < depth - 1 else out_dim
            cur_ks = ks_list[i]
            ch_out = cur_dim*cur_strd**2
            self.param_shapes[f'cnn_wb{i}'] = (ch_out, last_dim, cur_ks)
            cur_dilation = dilation_list[i]
            cur_pad = (cur_dilation * (cur_ks - 1)) // 2
            if cur_dilation > 1:
                assert cur_dilation % 2 == 0, 'some performance issues if dilation is odd'
            self.conv_shape_list.append((ch_out, last_dim, cur_ks, cur_pad, cur_dilation)) 
            self.ps_layers.append(nn.PixelShuffle(cur_strd))
            last_dim = cur_dim

        if act == 'relu':
            self.act = nn.ReLU()
        elif act == 'gelu':
            self.act = nn.GELU()
        else:
            NotImplementedError
        self.params = None
        self.out_bias = out_bias
        self.depth = depth

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
        B = x.shape[0]
        for i in range(self.depth):
            x = einops.rearrange(x, 'b c h w -> 1 (b c) h w')
            x = batched_conv(x, self.params[f'cnn_wb{i}'], self.conv_shape_list[i], self.ps_layers[i])
            if i < self.depth - 1:
                x = self.act(x)
            else:
                x = self.OutImg(x)
        x = einops.rearrange(x, '1 (b c) h w -> b c h w', b=B)
        return x
