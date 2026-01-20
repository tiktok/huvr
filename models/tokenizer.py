### inspired by and some code from https://github.com/yinboc/trans-inr/blob/f4bdc013286e2be00f9117e4e53913d6692fa49d/models/tokenizers/imgrec_tokenizer.py
### see the TransINR license at documentation/TransINR_usage.md

import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F

from .helpers import get_2d_sincos_pos_embed, interpolate_pos_embed_direct


class Tokenizer(nn.Module):

    def __init__(self, input_size, patch_size, dim, padding=0, img_channels=3, decoder_dim=None, 
                 use_decoder=False, embedding_dim=None, use_transcoder=False):
        super().__init__()
        if isinstance(input_size, int):
            input_size = (input_size, input_size)
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        if isinstance(padding, int):
            padding = (padding, padding)
        self.patch_size = patch_size
        self.padding = padding
        self.prefc = nn.Linear(patch_size[0] * patch_size[1] * img_channels, dim)
        n_patches = ((input_size[0] + padding[0] * 2) // patch_size[0]) * ((input_size[1] + padding[1] * 2)  // patch_size[1])

        if use_decoder:
            self.decoder_posemb = nn.Parameter(torch.randn(n_patches, decoder_dim), requires_grad=False)
            decoder_posemb = get_2d_sincos_pos_embed(decoder_dim, int(n_patches**.5), cls_token=False)
            self.decoder_posemb.data.copy_(torch.from_numpy(decoder_posemb).float())

        if use_transcoder:
            self.transcoder_posemb = nn.Parameter(torch.randn(n_patches, embedding_dim), requires_grad=False)
            transcoder_posemb = get_2d_sincos_pos_embed(embedding_dim, int(n_patches**.5), cls_token=False)
            self.transcoder_posemb.data.copy_(torch.from_numpy(transcoder_posemb).float())

    
    def forward_transcoder(self, x_):
        x = x_ + self.transcoder_posemb.unsqueeze(0)
        return x

    def forward_decoder(self, x_):
        if x_.shape[1] != self.decoder_posemb.shape[0]:
            decoder_posemb = self.decoder_posemb
            orig_size = int(math.sqrt(decoder_posemb.shape[0]))
            orig_size = (orig_size, orig_size)
            new_size = int(math.sqrt(x_.shape[1]))
            new_size = (new_size, new_size)
            decoder_posemb = interpolate_pos_embed_direct(decoder_posemb.unsqueeze(0), orig_size, new_size, mode="bicubic")
        else:
            decoder_posemb = self.decoder_posemb.unsqueeze(0)

        x = x_ + decoder_posemb
        return x
        
    def forward(self, x):
        p = self.patch_size
        x = F.unfold(x, p, stride=p, padding=self.padding) # (B, C * p * p, L)
        x = x.permute(0, 2, 1).contiguous()
        x_ = self.prefc(x)
        x = x_ + self.posemb.unsqueeze(0)
        return x
