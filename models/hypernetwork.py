### inspired by and some code from trans-inr, see https://github.com/yinboc/trans-inr/blob/f4bdc013286e2be00f9117e4e53913d6692fa49d/models/trans_inr.py
### see the TransINR license at documentation/TransINR_usage.md

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

from .hyponets import  HypoMLPC2F, HypoCNN
from .tokenizer import Tokenizer
from .decoder_transformer import TransformerEncoder
from .encoder_transformer import \
    vitb_16_norm_only_rope_fixed, \
    vitl_16_norm_only_rope_fixed


def make_distillation_layer(dim_map, potential_location, distill_dim):
    return nn.Sequential(
        nn.LayerNorm(dim_map[potential_location]),
        nn.Linear(dim_map[potential_location], distill_dim),
    )

def init_wb(shape):
    weight = torch.empty(shape[1], shape[0] - 1)
    nn.init.kaiming_uniform_(weight, a=math.sqrt(5))

    bias = torch.empty(shape[1], 1)
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
    nn.init.uniform_(bias, -bound, bound)

    return torch.cat([weight, bias], dim=1).t().detach()

def init_wb_cnn(shape):
    out_ch, in_ch, ks = shape
    weight = torch.empty(in_ch, out_ch, ks, ks)
    nn.init.kaiming_uniform_(weight, a=math.sqrt(5))

    bias = torch.empty(out_ch)
    _, fan_in = nn.init._calculate_fan_in_and_fan_out(weight)
    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
    nn.init.uniform_(bias, -bound, bound)

    wb_list = [weight.permute(0,2,3,1).flatten(end_dim=-2), bias[None]]
    return torch.cat(wb_list, dim=0).detach()


class HyperNetwork(nn.Module):

    def __init__(self, tokenizer_cfg, hyponet_cfg, hypocnn_cfg, transformer_encoder_cfg, transformer_decoder_cfg, transformer_transcoder_cfg, embedding_dim,
                 mod_idxs, distill_dim=512, distill_mode='global', distill_location='transcoder', decoder_type='old',
                 use_hypocnn=True, is_patch_mode=True, n_groups=None, use_global_token=True):
        super().__init__()

        if transformer_encoder_cfg['name'] == 'vitb_16_norm_only_rope_fixed':
            self.transformer_encoder = vitb_16_norm_only_rope_fixed(img_size=tokenizer_cfg['input_size'], use_registers=transformer_encoder_cfg['use_registers'])
        elif transformer_encoder_cfg['name'] == 'vitl_16_norm_only_rope_fixed':
            self.transformer_encoder = vitl_16_norm_only_rope_fixed(img_size=tokenizer_cfg['input_size'], use_registers=transformer_encoder_cfg['use_registers'])

        encoder_dim = transformer_encoder_cfg['dim']
        self.decoder_type = decoder_type
        if decoder_type == 'old':
            self.transformer_decoder = TransformerEncoder(**transformer_decoder_cfg)
            decoder_dim = transformer_decoder_cfg['dim']
            tokenizer_cfg['use_decoder'] = True
        elif decoder_type == 'pos_free':
            self.transformer_decoder = TransformerEncoder(**transformer_decoder_cfg)
            decoder_dim = transformer_decoder_cfg['dim']
            tokenizer_cfg['use_decoder'] = False
        else:
            decoder_dim = encoder_dim
        tokenizer_cfg['dim'] = encoder_dim
        tokenizer_cfg['decoder_dim'] = decoder_dim
        self.hyponet = HypoMLPC2F(**hyponet_cfg)
        if use_hypocnn:
            self.hypocnn = HypoCNN(**hypocnn_cfg)
        else:
            self.hypocnn = None
        self.is_patch_mode = is_patch_mode
        self.use_global_token = use_global_token
        if transformer_transcoder_cfg['dim'] > -1:
            self.transformer_transcoder = TransformerEncoder(**transformer_transcoder_cfg)
            tokenizer_cfg['use_transcoder'] = True
            tokenizer_cfg['embedding_dim'] = embedding_dim
            self.use_transcoder = True
            assert transformer_transcoder_cfg['dim'] == embedding_dim
        else:
            self.transformer_transcoder = None
            tokenizer_cfg['use_transcoder'] = False
            self.use_transcoder = False
        if decoder_type == 'old':
            self.tokenizer = Tokenizer(**tokenizer_cfg)
        if embedding_dim != encoder_dim:
            self.dimension_downsample = nn.Sequential(
                    nn.LayerNorm(encoder_dim),
                    nn.Linear(encoder_dim, embedding_dim),
                )
        else:
            self.dimension_downsample = nn.Identity()
        if embedding_dim != decoder_dim:
            self.dimension_upsample = nn.Sequential(
                    nn.LayerNorm(embedding_dim),
                    nn.Linear(embedding_dim, decoder_dim),
                )
        else:
            self.dimension_upsample = nn.Identity()
        self.mod_idxs = [int(x) for x in mod_idxs.split('_')]
        self.distill_mode = distill_mode
        if distill_location is not None:
            self.distill_locations = distill_location.split('-')
        else:
            self.distill_locations = None

        self.base_params = nn.ParameterDict()
        self.wtoken_postfc = nn.ModuleDict()
        self.num_unique_params = 0
        if not self.is_patch_mode:
            n_wtokens = 0
            self.wtoken_rng = dict()
        for name, shape in self.hyponet.param_shapes.items():
            self.base_params[name] = nn.Parameter(init_wb(shape))
            if int(name.replace('wb', '')) in self.mod_idxs:
                if use_global_token:
                    self.wtoken_postfc[f'{name}_global_postfc'] = nn.Sequential(
                        nn.LayerNorm(decoder_dim),
                        nn.Linear(decoder_dim, shape[1]),
                    )
                self.wtoken_postfc[f'{name}_patch_postfc'] = nn.Sequential(
                    nn.LayerNorm(decoder_dim),
                    nn.Linear(decoder_dim, shape[0] - 1),
                )
                self.patch_token_dim = shape[0] - 1
                self.global_token_dim = shape[1]
                if not self.is_patch_mode:
                    g = min(n_groups, shape[1])
                    assert shape[1] % g == 0
                    self.wtoken_rng[name] = (n_wtokens, n_wtokens + g)
                    n_wtokens += g
                    self.num_unique_params += (g * (shape[0] - 1))
        if not self.is_patch_mode:
            self.n_wtokens = n_wtokens
            self.wtokens = nn.Parameter(torch.randn(n_wtokens, encoder_dim))
        else:
            self.n_wtokens = 0

        if use_hypocnn:
            for name, shape in self.hypocnn.param_shapes.items():
                self.base_params[name] = nn.Parameter(init_wb_cnn(shape))

        dim_map = {
            'encoder': encoder_dim,
            'transcoder': embedding_dim,
            'decoder': decoder_dim
        }

        if isinstance(distill_dim, list):
            cls_distill_dims = {}
            patch_distill_dims = {}
            for dim_str in distill_dim:
                cur_distill_location = dim_str.split('_')[0]
                cur_distill_dim = int(dim_str.split('_')[1])
                if cur_distill_location == 'EC':
                    cls_distill_dims['encoder'] = cur_distill_dim
                if cur_distill_location == 'EP':
                    patch_distill_dims['encoder'] = cur_distill_dim
                if cur_distill_location == 'DC':
                    cls_distill_dims['decoder'] = cur_distill_dim
                if cur_distill_location == 'DP':
                    patch_distill_dims['decoder'] = cur_distill_dim

        if self.distill_locations is not None:
            for potential_location in ['encoder', 'transcoder', 'decoder']:
                if potential_location in self.distill_locations:
                    if isinstance(distill_dim, list):
                        cls_distill_dim = cls_distill_dims[potential_location]
                        patch_distill_dim = patch_distill_dims[potential_location]
                    else:
                        cls_distill_dim = distill_dim
                        patch_distill_dim = distill_dim
                    self.wtoken_postfc[f'distill_{potential_location}'] = make_distillation_layer(dim_map, potential_location, cls_distill_dim)
                    if 'separate' in self.distill_locations:
                        assert ('dense' in self.distill_mode or 'fancy' in self.distill_mode)
                        self.wtoken_postfc[f'distill_{potential_location}_patches'] = make_distillation_layer(dim_map, potential_location, patch_distill_dim)

    def get_param_counts(self):
        transformer_params = sum(p.numel() for p in self.transformer_encoder.parameters())
        wtoken_postfc_params = sum(p.numel() for p in self.wtoken_postfc.parameters())
        base_params = sum(p.numel() for p in self.base_params.values())
        print(f'Transformer encoder params: {transformer_params}')
        print(f'HyperNetwork Params: {transformer_params + wtoken_postfc_params + base_params}')
        print(f'Base params: {base_params}')
        if self.decoder_type in ['old', 'pos_free']:
            print(f'Decoder params: {sum(p.numel() for p in self.transformer_decoder.parameters())}')
        print(f'Global token dim: {self.global_token_dim}, patch token dim: {self.patch_token_dim}')

    def forward(self, data, unique_only=False, encoder_distill_layer=None, decoder_distill_layer=None, **kwargs):
        B = data.shape[0]
        distill_outs = []

        global_token_offset = 1 if self.use_global_token else 0

        if self.is_patch_mode:
            if encoder_distill_layer is not None:
                trans_enc_out, enc_to_distill = self.transformer_encoder(data, extra_layer_idx=encoder_distill_layer)
            else:
                trans_enc_out = self.transformer_encoder(data)
                enc_to_distill = trans_enc_out
        else:
            wtokens = einops.repeat(self.wtokens, 'n d -> b n d', b=B)
            trans_enc_out = self.transformer_encoder(data, wtokens=wtokens)
            enc_to_distill = trans_enc_out
        if self.distill_locations is not None and 'encoder' in self.distill_locations:
            if 'dense' in self.distill_mode:
                if 'separate' in self.distill_locations:
                    distill_enc_out = torch.cat([
                        self.wtoken_postfc['distill_encoder'](enc_to_distill[:, :1, :]),
                        self.wtoken_postfc['distill_encoder_patches'](enc_to_distill[:, 1:, :])
                    ], dim=1)
                else:
                    distill_enc_out = self.wtoken_postfc['distill_encoder'](enc_to_distill)
            elif 'fancy' in self.distill_mode:
                if 'separate' in self.distill_locations:
                    distill_enc_out = [
                        self.wtoken_postfc['distill_encoder'](enc_to_distill[:, :1, :]),
                        self.wtoken_postfc['distill_encoder_patches'](enc_to_distill[:, 1:, :])
                    ]
                else:
                    distill_enc_out = self.wtoken_postfc['distill_encoder'](enc_to_distill)
                    distill_enc_out = [
                        distill_enc_out[:, :1],
                        distill_enc_out[:, 1:],
                    ]
            else:
                distill_enc_out = self.wtoken_postfc['distill_encoder'](enc_to_distill[:, :global_token_offset, :]).flatten(1) # B x 512
            distill_outs.append(distill_enc_out)
        
        trans_enc_out_downsampled = self.dimension_downsample(trans_enc_out)
        if self.use_transcoder:
            if global_token_offset:
                trans_trans_out = self.transformer_transcoder(torch.cat([trans_enc_out_downsampled[:, :1, :],
                                                        self.tokenizer.forward_transcoder(trans_enc_out_downsampled[:, 1:, :])], dim=1))
            else:
                trans_trans_out = self.transformer_transcoder(trans_enc_out_downsampled)
        else:
            trans_trans_out = trans_enc_out_downsampled
        
        if self.distill_locations is not None and 'transcoder' in self.distill_locations:
            if 'dense' in self.distill_mode:
                if 'separate' in self.distill_locations:
                    distill_trans_out = torch.cat([
                        self.wtoken_postfc['distill_transcoder'](trans_trans_out[:, :1, :]),
                        self.wtoken_postfc['distill_transcoder_patches'](trans_trans_out[:, 1:, :])
                    ], dim=1)
                else:
                    distill_trans_out = self.wtoken_postfc['distill_transcoder'](trans_trans_out)
            elif 'fancy' in self.distill_mode:
                if 'separate' in self.distill_locations:
                    distill_trans_out = [
                        self.wtoken_postfc['distill_transcoder'](trans_trans_out[:, :1, :]),
                        self.wtoken_postfc['distill_transcoder_patches'](trans_trans_out[:, 1:, :])
                    ]
                else:
                    distill_trans_out = self.wtoken_postfc['distill_transcoder'](trans_trans_out)
                    distill_trans_out = [
                        distill_trans_out[:, :1],
                        distill_trans_out[:, 1:],
                    ]
            else:
                distill_trans_out = self.wtoken_postfc['distill_transcoder'](trans_trans_out[:, :1, :]).flatten(1) # B x 512
            distill_outs.append(distill_trans_out)
        
        trans_dec_in = self.dimension_upsample(trans_trans_out)
        
        if self.decoder_type == 'old':
            if global_token_offset:
                if decoder_distill_layer is not None:
                    trans_out, dec_to_distill = self.transformer_decoder(torch.cat([trans_dec_in[:, :1, :], 
                                                            self.tokenizer.forward_decoder(trans_dec_in[:, 1:, :])], dim=1), extra_layer_idx=decoder_distill_layer)
                else:
                    trans_out = self.transformer_decoder(torch.cat([trans_dec_in[:, :1, :], 
                                                            self.tokenizer.forward_decoder(trans_dec_in[:, 1:, :])], dim=1))
                    dec_to_distill = trans_out
            else:
                if decoder_distill_layer is not None:
                    trans_out, dec_to_distill = self.transformer_decoder(trans_dec_in, extra_layer_idx=decoder_distill_layer)
                else:
                    trans_out = self.transformer_decoder(trans_dec_in)
                    dec_to_distill = trans_out
        elif self.decoder_type == 'pos_free':
            if decoder_distill_layer is not None:
                trans_out, dec_to_distill = self.transformer_decoder(trans_dec_in, extra_layer_idx=decoder_distill_layer)
            else:
                trans_out = self.transformer_decoder(trans_dec_in)
                dec_to_distill = trans_out
        else:
            trans_out = trans_dec_in

        if self.distill_locations is not None and 'decoder' in self.distill_locations:
            if 'dense' in self.distill_mode:
                if 'separate' in self.distill_locations:
                    distill_dec_out = torch.cat([
                        self.wtoken_postfc['distill_decoder'](dec_to_distill[:, :1, :]),
                        self.wtoken_postfc['distill_decoder_patches'](dec_to_distill[:, 1:, :])
                    ], dim=1)
                else:
                    distill_dec_out = self.wtoken_postfc['distill_decoder'](dec_to_distill)
            elif 'fancy' in self.distill_mode:
                if 'separate' in self.distill_locations:
                    distill_dec_out = [
                        self.wtoken_postfc['distill_decoder'](dec_to_distill[:, :1, :]),
                        self.wtoken_postfc['distill_decoder_patches'](dec_to_distill[:, 1:, :])
                    ]
                else:
                    distill_dec_out = self.wtoken_postfc['distill_decoder'](dec_to_distill)
                    distill_dec_out = [
                        distill_dec_out[:, :1],
                        distill_dec_out[:, 1:],
                    ]
            else:
                distill_dec_out = self.wtoken_postfc['distill_decoder'](dec_to_distill[:, :global_token_offset, :]).flatten(1) # B x 512
            distill_outs.append(distill_dec_out)

        if unique_only:
            return {
                'patch_tokens_enc': trans_enc_out[:, global_token_offset:, :],
                'global_token': trans_enc_out[:, :global_token_offset, :].squeeze(),
                #'distill_token': distill_enc_out[:, :global_token_offset, :].squeeze(),
                'patch_tokens_trans': trans_trans_out[:, global_token_offset:, :],
                'global_token_trans': trans_trans_out[:, :global_token_offset, :].squeeze(),
                'patch_tokens_dec': trans_out[:, global_token_offset:, :],
                'global_token_dec': trans_out[:, :global_token_offset, :].squeeze(),
            }

        if self.is_patch_mode:
            if self.use_global_token:
                global_out = trans_out[:, :global_token_offset, :]
            patch_out = trans_out[:, global_token_offset:, :]

        params = dict()
        for name, shape in self.hyponet.param_shapes.items():
            if self.is_patch_mode:
                wb = einops.repeat(self.base_params[name], 'n m -> (b p) n m', b=B, p=patch_out.shape[1])
            else:
                wb = einops.repeat(self.base_params[name], 'n m -> b n m', b=B)
            w, b = wb[:, :-1, :], wb[:, -1:, :]

            if int(name.replace('wb', '')) in self.mod_idxs:
                if self.is_patch_mode:
                    if self.use_global_token:
                        c = self.wtoken_postfc[f'{name}_global_postfc'](global_out) # B x 1 x d
                    x = self.wtoken_postfc[f'{name}_patch_postfc'](patch_out) # B x P x d
                    x = x.unsqueeze(-1) # B x P x d x 1
                    if self.use_global_token:
                        c = einops.repeat(c, 'b 1 d -> b p d', p=x.shape[1]).unsqueeze(-2) # B x P x 1 x d
                        x = torch.einsum('b p j k, b p k l -> b p j l', x, c) # B x P x d x d
                        x = einops.rearrange(x, 'b p j l -> (b p) j l')
                    else:
                        x = einops.repeat(x, 'b p j 1 -> b p j l', l=self.global_token_dim)
                        x = einops.rearrange(x, 'b p j l -> (b p) j l')
                    w = F.normalize(w * x, dim=1)
                else:
                    l, r = self.wtoken_rng[name]
                    x = self.wtoken_postfc[f'{name}_patch_postfc'](trans_out[:, l: r, :])
                    x = x.transpose(-1, -2) # (B, shape[0] - 1, g)
                    w = F.normalize(w * x.repeat(1, 1, w.shape[2] // x.shape[2]), dim=1)
            else:
                w = F.normalize(w, dim=1)

            wb = torch.cat([w, b], dim=1)
            params[name] = wb

        if self.hypocnn is not None:
            cnn_params = dict()
            for name, shape in self.hypocnn.param_shapes.items():
                wb = einops.repeat(self.base_params[name], 'n m -> b n m', b=B)
                w, b = wb[:, :-1, :], wb[:, -1:, :]
                w = F.normalize(w, dim=1)
                wb = torch.cat([w, b], dim=1)
                cnn_params[name] = wb

        self.hyponet.set_params(params)  ### NOTE: we are batching B * p hyponetworks
        if self.hypocnn is not None:
            self.hypocnn.set_params(cnn_params)  ### NOTE: we are batching B * p hyponetworks
        return {
            'hyponet': self.hyponet,
            'hypocnn': self.hypocnn,
            'distill': distill_outs,
            'cls_token': trans_enc_out[:, :1, :].squeeze()
        }
