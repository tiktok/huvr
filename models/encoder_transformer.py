
# --------------------------------------------------------
# Lightly modified from https://github.com/baaivision/EVA/EVA02
# EVA-02: A Visual Representation for Neon Genesis
# Github source: https://github.com/baaivision/EVA/EVA02
# Copyright (c) 2023 Beijing Academy of Artificial Intelligence (BAAI)
# Licensed under The MIT License [see LICENSE for details]
# By Yuxin Fang
#
# some code from https://github.com/huggingface/pytorch-image-models/blob/b2034bb6c57fa6b41fda7398140bf21405361df7/timm/models/eva.py
# some code from https://github.com/baaivision/EVA/blob/master/EVA-02/asuka/modeling_pretrain.py
# --------------------------------------------------------'

import math
import torch.nn as nn
import torch
import torch.utils.checkpoint as checkpoint

from .helpers import VisionRotaryEmbeddingFast, trunc_normal_, Block, get_2d_sincos_pos_embed_rectangle, interpolate_pos_embed_direct, to_2tuple


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, 
                num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False,
                 qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, 
                 init_values=None, use_abs_pos_emb=True,
                 pool_type="mean", init_scale=0.001, sync_bn=False,
                 use_checkpoint=True, 
                 postnorm=False,
                 subln=False,
                 xattn=False,
                 naiveswiglu=False,
                 rope=False,
                 pt_hw_seq_len=16,
                 intp_freq=False,
                 use_registers=False,
                 num_weight_tokens=0,
                 use_global_token=True,
            ):
        super().__init__()
        """
        A
        patch_size: patch size
        in_chans: input channels
        num_classes: number of classes
        embed_dim: embedding dimension
        depth: number of layers
        num_heads: number of attention heads
        mlp_ratio: ratio of mlp hidden dim to embedding dim
        qkv_bias: whether to use bias in qkv projection
        qk_scale: scale factor for qk projection
        drop_rate: dropout rate
        attn_drop_rate: attention dropout rate
        drop_path_rate: drop path rate
        norm_layer: normalization layer
        init_values: initial values for gamma_1 and gamma_2 to control the residual connection
        use_abs_pos_emb: whether to use absolute position embedding
        use_mean_pooling: whether to use mean pooling
        init_scale: initial scale for the model
        use_checkpoint: whether to use checkpointing
        postnorm: whether to use post-norm
        subln: whether to use sub-layer norm
        xattn: whether to use xformers attention
        naiveswiglu: whether to use naive swiglu
        rope: whether to use rope
        pt_hw_seq_len: sequence length for the positional encoding
        intp_freq: whether to interpolate the frequency for rope
        """
        self.img_size = to_2tuple(img_size)
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        print(f'img size {self.img_size} patch size {patch_size} num patches {num_patches}')
        self.pos_embed_size = (self.img_size[0] // patch_size, self.img_size[1] // patch_size)

        if use_global_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.num_prefix_tokens = 1 if use_global_token else 0
        self.num_weight_tokens = num_weight_tokens
        if use_registers:
            self.num_registers = 4
            self.register_tokens = nn.Parameter(torch.zeros(1, self.num_registers, embed_dim))
        else:
            self.num_registers = 0
        if use_abs_pos_emb:
            #we use absolute position embedding with cosine-sinusoidal positional encoding
            #without gradient
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1 + self.num_registers, embed_dim), requires_grad=False)
        else:
            self.pos_embed = None
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.use_checkpoint = use_checkpoint
        self.sync_bn = sync_bn

        self.num_heads = num_heads
        self.pt_hw_seq_len = pt_hw_seq_len
        if rope:
            half_head_dim = embed_dim // num_heads // 2
            hw_seq_len = max(self.img_size) // patch_size
            self.rope = VisionRotaryEmbeddingFast(
                dim=half_head_dim,
                pt_seq_len=pt_hw_seq_len,
                ft_seq_len=hw_seq_len if intp_freq else None,
            )
        else:
            self.rope = None

        self.naiveswiglu = naiveswiglu

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.use_global_token = use_global_token
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values, 
                postnorm=postnorm,
                subln=subln,
                xattn=xattn,
                naiveswiglu=naiveswiglu,
                rope=self.rope,
                num_registers=self.num_registers,
                num_weight_tokens=self.num_weight_tokens,
                num_class_tokens=1 if use_global_token else 0,
            )
            for i in range(depth)])
        
        self.pool_type = pool_type
        if pool_type == "mean":
            self.norm = nn.Identity()
        elif pool_type == 'map':
            self.norm = norm_layer(embed_dim)
        elif pool_type == 'cls':
            self.norm = norm_layer(embed_dim)
        else:
            raise ValueError(f"Invalid pool type: {pool_type}")

        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        
        self.apply(self._init_weights)
        self.fix_init_weight()

        if isinstance(self.head, nn.Linear):
            self.head.weight.data.mul_(init_scale)
            self.head.bias.data.mul_(init_scale)
    
    def fix_init_weight(self):
        if self.pos_embed is not None:
            pos_embed = get_2d_sincos_pos_embed_rectangle(self.pos_embed.shape[2], self.pos_embed_size, True, self.num_registers)
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        if self.use_global_token:
            trunc_normal_(self.cls_token, std=.02)
        if self.num_registers:
            trunc_normal_(self.register_tokens, std=.02)
        if isinstance(self.head, nn.Linear):
            trunc_normal_(self.head.weight, std=.02)
        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            if self.naiveswiglu:
                rescale(layer.mlp.w3.weight.data, layer_id + 1)
            else:
                rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'register_tokens'}

    def forward_features(self, x, wtokens=None, extra_layer_idx=None):
        x = self.patch_embed(x)
        
        batch_size, seq_len, _ = x.size()

        if self.use_global_token:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            if self.num_registers:
                register_tokens = self.register_tokens.expand(batch_size, -1, -1)
                x = torch.cat((cls_tokens, register_tokens, x), dim=1)
            else:
                x = torch.cat((cls_tokens, x), dim=1)
        
        if self.pos_embed is not None:
            if x.shape[1] != self.pos_embed.shape[1]:
                pos_embed = self.pos_embed
                pos_tokens = pos_embed[:,1+self.num_registers:]
                orig_size = int(math.sqrt(self.pos_embed.shape[1]-1-self.num_registers))
                orig_size = (orig_size, orig_size)
                new_size = int(math.sqrt(x.shape[1]-1-self.num_registers))
                new_size = (new_size, new_size)
                pos_tokens = interpolate_pos_embed_direct(pos_tokens,orig_size,new_size,mode="bicubic")
                pos_embed = torch.cat((pos_embed[:,:1+self.num_registers],pos_tokens),dim=1)
            else:
                pos_embed = self.pos_embed
            x = x + pos_embed
        x = self.pos_drop(x)

        if wtokens is not None:
            x = torch.cat((x, wtokens), dim=1)

        for blk_idx, blk in enumerate(self.blocks):
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
            if extra_layer_idx is not None and blk_idx == extra_layer_idx - 1:
                extra_x = x.clone()

        x = self.norm(x)
        if self.num_registers:
            x = torch.cat((x[:,:1], x[:,1+self.num_registers:]), dim=1)
            if extra_layer_idx is not None:
                extra_x = torch.cat((extra_x[:,:1], extra_x[:,1+self.num_registers:]), dim=1)
        if wtokens is not None:
            x = x[:,-self.num_weight_tokens:]
            if extra_layer_idx is not None:
                extra_x = extra_x[:,-self.num_weight_tokens:]
        if extra_layer_idx is not None:
            return x, extra_x
        else:
            return x

    def forward(self, x, wtokens=None, extra_layer_idx=None):
        """
        x: [B, C, H, W]
        return_features: whether to return the features
        return: [B, N, C]
        """
        x = self.forward_features(x, wtokens, extra_layer_idx)
        return x


def vitb_16_norm_only_rope_fixed(**kwargs):
    model = VisionTransformer(
        patch_size=16, 
        embed_dim=768, 
        depth=12, 
        num_heads=16, 
        mlp_ratio=4*2/3, 
        qkv_bias=True,
        norm_layer=nn.LayerNorm, 
        use_abs_pos_emb=False,
        subln=True,
        xattn=True,
        naiveswiglu=True,
        rope=True, 
        pt_hw_seq_len=16,   # 512/16
        intp_freq=True,
        pool_type="map",
        num_classes=0,
        **kwargs)
    
    return model

def vitl_16_norm_only_rope_fixed(**kwargs):
    model = VisionTransformer(
        patch_size=16, 
        embed_dim=1024, 
        depth=24, 
        num_heads=16, 
        mlp_ratio=4*2/3, 
        qkv_bias=True,
        norm_layer=nn.LayerNorm, 
        use_abs_pos_emb=False,
        subln=True,
        xattn=True,
        naiveswiglu=True,
        rope=True, 
        pt_hw_seq_len=16,   # 512/16
        intp_freq=True,
        pool_type="map",
        num_classes=0,
        **kwargs)
    
    return model
