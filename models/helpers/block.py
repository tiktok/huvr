# adapted from https://github.com/baaivision/EVA/blob/master/EVA-02/asuka/modeling_finetune.py

import torch
import torch.nn as nn
from torch import Size
from typing import Union, List, Optional, Type


def maybe_add_mask(scores: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
    return scores if attn_mask is None else scores + attn_mask


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

try: 
    import xformers.ops as xops
except:
    xattn_flag = False
else:
    xattn_flag = True

_shape_t = Union[int, List[int], Size]

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., 
                norm_layer=nn.LayerNorm, 
                subln=False
            ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer() #https://docs.pytorch.org/docs/stable/generated/torch.nn.GELU.html

        self.ffn_ln = norm_layer(hidden_features) if subln else nn.Identity()

        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        
        x = self.ffn_ln(x)

        x = self.fc2(x)
        x = self.drop(x)
        return x

# https://azizbelaweid.substack.com/p/what-is-swiglu-how-to-implement-it
# https://github.com/huggingface/pytorch-image-models/blob/19f2bfb94cfedcc112cbe0ce737151ac8a79131b/timm/layers/mlp.py#L108
class SwiGLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.SiLU, drop=0., 
                norm_layer=nn.LayerNorm, subln=False
            ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(in_features, hidden_features)

        self.act = act_layer() 
        #https://docs.pytorch.org/docs/stable/generated/torch.nn.SiLU.html
        self.ffn_ln = norm_layer(hidden_features) if subln else nn.Identity()
        self.w3 = nn.Linear(hidden_features, out_features)
        
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = self.act(x1) * x2
        x = self.ffn_ln(hidden)
        x = self.w3(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False,
             qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None,
            xattn=False,
            rope=None,
            num_registers=0,
            num_weight_tokens=0,
            num_class_tokens=1,
        ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.num_registers = num_registers
        self.num_weight_tokens = num_weight_tokens
        self.num_class_tokens = num_class_tokens

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=qkv_bias)
        
        self.qk_float = True

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.xattn = xattn
        self.rope = rope

    def forward(self, x, attn_mask=None):
        B, N, C = x.shape   
        # Single linear transformation to get Q, K, V all at once
        #self.qkv_bias = self.qkv_bias.to(x.device)
        #qkv = F.linear(input=x, weight=self.qkv.weight, bias=self.qkv_bias)
        
        # Reshape and permute to separate Q, K, V
        # [B, N, 3*num_heads*head_dim] -> [3, B, num_heads, N, head_dim]
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)   # 3, B, num_heads, N, C
        q, k, v = qkv[0], qkv[1], qkv[2]   # Split into separate Q, K, V tensors   

        if self.rope:
            if self.num_weight_tokens > 0:
                q_t = q[:, :, self.num_class_tokens+self.num_registers:-self.num_weight_tokens, :]
            else:
                q_t = q[:, :, self.num_class_tokens+self.num_registers:, :]
            if q_t.shape[2] == self.rope.freqs_cos.shape[0]:
                ro_q_t = self.rope(q_t)
            else:
                ro_q_t = self.rope(q_t)
            if self.num_weight_tokens > 0:
                q = torch.cat((q[:, :, :self.num_class_tokens+self.num_registers, :], ro_q_t, q[:, :, -self.num_weight_tokens:, :]), -2).type_as(v)
            else:
                q = torch.cat((q[:, :, :self.num_class_tokens+self.num_registers, :], ro_q_t), -2).type_as(v)

            if self.num_weight_tokens > 0:
                k_t = k[:, :, self.num_class_tokens+self.num_registers:-self.num_weight_tokens, :]
            else:
                k_t = k[:, :, self.num_class_tokens+self.num_registers:, :]
            if k_t.shape[2] == self.rope.freqs_cos.shape[0]:
                ro_k_t = self.rope(k_t)
            else:
                ro_k_t = self.rope(k_t)
            if self.num_weight_tokens > 0:
                k = torch.cat((k[:, :, :self.num_class_tokens+self.num_registers, :], ro_k_t, k[:, :, -self.num_weight_tokens:, :]), -2).type_as(v)
            else:
                k = torch.cat((k[:, :, :self.num_class_tokens+self.num_registers, :], ro_k_t), -2).type_as(v)

        if self.xattn and xattn_flag:
            q = q.permute(0, 2, 1, 3)   # B, num_heads, N, C -> B, N, num_heads, C
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)

            x = xops.memory_efficient_attention(q, k, v)
            x = x.reshape(B, N, -1)
            x = self.proj(x)
            x = self.proj_drop(x)
        else:
            q = q * self.scale
            if self.qk_float:
                attn = (q.float() @ k.float().transpose(-2, -1))
            else:
                attn = (q @ k.transpose(-2, -1))
            
            if attn_mask is not None:
                attn_mask = attn_mask.bool()
                attn = attn.masked_fill(~attn_mask[:, None, None, :], float("-inf"))
            attn = attn.softmax(dim=-1).type_as(x)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
            x = self.proj(x)
            x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, 
                qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., 
                 init_values=None, norm_layer=nn.LayerNorm,
                 attn_head_dim=None, 
                 postnorm=False, 
                 subln=False,
                 xattn=False,
                 naiveswiglu=False,
                 rope=None,
                 num_registers=0,
                 num_weight_tokens=0,
                 num_class_tokens=1,
                ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim,
            xattn=xattn,
            rope=rope,
            num_registers=num_registers,
            num_weight_tokens=num_weight_tokens,
            num_class_tokens=num_class_tokens,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        if naiveswiglu:
            self.mlp = SwiGLU(
                in_features=dim, 
                hidden_features=mlp_hidden_dim, 
                subln=subln,
                norm_layer=norm_layer,
            )
        else:
            self.mlp = Mlp(
                in_features=dim, 
                hidden_features=mlp_hidden_dim, 
                subln=subln,
                norm_layer=norm_layer
            ) 

        if init_values is not None and init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

        
        self.postnorm = postnorm

    def forward(self, x,  attn_mask=None):
        """
        x: [B, N, C]
        attn_mask: [B, N, N]
        return: [B, N, C]
        """
        if self.gamma_1 is None:
            if self.postnorm:
                x = x + self.drop_path(
                    self.norm1(self.attn(x, attn_mask=attn_mask)))
                x = x + self.drop_path(self.norm2(self.mlp(x)))
            else:
                x = x + self.drop_path(
                    self.attn(self.norm1(x),  attn_mask=attn_mask))
                x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            if self.postnorm:
                x = x + self.drop_path(
                    self.gamma_1 * self.norm1(self.attn(x, attn_mask=attn_mask)))
                x = x + self.drop_path(self.gamma_2 * self.norm2(self.mlp(x)))
            else:
                x = x + self.drop_path(
                    self.gamma_1 * self.attn(self.norm1(x), attn_mask=attn_mask))
                x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x
