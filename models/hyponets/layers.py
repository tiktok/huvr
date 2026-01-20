### batched_linear_mm from https://github.com/yinboc/trans-inr/blob/f4bdc013286e2be00f9117e4e53913d6692fa49d/models/hyponets/hypo_mlp.py
### see the TransINR license at documentation/TransINR_usage.md

import torch


def batched_linear_mm(x, wb):
    # x: (B, N, D1); wb: (B, D1 + 1, D2) or (D1 + 1, D2)
    one = torch.ones(*x.shape[:-1], 1, device=x.device)
    return torch.matmul(torch.cat([x, one], dim=-1), wb)


def batched_conv(x, wb, conv_shape, ps_layer):
    B = wb.size(0)
    ch_out, ch_in, cur_ks, pad, dilation = conv_shape
    conv_weights = wb[:,:-1].view((B, ch_in, cur_ks, cur_ks, ch_out))
    conv_weights = conv_weights.permute(0,-1,1,2,3).flatten(end_dim=1) # (B, ch_out, ch_in, cur_ks, cur_ks) -> (B*ch_out, ch_in, cur_ks, cur_ks)
    conv_bias = wb[:,-1:].flatten() # B*ch_out
    if pad > 1:
        x = torch.nn.functional.pad(x, (pad, pad, pad, pad), mode='reflect')
        x = torch.nn.functional.conv2d(x, conv_weights, conv_bias, stride=1, padding=0, dilation=dilation, groups=B)
    else:
        x = torch.nn.functional.conv2d(x, conv_weights, conv_bias, stride=1, padding=pad, dilation=dilation, groups=B)

    return ps_layer(x)
