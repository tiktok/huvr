import torch
from torch import nn


class Head(nn.Module):
    def __init__(self, cfg, num_classes, **kwargs):
        super().__init__()
        self.unique_type = cfg['unique_type']

        if cfg["unique_type"] == 'global_token':
            fc_dim = cfg['transformer_encoder']['dim']
        elif cfg["unique_type"] in ['global_token_trans']:
            fc_dim = cfg['embedding_dim']
        elif cfg["unique_type"] in ['global_token_dec']:
            fc_dim = cfg['transformer_decoder']['dim']
        elif cfg["unique_type"] == 'global_token_post_fc':
            fc_dim = cfg["hyponet"]["hidden_dim"]
        elif '_fuse_' in cfg["unique_type"]:
            if '_flat' in cfg["unique_type"]:
                fc_dim = cfg['transformer_encoder']['dim'] * ((cfg["tokenizer"]["input_size"] // cfg["tokenizer"]["patch_size"]) ** 2) + \
                            cfg['transformer_encoder']['dim']
            elif '_avg' in cfg["unique_type"]:
                fc_dim = cfg['transformer_encoder']['dim'] * 2
        else:
            raise NotImplementedError(f"Unique type {cfg['unique_type']} not implemented.")

        if '_flat' in cfg["unique_type"] and '_fuse_' not in cfg["unique_type"]:
            n_patches = (cfg["tokenizer"]["input_size"] // cfg["tokenizer"]["patch_size"]) ** 2
            fc_dim = fc_dim * n_patches

        if 'type' in kwargs and kwargs['type'] == 'logistic':
            self.fc = torch.nn.Linear(fc_dim, num_classes)
        else:
            self.fc = torch.nn.Sequential(
                torch.nn.BatchNorm1d(fc_dim, affine=False, eps=1e-6),
                torch.nn.Linear(fc_dim, num_classes),
            )

    def forward(self, x):
        if self.unique_type == 'pre_mlp':
            x = self.proj(x)
        
        if 'fuse' in self.unique_type:
            if '_flat' in self.unique_type:
                return self.fc(torch.cat([x[0], x[1].flatten(1)], dim=1))
            elif '_avg' in self.unique_type:
                return self.fc(torch.cat([x[0], x[1].mean(dim=1)], dim=1))
        elif 'global' in self.unique_type:
            return self.fc(x)
