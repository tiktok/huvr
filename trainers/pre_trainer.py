import einops
import os

import torchvision
from torchvision import transforms
import torch
import torch.nn as nn
import wandb
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import gc
import copy

from models import HyperNetwork
from models.helpers import interpolate_pos_embed
from utils import make_coord_grid, adjust_learning_rate, NativeScalerWithGradNormCount
from .trainer import Trainer

class PreTrainer(Trainer):

    def __init__(self, rank, cfg):
        super().__init__(rank, cfg)
        if self.is_master:
            wandb.init(project="universal_vision_encoder", name=f"{cfg['dataset_name']}_{cfg['exp_name']}_backbone")
            wandb.config.update(cfg)

    def make_datasets(self):
        """
            By default, train dataset performs shuffle and drop_last.
            Distributed sampler will extend the dataset with a prefix to make the length divisible by tot_gpus, samplers should be stored in .dist_samplers.

            Cfg example:

            train/test_dataset:
                name:
                args:
                loader: {batch_size: , num_workers: }
        """
        cfg = self.cfg
        self.dist_samplers = []

        def make_distributed_loader(dataset, batch_size, num_workers, shuffle=False, drop_last=False):
            sampler = DistributedSampler(dataset, shuffle=shuffle) if self.distributed else None
            loader = DataLoader(
                dataset,
                batch_size,
                drop_last=drop_last,
                sampler=sampler,
                shuffle=(shuffle and (sampler is None)),
                num_workers=num_workers,
                pin_memory=True)
            return loader, sampler

        train_transforms = [transforms.Resize(cfg['tokenizer']['input_size'], interpolation=3), 
                                          transforms.CenterCrop(cfg['tokenizer']['input_size']), 
                                          transforms.RandomHorizontalFlip(), 
                                          transforms.ToTensor()]
        if cfg['normalize_images']:
            train_transforms.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

        train_dataset = torchvision.datasets.ImageFolder(
            root=os.path.join(cfg['working_root'], "data/imagenet/train"),
            transform=transforms.Compose(train_transforms)
        )
        print(f'Train dataset: len={len(train_dataset)}')
        self.train_loader, train_sampler = make_distributed_loader(
            train_dataset, cfg['batch_size'], cfg['num_workers'], shuffle=True, drop_last=True)
        self.dist_samplers.append(train_sampler)

    def make_model(self):
        if self.cfg['trainer'] in ['hypernetwork']:
            model = HyperNetwork(self.cfg['tokenizer'], self.cfg['hyponet'], self.cfg['hypocnn'], self.cfg['transformer_encoder'], self.cfg['transformer_decoder'],
                                            self.cfg['transformer_transcoder'], self.cfg['embedding_dim'], self.cfg['mod_idxs'], 
                                            self.cfg['distill_dim'], self.cfg['distill_mode'], self.cfg['distill_location'], self.cfg['decoder_type'],
                                            self.cfg['use_hypocnn'], self.cfg['is_patch_mode'], self.cfg['n_groups'], self.cfg['use_global_token'])
        print(model.get_param_counts(), flush=True)

        if os.path.exists(os.path.join(self.cfg['working_root'], f"checkpoints/{self.cfg['dataset_name']}_{self.cfg['exp_name']}_latest.pth")):
            checkpoint = torch.load(os.path.join(self.cfg['working_root'], f"checkpoints/{self.cfg['dataset_name']}_{self.cfg['exp_name']}_latest.pth"))
            model.load_state_dict(checkpoint["model"])
            self.epoch = checkpoint["epoch"]
            self.optimizer_state = copy.deepcopy(checkpoint["optimizer"])
            self.scaler_state = copy.deepcopy(checkpoint["scaler"])
            del checkpoint
            gc.collect()
            torch.cuda.empty_cache()
        elif self.cfg['pretrain_path'] is not None:
            checkpoint = torch.load(self.cfg['pretrain_path'])
            model_state = checkpoint["model"]
            interpolate_pos_embed(model, model_state, use_decoder=True)
            model_state = {k: v for k, v in model_state.items() if "rope" not in k}
            msg = model.load_state_dict(model_state, strict=False)
            print(msg)
            del checkpoint
            gc.collect()
            torch.cuda.empty_cache()
            self.epoch = 1
            self.optimizer_state = None
            self.scaler_state = None
        else:
            self.epoch = 1
            self.optimizer_state = None
            self.scaler_state = None

        if self.distributed:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model.cuda()
            model_ddp = DistributedDataParallel(model, device_ids=[self.rank])
        else:
            model.cuda()
            model_ddp = model
        self.model = model
        self.model_ddp = model_ddp

        if self.is_master and self.cfg['log_gradients']:
            wandb.watch(self.model_ddp, log_freq=100)

    def train(self):
        """
            For epochs perform training, evaluation, and visualization.
            Note that ave_scalars update ignores the actual current batch_size.
        """
        cfg = self.cfg

        self.optimizer = torch.optim.AdamW(self.model_ddp.parameters(), lr=cfg['lr'] * cfg['batch_size'] * cfg['env']['tot_gpus'] / 256, eps=cfg['opt_eps'], weight_decay=cfg['weight_decay'])
        if self.optimizer_state is not None:
            self.optimizer.load_state_dict(self.optimizer_state)

        self.scaler = NativeScalerWithGradNormCount()
        if self.scaler_state is not None:
            self.scaler.load_state_dict(self.scaler_state)

        max_epochs = cfg['max_epochs']

        for epoch in range(self.epoch, max_epochs + 1):
            self.epoch = epoch

            if self.distributed and self.cfg["dataset_name"] not in ['imagenet22k', 'datacomp']:
                for sampler in self.dist_samplers:
                    sampler.set_epoch(epoch)
            
            self.train_epoch()
            # break

            if epoch % self.cfg['ckpt_freq'] == 0:
                self.save_checkpoint(os.path.join(cfg['working_root'], f"checkpoints/{self.cfg['dataset_name']}_{self.cfg['exp_name']}_epoch{epoch}.pth"))
            self.save_checkpoint(os.path.join(cfg['working_root'], f"checkpoints/{self.cfg['dataset_name']}_{self.cfg['exp_name']}_latest.pth"))

    def _iter_step(self, cur_step, data, is_train):
        images, _ = data
        B = images.shape[0]
        images = images.cuda()

        hyponet = self.model_ddp(images)

        if self.cfg['is_patch_mode']:
            assert self.cfg['tokenizer']['patch_size'] % np.prod([int(x) for x in self.cfg['hyponet']['strides'].split('_')]) == 0
            grid_patch_size = self.cfg['tokenizer']['patch_size'] // np.prod([int(x) for x in self.cfg['hyponet']['strides'].split('_')])
            coords = make_coord_grid([grid_patch_size, grid_patch_size], (-1, 1), device=images.device)
            coords = einops.repeat(coords, 'h w d -> (b p) h w d', b=B, p=int((self.cfg['tokenizer']['input_size'] / self.cfg['tokenizer']['patch_size']) ** 2)) 
        else:
            coords = make_coord_grid(images.shape[-2:], (-1, 1), device=images.device)
            coords = einops.repeat(coords, 'h w d -> b h w d', b=B)
        output = hyponet(coords)
        images = einops.rearrange(images, 'b c h w -> b h w c')
        if self.cfg['is_patch_mode']:
            patches = einops.rearrange(output, '(b p) h w c -> b p h w c', b=B)
            output = einops.rearrange(patches, 'b (p1 p2) h w c -> b (p1 h) (p2 w) c', p1=int(np.sqrt(patches.shape[1])), p2=int(np.sqrt(patches.shape[1])))
        mses = ((output - images)**2).view(B, -1).mean(dim=-1)
        loss = mses.mean()
        with torch.no_grad():
            if self.cfg['normalize_images']:
                denormalized_mses = (((output * 0.5 + 0.5) - (images * 0.5 + 0.5))**2).view(B, -1).mean(dim=-1)
                psnr = (-10 * torch.log10(denormalized_mses)).mean()
            else:
                psnr = (-10 * torch.log10(mses)).mean()

        if is_train:
            learning_rate = adjust_learning_rate(self.optimizer, 
                                                cur_step + len(self.train_loader) * (self.epoch - 1), 
                                                self.cfg['warmup_epochs'] * len(self.train_loader), 
                                                self.cfg['max_epochs'] * len(self.train_loader), 
                                                1e-7, 
                                                self.cfg['lr'] * self.cfg['batch_size'] * self.cfg['env']['tot_gpus'] / 256
            )
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model_ddp.parameters(), self.cfg['clip'])
            self.optimizer.step()

        return {'loss': loss.item(), 'psnr': psnr.item(), 'lr': learning_rate}

    def train_step(self, iter_step, data):
        return self._iter_step(iter_step, data, is_train=True)

    def train_epoch(self):
        self.model_ddp.train()

        pbar = self.train_loader
        if self.is_master:
            pbar = tqdm(pbar, desc='train', leave=False)

        for cur_step, data in enumerate(pbar):
            ret = self.train_step(cur_step, data)
            # break
            if self.is_master:
                if 'contrastive_loss' in ret:
                    pbar.set_description(f"LR: {ret['lr']:.4f}, PSNR: {ret['psnr']:.4f}, Reconstruction Loss: {ret['loss']:.4f}, Contrastive Loss: {ret['contrastive_loss']:.4f}")
                elif 'psnr' in ret:
                    pbar.set_description(f"LR: {ret['lr']:.4f}, PSNR: {ret['psnr']:.4f}, Loss: {ret['loss']:.4f}")
                else:
                    pbar.set_description(f"LR: {ret['lr']:.4f}, Loss: {ret['loss']:.4f}")
                wandb.log(ret)

            if "debug" in self.cfg["exp_name"] and cur_step > 100:
                break

    def save_checkpoint(self, filename):
        if not self.is_master:
            return
        print(f'Saving checkpoint to {filename}', flush=True)
        model_sd = self.model.state_dict()
        optimizer_sd = self.optimizer.state_dict()
        scaler_sd = self.scaler.state_dict()
        checkpoint = {
            'model': model_sd,
            'optimizer': optimizer_sd,
            'scaler': scaler_sd,
            'epoch': self.epoch + 1,
            'cfg': self.cfg,
        }
        torch.save(checkpoint, filename)
