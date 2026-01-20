import einops

from tqdm import tqdm
from torchvision import transforms
import torch
import numpy as np
import os
import math
import lpips
from pytorch_msssim import ssim
import wandb
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from data.custom_datasets import DistillImageFolder
from utils import make_coord_grid, adjust_learning_rate, KoLeoLoss, \
    NativeScalerWithGradNormCount
from .pre_trainer import PreTrainer


class HyperNetworkPreTrainer(PreTrainer):

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

        train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(self.cfg['tokenizer']['input_size'], scale=(0.2, 1.0), interpolation=3),
                transforms.RandomHorizontalFlip()
            ]
        )

        our_transform = [transforms.ToTensor()]
        if cfg['normalize_images']:
            our_transform.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        our_transform = transforms.Compose(our_transform)

        if cfg['distill_mode'] in ['dinov3_fancy']:
            DISTILL_MEAN, DISTILL_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

        distill_transform = [transforms.ToTensor()]
        distill_transform.append(transforms.Normalize(DISTILL_MEAN, DISTILL_STD))
        distill_transform = transforms.Compose(distill_transform)

        if cfg['dataset_name'] == 'imagenet':
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
                    pin_memory=True,
                    persistent_workers=True)
                return loader, sampler

            train_transform = transforms.Compose([transforms.RandomResizedCrop(self.cfg['tokenizer']['input_size'], scale=(0.2, 1.0), interpolation=3),
                                transforms.RandomHorizontalFlip()])

            train_dataset = DistillImageFolder(
                root=os.path.join(cfg['working_root'], f"data/{cfg['dataset_name']}/train"),
                transform=train_transform,
                normalize_transform=our_transform,
                distill_transform=distill_transform
            )

            print(f'Train dataset: len={len(train_dataset)}')
            self.train_loader, train_sampler = make_distributed_loader(
                train_dataset, cfg['batch_size'], cfg['num_workers'], shuffle=True, drop_last=True)
            self.train_loader_length = len(self.train_loader)
            self.dist_samplers.append(train_sampler)
            

    def make_model(self):
        super().make_model()

        loaded_backbones = []
        self.distill = {}
        for model in self.cfg['distill_models']:
            cur_distill_location = model.split('_')[0]
            cur_dino_backbone = '_'.join(model.split('_')[1:])
            if cur_dino_backbone not in loaded_backbones:
                if 'dinov3' in model:
                    dino_code_path = os.path.join(self.cfg['working_root'], 'dinov3', 'dinov3')
                    dino_weights_path = os.path.join(self.cfg['working_root'], 'dinov3', f'{cur_dino_backbone}_pretrain.pth')
                    self.distill[cur_dino_backbone] = torch.hub.load(dino_code_path, cur_dino_backbone, source='local', pretrained=False)
                    dino_weights = torch.load(dino_weights_path, map_location='cpu', weights_only=True)
                    self.distill[cur_dino_backbone].load_state_dict(dino_weights, strict=True)
                for param in self.distill[cur_dino_backbone].parameters():
                    param.requires_grad = False
                self.distill[cur_dino_backbone].cuda()
                self.distill[cur_dino_backbone].eval()
                loaded_backbones.append(cur_dino_backbone)
            if cur_distill_location == 'EC':
                self.distill_encoder_cls = cur_dino_backbone
            if cur_distill_location == 'EP':
                self.distill_encoder_patch = cur_dino_backbone
            if cur_distill_location == 'DC':
                self.distill_decoder_cls = cur_dino_backbone
            if cur_distill_location == 'DP':
                self.distill_decoder_patch = cur_dino_backbone

        for distill_weight in self.cfg['distill_weights']:
            cur_distill_location = distill_weight.split('_')[0]
            cur_distill_weight = float(distill_weight.split('_')[1])
            if cur_distill_location == 'EC':
                self.distill_encoder_cls_weight = cur_distill_weight
            if cur_distill_location == 'EP':
                self.distill_encoder_patch_weight = cur_distill_weight
            if cur_distill_location == 'DC':
                self.distill_decoder_cls_weight = cur_distill_weight
            if cur_distill_location == 'DP':
                self.distill_decoder_patch_weight = cur_distill_weight

        if self.cfg['use_lpips']:
            self.lpips = lpips.LPIPS(net='vgg').requires_grad_(False).cuda()
            for param in self.lpips.parameters():
                param.requires_grad = False

        if self.cfg['use_koleo']:
            self.koleo_loss = KoLeoLoss()

    def _iter_step(self, cur_step, data, is_train):
        if len(data) == 4:
            _, clean_images, distill_normalized_images, _ = data
        elif len(data) == 2:
            clean_images, _ = data
            distill_normalized_images = clean_images.detach().clone()
        else:
            clean_images, distill_normalized_images, _ = data
        B = clean_images.shape[0]
        clean_images = clean_images.to(self.rank)
        distill_normalized_images = distill_normalized_images.to(self.rank)

        with torch.no_grad():
            #### get distillation features for Encoder CLS, Encoder patches, Decoder CLS, Decoder patches
            features_dicts = {}
            for cur_teacher_name, distill_model in self.distill.items():
                features_dicts[cur_teacher_name] = distill_model(distill_normalized_images, is_training=True)
            encoder_cls_features = features_dicts[self.distill_encoder_cls]["x_norm_clstoken"].unsqueeze(1)
            encoder_patch_features = features_dicts[self.distill_encoder_patch]["x_norm_patchtokens"]
            decoder_cls_features = features_dicts[self.distill_decoder_cls]["x_norm_clstoken"].unsqueeze(1)
            decoder_patch_features = features_dicts[self.distill_decoder_patch]["x_norm_patchtokens"]

        ret_dict = self.model_ddp(clean_images, encoder_distill_layer=self.cfg['encoder_distill_layer'], decoder_distill_layer=self.cfg['decoder_distill_layer'])
        hyponet = ret_dict['hyponet']
        hypocnn = ret_dict['hypocnn']
        distill_preds_raw = ret_dict['distill']
        distill_mses = 0.0
        if isinstance(distill_preds_raw, list):
            cls_weights = [self.distill_encoder_cls_weight, self.distill_decoder_cls_weight]
            patch_weights = [self.distill_encoder_patch_weight, self.distill_decoder_patch_weight]
            cls_features = [encoder_cls_features, decoder_cls_features]
            patch_features = [encoder_patch_features, decoder_patch_features]
            for t_idx, distill_preds in enumerate(distill_preds_raw):
                if cls_weights[t_idx] > 0.0:
                    distill_mses += (((distill_preds[0] - cls_features[t_idx])**2).view(B, -1).mean(dim=1).mean() * cls_weights[t_idx])
                else:
                    distill_mses += (distill_preds[0] * 0.0).mean(dim=1).mean()
                if patch_weights[t_idx] > 0.0:
                    distill_mses += (((distill_preds[1] - patch_features[t_idx])**2).view(B, -1).mean(dim=1).mean() * patch_weights[t_idx])
                else:
                    distill_mses += (distill_preds[1] * 0.0).mean(dim=1).mean()

        #assert self.cfg['is_patch_mode']

        if hypocnn is not None:
            upsample_factor = math.prod([int(strd) for strd in self.cfg['hyponet']['strides'].split('_')]) * math.prod([int(strd) for strd in self.cfg['hypocnn']['strds'].split('_')])
        else:
            upsample_factor = 1
        if not self.cfg['is_patch_mode']:
            coords = make_coord_grid([length // upsample_factor for length in clean_images.shape[-2:]], (-1, 1), device=clean_images.device)
            coords = einops.repeat(coords, 'h w d -> b h w d', b=B)
        elif self.cfg['coords_per_image']:
            coords = make_coord_grid([length // upsample_factor for length in clean_images.shape[-2:]], (-1, 1), device=clean_images.device)
            coords = einops.rearrange(coords, '(p1 h) (p2 w) d -> (p1 p2) h w d', 
                                   p1=int((self.cfg['tokenizer']['input_size'] / self.cfg['tokenizer']['patch_size'])), 
                                   p2=int((self.cfg['tokenizer']['input_size'] / self.cfg['tokenizer']['patch_size'])))
            coords = einops.repeat(coords, 'p h w d -> b p h w d', b=B)
        else:
            coords = make_coord_grid([self.cfg['tokenizer']['patch_size'] // upsample_factor, self.cfg['tokenizer']['patch_size'] // upsample_factor], (-1, 1), device=clean_images.device)
            coords = einops.repeat(coords, 'h w d -> (b p) h w d', b=B, p=int((self.cfg['tokenizer']['input_size'] / self.cfg['tokenizer']['patch_size']) ** 2)) 

        if self.cfg['coords_per_image']:
            coords = einops.rearrange(coords, 'b p h w d -> (b p) h w d').contiguous()
        output = hyponet(coords)
        patches = einops.rearrange(output, '(b p) h w c -> b p h w c', b=B)
        if patches.shape[1] > 1:
            output = einops.rearrange(patches, 'b (p1 p2) h w c -> b (p1 h) (p2 w) c', p1=int(np.sqrt(patches.shape[1])), p2=int(np.sqrt(patches.shape[1])))
        if hypocnn is not None:
            output = einops.rearrange(output, 'b h w c -> b c h w')
            output = hypocnn(output)
        else:
            output = einops.rearrange(output, 'b h w c -> b c h w').contiguous()
        assert output.shape[1] == 3
        assert clean_images.shape[1] == 3
        
        with torch.no_grad():
            if self.cfg['normalize_images']:
                denormalized_mses = (((output * 0.5 + 0.5) - (clean_images * 0.5 + 0.5))**2).view(B, -1).mean(dim=-1)
                psnr = (-10 * torch.log10(denormalized_mses)).mean()
            else:
                psnr = (-10 * torch.log10(((output - clean_images)**2).view(B, -1).mean(dim=-1))).mean()

        if self.cfg['use_lpips']:
            if not self.cfg['normalize_images']:
                lpips_loss = self.lpips(output * 2.0 - 1.0, clean_images * 2.0 - 1.0).mean()
            else:
                lpips_loss = self.lpips(output, clean_images).mean()
        if self.cfg['use_ssim']:
            if self.cfg['normalize_images']:
                ssim_loss = 1.0 - ssim(output * 0.5 + 0.5, clean_images * 0.5 + 0.5, data_range=1, size_average=False).mean()
            else:
                ssim_loss = 1.0 - ssim(output, clean_images, data_range=1, size_average=False).mean()
        if self.cfg['use_koleo']:
            koleo_loss = self.koleo_loss(ret_dict['cls_token'])

        mses = ((output - clean_images) ** 2).mean(dim=-1)
        loss = mses.mean()

        loss = loss + distill_mses
        if not isinstance(distill_mses, float):
            distill_loss = distill_mses.item()
        else:
            distill_loss = distill_mses

        if self.cfg['use_lpips']:
            loss = loss + lpips_loss * self.cfg['lpips_weight']

        if self.cfg['use_ssim']:
            loss = loss + ssim_loss * self.cfg['ssim_weight']

        if self.cfg['use_koleo']:
            loss = loss + koleo_loss * self.cfg['koleo_weight']

        if is_train:
            if cur_step % self.cfg['accum_iter'] == 0:
                if not self.cfg['skip_scheduler']:
                    learning_rate = adjust_learning_rate(self.optimizer,
                                                        cur_step,# + self.train_loader_length * (self.epoch - 1),
                                                        self.cfg['warmup_epochs'] * self.train_loader_length,
                                                        self.cfg['max_epochs'] * self.train_loader_length,
                                                        1e-7,
                                                        self.cfg['lr'] * self.cfg['batch_size'] * self.tot_gpus / 256
                    )
                else:
                    learning_rate = self.cfg['lr'] * self.cfg['batch_size'] * self.tot_gpus / 256

            if self.cfg['skip_scaler']:
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_ddp.parameters(), self.cfg['clip'])
                self.optimizer.step()
            else:
                loss /= self.cfg['accum_iter']
                self.scaler(loss, self.optimizer, clip_grad=self.cfg['clip'], 
                            parameters=self.model_ddp.parameters(), create_graph=False,
                            update_grad=(cur_step + 1) % self.cfg['accum_iter'] == 0)
                if (cur_step + 1) % self.cfg['accum_iter'] == 0:
                    self.optimizer.zero_grad()

            torch.cuda.synchronize()

        met_dict = {
            'loss': loss.item(),
            'distill_loss': distill_loss,
            'psnr': psnr.item(),
            'lr': learning_rate,
        }
        if self.cfg['use_lpips']:
            met_dict['lpips_loss'] = lpips_loss.item()
        if self.cfg['use_ssim']:
            met_dict['ssim_loss'] = ssim_loss.item()
        if self.cfg['use_koleo']:
            met_dict['koleo_loss'] = koleo_loss.item()
        return met_dict

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

        max_iters = cfg['max_epochs'] * self.train_loader_length

        self.model_ddp.train()
        if self.is_master:
            pbar = tqdm(range((self.epoch - 1) * self.train_loader_length, max_iters), desc='train', leave=False)
        train_loader = iter(self.train_loader)
        for cur_step in range((self.epoch - 1) * self.train_loader_length, max_iters):
            self.epoch = (cur_step // self.train_loader_length) + 1

            data = next(train_loader)
            
            ret = self.train_step(cur_step, data)
            if self.is_master:
                if 'contrastive_loss' in ret:
                    pbar.set_description(f"LR: {ret['lr']:.4f}, PSNR: {ret['psnr']:.4f}, Reconstruction Loss: {ret['loss']:.4f}, Contrastive Loss: {ret['contrastive_loss']:.4f}")
                elif 'psnr' in ret:
                    pbar.set_description(f"LR: {ret['lr']:.4f}, PSNR: {ret['psnr']:.4f}, Loss: {ret['loss']:.4f}")
                else:
                    pbar.set_description(f"LR: {ret['lr']:.4f}, Loss: {ret['loss']:.4f}")
                wandb.log(ret)
                pbar.update(1)

            if "debug" in self.cfg["exp_name"] and cur_step > 100:
                break

            if (cur_step + 1) % self.train_loader_length == 0 and cur_step > 0:
                train_loader = iter(self.train_loader)
                if self.epoch % self.cfg['ckpt_freq'] == 0:
                    self.save_checkpoint(os.path.join(cfg['working_root'], f"checkpoints/{self.cfg['dataset_name']}_{self.cfg['exp_name']}_epoch{self.epoch}.pth"))
                self.save_checkpoint(os.path.join(cfg['working_root'], f"checkpoints/{self.cfg['dataset_name']}_{self.cfg['exp_name']}_latest.pth"))
