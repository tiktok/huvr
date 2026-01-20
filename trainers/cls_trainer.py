import os

import torchvision
from torchvision import transforms
import torch
import torch.nn as nn
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import gc

from models import Head, HyperNetwork
from .trainer import Trainer


class ClsTrainer(Trainer):

    def __init__(self, rank, cfg):
        super().__init__(rank, cfg)
        if self.is_master:
            wandb.init(project="universal_vision_encoder", name=f"{cfg['dataset_name']}_{cfg['exp_name']}_{cfg['unique_type']}_cls")
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

        train_transforms = [transforms.RandomResizedCrop((self.cfg['tokenizer']['input_size'], self.cfg['tokenizer']['input_size']), scale=(0.8, 1.0), interpolation=3), transforms.RandomHorizontalFlip(), transforms.ToTensor()]
        val_transforms = [transforms.Resize((self.cfg['tokenizer']['input_size']), interpolation=3), transforms.CenterCrop((self.cfg['tokenizer']['input_size'], self.cfg['tokenizer']['input_size'])), transforms.ToTensor()]
        if cfg['normalize_images']:
            train_transforms.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
            val_transforms.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

        if cfg['downstream_dataset_name'] is None or cfg['downstream_dataset_name'] == 'imagenet':
            train_dataset = torchvision.datasets.ImageFolder(
                root=os.path.join(self.cfg['working_root'], "data/imagenet/train"),
                transform=transforms.Compose(train_transforms)
            )
            val_dataset = torchvision.datasets.ImageFolder(
                root=os.path.join(self.cfg['working_root'], "data/imagenet/val"),
                transform=transforms.Compose(val_transforms)
            )
            self.num_classes = 1000
        
        print(f'Train dataset: len={len(train_dataset)}')
        self.train_loader, train_sampler = make_distributed_loader(
            train_dataset, cfg['batch_size'], cfg['num_workers'], shuffle=True, drop_last=True)
        self.val_loader, val_sampler = make_distributed_loader(
            val_dataset, 25, cfg['num_workers'], shuffle=False, drop_last=False)
        self.dist_samplers.append(train_sampler)
        self.dist_samplers.append(val_sampler)
        self.val_dataset_length = len(self.val_loader.dataset)

    def make_model(self):
        if self.cfg['trainer'] == 'cls_hypernetwork':
            backbone = HyperNetwork(self.cfg['tokenizer'], self.cfg['hyponet'], self.cfg['hypocnn'], self.cfg['transformer_encoder'], self.cfg['transformer_decoder'],
                                            self.cfg['transformer_transcoder'], self.cfg['embedding_dim'], self.cfg['mod_idxs'], 
                                            self.cfg['distill_dim'], self.cfg['distill_mode'], self.cfg['distill_location'], self.cfg['decoder_type'],
                                            self.cfg['use_hypocnn'], self.cfg['is_patch_mode'], self.cfg['n_groups'], self.cfg['use_global_token'])

        checkpoint = torch.load(os.path.join(self.cfg['working_root'], f"checkpoints/{self.cfg['dataset_name']}_{self.cfg['exp_name']}_{self.cfg['ckpt_suffix']}.pth"))
        model_state = checkpoint["model"]
        if self.cfg['pretrain_path'] is not None:
            model_state = {k: v for k, v in model_state.items() if "pos_embed" not in k}
            model_state = {k: v for k, v in model_state.items() if "decoder_posemb" not in k}
        missing = backbone.load_state_dict(model_state, strict=False)
        del checkpoint
        gc.collect()
        torch.cuda.empty_cache()
        if self.is_master:
            print(f'missing keys: {missing}')
        for param in backbone.parameters():
            param.requires_grad = False
        backbone.cuda()
        self.backbone = backbone

        cls_head = Head(self.cfg, self.num_classes)

        self.epoch = 1
        self.optimizer_state = None
        self.scheduler_state = None
        if not os.path.exists(os.path.join(self.cfg['working_root'], "checkpoints")):
            os.mkdir(os.path.join(self.cfg['working_root'], "checkpoints"))

        if self.distributed:
            cls_head = nn.SyncBatchNorm.convert_sync_batchnorm(cls_head)
            cls_head.cuda()
            cls_head_ddp = DistributedDataParallel(cls_head, device_ids=[self.rank])
        else:
            cls_head.cuda()
            cls_head_ddp = cls_head
        self.cls_head = cls_head
        self.cls_head_ddp = cls_head_ddp

    def train(self):
        """
            For epochs perform training, evaluation, and visualization.
            Note that ave_scalars update ignores the actual current batch_size.
        """
        cfg = self.cfg

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.cls_head_ddp.parameters(), lr=cfg['lr'] * cfg['batch_size'] * cfg['env']['tot_gpus'] / 256)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 7, 0.1)
        if self.optimizer_state is not None:
            self.optimizer.load_state_dict(self.optimizer_state)
        if self.scheduler_state is not None:
            self.scheduler.load_state_dict(self.scheduler_state)

        max_epochs = cfg['max_epochs']

        for epoch in range(self.epoch, max_epochs + 1):
            self.epoch = epoch

            if self.distributed:
                for sampler in self.dist_samplers:
                    sampler.set_epoch(epoch)
            
            self.train_epoch()
            self.scheduler.step()

            self.val_epoch()

            self.save_checkpoint(os.path.join(self.cfg['working_root'], f"checkpoints/{self.cfg['dataset_name']}_{self.cfg['exp_name']}_{self.cfg['unique_type']}_{self.cfg['ckpt_suffix']}_cls_head_latest.pth"))

    def _iter_step(self, cur_step, data, is_train):
        images, labels = data
        B = images.shape[0]
        images = images.cuda()
        labels = labels.cuda()

        unique_params = self.backbone(images, unique_only=True, feat_select=self.cfg['feat_select'])
        preds = self.cls_head_ddp(unique_params[self.cfg['unique_type']])

        loss = self.loss_fn(preds, labels)
        num_correct = torch.sum(preds.argmax(dim=1) == labels)
        acc = num_correct / B

        if is_train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            return {'loss': loss.item(), 'acc': acc.item(), 'lr': self.scheduler.get_last_lr()[0]}
        else:
            return {'loss': loss.item(), 'acc': acc.item(), 'num_correct': num_correct}

    def train_step(self, iter_step, data):
        return self._iter_step(iter_step, data, is_train=True)
    
    def val_step(self, iter_step, data):
        return self._iter_step(iter_step, data, is_train=False)

    def train_epoch(self):
        self.backbone.eval()
        self.cls_head_ddp.train()

        pbar = self.train_loader
        if self.is_master:
            pbar = tqdm(pbar, desc='train', leave=False)

        for cur_step, data in enumerate(pbar):
            ret = self.train_step(cur_step, data)
            if self.is_master:
                pbar.set_description(f"Accuracy: {ret['acc']:.4f}, Loss: {ret['loss']:.4f}")
                wandb.log({
                    "Train Loss": ret["loss"],
                    "Train Accuracy": ret["acc"],
                    "lr": ret["lr"],
                })

    def val_epoch(self):
        self.backbone.eval()
        self.cls_head_ddp.eval()

        pbar = self.val_loader
        if self.is_master:
            pbar = tqdm(pbar, desc='val', leave=False)

        total_num_correct = 0
        for cur_step, data in enumerate(pbar):
            ret = self.val_step(cur_step, data)
            if self.is_master:
                pbar.set_description(f"Loss: {ret['loss']:.4f}")
            num_corrects = [torch.zeros(1, dtype=torch.long, device=torch.device(f"cuda:{self.rank}")) for _ in range(self.tot_gpus)]
            torch.distributed.all_gather(num_corrects, ret['num_correct'])
            total_num_correct += torch.stack(num_corrects, dim=0).sum().item()
        
        if self.is_master:
            wandb.log({
                "Val Loss": ret["loss"],
                "Val Accuracy": total_num_correct / self.val_dataset_length,
            })
            print(f"Val Accuracy: {total_num_correct / self.val_dataset_length}")

    def save_checkpoint(self, filename):
        if not self.is_master:
            return
        model_sd = self.cls_head.state_dict()
        optimizer_sd = self.optimizer.state_dict()
        scheduler_sd = self.scheduler.state_dict()
        checkpoint = {
            'model': model_sd,
            'optimizer': optimizer_sd,
            'scheduler': scheduler_sd,
            'epoch': self.epoch + 1,
            'cfg': self.cfg,
        }
        torch.save(checkpoint, filename)
