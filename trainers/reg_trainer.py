import gc
import os
import torchvision
from torchvision import transforms
import torch
import torch.distributed as dist
import wandb
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


from models import Head, HyperNetwork
from .trainer import Trainer


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output


def sync_initial_weights(model):
    """
    Broadcasts model weights from rank 0 to all other ranks.
    """
    for param in model.parameters():
        dist.broadcast(param.data, src=0) # Broadcast the tensor data


class RegTrainer(Trainer):

    def __init__(self, rank, cfg):
        super().__init__(rank, cfg)
        if self.is_master:
            wandb.init(project="universal_vision_encoder", name=f"{cfg['dataset_name']}_{cfg['exp_name']}_{cfg['unique_type']}_reg")
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
            sampler = DistributedSampler(dataset, num_replicas=self.tot_gpus, rank=self.rank, shuffle=shuffle) if self.distributed else None
            loader = DataLoader(
                dataset,
                batch_size,
                drop_last=drop_last,
                sampler=sampler,
                shuffle=(shuffle and (sampler is None)),
                num_workers=num_workers,
                pin_memory=True)
            return loader, sampler

        train_transforms = [
            transforms.Resize(int(self.cfg['tokenizer']['input_size'] * 1.25), interpolation=3),
            transforms.CenterCrop(self.cfg['tokenizer']['input_size']),
            transforms.ToTensor()]
        val_transforms = [
            transforms.Resize(int(self.cfg['tokenizer']['input_size'] * 1.25), interpolation=3),
            transforms.CenterCrop(self.cfg['tokenizer']['input_size']),
            transforms.ToTensor()]
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
            train_dataset, cfg['batch_size'], cfg['num_workers'], shuffle=False, drop_last=False)
        self.val_loader, val_sampler = make_distributed_loader(
            val_dataset, 25, cfg['num_workers'], shuffle=False, drop_last=False)
        self.dist_samplers.append(train_sampler)
        self.dist_samplers.append(val_sampler)

    def make_model(self):
        if self.cfg['trainer'] == 'reg_hypernetwork':
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
        
        reg_head = Head(self.cfg, self.num_classes, type='logistic')

        self.epoch = 1
        self.optimizer_state = None
        if not os.path.exists(os.path.join(self.cfg['working_root'], "checkpoints")):
            os.mkdir(os.path.join(self.cfg['working_root'], "checkpoints"))

        reg_head.cuda()
        self.reg_head = reg_head

    def extract_features(self):
        self.backbone.eval()
        feature_bank, feature_labels = [], []
        with torch.no_grad():
            for k, (data, target) in enumerate(self.train_loader):
                data = data.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)
                feature = self.backbone(data, unique_only=True, feat_select=self.cfg['feat_select'])[self.cfg['unique_type']]
                feature = torch.nn.functional.normalize(feature, dim=1)
                feature = concat_all_gather(feature)
                target = concat_all_gather(target)
                feature_bank.append(feature)
                feature_labels.append(target)
                print(f"Extracting feature from {k}/{len(self.train_loader)} batches", flush=True)
            torch.cuda.empty_cache()
            print("gpu consuming before combining:", torch.cuda.memory_allocated() / 1024 / 1024)
            feature_bank = torch.cat(feature_bank, dim=0).contiguous()
            print("feature bank size: ",feature_bank.size())

            feature_labels = torch.cat(feature_labels, dim=0).contiguous()
            print("feature label size:",feature_labels.size())
        return feature_bank, feature_labels

    def train(self):
        """
            For epochs perform training, evaluation, and visualization.
            Note that ave_scalars update ignores the actual current batch_size.
        """
        cfg = self.cfg
        device = torch.device(self.rank)

        feature_bank, feature_labels = self.extract_features()

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.LBFGS(self.reg_head.parameters(), lr=cfg['lr'])
        if self.optimizer_state is not None:
            self.optimizer.load_state_dict(self.optimizer_state)

        patience_counter, patience = 0, 10
        best_val_acc = 0.0
        max_epochs = cfg['max_epochs']
        for epoch in range(self.epoch, max_epochs + 1):
            self.epoch = epoch
            
            self.backbone.eval()
            self.reg_head.train()

            def closure():
                self.optimizer.zero_grad()
                output = self.reg_head(feature_bank)
                loss = self.loss_fn(output, feature_labels)
                loss.backward()
                return loss
            if self.is_master:
                loss = self.optimizer.step(closure)

            if self.is_master:
                print(f"Epoch {epoch} loss: {loss.item()}")
                wandb.log({
                    "Train Loss": loss.item()
                })
           
            torch.cuda.empty_cache()

            print("Evaluating on validation set ...")
            sync_initial_weights(self.reg_head)
            self.reg_head.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for data, target in self.val_loader:
                    data, target = data.to(device), target.to(device)
                    feature = self.backbone(data, unique_only=True, feat_select=self.cfg['feat_select'])[self.cfg['unique_type']]
                    feature = torch.nn.functional.normalize(feature, dim=1)
                    output = self.reg_head(feature)
                    output = concat_all_gather(output)
                    target = concat_all_gather(target)
                    pred = output.argmax(dim=1)
                    correct += (pred == target).sum().item()
                    total += target.shape[0]
                gc.collect()
                torch.cuda.empty_cache()
            val_acc = correct / total

            if self.is_master:
                wandb.log({"Val Accuracy": val_acc},step=epoch)
                print(f"Epoch {epoch} validation accuracy: {val_acc:.4f}")
            if val_acc <= best_val_acc:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered")
                    break
            else:
                patience_counter = 0
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                if self.is_master:
                    self.save_checkpoint(os.path.join(
                        self.cfg['working_root'], 
                        f"checkpoints/{self.cfg['dataset_name']}_{self.cfg['exp_name']}_{self.cfg['unique_type']}_{self.cfg['ckpt_suffix']}_reg_head_latest.pth")
                    )

        if self.is_master:
            print(f"Best validation accuracy: {best_val_acc:.4f}", flush=True)
            wandb.log({"Best Val Accuracy": best_val_acc})

        del feature_bank
        del feature_labels
        gc.collect()
        torch.cuda.empty_cache()

    def save_checkpoint(self, filename):
        if not self.is_master:
            return
        model_sd = self.reg_head.state_dict()
        optimizer_sd = self.optimizer.state_dict()
        checkpoint = {
            'model': model_sd,
            'optimizer': optimizer_sd,
            'epoch': self.epoch + 1,
            'cfg': self.cfg,
        }
        torch.save(checkpoint, filename)
