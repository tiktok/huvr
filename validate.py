import argparse
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import einops
import math
import os
import json
from tqdm import tqdm
import wandb
import numpy as np
import gc

from models import Head, HyperNetwork
from utils import make_coord_grid, make_cfg

from pytorch_msssim import ssim
import lpips

def main(args):
    cfg = make_cfg(args)

    wandb.init(project="universal_vision_encoder", name=f"{args.dataset_name}_{cfg['exp_name']}_{args.unique_type}_validate")
    wandb.config.update(args)

    if '_' in args.ckpt_suffix:
        ckpt_suffixes = args.ckpt_suffix.split('_')
    else:
        ckpt_suffixes = [args.ckpt_suffix]

    if args.dataset_name in ['imagenet']:
        val_transforms = [transforms.Resize((cfg['tokenizer']['input_size']), interpolation=3), transforms.CenterCrop((cfg['tokenizer']['input_size'], cfg['tokenizer']['input_size'])), transforms.ToTensor()]
        if cfg['normalize_images']:
            val_transforms.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
        val_dataset = torchvision.datasets.ImageFolder(
            root=os.path.join(cfg['working_root'], "data/imagenet/val"),
            transform=transforms.Compose(val_transforms)
        )
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=22, shuffle=False, drop_last=False)
        num_classes = 1000

    for ckpt_suffix in ckpt_suffixes:
        if cfg['trainer'] == 'cls_hypernetwork' or cfg['trainer'] == 'reg_hypernetwork':
            backbone = HyperNetwork(cfg['tokenizer'], cfg['hyponet'], cfg['hypocnn'], cfg['transformer_encoder'], cfg['transformer_decoder'],
                                            cfg['transformer_transcoder'], cfg['embedding_dim'], cfg['mod_idxs'],
                                            cfg['distill_dim'], cfg['distill_mode'], cfg['distill_location'], cfg['decoder_type'],
                                            cfg['use_hypocnn'], cfg['is_patch_mode'], cfg['n_groups'], cfg['use_global_token'])

        checkpoint = torch.load(os.path.join(cfg['working_root'], f"checkpoints/{args.dataset_name}_{cfg['exp_name']}_{ckpt_suffix}.pth"))
        model_state = checkpoint["model"]
        if cfg['pretrain_path'] is not None:
            model_state = {k: v for k, v in model_state.items() if "pos_embed" not in k}
            model_state = {k: v for k, v in model_state.items() if "decoder_posemb" not in k}
        missing = backbone.load_state_dict(model_state, strict=False)
        del checkpoint
        gc.collect()
        torch.cuda.empty_cache()
        print(f'missing keys: {missing}')
        backbone = backbone.cuda()
        ### freeze backbone
        for param in backbone.parameters():
            param.requires_grad = False

        if cfg['unique_type'] is not None:
            if 'reg_' in cfg['trainer']:
                cls_head = Head(cfg, num_classes, type='logistic')
                cls_head.load_state_dict(torch.load(os.path.join(cfg['working_root'], f"checkpoints/{args.dataset_name}_{cfg['exp_name']}_{args.unique_type}_{ckpt_suffix}_reg_head_latest.pth"))["model"])
            else:
                cls_head = Head(cfg, num_classes)
                cls_head.load_state_dict(torch.load(os.path.join(cfg['working_root'], f"checkpoints/{args.dataset_name}_{cfg['exp_name']}_{args.unique_type}_{ckpt_suffix}_cls_head_latest.pth"))["model"])
            cls_head = cls_head.cuda()
            for param in cls_head.parameters():
                param.requires_grad = False

        lpips_fn = lpips.LPIPS(net='vgg').cuda()
        for param in lpips_fn.parameters():
            param.requires_grad = False

        ### init tqdm
        pbar = tqdm(len(val_dataloader))
  
        backbone.eval()
        if cfg['unique_type'] is not None:
            cls_head.eval()

        all_preds = []
        total_correct, total_psnr, total_ssim, total_lpips = 0, 0.0, 0.0, 0.0
        for _, (images, labels) in enumerate(val_dataloader):
            B = images.shape[0]
            images = images.cuda()
            labels = labels.cuda()
            
            if cfg['unique_type'] is not None:
                unique_params = backbone(images, unique_only=True, feat_select=cfg['feat_select'])
                if 'global' in cfg['unique_type'] or 'distill' in cfg['unique_type']:
                    if 'reg_' in cfg['trainer']:
                        preds = cls_head(torch.nn.functional.normalize(unique_params[cfg['unique_type']], dim=1))
                    else:
                        preds = cls_head(unique_params[cfg['unique_type']])

                all_preds.extend(preds.argmax(dim=1).tolist())
                acc = torch.sum(preds.argmax(dim=1) == labels) / B
                total_correct += (torch.sum(preds.argmax(dim=1) == labels))

            backbone_ret = backbone(images)
            ## check if it's a dictionary
            if isinstance(backbone_ret, dict):
                if 'hyponet' in backbone_ret:
                    hyponet = backbone_ret['hyponet']
                else:
                    output = backbone_ret['image']
                    p = cfg['tokenizer']['patch_size']
                    h = w = images.shape[2] // p
                    pred_images = output.reshape(shape=(output.shape[0], h, w, p, p, 3))
                    pred_images = torch.einsum('nhwpqc->nchpwq', pred_images)
                    pred_images = pred_images.reshape(shape=(pred_images.shape[0], 3, h * p, w * p))
                    output = einops.rearrange(pred_images, 'b c h w -> b h w c').contiguous()
                if 'hypocnn' in backbone_ret:
                    hypocnn = backbone_ret['hypocnn']
            else:
                hyponet = backbone_ret
            if hypocnn is not None:
                upsample_factor = math.prod([int(strd) for strd in cfg['hyponet']['strides'].split('_')]) * math.prod([int(strd) for strd in cfg['hypocnn']['strds'].split('_')])
            else:
                upsample_factor = 1
            coords = make_coord_grid([cfg['tokenizer']['patch_size'] // upsample_factor, cfg['tokenizer']['patch_size'] // upsample_factor], (-1, 1), device=images.device)
            coords = einops.repeat(coords, 'h w d -> (b p) h w d', b=B, p=int((cfg['tokenizer']['input_size'] / cfg['tokenizer']['patch_size']) ** 2)) 
            output = hyponet(coords)
            patches = einops.rearrange(output, '(b p) h w c -> b p h w c', b=B)
            output = einops.rearrange(patches, 'b (p1 p2) h w c -> b (p1 h) (p2 w) c', p1=int(np.sqrt(patches.shape[1])), p2=int(np.sqrt(patches.shape[1])))
            output = einops.rearrange(output, 'b h w c -> b c h w')
            output = hypocnn(output)
            output = einops.rearrange(output, 'b c h w -> b h w c').contiguous()

            images = einops.rearrange(images, 'b c h w -> b h w c')

            denormalized_mses = (((output * 0.5 + 0.5) - (images * 0.5 + 0.5))**2).view(B, -1).mean(dim=-1)
            psnr = (-10 * torch.log10(denormalized_mses)).mean()
            cur_ssim = ssim(output * 0.5 + 0.5, images * 0.5 + 0.5, data_range=1, size_average=False).mean()
            lpips_output = -1 + ((output * 0.5 + 0.5) * 2) ## this is a no-op
            lpips_images = -1 + ((images * 0.5 + 0.5) * 2) ## this is a no-op
            lpips_output = einops.rearrange(lpips_output, 'b h w c -> b c h w')
            lpips_images = einops.rearrange(lpips_images, 'b h w c -> b c h w')
            cur_lpips = lpips_fn(lpips_output, lpips_images).mean()
                    
            total_psnr += (psnr * B / len(val_dataset))
            total_ssim += (cur_ssim * B / len(val_dataset))
            total_lpips += (cur_lpips * B / len(val_dataset))

            ### update pbar
            if cfg['unique_type'] is not None:
                pbar.set_description(f"Acc: {acc.item():.4f}, PSNR: {psnr.item():.4f}, SSIM: {cur_ssim.item():.4f}")
            else:
                pbar.set_description(f"PSNR: {psnr.item():.4f}, SSIM: {cur_ssim.item():.4f}")
            pbar.update(1)
        
        real_correct, real_total = 0, 0
        if args.dataset_name in ['imagenet'] :
            with open(os.path.join(cfg['working_root'], 'data', 'imagenet', 'real.json'), 'r') as f:
                real_labels = json.load(f)
            real_labels = {f'ILSVRC2012_val_{i + 1:08d}.JPEG': labels for i, labels in enumerate(real_labels)}
            for sample_idx, pred in enumerate(all_preds):
                filename = os.path.basename(val_dataset.imgs[sample_idx][0])
                if real_labels[filename]:
                    real_correct += (pred in real_labels[filename])
                    real_total += 1

        if cfg['unique_type'] is not None:
            total_acc = total_correct.item() / len(val_dataset)
            real_acc = real_correct / real_total
            print(f"Checkpoint : {ckpt_suffix}, ReaL Accuracy: {real_acc:.4f}, Accuracy: {total_acc:.4f}, PSNR: {total_psnr.item():.4f}, SSIM: {total_ssim.item():.4f}, LPIPS: {total_lpips.item():.4f}")
            wandb.log({
                f"ReaL Accuracy": real_acc,
                f"Accuracy": total_acc,
                f"PSNR": total_psnr.item(),
                f"SSIM": total_ssim.item(),
                f"LPIPS": total_lpips.item(),
            })
        else:
            print(f"Checkpoint : {ckpt_suffix}, PSNR: {total_psnr.item():.4f}, SSIM: {total_ssim.item():.4f}, LPIPS: {total_lpips.item():.4f}")
            wandb.log({
                f"PSNR": total_psnr.item(),
                f"SSIM": total_ssim.item(),
                f"LPIPS": total_lpips.item(),
            })


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default='hypernet')
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--dataset-name", type=str, default='cifar')
    parser.add_argument("--cfg", type=str, default='cfgs/cifar.yaml')
    parser.add_argument("--ckpt-suffix", type=str, default='latest')
    parser.add_argument("--ngroups", type=int, default=None)
    parser.add_argument("--hypo-hid-dim", type=int, default=None)
    parser.add_argument("--hyper-layers", type=int, default=None)
    parser.add_argument("--hypo-layers", type=int, default=None)
    parser.add_argument("--n-groups", type=int, default=None)
    parser.add_argument("--unique-type", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--is-patch-mode", action='store_true')
    parser.add_argument("--trainer", type=str, default=None)
    parser.add_argument("--normalize-images", action='store_true')
    parser.add_argument("--distill-mode", type=str, default="global")
    parser.add_argument("--mod-idxs", type=str, default=None)
    parser.add_argument("--distill-location", type=str, default=None)
    parser.add_argument("--feat-select", type=int, default=None)
    parser.add_argument("--working-root", type=str, default="/path/to/working")
    parser.add_argument("--coords-per-image", action="store_true")
    parser.add_argument("--decoder-type", type=str, default='old')
    parser.add_argument("--downstream-dataset-name", type=str, default="imagenet")
    parser.add_argument("--use-registers", action="store_true")
    parser.add_argument("--distill-dims", type=str, default=None)
    parser.add_argument("--pretrain-path", type=str, default=None)
    parser.add_argument("--distill-models", type=str, default=None)
    args = parser.parse_args()
    args.world_size = 1
    args.dist_url = None
    main(args)