# Implicit Neural Representation Facilitates Unified Universal Vision Encoding

## News

1.23.26 - Models available! <span>[Hugging Face](https://huggingface.co/collections/tiktok/huvr)</span>

1.21.26 - Paper available! <span>[arXiv](https://arxiv.org/abs/2601.14256)</span>

## Overview

This repository contains code and models for vision transformers that generate representations which not only do well for standard recognition tasks (classification, segmentation), but also support reconstruction generation. These models to operate well at standard embedding sizes (e.g. 768 dim for ViT-B) as well as with novel compressed embeddings (Tiny Tokens aka TinToks) which support the same tasks, but at 25-100x compressed size (8 dim, 16 dim, 32 dim).

## Pretrained models

Please follow the link to Hugging Face to download model weights for our open source ViT-B/16 and ViT-L/16 models.

<span>[Hugging Face](https://huggingface.co/collections/tiktok/huvr)</span>

## Installation

Training and evaluation was conducted with PyTorch 2.4.1, cuda 12.4, Python 3.11. Other versions may work. Install the following packages with pip:

```
pip install ftfy regex einops pytorch_msssim lpips
pip install xformers==0.0.27.post2
pip install safetensors==0.5.3
pip install datasets
pip install huggingface_hub
```

## Data Preparation

The code is compatible with ImageNet1k pretraining and evaluation. Simply ensure the data is formatted according to the ImageFolder specification.

## Training

Run `scripts/vitb_emb32.sh` for training and linear probe eval of our ViT-B/16 with tiny token dim=32. To only train, simply run:

```
torchrun --master_port=${MASTER_PORT} \
    --master_addr=${MASTER_ADDR} \
    --nproc_per_node=${NPROC_PER_NODE} \
    --nnodes=${NNODES} \
    --node_rank=${NODE_RANK} \
    train.py --trainer hypernetwork --exp-name vitb --tag emb32 --dataset-name imagenet \
        --cfg cfgs/vitb_emb32.yaml --is-patch-mode --normalize-images \
        --distill-location encoder-decoder-separate --distill-mode dinov3_fancy \
        --distill-dims EC_1024-EP_1024-DC_1024-DP_1024 \
        --distill-models EC_dinov3_vitl16-EP_dinov3_vitl16-DC_dinov3_vitl16-DP_dinov3_vitl16 \
        --distill-weights EC_4.0-EP_4.0-DC_1.0-DP_1.0 \
        --decoder-distill-layer 1 \
        --decoder-type pos_free \
        --use-koleo --koleo-weight 0.1 \
        --mask-ratio 0.0 \
        --mod-idxs 1 --ckpt-freq 5 \
        --max-epochs 50 --warmup-epochs 5 --batch-size 32 --lr 0.0005 --clip 0.01 \
        --working-root /path/to/working
```

## Evaluation

For linear probing, on standard-sized token:

```
torchrun --master_port=${MASTER_PORT} \
    --master_addr=${MASTER_ADDR} \
    --nproc_per_node=${NPROC_PER_NODE} \
    --nnodes=${NNODES} \
    --node_rank=${NODE_RANK} \
    train.py --trainer cls_hypernetwork --exp-name vitb --tag emb32 --dataset-name imagenet \
        --cfg cfgs/vitb_emb32.yaml --is-patch-mode --normalize-images \
        --distill-location encoder-decoder-separate --distill-mode dinov3_fancy \
        --distill-dims EC_1024-EP_1024-DC_1024-DP_1024 \
        --distill-models EC_dinov3_vitl16-EP_dinov3_vitl16-DC_dinov3_vitl16-DP_dinov3_vitl16 \
        --decoder-type pos_free \
        --max-epochs 15 --batch-size 32 --lr 0.1 \
        --mod-idxs 1 \
        --unique-type global_token --ckpt-suffix latest \
        --working-root /path/to/working

python3 validate.py --trainer cls_hypernetwork --exp-name vitb --tag emb32 --dataset-name imagenet \
    --cfg cfgs/vitb_emb32.yaml --is-patch-mode --normalize-images \
    --distill-location encoder-decoder-separate --distill-mode dinov3_fancy \
    --distill-dims EC_1024-EP_1024-DC_1024-DP_1024 \
    --distill-models EC_dinov3_vitl16-EP_dinov3_vitl16-DC_dinov3_vitl16-DP_dinov3_vitl16 \
    --decoder-type pos_free \
    --batch-size 32 \
    --mod-idxs 1 \
    --unique-type global_token --ckpt-suffix latest \
    --working-root /path/to/working
```

For regression, on tiny token:

```
torchrun --master_port=${MASTER_PORT} \
    --master_addr=${MASTER_ADDR} \
    --nproc_per_node=${NPROC_PER_NODE} \
    --nnodes=${NNODES} \
    --node_rank=${NODE_RANK} \
    train.py --trainer reg_hypernetwork --exp-name vitb --tag emb32 --dataset-name imagenet \
        --cfg cfgs/vitb_emb32.yaml --is-patch-mode --normalize-images \
        --distill-location encoder-decoder-separate --distill-mode dinov3_fancy \
        --distill-dims EC_1024-EP_1024-DC_1024-DP_1024 \
        --distill-models EC_dinov3_vitl16-EP_dinov3_vitl16-DC_dinov3_vitl16-DP_dinov3_vitl16 \
        --decoder-type pos_free \
        --max-epochs 100 --batch-size 32 --lr 0.1 \
        --mod-idxs 1 \
        --unique-type global_token_trans --ckpt-suffix latest \
        --working-root /path/to/working

python3 validate.py --trainer reg_hypernetwork --exp-name vitb --tag emb32 --dataset-name imagenet \
    --cfg cfgs/vitb_emb32.yaml --is-patch-mode --normalize-images \
    --distill-location encoder-decoder-separate --distill-mode dinov3_fancy \
    --distill-dims EC_1024-EP_1024-DC_1024-DP_1024 \
    --distill-models EC_dinov3_vitl16-EP_dinov3_vitl16-DC_dinov3_vitl16-DP_dinov3_vitl16 \
    --decoder-type pos_free \
    --batch-size 32 \
    --mod-idxs 1 \
    --unique-type global_token_trans --ckpt-suffix latest \
    --working-root /path/to/working
```

## License

For the license governing our code and weights, see License.md.

For the DINOv3 license (DINOv3 is used for distillation), see documentation/DINOv3_usage.md.

## Contributing

This implementation is shared primarily as a reference to clarify the ideas presented in the paper. Bug reports and contributions are welcome, but the project is not under active development.

## Citation

If you use our paper or code in your work, please cite:

```
@article{gwilliam2026HUVR,
  title={Accelerate High-Quality Diffusion Models with Inner Loop Feedback},
  author={Gwilliam, Matthew and Wang, Xiao and Hu, Xuefeng and Yang, Zhenheng},
  journal={arXiv preprint arXiv:2601.14256},
  year={2026}
}
```
