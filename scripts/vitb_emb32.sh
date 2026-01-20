pip install ftfy regex einops pytorch_msssim lpips
pip install xformers==0.0.27.post2
pip install safetensors==0.5.3
pip install datasets

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
        --mod-idxs 1 --ckpt-freq 10 \
        --max-epochs 50 --warmup-epochs 5 --batch-size 64 --lr 0.0005 --clip 0.01 \
        --working-root $WORKING_ROOT

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
        --max-epochs 15 --batch-size 64 --lr 0.1 \
        --mod-idxs 1 \
        --unique-type global_token --ckpt-suffix latest \
        --working-root $WORKING_ROOT

python3 validate.py --trainer cls_hypernetwork --exp-name vitb --tag emb32 --dataset-name imagenet \
    --cfg cfgs/vitb_emb32.yaml --is-patch-mode --normalize-images \
    --distill-location encoder-decoder-separate --distill-mode dinov3_fancy \
    --distill-dims EC_1024-EP_1024-DC_1024-DP_1024 \
    --distill-models EC_dinov3_vitl16-EP_dinov3_vitl16-DC_dinov3_vitl16-DP_dinov3_vitl16 \
    --decoder-type pos_free \
    --batch-size 128 \
    --mod-idxs 1 \
    --unique-type global_token --ckpt-suffix latest \
    --working-root $WORKING_ROOT

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
        --max-epochs 100 --batch-size 8 --lr 0.1 \
        --mod-idxs 1 \
        --unique-type global_token_trans --ckpt-suffix latest \
        --working-root $WORKING_ROOT

python3 validate.py --trainer reg_hypernetwork --exp-name vitb --tag emb32 --dataset-name imagenet \
    --cfg cfgs/vitb_emb32.yaml --is-patch-mode --normalize-images \
    --distill-location encoder-decoder-separate --distill-mode dinov3_fancy \
    --distill-dims EC_1024-EP_1024-DC_1024-DP_1024 \
    --distill-models EC_dinov3_vitl16-EP_dinov3_vitl16-DC_dinov3_vitl16-DP_dinov3_vitl16 \
    --decoder-type pos_free \
    --batch-size 128 \
    --mod-idxs 1 \
    --unique-type global_token_trans --ckpt-suffix latest \
    --working-root $WORKING_ROOT
