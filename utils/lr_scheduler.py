### LR Scheduler from https://github.com/facebookresearch/mae/blob/main/util/lr_sched.py
### see the MAE license at https://github.com/facebookresearch/mae/blob/main/LICENSE

import math

def adjust_learning_rate(optimizer, iter, warmup_iters, max_iters, min_lr, base_lr, restart_iters=0):
    if iter > restart_iters:
        iter = iter - restart_iters

    if iter < warmup_iters:
        lr = base_lr * iter / warmup_iters 
    else:
        lr = min_lr + (base_lr - min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (iter - warmup_iters) / (max_iters - warmup_iters)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr