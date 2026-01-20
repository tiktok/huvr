import os
import yaml


def make_cfg(args):
    with open(args.cfg, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    if args.exp_name is None:
        exp_name = os.path.basename(args.cfg).split('.')[0]
    else:
        exp_name = args.exp_name
    if args.tag is not None:
        exp_name += '_' + args.tag
    cfg['exp_name'] = exp_name

    env = dict()
    env['tot_gpus'] = args.world_size
    env['dist_url'] = args.dist_url
    cfg['env'] = env

    cfg['dataset_name'] = args.dataset_name
    if args.hypo_layers is not None:
        cfg['hyponet']['depth'] = args.hypo_layers
    if args.hyper_layers is not None:
        cfg['transformer_encoder']['depth'] = args.hyper_layers
    if args.n_groups is not None:
        cfg['n_groups'] = args.n_groups
    if args.hypo_hid_dim is not None:
        cfg['hyponet']['hidden_dim'] = args.hypo_hid_dim
    if args.ckpt_suffix is not None:
        cfg['ckpt_suffix'] = args.ckpt_suffix
    if args.unique_type is not None:
        cfg['unique_type'] = args.unique_type
    if args.batch_size is not None:
        cfg['batch_size'] = args.batch_size
    if 'max_epochs' in args and args.max_epochs is not None:
        cfg['max_epochs'] = args.max_epochs
    if 'trainer' in args and args.trainer is not None:
        cfg['trainer'] = args.trainer
    if 'lr' in args and args.lr is not None:
        cfg['lr'] = args.lr
    if 'clip' in args and args.clip is not None:
        cfg['clip'] = args.clip
    if 'coords_per_image' in args and args.coords_per_image is not None:
        cfg['coords_per_image'] = args.coords_per_image
    else:
        cfg['coords_per_image'] = False
    if 'is_patch_mode' in args and args.is_patch_mode is not None:
        cfg['is_patch_mode'] = args.is_patch_mode
    else:
        cfg['is_patch_mode'] = False
    if 'log_gradients' in args and args.log_gradients is not None:
        cfg['log_gradients'] = args.log_gradients
    else:
        cfg['log_gradients'] = False
    if 'weight_decay' in args and args.weight_decay is not None:
        cfg['weight_decay'] = args.weight_decay
    elif 'weight_decay' not in cfg:
        cfg['weight_decay'] = 0.01
    if 'opt_eps' in args and args.opt_eps is not None:
        cfg['opt_eps'] = args.opt_eps
    if 'warmup_epochs' in args and args.warmup_epochs is not None:
        cfg['warmup_epochs'] = args.warmup_epochs
    if 'restart_epochs' in args and args.restart_epochs is not None:
        cfg['restart_epochs'] = args.restart_epochs
    if 'mod_idxs' in args and args.mod_idxs is not None:
        cfg['mod_idxs'] = args.mod_idxs
    if 'normalize_images' in args and args.normalize_images is not None:
        cfg['normalize_images'] = args.normalize_images
    else:
        cfg['normalize_images'] = False
    if 'distill_mode' in args and args.distill_mode is not None:
        cfg['distill_mode'] = args.distill_mode
    else:
        cfg['distill_mode'] = None
    if 'accum_iter' in args and args.accum_iter is not None:
        cfg['accum_iter'] = args.accum_iter
    elif 'accum_iter' not in cfg:
        cfg['accum_iter'] = 1
    if 'use_lpips' in args and args.use_lpips is not None:
        cfg['use_lpips'] = args.use_lpips
        if 'lpips_weight' in args and args.lpips_weight is not None:
            cfg['lpips_weight'] = args.lpips_weight
        elif 'lpips_weight' not in cfg:
            cfg['lpips_weight'] = 0.5
    else:
        cfg['use_lpips'] = False
    if 'use_ssim' in args and args.use_ssim is not None:
        cfg['use_ssim'] = args.use_ssim
        if 'ssim_weight' in args and args.ssim_weight is not None:
            cfg['ssim_weight'] = args.ssim_weight
        elif 'ssim_weight' not in cfg:
            cfg['ssim_weight'] = 0.5
    else:
        cfg['use_ssim'] = False
    if 'use_koleo' in args and args.use_koleo is not None:
        cfg['use_koleo'] = args.use_koleo
        if 'koleo_weight' in args and args.koleo_weight is not None:
            cfg['koleo_weight'] = args.koleo_weight
        elif 'koleo_weight' not in cfg:
            cfg['koleo_weight'] = 0.1
    else:
        cfg['use_koleo'] = False
    if 'embedding_dim' in args and args.embedding_dim is not None:
        cfg['embedding_dim'] = args.embedding_dim
    cfg['distill_location'] = args.distill_location
    if 'downstream_dataset_name' in args and args.downstream_dataset_name is not None:
        cfg['downstream_dataset_name'] = args.downstream_dataset_name
    cfg['feat_select'] = args.feat_select
    cfg['working_root'] = args.working_root
    if 'ckpt_freq' in args:
        cfg['ckpt_freq'] = args.ckpt_freq
    if 'skip_scaler' in args:
        cfg['skip_scaler'] = args.skip_scaler
    if 'skip_scheduler' in args:
        cfg['skip_scheduler'] = args.skip_scheduler
    else:
        cfg['skip_scheduler'] = False
    if 'decoder_type' in args:
        cfg['decoder_type'] = args.decoder_type
    if 'use_registers' in args:
        cfg['transformer_encoder']['use_registers'] = args.use_registers
    else:
        cfg['transformer_encoder']['use_registers'] = False
    if 'distill_models' in args and args.distill_models is not None:
        if 'distill_weights' in args and args.distill_weights is not None:
            cfg['distill_weights'] = args.distill_weights.split('-')
        else:
            cfg['distill_weights'] = None
        if 'distill_models' in args and args.distill_models is not None:
            cfg['distill_models'] = args.distill_models.split('-')
        else:
            cfg['distill_models'] = None
        assert 'fancy' in cfg['distill_mode']
        assert 'distill_dims' in args and args.distill_dims is not None
    else:
        cfg['distill_models'] = None
        cfg['distill_weights'] = None
    if 'distill_dims' in args and args.distill_dims is not None:
        cfg['distill_dim'] = args.distill_dims.split('-')
    if 'pretrain_path' in args and args.pretrain_path is not None:
        cfg['pretrain_path'] = args.pretrain_path
    else:
        cfg['pretrain_path'] = None
    if 'use_hypocnn' not in cfg:
        cfg['use_hypocnn'] = True
    if 'is_patch_mode' not in cfg:
        cfg['is_patch_mode'] = True
    if 'n_groups' not in cfg:
        cfg['n_groups'] = None
    if 'use_global_token' not in cfg:
        cfg['use_global_token'] = True
    if 'encoder_distill_layer' in args:
        cfg['encoder_distill_layer'] = args.encoder_distill_layer
    if 'decoder_distill_layer' in args:
        cfg['decoder_distill_layer'] = args.decoder_distill_layer

    return cfg