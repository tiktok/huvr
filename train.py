import argparse
import socket
from contextlib import closing

import trainers
from utils import make_cfg, init_distributed_mode_torchrun


def get_open_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def main_worker(rank, cfg):
    if cfg["trainer"] == 'pre':
        trainer = trainers.PreTrainer(rank, cfg)
    elif cfg["trainer"] in ['cls_hypernetwork']:
        trainer = trainers.ClsTrainer(rank, cfg)
    elif cfg["trainer"] in ['reg_hypernetwork']:
        trainer = trainers.RegTrainer(rank, cfg)
    elif cfg["trainer"] == 'hypernetwork':
        trainer = trainers.HyperNetworkPreTrainer(rank, cfg)
    else:
        raise NotImplementedError(f"Trainer {cfg['trainer']} not implemented.")
    trainer.run()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default='cfgs/imagenet.yaml')
    parser.add_argument("--exp-name", type=str, default='hypernet')
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--dataset-name", type=str, default='imagenet')
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--warmup-epochs", type=int, default=None)
    parser.add_argument("--restart-epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None, help='learning rate')
    parser.add_argument("--hypo-layers", type=int, default=None)
    parser.add_argument("--hypo-hid-dim", type=int, default=None)
    parser.add_argument("--hyper-layers", type=int, default=None)
    parser.add_argument("--n-groups", type=int, default=None)
    parser.add_argument("--clip", type=float, default=None, help='gradient clipping')
    parser.add_argument("--ckpt-suffix", type=str, default=None)
    parser.add_argument("--unique-type", type=str, default=None)
    parser.add_argument("--trainer", type=str, default='pre')
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--is-patch-mode", action="store_true")
    parser.add_argument("--log-gradients", action="store_true")
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--opt-eps", type=float, default=1e-8)
    parser.add_argument("--mod-idxs", type=str, default=None)
    parser.add_argument("--normalize-images", action="store_true")
    parser.add_argument("--accum-iter", type=int, default=None)
    parser.add_argument("--use-lpips", action="store_true")
    parser.add_argument("--lpips-weight", type=float, default=None)
    parser.add_argument("--use-ssim", action="store_true")
    parser.add_argument("--ssim-weight", type=float, default=None)
    parser.add_argument("--use-koleo", action="store_true")
    parser.add_argument("--koleo-weight", type=float, default=None)
    parser.add_argument("--embedding-dim", type=int, default=None)
    parser.add_argument("--feat-select", type=int, default=None)
    parser.add_argument("--working-root", type=str, default="/path/to/working")
    parser.add_argument("--coords-per-image", action="store_true")
    parser.add_argument("--downstream-dataset-name", type=str, default="imagenet")
    parser.add_argument("--master_addr",default="localhost",type=str,help="master address for distributed training")
    parser.add_argument("--master_port",default="10001",type=str,help="master port for distributed training")
    parser.add_argument('--nproc_per_node', default=8, type=int, help='number of GPUs per node')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--ckpt-freq', type=int, default=50)
    parser.add_argument('--skip-scaler', action="store_true")
    parser.add_argument('--skip-scheduler', action="store_true")
    parser.add_argument('--decoder-type', type=str, default='old')
    parser.add_argument("--use-registers", action="store_true")
    parser.add_argument("--distill-mode", type=str, default=None)
    parser.add_argument("--distill-location", type=str, default=None)
    parser.add_argument("--distill-weights", type=str, default=None)
    parser.add_argument("--distill-models", type=str, default=None)
    parser.add_argument("--distill-dims", type=str, default=None)
    parser.add_argument("--pretrain-path", type=str, default=None)
    parser.add_argument("--encoder-distill-layer", type=int, default=None)
    parser.add_argument("--decoder-distill-layer", type=int, default=None)
    args = parser.parse_args()

    init_distributed_mode_torchrun(args)
    cfg = make_cfg(args)
    cfg['dist_mode'] = "torchrun"
    cfg['ngpus_per_node'] = args.nproc_per_node
    main_worker(args.gpu, cfg)


if __name__ == "__main__":
    main()
