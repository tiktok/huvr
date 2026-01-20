### inspired by and adapted from https://github.com/yinboc/trans-inr/blob/f4bdc013286e2be00f9117e4e53913d6692fa49d/trainers/base_trainer.py
### see the TransINR license at documentation/TransINR_usage.md

import torch.backends.cudnn as cudnn
import torch.distributed as dist
import wandb

from utils import get_world_size, get_rank

class Trainer():

    def __init__(self, rank, cfg):
        self.rank = rank
        self.cfg = cfg

        env = cfg['env']
        self.tot_gpus = env['tot_gpus']
        self.distributed = (env['tot_gpus'] > 1)

        master_addr = env['dist_url']
        #print the information line
        information_line="master_addr: {},  local rank: {}, world size: {}, \
            ngpus_per_node: {}, mode rank: {}".format(master_addr,rank,env['tot_gpus'],cfg['ngpus_per_node'],rank)
        print(information_line)
        
        if self.distributed:
            num_tasks = get_world_size()
            global_rank = get_rank()
        else:
            global_rank = -1
            num_tasks = 1
        self.is_master  = (global_rank == 0) or (num_tasks == 1)
        print(f'is_master: {self.is_master}, global_rank: {global_rank}, num_tasks: {num_tasks}')

        cudnn.benchmark = True

        print(f'Environment setup done.')

    def run(self):
        self.make_datasets()

        self.make_model()
        self.train()

        if self.is_master:
            wandb.finish()

        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()

    def make_datasets(self):
        pass

    def make_model(self):
        pass

    def train(self):
        pass

    def dist_all_reduce_mean_(self, x):
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        x.div_(self.tot_gpus)

    def _iter_step(self, cur_step, data, is_train):
        pass

    def train_step(self, iter_step, data):
        return self._iter_step(iter_step, data, is_train=True)

    def train_epoch(self):
        pass

    def save_checkpoint(self, filename):
        pass
