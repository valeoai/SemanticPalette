import os
import numpy as np
import torch
import torch.distributed as dist

from data import create_dataset, create_synthetic_dataset, create_semi_synthetic_dataset

try:
    from apex.parallel import DistributedDataParallel, SyncBatchNorm
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex")


class Engine(object):
    def __init__(self, opt):
        self.devices = None
        self.distributed = False

        self.opt = opt

        if 'WORLD_SIZE' in os.environ:
            self.distributed = int(os.environ['WORLD_SIZE']) > 1

        if self.distributed:
            self.local_rank = self.opt.local_rank
            self.world_size = int(os.environ['WORLD_SIZE'])
            torch.cuda.set_device(self.local_rank)
            dist.init_process_group(backend="nccl", init_method='env://')
            self.devices = [i for i in range(self.world_size)]
        else:
            gpus = os.environ["CUDA_VISIBLE_DEVICES"]
            self.devices =  [i for i in range(len(gpus.split(',')))]
            self.world_size = 1

        print(f"Distributed {self.opt.local_rank}: {self.distributed}")

    def data_parallel(self, model):
        if self.distributed:
            model = DistributedDataParallel(model, delay_allreduce=True)
        # else:
        #     model = torch.nn.DataParallel(model)
        return model

    def create_dataset(self, opt, load_seg=True, load_img=True, phase="train", is_synthetic=False, is_semi=False):
        if is_synthetic:
            if is_semi:
                return create_semi_synthetic_dataset(opt, batch_size=self.opt.batch_size // self.world_size, phase=phase)
            else:
                return create_synthetic_dataset(opt, batch_size=self.opt.batch_size // self.world_size, phase=phase)
        else:
            opt = opt["base"] if type(opt) is dict else opt
            return create_dataset(opt, load_seg=load_seg, load_img=load_img, phase=phase)

    def create_dataloader(self, dataset, batch_size=None, num_workers=1, is_train=True, is_synthetic=False):
        datasampler = None

        if is_synthetic:
            self.batch_size_per_gpu = dataset.batch_size
            drop_last = False
            is_shuffle = False
            batch_size = None
            num_workers = 0
            pin_memory = False

        else:
            is_shuffle = is_train
            drop_last = is_train
            pin_memory = True
            if self.distributed:
                datasampler = torch.utils.data.distributed.DistributedSampler(dataset)
                batch_size = batch_size // self.world_size
                is_shuffle = False
            self.batch_size_per_gpu = batch_size

        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=batch_size,
                                                 num_workers=num_workers,
                                                 drop_last=drop_last,
                                                 shuffle=is_shuffle,
                                                 pin_memory=pin_memory,
                                                 sampler=datasampler,
                                                 worker_init_fn=lambda _: np.random.seed())

        return dataloader, datasampler

    def all_reduce_tensor(self, tensor, norm=True):
        if self.distributed:
            return all_reduce_tensor(tensor, world_size=self.world_size, norm=norm)
        else:
            return tensor

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        torch.cuda.empty_cache()
        if type is not None:
            print("An exception occurred during Engine initialization, "
                  "give up running process")
            return False

def all_reduce_tensor(tensor, op=dist.ReduceOp.SUM, world_size=1, norm=True):
    tensor = tensor.clone()
    dist.all_reduce(tensor, op)
    if norm:
        tensor.div_(world_size)
    return tensor