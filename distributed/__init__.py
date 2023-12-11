import os
import torch.distributed as dist


def setup(rank, world_size):
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()
