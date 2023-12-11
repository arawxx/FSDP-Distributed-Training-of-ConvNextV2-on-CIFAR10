import os
import time
import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from convnextv2 import convnextv2_atto, convnextv2_huge
from dataset import CIFAR10DataModule
from distributed import setup, cleanup
from fsdp import FsdpPolicies
from utils import get_date_of_run
from flow import train, validation


def fsdp_main(args):
    # model = convnextv2_atto(num_classes=10)
    model = convnextv2_huge(num_classes=10)
    criteria = nn.CrossEntropyLoss()

    local_rank = int(os.environ['LOCAL_RANK'])
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    datamodule = CIFAR10DataModule()

    train_sampler = DistributedSampler(datamodule.train_dataset, rank=rank, num_replicas=world_size, shuffle=True)
    valid_sampler = DistributedSampler(datamodule.valid_dataset, rank=rank, num_replicas=world_size)

    setup(rank=rank, world_size=world_size)  # setup the distributed environment

    train_kwargs = {'batch_size': args.batch_size, 'sampler': train_sampler, 'collate_fn': datamodule.train_collate}
    valid_kwargs = {'batch_size': args.batch_size, 'sampler': valid_sampler, 'collate_fn': datamodule.valid_collate}

    cuda_kwargs = {'num_workers': 64, 'pin_memory': True, 'shuffle': False}
    train_kwargs.update(cuda_kwargs)
    valid_kwargs.update(cuda_kwargs)

    train_dataloader = DataLoader(datamodule.train_dataset, **train_kwargs)
    valid_dataloader = DataLoader(datamodule.valid_dataset, **valid_kwargs)

    torch.cuda.set_device(local_rank)

    # model is on CPU before input to FSDP
    model = FSDP(
        model,
        auto_wrap_policy=FsdpPolicies.auto_wrap_policy,
        mixed_precision=FsdpPolicies.mp_policy,
        sharding_strategy=FsdpPolicies.sharding_strategy,
        backward_prefetch=FsdpPolicies.backward_prefetch,
        device_id=torch.cuda.current_device(),
    )

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    if args.epochs < 30:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.gamma)
    else: 
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_valid_accuracy = -1.0
    curr_valid_accuracy = -1.0
    file_save_name = 'ConvNextV2-'

    if rank == 0:
        time_of_run = get_date_of_run()
        duration_tracking = []
        train_loss_tracking = []
        train_accu_tracking = []
        valid_loss_tracking = []
        valid_accu_tracking = []
        # training_start_time = time.time()

    # if rank == 0 and args.track_memory:
    #     mem_alloc_tracker = []
    #     mem_reserved_tracker = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        
        train_loss, train_accuracy = train(model, train_dataloader, rank, epoch, criteria, optimizer, train_sampler)
        
        if args.run_validation:
            with torch.no_grad(): valid_loss, curr_valid_accuracy = validation(model, valid_dataloader, rank, criteria)
        
        scheduler.step()

        if rank == 0:
            print(f"--> Epoch {epoch} completed. Saving training stats.")

            duration_tracking.append(time.time() - t0)

            train_loss_tracking.append(train_loss)
            train_accu_tracking.append(train_accuracy)

            if args.run_validation:
                valid_loss_tracking.append(valid_loss)
                valid_accu_tracking.append(curr_valid_accuracy)

            # if args.track_memory:
            #     mem_alloc_tracker.append(
            #         format_metrics_to_gb(torch.cuda.memory_allocated())
            #     )
            #     mem_reserved_tracker.append(
            #         format_metrics_to_gb(torch.cuda.memory_reserved())
            #     )
            print(f"Completed saving training stats.")

        # if args.save_model and curr_valid_accuracy > best_valid_accuracy:
        #     # save
        #     if rank == 0:
        #         print(f"--> Model saving procedure starting.")

        #     # save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        #     with FSDP.state_dict_type(
        #         model, FsdpPolicies.state_dict_type, FsdpPolicies.save_policy
        #     ):
        #         cpu_state = model.state_dict()

        #     if rank == 0:
        #         print(f"--> Saving the model...")
        #         curr_epoch = (
        #             "-" + str(epoch) + "-" + str(round(curr_valid_accuracy.item(), 4)) + ".pt"
        #         )
        #         print(f"--> Attempting to save the model with prefix {curr_epoch}")
        #         save_name = file_save_name + "-" + time_of_run + "-" + curr_epoch
        #         print(f"--> Saving the model with name: {save_name}")

        #         torch.save(cpu_state, save_name)

        if curr_valid_accuracy > best_valid_accuracy:
            best_valid_accuracy = curr_valid_accuracy
            if rank == 0:
                print(f"--> New Valid Accuracy Record: {best_valid_accuracy}")
    
        print()

    dist.barrier()
    cleanup()


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch FSDP Training w/ ConvNextV2')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=.002, metavar='LR',
                        help='learning rate (default: .002)')
    parser.add_argument('--gamma', type=float, default=0.9, metavar='M',
                        help='Learning rate step gamma (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    # parser.add_argument('--track_memory', action='store_false', default=True,
    #                     help='track the gpu memory')
    parser.add_argument('--run_validation', action='store_false', default=True,
                        help='running the validation')
    parser.add_argument('--save-model', action='store_false', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    fsdp_main(args)
