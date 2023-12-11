import functools

import torch
import torch.cuda.nccl as nccl
import torch.distributed as dist
from torch.distributed.fsdp import (
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy, size_based_auto_wrap_policy

from convnextv2.convnextv2 import Block


fp16_policy = MixedPrecision(
    param_dtype=torch.float16,
    # Gradient communication precision.
    reduce_dtype=torch.float16,
    # Buffer precision.
    buffer_dtype=torch.float16,
)

bf16_policy = MixedPrecision(
    param_dtype=torch.bfloat16,
    # Gradient communication precision.
    reduce_dtype=torch.bfloat16,
    # Buffer precision.
    buffer_dtype=torch.bfloat16,
)

fp32_policy = MixedPrecision(
    param_dtype=torch.float32,
    # Gradient communication precision.
    reduce_dtype=torch.float32,
    # Buffer precision.
    buffer_dtype=torch.float32,
)


class FsdpPolicies:
    '''Model Wrapping Policy'''
    auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy,
        min_num_params=100,
        # transformer_auto_wrap_policy,
        # transformer_layer_cls={
        #     Block,
        # },
    )

    '''Sharding Strategy; SHARD_GRAD_OP for Zero2 and FULL_SHARD for Zero3'''
    # sharding_strategy: ShardingStrategy = ShardingStrategy.SHARD_GRAD_OP
    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD

    '''Mixed Precision Policy'''
    bf16_ready = (
        torch.version.cuda
        and torch.cuda.is_bf16_supported()
        # and LooseVersion(torch.version.cuda) >= "11.0"
        and dist.is_nccl_available()
        and nccl.version() >= (2, 10)
    )

    if bf16_ready:
        mp_policy = bf16_policy
    else:
        mp_policy = None  # defaults to fp32

    '''Modely Saving Policy'''
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    state_dict_type = StateDictType.FULL_STATE_DICT

    '''Backward Prefetch Strategy'''
    backward_prefetch = BackwardPrefetch.BACKWARD_PRE
