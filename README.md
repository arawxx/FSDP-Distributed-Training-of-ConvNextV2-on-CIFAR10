# FSDP Distributed Training of ConvNextV2 on CIFAR10
A script for training the ConvNextV2 on CIFAR10 dataset using the FSDP technique for a distributed training scheme.
You can run the script using the `torchrun` with the `run.py` file, i.e.: `torchrun --nnodes 1 --nproc_per_node 2  run.py`

`run.py` script arguments include:
```
--batch-size
--epochs
--lr
--gamma
--no-cuda
--seed
--run_validation
--save-model
```

Additional info for the arguments can be seen using the `--help` argument.
