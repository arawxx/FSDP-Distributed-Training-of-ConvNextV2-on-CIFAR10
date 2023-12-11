import torch
import torch.distributed as dist

from tqdm import tqdm


def train(model, train_dataloader, rank, epoch, criteria, optimizer, sampler=None):
    model.train()
    fsdp_loss = torch.zeros(2).to(rank)
    fsdp_accu = torch.zeros(2).to(rank)

    if sampler:
        sampler.set_epoch(epoch)
    
    if rank == 0:
        inner_progress_bar = tqdm(range(len(train_dataloader)), colour='blue', desc=f'rank_0 epoch {epoch:5d}')

    for batch in train_dataloader:
        for key in batch:
            batch[key] = batch[key].to(rank)
        
        optimizer.zero_grad(set_to_none=True)  # set_to_none=True is needed for bfloat16

        model_predictions: torch.Tensor = model(batch['images'])
        actual_outputs: torch.Tensor = batch['labels']

        loss = criteria(model_predictions, actual_outputs)
        loss.backward()

        optimizer.step()

        fsdp_loss[0] += loss.item()
        fsdp_loss[1] += len(batch['labels'])

        fsdp_accu[0] += (model_predictions.argmax(dim=1) == actual_outputs.argmax(dim=1)).sum().item()
        fsdp_accu[1] += len(batch['labels'])

        if rank == 0:
            inner_progress_bar.update(1)
    
    dist.all_reduce(fsdp_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(fsdp_accu, op=dist.ReduceOp.SUM)

    train_loss = fsdp_loss[0] / fsdp_loss[1]
    train_accu = fsdp_accu[0] / fsdp_accu[1]

    if rank == 0:
        inner_progress_bar.close()
        print(f'Train Epoch {epoch:5d}, Loss: {train_loss:.4f}, Accuracy: {train_accu:.4f}')
        print('==='*45)
    
    return train_loss, train_accu
