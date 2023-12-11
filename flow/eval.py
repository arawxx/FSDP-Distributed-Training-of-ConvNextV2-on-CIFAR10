import torch
import torch.distributed as dist

from tqdm import tqdm


def validation(model, valid_dataloader, rank, criteria):
    model.eval()
    fsdp_loss = torch.zeros(2).to(rank)
    fsdp_accu = torch.zeros(2).to(rank)
    
    if rank == 0:
        inner_progress_bar = tqdm(range(len(valid_dataloader)), colour='green', desc='validation epoch')

    for batch in valid_dataloader:
        for key in batch:
            batch[key] = batch[key].to(rank)

        model_predictions: torch.Tensor = model(batch['images'])
        actual_outputs: torch.Tensor = batch['labels']

        loss = criteria(model_predictions, actual_outputs)

        fsdp_loss[0] += loss.item()
        fsdp_loss[1] += len(batch['labels'])

        fsdp_accu[0] += (model_predictions.argmax(dim=1) == actual_outputs.argmax(dim=1)).sum().item()
        fsdp_accu[1] += len(batch['labels'])

        if rank == 0:
            inner_progress_bar.update(1)
    
    dist.all_reduce(fsdp_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(fsdp_accu, op=dist.ReduceOp.SUM)

    valid_loss = fsdp_loss[0] / fsdp_loss[1]
    valid_accu = fsdp_accu[0] / fsdp_accu[1]

    if rank == 0:
        inner_progress_bar.close()
        print(f'Validation Epoch, Loss: {valid_loss:.4f}, Accuracy: {valid_accu:.4f}')
        print('==='*45)
    
    return valid_loss, valid_accu
