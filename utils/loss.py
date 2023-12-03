import torch

from utils.dataset import generate_auto_regressive_targets
from utils.types import TargetEnum


def compute_validation_loss(args, model, datasets, criterion, device, mask):
    """
    Function computes the validation loss for the epoch.

    Args:
        args: The args for the run
        model: The current model
        datasets: The datasets for the run
        criterion: The loss criterion
        device: The device to compile the data for
        mask: The mask for masked objective (optional)

    Returns:
        A tuple of the loss and the val_iter
    """
    epoch_val_loss = 0
    model.eval()
    with torch.no_grad():
        for val_iter, (src_seqs, tgt_seqs) in enumerate(datasets["validation"]):

            src_seqs, tgt_seqs = (
                src_seqs.to(device).float(),
                tgt_seqs.to(device).float(),
            )

            if args.target_type == TargetEnum.PRE_TRAIN.value:
                src_mask = mask.mask_joints(src_seqs)
                outputs = model(src_mask)
            else:
                outputs = model(src_seqs)

            if args.target_type == TargetEnum.AUTO_REGRESSIVE.value:
                loss = criterion(
                    outputs, generate_auto_regressive_targets(src_seqs, tgt_seqs)
                )
            elif args.target_type == TargetEnum.PRE_TRAIN.value:
                loss = criterion(outputs, src_seqs)
            else:
                loss = criterion(outputs, tgt_seqs)

            epoch_val_loss += loss.item()
            break  # TODO: Delete this

    model.train()
    return epoch_val_loss, val_iter
