import torch
import torch.nn as nn

from utils.dataset import generate_auto_regressive_targets
from utils.types import TargetEnum, ModelEnum


class PerJointMSELoss(nn.Module):
    def __init__(self, number_joints, joint_dimension):
        super(PerJointMSELoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction="sum")
        self.number_joints = number_joints
        self.joint_dimension = joint_dimension

    def forward(self, pred, target):
        # Reshape [batch size, seq length, feature dimension] to [batch size, seq length, number joints, joint dimension]
        pred = pred.reshape(
            pred.size(0), pred.size(1), self.number_joints, self.joint_dimension
        )
        target = target.reshape(
            target.size(0), target.size(1), self.number_joints, self.joint_dimension
        )

        # Initialize total loss
        total_loss = 0.0

        # Compute MSE loss for each joint and sum them up
        for j in range(self.number_joints):
            joint_pred = pred[:, :, j, :]
            joint_target = target[:, :, j, :]
            total_loss += self.mse_loss(joint_pred, joint_target)

        # Average the loss over all joints
        average_loss = total_loss / self.number_joints

        return average_loss


def attention_learning_rate_schedule(optimizer, step, dimension):
    """
    Implements the custom learning rate schedule defined in the 'Attention is All You Need' paper.

    :param optimizer: PyTorch optimizer
    :param epoch: int, current epoch number
    :param dimension: int, model dimension
    """
    warmup = 10000 ** -1.5
    lr = dimension ** -0.5 * min(step ** -0.5, step * warmup)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_learning_rate(optimizer, epoch, drop_factor, drop_interval):
    """
    Adjusts the learning rate of the optimizer based on the epoch.

    :param optimizer: PyTorch optimizer
    :param epoch: int, current epoch number
    :param drop_factor: float, factor by which to reduce the learning rate
    :param drop_interval: int, number of epochs between learning rate reductions
    """
    if (epoch + 1) % drop_interval == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= drop_factor


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
        A tuple of the loss and the batch_size
    """
    epoch_val_loss = 0
    model.eval()
    with torch.no_grad():
        for batch_size, (src_seqs, tgt_seqs) in enumerate(datasets["validation"]):

            src_seqs, tgt_seqs = (
                src_seqs.to(device).float(),
                tgt_seqs.to(device).float(),
            )

            if (args.model == ModelEnum.LSTM_SEQ2SEQ.value) or (
                args.model == ModelEnum.LSTM_SEQ2SEQ_ATT.value
            ):
                ar_tgt = generate_auto_regressive_targets(src_seqs, tgt_seqs)
                outputs = model(src_seqs, ar_tgt)
            else:
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

    batch_size += 1
    return epoch_val_loss, batch_size
