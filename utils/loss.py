import math
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


def setup_attention_learning_rate_schedule(dimension):
    """
    Lambda function hack see: https://stackoverflow.com/questions/77398956/how-to-pass-a-parameter-of-a-lambda-functon-to-another-function-which-accepts-th
    """
    def attention_learning_rate_schedule(epoch, dimension=dimension):
        """
        Implements the custom learning rate schedule defined in the STT paper
        """
        warmup = 10000**-1.5
        D = dimension
        return D**-.5 * math.min(epoch**-.5, epoch * warmup)

    return attention_learning_rate_schedule


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
