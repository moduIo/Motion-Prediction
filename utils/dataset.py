import os
import sys
import torch

sys.path.append(os.path.join(os.getcwd(), "fairmotion"))
from fairmotion.tasks.motion_prediction import utils

sys.path.remove(os.path.join(os.getcwd(), "fairmotion"))


def prepare_dataset(fpath, batch_size, device):
    """
    Function processes the AMASS DIP dataset by:
        1. Calling the fairmotion utils as a base

    Args:
        fpath: The relative path to the raw AMASS dataset
        batch_size: The batch_size to parse the data into
        device: The device to compile the data for

    Returns:
        A dict of train, val, and test datasets (iterators).
    """
    datasets, mean, std = utils.prepare_dataset(
        *[
            os.path.join(fpath, f"{split}.pkl")
            for split in ["train", "test", "validation"]
        ],
        batch_size=batch_size,
        device=device,
    )
    return datasets


def generate_auto_regressive_targets(src_seqs, tgt_seqs):
    """
    The auto-regressive targets are defined as in the Spatio-temporal Transformer paper:
        y_t = x_{t + 1}, ie. we shift the input sequence to predict the next movement

    Args:
        src_seqs: The original source from AMASS DIP
        tgt_seqs: The original targets from AMASS DIP

    Returns:
        A Tensor containing the shifted auto-regressive targets.
    """
    next_motion_seqs = src_seqs[:, 1:, :]
    final_motion_seqs = torch.unsqueeze(tgt_seqs[:, 0, :], 1)
    return torch.cat((next_motion_seqs, final_motion_seqs), 1)
