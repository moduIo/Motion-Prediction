import os

from fairmotion.tasks.motion_prediction import utils


def prepare_dataset():
    """
    Function processes the AMASS DIP dataset by:
        1. Calling the fairmotion utils as a base
        2. Post-processing the output by constructing new targets


    The auto-regressive targets in step 2 are defined as in the Spatio-temporal Transformer paper:
        y_t = x_{t + 1}, ie. we shift the input sequence to predict the next movement

    """

    fpath = "data/sampled/aa/"
    dataset, mean, std = utils.prepare_dataset(
        *[
            os.path.join(fpath, f"{split}.pkl")
            for split in ["train", "test", "validation"]
        ],
        batch_size=64,
        device='cpu',
    )
    return dataset
