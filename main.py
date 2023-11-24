import torch

from utils.dataset import prepare_dataset, generate_targets


def main():
    # Process train, val, test datasets
    fpath = "data/sampled/aa/"
    batch_size = 64
    device = 'cpu'
    datasets = prepare_dataset(fpath, batch_size, device)

    # Setup Model
    _, seq_len, num_predictions = next(iter(datasets["train"]))[1].shape
    num_joints = 24  # AMASS DIP has 24 joints
    M = num_predictions // num_joints

    # Train
    for iterations, (src_seqs, tgt_seqs) in enumerate(datasets["train"]):
        tgt_seqs = generate_targets(src_seqs, tgt_seqs)


if __name__ == "__main__":
    main()
