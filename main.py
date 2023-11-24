import torch

from models.SpatioTemporalTransformer import SpatioTemporalTransformer
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
    joint_dim = num_predictions // num_joints
    epochs = 2
    # Loss per epoch is the average loss per sequence
    num_training_sequences = len(datasets["train"]) * batch_size

    # Train
    print("Training model...")
    model = SpatioTemporalTransformer(num_joints, joint_dim)
    torch.autograd.set_detect_anomaly(True)

    opt = torch.optim.SGD(model.parameters(), lr=1e-8, momentum=0.9)
    criterion = torch.nn.MSELoss(reduction='sum')

    for epoch in range(epochs):
        epoch_loss = 0
        for iterations, (src_seqs, tgt_seqs) in enumerate(datasets["train"]):
            opt.zero_grad()
            src_seqs, tgt_seqs = src_seqs.to(device).float(), tgt_seqs.to(device).float()
            outputs = model(src_seqs)
            print(outputs.shape)

            loss = criterion(outputs, generate_targets(src_seqs, tgt_seqs))
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            break
        epoch_loss = epoch_loss / num_training_sequences
        print(f"Training loss {epoch_loss} | ")
        break


if __name__ == "__main__":
    main()
