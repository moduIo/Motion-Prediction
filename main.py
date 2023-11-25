import torch

from models.SpatioTemporalTransformer import SpatioTemporalTransformer
from utils.dataset import prepare_dataset, generate_auto_regressive_targets
from utils.types import ModelEnum, TargetEnum


def main(model_name: str = None, target_type: str = None) -> None:
    """
    Main runner.

    Args:
        model_name: The name of the model (used to select which model to run)
        target_type: The type of target (used to determine post processing logic)

    Returns:
        None
    """
    # Process train, val, test datasets
    fpath = "data/sampled/aa/"
    batch_size = 64
    device = 'cpu'
    datasets = prepare_dataset(fpath, batch_size, device)

    # Setup Model
    _, seq_len, raw_dim = next(iter(datasets["train"]))[1].shape
    embedding_dim = 64  # Size of embeddings
    num_joints = 24  # AMASS DIP has 24 joints
    joint_dim = raw_dim // num_joints
    epochs = 10
    # Loss per epoch is the average loss per sequence
    num_training_sequences = len(datasets["train"]) * batch_size

    # Train
    print("Training model...")

    if model_name == ModelEnum.SPATIO_TEMPORAL_TRANSFORMER:
        model = SpatioTemporalTransformer(num_joints, joint_dim, raw_dim, embedding_dim)
    else:
        print("Incorrect program usage.")
        return

    torch.autograd.set_detect_anomaly(True)

    opt = torch.optim.SGD(model.parameters(), lr=1e-8, momentum=0.9)
    criterion = torch.nn.MSELoss(reduction='sum')

    for epoch in range(epochs):
        epoch_loss = 0
        for iterations, (src_seqs, tgt_seqs) in enumerate(datasets["train"]):
            opt.zero_grad()
            src_seqs, tgt_seqs = src_seqs.to(device).float(), tgt_seqs.to(device).float()
            outputs = model(src_seqs)

            if target_type == TargetEnum.AUTO_REGRESSIVE:
                loss = criterion(outputs, generate_auto_regressive_targets(src_seqs, tgt_seqs))
            else:
                loss = criterion(outputs, tgt_seqs)

            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            break

        epoch_loss = epoch_loss / num_training_sequences
        print(f"Training loss {epoch_loss} | ")


if __name__ == "__main__":
    main(ModelEnum.SPATIO_TEMPORAL_TRANSFORMER, TargetEnum.AUTO_REGRESSIVE)
