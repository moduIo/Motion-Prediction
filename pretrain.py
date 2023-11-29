import torch

from models.BiDirectionalTransformer import BiDirectionalTransformer
from utils.maskgenerator import BatchJointMaskGenerator
from utils.dataset import prepare_dataset
from utils.types import ModelEnum, TargetEnum


def main(model_name: str = None, target_type: str = None) -> None:
    """
    Main runner.

    Args:
        model_name: The name of the model (used to select which model to run)
        target_type: The type of target (used to determine post-processing logic)

    Returns:
        None
    """
    # Process train, val, test datasets
    fpath = "data/sampled/aa/"
    batch_size = 64
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    datasets = prepare_dataset(fpath, batch_size, device)

    print("Using Device: ", device)

    # Setup Model
    _, seq_len, raw_dim = next(iter(datasets["train"]))[1].shape
    embedding_dim = 64  # Size of embeddings
    num_joints = 24  # AMASS DIP has 24 joints
    joint_dim = raw_dim // num_joints
    epochs = 10

    # Loss per epoch is the average loss per sequence
    num_training_sequences = len(datasets["train"]) * batch_size

    if model_name == ModelEnum.BIDIRECTIONAL_TRANSFORMER:
        model = BiDirectionalTransformer(num_joints,
                                         joint_dim,
                                         raw_dim,
                                         embedding_dim=32,
                                         nhead=2,
                                         num_encoder_layers=1,
                                         dim_feedforward=1024,
                                         dropout=0.1).to(device)
        mask = BatchJointMaskGenerator(num_joints,
                                       joint_dim,
                                       mask_value=-2,
                                       time_step_mask_prob=1.0,
                                       joint_mask_prob=1/24)
    else:
        print("Incorrect program usage.")
        return

    # Train
    print("Training model...")
    torch.autograd.set_detect_anomaly(True)
    opt = torch.optim.SGD(model.parameters(), lr=1e-8, momentum=0.9)
    criterion = torch.nn.MSELoss(reduction='sum')

    for epoch in range(epochs):
        epoch_loss = 0
        for iterations, (src_seqs, tgt_seqs) in enumerate(datasets["train"]):
            opt.zero_grad()
            src_seqs, tgt_seqs = src_seqs.to(device).float(), tgt_seqs.to(device).float()

            src_mask = mask.mask_joints(src_seqs)
            outputs = model(src_mask)

            loss = criterion(outputs, src_seqs)
            loss.backward()

            opt.step()
            epoch_loss += loss.item()
            break

        epoch_loss = epoch_loss / num_training_sequences
        print(f"Training loss {epoch_loss} | ")
        break


if __name__ == "__main__":
    main(ModelEnum.BIDIRECTIONAL_TRANSFORMER, TargetEnum.PRE_TRAIN)