import torch

from models.SpatioTemporalTransformer import SpatioTemporalTransformer
# from models.BiDrectionalTransformer import BiDirectionalTransformer
from utils.dataset import prepare_dataset, generate_auto_regressive_targets
from utils.types import ModelEnum, TargetEnum
# from utils.maskgenerator import BatchJointMaskGenerator
import argparse

def main() -> None:
    """
    Main runner.

    Args:
        None

    Returns:
        None
    """

    # Process command line arguments
    parser = argparse.ArgumentParser(prog="Model Training",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="Train selected model with given parameters on chosen dataset.")
    
    model_info = parser.add_argument_group("Model Information")
    model_info.add_argument("--model",choices =["STT","RNN","RNN_a","LSTM","LSTM_a","S2S","BDT"],default="STT")
    model_info.add_argument("-tt","--target_type",choices=["default","auto","pretrain"],default="default")
    model_info.add_argument("-dp","--data_path",default="data/sampled/aa/")

    parameters = parser.add_argument_group("Model Parameters")
    parameters.add_argument("-b","--batch_size",default=64)
    parameters.add_argument("-emb","--embedding_dim",default=64)
    parameters.add_argument("-ep","--epochs",default=10)
    parameters.add_argument("-h","--nhead",default=12)
    parameters.add_argument("-enc","--encoder_layers",default=6)
    parameters.add_argument("-f","--feedforward_dim",default=1024)

    args = parser.parse_args()

    # Process train, val, test datasets
    fpath = args.data_path
    batch_size = args.batch_size
    device = 'cpu'
    datasets = prepare_dataset(fpath, batch_size, device)

    # Setup Model
    _, seq_len, raw_dim = next(iter(datasets["train"]))[1].shape
    embedding_dim = args.embedding_dim  # Size of embeddings
    num_joints = 24  # AMASS DIP has 24 joints
    joint_dim = raw_dim // num_joints
    epochs = args.epochs
    # Loss per epoch is the average loss per sequence
    num_training_sequences = len(datasets["train"]) * batch_size

    if args.model == "STT":
        model = SpatioTemporalTransformer(num_joints, joint_dim, raw_dim, embedding_dim, .2)
    elif args.model == "BDT":
        # model = BiDirectionalTransformer(num_joints,
        #                                  joint_dim,
        #                                  raw_dim,
        #                                  embedding_dim,
        #                                  nhead=2,
        #                                  num_encoder_layers=1,
        #                                  dim_feedforward=1024,
        #                                  dropout=0.1).to(device)
        # mask = BatchJointMaskGenerator(num_joints,
        #                                joint_dim,
        #                                mask_value=-2,
        #                                time_step_mask_prob=1.0,
        #                                joint_mask_prob=1/24)
        pass
    elif args.model == "RNN":
        pass
    elif args.model == "RNN_a":
        pass
    elif args.model == "LSTM":
        pass
    elif args.model == "LSTM_a":
        pass
    elif args.model == "S2S":
        pass
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
            
            if args.target_type == "pretrain":
                # src_mask = mask.mask_joints(src_seqs)
                # outputs = model(src_mask)
                pass
            else:
                src_seqs, tgt_seqs = src_seqs.to(device).float(), tgt_seqs.to(device).float()
                outputs = model(src_seqs)

            if args.target_type == "auto":
                loss = criterion(outputs, generate_auto_regressive_targets(src_seqs, tgt_seqs))
            else:
                loss = criterion(outputs, tgt_seqs)

            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            break

        epoch_loss = epoch_loss / num_training_sequences
        print(f"Training loss {epoch_loss} | ")
        break


if __name__ == "__main__":
    main()
