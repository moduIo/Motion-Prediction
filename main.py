import torch

from models.SpatioTemporalTransformer import SpatioTemporalTransformer
from models.BiDrectionalTransformer import BiDirectionalTransformer
from utils.dataset import prepare_dataset, generate_auto_regressive_targets
from utils.types import ModelEnum, TargetEnum
from matplotlib import pyplot as plt
from utils.maskgenerator import BatchJointMaskGenerator
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
    model_info.add_argument("--model",choices =["spatio-temporal-transformer","rnn","rnn_a","lstm","lstm_a","seq2seq","bi-directional-transformer"],default="STT")
    model_info.add_argument("-tt","--target_type",choices=["default","auto-regressive","pretrain"],default="default")
    model_info.add_argument("-dp","--data_path",default="./data/sampled/aa/")
    model_info.add_argument("-sfreq","--save_model_frequency",default=5)
    model_info.add_argument("-spath","--save_model_path",default="./model_saves/")

    parameters = parser.add_argument_group("Model Parameters")
    parameters.add_argument("-b","--batch_size",default=64)
    parameters.add_argument("-emb","--embedding_dim",default=64)
    parameters.add_argument("-ep","--epochs",default=10)
    parameters.add_argument("-h","--nhead",default=12)
    parameters.add_argument("-enc","--nlayers",default=6)
    parameters.add_argument("-f","--feedforward_dim",default=1024)
    parameters.add_argument("-do","--dropout",default=0.1)

    args = parser.parse_args()

    # Process train, val, test datasets
    fpath = args.data_path
    batch_size = args.batch_size
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    datasets = prepare_dataset(fpath, batch_size, device)

    # Setup Model
    _, seq_len, raw_dim = next(iter(datasets["train"]))[0].shap
    
    #Hyperparameters
    embedding_dim = args.embedding_dim  # Size of embeddings
    num_joints = 24  # AMASS DIP has 24 joints
    joint_dim = raw_dim // num_joints # 3 for -aa and 9 for -rotmat
    epochs = args.epochs
    dropout = args.dropout
    nlayers = args.nlayers
    ff_dim = args.feedforward_dim
    nhead = args.nhead
    
    num_training_sequences = len(datasets["train"]) * batch_size
    
    #Model Selector
    if args.model == ModelEnum.SPATIO_TEMPORAL_TRANSFORMER:
        model = SpatioTemporalTransformer(num_joints,
                                          joint_dim,
                                          seq_len,
                                          raw_dim,
                                          embedding_dim,
                                          dropout,
                                          nlayers)
    elif args.model == ModelEnum.BI_DIRECTIONAL_TRANSFORMER:
        model = BiDirectionalTransformer(num_joints,
                                         joint_dim,
                                         raw_dim,
                                         embedding_dim,
                                         nhead=nhead,
                                         num_encoder_layers=nlayers,
                                         dim_feedforward=ff_dim,
                                         dropout=dropout).to(device)
        mask = BatchJointMaskGenerator(num_joints,
                                       joint_dim,
                                       mask_value=-2,
                                       time_step_mask_prob=1.0,
                                       joint_mask_prob=1/24)
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

    training_losses = []
    validation_losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        for iterations, (src_seqs, tgt_seqs) in enumerate(datasets["train"]):
            opt.zero_grad()
            
            src_seq, tgt_seqs = src_seqs.to(device).float(), tgt_seqs.to(device).float()
            
            if args.target_type == TargetEnum.PRETRAIN:
                src_mask = mask.mask_joints(src_seqs)
                outputs = model(src_mask)
            else:
                outputs = model(src_seqs)

            if args.target_type == TargetEnum.AUTO_REGRESSIVE:
                loss = criterion(outputs, generate_auto_regressive_targets(src_seqs, tgt_seqs))
            else:
                loss = criterion(outputs, tgt_seqs)

            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            break

        epoch_loss = epoch_loss / num_training_sequences
        training_losses.append(epoch_loss)

        # validation loss
        epoch_val_loss = 0
        with torch.no_grad():
            for val_iter, (src_seqs, tgt_seqs) in enumerate(datasets["validation"]):
                max_len = tgt_seqs.shape[1]
                src_seqs, tgt_seqs = (
                    src_seqs.to(device).float(),
                    tgt_seqs.to(device).float()
                )
                outputs = model(src_seqs, max_len=max_len)

                loss = criterion(outputs, tgt_seqs)
                epoch_val_loss += loss.item()
        validation_losses.append(epoch_val_loss / ((val_iter + 1) * batch_size))

        print(f"Training loss {epoch_loss} | ")
        print(f"Validation loss {epoch_val_loss} | ")

        if epoch % args.save_model_frequency == 0:
                        
            arguments = vars(args)
            with open(f"{args.save_model_path}/model_config.txt",'w') as f:
                for k,v in arguments.items():
                    line = str(k) + ": " + str(v)
                    f.write(line)

            torch.save(
                model.state_dict(), f"{args.save_model_path}/{epoch}.model"
            )
        if len(validation_losses) == 0 or epoch_val_loss <= min(validation_losses):
            torch.save(
                model.state_dict(), f"{args.save_model_path}/best_epoch_{epoch}.model"
            )

        break

    plt.plot(range(len(training_losses)), training_losses)
    plt.plot(range(len(validation_losses)), validation_losses)
    plt.ylabel("MSE Loss")
    plt.xlabel("Epoch")
    plt.savefig(f"{args.save_model_path}/loss.svg", format="svg")
    plt.savefig(f"{args.save_model_path}/loss.png", format="png")

if __name__ == "__main__":
    main()
