import torch
import os

from utils.dataset import prepare_dataset, generate_auto_regressive_targets
from utils.loss import compute_validation_loss, PerJointMSELoss
from utils.metrics import plot_training_metrics
from utils.model import get_model
from utils.types import TargetEnum,ModelEnum


def train(args):

    # Process train, val, test datasets
    fpath = args.data_path
    batch_size = args.batch_size
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    datasets = prepare_dataset(fpath, batch_size, device)

    # Get joint dimensions for PerJointMSELoss
    _, _, raw_dim = next(iter(datasets["train"]))[0].shape
    num_joints = 24 
    joint_dim = raw_dim // num_joints 

    # Setup Model
    model, mask = get_model(args, datasets, device)

    # Train
    print(f"=== Training model with args={args} on {device} ===")
    torch.autograd.set_detect_anomaly(True)
    opt = torch.optim.SGD(model.parameters(), lr=1e-8, momentum=0.9)
    criterion = PerJointMSELoss(number_joints=num_joints,joint_dimension=joint_dim)

    training_losses = []
    validation_losses = []
    os.makedirs(args.save_model_path,exist_ok=True)
    for epoch in range(args.epochs):
        print(f"Start Epoch {epoch}:")
        model.train()
        epoch_loss = 0
        for _, (src_seqs, tgt_seqs) in enumerate(datasets["train"]):
            opt.zero_grad()

            src_seqs, tgt_seqs = (
                src_seqs.to(device).float(),
                tgt_seqs.to(device).float(),
            )

            if (args.model == ModelEnum.LSTM_SEQ2SEQ.value) or (args.model == ModelEnum.LSTM_SEQ2SEQ_ATT.value):
                outputs = model(src_seqs,tgt_seqs)
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

            loss.backward()
            opt.step()
            epoch_loss += loss.item()

        # Compute epoch losses
        epoch_loss = epoch_loss / batch_size
        training_losses.append(epoch_loss)
        epoch_val_loss, val_batch_size = compute_validation_loss(args, model, datasets, criterion, device, mask)
        validation_losses.append(epoch_val_loss / val_batch_size)
        print(f"Training loss {epoch_loss} | ")
        print(f"Validation loss {epoch_val_loss} | ")

        # Save model
        if epoch % args.save_model_frequency == 0:
            arguments = vars(args)
            with open(f"{args.save_model_path}/model_config.txt", "w+") as f:
                for k, v in arguments.items():
                    line = str(k) + ": " + str(v) + "\n"
                    f.write(line)

            torch.save(model.state_dict(), f"{args.save_model_path}/{epoch}.model")

        if len(validation_losses) == 0 or epoch_val_loss <= min(validation_losses):
            torch.save(
                model.state_dict(), f"{args.save_model_path}/best_epoch_{epoch}.model"
            )

    plot_training_metrics(args, training_losses, validation_losses)
