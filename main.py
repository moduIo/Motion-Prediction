from matplotlib import pyplot as plt
import torch

from utils.args import parse_args
from utils.dataset import prepare_dataset, generate_auto_regressive_targets
from utils.loss import compute_validation_loss
from utils.model import get_model
from utils.types import TargetEnum


def main() -> None:
    """
    Main runner.

    Args:
        None

    Returns:
        None
    """
    args = parse_args()

    # Process train, val, test datasets
    fpath = args.data_path
    batch_size = args.batch_size
    device = "cpu"  #'cuda' if torch.cuda.is_available() else 'cpu'
    datasets = prepare_dataset(fpath, batch_size, device)

    # Setup Model
    _, seq_len, raw_dim = next(iter(datasets["train"]))[0].shape
    num_training_sequences = len(datasets["train"]) * batch_size

    # Model Selector
    model, mask = get_model(args, raw_dim)

    # Train
    print(f"=== Training model with args={args} ===")
    torch.autograd.set_detect_anomaly(True)
    opt = torch.optim.SGD(model.parameters(), lr=1e-8, momentum=0.9)
    criterion = torch.nn.MSELoss(reduction="sum")

    training_losses = []
    validation_losses = []
    for epoch in range(args.epochs):
        epoch_loss = 0
        for _, (src_seqs, tgt_seqs) in enumerate(datasets["train"]):
            opt.zero_grad()

            src_seqs, tgt_seqs = (
                src_seqs.to(device).float(),
                tgt_seqs.to(device).float(),
            )

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
            break  # TODO: Delete this

        epoch_loss = epoch_loss / num_training_sequences
        training_losses.append(epoch_loss)

        # validation loss
        epoch_val_loss, val_iter = compute_validation_loss(args, model, datasets, criterion, device, mask)
        validation_losses.append(epoch_val_loss / ((val_iter + 1) * batch_size))

        print(f"Training loss {epoch_loss} | ")
        print(f"Validation loss {epoch_val_loss} | ")

        # Save model
        if args.save_model_frequency == 0 or epoch % args.save_model_frequency == 0:  # TODO: remove args.save_model_frequency == 0
            arguments = vars(args)
            with open(f"{args.save_model_path}/model_config.txt", "w") as f:
                for k, v in arguments.items():
                    line = str(k) + ": " + str(v)
                    f.write(line)

            torch.save(model.state_dict(), f"{args.save_model_path}/{epoch}.model")

        if len(validation_losses) == 0 or epoch_val_loss <= min(validation_losses):
            torch.save(
                model.state_dict(), f"{args.save_model_path}/best_epoch_{epoch}.model"
            )

        break  # TODO: Delete this

    plt.plot(range(len(training_losses)), training_losses)
    plt.plot(range(len(validation_losses)), validation_losses)
    plt.ylabel("MSE Loss")
    plt.xlabel("Epoch")
    plt.savefig(f"{args.save_model_path}/loss.svg", format="svg")
    plt.savefig(f"{args.save_model_path}/loss.png", format="png")


if __name__ == "__main__":
    main()
