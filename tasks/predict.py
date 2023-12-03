import torch

from utils.dataset import prepare_dataset


def predict(args):
    # Process train, val, test datasets
    fpath = args.data_path
    batch_size = args.batch_size
    device = "cpu"  # TODO: 'cuda' if torch.cuda.is_available() else 'cpu'
    datasets = prepare_dataset(fpath, batch_size, device)
    num_test_sequences = len(datasets["score"]) * batch_size
    model_path = args.save_model_path
    # model = load(model_path)

    # Predict
    print(f"=== Computing test error with args={args} ===")
    criterion = torch.nn.MSELoss(reduction="sum")

    test_loss = 0
    for _, (src_seqs, tgt_seqs) in enumerate(datasets["test"]):
        src_seqs, tgt_seqs = (
            src_seqs.to(device).float(),
            tgt_seqs.to(device).float(),
        )
        outputs = model(src_seqs)  # TODO: Make this generate targets of the right shape
        loss = criterion(outputs, tgt_seqs)
        test_loss += loss.item()
        break  # TODO: Delete this

    # Normalize loss
    test_loss = test_loss / num_test_sequences
    print(f"Test loss for {model_path}={test_loss}")
