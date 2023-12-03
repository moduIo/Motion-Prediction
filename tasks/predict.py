import torch

from utils.dataset import prepare_dataset
from utils.model import get_model
from utils.motion_generator import generate_motion


def predict(args):
    # Setup dataset
    fpath = args.data_path
    batch_size = args.batch_size
    device = "cpu"  # TODO: 'cuda' if torch.cuda.is_available() else 'cpu'
    datasets = prepare_dataset(fpath, batch_size, device)
    num_test_sequences = len(datasets["test"]) * batch_size
    _, output_seq_len, raw_dim = next(iter(datasets["test"]))[1].shape

    # Setup model
    model_path = args.save_model_path
    model, mask = get_model(args, datasets, device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Predict
    print(f"=== Computing test error with args={args} ===")
    criterion = torch.nn.MSELoss(reduction="sum")

    test_loss = 0
    for _, (src_seqs, tgt_seqs) in enumerate(datasets["test"]):
        src_seqs, tgt_seqs = (
            src_seqs.to(device).float(),
            tgt_seqs.to(device).float(),
        )
        outputs = generate_motion(model, src_seqs, output_seq_len)
        loss = criterion(outputs, tgt_seqs)
        test_loss += loss.item()

    # Normalize loss
    test_loss = test_loss / num_test_sequences
    print(f"Test loss for {model_path}={test_loss}")
