import torch

from utils.dataset import prepare_dataset
from utils.loss import PerJointMSELoss
from utils.model import get_model
from utils.motion_generator import generate_motion


def predict(args):
    # Setup dataset
    fpath = args.data_path
    batch_size = args.batch_size
    device = "cuda" if torch.cuda.is_available() else "cpu"
    datasets = prepare_dataset(fpath, batch_size, device)
    num_test_sequences = len(datasets["test"]) * batch_size
    _, output_seq_len, raw_dim = next(iter(datasets["test"]))[1].shape
    num_joints = 24
    joint_dim = raw_dim // num_joints

    # Setup model
    model_path = args.save_model_path
    model, mask = get_model(args, datasets, device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Predict
    print(f"=== Computing test error with args={args} ===")
    criterion = PerJointMSELoss(number_joints=num_joints, joint_dimension=joint_dim)

    with torch.no_grad():
        test_loss = 0
        for _, (src_seqs, tgt_seqs) in enumerate(datasets["test"]):
            src_seqs, tgt_seqs = (
                src_seqs.to(device).float(),
                tgt_seqs.to(device).float(),
            )
            outputs = generate_motion(model, src_seqs, output_seq_len)

            # NOTE: It's assumed that the output will be auto-regressive
            loss = criterion(outputs, tgt_seqs)
            test_loss += loss.item()

        # Normalize loss
        test_loss = test_loss / num_test_sequences
        print("===" * 10, f"\nTest loss for {model_path}={test_loss}")
