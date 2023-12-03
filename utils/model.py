from models.SpatioTemporalTransformer import SpatioTemporalTransformer
from models.BiDirectionalTransformer import BiDirectionalTransformer
from utils.mask_generator import BatchJointMaskGenerator
from utils.types import ModelEnum, TargetEnum
from models.Encoders import LSTMEncoder
from models.Decoders import LSTMDecoder
from models.Seq2Seq import Seq2Seq

def get_model(args, datasets, device):
    """
    Function returns the model given the args.

    Args:
        args: The args for the run
        model: The current model
        datasets: The datasets for the run

    Returns:
        A tuple of the model and the mask.
    """
    # Hyperparameters
    _, seq_len, raw_dim = next(iter(datasets["train"]))[0].shape
    _, tgt_len, raw_dim = next(iter(datasets["train"]))[1].shape
    print(tgt_len)
    embedding_dim = args.embedding_dim  # Size of embeddings
    num_joints = 24  # AMASS DIP has 24 joints
    joint_dim = raw_dim // num_joints  # 3 for -aa and 9 for -rotmat
    dropout = args.dropout
    nlayers = args.nlayers
    ff_dim = args.feedforward_dim
    nhead = args.nhead
    hidden_dim = args.hidden_dim

    if args.model == ModelEnum.SPATIO_TEMPORAL_TRANSFORMER.value:
        model = SpatioTemporalTransformer(
            num_joints,
            joint_dim,
            seq_len,
            raw_dim,
            embedding_dim,
            ff_dim,
            dropout,
            nhead,
            nlayers,
        )
    elif args.model == ModelEnum.BIDIRECTIONAL_TRANSFORMER.value:
        model = BiDirectionalTransformer(
            num_joints,
            joint_dim,
            raw_dim,
            embedding_dim,
            nhead=nhead,
            num_encoder_layers=nlayers,
            dim_feedforward=ff_dim,
            dropout=dropout,
        ).to(device)
    elif args.model == "RNN":
        pass
    elif args.model == "RNN_a":
        pass
    elif args.model == ModelEnum.LSTM_SEQ2SEQ.value:
        enc = LSTMEncoder(input_dim=raw_dim, hidden_dim=hidden_dim, num_layers=nlayers).to(device)
        dec = LSTMDecoder(input_dim=raw_dim, hidden_dim=hidden_dim, output_dim=raw_dim,device='cpu').to(device)
        model = Seq2Seq(enc, dec)
    elif args.model == "LSTM_a":
        pass
    else:
        raise("Incorrect program usage.")

    if args.target_type == TargetEnum.PRE_TRAIN.value:
        mask = BatchJointMaskGenerator(
            num_joints,
            joint_dim,
            mask_value=-2,
            time_step_mask_prob=1.0,
            joint_mask_prob=1 / 24,
        )
    else:
        mask = None

    return model, mask
