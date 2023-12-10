from models.SpatioTemporalTransformer import SpatioTemporalTransformer
from models.BiDirectionalTransformer import BiDirectionalTransformer
from utils.mask_generator import BatchJointMaskGenerator
from utils.types import ModelEnum, TargetEnum
from models.Encoders import LSTMEncoder
from models.Decoders import LSTMDecoder, LSTMDecoderWithAttention
from models.Seq2Seq import Seq2Seq


def get_model(args, datasets, device):
    """
    Function returns the model given the args.

    Args:
        args: The args for the run
        datasets: The datasets for the run
        device: The device where the models and data will be housed
    Returns:
        A tuple of the model and the mask.
    """
    # Hyperparameters
    _, seq_len, raw_dim = next(iter(datasets["train"]))[0].shape
    embedding_dim = args.embedding_dim  # Size of embeddings
    num_joints = 24  # AMASS DIP has 24 joints
    joint_dim = raw_dim // num_joints  # 3 for -aa and 9 for -rotmat
    dropout = args.dropout
    nlayers = args.nlayers
    ff_dim = args.feedforward_dim
    nhead = args.nhead
    seq_mask_prob = args.seqmaskprob
    joint_mask_prob = args.jointmaskprob
    hidden_dim = args.hidden_dim

    if args.model == ModelEnum.SPATIO_TEMPORAL_TRANSFORMER.value:
        model = SpatioTemporalTransformer(
            num_joints,
            joint_dim,
            seq_len,
            raw_dim,
            device,
            embedding_dim,
            ff_dim,
            dropout,
            nhead,
            nlayers,
            args.temporal_attn_horizon,
        ).to(device)
    elif args.model == ModelEnum.BIDIRECTIONAL_TRANSFORMER.value:
        model = BiDirectionalTransformer(
            num_joints,
            joint_dim,
            raw_dim,
            device,
            embedding_dim,
            nhead=nhead,
            num_encoder_layers=nlayers,
            dim_feedforward=ff_dim,
            dropout=dropout,
        ).to(device)
    elif args.model == ModelEnum.LSTM_SEQ2SEQ.value:
        enc = LSTMEncoder(
            input_dim=raw_dim, hidden_dim=hidden_dim, num_layers=nlayers
        ).to(device)
        dec = LSTMDecoder(
            input_dim=raw_dim,
            hidden_dim=hidden_dim,
            output_dim=raw_dim,
            num_layers=nlayers,
            device=device,
        ).to(device)
        model = Seq2Seq(enc, dec)
    elif args.model == ModelEnum.LSTM_SEQ2SEQ_ATT.value:
        enc = LSTMEncoder(
            input_dim=raw_dim, hidden_dim=hidden_dim, num_layers=nlayers
        ).to(device)
        dec = LSTMDecoderWithAttention(
            input_dim=raw_dim,
            hidden_dim=hidden_dim,
            output_dim=raw_dim,
            max_source_length=seq_len,
            device=device,
        ).to(device)
        model = Seq2Seq(enc, dec)
    elif args.model == "S2S":
        pass
    else:
        raise ("Incorrect program usage.")

    if args.target_type == TargetEnum.PRE_TRAIN.value:
        mask = BatchJointMaskGenerator(
            num_joints,
            joint_dim,
            mask_value=-4,
            time_step_mask_prob=seq_mask_prob,
            joint_mask_prob=joint_mask_prob,
        )
    else:
        mask = None

    return model, mask
