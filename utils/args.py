import argparse


def parse_mode_arg():
    """
    Function parses the model execution mode.
    """
    parser = argparse.ArgumentParser(
        prog="Model Mode",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Switch for the model mode.",
    )
    model_info = parser.add_argument_group("Model Mode")
    model_info.add_argument("--mode", choices=["train", "predict"], default="train")
    return parser.parse_known_args()


def parse_train_args(args):
    """
    Function parses train() command line args.
    """
    parser = argparse.ArgumentParser(
        prog="Model Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Train selected model with given parameters on chosen dataset.",
    )

    model_info = parser.add_argument_group("Training Settings")
    model_info.add_argument(
        "--model",
        choices=[
            "spatio-temporal-transformer",
            "lstm_seq2seq",
            "lstm_seq2seq_att",
            "bi-directional-transformer",
        ],
        default="spatio-temporal-transformer",
    )
    model_info.add_argument(
        "-tt",
        "--target_type",
        choices=["default", "auto-regressive", "pre-train"],
        default="auto-regressive",
    )
    model_info.add_argument("-dp", "--data_path", default="./data/sampled/aa/")
    model_info.add_argument("-sfreq", "--save_model_frequency", default=5, type=int)
    model_info.add_argument("-spath", "--save_model_path", default="./model_saves/")

    parameters = parser.add_argument_group("Model Parameters")
    parameters.add_argument("-b", "--batch_size", default=32, type=int)
    parameters.add_argument("-emb", "--embedding_dim", default=128, type=int)
    parameters.add_argument("-ep", "--epochs", default=100, type=int)
    parameters.add_argument("-nh", "--nhead", default=8, type=int)
    parameters.add_argument("-enc", "--nlayers", default=8, type=int)
    parameters.add_argument("-ff", "--feedforward_dim", default=256, type=int)
    parameters.add_argument("-do", "--dropout", default=0.1, type=float)
    parameters.add_argument("-mp1", "--seqmaskprob", default=1.0, type=float)
    parameters.add_argument("-mp2", "--jointmaskprob", default=0.05, type=float)
    parameters.add_argument("-hid", "--hidden_dim", default=64, type=int)
    parameters.add_argument("-tah", "--temporal_attn_horizon", default=10, type=int)
    parameters.add_argument("-lrs", "--lr_start", default=.0001, type=float)
    parameters.add_argument("-lrf", "--lr_factor", default=.1, type=float)
    parameters.add_argument("-lrr", "--lr_rate", default=10, type=int)

    return parser.parse_known_args(args)


def parse_predict_args(args):
    """
    Function parses predict() command line args.
    """
    parser = argparse.ArgumentParser(
        prog="Model Prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Predict selected model with given parameters on chosen dataset.",
    )

    model_info = parser.add_argument_group("Prediction Settings")
    model_info.add_argument(
        "-tt",
        "--target_type",
        choices=["default", "auto-regressive", "pretrain"],
        default="default",
    )

    model_info.add_argument(
        "--model",
        choices=[
            "spatio-temporal-transformer",
            "rnn",
            "rnn_a",
            "lstm",
            "lstm_a",
            "seq2seq",
            "bi-directional-transformer",
        ],
        default="spatio-temporal-transformer",
    )
    model_info.add_argument("-dp", "--data_path", default="./data/sampled/aa/")
    model_info.add_argument("-spath", "--save_model_path", default="./model_saves/")
    model_info.add_argument("--save_preds_path", default="./motion_saves/")

    parameters = parser.add_argument_group("Model Parameters")
    parameters.add_argument("-b", "--batch_size", default=32, type=int)
    parameters.add_argument("-emb", "--embedding_dim", default=128, type=int)
    parameters.add_argument("-ep", "--epochs", default=100, type=int)
    parameters.add_argument("-nh", "--nhead", default=8, type=int)
    parameters.add_argument("-enc", "--nlayers", default=8, type=int)
    parameters.add_argument("-ff", "--feedforward_dim", default=256, type=int)
    parameters.add_argument("-do", "--dropout", default=0.1, type=float)
    parameters.add_argument("-mp1", "--seqmaskprob", default=1.0, type=float)
    parameters.add_argument("-mp2", "--jointmaskprob", default=0.05, type=float)
    parameters.add_argument("-hid", "--hidden_dim", default=64, type=int)
    parameters.add_argument("-tah", "--temporal_attn_horizon", default=10, type=int)
    parameters.add_argument("-lrs", "--lr_start", default=.0001, type=float)
    parameters.add_argument("-lrf", "--lr_factor", default=.1, type=float)
    parameters.add_argument("-lrr", "--lr_rate", default=10, type=int)

    return parser.parse_known_args(args)
