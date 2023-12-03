import argparse


def parse_args():
    # Process command line arguments
    parser = argparse.ArgumentParser(
        prog="Model Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Train selected model with given parameters on chosen dataset.",
    )

    model_info = parser.add_argument_group("Model Information")
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
    model_info.add_argument(
        "-tt",
        "--target_type",
        choices=["default", "auto-regressive", "pretrain"],
        default="default",
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

    return parser.parse_args()
