from utils.args import parse_train_args, parse_predict_args, parse_mode_arg
from tasks.train import train
from tasks.predict import predict


def main():
    mode_arg, args = parse_mode_arg()
    if mode_arg.mode == 'train':
        train(parse_train_args(args)[0])
    else:
        predict(parse_predict_args(args)[0])


if __name__ == "__main__":
    main()
