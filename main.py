from utils.args import parse_main_args
from tasks.train import train
from tasks.predict import predict


def main():
    args = parse_main_args()

    if args.mode == 'train':
        train(args)
    elif args.mode == 'predict':
        predict(args)
    else:
        print("Incorrect program usage: invalid args.mode value.")


if __name__ == "__main__":
    main()
