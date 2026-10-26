"""Train Go1: python train.py --help."""
from src.runner.config import parse_train_args


def main(argv=None):
    config = parse_train_args(argv)
    from src.runner.trainer import train
    return train(config)


if __name__ == '__main__':
    main()
