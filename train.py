"""Train Go1: python train.py --help."""
from src.algorithms.config import TrainConfig, parse_train_args


def main(argv=None):
    config = parse_train_args(argv)
    from src.algorithms.trainer import train
    return train(config)


if __name__ == '__main__':
    main()
