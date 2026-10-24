"""Evaluate or visualize Go1: python play.py CHECKPOINT --help."""
from src.algorithms.config import PlayConfig, parse_play_args


def main(argv=None):
    config = parse_play_args(argv)
    from src.algorithms.playback import play
    return play(config)


if __name__ == '__main__':
    main()
