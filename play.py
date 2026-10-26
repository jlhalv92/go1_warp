"""Evaluate or visualize Go1: python play.py CHECKPOINT --help."""
from src.runner.config import parse_play_args


def main(argv=None):
    config = parse_play_args(argv)
    from src.runner.playback import play
    return play(config)


if __name__ == '__main__':
    main()
