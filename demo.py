"""Run the nominal standing controller: python demo.py --help."""
from src.algorithms.config import config_parser, validate_environment
from src.envs import make_env
from src.envs.domain_randomized import DomainRandomizationConfig
import torch


def main(argv=None):
    parser = config_parser(DomainRandomizationConfig, __doc__)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--render', action='store_true')
    args = vars(parser.parse_args(argv))
    steps, render = args.pop('steps'), args.pop('render')
    if steps < 1:
        parser.error('steps must be positive')
    args.setdefault('environment', 'flat')
    config = DomainRandomizationConfig.from_dict(args)
    validate_environment(config)
    env = make_env(config)
    try:
        for _ in range(steps):
            _, _, terminated, truncated, _ = env.step(torch.zeros(env.num_envs,12,device=env.device))
            if render and not env.render(): break
            ids = torch.where(terminated | truncated)[0]
            if ids.numel(): env.reset(ids)
    finally:
        env.close()


if __name__ == '__main__':
    main()
