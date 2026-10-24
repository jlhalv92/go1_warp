"""One environment factory shared by training, playback and benchmarks."""
from .config import Config, EnvConfig
from .go1 import Go1WarpEnv


def make_env(config):
    if config.environment == 'flat':
        return Go1WarpEnv(config)
    if config.environment == 'domain_randomized':
        from .domain_randomized import DomainRandomizedGo1Env
        return DomainRandomizedGo1Env(config)
    if config.environment == 'blind':
        from .go1_blind import Go1WarpEnv as BlindGo1Env
        return BlindGo1Env(config)
    raise ValueError(f'Unknown environment: {config.environment}')


__all__ = ['Config', 'EnvConfig', 'Go1WarpEnv', 'make_env']
