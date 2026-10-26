"""Playback and fixed-window evaluation using the shared environment factory."""
import json
import time

import torch
from tqdm import trange

from src.envs import make_env
from src.networks import ActorCritic
from .config import load_checkpoint, resolve_play_config
from .evaluation import evaluate
from .trainer import make_config


def play(config):
    checkpoint = load_checkpoint(config.checkpoint)
    config = resolve_play_config(config, checkpoint)
    torch.set_num_threads(config.torch_threads)
    env = None
    try:
        env = make_env(make_config(config, config.num_envs, config.seed))
        if hasattr(env,'set_evaluation_mode'): env.set_evaluation_mode(True)
        print(f'Environment: {type(env).__module__}.{type(env).__name__}')
        agent = ActorCritic(env.num_observations, env.num_actions,
                            checkpoint['args']['n_features']).to(env.device)
        expected=checkpoint.get('observation_dim',checkpoint['agent']['actor.fc1.weight'].shape[1])
        if expected!=env.num_observations:
            raise ValueError(f'Checkpoint expects {expected} observations, but the environment supplies '
                             f'{env.num_observations}. Use matching terrain/layout/grid settings.')
        agent.load_state_dict(checkpoint['agent'])
        agent.eval()
        if config.headless:
            if env.device.type == 'cuda':
                torch.cuda.synchronize(env.device)
            start = time.perf_counter()
            result = evaluate(agent, env, config.steps, checkpoint['args']['gamma'],
                              config.seed, config.stochastic)
            if env.device.type == 'cuda':
                torch.cuda.synchronize(env.device)
            result['evaluation_seconds'] = time.perf_counter() - start
            result['transitions_per_second'] = result['eval_transitions'] / result['evaluation_seconds']
            print(json.dumps(result, indent=2))
            return result
        obs, _ = env.reset(seed=config.seed)
        with torch.no_grad():
            for _ in trange(config.steps):
                action = agent.distribution(obs).sample() if config.stochastic else agent.actor(obs)
                obs, _, terminated, truncated, _ = env.step(action)
                if not env.render(config.world_id, throttle=not config.unpaced):
                    break
                ids = torch.where(terminated | truncated)[0]
                if ids.numel():
                    obs, _ = env.reset(ids)
    finally:
        if env is not None:
            env.close()
