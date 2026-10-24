"""Evaluate or visualize a checkpoint produced by go1_warp.train."""
import argparse
from dataclasses import fields
import json
import torch
from . import Config, Go1WarpEnv
#from .env_new_contacts import Config, Go1WarpEnv

from .networks import ActorCritic
from .evaluation_utils import evaluate
from tqdm import trange


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--backend', choices=('warp', 'mujoco'), default='warp')
    p.add_argument('--num-envs', type=int, default=1)
    p.add_argument('--steps', type=int, default=10000)
    p.add_argument('--episodes', type=int, default=10)
    p.add_argument('--seed', type=int, default=10000)
    p.add_argument('--headless', action='store_true')
    p.add_argument('--stochastic', action='store_true')
    a = p.parse_args()
    if a.num_envs < 1 or a.steps < 1 or a.episodes < 1:
        p.error('num-envs, steps and episodes must be positive')
    torch.set_num_threads(1)
    ckpt = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    allowed = {f.name for f in fields(Config)}
    cfg = {k: v for k, v in ckpt['env_config'].items() if k in allowed}
    cfg.update(num_envs=a.num_envs, device=a.device, backend=a.backend, seed=a.seed)
    env = Go1WarpEnv(Config(**cfg))
    agent = ActorCritic(env.num_observations, env.num_actions, ckpt['args']['n_features']).to(env.device)
    agent.load_state_dict(ckpt['agent'])
    agent.eval()
    try:
        if a.headless:
            result = evaluate(agent, env, a.steps, ckpt['args']['gamma'], a.seed, a.stochastic)
            print(json.dumps(result, indent=2))
        else:
            obs, _ = env.reset(seed=a.seed)
            with torch.no_grad():
                for _ in trange(a.steps):

                    #obs[:,[9,10,11]]=torch.tensor([1.,0.,0.]).to(env.device)
                    actions = agent.distribution(obs).sample() if a.stochastic else agent.actor(obs)
                    obs, _, terminated, truncated, _ = env.step(actions)
                    if not env.render(0, throttle=True):
                        break
                    ids = torch.where(terminated | truncated)[0]
                    if ids.numel():
                        obs, _ = env.reset(ids)
    finally:
        env.close()


if __name__ == '__main__':
    main()
