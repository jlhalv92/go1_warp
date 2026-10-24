"""Fixed-step evaluation: no episode quotas and no early stopping.

    from src.algorithms.evaluation import evaluate
    metrics = evaluate(agent, eval_env, n_steps=1000, gamma=.99)

R and J describe the fixed window, including rewards after resets. They are
NOT mean episode returns. completed_episode_R/J describe completed episodes
only; short episodes can be overrepresented and unfinished tails are excluded.
Uses the existing env.step/reset API; does not remove its GPU synchronizations.
"""
import math
import operator
import torch
from .ppo import EpisodeReturns
from tqdm import trange


@torch.no_grad()
def evaluate(agent,
             env,
             n_steps=1000, gamma=.99,
             seed=10000, stochastic=False):
    """Run exactly n_steps batched calls = n_steps * env.num_envs transitions.

    Every world contributes at every step. Reset terminated/truncated worlds
    before the next call; reset does not advance physics. Weights are never
    updated. Default actions are deterministic actor means. A window's discount
    gamma**t does NOT restart at episode boundaries; episode diagnostics do.

    Resets the supplied env initially and modifies its simulation state. Use a
    separate eval_env, never the live training env with a pending rollout.
    Restores agent train/eval mode and Torch RNG state even on an exception.
    Returns Python numbers; final metric conversion synchronizes GPU work.
    """
    n_steps = operator.index(n_steps)
    if n_steps < 1:
        raise ValueError('n_steps must be positive')
    if not math.isfinite(gamma) or not 0 <= gamma <= 1:
        raise ValueError('gamma must be finite and in [0, 1]')
    device = torch.device(env.device)
    devices = ([device.index if device.index is not None else torch.cuda.current_device()]
               if device.type == 'cuda' else [])
    was_training = agent.training
    agent.eval()
    try:
        with torch.random.fork_rng(devices=devices):
            # Seed only generators whose states fork_rng preserves.
            torch.random.default_generator.manual_seed(seed)
            if device.type == 'cuda':
                with torch.cuda.device(device):
                    torch.cuda.manual_seed(seed)
            obs, _ = env.reset(seed=seed)
            window_R = torch.zeros(env.num_envs, dtype=torch.float64, device=device)
            window_J = torch.zeros_like(window_R)
            episodes = EpisodeReturns(env.num_envs, device, gamma)
            completed_R, completed_J, completed_lengths, failures = [], [], [], []
            discount = 1.
            for t in trange(n_steps):
                action = agent.distribution(obs).sample() if stochastic else agent.actor(obs)
                obs, reward, terminated, truncated, _ = env.step(action)
                window_R += reward
                window_J += discount * reward
                discount *= gamma
                done = terminated | truncated
                completed = episodes.update(reward, done)
                ids = completed['ids']
                if ids.numel():
                    completed_R.append(completed['R'])
                    completed_J.append(completed['J'])
                    completed_lengths.append(completed['length'])
                    failures.append(terminated[ids].float())
                    if t + 1 < n_steps:
                        obs, _ = env.reset(ids)
            count = sum(x.numel() for x in completed_R)
            def completed_mean(parts):
                return torch.cat(parts).double().mean().item() if parts else None
            return {
                'R': window_R.mean().item(),
                'J': window_J.mean().item(),
                'mean_step_reward': window_R.mean().item() / n_steps,
                'entropy': agent.entropy().item(),
                'eval_steps': n_steps,
                'eval_num_envs': env.num_envs,
                'eval_transitions': n_steps * env.num_envs,
                'completed_episodes': count,
                'completed_episode_R': completed_mean(completed_R),
                'completed_episode_J': completed_mean(completed_J),
                'completed_episode_length': completed_mean(completed_lengths),
                'completed_episode_failure_fraction': completed_mean(failures),
                'unfinished_worlds': int((episodes.length > 0).sum().item()),
            }
    finally:
        agent.train(was_training)
