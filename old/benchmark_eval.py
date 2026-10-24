"""Benchmark the existing evaluation loop without changing its semantics.

Place in go1_warp/ and run from the project parent directory:
python -m go1_warp.benchmark_eval runs/EXPERIMENT/last.pt \
    --env-counts 1 2 4 5 10 --episodes 10 --device cuda:0

Only load checkpoints you trust (same format/loading behavior as play.py).
"""
import argparse
import csv
from dataclasses import fields
import gc
import json
from pathlib import Path
import statistics
import time

import torch
from . import Config, Go1WarpEnv
from .networks import ActorCritic
from .train import evaluate


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


class CountedEnv:
    """Delegate to the real environment; count batched step calls only."""

    def __init__(self, env):
        self.env = env
        self.calls = 0

    def __getattr__(self, name):
        return getattr(self.env, name)

    def step(self, actions):
        self.calls += 1
        return self.env.step(actions)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('checkpoint', type=Path)
    p.add_argument('--env-counts', nargs='+', type=int, default=[1, 2, 4, 5, 10])
    p.add_argument('--episodes', type=int, default=10, help='Fixed TOTAL episodes per evaluation, not per world')
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--warmups', type=int, default=1, help='Full untimed evaluations per environment count')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--backend', choices=['warp', 'mujoco'], default='warp')
    p.add_argument('--seed', type=int, default=10000)
    p.add_argument('--stochastic', action='store_true')
    p.add_argument('--episode-seconds', type=float, help='Override checkpoint horizon, for smoke tests only')
    p.add_argument('--torch-threads', type=int, default=1)
    p.add_argument('--output', type=Path, default=Path('evaluation_benchmark.csv'))
    a = p.parse_args(argv)
    if min(a.episodes, a.repeats, a.torch_threads, *a.env_counts) < 1 or a.warmups < 0:
        p.error('Counts must be positive; warmups must be nonnegative')
    if max(a.env_counts) > a.episodes:
        p.error('Use env-counts <= episodes; otherwise some worlds have zero episode quota')
    if a.episode_seconds is not None and a.episode_seconds <= 0:
        p.error('episode-seconds must be positive')
    a.env_counts = list(dict.fromkeys(a.env_counts))
    return a


def main(argv=None):
    a = parse_args(argv)
    torch.set_num_threads(a.torch_threads)
    device = torch.device(a.device)
    checkpoint = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    allowed = {f.name for f in fields(Config)}
    config = {k: v for k, v in checkpoint['env_config'].items() if k in allowed}
    config.update(device=a.device, backend=a.backend, seed=a.seed)
    if a.episode_seconds is not None:
        config['episode_seconds'] = a.episode_seconds
    agent = None
    rows = []
    a.output.parent.mkdir(parents=True, exist_ok=True)
    print(f'Total episodes/evaluation: {a.episodes}; warmups: {a.warmups}; repeats: {a.repeats}', flush=True)
    print('Same seed for every repeat; environment-count changes do NOT preserve identical episode scenarios.',
          flush=True)
    if a.warmups == 0:
        print('WARNING: first measured evaluation can include lazy compilation/warmup.', flush=True)
    metadata = {
        'checkpoint': str(a.checkpoint.resolve()), 'device': str(device), 'backend': a.backend,
        'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
        'torch_version': torch.__version__, 'env_config': config,
        'episodes': a.episodes, 'env_counts': a.env_counts, 'repeats': a.repeats,
        'warmups': a.warmups, 'seed': a.seed, 'stochastic': a.stochastic,
        'gamma': checkpoint['args']['gamma'],
    }
    a.output.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')
    with a.output.open('w', newline='') as stream:
        writer = None
        for count in a.env_counts:
            env = None
            try:
                print(f'\nCreating {count} evaluation worlds...', flush=True)
                synchronize(device)
                start = time.perf_counter()
                env = Go1WarpEnv(Config(**dict(config, num_envs=count)))
                synchronize(device)
                setup_seconds = time.perf_counter() - start
                if agent is None:
                    agent = ActorCritic(env.num_observations, env.num_actions,
                                        checkpoint['args']['n_features']).to(device)
                    agent.load_state_dict(checkpoint['agent'])
                    agent.eval()
                wrapped = CountedEnv(env)

                def run():
                    return evaluate(agent, wrapped, a.episodes, checkpoint['args']['gamma'], a.seed, a.stochastic)

                for warmup in range(a.warmups):
                    synchronize(device)
                    start = time.perf_counter()
                    run()
                    synchronize(device)
                    print(f'  warmup {warmup + 1}: {time.perf_counter() - start:.3f}s (excluded)', flush=True)
                measured = []
                for repeat in range(1, a.repeats + 1):
                    wrapped.calls = 0
                    if device.type == 'cuda':
                        torch.cuda.reset_peak_memory_stats(device)
                    synchronize(device)
                    start = time.perf_counter()
                    metrics = run()
                    synchronize(device)
                    seconds = time.perf_counter() - start
                    simulated = wrapped.calls * count
                    # evaluate returns the mean over exactly the requested episodes.
                    scored = round(metrics['eval_episode_length'] * a.episodes)
                    row = dict(num_envs=count, repeat=repeat, episodes=a.episodes,
                               seconds=seconds, seconds_per_episode=seconds / a.episodes,
                               setup_seconds=setup_seconds, batched_step_calls=wrapped.calls,
                               simulated_transitions=simulated, scored_transitions=scored,
                               extra_transitions=simulated - scored,
                               simulated_sps=simulated / seconds, scored_sps=scored / seconds,
                               peak_torch_allocated_mb=(torch.cuda.max_memory_allocated(device) / 2 ** 20
                                                        if device.type == 'cuda' else 0.), **metrics)
                    if writer is None:
                        writer = csv.DictWriter(stream, fieldnames=list(row))
                        writer.writeheader()
                    writer.writerow(row)
                    stream.flush()
                    measured.append(seconds)
                    rows.append(row)
                    print(f'  repeat {repeat}: {seconds:.3f}s | calls={wrapped.calls} | '
                          f'extra={simulated - scored} | R={metrics["R"]:.4f} | J={metrics["J"]:.4f}', flush=True)
                print(f'  median={statistics.median(measured):.3f}s; '
                      f'range={min(measured):.3f}..{max(measured):.3f}s; setup={setup_seconds:.3f}s', flush=True)
            finally:
                if env is not None:
                    env.close()
                # Release wrapper references before trying the next batch size.
                env = None
                wrapped = None
                gc.collect()
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
    print('\nSummary (steady-state evaluation; lower seconds is better):')
    baseline = statistics.median(r['seconds'] for r in rows if r['num_envs'] == a.env_counts[0])
    for count in a.env_counts:
        seconds = statistics.median(r['seconds'] for r in rows if r['num_envs'] == count)
        print(f'  envs={count:5d}  median={seconds:9.3f}s  speedup_vs_{a.env_counts[0]}env={baseline / seconds:.2f}x')
    print(f'\nResults: {a.output}\nMetadata: {a.output.with_suffix(".json")}')
    print('Torch peak memory excludes allocations owned directly by Warp. Run with other GPU jobs stopped.')


if __name__ == '__main__':
    main()