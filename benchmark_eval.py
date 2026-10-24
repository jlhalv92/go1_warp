"""Benchmark fixed-step evaluation: python benchmark_eval.py CHECKPOINT --help."""
from dataclasses import replace
import csv
import json
import statistics
import time

import torch

from src.algorithms.config import (PlayConfig, config_parser, load_checkpoint,
                                   resolve_play_config)
from src.algorithms.paths import rooted
from src.algorithms.trainer import make_config
from src.algorithms.evaluation import evaluate
from src.envs import make_env
from src.networks import ActorCritic


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def main(argv=None):
    parser = config_parser(PlayConfig, __doc__)
    parser.add_argument('--env-counts', nargs='+', type=int, default=[1, 10, 100, 1024, 4096])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--warmups', type=int, default=1)
    parser.add_argument('--output', default='logs/evaluation_benchmark.csv')
    values = vars(parser.parse_args(argv))
    counts, repeats, warmups, output = (values.pop(name) for name in
                                       ('env_counts', 'repeats', 'warmups', 'output'))
    if min(*counts, repeats) < 1 or warmups < 0:
        parser.error('Environment counts and repeats must be positive; warmups must be nonnegative')
    checkpoint = load_checkpoint(values['checkpoint'])
    config = PlayConfig(**values)
    config._explicit_fields = frozenset(values)
    config = resolve_play_config(config, checkpoint)
    torch.set_num_threads(config.torch_threads)
    output = rooted(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for count in dict.fromkeys(counts):
        current = replace(config, num_envs=count)
        env = None
        try:
            env = make_env(make_config(current, count, config.seed))
            agent = ActorCritic(env.num_observations, env.num_actions,
                                checkpoint['args']['n_features']).to(env.device)
            agent.load_state_dict(checkpoint['agent'])
            def run():
                return evaluate(agent, env, config.steps, checkpoint['args']['gamma'],
                                config.seed, config.stochastic)
            for _ in range(warmups):
                run()
                synchronize(env.device)
            for repeat in range(repeats):
                synchronize(env.device)
                start = time.perf_counter()
                metrics = run()
                synchronize(env.device)
                seconds = time.perf_counter() - start
                rows.append(dict(num_envs=count, repeat=repeat, seconds=seconds,
                                 transitions_per_second=metrics['eval_transitions']/seconds, **metrics))
            times=[row['seconds'] for row in rows if row['num_envs']==count]
            print(f'envs={count}: median evaluation time={statistics.median(times):.3f}s')
        finally:
            if env is not None:
                env.close()
    with output.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    output.with_suffix('.json').write_text(json.dumps(dict(checkpoint=config.checkpoint,
                                                         steps=config.steps, warmups=warmups,
                                                         repeats=repeats, env_counts=counts),indent=2))
    print(f'Results: {output}')
    return rows


if __name__ == '__main__':
    main()
