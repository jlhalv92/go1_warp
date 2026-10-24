"""CleanRL-style continuous PPO with RSL-RL locomotion PPO choices.

Independent implementation; see TRAINING.md for sources, defaults and semantics.
No CleanRL, Gymnasium, MushroomRL or RSL-RL installation is required.

Based on your fixed-step evaluation / trange launcher. Fixes rollout-wide
statistics and SPS after flattening the collection loop. Add --profile-runtime
for collection/GAE/PPO timing in train.csv and epoch-level evaluation/save/log
wall times in runtime.csv. Profiling synchronizes CUDA at stage boundaries and
therefore adds overhead; use it for diagnosis, then disable it for throughput.
PPO, next-value calculation, environment behavior and save frequency are unchanged.
Uses your existing .evaluation_utils.evaluate fixed-step function.

Add --profile-collection to split collection into action/value inference,
env.step, next-state value evaluation, resets, rendering and remaining work.
This enables --profile-runtime and synchronizes CUDA around measured blocks
inside each control step. Diagnostic timings include this overhead; disable
profiling for throughput comparisons. env.step timing includes physics, PD,
rewards and observations; it is not physics-only timing. Reset counters count
training collection only, excluding initialization and evaluation. worlds_reset
counts reset events, so the same world may be counted more than once.
"""
import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from tqdm import trange, tqdm
# from . import Config, Go1WarpEnv
from .env_locomotion_contacts import Config, Go1WarpEnv
from .networks import ActorCritic
from .ppo import EpisodeReturns, Rollout, ppo_update
from .evaluation_utils import evaluate


def str2bool(x):
    if isinstance(x, bool):
        return x
    if x.lower() in ('true', 'yes', '1', 't'):
        return True
    if x.lower() in ('false', 'no', '0', 'f'):
        return False
    raise argparse.ArgumentTypeError('Expected true or false')


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    # Original script names retained, with hyphenated aliases.
    defaults = dict(n_envs=100, n_steps=10000, n_epochs=100, n_steps_per_fit=1000,
                    n_steps_test=1000, n_minibatches=20, n_epochs_policy=10,
                    n_features=256)
    for key, value in defaults.items():
        p.add_argument('--' + key.replace('_', '-'), '--' + key, type=int, default=value)
    for key, value in dict(lr=1e-3, eps_ppo=.2, lam=.95, ent_coeff=.01, gamma=.99,
                           target_vx=1.5, target_vy=.8, target_wz=1.).items():
        p.add_argument('--' + key.replace('_', '-'), '--' + key, type=float, default=value)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--backend', choices=('warp', 'mujoco'), default='warp')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--name-exp', '--name_exp', default='LLC_FLAT_PPO')
    p.add_argument('--output-dir', default='runs')
    p.add_argument('--resume', type=Path, help='Resume weights/optimizer; simulator starts fresh')
    p.add_argument('--schedule', choices=('adaptive', 'fixed'), default='adaptive')
    p.add_argument('--desired-kl', type=float, default=.01)
    p.add_argument('--max-grad-norm', type=float, default=1.)
    p.add_argument('--vf-coef', type=float, default=1.)
    p.add_argument('--clip-vloss', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--randomize-init-episode-length', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--log-std-min', type=float, default=-5.)
    p.add_argument('--log-std-max', type=float, default=2.)
    p.add_argument('--episode-seconds', type=float, default=20.)
    p.add_argument('--eval-envs', type=int, default=4)
    p.add_argument('--eval-every', type=int, default=1, help='Evaluate every this many outer epochs')
    p.add_argument('--stochastic-eval', action='store_true')
    p.add_argument('--save-every', type=int, default=10, help='Numbered checkpoint interval in outer epochs')
    p.add_argument('--collision-only', action='store_true')
    p.add_argument('--render', action='store_true', help='Unpaced training visualization; closing continues headless')
    p.add_argument('--profile-runtime', action='store_true',
                   help='Time stages with CUDA synchronization at rollout/stage boundaries; adds profiling overhead')
    p.add_argument('--profile-collection', action='store_true',
                   help='Split collection timings with CUDA synchronization around each measured block; implies --profile-runtime')
    p.add_argument('--torch-threads', type=int, default=1)
    p.add_argument('--tensorboard', action='store_true')
    p.add_argument('--log', '--track', type=str2bool, nargs='?', const=True, default=False)
    p.add_argument('--wandb-project', '--wandb_project', default='POLICY_TRAINING_WARP')
    p.add_argument('--wandb-mode', choices=('online', 'offline'), default='online')
    a = p.parse_args(argv)
    if a.profile_collection:
        a.profile_runtime = True
    for name in (*defaults, 'eval_envs', 'eval_every', 'save_every', 'torch_threads'):
        if getattr(a, name) < 1:
            p.error(f'{name} must be positive')
    if a.n_features < 2 or a.n_minibatches > a.n_envs * a.n_steps_per_fit:
        p.error('n_features must be >=2 and minibatches cannot exceed rollout transitions')
    if a.n_steps % a.n_steps_per_fit:
        p.error('n_steps must be divisible by n_steps_per_fit; no transitions are silently discarded')
    if not 0 <= a.gamma <= 1 or not 0 <= a.lam <= 1:
        p.error('gamma and lam must lie in [0,1]')
    if a.lr <= 0 or a.desired_kl <= 0 or a.eps_ppo <= 0 or a.max_grad_norm <= 0:
        p.error('lr, desired-kl, eps-ppo and max-grad-norm must be positive')
    if a.ent_coeff < 0 or a.vf_coef < 0 or a.log_std_min > 0 or a.log_std_max < 0:
        p.error('Loss coefficients must be nonnegative; log-std bounds must include initial log std=0')
    return a


class Logger:
    collection_fields = ['action_inference_seconds', 'next_value_seconds',
                         'inference_seconds', 'env_step_seconds', 'reset_seconds',
                         'render_seconds', 'collection_other_seconds']
    train_fields = ['epoch', 'update', 'global_step', 'train_R', 'train_J', 'train_episodes',
                    'train_episode_length', 'entropy', 'std_mean', 'learning_rate', 'policy_loss',
                    'value_loss', 'kl', 'clip_fraction', 'explained_variance', 'actor_grad_norm',
                    'critic_grad_norm', 'SPS', 'collection_seconds', 'gae_seconds', 'ppo_seconds',
                    'reset_calls', 'worlds_reset'] + collection_fields
    runtime_fields = ['epoch', 'update', 'global_step', 'epoch_seconds',
                      'evaluation_seconds', 'checkpoint_seconds', 'logging_seconds',
                      'reset_calls', 'worlds_reset'] + collection_fields
    eval_fields = [
        'epoch', 'update', 'global_step',
        'R', 'J', 'entropy', 'mean_step_reward',
        'eval_steps', 'eval_num_envs', 'eval_transitions',
        'completed_episodes',
        'completed_episode_R', 'completed_episode_J',
        'completed_episode_length',
        'completed_episode_failure_fraction',
        'unfinished_worlds',
    ]

    def __init__(self, path, args):
        self.path = path
        path.mkdir(parents=True, exist_ok=True)
        self.files = {}
        self.writers = {}
        self.tb = self.wb = None
        try:
            if args.tensorboard:
                from torch.utils.tensorboard import SummaryWriter
                self.tb = SummaryWriter(str(path / 'tensorboard'))
            if args.log:
                import wandb
                self.wb = wandb.init(project=args.wandb_project, group='flat',
                                     name=path.name, config=vars(args), mode=args.wandb_mode)
            for kind, fields in (('train', self.train_fields), ('evaluation', self.eval_fields),
                                 ('runtime', self.runtime_fields)):
                f = (path / f'{kind}.csv').open('w', newline='')
                self.files[kind] = f
                self.writers[kind] = csv.DictWriter(f, fieldnames=fields)
                self.writers[kind].writeheader()
        except Exception:
            self.close()
            raise

    def write(self, kind, metrics):
        self.writers[kind].writerow(metrics)
        self.files[kind].flush()
        prefix = {'train': 'train/', 'runtime': 'runtime/'}.get(kind, '')
        step = metrics['global_step']
        payload = {prefix + k: v for k, v in metrics.items() if v is not None and k != 'global_step'}
        if self.wb:
            # commit=False allows eval at the same transition count to join this update.
            self.wb.log(payload, step=step, commit=False)
        if self.tb:
            for k, v in payload.items():
                self.tb.add_scalar(k, v, step)

    def close(self):
        for f in self.files.values():
            f.close()
        if self.tb:
            self.tb.close()
        if self.wb:
            self.wb.finish()


def make_config(args, n_envs, seed):
    return Config(num_envs=n_envs, device=args.device, backend=args.backend, seed=seed,
                  collision_only=args.collision_only, episode_seconds=args.episode_seconds,
                  command_vx=args.target_vx, command_vy=args.target_vy, command_wz=args.target_wz)


def save_checkpoint(path, agent, optimizer, args, env, epoch, update, global_step, best_r):
    payload = dict(agent=agent.state_dict(), optimizer=optimizer.state_dict(), args=vars(args),
                   env_config=asdict(env.cfg), epoch=epoch, update=update, global_step=global_step,
                   best_R=best_r, torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(),
                   python_rng=random.getstate(), env_rng=env.rng.get_state(),
                   cuda_rng=torch.cuda.get_rng_state_all() if env.device.type == 'cuda' else None)
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.torch_threads)
    env = eval_env = logger = None
    global_step = update = start_epoch = 0
    best_r = -float('inf')
    try:
        env = Go1WarpEnv(make_config(args, args.n_envs, args.seed))
        eval_env = Go1WarpEnv(make_config(args, args.eval_envs, args.seed + 10000))
        agent = ActorCritic(env.num_observations, env.num_actions, args.n_features).to(env.device)
        optimizer = torch.optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)

        if args.resume:
            ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
            old = ckpt['args']
            for name in ('n_features', 'gamma', 'target_vx', 'target_vy', 'target_wz', 'episode_seconds'):
                if old[name] != getattr(args, name):
                    raise ValueError(f'Resume requires matching {name}: saved={old[name]}, requested={getattr(args, name)}')
            agent.load_state_dict(ckpt['agent'])
            optimizer.load_state_dict(ckpt['optimizer'])
            for group in optimizer.param_groups:
                for param in group['params']:
                    for key, value in optimizer.state[param].items():
                        if isinstance(value, torch.Tensor) and key != 'step':
                            optimizer.state[param][key] = value.to(env.device)
            start_epoch, update, global_step = ckpt['epoch'], ckpt['update'], ckpt['global_step']
            best_r = ckpt['best_R']
            torch.set_rng_state(ckpt['torch_rng'])
            random.setstate(ckpt['python_rng'])
            np.random.set_state(ckpt['numpy_rng'])
            if env.device.type == 'cuda' and ckpt['cuda_rng'] is not None:
                torch.cuda.set_rng_state_all(ckpt['cuda_rng'])
            print('Resumed policy and optimizer; episodes restart with fresh simulator state.')

        if start_epoch >= args.n_epochs:
            raise ValueError('n_epochs must exceed the checkpoint epoch')

        stamp = time.strftime('%Y%m%d-%H%M%S') + f'-{time.time_ns() % 1000000:06d}'
        run_dir = Path(args.output_dir) / f'{args.name_exp}_seed{args.seed}_{stamp}'
        logger = Logger(run_dir, args)

        config = dict(args=vars(args), env=asdict(env.cfg), batch_size=args.n_envs * args.n_steps_per_fit,
                      updates_per_epoch=args.n_steps // args.n_steps_per_fit,
                      total_target_transitions=args.n_envs * args.n_steps * args.n_epochs)
        (run_dir / 'config.json').write_text(json.dumps(config, indent=2, default=str))
        print(f'Run directory: {run_dir}\nBatch transitions: {config["batch_size"]}; '
              f'target training transitions: {config["total_target_transitions"]}')
        obs, _ = env.reset(seed=args.seed)

        if args.randomize_init_episode_length:
            env.steps.copy_(torch.randint(env.max_episode_steps, (env.num_envs,), device=env.device,
                                          generator=env.rng))

        tracker = EpisodeReturns(env.num_envs, env.device, args.gamma)
        storage = Rollout(args.n_steps_per_fit, args.n_envs, env.num_observations, env.num_actions, env.device)
        def stamp():
            # Detailed collection profiling also uses this at block boundaries
            # inside a step. Synchronizing attributes CUDA work to the right block.
            if args.profile_runtime and env.device.type == 'cuda':
                torch.cuda.synchronize(env.device)
            return time.perf_counter()

        initial_evaluation_start = stamp()
        metrics = evaluate(agent, eval_env, args.n_steps_test, args.gamma, args.seed + 10000, args.stochastic_eval)
        initial_evaluation_seconds = stamp() - initial_evaluation_start
        tqdm.write(f'Initial evaluation time: {initial_evaluation_seconds:.3f}s (includes first-use warmup)')
        metrics.update(epoch=start_epoch, update=update, global_step=global_step)
        logger.write('evaluation', metrics)
        print(f'Initial evaluation: R={metrics["R"]:.4f} J={metrics["J"]:.4f} entropy={metrics["entropy"]:.4f}')

        if metrics['R'] > best_r:
            best_r = metrics['R']
            save_checkpoint(run_dir / 'best.pt', agent, optimizer, args, env, start_epoch, update, global_step, best_r)
        show = args.render

        for epoch in trange(start_epoch + 1, args.n_epochs + 1, desc='epochs', mininterval=1.0):
            epoch_start = stamp()
            logging_seconds = evaluation_seconds = checkpoint_seconds = 0.0
            epoch_collection = dict.fromkeys(Logger.collection_fields, 0.0)
            epoch_reset_calls = epoch_worlds_reset = 0
            t = 0
            for _ in trange(args.n_steps, desc='control steps', leave=False, mininterval=1.0):
                # These belong to the entire rollout, not to one simulation step.
                if t == 0:
                    collection_start = stamp()
                    completed_r, completed_j, completed_len = [], [], []
                    collection_parts = dict.fromkeys(Logger.collection_fields, 0.0)
                    reset_calls = worlds_reset = 0
                with torch.no_grad():

                    storage.obs[t].copy_(obs)
                    if args.profile_collection:
                        block_start = stamp()
                    action, logprob, _, value, mu, sigma = agent.action_value(obs)
                    if args.profile_collection:
                        collection_parts['action_inference_seconds'] += stamp() - block_start
                    storage.actions[t].copy_(action)  # Raw Gaussian sample, NOT env-clipped action.
                    storage.logprob[t].copy_(logprob)
                    storage.values[t].copy_(value)
                    storage.mu[t].copy_(mu)
                    storage.sigma[t].copy_(sigma)
                    if args.profile_collection:
                        block_start = stamp()
                    next_obs, reward, terminated, truncated, _ = env.step(action)
                    if args.profile_collection:
                        collection_parts['env_step_seconds'] += stamp() - block_start
                    storage.rewards[t].copy_(reward)
                    storage.terminated[t].copy_(terminated)
                    storage.truncated[t].copy_(truncated)
                    # Only valid nonterminal next states enter the critic. Timeouts bootstrap.
                    if args.profile_collection:
                        block_start = stamp()
                    storage.next_values[t].zero_()
                    eligible = ~terminated
                    if bool(eligible.any()):
                        if not bool(torch.isfinite(next_obs[eligible]).all()):
                            raise FloatingPointError('Nonfinite nonterminal observation')
                        storage.next_values[t, eligible] = agent.value(next_obs[eligible])
                    if args.profile_collection:
                        # Includes masks, finite-state checks, indexing and value copy.
                        collection_parts['next_value_seconds'] += stamp() - block_start
                    done = terminated | truncated
                    completed = tracker.update(reward, done)
                    ids = completed['ids']
                    if ids.numel():
                        completed_r.append(completed['R'])
                        completed_j.append(completed['J'])
                        completed_len.append(completed['length'])
                    if show:
                        if args.profile_collection:
                            block_start = stamp()
                        if not env.render(0, throttle=False):
                            show = False
                        if args.profile_collection:
                            collection_parts['render_seconds'] += stamp() - block_start
                    obs = next_obs
                    if ids.numel():
                        reset_calls += 1
                        worlds_reset += ids.numel()
                        if args.profile_collection:
                            block_start = stamp()
                        obs, _ = env.reset(ids)
                        if args.profile_collection:
                            collection_parts['reset_seconds'] += stamp() - block_start
                    global_step += env.num_envs
                    t += 1
                if t == args.n_steps_per_fit:
                    t = 0
                    collection_end = stamp()
                    if args.profile_collection:
                        collection_parts['inference_seconds'] = (
                            collection_parts['action_inference_seconds'] + collection_parts['next_value_seconds'])
                        measured = sum(collection_parts[k] for k in (
                            'inference_seconds', 'env_step_seconds', 'reset_seconds', 'render_seconds'))
                        # Buffer writes, episode tracking, Python checks, progress-bar
                        # work, and profiling overhead fall in this residual.
                        collection_parts['collection_other_seconds'] = collection_end - collection_start - measured
                    storage.finish(args.gamma, args.lam)
                    gae_end = stamp()
                    stats = ppo_update(agent, optimizer, storage, args)
                    update += 1

                    if env.device.type == 'cuda':
                        torch.cuda.synchronize(env.device)

                    update_end = time.perf_counter()
                    log_start = time.perf_counter()
                    stats.update(epoch=epoch, update=update, global_step=global_step,
                                 train_R=torch.cat(completed_r).mean().item() if completed_r else None,
                                 train_J=torch.cat(completed_j).mean().item() if completed_j else None,
                                 train_episodes=sum(len(x) for x in completed_r),
                                 train_episode_length=torch.cat(completed_len).float().mean().item() if completed_len else None,
                                 SPS=config['batch_size'] / (update_end - collection_start),
                                 collection_seconds=collection_end - collection_start if args.profile_runtime else None,
                                 gae_seconds=gae_end - collection_end if args.profile_runtime else None,
                                 ppo_seconds=update_end - gae_end if args.profile_runtime else None,
                                 reset_calls=reset_calls, worlds_reset=worlds_reset)
                    stats.update({key: value if args.profile_collection else None
                                  for key, value in collection_parts.items()})
                    epoch_reset_calls += reset_calls
                    epoch_worlds_reset += worlds_reset
                    if args.profile_collection:
                        for key, value in collection_parts.items():
                            epoch_collection[key] += value
                    logger.write('train', stats)
                    tqdm.write(f'epoch={epoch} update={update} step={global_step} '
                          f'entropy={stats["entropy"]:.3f} KL={stats["kl"]:.5f} '
                          f'lr={stats["learning_rate"]:.2e} SPS={stats["SPS"]:.0f}')
                    if args.profile_runtime:
                        tqdm.write(f'  collection={stats["collection_seconds"]:.3f}s '
                                   f'GAE={stats["gae_seconds"]:.3f}s PPO={stats["ppo_seconds"]:.3f}s')
                    if args.profile_collection:
                        tqdm.write(f'  inference={stats["inference_seconds"]:.3f}s '
                                   f'(action/value={stats["action_inference_seconds"]:.3f}s '
                                   f'next-value={stats["next_value_seconds"]:.3f}s) '
                                   f'env.step={stats["env_step_seconds"]:.3f}s '
                                   f'resets={stats["reset_seconds"]:.3f}s '
                                   f'render={stats["render_seconds"]:.3f}s '
                                   f'other={stats["collection_other_seconds"]:.3f}s')
                        mean_reset_ms = stats['reset_seconds'] * 1000 / reset_calls if reset_calls else 0.0
                        tqdm.write(f'  reset_calls={reset_calls}/{args.n_steps_per_fit} control steps; '
                                   f'worlds_reset={worlds_reset}; '
                                   f'mean_reset_call={mean_reset_ms:.2f}ms')
                    logging_seconds += time.perf_counter() - log_start

            if epoch % args.eval_every == 0 or epoch == args.n_epochs:
                evaluation_start = stamp()
                metrics = evaluate(agent, eval_env, args.n_steps_test, args.gamma,
                                   args.seed + 10000, args.stochastic_eval)
                evaluation_seconds = stamp() - evaluation_start
                metrics.update(epoch=epoch, update=update, global_step=global_step)
                log_start = time.perf_counter()
                logger.write('evaluation', metrics)
                tqdm.write(f'Evaluation epoch={epoch}: R={metrics["R"]:.4f} J={metrics["J"]:.4f} '
                      f'entropy={metrics["entropy"]:.4f} time={evaluation_seconds:.3f}s')
                logging_seconds += time.perf_counter() - log_start
                if metrics['R'] > best_r:
                    best_r = metrics['R']
                    checkpoint_start = stamp()
                    save_checkpoint(run_dir / 'best.pt', agent, optimizer, args, env, epoch, update, global_step, best_r)
                    checkpoint_seconds += stamp() - checkpoint_start
            checkpoint_start = stamp()
            save_checkpoint(run_dir / 'last.pt', agent, optimizer, args, env, epoch, update, global_step, best_r)
            if epoch % args.save_every == 0:
                save_checkpoint(run_dir / f'epoch_{epoch:04d}.pt', agent, optimizer, args, env, epoch, update, global_step, best_r)
            checkpoint_seconds += stamp() - checkpoint_start
            if args.profile_runtime:
                runtime = dict(epoch=epoch, update=update, global_step=global_step,
                               epoch_seconds=stamp() - epoch_start,
                               evaluation_seconds=evaluation_seconds,
                               checkpoint_seconds=checkpoint_seconds, logging_seconds=logging_seconds,
                               reset_calls=epoch_reset_calls, worlds_reset=epoch_worlds_reset)
                runtime.update({key: value if args.profile_collection else None
                                for key, value in epoch_collection.items()})
                logger.write('runtime', runtime)
                tqdm.write(f'Epoch {epoch}: total={runtime["epoch_seconds"]:.3f}s '
                           f'evaluation={evaluation_seconds:.3f}s saves={checkpoint_seconds:.3f}s '
                           f'logging={logging_seconds:.3f}s')
        return run_dir
    finally:
        if logger is not None:
            logger.close()
        if eval_env is not None:
            eval_env.close()
        if env is not None:
            env.close()


def main():
    train(parse_args())


if __name__ == '__main__':
    main()