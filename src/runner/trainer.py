"""CleanRL-style continuous PPO with RSL-RL locomotion PPO choices.

Independent implementation; see docs/TRAINING.md for defaults and semantics.
No CleanRL, Gymnasium, MushroomRL or RSL-RL installation is required.

Based on your fixed-step evaluation / trange launcher. Fixes rollout-wide
statistics and SPS after flattening the collection loop. Add --profile-runtime
for collection/GAE/PPO timing in train.csv and epoch-level evaluation/save/log
wall times in runtime.csv. Profiling synchronizes CUDA at stage boundaries and
therefore adds overhead; use it for diagnosis, then disable it for throughput.
PPO and next-value calculations are preserved. Uses fixed-step evaluation.

Add --profile-collection to split collection into action/value inference,
env.step, next-state value evaluation, resets, rendering and remaining work.
This enables --profile-runtime and synchronizes CUDA around measured blocks
inside each control step. Diagnostic timings include this overhead; disable
profiling for throughput comparisons. env.step timing includes physics, PD,
rewards and observations; it is not physics-only timing. Reset counters count
training collection only, excluding initialization and evaluation. worlds_reset
counts reset events, so the same world may be counted more than once.
"""
import csv
from dataclasses import asdict
import json
import random
import time
import numpy as np
import torch
from tqdm import trange, tqdm
from src.envs import make_env
from .config import parse_train_args as parse_args
from src.runner.utils.paths import run_paths
from src.networks import ActorCritic
from src.algorithms.ppo import EpisodeReturns, Rollout, ppo_update
from .evaluation import evaluate


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

    def __init__(self, path, args, append=False):
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
                self.wb = wandb.init(project=args.wandb_project, group=args.experiment_name,
                                     name=f'{args.experiment_name}/run_{args.run_id}',
                                     dir=str(path), config=asdict(args), mode=args.wandb_mode)
            for kind, fields in (('train', self.train_fields), ('evaluation', self.eval_fields),
                                 ('runtime', self.runtime_fields)):
                csv_path = path / f'{kind}.csv'
                existing = append and csv_path.exists() and csv_path.stat().st_size > 0
                if existing:
                    with csv_path.open(newline='') as old:
                        if next(csv.reader(old), None) != fields:
                            raise ValueError(f'Cannot append incompatible CSV schema: {csv_path}')
                f = csv_path.open('a' if existing else 'w', newline='')
                self.files[kind] = f
                self.writers[kind] = csv.DictWriter(f, fieldnames=fields)
                if not existing:
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


def make_config(args, num_envs, seed):
    from src.envs.config import EnvConfig
    from src.envs.domain_randomized import DomainRandomizationConfig
    from src.envs.terrain_config import TerrainConfig
    cls = ({'domain_randomized':DomainRandomizationConfig,'terrain':TerrainConfig}.get(args.environment,EnvConfig))
    return cls.from_dict(dict(asdict(args), num_envs=num_envs, seed=seed))


def save_checkpoint(path, agent, optimizer, args, env, epoch, update, global_step, best_r):
    payload = dict(agent=agent.state_dict(), optimizer=optimizer.state_dict(), args=asdict(args),
                   env_config=asdict(env.cfg), epoch=epoch, update=update, global_step=global_step,
                   best_R=best_r, torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(),
                   python_rng=random.getstate(), env_rng=env.rng.get_state(),
                   cuda_rng=torch.cuda.get_rng_state_all() if env.device.type == 'cuda' else None)
    payload['schema_version'] = 2
    payload['observation_dim']=env.num_observations
    if hasattr(env,'get_curriculum_state'):
        payload['environment_state']=env.get_curriculum_state()
    if hasattr(env, 'domain_parameters'):
        payload['domain_parameters'] = {name: value.detach().cpu().clone()
                                        for name, value in env.domain_parameters.items()}
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def train(args):
    run_dir, log_dir = run_paths(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.torch_threads)
    env = eval_env = logger = None
    global_step = update = start_epoch = 0
    best_r = -float('inf')
    try:
        env = make_env(make_config(args, args.num_envs, args.seed))
        eval_env = make_env(make_config(args, args.eval_envs, args.seed + 10000))
        agent = ActorCritic(env.num_observations, env.num_actions, args.n_features).to(env.device)
        optimizer = torch.optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)

        if args.resume:
            ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
            old = ckpt['args']
            for name in ('n_features', 'gamma'):
                if old[name] != getattr(args, name):
                    raise ValueError(f'Resume requires matching {name}: saved={old[name]}, requested={getattr(args, name)}')
            agent.load_state_dict(ckpt['agent'])
            if 'environment_state' in ckpt and hasattr(env,'load_curriculum_state'):
                env.load_curriculum_state(ckpt['environment_state'])
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

        logger = Logger(log_dir, args, append=bool(args.resume))

        config = dict(args=asdict(args), env=asdict(env.cfg), batch_size=args.num_envs * args.n_steps_per_fit,
                      updates_per_epoch=args.n_steps // args.n_steps_per_fit,
                      total_target_transitions=args.num_envs * args.n_steps * args.n_epochs,
                      run_directory=str(run_dir), log_directory=str(log_dir))
        (run_dir / 'config.json').write_text(json.dumps(config, indent=2, default=str))
        print(f'Environment: {type(env).__module__}.{type(env).__name__}\n'
              f'Run directory: {run_dir}\nLog directory: {log_dir}\nBatch transitions: {config["batch_size"]}; '
              f'target training transitions: {config["total_target_transitions"]}')
        obs, _ = env.reset(seed=args.seed)

        if args.randomize_init_episode_length:
            env.steps.copy_(torch.randint(env.max_episode_steps, (env.num_envs,), device=env.device,
                                          generator=env.rng))

        tracker = EpisodeReturns(env.num_envs, env.device, args.gamma)
        storage = Rollout(args.n_steps_per_fit, args.num_envs, env.num_observations, env.num_actions, env.device)
        def stamp():
            # Detailed collection profiling also uses this at block boundaries
            # inside a step. Synchronizing attributes CUDA work to the right block.
            if args.profile_runtime and env.device.type == 'cuda':
                torch.cuda.synchronize(env.device)
            return time.perf_counter()

        def evaluate_policy(epoch):
            if not args.record_video:
                return evaluate(agent, eval_env, args.n_steps_test, args.gamma,
                                args.seed + 10000, args.stochastic_eval)
            from .video import make_video
            label = (f'{args.experiment_name}_run{args.run_id}_eval_epoch{epoch:04d}'
                     f'_world{args.video_world_id}')
            with make_video(eval_env, args, label, args.video_world_id) as video:
                result = evaluate(agent, eval_env, args.n_steps_test, args.gamma,
                                  args.seed + 10000, args.stochastic_eval, step_callback=video.capture)
            tqdm.write(f'Evaluation video: {video.path} ({video.frames} frames)')
            return result

        initial_evaluation_start = stamp()
        metrics = evaluate_policy(start_epoch)
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
                metrics = evaluate_policy(epoch)
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
            if args.save_epochs and epoch % args.save_every == 0:
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
