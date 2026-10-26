"""Dataclasses are the defaults; argparse supplies explicit overrides."""
import argparse
from dataclasses import dataclass, fields
from pathlib import Path
from typing import get_args, get_origin, get_type_hints

from src.envs.domain_randomized import DomainRandomizationConfig
from src.envs.terrain_config import TerrainConfig, TERRAIN_TYPES
from .paths import rooted


@dataclass
class TrainConfig(TerrainConfig):
    environment: str = 'flat'
    kp: float = 30.
    kd: float = .8
    reset_joint_noise: float = .03
    command_seconds: float = 5.
    num_envs: int = 100
    command_vx: float = 1.5
    command_vy: float = .8
    n_steps: int = 10000
    n_epochs: int = 100
    n_steps_per_fit: int = 1000
    n_steps_test: int = 1000
    n_minibatches: int = 20
    n_epochs_policy: int = 10
    n_features: int = 256
    lr: float = 1e-3
    eps_ppo: float = .2
    lam: float = .95
    ent_coeff: float = .01
    gamma: float = .99
    experiment_name: str = 'LLC_PPO'
    run_id: int = 0
    output_dir: str = 'runs'
    log_dir: str = 'logs'
    resume: str | None = None
    schedule: str = 'adaptive'
    desired_kl: float = .01
    max_grad_norm: float = 1.
    vf_coef: float = 1.
    clip_vloss: bool = True
    randomize_init_episode_length: bool = True
    log_std_min: float = -5.
    log_std_max: float = 2.
    eval_envs: int = 4
    eval_every: int = 1
    stochastic_eval: bool = False
    save_epochs: bool = False
    save_every: int = 10
    render: bool = False
    profile_runtime: bool = False
    profile_collection: bool = False
    torch_threads: int = 1
    tensorboard: bool = False
    log: bool = False
    wandb_project: str = 'POLICY_TRAINING_WARP'
    wandb_mode: str = 'online'


@dataclass
class PlayConfig(TerrainConfig):
    environment: str = 'flat'
    kp: float = 30.
    kd: float = .8
    reset_joint_noise: float = .03
    command_seconds: float = 5.
    num_envs: int = 1
    seed: int = 10000
    checkpoint: str = ''
    steps: int = 10000
    headless: bool = False
    stochastic: bool = False
    unpaced: bool = False
    world_id: int = 0
    torch_threads: int = 1


ALIASES = {
    'num_envs': ('--n-envs', '--n_envs'),
    'experiment_name': ('--name-exp', '--name_exp'),
    'command_vx': ('--target-vx', '--target_vx'),
    'command_vy': ('--target-vy', '--target_vy'),
    'command_wz': ('--target-wz', '--target_wz'),
    'environment': ('--env',),
    'use_physics_graph': ('--physics-graph',),
    'log': ('--track',),
}
CHOICES = {'environment': ('flat', 'domain_randomized', 'blind', 'terrain'),
           'terrain_types': TERRAIN_TYPES, 'terrain_observation_layout': ('legacy','compact'),
           'backend': ('warp', 'mujoco'), 'schedule': ('adaptive', 'fixed'),
           'wandb_mode': ('online', 'offline', 'disabled')}


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('true', 'yes', '1', 't'):
        return True
    if value.lower() in ('false', 'no', '0', 'f'):
        return False
    raise argparse.ArgumentTypeError('Expected true or false')


def config_parser(cls, description):
    parser = argparse.ArgumentParser(description=description,
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    types = get_type_hints(cls)
    for field in fields(cls):
        name = field.name
        if name == 'checkpoint':
            parser.add_argument('checkpoint', help='Checkpoint path, relative to the repository or absolute')
            continue
        flags = list(dict.fromkeys(('--' + name.replace('_', '-'), '--' + name, *ALIASES.get(name, ()))))
        opts = dict(dest=name, default=argparse.SUPPRESS,
                    help=f'{field.metadata.get("help", name.replace("_", " "))} (default: {field.default})')
        typ = types[name]
        if name == 'log':
            opts.update(type=str2bool, nargs='?', const=True)
        elif typ is bool:
            opts['action'] = argparse.BooleanOptionalAction
        elif get_origin(typ) is tuple:
            item_type=get_args(typ)[0]
            opts.update(type=item_type,nargs='+' if Ellipsis in get_args(typ) else len(get_args(typ)))
        else:
            opts['type'] = next((t for t in get_args(typ) if t is not type(None)), typ)
        if name in CHOICES:
            opts['choices'] = CHOICES[name]
        parser.add_argument(*flags, **opts)
    return parser


def migrated(values):
    """Old checkpoints stored these argument names; network keys are unchanged."""
    aliases = {'n_envs': 'num_envs', 'name_exp': 'experiment_name',
               'target_vx': 'command_vx', 'target_vy': 'command_vy', 'target_wz': 'command_wz'}
    return {aliases.get(name, name): value for name, value in values.items()}


def saved_values(checkpoint):
    # env_config records actual resolved physics options, including old exports.
    values = {**migrated(checkpoint.get('args', {})), **checkpoint.get('env_config', {})}
    path = values.get('xml_path')
    if path and not rooted(path).exists():
        old = Path(path)
        if old.parent.name == 'models' and old.name in ('walk_go1.xml', 'go1_robot.xml'):
            values['xml_path'] = str(rooted('src/models') / old.name)
    return values


def load_checkpoint(path):
    import torch
    return torch.load(rooted(path), map_location='cpu', weights_only=False)


def _construct(cls, explicit, saved=None):
    allowed = {field.name for field in fields(cls)}
    values = {key: value for key, value in (saved or {}).items() if key in allowed}
    values.update(explicit)
    if values.get('environment')=='terrain':
        defaults=TerrainConfig()
        for name in ('kp','kd','reset_joint_noise','command_seconds'):
            values.setdefault(name,getattr(defaults,name))
    for field in fields(cls):
        if field.name in values and isinstance(field.default, tuple):
            values[field.name] = tuple(values[field.name])
    result = cls(**values)
    result._explicit_fields = frozenset(explicit)
    return result


def validate_environment(config):
    if config.num_envs < 1 or config.decimation < 1 or config.nconmax < 1 or config.njmax < 1:
        raise ValueError('Environment counts, decimation and buffer capacities must be positive')
    if config.solver_iterations < 1 or config.solver_ls_iterations < 1:
        raise ValueError('Solver iteration limits must be positive')
    if config.episode_seconds <= 0 or config.command_seconds <= 0:
        raise ValueError('Episode and command durations must be positive')
    config.validate_randomization()
    if config.environment=='terrain':
        config.validate_terrain()


def parse_train_args(argv=None):
    parser = config_parser(TrainConfig, 'Train Go1 with continuous PPO.')
    explicit = vars(parser.parse_args(argv))
    saved = saved_values(load_checkpoint(explicit['resume'])) if explicit.get('resume') else None
    config = _construct(TrainConfig, explicit, saved)
    if config.resume:
        config.resume = str(rooted(config.resume))
    if 'save_every' in explicit and 'save_epochs' not in explicit:
        config.save_epochs = config.save_every > 0
    if config.profile_collection:
        config.profile_runtime = True
    try:
        validate_environment(config)
        for name in ('n_steps', 'n_epochs', 'n_steps_per_fit', 'n_steps_test', 'n_minibatches',
                     'n_epochs_policy', 'eval_envs', 'eval_every', 'torch_threads'):
            if getattr(config, name) < 1:
                raise ValueError(f'{name} must be positive')
        if config.n_features < 2 or config.n_minibatches > config.num_envs * config.n_steps_per_fit:
            raise ValueError('n_features must be >=2; minibatches cannot exceed rollout transitions')
        if config.n_steps % config.n_steps_per_fit:
            raise ValueError('n_steps must be divisible by n_steps_per_fit')
        if config.save_every < 0 or (config.save_epochs and config.save_every < 1):
            raise ValueError('Use --save-every >=1 with --save-epochs, or --no-save-epochs to disable')
        if not 0 <= config.gamma <= 1 or not 0 <= config.lam <= 1:
            raise ValueError('gamma and lam must be in [0,1]')
        if min(config.lr, config.desired_kl, config.eps_ppo, config.max_grad_norm) <= 0:
            raise ValueError('Learning rate, KL, PPO clip and gradient norm must be positive')
        if config.ent_coeff < 0 or config.vf_coef < 0 or config.log_std_min > 0 or config.log_std_max < 0:
            raise ValueError('Invalid loss coefficients or log-standard-deviation bounds')
        if (not config.experiment_name or config.experiment_name in ('.', '..')
                or any(char in config.experiment_name for char in ('/', '\\', '\0'))
                or config.run_id < 0):
            raise ValueError('Experiment name must be one directory name; run ID must be nonnegative')
    except ValueError as error:
        parser.error(str(error))
    return config


def parse_play_args(argv=None):
    parser = config_parser(PlayConfig, 'Evaluate or visualize a Go1 checkpoint.')
    explicit = vars(parser.parse_args(argv))
    config = _construct(PlayConfig, explicit)
    if config.steps < 1 or config.torch_threads < 1:
        parser.error('steps and torch-threads must be positive')
    return config


def resolve_play_config(config, checkpoint):
    # Saved physics/DR settings are inherited; playback has its own runtime defaults.
    saved = {key: value for key, value in saved_values(checkpoint).items()
             if key in {field.name for field in fields(TerrainConfig)}}
    for name in ('num_envs', 'device', 'backend', 'seed'):
        saved.pop(name, None)
    from dataclasses import asdict
    explicit_fields = getattr(config, '_explicit_fields', set(asdict(config)))
    explicit = {name: value for name, value in asdict(config).items() if name in explicit_fields}
    result = _construct(PlayConfig, explicit, saved)
    if 'terrain_start_level' in explicit and 'terrain_eval_level' not in explicit:
        result.terrain_eval_level=result.terrain_start_level
    result.checkpoint = str(rooted(result.checkpoint))
    validate_environment(result)
    if not 0 <= result.world_id < result.num_envs:
        raise ValueError('world-id must be within num-envs')
    return result
