"""Playback and fixed-window evaluation using the shared environment factory."""
import json
from pathlib import Path
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
    config.terrain_eval_all_levels = False
    torch.set_num_threads(config.torch_threads)
    env = None
    video = None
    try:

        env = make_env(make_config(config, config.num_envs, config.seed))
        if hasattr(env,'set_evaluation_mode'): env.set_evaluation_mode(True)
        if config.joystick:
            # Retain keyboard commands across periodic resampling and resets.
            env.manual_commands = True
            env.commands.zero_()
        print(f'Environment: {type(env).__module__}.{type(env).__name__}')
        agent = ActorCritic(env.num_observations, env.num_actions,
                            checkpoint['args']['n_features']).to(env.device)
        expected=checkpoint.get('observation_dim',checkpoint['agent']['actor.fc1.weight'].shape[1])
        if expected!=env.num_observations:
            raise ValueError(f'Checkpoint expects {expected} observations, but the environment supplies '
                             f'{env.num_observations}. Use matching terrain/layout/grid settings.')
        agent.load_state_dict(checkpoint['agent'])
        agent.eval()
        if config.record_video:
            from src.runner.utils.video import make_video
            label = f'{Path(config.checkpoint).parent.name}_{Path(config.checkpoint).stem}_world{config.world_id}'
            video = make_video(env, config, label, config.world_id)
        if config.headless:
            if env.device.type == 'cuda':
                torch.cuda.synchronize(env.device)
            start = time.perf_counter()
            result = evaluate(agent, env, config.steps, checkpoint['args']['gamma'],
                              config.seed, config.stochastic,
                              step_callback=video.capture if video is not None else None)
            if video is not None:
                video.close()
                result['video_path'] = str(video.path)
                result['video_frames'] = video.frames
            if env.device.type == 'cuda':
                torch.cuda.synchronize(env.device)
            result['evaluation_seconds'] = time.perf_counter() - start
            result['transitions_per_second'] = result['eval_transitions'] / result['evaluation_seconds']
            print(json.dumps(result, indent=2))
            return result
        obs, _ = env.reset(seed=config.seed)
        if config.joystick:
            print('Keyboard commands (focus the viewer): Up/Down = +/-vx, '
                  'Left/Right = +/-vy, A/D = +/-wz. Release to stop. '
                  f'Speeds: vx={config.command_vx:g} m/s, vy={config.command_vy:g} m/s, '
                  f'wz={config.command_wz:g} rad/s. Commands apply to all worlds.')
            command_speeds = (config.command_vx, config.command_vy, config.command_wz)
        with torch.no_grad():
            for step in trange(config.steps):
                if config.joystick:
                    # Poll events and handle pause/close before the next action.
                    # This also creates the viewer lazily on the first iteration.
                    if not env.render(config.world_id, throttle=not config.unpaced):
                        break
                    viewer = env.viewer
                    viewer.joystick = True
                    command = tuple(direction * speed for direction, speed in
                                    zip(viewer.command_directions(), command_speeds))
                    env.commands[:] = env.commands.new_tensor(command)
                    viewer.command_values = command
                    # Rebuild through the env API: flat and terrain layouts differ.
                    obs = env.observations()
                action = agent.distribution(obs).sample() if config.stochastic else agent.actor(obs)
                obs, _, terminated, truncated, _ = env.step(action)
                if video is not None:
                    video.capture(env, step + 1)
                if not config.joystick and not env.render(config.world_id, throttle=not config.unpaced):
                    break
                ids = torch.where(terminated | truncated)[0]
                if ids.numel():
                    obs, _ = env.reset(ids)
    finally:
        try:
            if video is not None:
                video.close()
                if not config.headless and video.frames:
                    print(f'Video saved: {video.path} ({video.frames} frames)')
        finally:
            if env is not None:
                env.close()
