import argparse
import torch
from . import Config, Go1WarpEnv


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--num-envs', type=int, default=1024)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--backend', choices=['warp', 'mujoco'], default='warp')
    p.add_argument('--steps', type=int, default=500)
    p.add_argument('--render', action='store_true')
    p.add_argument('--world-id', type=int, default=0)
    p.add_argument('--unpaced', action='store_true', help='Disable real-time viewer pacing')
    p.add_argument('--collision-only', action='store_true', help='Strip visual meshes for debugging')
    args = p.parse_args()
    env = Go1WarpEnv(Config(num_envs=args.num_envs, device=args.device,
                          backend=args.backend, collision_only=args.collision_only))
    # Zero actions test the PD standing controller, not a trained walking policy.
    actions = torch.zeros((args.num_envs, 12), device=env.device)
    try:
        for i in range(args.steps):
            obs, reward, terminated, truncated, info = env.step(actions)
            # Render terminal state before resetting these worlds.
            if args.render and not env.render(args.world_id, throttle=not args.unpaced):
                break
            # PPO must evaluate V(obs) / store final obs BEFORE resetting these rows.
            done_ids = torch.where(terminated | truncated)[0]
            if done_ids.numel():
                obs, _ = env.reset(done_ids)
            if i % 100 == 0:
                print(f'step={i}, mean reward={reward.mean().item():.5f}, obs={tuple(obs.shape)}')
    finally:
        env.close()


if __name__ == '__main__':
    main()
