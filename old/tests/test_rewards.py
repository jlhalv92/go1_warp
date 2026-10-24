import unittest
import numpy as np
import torch
import mujoco
from go1_warp import Config, Go1WarpEnv


class RewardTests(unittest.TestCase):
    def make(self):
        return Go1WarpEnv(Config(num_envs=2, device='cpu', backend='mujoco', reset_joint_noise=0))

    def test_rates_and_nonnegative_total(self):
        e = self.make()
        e.commands.zero_()
        e.qpos[:, 2] = .34
        e.qvel.zero_()
        previous = torch.ones_like(e.actions)
        prev_velocity = torch.ones_like(e.actions)
        reward, terms, tracking = e._reward(previous, prev_velocity)
        torch.testing.assert_close(terms['track_xy'], torch.ones(2))
        torch.testing.assert_close(terms['track_yaw'], torch.full((2,), .5))
        torch.testing.assert_close(terms['action_rate'], torch.full((2,), -.24))
        torch.testing.assert_close(terms['joint_acceleration'], torch.full((2,), -.0075))
        torch.testing.assert_close(terms['height'], torch.zeros(2))
        torch.testing.assert_close(tracking, torch.ones(2))
        torch.testing.assert_close(reward, (sum(terms.values()) * e.dt).clamp_min(0))
        e.qpos[:, 2] = 10
        self.assertTrue((e._reward(previous, prev_velocity)[0] == 0).all())

    def test_airtime_touchdown_and_partial_reset(self):
        e = self.make()
        e.commands[:, 0] = 1
        e.feet_air_time[:] = .6
        # Controlled floor-foot snapshot to test touchdown bookkeeping.
        from go1_warp.collisions import ContactSnapshot
        floor = e.model.geom('floor').id
        foot = e.model.geom('FR_foot').id
        snapshot = ContactSnapshot(e._collisions, torch.tensor([[floor, foot]]),
                                   torch.tensor([0]), torch.tensor([0.]), None)
        e.collision_snapshot = lambda **kwargs: snapshot
        # This fixture has no corresponding solver contact; isolate force query.
        e._collision_force_flags = lambda contacts, groups: torch.zeros((2, len(groups)), dtype=torch.bool)
        _, terms, _ = e._reward(e.actions.clone(), e.qvel[:, e.v_index].clone())
        self.assertAlmostEqual(terms['airtime'][0].item(), .8 * .12, places=6)
        self.assertEqual(terms['airtime'][1].item(), 0)
        self.assertEqual(e.feet_air_time[0, 0].item(), 0)
        other = e.feet_air_time[1].clone()
        e.reset([0])
        self.assertTrue((e.feet_air_time[0] == 0).all())
        torch.testing.assert_close(e.feet_air_time[1], other)

    def test_actual_contact_force_threshold(self):
        e = self.make()
        for i, d in enumerate(e.cpu_data):
            d.qpos[2] = 0. if i == 0 else 2.
            mujoco.mj_forward(e.model, d)
        contacts = e.collision_snapshot()
        flags = e._collision_force_flags(contacts, ['trunk'])
        self.assertTrue(flags[0, 0])
        self.assertFalse(flags[1, 0])
        e.cfg.collision_force_threshold = 1e20
        self.assertFalse(e._collision_force_flags(contacts, ['trunk']).any())

    def test_actions_remain_unbounded(self):
        e = self.make()
        action = torch.full((2, 12), 2.)
        e.step(action)
        torch.testing.assert_close(e.actions, action)


if __name__ == '__main__':
    unittest.main()
