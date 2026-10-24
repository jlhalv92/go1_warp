import unittest
import numpy as np
import torch
from go1_warp import Config, Go1WarpEnv
from go1_warp.env import rotate_inverse


class EnvironmentTests(unittest.TestCase):
    def make(self, **kwargs):
        return Go1WarpEnv(Config(num_envs=3, device='cpu', backend='mujoco', **kwargs))

    def test_xml_and_signed_joint_mapping(self):
        e = self.make()
        self.assertEqual((e.model.nq, e.model.nv, e.model.nu), (19, 18, 12))
        self.assertEqual(e.joint_names[:3], ['FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint'])
        torch.testing.assert_close(e.nominal, torch.tensor([-.1,-.9,1.8,.1,.9,-1.8,.1,-.9,1.8,-.1,.9,-1.8]))
        self.assertTrue(bool((e.ctrl == 0).all()))

    def test_step_and_torque_bounds(self):
        e = self.make()
        o, r, t, tr, _ = e.step(torch.ones(3, 12) * 10)
        self.assertEqual(tuple(o.shape), (3, 48))
        self.assertEqual(tuple(r.shape), (3,))
        self.assertTrue(bool(torch.isfinite(o).all()))
        self.assertTrue(bool((e.ctrl >= e.torque_limits[:, 0]).all()))
        self.assertTrue(bool((e.ctrl <= e.torque_limits[:, 1]).all()))
        np.testing.assert_allclose([d.time for d in e.cpu_data], e.dt)

    def test_partial_reset_isolation_and_seed(self):
        e = self.make()
        e.step(torch.zeros(3, 12))
        q = e.qpos.clone()
        v = e.qvel.clone()
        commands = e.commands.clone()
        e.reset([1], seed=33)
        torch.testing.assert_close(e.qpos[[0, 2]], q[[0, 2]])
        torch.testing.assert_close(e.qvel[[0, 2]], v[[0, 2]])
        torch.testing.assert_close(e.commands[[0, 2]], commands[[0, 2]])
        first = e.qpos[1].clone()
        e.reset([1], seed=33)
        torch.testing.assert_close(e.qpos[1], first)

    def test_timeout_and_explicit_reset(self):
        e = self.make(episode_seconds=.02)
        o, r, terminated, truncated, _ = e.step(torch.zeros(3, 12))
        self.assertTrue(bool(truncated.all()))
        self.assertFalse(bool(terminated.any()))
        final = o.clone()
        with self.assertRaises(RuntimeError):
            e.step(torch.zeros(3,12))
        e.reset(torch.where(terminated | truncated)[0])
        torch.testing.assert_close(final, o)
        self.assertTrue(bool((e.steps == 0).all()))

    def test_tilt_termination_and_rotation(self):
        # 90 degrees about z maps world +x to body -y.
        q = torch.tensor([[2**-.5, 0., 0., 2**-.5]])
        torch.testing.assert_close(rotate_inverse(q, torch.tensor([[1., 0., 0.]])),
                                   torch.tensor([[0., -1., 0.]]), atol=1e-6, rtol=0)
        e = self.make(reset_joint_noise=0.)
        # Use the same public state mutation as the GPU path, and synchronize the reference data.
        for i,d in enumerate(e.cpu_data):
            d.qpos[3:7] = [0.,1.,0.,0.]
            e.qpos[i, 3:7] = torch.tensor([0.,1.,0.,0.])
        _, _, terminated, _, _ = e.step(torch.zeros(3,12))
        self.assertTrue(bool(terminated.all()))


if __name__ == '__main__':
    unittest.main()
