import contextlib
import io
import unittest

import numpy as np
import torch

from src.envs import EnvConfig, make_env
from src.envs.domain_randomized import DomainRandomizedGo1Env, DomainRandomizationConfig, restitution_solref


class DomainRandomizationTests(unittest.TestCase):
    def make(self, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            env = make_env(DomainRandomizationConfig(num_envs=3, device='cpu', backend='mujoco', **kwargs))
        self.addCleanup(env.close)
        return env

    def test_independent_reproducible_domains_and_parameters_reach_physics(self):
        env, same = self.make(), self.make()
        self.assertEqual(len({id(m) for m in env.cpu_models}), 3)
        for name, value in env.domain_parameters.items():
            torch.testing.assert_close(value, same.domain_parameters[name])
            low, high = getattr(env.cfg, f'dr_{name}_range')
            self.assertTrue(((value >= low) & (value <= high)).all())
        self.assertFalse(torch.equal(env.domain_parameters['com'][0], env.domain_parameters['com'][1]))
        for world, model in enumerate(env.cpu_models):
            p = env.domain_parameters
            self.assertAlmostEqual(model.body_mass[env.trunk_id], env._nominal_mass+p['payload_mass'][world].item(), places=5)
            np.testing.assert_allclose(model.body_ipos[env.trunk_id],
                                       env._nominal_com.numpy()+p['com'][world].numpy(), atol=1e-7)
            np.testing.assert_allclose(model.actuator_gainprm[:, 0], p['motor_strength'][world].numpy(), atol=1e-7)
            self.assertAlmostEqual(model.geom_friction[env.floor_id, 0], p['friction'][world].item())
            self.assertNotEqual(model.body_subtreemass[env.trunk_id], env.model.body_subtreemass[env.trunk_id])
        obs, rewards, term, trunc, info = env.step(torch.zeros(3,12))
        self.assertEqual(obs.shape, (3,48))
        self.assertTrue(torch.isfinite(obs).all())
        self.assertTrue(torch.isfinite(rewards).all())
        self.assertIn('collision', info['reward_terms']); self.assertIn('airtime', info['reward_terms'])

    def test_static_domains_survive_resets(self):
        env = self.make()
        before = {key: value.clone() for key,value in env.domain_parameters.items()}
        env.reset([0,2],seed=99)
        for key,value in before.items(): torch.testing.assert_close(env.domain_parameters[key],value)

    def test_optional_resampling_changes_only_selected_worlds(self):
        env = self.make(dr_resample_on_reset=True)
        env.step(torch.zeros(3,12))
        parameters = {key: value.clone() for key,value in env.domain_parameters.items()}
        poses = env.qpos.clone()
        times = [data.time for data in env.cpu_data]
        env.reset([1],seed=19)
        for key,value in parameters.items():
            torch.testing.assert_close(env.domain_parameters[key][[0,2]], value[[0,2]])
            self.assertFalse(torch.equal(env.domain_parameters[key][1],value[1]))
        torch.testing.assert_close(env.qpos[[0,2]],poses[[0,2]])
        self.assertEqual([env.cpu_data[i].time for i in (0,2)], [times[i] for i in (0,2)])
        first={key:value[1].clone() for key,value in env.domain_parameters.items()}
        env.reset([1],seed=19)
        for key,value in first.items(): torch.testing.assert_close(env.domain_parameters[key][1],value)

    def test_restitution_mapping_and_invalid_mass(self):
        solref=restitution_solref(torch.tensor([0.,.5,1.]),1000.)
        self.assertTrue((solref<0).all())
        self.assertAlmostEqual(solref[0,1].item(),-2*1000**.5,places=5)
        self.assertTrue((solref[1:,1]>solref[:-1,1]).all())
        with self.assertRaises(ValueError): self.make(dr_payload_mass_range=(-100.,0.))
        with self.assertRaises(ValueError): self.make(dr_friction_range=(2.,1.))
        # Explicit construction with a base config must still save the DR variant.
        config=EnvConfig(num_envs=1,device='cpu',backend='mujoco')
        with contextlib.redirect_stdout(io.StringIO()):
            env=DomainRandomizedGo1Env(config)
        self.addCleanup(env.close)
        self.assertEqual(env.cfg.environment,'domain_randomized')
        self.assertEqual(config.environment,'flat')

    def test_native_warp_batched_parameters_and_derived_constants(self):
        # Run real Warp kernels on CPU to validate MuJoCo Warp 3.16 APIs/layouts.
        import warp as wp
        import mujoco_warp as mjw
        wp.init()
        reference = self.make()
        env = self.make()
        with wp.ScopedDevice('cpu'):
            env._wp, env._mjw = wp, mjw
            env.warp_model = mjw.put_model(env.model, batch_sizes=env._model_batch_sizes())
            env.data = mjw.make_data(env.model,nworld=3,nconmax=128,njmax=512)
            env._initialize_domain()
            for world, model in enumerate(reference.cpu_models):
                for name in ('body_mass','body_inertia','body_ipos','body_subtreemass',
                             'body_invweight0','dof_invweight0','actuator_acc0'):
                    actual=wp.to_torch(getattr(env.warp_model,name))[world].numpy()
                    np.testing.assert_allclose(actual,getattr(model,name),atol=3e-5,rtol=3e-4,err_msg=name)
            pointers={name:view.data_ptr() for name,view in env._domain_model_views.items()}
            env._sample_domains(torch.tensor([1]))
            self.assertEqual(pointers,{name:view.data_ptr() for name,view in env._domain_model_views.items()})
        # Native harness used CPU Warp, so skip the base CUDA close path.
        env._wp=None
