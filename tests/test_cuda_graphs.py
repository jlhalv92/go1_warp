"""Run on a CUDA machine to validate captured/direct simulation parity."""
from dataclasses import replace
import unittest

import torch

from src.envs import make_env
from src.envs.domain_randomized import DomainRandomizationConfig


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA hardware is required')
class CudaGraphTests(unittest.TestCase):
    def check_variant(self, environment):
        config = DomainRandomizationConfig(environment=environment, num_envs=8,
                                            device='cuda:0', backend='warp', seed=23,
                                            dr_resample_on_reset=True)
        direct = make_env(replace(config, use_physics_graph=False))
        self.addCleanup(direct.close)
        graphed = make_env(replace(config, use_physics_graph=True))
        self.addCleanup(graphed.close)
        if graphed.physics_graph is None:
            self.skipTest('This CUDA driver / Warp memory-pool configuration does not support capture')
        self.assertIsNotNone(graphed.forward_graph)
        self.assertIsNotNone(graphed.reset_graph)
        generator = torch.Generator(device='cuda:0').manual_seed(91)
        for index in range(10):
            action = torch.rand((8,12),device='cuda:0',generator=generator)*.2-.1
            actual, expected = graphed.step(action), direct.step(action)
            for value, reference in zip(actual[:4],expected[:4]):
                torch.testing.assert_close(value,reference,atol=5e-5,rtol=5e-4)
            if index==4:
                # DR changes parameters in place here; captured addresses must
                # still refer to the randomized models after a partial reset.
                graphed.reset([1,5],seed=99)
                direct.reset([1,5],seed=99)
            ids=torch.where(actual[2]|actual[3])[0]
            if ids.numel():
                graphed.reset(ids)
                direct.reset(ids)

    def test_flat_graph_and_direct_paths(self):
        self.check_variant('flat')

    def test_domain_randomization_graph_and_direct_paths(self):
        self.check_variant('domain_randomized')
