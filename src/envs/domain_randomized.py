"""Per-world dynamics randomization inspired by Margolis et al., Table I.

https://arxiv.org/abs/2205.02824 (Rapid Locomotion via Reinforcement Learning).
This is the domain-randomized policy setting, not teacher/student adaptation
or the paper's velocity curriculum. The existing 48-D observation is retained.

Domains are sampled once per world at initialization by default. Optional
per-reset resampling recomputes model constants and is more expensive on Warp.
All model arrays keep their addresses after initialization, preserving graphs.
MuJoCo has compliant contacts rather than a restitution coefficient: the
sampled restitution is approximated by a spring/damper damping ratio.
"""
import copy
from dataclasses import dataclass, replace
import math

import mujoco
import numpy as np
import torch

from .config import EnvConfig
from .go1 import Go1WarpEnv


@dataclass
class DomainRandomizationConfig(EnvConfig):
    environment: str = 'domain_randomized'
    dr_friction_range: tuple[float, float] = (.05, 4.)
    dr_restitution_range: tuple[float, float] = (0., 1.)
    dr_payload_mass_range: tuple[float, float] = (-1., 3.)
    dr_com_range: tuple[float, float] = (-.1, .1)
    dr_motor_strength_range: tuple[float, float] = (.9, 1.1)
    dr_contact_stiffness: float = 1000.
    dr_resample_on_reset: bool = False

    def validate_randomization(self):
        for name in ('friction', 'restitution', 'payload_mass', 'com', 'motor_strength'):
            limits = getattr(self, f'dr_{name}_range')
            if len(limits) != 2 or not all(math.isfinite(x) for x in limits) or limits[0] > limits[1]:
                raise ValueError(f'Invalid dr_{name}_range: {limits}')
        if self.dr_friction_range[0] <= 0 or self.dr_motor_strength_range[0] <= 0:
            raise ValueError('Friction and motor strength must be positive')
        if self.dr_restitution_range[0] < 0 or self.dr_restitution_range[1] > 1:
            raise ValueError('Restitution must be in [0,1]')
        if not math.isfinite(self.dr_contact_stiffness) or self.dr_contact_stiffness <= 0:
            raise ValueError('dr_contact_stiffness must be finite and positive')


def restitution_solref(restitution, stiffness):
    """Approximate restitution using a direct-format contact spring and damper.

    zeta = -log(e) / sqrt(pi^2 + log(e)^2). e=0 is critically damped;
    e=1 is capped slightly below one to retain a positive damping coefficient.
    MuJoCo's impedance/solver/discretization mean actual rebound can differ.
    """
    log_e = torch.log(restitution.clamp(1e-6, 1. - 1e-6))
    zeta = -log_e / torch.sqrt(math.pi ** 2 + log_e.square())
    zeta = torch.where(restitution <= 0, torch.ones_like(zeta), zeta)
    return torch.stack((torch.full_like(restitution, -stiffness),
                        -2. * math.sqrt(stiffness) * zeta), dim=-1)


class DomainRandomizedGo1Env(Go1WarpEnv):
    def __init__(self, cfg=None):
        if cfg is None:
            cfg = DomainRandomizationConfig()
        elif not isinstance(cfg, DomainRandomizationConfig):
            from dataclasses import asdict
            cfg = DomainRandomizationConfig.from_dict(asdict(cfg))
        cfg = replace(cfg, environment='domain_randomized')
        cfg.validate_randomization()
        super().__init__(cfg)

    def _configure_model(self):
        # Ground friction/solref must win contact mixing, including custom XMLs.
        floor = self.model.geom('floor').id
        self.model.geom_priority[floor] = int(self.model.geom_priority.max()) + 1

    def _model_batch_sizes(self):
        # Include dependent constants, not only primary randomized parameters.
        names = ('body_mass', 'body_inertia', 'body_ipos', 'body_subtreemass',
                 'body_invweight0', 'dof_invweight0', 'actuator_acc0',
                 'geom_friction', 'geom_solref', 'actuator_gainprm',
                 'cam_pos0', 'cam_poscom0', 'cam_mat0',
                 'light_pos0', 'light_poscom0', 'light_dir0')
        return dict.fromkeys(names, self.num_envs)

    def _initialize_domain(self):
        self.trunk_id = self.model.body('trunk').id
        self.floor_id = self.model.geom('floor').id
        if self.model.body_mass[self.trunk_id] + self.cfg.dr_payload_mass_range[0] <= 0:
            raise ValueError('Payload range would produce a nonpositive trunk mass')
        if not np.all(self.model.actuator_gaintype == mujoco.mjtGain.mjGAIN_FIXED):
            raise ValueError('Motor-strength randomization requires fixed-gain torque motors')
        self.domain_rng = torch.Generator(device=self.device).manual_seed(self.cfg.seed + 104729)
        self.domain_parameters = {
            'friction': torch.empty(self.num_envs, device=self.device),
            'restitution': torch.empty(self.num_envs, device=self.device),
            'payload_mass': torch.empty(self.num_envs, device=self.device),
            'com': torch.empty((self.num_envs, 3), device=self.device),
            'motor_strength': torch.empty((self.num_envs, self.num_actions), device=self.device),
        }
        self._nominal_mass = float(self.model.body_mass[self.trunk_id])
        self._nominal_inertia = torch.tensor(self.model.body_inertia[self.trunk_id].copy(),
                                             dtype=torch.float32, device=self.device)
        self._nominal_com = torch.tensor(self.model.body_ipos[self.trunk_id].copy(),
                                        dtype=torch.float32, device=self.device)
        self._nominal_gain = torch.tensor(self.model.actuator_gainprm[:, 0].copy(),
                                         dtype=torch.float32, device=self.device)
        if self._wp is None:
            # Every CPU world owns a distinct model. Never mutate the shared model.
            self.cpu_models = [copy.copy(self.model) for _ in range(self.num_envs)]
        else:
            wp = self._wp
            self._domain_model_views = {name: wp.to_torch(getattr(self.warp_model, name))
                                        for name in ('body_mass', 'body_inertia', 'body_ipos',
                                                     'geom_friction', 'geom_solref', 'actuator_gainprm')}
            # put_model's batch_sizes covers top-level fields; this nested field
            # must also be per-world before set_const and graph capture.
            self.warp_model.stat.meaninertia = wp.full(self.num_envs, float(self.model.stat.meaninertia),
                                                      dtype=float, device=str(self.device))
        self._sample_domains(torch.arange(self.num_envs, device=self.device))
        # The base constructor performs its first reset immediately after warmup.
        self._first_domain_reset = True

    def _sample_domains(self, ids):
        if ids.numel() == 0:
            return
        for name, values in self.domain_parameters.items():
            low, high = getattr(self.cfg, f'dr_{name}_range')
            values[ids] = low + (high - low) * torch.rand(
                (len(ids), *values.shape[1:]), generator=self.domain_rng, device=self.device)
        params = self.domain_parameters
        mass = self._nominal_mass + params['payload_mass'][ids]
        # Scale trunk inertia with mass at fixed shape, about the new COM.
        inertia = self._nominal_inertia * (mass / self._nominal_mass)[:, None]
        com = self._nominal_com + params['com'][ids]
        gain = self._nominal_gain * params['motor_strength'][ids]
        solref = restitution_solref(params['restitution'][ids], self.cfg.dr_contact_stiffness)
        if self._wp is not None:
            views = self._domain_model_views
            views['body_mass'][ids, self.trunk_id] = mass
            views['body_inertia'][ids, self.trunk_id] = inertia
            views['body_ipos'][ids, self.trunk_id] = com
            # Higher-priority floor parameters determine floor/robot contacts.
            # Robot/robot contact materials retain their XML values.
            views['geom_friction'][ids, self.floor_id, 0] = params['friction'][ids]
            views['geom_solref'][ids, self.floor_id] = solref
            views['actuator_gainprm'][ids, :, 0] = gain
            self._mjw.set_const(self.warp_model, self.data)
        else:
            for row, world in enumerate(ids.tolist()):
                model = self.cpu_models[world]
                model.body_mass[self.trunk_id] = mass[row].item()
                model.body_inertia[self.trunk_id] = inertia[row].numpy()
                model.body_ipos[self.trunk_id] = com[row].numpy()
                model.geom_friction[self.floor_id, 0] = params['friction'][world].item()
                model.geom_solref[self.floor_id] = solref[row].numpy()
                model.actuator_gainprm[:, 0] = gain[row].numpy()
                mujoco.mj_setConst(model, self.cpu_data[world])

    def _before_reset(self, ids, seed):
        if self._first_domain_reset:
            self._first_domain_reset = False
            return
        if self.cfg.dr_resample_on_reset:
            if seed is not None:
                self.domain_rng.manual_seed(seed + 104729)
            self._sample_domains(ids)

    def _cpu_model(self, world_id):
        return self.cpu_models[world_id] if self._wp is None else self.model
