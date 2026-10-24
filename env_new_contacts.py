"""Batched Go1 velocity tracking using native MuJoCo Warp and PyTorch.

Explicit reset semantics: step returns pre-reset observations; call reset(done_ids).
All environment tensors use the device selected in Config.

DIAGNOSTIC VARIANT: reward-side contact checking is DISABLED by default.
Physics collision detection/response remains enabled. No foot-contact queries,
force extraction, group reductions, airtime tracking/reward, or collision penalty
run in this mode. All other reward terms, PD control, observations, reset and
termination behavior are unchanged. Use the same fixed checkpoint for timings;
R/J are not comparable across modes because the reward definition changes.

Set GO1_WARP_DISABLE_CONTACT_CHECKS=0 to restore normal contact processing in
this file; =1 disables it. This overrides Config.disable_contact_checks.
GO1_WARP_FAST_CONTACTS still selects the fast/reference path when checks are ON.
Generic contact-query methods raise when checks are disabled.

"""
from contextlib import nullcontext
from dataclasses import dataclass
from functools import lru_cache
import os
import math
import numpy as np
import torch
import mujoco
from .model import load_model
from .collisions import CollisionDetector


@dataclass
class Config:
    num_envs: int = 1024
    device: str = 'cuda:0'
    backend: str = 'warp'  # 'mujoco' is a slow reference backend for debugging
    xml_path: str | None = None
    collision_only: bool = False
    decimation: int = 4  # XML dt=0.005 -> control dt=0.02 s
    episode_seconds: float = 20.0
    kp: float = 30.0
    kd: float = 0.8
    action_scale: float = 0.25
    nconmax: int = 128
    njmax: int = 512
    seed: int = 0
    reset_joint_noise: float = 0.03
    command_seconds: float = 5.0
    command_vx: float = 1.0
    command_vy: float = 0.5
    command_wz: float = 1.0
    nominal_height: float = 0.34
    soft_joint_position_limit: float = 0.9
    max_airtime: float = 0.5
    collision_force_threshold: float = 0.1
    fast_contacts: bool = True  # Warp reward queries; CPU backend keeps reference path
    disable_contact_checks: bool = True  # Diagnostic: also disables airtime reward


def rotate_inverse(q, v):
    """World -> body; MuJoCo quaternion order is (w, x, y, z)."""
    xyz = q[:, 1:]
    t = 2.0 * torch.linalg.cross(xyz, v, dim=-1)
    return v - q[:, :1] * t + torch.linalg.cross(xyz, t, dim=-1)


# These kernels are defined lazily so the CPU reference backend needs no Warp import.
@lru_cache(maxsize=1)
def _reward_contact_kernels():
    import warp as wp

    @wp.kernel
    def clear_contacts(
        nacon: wp.array[int], capacity: int,
        foot_hits: wp.array2d[int], group_force: wp.array3d[float],
        capacity_ok: wp.array[bool],
    ):
        world, group = wp.tid()
        if group < 4:
            foot_hits[world, group] = 0
        for axis in range(3):
            group_force[world, group, axis] = 0.0
        if world == 0 and group == 0:
            capacity_ok[0] = nacon[0] >= 0 and nacon[0] <= capacity

    @wp.kernel
    def reduce_contacts(
        nacon: wp.array[int], geom: wp.array[wp.vec2i], worldid: wp.array[int],
        wrench: wp.array[wp.spatial_vector],
        floor_mask: wp.array[int], foot_mask: wp.array2d[int],
        group_mask: wp.array2d[int], nworld: int, ngeom: int,
        foot_hits: wp.array2d[int], group_force: wp.array3d[float],
    ):
        contact = wp.tid()
        if contact >= nacon[0]:
            return
        pair = geom[contact]
        first, second = pair[0], pair[1]
        world = worldid[contact]
        if world < 0 or world >= nworld:
            return
        if first < 0 or second < 0 or first >= ngeom or second >= ngeom:
            return
        # Preserve snapshot(max_distance=None): any generated foot-floor contact.
        for foot in range(4):
            if ((floor_mask[first] != 0 and foot_mask[second, foot] != 0)
                    or (floor_mask[second] != 0 and foot_mask[first, foot] != 0)):
                wp.atomic_max(foot_hits, world, foot, 1)
        force = wrench[contact]
        for group in range(8):
            sign = group_mask[second, group] - group_mask[first, group]
            if sign != 0:
                for axis in range(3):
                    wp.atomic_add(group_force, world, group, axis, float(sign) * force[axis])

    @wp.kernel
    def finish_contacts(
        foot_hits: wp.array2d[int], group_force: wp.array3d[float], threshold: float,
        foot_contact: wp.array2d[bool], collision_flags: wp.array2d[bool],
    ):
        world, group = wp.tid()
        if group < 4:
            foot_contact[world, group] = foot_hits[world, group] != 0
        force = wp.vec3(group_force[world, group, 0], group_force[world, group, 1],
                        group_force[world, group, 2])
        collision_flags[world, group] = wp.length(force) > threshold

    return clear_contacts, reduce_contacts, finish_contacts


class _WarpRewardContacts:
    """Persistent Warp contact buffers for the exact existing Go1 reward queries.

    Uses public mujoco_warp.contact_force; no solver decoding is duplicated.
    Output Torch views are overwritten on the next update, so clone if retaining.
    Must run on the same stream as physics and Torch reward calculations.
    """
    def __init__(self, model, data, detector):
        import warp as wp
        self.wp = wp
        self.data = data
        self.nworld, self.ngeom = data.nworld, model.ngeom
        self.capacity = data.contact.geom.shape[0]
        self.kernels = _reward_contact_kernels()
        device = data.qpos.device
        legs = ('FR', 'FL', 'RR', 'RL')
        floor = np.zeros(model.ngeom, dtype=np.int32)
        feet = np.zeros((model.ngeom, 4), dtype=np.int32)
        groups = np.zeros((model.ngeom, 8), dtype=np.int32)
        floor[list(detector.groups['floor'])] = 1
        for index, leg in enumerate(legs):
            feet[list(detector.groups[f'{leg}_foot']), index] = 1
        names = [f'{leg}_{part}' for part in ('calf', 'thigh') for leg in legs]
        for index, name in enumerate(names):
            groups[list(detector.groups[name]), index] = 1
        self.floor = wp.array(floor, dtype=int, device=device)
        self.feet = wp.array(feet, dtype=int, device=device)
        self.groups = wp.array(groups, dtype=int, device=device)
        self.ids = wp.array(np.arange(self.capacity, dtype=np.int32), dtype=int, device=device)
        self.wrench = wp.empty(self.capacity, dtype=wp.spatial_vector, device=device)
        self.foot_hits = wp.empty((self.nworld, 4), dtype=int, device=device)
        self.group_force = wp.empty((self.nworld, 8, 3), dtype=float, device=device)
        self.foot_contact_wp = wp.empty((self.nworld, 4), dtype=bool, device=device)
        self.collision_flags_wp = wp.empty((self.nworld, 8), dtype=bool, device=device)
        self.capacity_ok_wp = wp.empty(1, dtype=bool, device=device)
        self.foot_contact = wp.to_torch(self.foot_contact_wp)
        self.collision_flags = wp.to_torch(self.collision_flags_wp)
        self.group_force_t = wp.to_torch(self.group_force)
        self.capacity_ok = wp.to_torch(self.capacity_ok_wp).view(())

    def update(self, model_wp, mjw, threshold):
        wp, data = self.wp, self.data
        clear, reduce, finish = self.kernels
        wp.launch(clear, dim=(self.nworld, 8), inputs=[
            data.nacon, self.capacity, self.foot_hits, self.group_force, self.capacity_ok_wp],
            device=data.qpos.device)
        # CUDA assertion checks the scalar on-device, without .item() or a host wait.
        # On CPU it raises synchronously. A failed CUDA assertion invalidates the
        # CUDA context; restart and raise capacity if that invariant ever fails.
        torch._assert_async(self.capacity_ok,
                            'Contact buffer overflow: increase nconmax/naconmax before querying collisions')
        # Fixed IDs cover the allocated capacity. MJWarp's public kernel skips
        # IDs >= nacon on-device; stale wrench slots are never read by reduce.
        mjw.contact_force(model_wp, data, self.ids, True, self.wrench)
        wp.launch(reduce, dim=self.capacity, inputs=[
            data.nacon, data.contact.geom, data.contact.worldid, self.wrench,
            self.floor, self.feet, self.groups, self.nworld, self.ngeom,
            self.foot_hits, self.group_force], device=data.qpos.device)
        wp.launch(finish, dim=(self.nworld, 8), inputs=[
            self.foot_hits, self.group_force, threshold,
            self.foot_contact_wp, self.collision_flags_wp], device=data.qpos.device)
        return self.foot_contact, self.collision_flags


class Go1WarpEnv:
    num_actions = 12
    num_observations = 48

    def __init__(self, cfg=None):
        self.cfg = cfg or Config()
        c = self.cfg
        disabled = os.environ.get('GO1_WARP_DISABLE_CONTACT_CHECKS')
        if disabled is not None:
            if disabled not in ('0', '1'):
                raise ValueError('GO1_WARP_DISABLE_CONTACT_CHECKS must be 0 or 1')
            c.disable_contact_checks = disabled == '1'
        # Override works with existing train/play/benchmark scripts and checkpoints.
        mode = os.environ.get('GO1_WARP_FAST_CONTACTS')
        if mode is not None:
            if mode not in ('0', '1'):
                raise ValueError('GO1_WARP_FAST_CONTACTS must be 0 or 1')
            c.fast_contacts = mode == '1'
        if c.num_envs < 1 or c.decimation < 1 or c.episode_seconds <= 0 or c.command_seconds <= 0:
            raise ValueError('num_envs, decimation and time intervals must be positive')
        if c.backend not in ('warp', 'mujoco'):
            raise ValueError('backend must be warp or mujoco')
        if c.kp <= 0 or c.kd < 0 or c.reset_joint_noise < 0:
            raise ValueError('Invalid PD gains or reset noise')
        if any(not math.isfinite(v) or v < 0 for v in (c.command_vx, c.command_vy, c.command_wz)):
            raise ValueError('Command limits must be finite and nonnegative')
        self.device = torch.device(c.device)
        if c.backend == 'warp' and self.device.type != 'cuda':
            raise ValueError('Use CUDA for Warp; use backend="mujoco", device="cpu" for reference tests')
        if c.backend == 'mujoco' and self.device.type != 'cpu':
            raise ValueError('Reference backend requires device="cpu"')
        self.num_envs = c.num_envs
        self.model = load_model(c.xml_path, c.collision_only)
        m = self.model
        if (m.nq, m.nv, m.nu) != (19, 18, 12):
            raise ValueError('Expected a floating-base Go1 with 12 motors')
        if not np.all(m.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT):
            raise ValueError('Expected joint motors')
        if not np.allclose(m.actuator_gear[:, 0], 1):
            raise ValueError('This controller expects unit motor gears')
        joints = m.actuator_trnid[:, 0]
        if len(set(joints.tolist())) != 12:
            raise ValueError('Motor joints must be unique')
        self.joint_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, int(j)) for j in joints]
        self.q_index = torch.tensor(m.jnt_qposadr[joints].copy(), device=self.device, dtype=torch.long)
        self.v_index = torch.tensor(m.jnt_dofadr[joints].copy(), device=self.device, dtype=torch.long)
        key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, 'home')
        if key < 0:
            raise ValueError('Missing home keyframe')
        self.home = torch.tensor(m.key_qpos[key].copy(), device=self.device, dtype=torch.float32)
        self.nominal = self.home[self.q_index]
        self.joint_limits = torch.tensor(m.jnt_range[joints].copy(), device=self.device, dtype=torch.float32)
        self.torque_limits = torch.tensor(m.actuator_ctrlrange.copy(), device=self.device, dtype=torch.float32)
        self.dt = float(m.opt.timestep) * c.decimation
        self.max_episode_steps = math.ceil(c.episode_seconds / self.dt)
        self.command_steps = max(1, round(c.command_seconds / self.dt))
        self.rng = torch.Generator(device=self.device).manual_seed(c.seed)
        self._wp = None
        if c.backend == 'warp':
            import warp as wp
            import mujoco_warp as mjw
            wp.init()
            self._wp, self._mjw = wp, mjw
            with wp.ScopedDevice(str(self.device)), self._scope():
                self.warp_model = mjw.put_model(m)
                self.data = mjw.make_data(m, nworld=c.num_envs, nconmax=c.nconmax, njmax=c.njmax)
                self.qpos = wp.to_torch(self.data.qpos)
                self.qvel = wp.to_torch(self.data.qvel)
                self.ctrl = wp.to_torch(self.data.ctrl)
        else:
            self.cpu_data = [mujoco.MjData(m) for _ in range(c.num_envs)]
            self.qpos = torch.zeros((c.num_envs, m.nq))
            self.qvel = torch.zeros((c.num_envs, m.nv))
            self.ctrl = torch.zeros((c.num_envs, m.nu))
        self.actions = torch.zeros((c.num_envs, 12), device=self.device)
        self.commands = torch.zeros((c.num_envs, 3), device=self.device)
        self.steps = torch.zeros(c.num_envs, dtype=torch.long, device=self.device)
        self.needs_reset = torch.zeros(c.num_envs, dtype=torch.bool, device=self.device)
        self._collisions = (None if c.disable_contact_checks else
                            CollisionDetector(m, self.num_envs, self.device))
        self.collision_groups = {} if self._collisions is None else self._collisions.groups
        self._contact_zero = torch.zeros(c.num_envs, device=self.device)
        self._reward_contacts = None
        if self._wp is not None and c.fast_contacts and not c.disable_contact_checks:
            with self._wp.ScopedDevice(str(self.device)), self._scope():
                self._reward_contacts = _WarpRewardContacts(m, self.data, self._collisions)
        self.feet_air_time = torch.zeros((c.num_envs, 4), device=self.device)
        self.last_foot_contact = torch.zeros_like(self.feet_air_time, dtype=torch.bool)
        self.tracking_sum = torch.zeros(c.num_envs, device=self.device)
        midpoint = self.joint_limits.mean(-1)
        half_range = (self.joint_limits[:, 1] - self.joint_limits[:, 0]) * .5 * c.soft_joint_position_limit
        self.soft_joint_limits = torch.stack((midpoint - half_range, midpoint + half_range), -1)
        self.viewer = None
        self._render_data = None
        self.reset()

    def _scope(self):
        # All Torch/Warp operations use the caller's current Torch CUDA stream.
        if self._wp is None:
            return nullcontext()
        return self._wp.ScopedStream(self._wp.stream_from_torch(torch.cuda.current_stream(self.device)))

    def _uniform(self, shape):
        return torch.rand(shape, device=self.device, generator=self.rng)

    def _sample_commands(self, ids):
        hi = self.commands.new_tensor([self.cfg.command_vx, self.cfg.command_vy, self.cfg.command_wz])
        lo = -hi
        self.commands[ids] = lo + (hi - lo) * self._uniform((len(ids), 3))

    @torch.no_grad()
    def reset(self, env_ids=None, seed=None):
        if seed is not None:
            self.rng.manual_seed(seed)
        ids = (torch.arange(self.num_envs, device=self.device) if env_ids is None
               else torch.as_tensor(env_ids, dtype=torch.long, device=self.device).flatten().unique())
        if ids.numel() == 0:
            return self.observations(), {}
        if bool(((ids < 0) | (ids >= self.num_envs)).any()):
            raise IndexError('Environment index out of range')
        with self._scope():
            if self._wp is not None:
                mask = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
                mask[ids] = 1
                self._mjw.reset_data(self.warp_model, self.data, self._wp.from_torch(mask))
            self.qpos[ids] = self.home
            noise = (self._uniform((len(ids), 12)) * 2 - 1) * self.cfg.reset_joint_noise
            self.qpos[ids[:, None], self.q_index] = (self.nominal + noise).clamp(
                self.joint_limits[:, 0], self.joint_limits[:, 1])
            self.qvel[ids] = 0
            # home.ctrl is inconsistent with torque motors: reset torque to zero.
            self.ctrl[ids] = 0
            self.actions[ids] = 0
            self.feet_air_time[ids] = 0
            self.last_foot_contact[ids] = False
            self.tracking_sum[ids] = 0
            self.steps[ids] = 0
            self.needs_reset[ids] = False
            self._sample_commands(ids)
            if self._wp is not None:
                # Rebuild packed contact/constraint buffers for all worlds after a masked reset.
                self._mjw.forward(self.warp_model, self.data)
            else:
                for i in ids.tolist():
                    mujoco.mj_resetData(self.model, self.cpu_data[i])
                    self.cpu_data[i].qpos[:] = self.qpos[i].numpy()
                    mujoco.mj_forward(self.model, self.cpu_data[i])
        return self.observations(), {}

    def body_state(self):
        quat = self.qpos[:, 3:7]
        # Free-joint linear velocity is world-frame; angular velocity is local-frame.
        linear = rotate_inverse(quat, self.qvel[:, :3])
        angular = self.qvel[:, 3:6]
        gravity = rotate_inverse(quat, self.qvel.new_tensor([0., 0., -1.]).expand(self.num_envs, -1))
        return linear, angular, gravity

    def observations(self):
        linear, angular, gravity = self.body_state()
        # 3 + 3 + 3 + 3 + 12 + 12 + 12 = 48; outputs have independent storage.
        return torch.cat((linear * 2., angular * .25, gravity, self.commands,
                          self.qpos[:, self.q_index] - self.nominal,
                          self.qvel[:, self.v_index] * .05, self.actions), dim=-1)

    @torch.no_grad()
    def step(self, actions):
        if bool(self.needs_reset.any()):
            raise RuntimeError('Reset completed worlds before calling step again')
        a = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        if a.shape != (self.num_envs, 12) or not bool(torch.isfinite(a).all()):
            raise ValueError(f'Expected finite actions with shape ({self.num_envs}, 12)')
        previous = self.actions.clone()
        previous_joint_velocity = self.qvel[:, self.v_index].clone()
        self.actions.copy_(a)
        target = self.nominal + self.cfg.action_scale * self.actions
        with self._scope():
            for _ in range(self.cfg.decimation):
                torque = self.cfg.kp * (target - self.qpos[:, self.q_index]) - self.cfg.kd * self.qvel[:, self.v_index]
                self.ctrl.copy_(torque.clamp(self.torque_limits[:, 0], self.torque_limits[:, 1]))
                if self._wp is not None:
                    self._mjw.step(self.warp_model, self.data)
                else:
                    for i, d in enumerate(self.cpu_data):
                        d.ctrl[:] = self.ctrl[i].numpy()
                        mujoco.mj_step(self.model, d)
                        self.qpos[i] = torch.from_numpy(d.qpos.copy()).float()
                        self.qvel[i] = torch.from_numpy(d.qvel.copy()).float()
        self.steps += 1
        linear, angular, gravity = self.body_state()
        with self._scope():
            reward, terms, tracking = self._reward(previous, previous_joint_velocity)
        finite = torch.isfinite(self.qpos).all(-1) & torch.isfinite(self.qvel).all(-1)
        terminated = (~finite) | (self.qpos[:, 2] < .16) | (gravity[:, 2] > -.5)
        truncated = self.steps >= self.max_episode_steps
        self.needs_reset.copy_(terminated | truncated)
        # Reward uses the command just executed; next obs gets any new command.
        ids = torch.where((self.steps % self.command_steps == 0) & ~self.needs_reset)[0]
        self._sample_commands(ids)
        obs = self.observations()
        self.tracking_sum += tracking
        done = terminated | truncated
        info = {
            'reward_terms': terms,  # Weighted reward rates; total multiplies by dt then clamps.
            'tracking_performance': tracking,
            'episode_tracking_performance': torch.where(
                done, self.tracking_sum / self.steps.clamp_min(1), torch.nan),
        }
        return obs, torch.nan_to_num(reward), terminated, truncated, info

    def collision_snapshot(self, *, max_distance=None):
        if self.cfg.disable_contact_checks:
            raise RuntimeError('Contact queries are disabled in this diagnostic configuration')
        with self._scope():
            source = self.data if self._wp is not None else self.cpu_data
            return self._collisions.snapshot(source, max_distance=max_distance)

    def check_collision(self, group1, group2, *, max_distance=None):
        return self.collision_snapshot(max_distance=max_distance).check_collision(group1, group2)

    def check_any_collision(self, groups, *, max_distance=None):
        return self.collision_snapshot(max_distance=max_distance).check_any_collision(groups)

    def _collision_force_flags(self, contacts, groups):
        """Threshold signed net world force per geom group, excluding internal contacts.

        Feet sharing a calf body do not enter the calf geom group. Returns (N, G).
        Call in _scope before physics advances; snapshot and forces must correspond.
        """
        if self.cfg.disable_contact_checks:
            raise RuntimeError('Contact queries are disabled in this diagnostic configuration')
        count = contacts.geom.shape[0]
        force = torch.zeros((count, 3), device=self.device)
        if count and self._wp is not None:
            wp = self._wp
            ids = torch.arange(count, dtype=torch.int32, device=self.device)
            wrench = wp.empty(count, dtype=wp.spatial_vector, device=str(self.device))
            self._mjw.contact_force(self.warp_model, self.data, wp.from_torch(ids), True, wrench)
            force = wp.to_torch(wrench)[:, :3]
        elif count:
            rows = []
            for d in self.cpu_data:
                for index in range(d.ncon):
                    wrench = np.zeros(6)
                    mujoco.mj_contactForce(self.model, d, index, wrench)
                    rows.append(d.contact[index].frame.reshape(3, 3).T @ wrench[:3])
            force = torch.as_tensor(np.asarray(rows), dtype=torch.float32, device=self.device)
        flags = []
        for group in groups:
            mask = self._collisions.group_mask(group)
            sign = mask[contacts.g2].float() - mask[contacts.g1].float()
            signed = force * (sign * contacts.valid)[:, None]
            net = torch.zeros((self.num_envs, 3), device=self.device)
            net.index_add_(0, contacts.world, signed)
            flags.append(net.norm(dim=-1) > self.cfg.collision_force_threshold)
        return torch.stack(flags, dim=-1)

    def _reward(self, previous_actions, previous_joint_velocity):
        """Batched port of the supplied Reward class; coefficients are reward rates."""
        linear, angular, _ = self.body_state()
        q, velocity = self.qpos[:, self.q_index], self.qvel[:, self.v_index]
        w, x, y, z = self.qpos[:, 3:7].unbind(-1)
        roll = torch.atan2(2 * (w*x + y*z), 1 - 2 * (x*x + y*y))
        pitch = torch.asin((2 * (w*y - z*x)).clamp(-1, 1))
        if self.cfg.disable_contact_checks:
            # No contact state is read or processed in this branch.
            airtime_reward = self._contact_zero
            collision_penalty = self._contact_zero
        else:
            if self._reward_contacts is not None:
                foot_contact, collision_flags = self._reward_contacts.update(
                    self.warp_model, self._mjw, self.cfg.collision_force_threshold)
            else:
                contacts = self.collision_snapshot()
                legs = ('FR', 'FL', 'RR', 'RL')
                foot_contact = contacts.check_pairs([('floor', f'{leg}_foot') for leg in legs])
                penalized = [f'{leg}_{part}' for part in ('calf', 'thigh') for leg in legs]
                collision_flags = self._collision_force_flags(contacts, penalized)
            filtered_contact = foot_contact | self.last_foot_contact
            first_contact = (self.feet_air_time > 0) & filtered_contact
            self.feet_air_time += self.dt
            airtime = ((self.feet_air_time - self.cfg.max_airtime) * first_contact).sum(-1)
            self.feet_air_time *= ~foot_contact
            self.last_foot_contact.copy_(foot_contact)
            collisions = collision_flags.sum(-1)
            airtime_reward = .8 * airtime * (self.commands[:, :2].norm(dim=-1) > .1)
            collision_penalty = -collisions.float()
        if self._wp is not None:
            actuator_torque = self._wp.to_torch(self.data.qfrc_actuator)[:, self.v_index]
        else:
            actuator_torque = torch.as_tensor(np.stack([d.qfrc_actuator for d in self.cpu_data]),
                                             dtype=torch.float32)[:, self.v_index]
        track_xy = torch.exp(-(linear[:, :2] - self.commands[:, :2]).square().sum(-1) / .25)
        track_yaw = torch.exp(-(angular[:, 2] - self.commands[:, 2]).square() / .25)
        terms = {
            'track_xy': track_xy,
            'track_yaw': .5 * track_yaw,
            'airtime': airtime_reward,
            'vertical': -2. * linear[:, 2].square(),
            'roll_pitch': -.05 * angular[:, :2].square().sum(-1),
            'tilt': -5. * (roll.square() + pitch.square()),
            'joint_limits': -10. * ((self.soft_joint_limits[:, 0] - q).clamp_min(0)
                                   + (q - self.soft_joint_limits[:, 1]).clamp_min(0)).sum(-1),
            'joint_acceleration': -2.5e-7 * ((velocity - previous_joint_velocity) / self.dt).square().sum(-1),
            'joint_velocity': 0. * velocity.square().sum(-1),
            'torque': -.0002 * actuator_torque.square().sum(-1),
            'action_rate': -.02 * (self.actions - previous_actions).square().sum(-1),
            'collision': collision_penalty,
            'height': -30. * (self.qpos[:, 2] - self.cfg.nominal_height).square(),
            # Original uses -0.5 * 0.02; express as rate for arbitrary control dt.
            'stand_still': -.5 * (q - self.nominal).abs().sum(-1) * (self.commands.norm(dim=-1) < .1),
        }
        tracking = (1 - (torch.cat((linear[:, :2], angular[:, 2:3]), -1)
                         - self.commands).abs()).mean(-1).clamp_min(0)
        return (sum(terms.values()) * self.dt).clamp_min(0), terms, tracking

    def copy_render_data(self, world_id=0, full_state=False):
        """Reusable host snapshot; never advance or modify simulation state.

        Fast path copies one world's state and runs CPU forward for display.
        full_state reads Warp solver/contact results using get_data_into, which
        may transfer batched buffers internally and is much more expensive.
        The returned MjData is overwritten by the next snapshot call.
        """
        if not 0 <= world_id < self.num_envs:
            raise IndexError(world_id)
        if self._render_data is None:
            self._render_data = mujoco.MjData(self.model)
        d = self._render_data
        if self._wp is None:
            mujoco.mj_copyData(d, self.model, self.cpu_data[world_id])
            if not full_state:
                mujoco.mj_forward(self.model, d)
        elif full_state:
            with self._scope():
                self._wp.synchronize_device(str(self.device))
                self._mjw.get_data_into(d, self.model, self.data, world_id=world_id)
        else:
            with self._scope():
                # Pack only the chosen world. Tensor.cpu() completes its D2H copy.
                state = torch.cat((self.qpos[world_id], self.qvel[world_id],
                                   self.ctrl[world_id])).cpu().numpy()
                sim_time = self._wp.to_torch(self.data.time)[world_id].item()
            mujoco.mj_resetData(self.model, d)
            nq, nv = self.model.nq, self.model.nv
            d.qpos[:] = state[:nq]
            d.qvel[:] = state[nq:nq + nv]
            d.ctrl[:] = state[nq + nv:]
            d.time = sim_time
            # Display-only CPU reconstruction; does not step GPU physics.
            mujoco.mj_forward(self.model, d)
        return d

    def render(self, world_id=0, throttle=True, full_state=False):
        """Update custom GLFW viewer; return False when its window is closed.

        Call on the main thread once per control step. SPACE pauses this loop.
        C enables actual backend contact/constraint readback automatically.
        Set throttle=False to leave simulation unpaced and display at <=60 Hz.
        """
        if not 0 <= world_id < self.num_envs:
            raise IndexError(world_id)
        if self.viewer is None:
            from .viewer import MujocoViewer
            self.viewer = MujocoViewer(self.model, self.dt, throttle=throttle)
        viewer = self.viewer
        viewer.throttle = throttle
        viewer.world_id = world_id
        return viewer.render(lambda: self.copy_render_data(
            world_id, full_state=full_state or viewer.show_contacts))

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
        if self._wp is not None:
            self._wp.synchronize_device(str(self.device))