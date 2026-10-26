"""Go1 velocity tracking with native MuJoCo/Warp physics and PyTorch rewards.

Collision-blind reward variant, selected with --environment blind.
MuJoCo handles physical collisions normally. This module contains no contact
queries, sensors, group reductions, collision penalties, or foot airtime state.

Warp uses separate CUDA graphs for step, forward, and masked reset. Capture
uses a persistent non-default Warp stream; replay uses the caller's Torch CUDA
stream. Reset masks are persistent, and PD is recomputed at
every physics substep. Raw actions are scaled; only motor torques are limited.

Defaults: 10 solver iterations, 50 line-search iterations, graphs requested,
Python tensor-value validation disabled. Numerical settings otherwise come
from the XML. In particular the integrator, gains, timestep and geometry stay
as supplied; the default Go1 XML uses dt=0.005 and decimation=4 (50 Hz control).

Configure the shared EnvConfig dataclass or train.py/play.py arguments:
--physics-graph / --no-physics-graph, --debug-checks,
--solver-iterations 100 --solver-ls-iterations 50.

step() returns terminal/pre-reset observations. Call reset(done_ids) before
stepping completed worlds again, even when debug checks are disabled.

For older checkpoint configuration dictionaries with retired fields, construct
Config.from_dict(saved_config) in play.py rather than Config(**saved_config).
CUDA performance and graph/direct numerical parity must be checked on the GPU.
"""
from contextlib import nullcontext
import math
import numpy as np
import torch
import mujoco
from .config import EnvConfig as Config
from .model import load_model
import gc
import warnings




def rotate_inverse(q, v):
    """World -> body; MuJoCo quaternion order is (w, x, y, z)."""
    xyz = q[:, 1:]
    t = 2.0 * torch.linalg.cross(xyz, v, dim=-1)
    return v - q[:, :1] * t + torch.linalg.cross(xyz, t, dim=-1)


class Go1WarpEnv:
    num_actions = 12
    num_observations = 48

    def __init__(self, cfg=None):
        self.cfg = cfg or Config()
        c = self.cfg
        if c.solver_iterations < 1 or c.solver_ls_iterations < 1:
            raise ValueError('Solver iteration limits must be positive')
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
        # Set host options before put_model so the Warp model sees these values.
        m.opt.iterations = c.solver_iterations
        m.opt.ls_iterations = c.solver_ls_iterations
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
                self.qfrc_actuator = wp.to_torch(self.data.qfrc_actuator)
                self._sim_time = wp.to_torch(self.data.time)
                self._reset_mask_wp = wp.zeros(c.num_envs, dtype=bool, device=str(self.device))
                self._reset_mask = wp.to_torch(self._reset_mask_wp)
        else:
            self.cpu_data = [mujoco.MjData(m) for _ in range(c.num_envs)]
            self.qpos = torch.zeros((c.num_envs, m.nq))
            self.qvel = torch.zeros((c.num_envs, m.nv))
            self.ctrl = torch.zeros((c.num_envs, m.nu))
            self.qfrc_actuator = torch.zeros((c.num_envs, m.nv))
        self.actions = torch.zeros((c.num_envs, 12), device=self.device)
        self.commands = torch.zeros((c.num_envs, 3), device=self.device)
        self.manual_commands = False  # Playback may supply commands instead of sampling.
        self.steps = torch.zeros(c.num_envs, dtype=torch.long, device=self.device)
        self.needs_reset = torch.zeros(c.num_envs, dtype=torch.bool, device=self.device)
        self._previous_actions = torch.empty_like(self.actions)
        self._previous_joint_velocity = torch.empty_like(self.actions)
        self._command_limits = self.commands.new_tensor([c.command_vx, c.command_vy, c.command_wz])
        self.tracking_sum = torch.zeros(c.num_envs, device=self.device)
        midpoint = self.joint_limits.mean(-1)
        half_range = (self.joint_limits[:, 1] - self.joint_limits[:, 0]) * .5 * c.soft_joint_position_limit
        self.soft_joint_limits = torch.stack((midpoint - half_range, midpoint + half_range), -1)

        self.viewer = None
        self._render_data = None

        self.physics_graph = None
        self.forward_graph = None
        self.reset_graph = None
        self._capture_stream = None
        self._init_physics_graph()

        # Restore the initial state after physics warmup.
        self.reset()
        print(
            f'[Go1WarpEnv] backend={c.backend}; '
            f'graphs(step/forward/reset)='
            f'{self.physics_graph is not None}/'
            f'{self.forward_graph is not None}/'
            f'{self.reset_graph is not None}; '
            f'solver_iterations={m.opt.iterations}; '
            f'ls_iterations={m.opt.ls_iterations}; debug_checks={c.debug_checks}'
        )

    @torch.no_grad()
    def _init_physics_graph(self):
        """Initialization only; the caller resets all worlds after warmup.

        Graphs retain array addresses. In-place writes remain visible, while
        replacing model/data arrays requires rebuilding graphs and Torch views.
        """
        if self._wp is None or not self.cfg.use_physics_graph:
            return
        wp = self._wp
        device = wp.get_device(str(self.device))
        driver = wp.get_cuda_driver_version()
        if driver is None or driver < (12, 4) or not wp.is_mempool_enabled(device):
            warnings.warn(
                'CUDA graphs unavailable: need CUDA driver support for 12.4+ '
                'and an enabled Warp memory pool. Using direct simulation calls.',
                RuntimeWarning,
            )
            return
        with wp.ScopedDevice(device), self._scope():
            # Compile each operation outside capture. No random numbers are
            # consumed, so graph and direct modes use the same reset RNG stream.
            self._reset_mask.fill_(True)
            self._mjw.reset_data(self.warp_model, self.data, reset=self._reset_mask_wp)
            self._reset_mask.fill_(False)
            self.qpos.copy_(self.home.expand_as(self.qpos))
            self.qvel.zero_()
            self.ctrl.zero_()
            self._mjw.forward(self.warp_model, self.data)
            self._mjw.step(self.warp_model, self.data)
            wp.synchronize_device(device)  # Startup only, not a training-step wait.

            # Torch's current stream can be the legacy/null stream (handle 0).
            # That stream is valid for replay, but cannot start CUDA capture.
            # Capture only Warp operations on this registered, owned stream.
            # Warmup's device synchronization has made input writes visible.
            self._capture_stream = wp.Stream(device)
            enabled = gc.isenabled()
            gc.disable()
            try:
                # No cross-stream events are needed: capture does not execute
                # these kernels. On exit, restore the caller's Warp/Torch stream.
                with wp.ScopedStream(self._capture_stream, sync_enter=False, sync_exit=False):
                    with wp.ScopedCapture(stream=self._capture_stream) as capture:
                        self._mjw.step(self.warp_model, self.data)
                    self.physics_graph = capture.graph
                    with wp.ScopedCapture(stream=self._capture_stream) as capture:
                        self._mjw.forward(self.warp_model, self.data)
                    self.forward_graph = capture.graph
                    with wp.ScopedCapture(stream=self._capture_stream) as capture:
                        self._mjw.reset_data(self.warp_model, self.data, reset=self._reset_mask_wp)
                    self.reset_graph = capture.graph
            finally:
                if enabled:
                    gc.enable()

    def _scope(self):
        # All Torch/Warp operations use the caller's current Torch CUDA stream.
        if self._wp is None:
            return nullcontext()
        return self._wp.ScopedStream(self._wp.stream_from_torch(torch.cuda.current_stream(self.device)))

    def _uniform(self, shape):
        return torch.rand(shape, device=self.device, generator=self.rng)

    def _sample_commands(self, ids):
        if self.manual_commands or ids.numel() == 0:
            return
        hi = self._command_limits
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
        if self.cfg.debug_checks and bool(((ids < 0) | (ids >= self.num_envs)).any()):
            raise IndexError('Environment index out of range')
        with self._scope():
            if self._wp is not None:
                self._reset_mask.fill_(False)
                self._reset_mask[ids] = True
                if self.reset_graph is not None:
                    self._wp.capture_launch(self.reset_graph)
                else:
                    self._mjw.reset_data(self.warp_model, self.data, reset=self._reset_mask_wp)
            self.qpos[ids] = self.home
            noise = (self._uniform((len(ids), 12)) * 2 - 1) * self.cfg.reset_joint_noise
            self.qpos[ids[:, None], self.q_index] = (self.nominal + noise).clamp(
                self.joint_limits[:, 0], self.joint_limits[:, 1])
            self.qvel[ids] = 0
            # home.ctrl is inconsistent with torque motors: reset torque to zero.
            self.ctrl[ids] = 0
            self.actions[ids] = 0
            self.tracking_sum[ids] = 0
            self.steps[ids] = 0
            self.needs_reset[ids] = False
            self._sample_commands(ids)
            if self._wp is not None:
                # Refresh derived simulation state after writing the reset pose.
                if self.forward_graph is not None:
                    self._wp.capture_launch(self.forward_graph)
                else:
                    self._mjw.forward(self.warp_model, self.data)
            else:
                for i in ids.tolist():
                    mujoco.mj_resetData(self.model, self.cpu_data[i])
                    self.cpu_data[i].qpos[:] = self.qpos[i].numpy()
                    mujoco.mj_forward(self.model, self.cpu_data[i])
                    self.qfrc_actuator[i].copy_(torch.from_numpy(self.cpu_data[i].qfrc_actuator))
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
        if self.cfg.debug_checks and bool(self.needs_reset.any()):
            raise RuntimeError('Reset completed worlds before calling step again')
        a = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        if a.shape != (self.num_envs, 12):
            raise ValueError(f'Expected actions with shape ({self.num_envs}, 12)')
        if self.cfg.debug_checks and not bool(torch.isfinite(a).all()):
            raise ValueError('Actions must be finite')
        self._previous_actions.copy_(self.actions)
        self._previous_joint_velocity.copy_(self.qvel[:, self.v_index])
        self.actions.copy_(a)
        target = self.nominal + self.cfg.action_scale * self.actions
        with self._scope():
            for _ in range(self.cfg.decimation):
                torque = (
                    self.cfg.kp * (target - self.qpos[:, self.q_index])
                    - self.cfg.kd * self.qvel[:, self.v_index]
                )
                self.ctrl.copy_(torque.clamp(self.torque_limits[:, 0], self.torque_limits[:, 1]))
                if self._wp is not None:
                    if self.physics_graph is not None:
                        self._wp.capture_launch(self.physics_graph)
                    else:
                        self._mjw.step(self.warp_model, self.data)

                else:
                    for i, d in enumerate(self.cpu_data):
                        d.ctrl[:] = self.ctrl[i].numpy()
                        mujoco.mj_step(self.model, d)
                        self.qpos[i] = torch.from_numpy(d.qpos.copy()).float()
                        self.qvel[i] = torch.from_numpy(d.qvel.copy()).float()
                        self.qfrc_actuator[i].copy_(torch.from_numpy(d.qfrc_actuator))
        self.steps += 1
        linear, angular, gravity = self.body_state()
        with self._scope():
            reward, terms, tracking = self._reward(
                self._previous_actions, self._previous_joint_velocity)
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

    def _reward(self, previous_actions, previous_joint_velocity):
        """Remaining reward rates; sum, multiply by dt, then clamp at zero."""
        linear, angular, _ = self.body_state()
        q, velocity = self.qpos[:, self.q_index], self.qvel[:, self.v_index]
        w, x, y, z = self.qpos[:, 3:7].unbind(-1)
        roll = torch.atan2(2 * (w*x + y*z), 1 - 2 * (x*x + y*y))
        pitch = torch.asin((2 * (w*y - z*x)).clamp(-1, 1))
        actuator_torque = self.qfrc_actuator[:, self.v_index]
        track_xy = torch.exp(-(linear[:, :2] - self.commands[:, :2]).square().sum(-1) / .25)
        track_yaw = torch.exp(-(angular[:, 2] - self.commands[:, 2]).square() / .25)
        terms = {
            'track_xy': track_xy,
            'track_yaw': .5 * track_yaw,
            'vertical': -2. * linear[:, 2].square(),
            'roll_pitch': -.05 * angular[:, :2].square().sum(-1),
            'tilt': -5. * (roll.square() + pitch.square()),
            'joint_limits': -10. * ((self.soft_joint_limits[:, 0] - q).clamp_min(0)
                                   + (q - self.soft_joint_limits[:, 1]).clamp_min(0)).sum(-1),
            'joint_acceleration': -2.5e-7 * ((velocity - previous_joint_velocity) / self.dt).square().sum(-1),
            'joint_velocity': 0. * velocity.square().sum(-1),
            'torque': -.0002 * actuator_torque.square().sum(-1),
            'action_rate': -.02 * (self.actions - previous_actions).square().sum(-1),
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
        full_state reads Warp solver results using get_data_into, which
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
                sim_time = self._sim_time[world_id].item()
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
            world_id, full_state=full_state))

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
        if self._wp is not None:
            self._wp.synchronize_device(str(self.device))
