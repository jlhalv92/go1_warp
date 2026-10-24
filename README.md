# Go1 MuJoCo Warp environment

A first flat-ground velocity-tracking environment built from your supplied XMLs.
Native `mujoco_warp` handles batched physics; PyTorch handles control, observations,
rewards, commands and episode bookkeeping. This is an environment, not a trained
walking policy or a complete PPO implementation.

## Run

Use Python 3.10+ and an NVIDIA GPU with a driver compatible with your installed
Warp and PyTorch CUDA builds. Install a CUDA-enabled PyTorch build using the
instructions on https://pytorch.org/get-started/locally/, then:

```bash
python -m pip install -r go1_warp/requirements.txt
python -m go1_warp.demo --num-envs 1024 --device cuda:0
```

Run commands from the directory containing `go1_warp/` after extracting the ZIP.
First execution compiles Warp kernels and can take time. Start with 16 worlds
before increasing the batch size; monitor contact/constraint capacity warnings.

CPU reference smoke test (ordinary MuJoCo, not Warp):

```bash
python -m go1_warp.demo --backend mujoco --device cpu --num-envs 4 --steps 100
python -m unittest discover -s go1_warp/tests -v
```

The demo uses zero actions to test the standing PD controller. It does not
produce a walking gait. `--render` opens the adapted custom GLFW viewer and
updates it during the rollout. It displays one selected world, not the whole batch.

```bash
python -m go1_warp.demo --num-envs 16 --device cuda:0 --render --world-id 0 --steps 10000
# CPU reference with the same viewer:
python -m go1_warp.demo --backend mujoco --device cpu --num-envs 4 --render
```

In your own control loop, call `env.render(world_id=0)` once per control step,
preferably before resetting completed worlds. It returns False when closed;
break out of the loop or continue headless, and always call `env.close()` in
`finally`. It runs on the main thread and needs a working desktop/OpenGL display.
SPACE pauses the calling loop (all worlds); TAB switches static/follow camera;
S/F change playback speed; H toggles overlays; C toggles contacts/constraints;
T toggles transparency; 0-5 toggle geom groups; ESC closes the viewer.

Default real-time pacing slows the full batch for visualization. Use
`env.render(world_id=0, throttle=False)` or demo `--unpaced` for unpaced
simulation with a display capped at 60 Hz. S/F affect paced mode only.
Snapshots are copied only on frames actually drawn, including while paused.
The fast path copies the selected world's qpos/qvel/ctrl to a reusable CPU
MjData and calls CPU `mj_forward` for visualization. This never steps the GPU.
Pressing C automatically enables full `mjw.get_data_into` readback, preserving
the backend's contact and solver fields. You can also force this using
`env.render(world_id=0, full_state=True)`. Full readback can transfer large batched
buffers internally and synchronizes GPU work, so use it for debugging rather
than throughput measurements. Solver/contact geometry reflects the last backend
forward evaluation (as in ordinary MuJoCo stepping).

To use an existing external viewer instance, pass a lazy snapshot callback:

```python
from go1_warp.viewer import MujocoViewer
viewer = MujocoViewer(env.model, env.dt)
try:
    # Inside your simulation loop, after env.step(actions):
    running = viewer.render(lambda: env.copy_render_data(
        world_id=0, full_state=viewer.show_contacts))
finally:
    viewer.close()
```

The viewer owns GLFW initialization/termination. If embedding alongside another
GLFW application, share that lifecycle explicitly. Structural model changes
require closing/recreating both the environment and its viewer.

## Interface

```python
import torch
from go1_warp import Config, Go1WarpEnv

env = Go1WarpEnv(Config(num_envs=1024, device='cuda:0'))
obs, info = env.reset(seed=42)
actions = torch.zeros((env.num_envs, 12), device=env.device)
next_obs, reward, terminated, truncated, info = env.step(actions)
# Save next_obs and evaluate next values BEFORE reset.
done_ids = torch.where(terminated | truncated)[0]
obs, info = env.reset(done_ids)
```

`reset(ids)` returns observations for the entire batch. `step()` never autoresets
and refuses to advance if any world still needs reset. Tensors returned by
`observations()` and rewards own their storage; `qpos`, `qvel`, `ctrl`, and other
environment fields are mutable live state. Do not edit those directly during
training. Commands are in `env.commands` and can be set manually (they are
resampled every five seconds and on reset).

| Quantity | Shape | Meaning |
|---|---|---|
| actions | (N, 12) | Joint position residuals, clipped to [-1,1] |
| observations | (N, 48) | Detailed order below |
| reward | (N,) | Sum of reward rates multiplied by control dt |
| terminated | (N,) bool | Low base height, excessive tilt, or nonfinite state |
| truncated | (N,) bool | Episode time limit |

For GAE use `~terminated` for the value bootstrap and
`~(terminated | truncated)` for recursion across transitions. Bootstrap timeouts
from the pre-reset observation. Both flags may be true on the same transition.
Nonfinite terminal observations must not be evaluated for bootstrap.

Observation slices:

| Slice | Quantity | Scale |
|---|---|---|
| 0:3 | Body-frame linear velocity | 2.0 |
| 3:6 | Body-frame angular velocity | 0.25 |
| 6:9 | Projected unit gravity | 1.0 |
| 9:12 | Desired body vx, vy, yaw rate | 1.0 |
| 12:24 | Joint position minus home pose | 1.0 |
| 24:36 | Joint velocity | 0.05 |
| 36:48 | Most recently applied action | 1.0 |

Joint order follows XML motors: FR hip/thigh/calf, FL hip/thigh/calf,
RR hip/thigh/calf, RL hip/thigh/calf. Nominal angles come from `home.qpos`:
`[-.1,-.9,1.8, .1,.9,-1.8, .1,-.9,1.8, -.1,.9,-1.8]`.
Do not replace this with a conventional Go1 sign convention.

## Control and task defaults

- Original XML timestep: 0.005 s (200 Hz); decimation 4 gives 50 Hz control.
- Actions specify `home + 0.25 * action` radians, clipped to joint limits.
- PD torques recomputed every physics step: kp=30, kd=0.8.
- XML motor torque limits retained: 23.7 Nm hips/thighs, 35.55 Nm calves.
- Torque actuators are preserved. The position-like `home.ctrl` values are
  deliberately ignored; reset initializes torque to zero.
- Commands: vx in [-1,1] m/s, vy in [-0.5,0.5] m/s, yaw in [-1,1] rad/s.
- Episodes: 20 s; failure if base z<0.16 m or projected gravity z>-0.5.
- Initial pose: supplied home pose, with +/-0.03 rad joint noise.
- Rewards: xy/yaw tracking, with vertical velocity, roll/pitch angular velocity,
  tilt, torque and action-change penalties. These are starting coefficients;
  locomotion learning and gait quality have not been validated.

## Meshes and model handling

The archive includes unchanged copies of the uploaded XMLs and all six uploaded
STL assets in `go1_warp/models/assets/`. The robot XML already declares
`meshdir="assets"` and references `trunk.stl`, `hip.stl`, `thigh_mirror.stl`,
`calf.stl`, and `thigh.stl`. `shell.stl` is included for completeness but is not
referenced by the supplied XML; no extra shell geom has been added.

Default `Config(collision_only=False)` now loads the visual meshes. No XML
changes are needed. Paths resolve relative to the model, so the asset directory
does not depend on the working directory.

For collision-only debugging use `Config(collision_only=True)` or the demo's
`--collision-only` flag. This removes visual mesh geoms and mesh declarations
in memory while retaining all explicit inertias, joints and collision geoms.
The viewer enables collision geom group 3 automatically when meshes are absent;
normal mesh rendering uses group 2 with collision group 3 initially hidden.

Original timestep, solver, contact settings and collision masks are retained.
No rough terrain, domain randomization beyond reset joint noise, contact
observations, privileged critic, CUDA graphs or Gymnasium
VectorEnv adapter is included yet. This class has a tensor API with separate
termination and truncation flags; it does not subclass Gymnasium VectorEnv.

## PPO training

See **TRAINING.md** for the new trainer, exact reference hyperparameters,
RSL-RL modifications, R/J/entropy logging, checkpoints, and playback.

```bash
python -m go1_warp.train --n-envs 16 --device cuda:0
```

This uses the working CUDA-enabled PyTorch and Warp builds already installed.
Do not reinstall requirements merely to add the trainer. Optional W&B and
TensorBoard dependencies are listed separately in requirements-training.txt.

## Validation and limitations

Validated with MuJoCo 3.16.0, MuJoCo Warp 3.16.0, Warp 1.18.0 and CPU PyTorch
2.14.1. See VALIDATION.md for the checks actually run. Requirements are pinned
to this simulator stack; install the appropriate PyTorch CUDA wheel separately.
GPU simulation and training throughput have not been executed here because
this machine has no NVIDIA GPU. No speedup or learned performance is claimed.

Torch and Warp share the current Torch CUDA stream for reset and stepping.
Keep calls on the same stream; cross-stream policy code must establish its own
dependencies. A full forward pass follows masked resets to rebuild batched
contact bookkeeping. Contact allocation defaults are conservative starting
values, not capacity guarantees.

Official references:
- https://mujoco.readthedocs.io/en/stable/mjwarp/index.html
- https://mujoco.readthedocs.io/en/stable/mjwarp/api.html
- https://nvidia.github.io/warp/latest/user_guide/interoperability/pytorch.html
