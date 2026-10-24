# Go1 MuJoCo Warp locomotion

Batched Unitree Go1 velocity tracking with MuJoCo Warp physics, PyTorch control
and rewards, and continuous PPO. Training and playback select their environment
through the same factory. The default `flat` environment includes collision
penalties and foot airtime rewards; `blind` retains physical collisions but
omits contact queries and contact rewards.

## Install

Use Python 3.10+ (the development environment uses Python 3.12). Install a
CUDA-enabled PyTorch build compatible with your NVIDIA driver, then run from
the repository root:

```bash
python -m pip install -r requirements.txt
# Optional W&B / TensorBoard integrations:
python -m pip install -r requirements-training.txt
```

The working simulator versions are MuJoCo/MuJoCo Warp 3.16.0 and Warp 1.18.0.
Training requires an NVIDIA GPU for the `warp` backend. A slow CPU reference
backend is available with `--backend mujoco --device cpu`.

## Structure

```text
train.py                         training entry point
play.py                          visualization / fixed-step evaluation entry point
benchmark_eval.py                evaluation timing across environment counts
demo.py                          zero-action simulation smoke test
src/
    models/                      XML files and assets/ meshes
    envs/
        config.py                EnvConfig dataclass
        go1.py                   fast contact-enabled flat environment
        go1_blind.py             contact-blind reward variant
        domain_randomized.py     DomainRandomizationConfig and randomized environment
        collisions.py            geom-group contact helpers
        model.py                 model loading
        viewer.py                custom GLFW viewer
    algorithms/
        config.py                TrainConfig, PlayConfig and argparse integration
        trainer.py               rollout collection, updates, logging and checkpoints
        ppo.py                   PPO, GAE and rollout storage
        evaluation.py            fixed-window evaluation
        playback.py              checkpoint playback
        paths.py                 repository-relative output paths
    networks/
        actor_critic.py          SiLU / LayerNorm actor, critic and Gaussian policy
runs/                            checkpoints and resolved configuration (generated)
logs/                            CSV, TensorBoard and W&B logs (generated)
tests/
docs/
```

XML includes and mesh paths remain relative to `src/models/`. Default asset,
run and logging paths are resolved from the repository location, independently
of the shell's current working directory. Relative custom XML/checkpoint/output
paths also resolve from the repository root; absolute paths are accepted.

## Train

```bash
python train.py \
    --experiment-name LLC_PPO --run-id 0 \
    --n-envs 4096 --n-epochs 150 \
    --n-steps-per-fit 24 --n-steps 240 \
    --n-epochs-policy 5 --n-minibatches 4 \
    --eval-envs 10 --eval-every 5 \
    --device cuda:0 \
    --solver-iterations 100 --solver-ls-iterations 50
```

These outputs have stable, user-selected names (no timestamp or implicit seed
suffix):

```text
runs/LLC_PPO/run_0/config.json
runs/LLC_PPO/run_0/best.pt
runs/LLC_PPO/run_0/last.pt
logs/LLC_PPO/run_0/train.csv
logs/LLC_PPO/run_0/evaluation.csv
logs/LLC_PPO/run_0/runtime.csv
```

`--experiment-name` defaults to `LLC_PPO`; `--run-id` defaults to `0`.
Choose another run ID for a new run. Reusing an existing output location requires
`--resume`, which appends CSV rows rather than overwriting earlier logs.
`--output-dir` and `--log-dir` change the two parent directories.

`last.pt` is updated after each completed epoch. `best.pt` is updated when the
fixed-window evaluation return improves, including the initial evaluation.
Numbered epoch checkpoints are optional:

```bash
# Save epoch_0001.pt, epoch_0002.pt, ... in addition to best/last:
python train.py --experiment-name LLC_PPO --run-id 1 --save-epochs --save-every 1
# Save every tenth epoch (10 is the default numbered-checkpoint interval):
python train.py --experiment-name LLC_PPO --run-id 2 --save-epochs
# Only best and last:
python train.py --experiment-name LLC_PPO --run-id 3 --no-save-epochs
```

For compatibility, explicitly supplying `--save-every N` also enables numbered
checkpoints when N > 0; `--save-every 0` disables them. An explicit
`--no-save-epochs` takes precedence.

Resume to a larger **total** epoch count:

```bash
python train.py --resume runs/LLC_PPO/run_0/last.pt --n-epochs 200
```

Policy/optimizer and training counters resume; simulation episodes restart.
Configuration precedence is dataclass defaults, then saved configuration, then
explicit CLI arguments. Architecture width and gamma must match on resume.

## Dataclasses and command-line configuration

Edit defaults in `EnvConfig`, `DomainRandomizationConfig`, `TrainConfig` or
`PlayConfig`, or override them through argparse:

```bash
python train.py --help
python play.py --help
python train.py --experiment-name graph_test --run-id 0 \
    --no-physics-graph --no-fast-contacts --debug-checks \
    --solver-iterations 100 --solver-ls-iterations 50
```

`GO1_WARP_*` environment-variable overrides have been removed. Boolean settings
have positive/negative flags; physics graphs and fast contact rewards are enabled
by default. Debug checks are disabled by default because Python tensor-value
checks synchronize the GPU. Legacy aliases such as `--n-envs`, `--name-exp`,
`--target-vx` and underscore spellings remain accepted. The canonical dataclass
fields are `num_envs`, `experiment_name` and `command_vx/command_vy/command_wz`.
The original PPO defaults are retained; the 4096-world command above explicitly
overrides them with your working training configuration.

Use `--log` for W&B (`--wandb-mode offline` is supported) and `--tensorboard` for
TensorBoard. Both integrations write under the selected `logs/<experiment>/run_<id>`.
`--profile-runtime` and `--profile-collection` retain the existing diagnostic
measurements and their synchronization overhead.

## Play / evaluate

```bash
python play.py runs/LLC_PPO/run_0/best.pt --num-envs 1
python play.py runs/LLC_PPO/run_0/best.pt --num-envs 10 --headless --steps 1000
python play.py runs/LLC_PPO/run_0/best.pt --unpaced --world-id 0
```

Saved physics, environment variant and randomization settings are inherited.
Explicit CLI options override them. Playback has its own runtime defaults for
world count (1), seed (10000), device (`cuda:0`) and backend (`warp`). Use
`--backend mujoco --device cpu` for CPU playback. Old dictionary/state-dict
checkpoints retain their network keys and load through the compatibility filter;
old checkpoints without an environment selector default to `flat`. Select
`--environment blind` explicitly for a policy trained with the blind variant.

Headless evaluation always runs exactly `steps * num_envs` transitions; there
are no episode quotas. `R` is mean undiscounted **window return** and `J` is mean
discounted window return. Their discount does not restart at resets. Completed
per-episode diagnostics are logged separately. Playback prints elapsed evaluation
time and transition throughput; the first run can include first-use warmup.
Training SPS counts transitions across all worlds, not batched step calls.

Viewer: SPACE pauses, TAB changes camera, S/F adjust paced speed, H toggles the
menu, and C displays backend contact forces/constraints. Contact display uses
full solver readback and can reduce rendering throughput. `--unpaced` removes
real-time pacing while limiting display refreshes to 60 Hz.

For a warmed-up fixed-work evaluation benchmark:

```bash
python benchmark_eval.py runs/LLC_PPO/run_0/best.pt \
    --env-counts 1 10 100 1024 4096 --steps 1000 --warmups 1 --repeats 3
```

## Domain-randomized environment

```bash
python train.py --environment domain_randomized \
    --experiment-name LLC_PPO_DR --run-id 0 \
    --n-envs 4096 --n-steps-per-fit 24 --n-steps 240 \
    --n-epochs-policy 5 --n-minibatches 4 --device cuda:0
```

The defaults follow Table I of
[Rapid Locomotion via Reinforcement Learning (Margolis et al.)](https://arxiv.org/abs/2205.02824):

| Parameter | Default uniform range | Application |
|---|---|---|
| Ground sliding friction | 0.05–4.0 | Floor contact coefficient |
| Target restitution | 0–1 | Approximate ground spring/damper response |
| Payload mass | −1 to +3 kg | Added to trunk mass |
| Body COM offset | −0.10 to +0.10 m per axis | Trunk inertial position |
| Motor strength | 0.9–1.1 per motor | Multiplies actuator gain after commanded-torque limits |

Each world samples its own values with a dedicated seeded Torch generator.
Trunk inertia scales proportionally with its randomized mass at fixed shape.
The floor receives higher contact priority so its friction/reference settings
control robot–ground contacts; robot–robot materials retain their XML values.
The actor/critic retain the existing 48-D observation and 12-D action dimensions;
this adds domain randomization, not the paper's teacher/student adaptation,
privileged observations or velocity curriculum.

By default, domains are sampled at initialization and retained across resets.
This avoids recomputing mass/inertia constants during every reset. To resample
only completed worlds on reset, enable:

```bash
python train.py --environment domain_randomized --experiment-name LLC_PPO_DR \
    --run-id 1 --dr-resample-on-reset \
    --dr-friction-range 0.2 1.5 \
    --dr-payload-mass-range -0.5 2.0 \
    --dr-com-range -0.03 0.03 \
    --dr-motor-strength-range 0.9 1.1 \
    --dr-restitution-range 0.0 0.5
```

On Warp, per-world model fields and derived constants are allocated before graph
capture, and updated in place. `mujoco_warp.set_const` refreshes mass/inertia
constants; optional per-reset resampling adds this work to reset handling. The
CPU reference uses a separate MjModel for each world. Checkpoints contain the
resolved ranges and sampled domain parameters for inspection/reproducibility.
Playback regenerates evaluation domains from its evaluation seed; it does not
reuse the training worlds' domain assignments.

**Restitution approximation:** MuJoCo's compliant-contact solver has no direct
PhysX-style restitution coefficient. We convert the sampled target to a damping
ratio using `zeta = -log(e)/sqrt(pi² + log(e)²)` and direct-format negative
`solref` stiffness/damping (`--dr-contact-stiffness`, default 1000). Zero is
critically damped and one is capped slightly below one. Actual rebound depends
on impedance, timestep and solver convergence; this is not an exact match to
PhysX restitution. The broad paper ranges can be narrowed through CLI arguments.

## Validation

```bash
python -m unittest discover -s tests -v
python demo.py --environment flat --backend mujoco --device cpu --num-envs 3 --steps 20
```

Tests cover PPO/GAE, reset isolation, collision and airtime rewards, configuration
precedence, checkpoint/log output, playback, asset paths, and domain physics.
Real Warp CPU kernels validate batched randomized model fields and derived
constants against separate CPU MuJoCo models. CUDA graph execution and RTX 4090
throughput require a GPU validation run; no new speed claim is made here.
