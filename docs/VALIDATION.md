# Refactor validation

The source refactor starts from upstream commit
`68d38c5fafda4e465635629e5e325efa372c1d84`.

Run the regression suite with:

```bash
python -m unittest discover -s tests -v
```

The suite exercises the CPU reference simulator and real Warp kernels on CPU.
It checks network outputs/log probabilities, timeout-aware GAE, PPO updates,
reset isolation, contact-force thresholds, foot airtime bookkeeping, viewer
callbacks without a desktop, asset loading, dataclass/CLI precedence and output
paths, numbered checkpoints, save/resume/play, and randomized physical models.
Warp mass/inertia constants are compared to independent CPU MuJoCo models.

Validation on 24 October 2026: 29 tests passed with Python 3.12,
MuJoCo/MuJoCo Warp 3.16.0, Warp 1.18.0 and Torch 2.5.1 (CPU). Direct scripts
were launched from a foreign working directory, including legacy checkpoint
playback and the benchmark launcher. A separate 100-step, three-world comparison
matched the upstream active flat environment exactly for observations, state,
all reward terms and termination/reset behavior. A short domain-randomized PPO
training run with per-reset resampling saved best/last/numbered checkpoints and
completed headless playback.

For a GPU check, use the same environment count, rollout length, solver settings
and profiling flags as your baseline. Confirm the startup line reports
`graphs(step/forward/reset)=True/True/True` and `contacts=gpu` for `flat` and
`domain_randomized`. Compare warmed-up SPS with profiling disabled. Domain
randomization changes physical trajectories and training returns, so compare
throughput separately from reward progress.

No GPU is available in the refactor test workspace. CUDA capture/replay and
4090 throughput remain hardware validation items. Two CUDA-only regression tests
are included and skipped automatically when CUDA is unavailable. They compare
graph/direct stepping, including partial resets and randomized-model updates.
