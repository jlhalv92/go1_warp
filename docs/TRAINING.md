# PPO implementation notes

Launch from the repository root with `python train.py`; configuration is defined
by `TrainConfig` in `src/algorithms/config.py`. See [the README](../README.md) for
complete examples, output paths, randomization and playback.

The actor and critic preserve the supplied SiLU / LayerNorm network structure
and parameter names. A state-independent diagonal Gaussian samples raw actions.
PPO log probabilities refer to those raw samples. The environment maps actions
to joint-position offsets and recomputes PD torque on every physics substep.
Only controller torque commands are clipped; Gaussian actions are not clipped.

PPO retains clipped surrogate loss, optional clipped value loss, entropy bonus,
separate actor/critic gradient clipping, analytic Gaussian KL and adaptive
learning rate. GAE bootstraps timeouts but not true terminations, and stops its
recursion at both episode boundaries. Values are evaluated before reset.

The rollout shape is `(n_steps_per_fit, num_envs, ...)`. Each update uses
`num_envs * n_steps_per_fit` transitions. Each epoch collects `n_steps` batched
control steps, so total training transitions are:

```text
num_envs * n_steps * n_epochs
```

`n_epochs_policy` is the number of PPO optimization passes per update; it does
not collect additional transitions. `n_minibatches` divides each rollout for
optimization, independently of the number of vectorized worlds.

The default hyperparameters remain those of the upstream script:

| Setting | Default |
|---|---:|
| num_envs | 100 |
| n_steps | 10000 |
| n_steps_per_fit | 1000 |
| n_epochs | 100 |
| n_epochs_policy | 10 |
| n_minibatches | 20 |
| n_features | 256 |
| lr | 0.001 |
| gamma / lambda | 0.99 / 0.95 |
| PPO clip | 0.2 |
| entropy coefficient | 0.01 |

Training CSV logs contain completed-episode R and J, Gaussian policy entropy,
losses, KL, learning rate and transition throughput. Fixed-step evaluation R/J
refer to the whole evaluation window, including rewards following resets.
Completed-episode diagnostics and unfinished-world counts are separate fields.

The implementation follows the continuous PPO organization used by
[CleanRL](https://github.com/vwxyzjn/cleanrl) with locomotion-oriented choices
from [RSL-RL](https://github.com/leggedrobotics/rsl_rl). Neither framework is a
runtime dependency.
