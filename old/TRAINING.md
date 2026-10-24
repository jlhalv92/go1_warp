# Go1 PPO training

`python -m go1_warp.train` trains the existing flat-ground environment. It uses
CleanRL-style continuous-action PPO collection and minibatch updates, with
RSL-RL-style locomotion PPO choices. There is no dependency on either framework,
Gymnasium, or MushroomRL. Networks are in networks.py, PPO math/storage in ppo.py,
and the executable training loop is in train.py.

## Start on your NVIDIA T550 (4 GB)

Keep the working Python 3.12 / CUDA 12 PyTorch and Warp environment. Adding this
trainer does not require reinstalling MuJoCo, PyTorch or Warp. The package includes
all previous viewer fixes, XMLs and STL assets.

```bash
conda activate go1_warp
# From the directory containing go1_warp/:
python -m go1_warp.train --n-envs 16 --device cuda:0 --name-exp GO1_FLAT
```

16 worlds is a starting batch size for this laptop, not a throughput optimum.
The exact reference default is 100 worlds; use --n-envs 100 if resources permit.
No GPU memory or performance claim is made here. Start headless; --render adds
unpaced training visualization, and closing the window continues training headless.

A short CPU reference check:

```bash
python -m go1_warp.train --backend mujoco --device cpu \
  --n-envs 2 --n-epochs 1 --n-steps 32 --n-steps-per-fit 16 \
  --n-minibatches 2 --n-epochs-policy 2 --n-features 32 \
  --n-episodes-test 2 --eval-envs 2 --episode-seconds 0.1 --collision-only
```

This short check validates execution; it cannot learn walking.

## Exact defaults taken from your script

| Reference argument | Trainer argument (underscore alias also accepted) | Default |
|---|---|---:|
| n_envs | --n-envs | 100 |
| n_steps | --n-steps | 10000 per world per outer epoch |
| n_epochs | --n-epochs | 100 outer epochs |
| n_steps_per_fit | --n-steps-per-fit | 1000 per world per PPO rollout |
| n_minibatches | --n-minibatches | 20 |
| n_epochs_policy | --n-epochs-policy | 10 |
| n_features | --n-features | 256 |
| lr | --lr | 0.001 initial |
| eps_ppo | --eps-ppo | 0.2 |
| lam | --lam | 0.95 |
| ent_coeff | --ent-coeff | 0.01 |
| gamma | --gamma | 0.99 |
| std_0 | Gaussian initial std | 1.0 per joint |
| n_episodes_test | --n-episodes-test | 10 |
| target_vx | --target-vx | 1.5 m/s |
| target_vy | --target-vy | 0.8 m/s |
| target_wz | --target-wz | 1.0 rad/s |

Command samples are uniform in symmetric ranges [-limit,+limit], resampled every
five seconds and on reset. The source script's environment-specific command
sampling implementation was not supplied, so its other modes cannot be reproduced.

At reference defaults: batch=100*1000=100000 transitions; minibatch=5000;
10 PPO updates per outer epoch; 1000 PPO updates / 100 million training transitions
over the whole run. Evaluation steps are excluded from global_step. Using 16
worlds reduces the batch to 16000, minibatches to 800 and the total to 16 million.
`n_steps` must be divisible by `n_steps_per_fit`; no rollout data is dropped.
Minibatches are reshuffled every PPO epoch, and unequal splits retain every sample.

## Requested networks

Both actor and critic use:

`48 -> Linear(512) -> LayerNorm -> SiLU -> Linear(256) -> LayerNorm -> SiLU -> Linear(128) -> LayerNorm -> SiLU -> output`

The actor output has 12 unbounded means. Its Linear layers retain the default
PyTorch initialization, as in your code. A separate learned, state-independent
12-vector of log standard deviations defines a diagonal Gaussian; initial std=1.
The critic has a single scalar output, with orthogonal gain sqrt(2) on its hidden
Linear layers and 0.01 on its output layer, zero biases, exactly as requested.
Forward passes accept (B,D) and (B,1,D); batch size 1 is preserved.

Both networks receive the current 48-dimensional observation. No privileged
critic input, observation running normalization, reward normalization, shared
network body or pretrained actor transfer is introduced. The existing environment
observation scales and the requested LayerNorm remain in effect.

PPO stores and scores the original Gaussian sample. The environment clips it
to [-1,1], converts it to a home-pose residual and computes PD torques. Never
replace stored actions with clipped actions: their Gaussian likelihoods differ.
Logged entropy is the sum of analytic Gaussian differential entropies over the
12 action dimensions, matching an unsquashed Gaussian policy; it is not the
entropy of the physically clipped joint commands. Log std is bounded to [-5,2]
after updates for numerical stability (configurable).

## RSL-RL-inspired additions and explicit choices

- Exact analytic KL(old Gaussian || new Gaussian), not only sampled log-ratio KL.
- Adaptive LR by default: target KL .01; divide LR by 1.5 above 2*target,
  multiply by 1.5 below target/2 when KL>0; bound LR to [1e-5,1e-2]. Applied per
  minibatch, as in the referenced RSL-RL implementation. --schedule fixed keeps LR
  fixed at its current optimizer value. Initial LR is your requested 1e-3.
- Whole-rollout advantage normalization before optimization; no minibatch normalization.
- Clipped value loss (clip .2), value-loss coefficient 1.0 and no extra 0.5 factor,
  following RSL-RL's MSE convention. --no-clip-vloss disables value clipping.
- Max gradient norm 1.0, separately for actor (including log std) and critic.
- Learned global action std, entropy coefficient .01, Adam eps=1e-5.
- Initial episode counters are randomized once to desynchronize timeouts;
  --no-randomize-init-episode-length disables this. This shortens the first
  training episode in each world without pretending that unobserved rewards
  occurred; evaluation always starts from a full reset.
- No learning-rate annealing or KL early-stop is combined with adaptive LR.
- No symmetry augmentation, RND, teacher/student distillation, recurrent policy,
  AMP, torch.compile, multi-GPU training or terrain curriculum is included.

This is a selected implementation of the relevant PPO choices, not a claim to
reproduce all settings/results of Rudin et al. or current RSL-RL.

## Correct termination and timeout handling

The rollout stores r_t, V(s_t), V(s_{t+1}), terminated and truncated separately.
The next value is evaluated on the final pre-reset observation. True terminal
next states are never passed into the critic (important for nonfinite failures).

`delta_t = r_t + gamma * (1-terminated_t) * V(next_state_t) - V(state_t)`

`A_t = delta_t + gamma * lambda * (1-done_t) * A_(t+1)`

where done=terminated|truncated. Thus a timeout bootstraps, while a fall does not;
neither allows GAE to cross into the next episode. The final rollout transition
also bootstraps when nonterminal. No extra timeout reward correction is added,
which would otherwise double-count this bootstrap.

## Logging R, J and entropy

Your script's R and J are **evaluation episode returns**, so these exact names
are reserved for evaluation here:

- R = mean over episodes of sum_t r_t.
- J = mean over episodes of sum_t gamma^t * r_t, starting exponent t=0.
- entropy = sum of Gaussian action entropies from the current policy.

Evaluation runs before learning and after each outer epoch (--eval-every can
change the interval), using a separate environment, fixed reset/command seed,
and a fixed per-world quota totaling exactly --n-episodes-test. Quotas prevent
short failed episodes from being overrepresented. Deterministic mean actions
are the default; --stochastic-eval uses the Gaussian policy without changing
the training action RNG. Eval includes raw environmental rewards, no GAE or
value bootstrap corrections. Failure fraction and episode length are also logged.

Your source averaged R/J over four terrains after each epoch. This package has
only flat terrain: results here are **flat-ground evaluation only**. The source's
other terrains, current_level logic, and domain randomization are not silently
replaced with equivalent-looking labels. Metrics across the two environments
are not directly comparable, since their reward implementations also differ.

Always saved locally under runs/<name>_seed<seed>_<timestamp>/:

- config.json: full effective trainer/environment parameters and step counts.
- train.csv: each PPO update; train_R/train_J over completed training episodes,
  plus losses, KL, entropy, LR, std, clipping, gradient norms, explained variance,
  episode counts/lengths and training transitions per second (SPS).
- evaluation.csv: R, J, entropy and evaluation diagnostics.
- last.pt: latest completed outer epoch.
- best.pt: best evaluation R in this run (or a new historical best after resume).
- epoch_0010.pt etc.: numbered snapshots every ten outer epochs.

Training return tracking preserves unfinished episodes across rollouts. A CSV
return cell is blank when no episode completed in that update, rather than
writing a misleading zero. Initially randomized horizons affect only the first
training episodes, not evaluation. SPS includes collection and PPO optimization,
not evaluation or checkpoint writing; first-time compilation distorts early SPS.

Optional logging integrations (install only if needed):

```bash
python -m pip install -r go1_warp/requirements-training.txt
python -m go1_warp.train --n-envs 16 --log --wandb-project A_POLICY_TRAINING
# Offline W&B:
python -m go1_warp.train --n-envs 16 --log --wandb-mode offline
# TensorBoard:
python -m go1_warp.train --n-envs 16 --tensorboard
tensorboard --logdir runs
```

W&B evaluation keys are exactly R, J, entropy. Other training metrics use train/.
W&B uses global training transition count as its step, not evaluation steps.

## Resume and playback

```bash
python -m go1_warp.train --n-envs 16 --n-epochs 200 \
  --resume runs/<run>/last.pt

python -m go1_warp.play runs/<run>/best.pt --device cuda:0
python -m go1_warp.play runs/<run>/best.pt --device cuda:0 --headless --episodes 10
```

Resume restores weights, optimizer/adaptive LR and counters in a new run directory.
`n_epochs` is the total target, not an extra number of epochs. Pass matching
network width, gamma, command limits and episode duration; other effective
arguments are recorded for review. Simulator state and pending episodes restart,
so this is not a bitwise continuation. GPU/CPU optimizer state migration is
supported. Checkpoints include trusted Python configuration/RNG data and are
loaded with weights_only=False. Playback reconstructs the saved network and
environment parameters and uses mean actions unless --stochastic is supplied.

## Current validation

CPU tests execute collection, updates, timeouts, evaluation, logging, checkpointing
and resume; see VALIDATION.md. CUDA training and convergence to a walking gait
have not been run here. Reward coefficients, command ranges and PD gains remain
starting values and may need locomotion tuning. The trainer does not guarantee
that the current flat-ground reward alone produces a desirable gait.

## Sources

Independent trainer implementation informed by:

- CleanRL continuous PPO: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py
- CleanRL PPO documentation: https://docs.cleanrl.dev/rl-algorithms/ppo/
- RSL-RL PPO: https://github.com/leggedrobotics/rsl_rl/blob/main/rsl_rl/algorithms/ppo.py
- RSL-RL on-policy runner: https://github.com/leggedrobotics/rsl_rl/blob/main/rsl_rl/runners/on_policy_runner.py
- Rudin et al., Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning: https://arxiv.org/abs/2109.11978
- Your attached PPO script supplies the networks, primary hyperparameters and R/J/entropy definitions.
