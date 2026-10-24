# Validation performed

- Five unittest cases passed on the ordinary MuJoCo CPU reference backend.
- Both supplied XMLs compile with visual meshes removed in memory: nq=19, nv=18, nu=12.
- Confirmed XML actuator joint order and signed home angles.
- Confirmed reset clears position-like keyframe controls to zero torque.
- Checked observation shape (N,48), finite rollout state, torque saturation and physics time.
- Checked deterministic seeded partial reset and preservation of unselected worlds.
- Checked timeout flags, explicit-reset guard, preservation of terminal observation storage.
- Checked inverse quaternion rotation and inverted-body termination.
- A 100-control-step, four-world zero-action CPU reference rollout completed.
- MuJoCo Warp put_model and make_data succeeded on CPU with two worlds.

No NVIDIA driver/device is available in this runtime. CUDA stepping, CUDA stream
interop, GPU partial resets, GPU/CPU numerical parity, capacity behavior under
large batches, converged PPO learning, and locomotion quality have not been validated.

Warp warns that capsule-cylinder CCD pairs have at most one contact when
MULTICCD is enabled. This comes from the supplied collision geometry and
backend support, not missing visual meshes. No contact tuning was applied.

## Custom viewer integration

- Fixed both mouse-drag and scroll calls to use MuJoCo 3.16's five-argument
  `mjv_moveCamera` signature. An additional headless regression test invokes
  both callbacks against the real MuJoCo API and checks camera changes.

- Three additional headless tests cover selected-world snapshot isolation,
  no simulation advancement during rendering, scene update acceptance, camera
  mode changes, safe geom-group keys, skipped-frame readback and clean-close return.
- Native MuJoCo Warp CPU full-state readback (`get_data_into`) was checked
  with two worlds, different x positions, four contacts and a scene update.
- Native Warp CPU reset, forward and step also succeeded during initial setup.
- Live GLFW window creation, mouse controls, visual appearance and real-time
  pacing were not tested: this runtime has no desktop display. CUDA readback
  and throughput remain untested because there is no NVIDIA GPU.

## Uploaded visual assets

- Bundled all six STL uploads in models/assets; the original XML references five.
- Full model compiles with visual meshes enabled. Compared against collision-only
  loading: body masses/inertias, joint and actuator limits, home pose, and collision
  geometry/contact parameters match.
- Default configuration loads visual meshes; --collision-only remains available.
- Live visual appearance still requires desktop validation.

## PPO trainer

- All 15 environment/viewer/PPO tests pass on the CPU reference backend.
- Hand-computed GAE cases verify timeout bootstrapping, no bootstrap on true
  termination, no recursion into reset episodes, and rollout-tail bootstrapping.
- Episode return tracking preserves pending episodes and discounted exponents.
- Analytic Gaussian KL matches torch.distributions.kl_divergence.
- Actor/critic support batches of one and optional singleton history dimension;
  action likelihood is checked on original Gaussian samples outside [-1,1].
- End-to-end short training executes rollout collection, PPO updates, evaluation
  with unequal per-world episode quotas, CSV logging, checkpoint save/load and
  resume. Parameters change and remain finite; step/update counters are checked.
- This validates execution and bookkeeping, not learned walking performance.
