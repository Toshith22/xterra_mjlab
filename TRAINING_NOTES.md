# ISL V7 formulation on public SvanM2

## Continuous mode (user requested, September 13)

The original 100-update stop is superseded by scripts/continue_isl.py.
Both seeds resume actor, critic and optimizer at update 100, unlocked level 1,
and target 12,000 total updates. Save checkpoints every 100; evaluate every
500 in a separate process so the training simulator and random state are not
replaced by evaluation state. Training continues automatically after evaluation.
The current episode states are freshly reset on process resume; this is not
a bit-for-bit replay of the interrupted simulator trajectory.

Functional ISL acceptance still gates difficulty increases. Ordinary secondary
style/scuff/landing proxies are recorded but no longer require a human pause
or veto an otherwise functional promotion. The unmodified review is preserved.
Numerical failures, evaluation crashes/timeouts, or two consecutive severe
fall regressions on previously validated clean cases stop and write ALERT.json.
These checks do not replace visual review or establish hardware safety.

scripts/watch_isl.py checks both runs every five minutes for errors, stalled
heartbeats or unexpected process exits; repeated connectivity failures also
alert. It queues an alert into the current Codex chat using codex queue.
This local watcher needs this workstation and its Codex service available,
expires after 48 hours, and stops when both runs finish. It is not a cloud
scheduled task. Delivery attempts/errors are in ISL_WATCHER_STATE.json.
The OpenAI Docs check established that CLI lacks the Scheduled management UI:
https://learn.chatgpt.com/docs/automations?surface=app

The remaining sections describe the original port and first 100-update block.

Runtime: public xterra_mjlab, upstream commit
522da8e02d08462be550db223ad4adc2d5ff338b, on native MuJoCo-Warp/mjlab.
All tracked upstream robot, actuator and task files remain unchanged.
Entry point: scripts/train_isl.py -> isl_cfg.py -> isl_native.py/ref_task.py.
The earlier isl_task.py prototype and reference_*.py helpers are superseded;
they are not imported by the V7 training entry point.

Reference: frozen ISL seed97003/continue_1000/block2/source from
fresh20260912_heading_v7_isl. Pure ref_*.py modules retain the ISL numerical
rewards, recovery, terrain queries, height surrogate and acceptance criteria.
There is no Genesis runtime, old URDF, HIM policy or inherited checkpoint.

Ported formulation:
- Stock PPO (24 steps, five epochs, four minibatches, initial LR 3e-4),
  512/256/128 MLPs, symmetry augmentation and mirror loss 0.01.
- Six proprioceptive frames plus delayed/noisy finite-view height observations;
  asymmetric critic adds true velocity, clean terrain, foot state and actual DR.
- Ten levels, seven lanes, solid slopes and six-step ascending/descending stairs,
  25/30/35/40 cm treads, 6–18 cm risers. Curriculum starts at level zero.
- Frontier/easier-level mixing; move/stop/resume/stand timing; bounded turn bouts,
  integrated heading reference; nine finite-force/torque pulse families.
- ISL foot-state, phase-lag symmetry, soft abduction and recovery objectives.
  No extra native landing/joint-limit reward stack.
- Independent-seed mean-action evaluation, first episode per case, 64 trials,
  calibrated tracking/heading, quiet support, recovery and warning review.
  Missing, failed, smoke or incomplete evaluations cannot authorize promotion.

Embodiment adaptations:
- Preserve public M2 nominal pose, XML, mass, effort/position limits, transmission,
  armature, gains, action scales and actuator delay. Spawn 0.34 m; height target
  0.32 m. ISL randomization ranges apply around M2 nominal hardware.
- Native term-major history/joint order; equivalent feature categories and scales.
- Public foot sites and body contact sensors. MuJoCo contact forces are negated
  to represent force ON the robot, verified with a resting one-kilogram sphere.
- Native simulator/reset timing; no claim of numerical equivalence to Genesis.
- Stair-edge telemetry uses sphere bottom minus upper edge height. Reward foot
  heights retain the public site convention. Neither metric is hardware proof.

Verification: 75 tensor/native-interface regression tests; 70 CPU collision
profiles (max error < 1e-7 m); GPU PPO, reflection, reset, pulse and evaluator
smokes. Resting-robot force/loaded-contact assertions guard the sensor bridge.

Each run freezes source/configuration and starts from scratch. It saves training
episode records, model_review.pt, training_ledger.json, development_review.json,
review_ledger.json and progress.json. A block stops after 100 updates for review;
promotion eligibility does not silently start another run. No trained stair
capability is claimed until later-level evaluation and recorded-motion review.

Earlier seed98001_block1/seed98002_block1 pilots were stopped during setup after
the user clarified the full ISL port; preserve them as aborted prototypes.
New V7 runs use separate names and do not resume their checkpoints.

Isolated environment logging dependencies:
tensorboard==2.20.0, wandb==0.22.3, protobuf==6.33.5 (pip check passed).
