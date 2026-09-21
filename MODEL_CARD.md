# Model card — SvanM2 flat velocity policy (mjlab)

A velocity-tracking policy for the SvanM2 quadruped on flat terrain, shipped
under `checkpoints/svanm2_flat/`.

## Overview

- **Task:** `xTerra-Mjlab-Velocity-Flat-SvanM2` (flat-terrain velocity tracking).
- **Robot:** SvanM2 quadruped, 12 DoF (FL/FR/RL/RR × hip/thigh/calf).
- **Algorithm:** rsl_rl PPO (rsl-rl-lib 5.x) via mjlab's `VelocityOnPolicyRunner`;
  actor/critic MLP `[512, 256, 128]`, ELU, adaptive-KL LR schedule.
- **Control:** joint-position targets at 50 Hz, per-joint action scale
  (0.25·effort/kp), parallel-mechanism actuator (kp = 20, kd = 0.7, per-stage
  gear 8/8/16, thigh→calf belt coupling 0.5, joint-side effort limits 12 N·m
  hip/thigh, 24 N·m calf).

## Interfaces

- **Policy observation:** a **6-frame history** (term-major) of the per-frame
  layout — velocity command (3) + base angular velocity (3) + projected gravity
  (3) + joint positions (12) + joint velocities (12) + last action (12) =
  45/frame × 6 = **270 dims**.
- **Action:** 12 joint-position targets (leg order FL, FR, RL, RR; within leg
  hip, thigh, calf).
- **Critic observation:** proprioception + true base linear velocity + foot
  contact/air-time features (privileged, no history).

## Training

- **Command:** `python scripts/train.py xTerra-Mjlab-Velocity-Flat-SvanM2
  --env.scene.num-envs 4096`
- **Environments:** 4096. **Iterations:** 3000. **Seed:** 42.
- **Final mean reward:** 158.3.
- **Stack:** mjlab 1.5.x, MuJoCo 3.10.0, MuJoCo-Warp 3.10.0.x, Warp 1.15.0,
  rsl-rl-lib 5.x, PyTorch 2.7.0 (CUDA 12.8), Python 3.11.
- **Hardware:** 1× NVIDIA RTX PRO 4500 Blackwell (32 GB).

## Artifacts (`checkpoints/svanm2_flat/`)

| File | Purpose | SHA256 |
|------|---------|--------|
| `svanm2_flat.pt` | rsl_rl checkpoint (`play.py --checkpoint-file`) | `8101dbb28b103a47dd2219d97eac395a79819bcfa65acb8e8457a18c1aa1a9d3` |
| `policy.onnx` | ONNX actor, 270→12 (deployment) | `f13055966f94443206097cd05154c9e717e845f390171e02708280e7d918f5ab` |

## Evaluation

Rollout in the mjlab environment, 4 s per command, mean of the last 2 s (nominal
trunk height 0.32 m):

| Command | Measured | Height (m) | Stable |
|---------|----------|-----------|--------|
| forward vx +1.0    | vx +1.00          | 0.35 | yes |
| backward vx −1.0   | vx −0.99          | 0.35 | yes |
| strafe left vy +0.6  | vy +0.59        | 0.35 | yes |
| strafe right vy −0.6 | vy −0.61        | 0.35 | yes |
| turn left ωz +0.6  | ωz +0.49          | 0.36 | yes |
| turn right ωz −0.6 | ωz −0.50          | 0.33 | yes |
| diagonal (0.7, 0.5)  | vx 0.71, vy 0.51 | 0.34 | yes |
| fwd + turn (1.0, 0.4)| vx 1.00, ωz 0.31 | 0.35 | yes |

Linear tracking is near-exact; angular tracking slightly undershoots; stable at
nominal height across all commands.

## Known limitations

- **Flat terrain only** — (for `svanm2_flat`) no rough/stairs.
- **Angular undershoot** — yaw-rate commands settle a little below target.
- **Not hardware-validated** — (`svanm2_flat`) simulation only.

---

# Model card — SvanM2 HIMLoco policy (mjlab)

A high-speed terrain locomotion policy trained with the HIMLoco estimator-actor architecture, shipped under `checkpoints/svanm2_himloco/`.

## Overview
- **Task:** Native MuJoCo SvanM2 terrain locomotion (flat, stairs, slopes).
- **Robot:** SvanM2 quadruped, 12 DoF.
- **Algorithm:** HIMLoco (temporal convolutional estimator + PPO actor), 20,000 updates.
- **Interfaces:** 6-frame history of 45-dim observations (270 dims total) $\to$ 12 raw joint actions.
- **Artifacts:**
  - `model_20000.pt` (PyTorch state dict)
  - `policy.onnx` (ONNX export)
  - `offline_inference.ts` (TorchScript export)
  - `env.yaml` (training configuration)

---

# Model card — SvanM2 MoE-CTS policy (mjlab)

A multi-terrain locomotion policy trained with Mixture of Experts Concurrent Teacher-Student (MoE-CTS), shipped under `checkpoints/svanm2_moe/`.

## Overview
- **Task:** Native MuJoCo SvanM2 terrain locomotion (flat, stairs, slopes).
- **Robot:** SvanM2 quadruped, 12 DoF.
- **Algorithm:** MoE-CTS (8 concurrent latent experts, teacher/student distillation with load balancing), 90,000 updates. Selected as best-performing reference from multi-seed benchmarking (643/720 qualified passes).
- **Interfaces:** 5-frame history of 45-dim observations (225 dims total) $\to$ 12 raw joint actions.
- **Artifacts:**
  - `model_90000.pt` (PyTorch state dict)
  - `policy.onnx` (ONNX export)
  - `offline_inference.ts` (TorchScript export)

## Provenance & license

- Config and code: this repository (Apache-2.0, © xTerra Robotics).
- Robot description (MJCF/STL): xTerra Robotics (Apache-2.0, see `NOTICE`).
