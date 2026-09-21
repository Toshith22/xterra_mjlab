# SvanM2 Mixture of Experts (MoE-CTS) Policy

Trained using native MuJoCo-Warp dynamics.
- Checkpoint update: 90,000 (selected reference from multi-seed evaluation)
- Environments: 8,192
- Rollout length: 24 steps
- Samples per update: 196,608
- Experts: 8 concurrent teacher/student latent experts

## Files
| File | Use | Format / Shape |
|------|-----|----------------|
| `model_90000.pt` | rsl_rl model checkpoint | PyTorch state dict |
| `policy.onnx` | Deterministic student encoder + actor deployment graph | `[1, 225]` input $\to$ `[1, 12]` raw actions |
| `offline_inference.ts` | TorchScript traced evaluation model | `[1, 225]` oldest-first input $\to$ `[1, 12]` raw actions |
