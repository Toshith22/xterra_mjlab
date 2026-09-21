# SvanM2 HIMLoco Policy

Trained using native MuJoCo-Warp dynamics.
- Checkpoint update: 20,000
- Environments: 4,096
- Rollout length: 100 steps
- Samples per update: 409,600

## Files
| File | Use | Format / Shape |
|------|-----|----------------|
| `model_20000.pt` | rsl_rl model checkpoint | PyTorch state dict |
| `policy.onnx` | Deterministic estimator + actor deployment graph | `[1, 270]` input $\to$ `[1, 12]` raw actions |
| `offline_inference.ts` | TorchScript traced evaluation model | `[1, 270]` input $\to$ `[1, 12]` raw actions |
| `env.yaml` | Training environment configuration | YAML |
