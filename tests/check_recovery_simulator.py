"""GPU integration check: terminate/reset a corrupt observation, then keep stepping."""
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from m2_algorithm_env import Adapter, make_cfg

torch.set_num_threads(4)
adapter = Adapter(make_cfg(64, 103001), "cuda:0", numerical_guard=True)
original = adapter.env.step

def inject(actions):
    obs, rewards, terminated, truncated, extras = original(actions)
    obs["actor"][0, 0] = float("nan")
    obs["critic"][0, 0] = float("inf")
    truncated[0] = True
    return obs, rewards, terminated, truncated, extras

adapter.env.step = inject
reward, done, info, successor = adapter.step(torch.zeros(64, 12, device="cuda:0"))
assert done[0] and not info["time_outs"][0] and reward[0] == -1
assert adapter.numerical_resets == 1
assert torch.isfinite(adapter.history).all() and torch.isfinite(successor).all()
adapter.env.step = original
for _ in range(12):
    adapter.step(torch.zeros(64, 12, device="cuda:0"))
assert torch.isfinite(adapter.history).all()
adapter.env.close()
print("RECOVERY_SIMULATOR_TEST_PASSED", flush=True)
