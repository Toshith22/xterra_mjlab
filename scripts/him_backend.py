"""Isolated, pinned upstream HIMLoco with explicit distributed adaptations.

Upstream files remain unmodified and retain their licenses. No replacement of
the installed rsl_rl package. Both PPO and estimator gradients are synchronized
BEFORE clipping; KL learning-rate decisions are global. Sinkhorn assignments
and advantage normalization remain rank-local (documented port difference).
"""
import sys
import types
from pathlib import Path

import torch
import torch.distributed as dist


def mean(value):
    value = value.clone()
    if dist.is_initialized():
        dist.all_reduce(value)
        value /= dist.get_world_size()
    return value


def clip(parameters, maximum):
    parameters = list(parameters)
    active = [p for p in parameters if p.grad is not None]
    if dist.is_initialized() and active:
        flat = torch.cat([p.grad.flatten() for p in active])
        dist.all_reduce(flat)
        flat /= dist.get_world_size()
        offset = 0
        for p in active:
            p.grad.copy_(flat[offset:offset+p.numel()].view_as(p))
            offset += p.numel()
    return torch.nn.utils.clip_grad_norm_(parameters, maximum, error_if_nonfinite=True)


def load(source):
    root = Path(source) / "rsl_rl/rsl_rl"
    namespace = "m2_him_upstream"
    for package in ("", ".modules", ".storage", ".algorithms", ".utils"):
        module = types.ModuleType(namespace + package)
        module.__path__ = []
        sys.modules[module.__name__] = module
    specs = [
        ("utils.utils", "split_and_pad_trajectories"),
        ("modules.actor_critic", "ActorCritic"),
        ("modules.him_estimator", "HIMEstimator"),
        ("modules.him_actor_critic", "HIMActorCritic"),
        ("storage.him_rollout_storage", "HIMRolloutStorage"),
        ("algorithms.him_ppo", "HIMPPO"),
    ]
    for name, exported in specs:
        path = root / (name.replace(".", "/") + ".py")
        code = path.read_text().replace("rsl_rl", namespace)
        code = code.replace("nn.utils.clip_grad_norm_", "_distributed_clip")
        code = code.replace("kl_mean = torch.mean(kl)", "kl_mean = _distributed_mean(torch.mean(kl))")
        # Upstream computes means but accidentally returns the last estimator batch.
        code = code.replace(
            "return mean_value_loss, mean_surrogate_loss, estimation_loss, swap_loss",
            "return mean_value_loss, mean_surrogate_loss, mean_estimation_loss, mean_swap_loss")
        module = types.ModuleType(namespace + "." + name)
        module.__file__ = str(path)
        module.__package__ = module.__name__.rsplit(".", 1)[0]
        module.__dict__.update(_distributed_clip=clip, _distributed_mean=mean)
        sys.modules[module.__name__] = module
        exec(compile(code, str(path), "exec"), module.__dict__)
        setattr(sys.modules[module.__package__], exported, getattr(module, exported))
    return (sys.modules[namespace+".modules"].HIMActorCritic,
            sys.modules[namespace+".algorithms"].HIMPPO)
