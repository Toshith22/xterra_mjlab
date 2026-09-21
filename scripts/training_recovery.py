"""Explicit numerical safeguards and checkpoint restoration for recovery runs."""
import torch


def require_finite(value, label="state"):
    tensors = []
    def collect(item, name):
        if isinstance(item, torch.Tensor):
            tensors.append((name, item))
        elif isinstance(item, dict):
            for key, child in item.items():
                collect(child, f"{name}.{key}")
        elif isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                collect(child, f"{name}[{index}]")
    collect(value, label)
    for device in {t.device for _, t in tensors}:
        group = [(name, t) for name, t in tensors if t.device == device]
        if not torch.stack([torch.isfinite(t).all() for _, t in group]).all():
            bad = [name for name, t in group if not torch.isfinite(t).all()]
            raise FloatingPointError(f"Non-finite tensors: {bad}")


def restore(checkpoint, model, optimizers, algorithm, arguments):
    saved = checkpoint["arguments"]
    for key in ("algorithm", "num_envs", "steps", "seed"):
        if saved[key] != getattr(arguments, key):
            raise ValueError(f"Resume mismatch for {key}: {saved[key]} != {getattr(arguments, key)}")
    require_finite(checkpoint["model_state_dict"], "model")
    require_finite(checkpoint["optimizer_states"], "optimizers")
    if len(optimizers) != len(checkpoint["optimizer_states"]):
        raise ValueError("Optimizer count changed")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if (model.std <= 0).any():
        raise ValueError("Checkpoint has non-positive action standard deviation")
    for optimizer, state in zip(optimizers, checkpoint["optimizer_states"]):
        optimizer.load_state_dict(state)
    algorithm.learning_rate = float(checkpoint["learning_rate"])
    return int(checkpoint["update"])


def install_optimizer_guards(model, optimizers, algorithm, max_lr, std_min, std_max):
    """Check before any Adam mutation; bound noise after each minibatch update.

    No upstream source is modified. Raises on non-finite gradients/weights;
    never silently replaces a failed model or optimizer state with zeros.
    """
    if not 0 < std_min < std_max or max_lr <= 0:
        raise ValueError("Invalid recovery bounds")
    if ((model.std < std_min) | (model.std > std_max)).any():
        raise ValueError("Resume from a checkpoint with healthy exploration noise")
    stats = {"std_bound_events": 0}

    def before(optimizer, args, kwargs):
        algorithm.learning_rate = min(algorithm.learning_rate, max_lr)
        parameters = []
        for group in optimizer.param_groups:
            group["lr"] = min(group["lr"], max_lr)
            parameters.extend(p for p in group["params"] if p.grad is not None)
        # Also clips the WTW adaptation update, which upstream leaves unclipped.
        torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)

    def after(optimizer, args, kwargs):
        with torch.no_grad():
            require_finite(model.state_dict(), "post_update_model")
            outside = (model.std < std_min) | (model.std > std_max)
            if outside.any():
                stats["std_bound_events"] += 1
                model.std.clamp_(std_min, std_max)
                # Remove outward Adam momentum only for projected dimensions.
                for opt in optimizers:
                    state = opt.state.get(model.std, {})
                    for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                        if key in state:
                            state[key][outside] = 0

    for optimizer in optimizers:
        for group in optimizer.param_groups:
            group["lr"] = min(group["lr"], max_lr)
        optimizer.register_step_pre_hook(before)
        optimizer.register_step_post_hook(after)
    algorithm.learning_rate = min(algorithm.learning_rate, max_lr)
    return stats


def timeout_bootstrap(rewards, values, timeouts, gamma=.99):
    """Do not let NaN terminal values contaminate non-timeout transitions."""
    require_finite(values[timeouts], "timeout_values")
    return rewards + gamma * torch.where(timeouts, values, 0.)
