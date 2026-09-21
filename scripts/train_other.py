"""Long-running native M2 ports of pinned WTW and MoE-CTS upstream learners."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from mjlab.utils.os import dump_yaml
from xterra_mjlab.assets.svanm2.constants import SVANM2_XML
from m2_algorithm_env import Adapter, make_cfg
from other_backends import load
from train_him import write
from training_recovery import restore, install_optimizer_guards, timeout_bootstrap, require_finite


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--algorithm", choices=("wtw", "moe"), required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--num-envs", type=int, required=True)
    p.add_argument("--steps", type=int, default=24)
    p.add_argument("--updates", type=int, default=150000)
    p.add_argument("--seed", type=int, default=102001)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--resume", type=Path)
    p.add_argument("--stabilize", action="store_true")
    p.add_argument("--max-learning-rate", type=float, default=3e-4)
    p.add_argument("--std-min", type=float, default=.05)
    p.add_argument("--std-max", type=float, default=1.)
    a = p.parse_args()
    torch.set_num_threads(int(os.environ.get("M2_TORCH_THREADS", "6")))
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)
    device = "cuda:0"
    cfg = make_cfg(a.num_envs, a.seed)
    if a.smoke:
        cfg.episode_length_s = .20
    wtw = a.algorithm == "wtw"
    if wtw:
        import wtw_m2
        cfg = wtw_m2.configure(cfg)
    history_length, frame_size = (30, 55) if wtw else (5, 45)
    xml_hash = hashlib.sha256(SVANM2_XML.read_bytes()).hexdigest()
    assert xml_hash == "2c37cc2f9eaa7be52a8c2312a1fffca779be08ce9a34406129bc2d2226527163"
    a.out.mkdir(parents=True, exist_ok=False)
    write(a.out/"manifest.json", dict(algorithm=a.algorithm+"/native-MuJoCo SvanM2 adaptation",
        arguments=vars(a), initialized_from=str(a.resume) if a.resume else "scratch", deployable=False,
        numerical_recovery=(dict(max_learning_rate=a.max_learning_rate,
            std_bounds=[a.std_min, a.std_max], gradient_norm_limit=1.,
            invalid_worlds="terminate with reward -1, no timeout bootstrap, reset; stop if >1% in one step") if a.stabilize else None),
        upstream_commit=("0e7236bdc81ce855cbe3d70345a7899452bdeb1c" if wtw else "30e74dc507bec7a642a8c98be26081f2c6f0822d"),
        robot_xml_sha256=xml_hash, public_commit="522da8e02d08462be550db223ad4adc2d5ff338b",
        targets={"stairs_m": [.20, .25], "speed_m_s": 2., "slope_degrees": 30},
        differences=["Native public M2 hardware/actions, native task rewards and per-environment terrain curriculum",
            "True terminal-value timeout bootstrapping; no terminal history leakage",
            "No exact IsaacGym/Go1/Go2 paper reproduction or achieved-capability claim"] + (
            ["WTW original PPO/adaptation network; 30-frame public proprioception plus six gait commands and four clocks",
             "WTW phase-conditioned contact force/velocity and terrain-relative swing clearance",
             "Native friction/base-mass privileged targets replace IsaacGym friction/restitution",
             "No body-geometry commands, two-previous-action observation, or multidimensional command-bin curriculum"]
            if wtw else ["Original eight-expert MoE-CTS learner, 75% teacher / 25% student concurrent rollouts",
             "Original student latent distillation and load balance loss; five-frame history",
             "Public native privileged critic observations replace Go2 privileged observations"])))
    dump_yaml(a.out/"env.yaml", asdict(cfg))
    source = Path(__file__).resolve().parents[1]
    shutil.copytree(source/"scripts", a.out/"source"/"scripts", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(source/"xterra_mjlab", a.out/"source"/"xterra_mjlab", ignore=shutil.ignore_patterns("__pycache__"))
    write(a.out/"source_hashes.json", {str(f.relative_to(a.source)): hashlib.sha256(f.read_bytes()).hexdigest()
          for f in a.source.rglob("*.py")})
    adapter = Adapter(cfg, device, history=history_length, frame_size=frame_size, numerical_guard=a.stabilize)
    Model, Algorithm = load(a.source, a.algorithm)
    if wtw:
        model = Model(frame_size, 2, frame_size*history_length, 12).to(device)
        alg = Algorithm(model, device=device)
        alg.init_storage(a.num_envs, a.steps, [frame_size], [2], [frame_size*history_length], [12])
        optimizers = [alg.optimizer, alg.adaptation_module_optimizer]
    else:
        model = Model(frame_size, adapter.critic.shape[1], 12, a.num_envs, history_length).to(device)
        alg = Algorithm(model, a.num_envs, history_length, device=device,
            num_learning_epochs=5, num_mini_batches=4, entropy_coef=.01,
            schedule="adaptive", gamma=.99, lam=.95)
        alg.init_storage(a.num_envs, a.steps, [frame_size], [adapter.critic.shape[1]], [12])
        optimizers = [alg.optimizer1, alg.optimizer2]
    completed = 0
    if a.resume:
        checkpoint = torch.load(a.resume, map_location="cpu", weights_only=False)
        completed = restore(checkpoint, model, optimizers, alg, a)
        if a.updates <= completed:
            raise ValueError("Target must exceed the checkpoint update")
        adapter.env.common_step_counter = checkpoint["common_step_counter"]
        # Old checkpoints did not save terrain levels or simulator state.
        # Reset episodes on the initial terrain; preserve command schedule.
        if "terrain_levels" in checkpoint:
            terrain = adapter.env.scene.terrain
            terrain.terrain_levels[:] = checkpoint["terrain_levels"].to(device)
            terrain.terrain_types[:] = checkpoint["terrain_types"].to(device)
            terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
        adapter.env._m2_restoring = True
        obs, _ = adapter.env.reset()
        adapter.env._m2_restoring = False
        adapter.history = obs["actor"][:, None].repeat(1, history_length, 1)
        adapter.critic = obs["critic"]
        torch.set_rng_state(checkpoint["rng"].cpu())
        torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng"]])
        if "numpy_rng" in checkpoint:
            np.random.set_state(checkpoint["numpy_rng"])
            random.setstate(checkpoint["python_rng"])
        write(a.out/"resume.json", dict(checkpoint=str(a.resume.resolve()),
            sha256=hashlib.sha256(a.resume.read_bytes()).hexdigest(), update=completed,
            model_and_optimizers_restored=True, simulator_episodes_reset=True,
            terrain_levels_restored="terrain_levels" in checkpoint,
            bit_exact_continuation=False))
    guard_stats = install_optimizer_guards(model, optimizers, alg, a.max_learning_rate,
        a.std_min, a.std_max) if a.stabilize else {}
    writer = SummaryWriter(str(a.out))
    started = time.monotonic()
    write(a.out/"progress.json", dict(status="running", algorithm=a.algorithm, completed_updates=completed,
        target=a.updates, pid=os.getpid(), heartbeat=time.time(), resumed_from=completed))
    print("RECOVERY_READY", a.algorithm, completed, a.updates, flush=True)
    for update in range(completed+1, a.updates+1):
        tick = time.monotonic()
        episode_stats = torch.zeros(3, device=device)
        with torch.no_grad():
            for _ in range(a.steps):
                history = adapter.history.flip(1).flatten(1)
                current = adapter.history[:, 0]
                privileged = wtw_m2.privileged(adapter.env) if wtw else adapter.critic
                action = alg.act(current, privileged, history)
                old_history = adapter.history
                rewards, done, info, successor = adapter.step(action)
                episode_stats += torch.stack([info[k] for k in
                    ("terminal_events", "finished_episodes", "finished_episode_seconds")])
                timeout = info.pop("time_outs")
                terminal_history = torch.cat((adapter.successor_actor[:, None], old_history[:, :-1]), 1).flip(1).flatten(1)
                if wtw:
                    terminal_value = model.evaluate(terminal_history, privileged).squeeze(-1)
                    info["env_bins"] = torch.zeros(a.num_envs, device=device)
                else:
                    terminal_value = torch.zeros(a.num_envs, device=device)
                    for idx, teacher in ((alg.teacher_env_idxs, True), (alg.student_env_idxs, False)):
                        terminal_value[idx] = model.evaluate(successor[idx], terminal_history[idx], teacher).squeeze(-1)
                rewards = timeout_bootstrap(rewards, terminal_value, timeout)
                alg.process_env_step(rewards, done, info)
            history = adapter.history.flip(1).flatten(1)
            if wtw:
                alg.compute_returns(history, wtw_m2.privileged(adapter.env))
            else:
                alg.compute_returns(adapter.critic, history)
        losses = alg.update()
        assert np.isfinite(losses).all(), "Non-finite training losses"
        assert torch.isfinite(model.std).all() and (model.std > 0).all(), "Invalid action distribution"
        torch.cuda.synchronize()
        elapsed = time.monotonic()-tick
        levels = adapter.env.scene.terrain.terrain_levels.float()
        result = dict(status="running", algorithm=a.algorithm, completed_updates=update, target=a.updates,
            pid=os.getpid(), heartbeat=time.time(),
            elapsed_s=time.monotonic()-started, update_seconds=elapsed,
            steps_per_second=a.num_envs*a.steps/elapsed, losses=list(map(float, losses)),
            mean_terrain_level=levels.mean().item(), max_terrain_level=levels.max().item(),
            learning_rate=alg.learning_rate, std_min=model.std.min().item(), std_max=model.std.max().item(),
            mean_episode_seconds=(episode_stats[2]/episode_stats[1].clamp(min=1)).item(),
            termination_fraction=(episode_stats[0]/episode_stats[1].clamp(min=1)).item(),
            allocated_gb=torch.cuda.max_memory_allocated()/2**30, checkpoint_validated=False,
            numerical_resets=adapter.numerical_resets, recovery_guards=guard_stats)
        write(a.out/"progress.json", result)
        if update % 10 == 0 or update <= 5:
            print(json.dumps(result), flush=True)
        for index, loss in enumerate(losses):
            writer.add_scalar(f"Loss/{index}", loss, update)
        writer.add_scalar("Terrain/mean", result["mean_terrain_level"], update)
        if update % 100 == 0 or update == a.updates:
            checkpoint = dict(model_state_dict=model.state_dict(),
                optimizer_states=[o.state_dict() for o in optimizers], update=update,
                learning_rate=alg.learning_rate, arguments=vars(a),
                common_step_counter=adapter.env.common_step_counter,
                rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                numpy_rng=np.random.get_state(), python_rng=random.getstate(),
                terrain_levels=adapter.env.scene.terrain.terrain_levels.cpu(),
                terrain_types=adapter.env.scene.terrain.terrain_types.cpu())
            require_finite(checkpoint["model_state_dict"], "checkpoint_model")
            require_finite(checkpoint["optimizer_states"], "checkpoint_optimizers")
            temp = a.out/"latest.tmp"
            torch.save(checkpoint, temp)
            temp.replace(a.out/"latest.pt")
            if update % 5000 == 0:
                torch.save(checkpoint, a.out/f"model_{update}.pt")
    result["status"] = "completed"
    write(a.out/"progress.json", result)
    writer.close()
    adapter.env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        # Persist a truthful failure status instead of leaving stale 'running'.
        import sys
        if "--out" in sys.argv:
            out = Path(sys.argv[sys.argv.index("--out")+1])
            if out.exists() and (out/"progress.json").exists():
                state = json.loads((out/"progress.json").read_text())
                if state.get("pid") == os.getpid():
                    state.update(status="stopped_error", error=str(exc), heartbeat=time.time())
                    write(out/"progress.json", state)
                    write(out/"ALERT.json", state)
        raise
