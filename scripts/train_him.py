"""Pinned HIMLoco on native public M2; supports torchrun two-GPU training."""
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
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from mjlab.utils.os import dump_yaml
from xterra_mjlab.assets.svanm2.constants import SVANM2_XML
from him_backend import load, mean
from m2_algorithm_env import Adapter, make_cfg


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False, default=str)+"\n")
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--num-envs", type=int, default=2048, help="Per GPU")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--updates", type=int, default=20000)
    p.add_argument("--seed", type=int, default=101001)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    rank = int(os.environ.get("LOCAL_RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"
    if world > 1:
        dist.init_process_group("nccl", device_id=torch.device(device))
    torch.set_num_threads(int(os.environ.get("M2_TORCH_THREADS", "4")))
    torch.manual_seed(a.seed)
    np.random.seed(a.seed + rank)
    random.seed(a.seed + rank)
    xml_hash = hashlib.sha256(SVANM2_XML.read_bytes()).hexdigest()
    assert xml_hash == "2c37cc2f9eaa7be52a8c2312a1fffca779be08ce9a34406129bc2d2226527163"
    cfg = make_cfg(a.num_envs, a.seed + rank)
    if a.smoke:
        cfg.episode_length_s = .2
    if rank == 0:
        a.out.mkdir(parents=True, exist_ok=False)
        write(a.out / "manifest.json", dict(
            algorithm="HIMLoco/native-MuJoCo SvanM2 adaptation", arguments=vars(a),
            upstream_commit="ef289acaa62795009363b7b819c9186690630441",
            public_commit="522da8e02d08462be550db223ad4adc2d5ff338b",
            robot_xml_sha256=xml_hash, world_size=world,
            global_environments=a.num_envs*world, batch=a.num_envs*world*a.steps,
            initialized_from="scratch", deployable=False,
            differences=["Native public M2 dynamics/rewards/terrain curriculum, not IsaacGym Go1",
                "Public 45D observations; newest-first six-frame history; privileged native sensors",
                "True terminal successor targets; upstream target excludes command and includes velocity",
                "Distributed gradients synchronized before clipping; globally synchronized KL",
                "Rank-local Sinkhorn assignments and advantage normalization",
                "Public robot gains, torque limits and action scales unchanged"],
            targets={"stairs_m": [.20, .25], "speed_m_s": 2., "slope_degrees": 30},
            license_notice="HIMLoco upstream license applies; no public redistribution authorized"))
        dump_yaml(a.out/"env.yaml", asdict(cfg))
        source = Path(__file__).resolve().parents[1]
        shutil.copytree(source/"scripts", a.out/"source"/"scripts", ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copytree(source/"xterra_mjlab", a.out/"source"/"xterra_mjlab", ignore=shutil.ignore_patterns("__pycache__"))
        write(a.out/"source_hashes.json", {str(f.relative_to(a.source)): hashlib.sha256(f.read_bytes()).hexdigest()
              for f in a.source.rglob("*.py")})
    if world > 1:
        dist.barrier()
    adapter = Adapter(cfg, device)
    Model, Algorithm = load(a.source)
    model = Model(270, adapter.critic.shape[1], 45, 12).to(device)
    if world > 1:
        for value in model.state_dict().values():
            dist.broadcast(value, 0)
    alg = Algorithm(model, device=device, num_learning_epochs=5, num_mini_batches=4,
        entropy_coef=.01, learning_rate=1e-3, schedule="adaptive", gamma=.99, lam=.95)
    alg.init_storage(a.num_envs, a.steps, [270], [adapter.critic.shape[1]], [12])
    torch.manual_seed(a.seed + 1000*rank)
    writer = SummaryWriter(str(a.out)) if rank == 0 else None
    started = time.monotonic()
    timings = []
    for update in range(1, a.updates+1):
        tick = time.monotonic()
        episode_stats = torch.zeros(3, device=device)
        with torch.no_grad():
            for step in range(a.steps):
                actions = alg.act(adapter.actor, adapter.critic)
                rewards, done, info, successor = adapter.step(actions)
                episode_stats += torch.stack([info[k] for k in
                    ("terminal_events", "finished_episodes", "finished_episode_seconds")])
                # Bootstrapping uses the terminal value, not the pre-action value.
                timeout = info.pop("time_outs")
                rewards += alg.gamma * model.evaluate(successor).squeeze(-1) * timeout
                alg.process_env_step(rewards, done, info, successor)
            alg.compute_returns(adapter.critic)
        losses = alg.update()
        if not np.isfinite(losses).all():
            raise RuntimeError("Non-finite HIMLoco loss")
        with torch.no_grad():
            if not torch.isfinite(model.std).all() or (model.std <= 0).any():
                raise RuntimeError("Invalid policy action distribution")
            stats = torch.tensor(losses, device=device)
            stats = mean(stats).tolist()
            episode_stats = mean(episode_stats)
        torch.cuda.synchronize()
        elapsed = time.monotonic()-tick
        timings.append(elapsed)
        if rank == 0:
            levels = adapter.env.scene.terrain.terrain_levels.float()
            result = dict(status="running", completed_updates=update, target=a.updates,
                pid=os.getpid(), heartbeat=time.time(),
                elapsed_s=time.monotonic()-started, update_seconds=elapsed,
                steps_per_second=a.num_envs*world*a.steps/elapsed,
                mean_terrain_level=levels.mean().item(), max_terrain_level=levels.max().item(),
                loss=dict(zip(("value", "surrogate", "velocity", "contrastive"), stats)),
                learning_rate=alg.learning_rate, std_min=model.std.min().item(),
                std_max=model.std.max().item(), gpu_count=world,
                allocated_gb=torch.cuda.max_memory_allocated()/2**30,
                mean_episode_seconds=(episode_stats[2]/episode_stats[1].clamp(min=1)).item(),
                termination_fraction=(episode_stats[0]/episode_stats[1].clamp(min=1)).item(),
                checkpoint_validated=False)
            write(a.out/"progress.json", result)
            for k, v in result["loss"].items():
                writer.add_scalar("Loss/"+k, v, update)
            writer.add_scalar("Terrain/mean", result["mean_terrain_level"], update)
            writer.add_scalar("Performance/steps_per_second", result["steps_per_second"], update)
            print(json.dumps(result), flush=True)
            if update % 100 == 0 or update == a.updates:
                checkpoint = dict(model_state_dict=model.state_dict(),
                    optimizer_state_dict=alg.optimizer.state_dict(),
                    estimator_optimizer_state_dict=model.estimator.optimizer.state_dict(),
                    learning_rate=alg.learning_rate, update=update, arguments=vars(a),
                    common_step_counter=adapter.env.common_step_counter,
                    rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all())
                temp = a.out/"latest.tmp"
                torch.save(checkpoint, temp)
                temp.replace(a.out/"latest.pt")
                if update % 2000 == 0:
                    torch.save(checkpoint, a.out/f"model_{update}.pt")
        if world > 1:
            # Catch accidental estimator/PPO divergence early, before a long run.
            if update <= 2 or update % 100 == 0:
                for value in model.parameters():
                    reference = value.detach().clone()
                    dist.broadcast(reference, 0)
                    if not torch.allclose(value, reference, atol=1e-6, rtol=1e-6):
                        raise RuntimeError("Distributed model parameters diverged")
            dist.barrier()
    if rank == 0:
        result.update(status="completed", median_update_seconds=float(np.median(timings[2:] or timings)))
        write(a.out/"progress.json", result)
        writer.close()
    adapter.env.close()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
