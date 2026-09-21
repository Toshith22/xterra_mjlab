"""Run one ISL V7 public-M2 block, evaluate, and stop at the review gate."""
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
from xterra_mjlab.tasks.velocity.config.svanm2.isl_native import IslEnv as ManagerBasedRlEnv, task
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.utils.os import dump_yaml
from xterra_mjlab.assets.svanm2.constants import SVANM2_XML
from xterra_mjlab.tasks.velocity.config.svanm2.isl_cfg import make_isl_cfg
from xterra_mjlab.tasks.velocity.config.svanm2.rl_cfg import svanm2_ppo_runner_cfg
from xterra_mjlab.tasks.velocity.config.svanm2.isl_observations import reflect


def write(path, data):
    path.write_text(json.dumps(data, indent=2, default=str, allow_nan=False)+"\n")


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=98001)
    p.add_argument("--num-envs", type=int, default=4096)
    p.add_argument("--updates", type=int, default=100)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    if not 1 <= a.updates <= 100:
        p.error("ISL review blocks contain at most 100 updates.")
    torch.set_num_threads(int(os.environ.get("M2_TORCH_THREADS", "6")))
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)
    assert torch.cuda.is_available(), "GPU required"
    assert hashlib.sha256(SVANM2_XML.read_bytes()).hexdigest() == "2c37cc2f9eaa7be52a8c2312a1fffca779be08ce9a34406129bc2d2226527163", "Public XML changed"
    cfg = make_isl_cfg(level=0)
    cfg.scene.num_envs = a.num_envs
    cfg.seed = a.seed
    if a.smoke:
        cfg.episode_length_s = .30
    agent = svanm2_ppo_runner_cfg()
    agent.logger = "tensorboard"
    agent.seed = a.seed
    agent.algorithm.learning_rate = 3e-4
    agent.obs_groups = {"actor": ("actor", "height"),
        "critic": ("actor", "priv_vel", "priv_height", "priv_feet", "dynamics", "height")}
    agent.max_iterations = a.updates
    agent.save_interval = 50
    agent.clip_actions = 6.
    a.out.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve().parents[1]
    shutil.copytree(source/"xterra_mjlab", a.out/"source"/"xterra_mjlab", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(__file__, a.out/"source"/"train_isl.py")
    shutil.copytree(source/"scripts",a.out/"source"/"scripts",ignore=shutil.ignore_patterns("__pycache__"))
    manifest = dict(formulation="frozen ISL V7 formulation port to public xterra_mjlab SvanM2",
        public_commit="522da8e02d08462be550db223ad4adc2d5ff338b", robot_xml_sha256=hashlib.sha256(SVANM2_XML.read_bytes()).hexdigest(),
        arguments=vars(a), gpu=os.environ.get("CUDA_VISIBLE_DEVICES"), initialization="scratch",
        robot_and_actuator="unchanged public xterra_mjlab files", level=0, deployable=False,
        reference="ISL V7: stock PPO, perceptive actor, asymmetric critic, move/stop/stand, heading, foot-state costs",
        adaptation_differences=["native joint/term-major history order; ISL feature scales and PPO unchanged",
            "native public M2 foot sites/body contact forces and MuJoCo dynamics",
            "M2 spawn 0.34 m, height target 0.32 m, public nominal hardware retained",
            "ISL dynamics perturbations around M2 nominal parameters",
            "native simulator step/reset/sensing timing; no cross-simulator numerical equivalence claimed"],
        artifact_scope="first 100-update training block; no validated stair capability or hardware deployment claim",
        source_hashes={str(f.relative_to(source)): hashlib.sha256(f.read_bytes()).hexdigest()
            for f in (source/"xterra_mjlab").rglob("*.py")})
    write(a.out/"manifest.json", manifest)
    dump_yaml(a.out/"env.yaml", asdict(cfg))
    dump_yaml(a.out/"agent.yaml", asdict(agent))
    env = ManagerBasedRlEnv(cfg, device="cuda:0")
    if a.smoke:
        task(env).env_cfg["fresh_experiment"]["smoke_episode_seconds"] = .30
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    train_cfg = asdict(agent)
    train_cfg["algorithm"]["symmetry_cfg"] = dict(
        data_augmentation_func="xterra_mjlab.tasks.velocity.config.svanm2.isl_observations.augment",
        use_data_augmentation=True,use_mirror_loss=True,mirror_loss_coeff=.01)
    dump_yaml(a.out/"agent.yaml", train_cfg)
    runner = MjlabOnPolicyRunner(wrapped, train_cfg, str(a.out), "cuda:0")
    obs = wrapped.get_observations()
    assert obs["actor"].shape[1] == 270 and obs["height"].shape[1] == 381
    assert all(torch.isfinite(v).all() for v in obs.values())
    twice = reflect(env,reflect(env,obs))
    assert all(torch.allclose(obs[k],twice[k]) for k in obs.keys()), "Reflection is not involutive"
    updates = 0
    original_update = runner.alg.update
    started = time.time()

    def update():
        nonlocal updates
        result = original_update()
        updates += 1
        s=task(env)
        with (a.out/"training_episodes.jsonl").open("a") as f:
            for row in s.completed_records:
                f.write(json.dumps(row,allow_nan=False)+"\n")
        s.completed_records.clear()
        if not all(np.isfinite(float(v)) for v in result.values()):
            raise RuntimeError("Nonfinite PPO loss")
        write(a.out/"progress.json", dict(completed_updates=updates, target=a.updates,
            elapsed_s=time.time()-started, level=0, status="running", smoke=a.smoke))
        return result

    runner.alg.update = update
    print("M2_PUBLIC_PORT_READY", {k: list(v.shape) for k,v in obs.items()}, flush=True)
    runner.learn(num_learning_iterations=a.updates, init_at_random_ep_len=False)
    runner.save(str(a.out/"model_review.pt"))
    model_hash=hashlib.sha256((a.out/"model_review.pt").read_bytes()).hexdigest()
    s=task(env)
    write(a.out/"training_ledger.json",s.ledger.report(updates))
    write(a.out/"task_diagnostics.json",dict(
        body_names=s.native.body_names,
        contact_primary_names=env.scene["isl_body_contact"].primary_names,
        foot_sites=s.feet_site_ids,shank_bodies=s.feet_indices,
        measured_foot_height=s._measure()["height"].mean(0).tolist(),
        normal_contact_force=s._measure()["force_normal"].mean(0).tolist()))
    if not a.smoke:
        from evaluate_isl_block import evaluate
        result=evaluate(env,wrapped,runner.get_inference_policy(device="cuda:0"),
                        seed=a.seed+12000,model_hash=model_hash)
        result["promotion_eligible"]=s.ledger.accept_evaluation(result,updates)
        write(a.out/"development_review.json",result)
        write(a.out/"review_ledger.json",s.ledger.report(updates))
    else:
        from check_isl_runtime import check
        write(a.out/"runtime_checks.json",check(env,wrapped,runner.get_inference_policy(device="cuda:0"),a.seed,model_hash))
    write(a.out/"progress.json", dict(completed_updates=updates, target=a.updates,
        elapsed_s=time.time()-started, level=0, status="review_required", smoke=a.smoke,
        reason="ISL 100-update gate: review acceptance and foot telemetry before continuation."))
    env.close()


if __name__ == "__main__":
    main()
