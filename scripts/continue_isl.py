"""Persistent ISL training, checkpointing and automatic functional gates."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.utils.os import dump_yaml
from xterra_mjlab.tasks.velocity.config.svanm2.isl_native import IslEnv,task
from xterra_mjlab.tasks.velocity.config.svanm2.isl_cfg import make_isl_cfg
from xterra_mjlab.tasks.velocity.config.svanm2.rl_cfg import svanm2_ppo_runner_cfg
from xterra_mjlab.tasks.velocity.config.svanm2.ref_fresh_objective_review import assess_objectives
from xterra_mjlab.tasks.velocity.config.svanm2.isl_observations import reflect


def write(path,data):
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False,default=str)+"\n")
    tmp.replace(path)


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def resume_state(parent,checkpoint):
    # Explicit completed-update metadata, not RSL's zero-based iteration index.
    progress=json.loads((parent/"progress.json").read_text())
    completed=int(progress["completed_updates"])
    ledger_path=parent/"review_ledger.json"
    if not ledger_path.exists(): ledger_path=parent/"training_ledger.json"
    ledger=json.loads(ledger_path.read_text())
    review_path=parent/"development_review.json"
    if progress.get("checkpoint_sha256"):
        if progress["checkpoint_sha256"]!=digest(checkpoint):
            raise ValueError("Parent checkpoint does not match its completed-run metadata")
    elif review_path.exists():
        review=json.loads(review_path.read_text())
        if review["model_sha256"]!=digest(checkpoint):
            raise ValueError("Parent checkpoint does not match its evaluation")
    return completed,ledger


def load_checkpoint(runner,path,completed):
    saved=torch.load(path,map_location="cuda:0",weights_only=True)
    runner.alg.load(saved,None,strict=True)
    runner.alg.learning_rate=runner.alg.optimizer.param_groups[0]["lr"]
    runner.current_learning_iteration=completed
    runner.env.unwrapped.common_step_counter=saved.get("infos",{}).get("env_state",{}).get("common_step_counter",0)
    return saved


def severe_regression(result):
    # Poor performance on newly unlocked terrain is expected during learning.
    # Stop only for gross collapse on previously validated clean terrain.
    prior=result.get("previous_validated_level",-1)
    rows=[c for c in result["cases"] if c["scenario"]["level"]<=prior
          and c["scenario"].get("pulse_family",0)==0]
    return any(c["trials"]>=32 and c["physical_falls"]/c["trials"]>=.5 for c in rows)


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--parent",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--seed",type=int,required=True)
    p.add_argument("--target",type=int,default=12000)
    p.add_argument("--num-envs",type=int,default=4096)
    p.add_argument("--eval-every",type=int,default=500)
    p.add_argument("--evaluate",action="store_true")
    p.add_argument("--level",type=int,default=0)
    a=p.parse_args()
    a.parent=a.parent.resolve()
    a.out=a.out.resolve()
    source=Path(__file__).resolve().parents[1]
    checkpoint=a.parent if a.evaluate else a.parent/"model_review.pt"
    if a.evaluate:
        completed,ledger=0,{}
    else:
        completed,ledger=resume_state(a.parent,checkpoint)
        if a.target<=completed or a.eval_every<1:
            p.error("Target must exceed completed updates; evaluation interval must be positive")
    torch.set_num_threads(int(os.environ.get("M2_TORCH_THREADS", "6")))
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)
    a.out.mkdir(parents=True,exist_ok=False)
    cfg=make_isl_cfg()
    cfg.seed=a.seed
    cfg.scene.num_envs=a.num_envs
    agent=svanm2_ppo_runner_cfg()
    agent.seed=a.seed
    agent.logger="tensorboard"
    agent.algorithm.learning_rate=3e-4
    agent.obs_groups={"actor":("actor","height"),
        "critic":("actor","priv_vel","priv_height","priv_feet","dynamics","height")}
    agent.clip_actions=6.
    agent.save_interval=100
    agent.max_iterations=a.target
    train_cfg=asdict(agent)
    train_cfg["algorithm"]["symmetry_cfg"]=dict(
        data_augmentation_func="xterra_mjlab.tasks.velocity.config.svanm2.isl_observations.augment",
        use_data_augmentation=True,use_mirror_loss=True,mirror_loss_coeff=.01)
    if not a.evaluate:
        subprocess.run(["sha256sum","--check","--quiet","UPSTREAM_SHA256SUMS"],cwd=source,check=True)
        # Task/robot changes require an explicit new experiment, not silent resume.
        previous=json.loads((a.parent/"manifest.json").read_text())
        for name,expected in previous["source_hashes"].items():
            if digest(source/name)!=expected:
                raise ValueError("Resume source mismatch: "+name)
        shutil.copytree(source/"xterra_mjlab",a.out/"source"/"xterra_mjlab",
            ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copytree(source/"scripts",a.out/"source"/"scripts",
            ignore=shutil.ignore_patterns("__pycache__"))
        write(a.out/"manifest.json",dict(**{k:v for k,v in previous.items() if k not in
            ("arguments","initialization","artifact_scope")},arguments=vars(a),
            initialization="resume actor, critic and optimizer; fresh simulator episodes",
            parent_checkpoint=str(checkpoint),parent_sha256=digest(checkpoint),
            start_update=completed,target=a.target,checkpoint_interval=100,evaluation_interval=a.eval_every,
            artifact_scope="continuous training; no automatic 100-update review stop",
            monitoring_policy="warnings advisory; functional gates retained; nonfinite/crash or repeated severe regression stops"))
        dump_yaml(a.out/"env.yaml",asdict(cfg))
        dump_yaml(a.out/"agent.yaml",train_cfg)
    env=IslEnv(cfg,device="cuda:0")
    wrapped=RslRlVecEnvWrapper(env,clip_actions=6.)
    runner=MjlabOnPolicyRunner(wrapped,train_cfg,None if a.evaluate else str(a.out),"cuda:0")
    load_checkpoint(runner,checkpoint,completed)
    s=task(env)
    s.ledger.level=ledger.get("unlocked_level",a.level)
    s.ledger.transitions=ledger.get("transitions",[])
    s.ledger.development_evaluations=ledger.get("development_evaluations",[])
    with torch.inference_mode(): env.reset()
    obs=wrapped.get_observations()
    assert all(torch.isfinite(x).all() for x in obs.values())
    twice=reflect(env,reflect(env,obs))
    assert all(torch.allclose(obs[k],twice[k]) for k in obs.keys())
    if a.evaluate:
        from evaluate_isl_block import evaluate
        result=evaluate(env,wrapped,runner.get_inference_policy(device="cuda:0"),
            seed=a.seed,model_hash=digest(checkpoint),through=a.level)
        write(a.out/"result.json",result)
        env.close()
        return
    updates=completed
    original_update=runner.alg.update
    started=time.time()
    bad_gates=0

    def progress(status,**extra):
        write(a.out/"progress.json",dict(completed_updates=updates,target=a.target,
            level=s.ledger.level,status=status,smoke=False,pid=os.getpid(),
            heartbeat=time.time(),elapsed_s=time.time()-started,**extra))

    def save():
        runner.current_learning_iteration=updates
        path=a.out/f"checkpoint_{updates:06d}.pt"
        runner.save(str(path),infos={"completed_updates":updates,"level":s.ledger.level})
        write(a.out/"training_ledger.json",s.ledger.report(updates))
        return path

    def update():
        nonlocal updates,bad_gates
        result=original_update()
        updates+=1
        if not all(np.isfinite(float(v)) for v in result.values()):
            raise RuntimeError("Nonfinite PPO loss")
        if any(not torch.isfinite(p).all() for p in runner.alg.get_policy().parameters()):
            raise RuntimeError("Nonfinite actor parameter")
        rows=s.completed_records
        with (a.out/"training_episodes.jsonl").open("a") as f:
            for row in rows: f.write(json.dumps(row,allow_nan=False)+"\n")
        if rows:
            write(a.out/"health.json",dict(update=updates,episodes=len(rows),
                physical_fall_fraction=sum(r["physical_fall"] for r in rows)/len(rows),
                mean_scuff=sum(r["scuff_mean"] for r in rows)/len(rows)))
        s.completed_records.clear()
        progress("running")
        path=save() if updates%100==0 or updates==a.target else None
        if updates%a.eval_every==0 or updates==a.target:
            path=path or save()
            progress("evaluating")
            evaluation=a.out/f"eval_{updates:06d}"
            with (a.out/f"eval_{updates:06d}.log").open("w") as log:
                subprocess.run([sys.executable,str(Path(__file__).resolve()),"--evaluate",
                    "--parent",str(path),"--out",str(evaluation),"--seed",str(a.seed+12000),
                    "--level",str(s.ledger.level),"--num-envs","512"],
                    cwd=source,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=7200)
            review=json.loads((evaluation/"result.json").read_text())
            review["objective_review"]=assess_objectives(review,s.ledger.level,s.ledger.development_evaluations)
            validated=[r["validated_level"] for r in s.ledger.development_evaluations if r.get("validated_level") is not None]
            review["previous_validated_level"]=max(validated,default=-1)
            bad_gates=bad_gates+1 if severe_regression(review) else 0
            # The user removed human pauses for ordinary style/proxy warnings.
            # Retain their unmodified report; only functional acceptance promotes.
            gate={**review,"objective_review":{**review["objective_review"],"pause_recommended":False}}
            review["promoted"]=s.ledger.accept_evaluation(gate,updates)
            write(a.out/"development_review.json",review)
            write(a.out/"review_ledger.json",s.ledger.report(updates))
            if bad_gates>=2:
                raise RuntimeError("Repeated severe fall regression on previously validated clean terrain")
            progress("running")
        return result

    runner.alg.update=update
    progress("running",resumed_from=completed)
    print("M2_CONTINUOUS_READY",dict(update=completed,level=s.ledger.level,target=a.target,
        optimizer_lr=runner.alg.learning_rate),flush=True)
    try:
        runner.learn(num_learning_iterations=a.target-completed,init_at_random_ep_len=False)
        runner.current_learning_iteration=updates
        runner.save(str(a.out/"model_review.pt"),infos={"completed_updates":updates,"level":s.ledger.level})
        progress("complete",checkpoint_sha256=digest(a.out/"model_review.pt"))
    except BaseException as exc:
        write(a.out/"ALERT.json",dict(update=updates,error=str(exc),timestamp=time.time()))
        progress("stopped_error",error=str(exc))
        raise
    finally:
        env.close()


if __name__=="__main__":
    main()
