"""Compare throughput at fixed PPO/task settings; benchmark weights are discarded."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np
import torch
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.runner import MjlabOnPolicyRunner
from xterra_mjlab.tasks.velocity.config.svanm2.isl_cfg import make_isl_cfg
from xterra_mjlab.tasks.velocity.config.svanm2.isl_native import IslEnv,task
from xterra_mjlab.tasks.velocity.config.svanm2.rl_cfg import svanm2_ppo_runner_cfg


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--threads",type=int,default=6)
    p.add_argument("--updates",type=int,default=30)
    a=p.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(99991)
    cfg=make_isl_cfg()
    cfg.seed=99991
    cfg.scene.num_envs=4096
    env=IslEnv(cfg,device="cuda:0")
    wrapped=RslRlVecEnvWrapper(env,clip_actions=6.)
    agent=svanm2_ppo_runner_cfg()
    agent.obs_groups={"actor":("actor","height"),
        "critic":("actor","priv_vel","priv_height","priv_feet","dynamics","height")}
    agent.algorithm.learning_rate=3e-4
    ac=asdict(agent)
    ac["algorithm"]["symmetry_cfg"]=dict(
        data_augmentation_func="xterra_mjlab.tasks.velocity.config.svanm2.isl_observations.augment",
        use_data_augmentation=True,use_mirror_loss=True,mirror_loss_coeff=.01)
    runner=MjlabOnPolicyRunner(wrapped,ac,None,"cuda:0")
    # Initialize buffers through the wrapper before the inference-mode reset.
    task(env).ledger.level=3
    with torch.inference_mode(): env.reset()
    times=[]
    original=runner.alg.update
    started=time.perf_counter()
    previous=started
    def update():
        nonlocal previous
        result=original()
        assert all(np.isfinite(float(v)) for v in result.values())
        task(env).completed_records.clear()
        torch.cuda.synchronize()
        now=time.perf_counter()
        times.append(now-previous)
        previous=now
        return result
    runner.alg.update=update
    runner.learn(num_learning_iterations=a.updates,init_at_random_ep_len=False)
    result=dict(threads=a.threads,num_envs=4096,updates=a.updates,
        seconds_per_update=float(np.median(times[5:])),
        environment_steps_per_second=4096*24/float(np.median(times[5:])),
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
        purpose="throughput-only probe; random seed and weights not used for training")
    a.out.write_text(json.dumps(result,indent=2)+"\n")
    print("BENCHMARK_RESULT",result,flush=True)
    env.close()


if __name__=="__main__": main()
