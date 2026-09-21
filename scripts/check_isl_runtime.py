"""Bounded integration assertions; never counts as policy acceptance."""
import torch
from xterra_mjlab.tasks.velocity.config.svanm2.isl_native import task
from evaluate_isl_block import evaluate


@torch.inference_mode()
def check(env,wrapped,policy,seed,model_hash):
    s=task(env)
    assert [n.split("/")[-1] for n in env.scene["isl_body_contact"].primary_names] == [n.split("/")[-1] for n in s.native.body_names]
    assert env.step_dt == .02
    s.env_cfg["fresh_experiment"]["smoke_episode_seconds"]=3.
    env.reset()
    for _ in range(100):
        wrapped.step(torch.zeros(env.num_envs,12,device=env.device))
    s._foot_cache=None
    contact=s._measure()
    assert contact["force_normal"].sum(-1).mean()>30., "Ground force must support robot weight"
    assert contact["contact"].float().mean()>.5, "Resting feet are not recognized"
    resting_force=contact["force_normal"].mean(0).tolist()
    s.env_cfg["fresh_experiment"]["smoke_episode_seconds"]=.30
    s.ledger.level=3
    env.reset()
    # All terrain families and widths are available at their intended levels.
    assert s.terrain_level.max()<=3
    assert ((s.base_pos[:,1]%6.)>2.89).all() and ((s.base_pos[:,1]%6.)<3.11).all()
    assert torch.isfinite(s._measure()["height"]).all()
    s.terrain_level[:]=3
    s.pulse_family[:]=torch.arange(env.num_envs,device=env.device)%9
    s.pulse_start[:]=0.
    s.pulse_latched[:]=False
    s.pulse_count[:]=0
    s.duration[:]=12.
    s._apply_fresh_pulse()
    assert torch.equal(s.pulse_count>0,s.pulse_family>0)
    counts=s.pulse_count.clone()
    s._apply_fresh_pulse()
    assert torch.equal(s.pulse_count,counts), "Pulse counted twice"
    s.terrain_level[:]=0
    s.pulse_latched[:]=False
    s._apply_fresh_pulse()
    assert not s.last_pulse_active.any(), "Zero-amplitude disturbance credited"
    s.ledger.level=0
    result=evaluate(env,wrapped,policy,seed+12000,model_hash,trials=8)
    result["smoke"]=True
    assert not s.ledger.accept_evaluation(result,2), "Smoke promoted curriculum"
    return dict(passed=True,pulse_families=9,zero_force_credit=False,
        resting_force_per_foot_n=resting_force,
        contact_order_verified=True,evaluation_cases=len(result["cases"]),
        evaluation_trials=sum(c["trials"] for c in result["cases"]),promotion=False)
