import math
from copy import deepcopy
import torch
from xterra_mjlab.tasks.velocity.config.svanm2.ref_fresh_acceptance import AcceptanceMonitor, apply_acceptance


def simulate(mode="normal", seconds=12., n=1):
    m=AcceptanceMonitor(n,.02,"cpu")
    for i in range(round(seconds/.02)):
        t=i*.02
        cmd=torch.tensor([[.65,0.,0.]]).repeat(n,1)
        vel=cmd.clone()
        if mode=="periodic": vel[:,1]=.18*math.sin(2*math.pi*5*t)
        if mode=="biased": vel[:,0]=.3
        if mode=="violent": vel[:,1]=1.*math.sin(2*math.pi*5*t)
        if mode=="nonfinite" and i==200: vel[:,0]=float("nan")
        q=torch.tensor([[1.,0.,0.,0.]]).repeat(n,1)
        m.update(torch.full((n,),t),cmd,vel,torch.ones(n,dtype=torch.bool),q,torch.zeros(n,3),torch.zeros(n,dtype=torch.bool))
    return m


def row(metrics=None):
    base=dict(task="move",family="flat",failure_reasons=["scuff_mean","impact_mean"],success=False,
        crossed=True,episode_complete=True,slip_mean=.01,scuff_mean=.08,impact_mean=.08,
        applied_pulses=0,quiet_steps=0,quiet_fraction=0.,out_of_bounds=False)
    base.update(metrics if metrics is not None else simulate().records(torch.tensor([0]))[0])
    return base


def test_normal_and_periodic_motion_pass_without_erasing_raw_proxy_scores():
    for mode in ("normal","periodic"):
        r=row(simulate(mode).records(torch.tensor([0]))[0]);apply_acceptance(r)
        assert r["success"]
        assert r["legacy_failure_reasons"]==["scuff_mean","impact_mean"]
        assert r["motion_warnings"]==["scuff_mean","impact_mean"]
        assert r["scuff_mean"]==.08


def test_sustained_bias_violent_oscillation_and_nonfinite_do_not_pass():
    for mode in ("biased","violent","nonfinite"):
        r=row(simulate(mode).records(torch.tensor([0]))[0]);apply_acceptance(r)
        assert not r["success"], mode
    assert simulate("biased").response_failures.item()==1


def test_sliding_falls_and_unrecovered_push_are_not_waived():
    for changes in ({"slip_mean":.1},{"physical_fall":True},{"strict_angle_failure":True},
                    {"applied_pulses":1,"failure_reasons":["recovery"]}, {"impact_mean":float("nan")}):
        r=row();r.update(changes);apply_acceptance(r);assert not r["success"]


def test_arena_censoring_does_not_grant_pass_or_invent_quiet_failure():
    r=row();r.update(task="stop",out_of_bounds=True,episode_complete=False,quiet_steps=0,
                    failure_reasons=["out_of_bounds","incomplete_episode","quiet_standing"])
    apply_acceptance(r)
    assert r["acceptance_status"]=="inconclusive" and not r["success"]
    assert not r["failure_reasons"]
    r=row();r.update(out_of_bounds=True,physical_fall=True);apply_acceptance(r)
    assert r["acceptance_status"]=="failed"


def test_filter_reset_does_not_leak_between_envs_or_episodes():
    m=simulate(n=2)
    previous=deepcopy(m.records(torch.tensor([1])))
    m.reset(torch.tensor([0]))
    assert not m.seen[0] and m.seen[1]
    assert m.steady_count[0]==0 and m.history[:,0].sum()==0
    assert m.records(torch.tensor([1]))==previous


def test_monitor_can_reset_after_inference_mode_batch():
    m=AcceptanceMonitor(1,.02,"cpu")
    with torch.inference_mode():
        m.update(torch.tensor([0.]),torch.tensor([[.4,0.,0.]]),torch.tensor([[.4,0.,0.]]),
                 torch.tensor([False]),torch.tensor([[1.,0.,0.,0.]]),torch.zeros(1,3),torch.tensor([False]))
    m.reset(torch.tensor([0]))
    assert not m.seen[0] and m.heading_count[0] == 0


def test_stop_resume_requires_each_response_and_rejects_no_stop():
    for bad in (False,True):
        m=AcceptanceMonitor(1,.02,"cpu")
        for i in range(950):
            t=i*.02
            stop=6<=t<9 or t>=14
            cmd=torch.tensor([[0. if stop else .65,0.,0.]])
            vel=cmd.clone()
            if bad and stop:vel[:,0]=.65
            # A plausible finite response, not instantaneous command following.
            if not bad and 9<=t<9.6:vel[:,0]=.65*(t-9)/.6
            quiet=torch.tensor([stop and not bad and (t>=6.6 if t<9 else t>=14.6)])
            m.update(torch.tensor([t]),cmd,vel,quiet,torch.tensor([[1.,0.,0.,0.]]),torch.zeros(1,3),torch.zeros(1,dtype=torch.bool))
        assert m.response_events.item()==4
        assert (m.response_failures.item()>0)==bad
        if not bad:assert 1.0<m.max_stop_response.item()<1.3


def test_quaternion_heading_change_is_a_hard_primary_gate():
    m=AcceptanceMonitor(1,.02,"cpu")
    for i in range(600):
        yaw=math.radians(45)*i/599
        q=torch.tensor([[math.cos(yaw/2),0.,0.,math.sin(yaw/2)]])
        m.update(torch.tensor([i*.02]),torch.tensor([[.65,0.,0.]]),torch.tensor([[.65,0.,0.]]),
                 torch.tensor([False]),q,torch.tensor([[0.,0.,math.radians(45)/12]]),torch.tensor([False]))
    r=row(m.records(torch.tensor([0]))[0]);apply_acceptance(r)
    assert abs(r["quaternion_heading_change_deg"]-45)<.001
    assert not r["success"]
    assert "heading_tracking" in r["failure_reasons"]


def test_integrated_turn_command_passes_when_quaternion_tracks_it():
    m=AcceptanceMonitor(1,.02,"cpu")
    rate=.2
    for i in range(600):
        yaw=rate*i*.02
        q=torch.tensor([[math.cos(yaw/2),0.,0.,math.sin(yaw/2)]])
        m.update(torch.tensor([i*.02]),torch.tensor([[.65,0.,rate]]),torch.tensor([[.65,0.,0.]]),
                 torch.tensor([False]),q,torch.tensor([[0.,0.,rate]]),torch.tensor([False]))
    r=row(m.records(torch.tensor([0]))[0]);apply_acceptance(r)
    assert r["success"]


def test_insufficient_completed_coverage_does_not_pass():
    r=row(simulate(seconds=.8).records(torch.tensor([0]))[0]);apply_acceptance(r)
    assert "insufficient_tracking_coverage" in r["failure_reasons"]


def test_diagnostic_control_never_promotes_even_with_forged_pass_summary():
    from xterra_mjlab.tasks.velocity.config.svanm2.ref_fresh_curriculum import ProgressLedger
    ledger=ProgressLedger()
    assert not ledger.accept_evaluation(dict(diagnostic_only=True,model_sha256="test",
                                            highest_consecutively_validated_level=0),1)
    assert ledger.level==0


def test_motion_proxy_warnings_are_retained_in_secondary_review():
    from xterra_mjlab.tasks.velocity.config.svanm2.ref_fresh_objective_review import assess_objectives
    result=dict(evaluator_version=9,seed=1,highest_consecutively_validated_level=0,
        cases=[dict(scenario=dict(level=0,family="flat",task="move",pulse_family=0),trials=128,passed=True,
                    motion_warnings={"scuff_mean":20},motion_review_trial_ids=[1,2,3])])
    review=assess_objectives(result,0)
    assert review["primary_current_level_passed"] and review["secondary_review_required"]
    assert review["secondary_cases"][0]["review_trial_ids"]==[1,2,3]
