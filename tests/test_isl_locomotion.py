import math
import numpy as np
import pytest
import torch
from xterra_mjlab.tasks.velocity.config.svanm2.ref_fresh_rewards import (foot_costs, quiet_support, support_loss_pairs, support_margin,
                           heading_command, phase_aligned_contact_cost, reference_is_stopped,
                           abduction_costs)
from xterra_mjlab.tasks.velocity.config.svanm2.ref_fresh_curriculum import ProgressLedger, families, TASKS, LEVELS, FAMILIES, speed_limit
from xterra_mjlab.tasks.velocity.config.svanm2.ref_fresh_terrain import profile, query_surface, LENGTH, WIDTH
from xterra_mjlab.tasks.velocity.config.svanm2.ref_evaluation import summarize, assign_pair_evidence


def test_abduction_cost_relaxes_for_recovery_without_free_joint_loophole():
    q = torch.zeros(4, 12)
    q[:, [0, 3, 6, 9]] = .4
    v = torch.ones_like(q)
    command = torch.zeros(4, 3)
    command[2, 1] = .3
    heights = torch.zeros(4, 4)
    heights[3, 0] = .12
    pose, motion = abduction_costs(q, v, command, torch.tensor([0., 1., 0., 0.]), heights)
    assert (pose > 0).all() and (motion > 0).all()
    assert torch.allclose(pose[1:], pose[:1].expand(3)*.1, atol=1e-6)
    assert torch.allclose(motion[1:], motion[:1].expand(3)*.1, atol=1e-6)
    q[:, [1, 2, 4, 5, 7, 8, 10, 11]] = 50.
    assert torch.equal(pose, abduction_costs(q, v, command, torch.tensor([0., 1., 0., 0.]), heights)[0])


def test_static_splay_cannot_evade_abduction_pose_cost():
    q = torch.zeros(1, 12)
    q[0, 6] = -.5
    pose, motion = abduction_costs(q, torch.zeros_like(q), torch.zeros(1, 3), torch.zeros(1), torch.zeros(1, 4))
    assert pose.item() > 0 and motion.item() == 0
    q.zero_()
    pose, _ = abduction_costs(q, torch.zeros_like(q), torch.zeros(1, 3), torch.zeros(1), torch.zeros(1, 4))
    assert pose.item() == 0


def test_heading_command_corrects_drift_without_changing_command_shape():
    yaw=math.radians(20)
    q=torch.tensor([[math.cos(yaw/2),0.,0.,math.sin(yaw/2)]])
    command,error=heading_command(torch.tensor([0.]),torch.tensor([0.]),q)
    assert command.shape == error.shape == (1,)
    assert command.item() < 0 and abs(error.item()+yaw) < 1e-6


def test_internal_heading_correction_does_not_change_user_stop_state():
    reference=torch.zeros(2,3)
    effective=reference.clone(); effective[:,2]=torch.tensor([.1,-.2])
    assert reference_is_stopped(reference).all()
    assert not reference_is_stopped(effective).any()


def test_moving_orientation_penalty_is_terrain_relative_and_strengthened_softly():
    from xterra_mjlab.tasks.velocity.config.svanm2.ref_task import ReferenceTask as FreshLocomotionMixin
    env = object.__new__(FreshLocomotionMixin)
    env.env_cfg = {"fresh_experiment": {"posture_revision": {"moving_orientation_scale": .6}}}
    env._measure = lambda: {"orientation": torch.tensor([.1, .1]),
                            "stopped": torch.tensor([False, True])}
    assert torch.allclose(env._reward_fresh_orientation(), torch.tensor([.06, .1]))


def test_contact_symmetry_compares_half_cycle_not_same_instant():
    history=torch.zeros(30,1,4,dtype=torch.bool)
    # Alternating diagonal trot. Left/right mirror occurs two ticks later.
    pattern=torch.tensor([[1,0,0,1],[1,0,0,1],[0,1,1,0],[0,1,1,0]],dtype=torch.bool)
    for i in range(30): history[i,0]=pattern[i%4]
    ema=torch.zeros(1,2)
    cost=phase_aligned_contact_cost(history,0,(2,3),ema,1.)
    assert cost.item() == 0.


def costs(velocity, height=.03, contact=True, stopped=True, recovery=0., age=1., previous_speed=0.):
    v = torch.tensor(velocity, dtype=torch.float).reshape(1, 1, 3).expand(1, 4, 3)
    normal = torch.tensor([0., 0., 1.]).reshape(1, 1, 3).expand_as(v)
    return foot_costs(torch.full((1, 4), height), v, normal, torch.full((1, 4), contact),
        torch.zeros(1, 4, dtype=torch.bool), torch.full((1, 4), previous_speed), torch.full((1, 4), age),
        torch.tensor([stopped]), torch.tensor([recovery]), torch.tensor([True]))


def test_dragging_cost_remains_during_stop_and_recovery():
    for recovery in (0., 1.):
        c = costs([.5, 0., 0.], recovery=recovery)
        assert c["slip"].item() == 1.
        assert c["scuff"].item() > 0


def test_airborne_lowering_allowed_but_low_horizontal_scraping_not_free():
    lower = costs([0., 0., -.2], contact=False)
    drag = costs([.5, 0., -.2], contact=False)
    assert lower["slip"].item() == lower["scuff"].item() == lower["quiet"].item() == 0
    assert drag["scuff"].item() > 0


def test_hover_and_excess_lift_cannot_collect_contact_rewards():
    hover = costs([0., 0., 0.], height=.30, contact=False)
    assert hover["hover"].item() > 0 and hover["over_lift"].item() > 0
    assert all(value.item() >= 0 for value in hover.values())
    # A normal swing does not attract the delayed placement cost.
    assert costs([0., 0., -.1], contact=False, age=.2)["hover"].item() == 0
    assert costs([0., 0., 0.], contact=False, stopped=False, age=2.)["hover"].item() > 0


def test_touchdown_uses_preimpact_velocity_not_zero_postimpact_velocity():
    assert costs([0., 0., 0.], previous_speed=-2.)["impact"].item() == 9.
    assert costs([0., 0., 0.], previous_speed=-.2)["impact"].item() == 0


def test_slope_tangent_sliding_is_penalized_but_normal_lowering_is_not():
    n = torch.tensor([[[-.6, 0., .8]]])
    args = (torch.tensor([[.03]]), n*-.2, n, torch.tensor([[True]]), torch.tensor([[True]]),
            torch.zeros(1, 1), torch.zeros(1, 1), torch.tensor([True]), torch.zeros(1), torch.tensor([True]))
    assert foot_costs(*args)["slip"].item() < 1e-12
    assert foot_costs(args[0], torch.tensor([[[.8, 0., .6]]]), *args[2:])["slip"].item() > .99


def test_crouching_or_three_feet_cannot_pass_quiet_standing():
    contact = torch.ones(2, 4, dtype=torch.bool)
    contact[1, 3] = False
    assert not quiet_support(contact, torch.zeros(2, 4, 3), torch.zeros(2, 3),
        torch.zeros(2, 3), torch.tensor([.12, .3]), torch.zeros(2)).any()


def test_pair_events_require_clearance_unloading_and_other_support():
    h = torch.tensor([[.12, .12, .03, .03], [.12, .12, .12, .12], [.03, .03, .03, .03]])
    f = torch.tensor([[0., 0., 30., 30.], [0., 0., 0., 0.], [0., 0., 30., 30.]])
    events = support_loss_pairs(h, f)
    assert events[0, 0] and not events[1:].any()


def test_support_margin_distinguishes_inside_outside_and_two_contact_gait():
    feet = torch.tensor([[[.3, .2], [.3, -.2], [-.3, .2], [-.3, -.2]]]).expand(3, 4, 2)
    contacts = torch.ones(3, 4, dtype=torch.bool)
    contacts[2, 1:3] = False
    points = torch.tensor([[0., 0.], [.4, 0.], [0., 0.]])
    margin = support_margin(feet, contacts, points)
    assert torch.allclose(margin[:2], torch.tensor([.2, -.1]), atol=1e-6)
    assert torch.isnan(margin[2])


def fill(ledger, level, success=True, ceiling=True, disturbed=True):
    for family in families(level):
        for task in TASKS:
            for _ in range(ledger.minimum):
                ledger.add(level, family, task, success, ceiling=ceiling, disturbed=disturbed)


def test_curriculum_cannot_promote_on_missing_stand_or_ceiling_evidence():
    ledger = ProgressLedger(minimum=4, window=4)
    fill(ledger, 0, ceiling=False)
    assert not ledger.try_promote(100)
    fill(ledger, 0)
    assert ledger.try_promote(200) and ledger.level == 1
    fill(ledger, 1)
    fill(ledger, 0, success=False)
    assert not ledger.try_promote(300)  # old skill regression blocks promotion
    assert ledger.report(300)["reliable_1_5_mps"] is False


@pytest.mark.parametrize("row", [0, 3, 7, 8, 9])
def test_analytic_training_queries_match_physical_profiles(row):
    for col in range(7):
        x = np.array([1., 3.99, 4.01, 4.51, 5.91, 8.1, 12.])
        xy = torch.tensor(np.stack((row*LENGTH+x, np.full_like(x, col*WIDTH+WIDTH/2)), -1), dtype=torch.float64)
        z, n = query_surface(xy)
        assert np.allclose(z.numpy(), profile(x, row, col), atol=1e-6)
        assert torch.allclose(n.norm(dim=-1), torch.ones_like(z))


def test_empty_or_stand_only_evaluation_does_not_claim_speed():
    assert not summarize([])["reliable_1_5_mps_on_flat_in_this_suite"]
    assert summarize([])["highest_consecutively_validated_level"] is None


def test_final_curriculum_requires_all_terrain_speed():
    assert all(speed_limit(len(LEVELS)-1, f) == 1.5 for f in FAMILIES)
    assert LEVELS[7].stair_speed < LEVELS[8].stair_speed < LEVELS[9].stair_speed
    assert not summarize([])["reliable_1_5_mps_all_terrains"]


def test_mean_action_promotion_requires_complete_prior_level_evidence():
    ledger = ProgressLedger()
    evidence = dict(model_sha256="test", seed=83001, highest_consecutively_validated_level=0)
    assert not ledger.accept_evaluation({**evidence, "smoke": True}, 10)
    assert not ledger.accept_evaluation({**evidence, "model_sha256": None}, 20)
    assert ledger.accept_evaluation(evidence, 30)
    assert ledger.level == 1
    assert not ledger.accept_evaluation(evidence, 40)
    assert ledger.accept_evaluation({**evidence, "highest_consecutively_validated_level": 1}, 50)


def test_flat_speed_pass_does_not_certify_stairs_or_slopes():
    row = dict(success=True, verified_pair_event=False, strict_angle_failure=False, physical_fall=False)
    cases = [dict(scenario=dict(level=9, family="flat", task=t, pulse_family=p, speed=1.5),
                  requested_trials=128, episodes=[row]*128) for t in ("move", "stop") for p in (0, 1)]
    result = summarize(cases)
    assert result["reliable_1_5_mps_on_flat_in_this_suite"]
    assert not result["reliable_1_5_mps_all_terrains"]
    assert not result["reliable_1_5_mps_by_terrain"]["stairs_up"]


def test_native_reset_records_outgoing_episodes_once(monkeypatch):
    from types import SimpleNamespace
    from mjlab.envs import ManagerBasedRlEnv
    from xterra_mjlab.tasks.velocity.config.svanm2.isl_native import IslEnv
    env=object.__new__(IslEnv)
    env.episode_length_buf=torch.tensor([5,0,1,0])
    recorded=[]
    resets=[]
    state=SimpleNamespace(_env_indices=lambda ids:ids,
        _out_of_bounds=lambda:torch.zeros(4,dtype=torch.bool),
        _record_episodes=lambda ids,bounds:recorded.append(ids.tolist()))
    env.command_manager=SimpleNamespace(get_term=lambda name:SimpleNamespace(task=state))
    monkeypatch.setattr(ManagerBasedRlEnv,"_reset_idx",lambda self,ids:resets.append(ids.tolist()))
    env._reset_idx(torch.tensor([0,1,2]))
    assert recorded==[[0,2]]
    assert resets==[[0,1,2]]


def test_wrong_airborne_pair_cannot_pass_targeted_recovery_coverage():
    records = [{}, {}]
    assign_pair_evidence(records, [[False, False, True, False], [True, False, False, False]], 5)
    assert not records[0]["verified_pair_event"]
    assert records[1]["verified_pair_event"]
