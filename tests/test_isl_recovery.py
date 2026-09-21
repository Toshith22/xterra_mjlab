import math
import pickle
from pathlib import Path

import torch
import pytest

from xterra_mjlab.tasks.velocity.config.svanm2.ref_fresh_recovery import MovingRecoveryWindow, failure_reasons, timeout_mask, stalled_evaluations


def test_periodic_walking_passes_but_drift_violent_wobble_drag_and_falls_do_not():
    window = MovingRecoveryWindow(7, .02, "cpu")
    for tick in range(150):
        angular = torch.zeros(7, 2)
        angular[:, 0] = math.sin(tick*.02*2*math.pi*3)
        angular[1, 0] = .6  # sustained rotation, not a periodic gait
        angular[2, 0] *= 1.9  # too much angular energy even if signed mean cancels
        slip = torch.tensor([.01, .01, .01, .08, .01, .01, .01])
        error = torch.tensor([.03, .03, .03, .03, .3, .03, .03])
        tilt = torch.tensor([.05]*5+[math.radians(25), .05])
        result = window.update(error, torch.zeros(7), angular, tilt, slip,
            torch.ones(7, dtype=torch.bool), torch.tensor([False]*6+[True]))
    assert result.tolist() == [True, False, False, False, False, False, False]


def test_no_prepush_credit_and_reset_does_not_leak_between_episodes():
    window = MovingRecoveryWindow(2, .02, "cpu")
    def update(eligible):
        return window.update(torch.zeros(2), torch.zeros(2), torch.zeros(2, 2),
            torch.zeros(2), torch.zeros(2), torch.tensor(eligible), torch.zeros(2, dtype=torch.bool))
    for _ in range(50): assert not update([False, False]).any()
    for _ in range(29): assert not update([True, True]).any()
    assert update([True, True]).all()
    window.reset(torch.tensor([0]))
    assert update([True, True]).tolist() == [False, True]


def test_one_unsafe_or_nonfinite_frame_invalidates_entire_recovery_window():
    window = MovingRecoveryWindow(1, .02, "cpu")
    def update(omega=0.):
        return window.update(torch.zeros(1), torch.zeros(1), torch.tensor([[omega, 0.]]),
            torch.zeros(1), torch.zeros(1), torch.ones(1, dtype=torch.bool), torch.zeros(1, dtype=torch.bool))
    for _ in range(30): update()
    assert update().item()
    assert not update(3.).item()
    for _ in range(29): assert not update().item()
    assert update().item()
    assert not update(float("nan")).item()


def test_simultaneous_timeout_and_failure_never_bootstraps():
    result = timeout_mask(torch.tensor([False, True, False, False]),
        torch.tensor([False, False, True, False]), torch.tensor([False, False, False, True]),
        torch.ones(4, dtype=torch.bool))
    assert result.tolist() == [True, False, False, False]


def episode():
    return dict(task="move", crossed=True, slip_mean=.01, scuff_mean=.01, impact_mean=.001,
        tracking_steps=200, tracking_fraction=.99, quiet_steps=0, quiet_fraction=0.,
        applied_pulses=1, post_pulse_recovery_seconds=.8)


def test_strict_failure_prior_unrecovered_push_and_incomplete_episode_cannot_pass():
    assert not failure_reasons(episode())
    for key, value, reason in (("strict_angle_failure", True, "strict_angle_failure"),
            ("missed_recoveries", 1, "recovery"), ("post_pulse_recovery_seconds", None, "recovery"),
            ("episode_complete", False, "incomplete_episode"), ("simulation_failure", True, "simulation_failure")):
        assert reason in failure_reasons({**episode(), key: value})
    assert "quiet_standing" in failure_reasons({**episode(), "task": "stand"})


def test_repeated_gate_failures_request_review_but_promotion_does_not():
    fail = dict(current_level=1, validated_level=0)
    assert stalled_evaluations([fail, fail], 2)
    assert not stalled_evaluations([fail], 2)
    assert not stalled_evaluations([fail, dict(current_level=1, validated_level=1)], 2)
    assert not stalled_evaluations([fail, dict(current_level=2, validated_level=1)], 2)


def test_clean_walking_recovery_mismatch_is_reported_and_blocks_promotion():
    from xterra_mjlab.tasks.velocity.config.svanm2.ref_evaluation import summarize
    from xterra_mjlab.tasks.velocity.config.svanm2.ref_fresh_curriculum import ProgressLedger
    row = dict(success=True, verified_pair_event=False, strict_angle_failure=False,
        physical_fall=False, clean_walking_recovery_compatible=False)
    result = summarize([dict(scenario=dict(level=0, family="flat", task="move", pulse_family=0),
        requested_trials=128, episodes=[row]*128)])
    assert result["review_alerts"]
    ledger = ProgressLedger()
    assert not ledger.accept_evaluation(dict(model_sha256="test", seed=1,
        highest_consecutively_validated_level=0, review_alerts=result["review_alerts"]), 250)


def test_native_critic_does_not_leak_privileged_groups_to_actor():
    from pathlib import Path
    import ast
    tree=ast.parse((Path(__file__).resolve().parents[1]/"scripts/train_isl.py").read_text())
    assignment=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Attribute) and t.attr=="obs_groups" for t in n.targets))
    groups=ast.literal_eval(assignment.value)
    assert groups["actor"]==("actor","height")
    assert set(groups["critic"])=={"actor","height","priv_vel","priv_height","priv_feet","dynamics"}


@pytest.mark.parametrize("start", [1., 6.8, 4.02, 4.17])
@pytest.mark.parametrize("force_strength", [0., 15.])
def test_repeated_force_pulses_have_finite_duration_and_preserve_previous_failure(start, force_strength):
    from types import SimpleNamespace
    from xterra_mjlab.tasks.velocity.config.svanm2.ref_task import ReferenceTask as FreshLocomotionMixin
    class Fake(FreshLocomotionMixin):
        pass
    env = Fake()
    env.num_envs, env.dt, env.device = 1, .02, "cpu"
    env.episode_length_buf = torch.zeros(1)
    env.duration = torch.tensor([start+11.])
    env.repeated_pulses = torch.tensor([True])
    env.next_pulse_start = torch.tensor([float("inf")])
    env.pulse_start = torch.tensor([start])
    env.pulse_count = torch.zeros(1, dtype=torch.long)
    env.pulse_latched = torch.tensor([False])
    env.pulse_family = torch.tensor([1])
    env.single_foot = torch.tensor([0])
    env.terrain_level = torch.tensor([0])
    env._push_strengths, env._push_torques = torch.tensor([force_strength]), torch.tensor([2. if force_strength else 0.])
    env.pulse_direction = torch.tensor([[1., 0., 0.]])
    env.recovery_time = torch.tensor([-1.])  # first pulse never recovers
    env.recovery_dwell = torch.zeros(1)
    env.missed_recoveries = torch.zeros(1)
    env.last_pulse_active = torch.tensor([False])
    env.applied_impulse = torch.zeros(1)
    env.moving_recovery = MovingRecoveryWindow(1, .02, "cpu")
    env._rand = lambda shape: torch.full(shape, .5)
    env.apply_wrenches = lambda force, moment, leg: None
    env.robot = SimpleNamespace(get_link=lambda name: SimpleNamespace(idx=0))
    env.scene = SimpleNamespace(rigid_solver=SimpleNamespace(
        apply_links_external_force=lambda *args, **kwargs: None,
        apply_links_external_torque=lambda *args, **kwargs: None))
    for tick in range(round((start+10)/env.dt)):
        env.episode_length_buf[:] = tick
        env._apply_fresh_pulse()
    assert env.pulse_count.item() == (2 if force_strength else 0)
    assert env.missed_recoveries.item() == (1 if force_strength else 0)
    assert torch.allclose(env.applied_impulse, torch.tensor([force_strength*.24]), atol=1e-5)
    assert not env.last_pulse_active.item()
