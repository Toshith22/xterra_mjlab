from copy import deepcopy

import pytest
from xterra_mjlab.tasks.velocity.config.svanm2.ref_fresh_abduction_gate import annotate_abduction


def sample(**kwargs):
    return dict(dict(stable_straight_hip_samples=500, hip_rms_error_deg=[14., 5., 4., 3.],
        episode_seconds=12., success=True, failure_reasons=[], quiet_fraction=1.), **kwargs)


def test_quiet_wider_stance_is_advisory_not_failure():
    row = sample()
    annotate_abduction(row)
    assert row["success"] and row["failure_reasons"] == []
    assert not row["hip_quality_pass"]
    assert row["posture_warnings"] == ["abduction_posture_reference_exceeded"]
    assert row["abduction_reference_only"] and row["abduction_gate_version"] == 2


@pytest.mark.parametrize("reason", ["physical_fall", "strict_angle_failure", "tracking", "quiet_standing",
                                  "scuff_mean", "slip_mean", "impact_mean", "incomplete_episode"])
def test_posture_exception_never_erases_functional_failure(reason):
    row = sample(success=False, failure_reasons=[reason])
    before = deepcopy(row)
    annotate_abduction(row)
    assert not row["success"] and row["failure_reasons"] == before["failure_reasons"]


def test_balance_correction_without_stable_window_is_not_automatic_failure():
    row = sample(stable_straight_hip_samples=40)
    annotate_abduction(row, pulse_family=1)
    assert row["success"]
    assert row["posture_warnings"] == ["abduction_insufficient_stable_samples"]


def test_normal_stance_has_no_warning_and_dt_is_respected():
    row = sample(hip_rms_error_deg=[4., 3., 2., 1.], stable_straight_hip_samples=1000)
    annotate_abduction(row, dt=.01)
    assert row["hip_quality_pass"] and row["posture_warnings"] == []
    assert row["stable_straight_fraction"] == 1.


def test_summary_keeps_posture_warnings_separate_from_success_and_failures():
    from xterra_mjlab.tasks.velocity.config.svanm2.ref_evaluation import summarize
    row = sample(verified_pair_event=False, physical_fall=False, strict_angle_failure=False)
    annotate_abduction(row)
    result = summarize([dict(scenario=dict(level=0, family="flat", task="stand", pulse_family=0),
        episodes=[row]*128, requested_trials=128)])
    assert result["cases"][0]["passed"]
    assert result["cases"][0]["failure_reasons"] == {}
    assert result["cases"][0]["posture_warnings"] == {"abduction_posture_reference_exceeded": 128}
    assert result["review_alerts"] == []
