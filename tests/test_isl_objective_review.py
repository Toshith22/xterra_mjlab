from copy import deepcopy

from xterra_mjlab.tasks.velocity.config.svanm2.ref_fresh_objective_review import assess_objectives
from xterra_mjlab.tasks.velocity.config.svanm2.ref_fresh_curriculum import ProgressLedger


def result(warnings=16, passed=True, hip=14., seed=123, version=8):
    return dict(evaluator_version=version, seed=seed, model_sha256="test",
        highest_consecutively_validated_level=0 if passed else None,
        cases=[dict(scenario=dict(level=0, family="flat", task="stand", pulse_family=0),
            trials=128, passed=passed, maximum_hip_rms_error_deg=hip,
            posture_warnings={"abduction_posture_reference_exceeded": warnings} if warnings else {})])


def test_primary_pass_with_secondary_warning_is_not_primary_failure_or_auto_acceptance():
    review = assess_objectives(result(), 0)
    assert review["primary_current_level_passed"]
    assert review["decision"] == "primary_met_secondary_review"
    assert review["secondary_review_required"] and not review["pause_recommended"]
    assert review["secondary_cases"][0]["acceptable"] is None
    assert not review["final_speed_goal_passed"]


def test_secondary_success_never_excuses_primary_failure():
    review = assess_objectives(result(warnings=0, passed=False), 0)
    assert review["decision"] == "primary_not_met"
    assert not review["primary_current_level_passed"]


def test_no_warning_does_not_claim_unmeasured_style_is_perfect():
    review = assess_objectives(result(warnings=0), 0)
    assert review["decision"] == "primary_met_no_measured_secondary_warning"
    assert not review["secondary_review_required"]


def test_repeated_warning_pauses_for_review_without_changing_primary_result():
    data = result()
    before = deepcopy(data)
    prior = assess_objectives(data, 0)
    review = assess_objectives(data, 0, [dict(objective_review=prior)])
    assert data == before
    assert review["pause_recommended"] and review["primary_current_level_passed"]
    assert review["secondary_cases"][0]["persistent_warnings"]


def test_worsening_prevalence_and_severity_are_recorded():
    prior = assess_objectives(result(warnings=16, hip=14.), 0)
    review = assess_objectives(result(warnings=40, hip=17.), 0, [dict(objective_review=prior)])
    assert len(review["secondary_cases"][0]["worsening"]) == 2


def test_changed_seed_protocol_trial_count_or_missing_history_is_not_false_trend():
    prior = [dict(objective_review=assess_objectives(result(), 0))]
    for data in (result(seed=124), result(version=9)):
        assert not assess_objectives(data, 0, prior)["pause_recommended"]
    data = result()
    data["cases"][0]["trials"] = 64
    assert not assess_objectives(data, 0, prior)["pause_recommended"]
    assert not assess_objectives(result(), 0, [dict(validated_level=0)])["pause_recommended"]


def test_resolved_warning_clears_review_requirement():
    prior = [dict(objective_review=assess_objectives(result(), 0))]
    assert not assess_objectives(result(warnings=0), 0, prior)["secondary_review_required"]


def test_review_hold_is_saved_without_promoting_or_falsifying_primary_pass():
    ledger = ProgressLedger()
    data = result()
    previous = assess_objectives(data, 0)
    data["objective_review"] = assess_objectives(data, 0, [dict(objective_review=previous)])
    assert not ledger.accept_evaluation(data, 500)
    assert ledger.level == 0
    assert ledger.development_evaluations[-1]["validated_level"] == 0
    report = ledger.report(500)
    assert report["secondary_review"]["pause_recommended"]
    assert report["promotion_blockers"]


def test_current_difficulty_does_not_pass_from_an_easier_level():
    assert not assess_objectives(result(warnings=0), 1)["primary_current_level_passed"]
