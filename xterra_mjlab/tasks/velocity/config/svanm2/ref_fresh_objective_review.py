"""Primary task acceptance and secondary-style review, not a new PPO loss.

Warnings never excuse primary failures. Repeated/worsening style warnings pause
for human analysis, not automatic rejection. A visual acceptance is conditional
evidence, never a guarantee about future learning or a permanent exemption.
"""
import json


def objective_manifest():
    return dict(version=2,
        primary=["tracking at current difficulty", "stop/resume and quiet support", "terrain crossing",
                 "disturbance recovery when required", "fall/angle/contact-slip checks; calibrated settled tracking and command response",
                 "complete task coverage; arena exits are censored rather than falls"],
        final_primary="held-out 1.5 m/s on flat, slopes and stairs; current-level success is not this goal",
        secondary_measured=["abduction posture reference", "availability of stable posture samples",
                            "near-ground swing and landing-speed proxies"],
        secondary_manual=["motion economy", "unnecessary joint activity", "symmetry/style under matched conditions",
                          "near-ground swing and landing-speed proxies; separate from confirmed contact sliding or damaging force"],
        trend_review=dict(consecutive_warning_gates=2, warning_fraction_increase=.10,
                          maximum_hip_rms_increase_degrees=2.),
        acceptance="human visual/telemetry review: acceptable with monitoring, investigate, or correct before continuation",
        changes_ppo_or_rewards=False)


def assess_objectives(result, current_level, history=()):
    spec = objective_manifest()
    validated = result.get("highest_consecutively_validated_level")
    primary_pass = validated is not None and validated >= current_level and not result.get("smoke", False)
    # Compare only the immediately preceding comparable evaluator/seed. Never
    # call a new terrain, changed protocol, or changed trial count a regression.
    prior = history[-1].get("objective_review", {}) if history else {}
    comparable = (prior.get("version") == spec["version"]
        and prior.get("evaluator_version") == result.get("evaluator_version")
        and prior.get("seed") == result.get("seed"))
    previous_cases = {c["case_key"]: c for c in prior.get("secondary_cases", [])} if comparable else {}
    secondary = []
    for case in result.get("cases", []):
        warnings = {**case.get("posture_warnings", {}), **case.get("motion_warnings", {})}
        if not warnings:
            continue
        key = json.dumps(case["scenario"], sort_keys=True)
        old = previous_cases.get(key, {})
        n = case["trials"]
        old = old if old.get("trials") == n else {}
        fractions = {k: count/max(1, n) for k, count in warnings.items()}
        repeated = sorted(set(fractions) & set(old.get("warning_fractions", {})))
        worsening = []
        for reason in repeated:
            if fractions[reason]-old["warning_fractions"][reason] >= spec["trend_review"]["warning_fraction_increase"]-1e-9:
                worsening.append(reason+": increased prevalence")
        hip = case.get("maximum_hip_rms_error_deg")
        old_hip = old.get("maximum_hip_rms_error_deg")
        if repeated and hip is not None and old_hip is not None and hip-old_hip >= spec["trend_review"]["maximum_hip_rms_increase_degrees"]:
            worsening.append("abduction: increased maximum offset")
        secondary.append(dict(case_key=key, scenario=case["scenario"], trials=n,
            review_trial_ids=list(dict.fromkeys(case.get("secondary_review_trial_ids", [])+case.get("motion_review_trial_ids", []))),
            primary_case_passed=bool(case["passed"]), warning_fractions=fractions,
            maximum_hip_rms_error_deg=hip, persistent_warnings=repeated, worsening=worsening,
            review_status="pending_human_review", acceptable=None,
            review_evidence_needed=["actual flagged trial video", "contacts/body stability/tracking telemetry",
                                    "comparison with earlier matched checkpoints", "possible future compensation or reward exploit"]))
    pause = any(c["persistent_warnings"] or c["worsening"] for c in secondary)
    return dict(version=spec["version"], evaluator_version=result.get("evaluator_version"), seed=result.get("seed"),
        current_level=current_level, primary_current_level_passed=primary_pass,
        final_speed_goal_passed=bool(result.get("reliable_1_5_mps_all_terrains", False)),
        secondary_cases=secondary, secondary_review_required=bool(secondary), pause_recommended=pause,
        decision=("primary_not_met" if not primary_pass else
                  "primary_met_secondary_review" if secondary else "primary_met_no_measured_secondary_warning"),
        note="Secondary review is not a primary failure or a prediction of future learning; manual qualities remain unscored.")
