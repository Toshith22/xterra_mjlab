"""Unchanged frozen ISL V7 evaluator aggregation (simulator independent)."""
import math
from collections import Counter
from .ref_fresh_curriculum import LEVELS,FAMILIES,families
from .ref_fresh_objective_review import objective_manifest

def assign_pair_evidence(records, flags, pulse_family):
    for row, pair in zip(records, flags):
        row["verified_pair_events"] = dict(zip(("front", "rear", "left", "right"), map(bool, pair)))
        row["verified_pair_event"] = bool(pair[pulse_family-5]) if pulse_family >= 5 else any(pair)


def wilson_lower(passed, total, z=1.96):
    if total < 1:
        return 0.
    p = passed/total
    return (p+z*z/(2*total)-z*math.sqrt(p*(1-p)/total+z*z/(4*total*total)))/(1+z*z/total)


def summarize(cases):
    results = []
    review_alerts = []
    for case in cases:
        rows = case["episodes"]
        successes = sum(r["success"] for r in rows)
        complete = len(rows) == case["requested_trials"]
        passed = complete and successes/len(rows) >= .95 and wilson_lower(successes, len(rows)) >= .90
        pairs = sum(r["verified_pair_event"] for r in rows)
        # Deliberately lifting a pair is a separate challenge from applying a
        # pulse. Require verified exposure before passing a pair-loss case.
        if case["scenario"].get("pulse_family", 0) >= 5:
            exposed = [r for r in rows if r["verified_pair_event"]]
            passed &= len(exposed) >= 32 and sum(r["success"] for r in exposed)/len(exposed) >= .95
        results.append(dict(scenario=case["scenario"], trials=len(rows), requested_trials=case["requested_trials"],
            successes=successes, success_rate=successes/len(rows) if rows else 0.,
            lower_95pct_confidence=wilson_lower(successes, len(rows)), passed=bool(passed),
            actual_pair_events=pairs, strict_angle_failures=sum(r["strict_angle_failure"] for r in rows),
            physical_falls=sum(r["physical_fall"] for r in rows),
            mean_actual_forward_mps=_mean(rows, "mean_actual_forward_mps"),
            terrain_mean_actual_forward_mps=_mean(rows, "terrain_mean_actual_forward_mps"),
            terrain_tracking_fraction=_mean(rows, "terrain_tracking_fraction"),
            mean_max_hip_rms_error_deg=_mean(rows, "max_hip_rms_error_deg"),
            maximum_hip_rms_error_deg=max((r["max_hip_rms_error_deg"] for r in rows if r.get("max_hip_rms_error_deg") is not None), default=None),
            mean_commanded_forward_mps=_mean(rows, "mean_commanded_forward_mps"),
            mean_tracking_error_mps=_mean(rows, "speed_error")))
        results[-1]["failure_reasons"] = dict(Counter(reason for r in rows for reason in r.get("failure_reasons", [])))
        results[-1]["motion_warnings"] = dict(Counter(reason for r in rows for reason in r.get("motion_warnings", [])))
        motion_flagged = sorted((r for r in rows if r.get("motion_warnings") and "env_id" in r),
                                key=lambda r: max(r.get("scuff_mean", 0.), r.get("impact_mean", 0.)))
        results[-1]["motion_review_trial_ids"] = list(dict.fromkeys(
            motion_flagged[i]["env_id"] for i in (0, len(motion_flagged)//2, len(motion_flagged)-1))) if motion_flagged else []
        results[-1]["censored_trials"] = sum(bool(r.get("evaluation_censored")) for r in rows)
        results[-1]["functional_failures"] = sum(bool(r.get("failure_reasons")) for r in rows)
        results[-1]["mean_settled_tracking_fraction"] = _mean(rows, "steady_tracking_fraction")
        results[-1]["mean_absolute_heading_change_deg"] = _mean(
            [{"heading": abs(r["quaternion_heading_change_deg"])} for r in rows if "quaternion_heading_change_deg" in r], "heading")
        results[-1]["posture_warnings"] = dict(Counter(reason for r in rows for reason in r.get("posture_warnings", [])))
        flagged = sorted((r for r in rows if r.get("posture_warnings") and "env_id" in r),
                         key=lambda r: r.get("max_hip_rms_error_deg", 0.))
        results[-1]["secondary_review_trial_ids"] = list(dict.fromkeys(
            flagged[i]["env_id"] for i in (0, len(flagged)//2, len(flagged)-1))) if flagged else []
        checked = [r for r in rows if r.get("clean_walking_recovery_compatible") is not None]
        if passed and checked and sum(r["clean_walking_recovery_compatible"] for r in checked)/len(checked) < .95:
            review_alerts.append(dict(scenario=case["scenario"],
                reason="recovery criterion rejects otherwise accepted clean walking; inspect before training further"))
    # Never infer 1.5 m/s from a command cap or from stand-only trials.
    required = {(task, pulse) for task in ("move", "stop") for pulse in (0, 1)}
    terrain_speed = {}
    terrain_sweep = {}
    for family in FAMILIES:
        speed_rows = [r for r in results if r["scenario"]["family"] == family and r["scenario"].get("speed") == 1.5]
        covered = {(r["scenario"]["task"], r["scenario"]["pulse_family"]) for r in speed_rows if r["passed"]}
        terrain_speed[family] = required <= covered
        sweep_required = {(speed, task, pulse) for speed in (.3, .65, 1., 1.2, 1.5) for task, pulse in required}
        sweep_covered = {(r["scenario"].get("speed"), r["scenario"]["task"], r["scenario"]["pulse_family"])
                         for r in results if r["scenario"]["family"] == family and r["passed"]}
        terrain_sweep[family] = sweep_required <= sweep_covered
    level_passes = []
    for level in range(len(LEVELS)):
        expected = {(f, t, p) for f in families(level) for t in ("move", "stop", "stand")
                    for p in ((0,) if level == 0 else (0, 1))}
        if level == len(LEVELS)-1:
            expected |= {("flat", "stand", p) for p in range(2, 9)}
        observed = {(r["scenario"]["family"], r["scenario"]["task"], r["scenario"]["pulse_family"])
                    for r in results if r["scenario"]["level"] == level and r["passed"]
                    and "speed" not in r["scenario"]}
        if not expected <= observed:
            break
        level_passes.append(level)
    return dict(cases=results, review_alerts=review_alerts, objective_hierarchy=objective_manifest(), reliable_1_5_mps_on_flat_in_this_suite=terrain_speed["flat"],
        reliable_1_5_mps_by_terrain=terrain_speed, reliable_1_5_mps_all_terrains=all(terrain_speed.values()),
        reliable_up_to_1_5_mps_by_terrain=terrain_sweep, reliable_up_to_1_5_mps_all_terrains=all(terrain_sweep.values()),
        highest_consecutively_validated_level=max(level_passes, default=None),
        note="Simulation suite result, not a hardware guarantee. Aggregate independent training/evaluation seeds separately.")


def _mean(rows, key):
    values = [r[key] for r in rows if r.get(key) is not None]
    return sum(values)/len(values) if values else None
