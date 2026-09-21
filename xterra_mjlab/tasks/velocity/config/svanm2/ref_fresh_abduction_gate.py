"""Posture reference is advisory; functional control failures remain binding.

Ten degrees from the nominal pose is a style/reference threshold, not a
manufacturer joint limit or a demonstrated stability boundary. Never reject
otherwise successful locomotion solely for exceeding it or for needing balance
correction. Reward-based soft posture/motion preferences remain unchanged.
"""


def abduction_gate_manifest():
    return dict(version=2, reference_rms_degrees=10., reference_only=True,
        minimum_stable_samples=100, clean_stable_fraction_reference=.9,
        hard_failures="existing falls, strict body-angle, tracking, slip/scuff/impact, quiet-support and completion checks",
        note="Posture warnings do not grant a safety exception or prove the wider stance is necessary.")


def annotate_abduction(row, pulse_family=0, dt=.02):
    """Annotate a fresh functional result without changing its success/reasons."""
    spec = abduction_gate_manifest()
    samples = row["stable_straight_hip_samples"]
    row["max_hip_rms_error_deg"] = max(row["hip_rms_error_deg"])
    row["stable_straight_fraction"] = samples/max(1., (row["episode_seconds"]-2.)/dt)
    assessable = samples >= spec["minimum_stable_samples"]
    within = row["max_hip_rms_error_deg"] <= spec["reference_rms_degrees"]
    stable_reference = pulse_family != 0 or row["stable_straight_fraction"] >= spec["clean_stable_fraction_reference"]
    # Retain this legacy field for comparison; it no longer determines success.
    row["hip_quality_pass"] = assessable and within and stable_reference
    row["abduction_gate_version"] = spec["version"]
    row["abduction_reference_only"] = True
    warnings = []
    if not assessable:
        warnings.append("abduction_insufficient_stable_samples")
    elif not within:
        warnings.append("abduction_posture_reference_exceeded")
    if not stable_reference:
        warnings.append("abduction_stable_window_reference_unmet")
    row["posture_warnings"] = warnings
