"""Versioned task acceptance, separate from the PPO objective.

Windowed tracking/attitude measures allow periodic locomotion; quiet standing
remains a distinct requirement. Thresholds are engineering hypotheses tested
on development rollouts, not numerical limits prescribed by ETH or Unitree.
"""
from dataclasses import asdict, dataclass
import math
import torch


@dataclass(frozen=True)
class RecoverySpec:
    version: int = 2
    window_seconds: float = .6
    speed_error_mps: float = .15
    yaw_error_rad_s: float = .4
    tracking_fraction: float = .9
    angular_rms_rad_s: float = 1.
    angular_drift_rad_s: float = .4
    angular_peak_rad_s: float = 2.
    terrain_relative_tilt_deg: float = 20.
    mean_slip_cost: float = .04
    deadline_seconds: float = 3.
    pulse_seconds: float = .12


RECOVERY = RecoverySpec()


def scheduled_tick(seconds, dt):
    """Round requested start up to a control tick, tolerant of float32 roundoff."""
    return torch.ceil(seconds/dt-1e-4)


class MovingRecoveryWindow:
    def __init__(self, n, dt, device, spec=RECOVERY):
        self.spec, self.dt = spec, dt
        self.length = round(spec.window_seconds/dt)
        if self.length < 1 or abs(self.length*dt-spec.window_seconds) > 1e-6:
            raise ValueError("recovery window must contain whole control steps")
        self.history = torch.zeros(self.length, n, 8, device=device)
        self.total = torch.zeros(n, 8, device=device)
        self.cursor = 0

    def reset(self, ids):
        self.history[:, ids] = 0.
        self.total[ids] = 0.

    def update(self, speed_error, yaw_error, angular_xy, tilt, slip, eligible, unsafe):
        s = self.spec
        finite = (torch.isfinite(speed_error) & torch.isfinite(yaw_error)
                  & torch.isfinite(angular_xy).all(-1) & torch.isfinite(tilt) & torch.isfinite(slip))
        omega2 = angular_xy.square().sum(-1)
        safe = (finite & ~unsafe & (omega2 <= s.angular_peak_rad_s**2)
                & (tilt <= math.radians(s.terrain_relative_tilt_deg)))
        features = torch.stack((torch.ones_like(slip), safe.float(),
            (speed_error < s.speed_error_mps).float(), (yaw_error.abs() < s.yaw_error_rad_s).float(),
            slip, angular_xy[:, 0], angular_xy[:, 1], omega2), -1)
        features = torch.where((eligible & finite)[:, None], features, 0.)
        self.total += features-self.history[self.cursor]
        self.history[self.cursor].copy_(features)
        self.cursor = (self.cursor+1) % self.length
        mean = self.total/self.length
        return ((self.total[:, 0] >= self.length-.01) & (self.total[:, 1] >= self.length-.01)
                & (mean[:, 2] >= s.tracking_fraction) & (mean[:, 3] >= s.tracking_fraction)
                & (mean[:, 4] < s.mean_slip_cost) & (mean[:, 5:7].norm(dim=-1) < s.angular_drift_rad_s)
                & (mean[:, 7] <= s.angular_rms_rad_s**2))


def timeout_mask(initial_failure, bounds, fall, finish):
    """A simultaneous fall/solver/angle failure is never bootstrapped as timeout."""
    return finish & ~(initial_failure | bounds | fall)


def failure_reasons(row):
    failed = []
    for key in ("physical_fall", "strict_angle_failure", "out_of_bounds", "simulation_failure"):
        if row.get(key, False):
            failed.append(key)
    if not row.get("episode_complete", True): failed.append("incomplete_episode")
    if not row["crossed"]: failed.append("obstacle_not_crossed")
    for key in ("slip_mean", "scuff_mean", "impact_mean"):
        if not math.isfinite(row[key]) or row[key] >= .04: failed.append(key)
    if row["task"] != "stand" and (row["tracking_steps"] < 100 or row["tracking_fraction"] < .9):
        failed.append("tracking")
    if row["task"] != "move" and (row["quiet_steps"] < 140 or row["quiet_fraction"] < .9):
        failed.append("quiet_standing")
    if row["applied_pulses"]:
        recovery = row["post_pulse_recovery_seconds"]
        if recovery is None or not 0 <= recovery <= RECOVERY.deadline_seconds or row.get("missed_recoveries", 0):
            failed.append("recovery")
    return failed


def stalled_evaluations(evaluations, limit):
    if limit <= 0 or len(evaluations) < limit:
        return False
    tail = evaluations[-limit:]
    level = tail[-1]["current_level"]
    return all(e["current_level"] == level and
               (e["validated_level"] is None or e["validated_level"] < level) for e in tail)


def recovery_manifest():
    return asdict(RECOVERY)
