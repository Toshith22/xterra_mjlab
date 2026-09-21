"""Versioned evaluation-only acceptance; does not change rewards or observations.

Thresholds are simulation engineering criteria, not hardware load limits.
Low swing/landing-speed proxies are warnings, not proof of scraping/damage.
"""
import math
import torch


def acceptance_manifest():
    return dict(version=2, velocity_window_seconds=.2, transition_grace_seconds=.5,
        tracking_tolerance_mps=.12, tracking_fraction=.9, steady_raw_error_rms_limit_mps=.25,
        response_deadline_seconds=2., moving_response_dwell_seconds=.3,
        stopped_response_dwell_seconds=.5, contact_slip_mean_limit=.04,
        clearance_and_landing_speed_proxies="advisory; not contact or damaging-load certificates",
        arena_exit="censored task coverage, not a fall; never grants full-suite pass",
        heading_tolerance_deg=10., heading_fraction=.9, maximum_heading_error_deg=25.,
        heading="quaternion yaw against integrated user command; hard primary path-direction gate",
        calibration_scope="simulation; synthetic controls and independent seed checks; not hardware validation")


class AcceptanceMonitor:
    def __init__(self, n, dt, device):
        self.n, self.dt, self.device = n, dt, device
        self.spec = acceptance_manifest()
        self.length = round(self.spec["velocity_window_seconds"]/dt)
        if self.length < 1 or abs(self.length*dt-self.spec["velocity_window_seconds"]) > 1e-6:
            raise ValueError("velocity window must contain whole control steps")
        self.history = torch.zeros(self.length, n, 2, device=device)
        self.total = torch.zeros(n, 2, device=device)
        self.command = torch.zeros(n, 3, device=device)
        self.cursor = 0
        for name in ("age", "segment_start", "dwell", "response_failures", "response_events",
                     "max_move_response", "max_stop_response", "steady_count", "steady_good",
                     "steady_error_sum", "steady_raw_error_sq", "terrain_count", "terrain_good",
                     "initial_yaw", "yaw", "heading_target", "max_heading_error", "yaw_rate_error_sq", "yaw_count",
                     "heading_count", "heading_good", "heading_error_sq"):
            setattr(self, name, torch.zeros(n, device=device))
        self.seen = torch.zeros(n, dtype=torch.bool, device=device)
        self.pending = torch.zeros_like(self.seen)
        self.invalid = torch.zeros_like(self.seen)

    def reset(self, ids):
        # Some accumulators are reassigned from torch.where/maximum while the
        # evaluator is in inference mode. Reset them under that same mode so a
        # multi-batch suite can safely reuse the monitor.
        with torch.inference_mode():
            for key, value in vars(self).items():
                if not isinstance(value, torch.Tensor):
                    continue
                if key == "history": value[:, ids] = 0
                else: value[ids] = 0

    def update(self, elapsed, command, velocity, quiet, quaternion, angular_velocity, on_terrain):
        spec = self.spec
        first = ~self.seen
        changed = first | ((command-self.command).abs().amax(-1) > .02)
        self.response_failures += (changed & self.pending).float()
        self.pending |= changed
        self.response_events += changed.float()
        self.segment_start[changed] = elapsed[changed]
        self.age[changed] = 0
        self.dwell[changed] = 0
        self.history[:, changed] = 0
        self.total[changed] = 0
        self.command.copy_(command)
        finite = (torch.isfinite(velocity).all(-1) & torch.isfinite(command).all(-1)
                  & torch.isfinite(quaternion).all(-1) & (quaternion.norm(dim=-1) > 1e-6)
                  & torch.isfinite(angular_velocity).all(-1))
        self.invalid |= ~finite
        safe_velocity = torch.nan_to_num(velocity[:, :2])
        self.total += safe_velocity-self.history[self.cursor]
        self.history[self.cursor].copy_(safe_velocity)
        self.cursor = (self.cursor+1) % self.length
        self.age += 1
        mean = self.total/self.age.clamp(max=self.length)[:, None]
        error = (mean-command[:, :2]).norm(dim=-1)
        raw_error_sq = (velocity[:, :2]-command[:, :2]).square().sum(-1)
        moving = command.norm(dim=-1) >= .02
        ready = (self.age >= self.length) & finite
        response_good = torch.where(moving, ready & (error < spec["tracking_tolerance_mps"]), quiet & finite)
        self.dwell = torch.where(response_good, self.dwell+self.dt, 0.)
        needed = torch.where(moving, spec["moving_response_dwell_seconds"], spec["stopped_response_dwell_seconds"])
        responded = self.pending & (self.dwell >= needed-1e-6)
        latency = elapsed-self.segment_start+self.dt
        # Late responses are not silently credited after the deadline.
        timely = responded & (latency <= spec["response_deadline_seconds"]+1e-6)
        self.max_move_response = torch.maximum(self.max_move_response, torch.where(timely & moving, latency, 0.))
        self.max_stop_response = torch.maximum(self.max_stop_response, torch.where(timely & ~moving, latency, 0.))
        overdue = self.pending & ~timely & (latency > spec["response_deadline_seconds"]+1e-6)
        self.response_failures += overdue.float()
        self.pending &= ~(timely | overdue)
        eligible = ready & moving & (elapsed > 2.) & (elapsed-self.segment_start >= spec["transition_grace_seconds"]-1e-6)
        self.steady_count += eligible
        self.steady_good += eligible & (error < spec["tracking_tolerance_mps"])
        self.steady_error_sum += torch.where(eligible, error, 0.)
        self.steady_raw_error_sq += torch.where(eligible, raw_error_sq, 0.)
        self.terrain_count += eligible & on_terrain
        self.terrain_good += eligible & on_terrain & (error < spec["tracking_tolerance_mps"])
        # Genesis quaternion order w,x,y,z. atan2 of projected forward axis is
        # independent of the parent environment's Euler-angle conversion.
        q = quaternion/quaternion.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        w,x,y,z = q.unbind(-1)
        yaw = torch.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
        self.initial_yaw[first] = yaw[first]
        self.heading_target[first] = yaw[first]
        self.heading_target += command[:, 2]*self.dt
        self.yaw.copy_(yaw)
        heading_error = torch.atan2(torch.sin(yaw-self.heading_target), torch.cos(yaw-self.heading_target))
        heading_eligible = finite & (elapsed > 2.) & (elapsed-self.segment_start >= spec["transition_grace_seconds"]-1e-6)
        self.max_heading_error = torch.maximum(self.max_heading_error,
            torch.where(heading_eligible, heading_error.abs(), 0.))
        self.heading_count += heading_eligible
        self.heading_good += heading_eligible & (heading_error.abs() <= math.radians(spec["heading_tolerance_deg"]))
        self.heading_error_sq += torch.where(heading_eligible, heading_error.square(), 0.)
        self.yaw_rate_error_sq += (angular_velocity[:, 2]-command[:, 2]).square()
        self.yaw_count += 1
        self.seen[:] = True

    def records(self, ids):
        drift = torch.atan2(torch.sin(self.yaw-self.initial_yaw), torch.cos(self.yaw-self.initial_yaw))
        values = torch.stack((self.steady_count, self.steady_good/self.steady_count.clamp_min(1),
            self.steady_error_sum/self.steady_count.clamp_min(1),
            (self.steady_raw_error_sq/self.steady_count.clamp_min(1)).sqrt(),
            self.response_events, self.response_failures, self.pending,
            self.max_move_response, self.max_stop_response, self.invalid,
            self.terrain_count, self.terrain_good/self.terrain_count.clamp_min(1),
            drift.rad2deg(), self.max_heading_error.rad2deg(), self.heading_count,
            self.heading_good/self.heading_count.clamp_min(1),
            (self.heading_error_sq/self.heading_count.clamp_min(1)).sqrt().rad2deg(),
            (self.yaw_rate_error_sq/self.yaw_count.clamp_min(1)).sqrt()), -1)[ids].detach().cpu().tolist()
        keys = ("steady_tracking_steps", "steady_tracking_fraction", "steady_speed_error_mps",
            "steady_raw_error_rms_mps", "command_response_events", "command_response_failures",
            "command_response_pending", "maximum_move_response_seconds", "maximum_stop_response_seconds",
            "nonfinite_measurements", "settled_terrain_tracking_steps", "settled_terrain_tracking_fraction",
            "quaternion_heading_change_deg", "maximum_command_heading_error_deg", "settled_heading_steps",
            "settled_heading_fraction", "settled_heading_error_rms_deg", "yaw_rate_tracking_rms_rad_s")
        return [dict(zip(keys, row)) for row in values]


def apply_acceptance(row):
    """Preserve legacy scores; distinguish primary faults, advisories and censoring."""
    spec = acceptance_manifest()
    row["legacy_failure_reasons"] = list(row["failure_reasons"])
    row["legacy_success"] = row["success"]
    failure, warnings = [], []
    for name in ("physical_fall", "strict_angle_failure", "simulation_failure", "nonfinite_measurements"):
        if row.get(name, False): failure.append(name)
    censored = bool(row.get("out_of_bounds", False))
    for name in ("slip_mean", "scuff_mean", "impact_mean"):
        if not math.isfinite(row[name]): failure.append(name+"_nonfinite")
        elif row[name] >= (spec["contact_slip_mean_limit"] if name == "slip_mean" else .04):
            (failure if name == "slip_mean" else warnings).append(name)
    if row["task"] != "stand":
        if row["steady_tracking_steps"] >= 100:
            if row["steady_tracking_fraction"] < spec["tracking_fraction"] or row["steady_raw_error_rms_mps"] > spec["steady_raw_error_rms_limit_mps"]:
                failure.append("settled_tracking")
        elif not censored: failure.append("insufficient_tracking_coverage")
    if row.get("settled_heading_steps", 0) >= 100:
        if (row.get("settled_heading_fraction", 0.) < spec["heading_fraction"]
                or row.get("maximum_command_heading_error_deg", float("inf")) > spec["maximum_heading_error_deg"]):
            failure.append("heading_tracking")
    elif not censored:
        failure.append("insufficient_heading_coverage")
    if row["command_response_failures"] or (row["command_response_pending"] and not censored):
        failure.append("command_response")
    if row["task"] != "move":
        if row["quiet_steps"] >= 140:
            if row["quiet_fraction"] < .9: failure.append("quiet_standing")
        elif not censored: failure.append("insufficient_quiet_coverage")
    if not row.get("episode_complete", True) and not censored: failure.append("incomplete_episode")
    if not row["crossed"] and not censored: failure.append("obstacle_not_crossed")
    if row["family"] != "flat" and row["task"] != "stand":
        if not censored and (row["settled_terrain_tracking_steps"] < 10 or row["settled_terrain_tracking_fraction"] < spec["tracking_fraction"]):
            failure.append("obstacle_tracking")
    if row["applied_pulses"] and "recovery" in row["legacy_failure_reasons"]:
        failure.append("recovery")
    row.update(acceptance_version=2, failure_reasons=list(dict.fromkeys(failure)), motion_warnings=warnings,
        evaluation_censored=censored, censor_reason="arena_boundary" if censored else None,
        success=not failure and not censored,
        acceptance_status="failed" if failure else "inconclusive" if censored else "passed")
