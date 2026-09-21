"""Fresh-task adapter: joint objectives, measured foot motion, honest progression.

No parent weights, simulator-force flags, terrain labels or oracle contacts are
fed to the actor. Difficulty/task/push state is for training and reporting only.
"""
import math
import torch
from .ref_fresh_curriculum import LEVELS, FAMILIES, TASKS, ProgressLedger, families, speed_limit
from .ref_fresh_rewards import (foot_costs, quiet_support, support_loss_pairs, support_margin,
                           abduction_costs, quaternion_yaw, heading_command,
                           phase_aligned_contact_cost, reference_is_stopped)
from .ref_fresh_terrain import COLUMN_FAMILY, LENGTH, WIDTH, query_surface, tread_width
from .ref_stair_sensing import rotation_matrix
from .ref_fresh_recovery import MovingRecoveryWindow, RECOVERY, failure_reasons, timeout_mask, scheduled_tick


class ReferenceTask:
    def terrain_heights(self, xy):
        return query_surface(xy)[0]

    def _init_terrain_buffers(self):
        n, device = self.num_envs, self.device
        self.moving_recovery = MovingRecoveryWindow(n, self.dt, device)
        self.next_pulse_start = torch.full((n,), float("inf"), device=device)
        self.pulse_latched = torch.zeros(n, dtype=torch.bool, device=device)
        self.missed_recoveries = torch.zeros(n, device=device)
        self.worst_recovery_time = torch.full((n,), -1., device=device)
        self.repeated_pulses = torch.zeros(n, dtype=torch.bool, device=device)
        self.last_pulse_active = torch.zeros(n, dtype=torch.bool, device=device)
        self.applied_impulse = torch.zeros(n, device=device)
        self.clean_recovery_windows = torch.zeros(n, device=device)
        if self.env_cfg["fresh_experiment"].get("policy_backbone") == "rsl_height":
            from .ref_stair_sensing import HeightSensorSurrogate, FEATURE_DIM
            self.height_sensor = HeightSensorSurrogate(n, device, self.env_cfg["fresh_experiment"]["seed"]+71000)
            self.height_observation = torch.zeros(n, FEATURE_DIM, device=device)
        self._speed_limits = torch.tensor([[speed_limit(l, f) for f in FAMILIES] for l in range(len(LEVELS))], device=device)
        self._push_strengths = torch.tensor([d.push_newtons for d in LEVELS], device=device)
        self._push_torques = torch.tensor([d.torque_nm for d in LEVELS], device=device)
        self.hip_samples = torch.zeros(n, device=device)
        self.hip_error_sq = torch.zeros(n, 4, device=device)
        self.hip_velocity_sq = torch.zeros(n, 4, device=device)
        self.ledger = ProgressLedger()
        self.fresh_update = 0
        self.fresh_rng = torch.Generator(device=device).manual_seed(self.env_cfg["fresh_experiment"]["seed"]+31000)
        self.eval_scenario = None
        self.completed_records = []
        self._fresh_reset = False
        self._foot_cache = None
        self.foot_previous = torch.zeros(n, 4, 3, device=device)
        self.foot_valid = torch.zeros(n, dtype=torch.bool, device=device)
        self.contact_previous = torch.zeros(n, 4, dtype=torch.bool, device=device)
        self.normal_speed_previous = torch.zeros(n, 4, device=device)
        self.air_age = torch.zeros(n, 4, device=device)
        self.support_age = torch.zeros(n, 4, device=device)
        self.pair_duration = torch.zeros(n, 4, device=device)
        self.pair_seen = torch.zeros(n, 4, dtype=torch.bool, device=device)
        self.nominal_commands = torch.zeros(n, 3, device=device)
        self.reference_commands = torch.zeros(n, 3, device=device)
        self.reference_heading = torch.zeros(n, device=device)
        self.heading_error = torch.zeros(n, device=device)
        self.phase_contact_lags = tuple(range(round(.16/self.dt), round(.50/self.dt)+1, 2))
        self.phase_contact_history = torch.zeros(round(.52/self.dt)+1, n, 4,
                                                 dtype=torch.bool, device=device)
        self.phase_contact_cursor = 0
        self.phase_contact_steps = torch.zeros(n, dtype=torch.long, device=device)
        self.phase_contact_ema = torch.zeros(n, len(self.phase_contact_lags), device=device)
        self.phase_contact_cost = torch.zeros(n, device=device)
        self.phase_contact_gate = torch.zeros(n, dtype=torch.bool, device=device)
        self.phase_contact_sum = torch.zeros(n, device=device)
        self.phase_contact_count = torch.zeros(n, device=device)
        for name in ("task", "family", "pulse_family", "single_foot", "pulse_count", "metric_steps", "tracking_steps", "quiet_steps"):
            setattr(self, name, torch.zeros(n, dtype=torch.long, device=device))
        for name in ("duration", "stop_time", "stop_until", "speed_error", "tracking_good", "quiet_good",
                     "slip_sum", "scuff_sum", "impact_sum", "max_roll", "max_pitch", "pulse_start", "recovery_dwell", "recovery_time"):
            setattr(self, name, torch.zeros(n, device=device))
        self.pulse_direction = torch.zeros(n, 3, device=device)
        self.at_ceiling = torch.zeros(n, dtype=torch.bool, device=device)
        self.ever_fall = torch.zeros(n, dtype=torch.bool, device=device)
        self.ever_strict = torch.zeros(n, dtype=torch.bool, device=device)
        self.recent_episodes = 0

    def _rand(self, shape):
        return torch.rand(shape, generator=self.fresh_rng, device=self.device)

    def _resample_commands(self, envs_idx):
        # This experiment owns command timing; parent periodic resampling must
        # not silently turn a stand episode into walking.
        return

    def _sample_tiles_for_levels(self, levels):
        result = torch.zeros_like(levels, dtype=self._env_tile_idx.dtype)
        for level in levels.unique().tolist():
            mask = levels == level
            allowed = [i for i, f in enumerate(COLUMN_FAMILY) if FAMILIES[f] in families(level)]
            if self.eval_scenario is not None:
                allowed = [i for i in allowed if FAMILIES[COLUMN_FAMILY[i]] == self.eval_scenario["family"]]
            if not allowed:
                raise ValueError("evaluation terrain not present at level")
            indices = (self._rand((int(mask.sum()),))*len(allowed)).long()
            result[mask] = (level*len(COLUMN_FAMILY)+torch.tensor(allowed, device=self.device)[indices]).to(result.dtype)
        return result

    def _update_terrain_curriculum(self, env_ids):
        # Promotion is coordinated across task/terrain strata in the trainer.
        # Preserve the outgoing level until its completed episode is recorded.
        return

    def _reset_idx(self, envs_idx=None):
        ids = self._env_indices(envs_idx).long()
        if not len(ids):
            return
        if self.eval_scenario is None:
            # 60% frontier, 40% all previously unlocked levels. Never drop
            # standing, stopping, flat or easier terrain from later training.
            frontier = min(self.ledger.level, self._max_level)
            level = torch.full((len(ids),), frontier, dtype=torch.long, device=self.device)
            earlier = self._rand((len(ids),)) < .4
            level[earlier] = (self._rand((int(earlier.sum()),))*(frontier+1)).long()
        else:
            level = torch.full((len(ids),), self.eval_scenario["level"], device=self.device, dtype=torch.long)
        self.terrain_level[ids] = level
        self.native_reset(ids)
        col = self._env_tile_idx[ids].long()%len(COLUMN_FAMILY)
        self.family[ids] = torch.tensor(COLUMN_FAMILY, device=self.device)[col]
        draw = self._rand((len(ids),))
        self.task[ids] = torch.where(draw < .2, 2, torch.where(draw < .5, 1, 0))
        if self.eval_scenario is not None:
            self.task[ids] = TASKS.index(self.eval_scenario["task"])
        limits = self._speed_limits[level, self.family[ids]]
        self.at_ceiling[ids] = self._rand((len(ids),)) < .5
        speed = limits*torch.where(self.at_ceiling[ids], 1., .5+.5*self._rand((len(ids),)))
        if self.eval_scenario is not None:
            speed[:] = self.eval_scenario.get("speed", float(limits[0]))
            self.at_ceiling[ids] = speed >= limits-.01
        self.nominal_commands[ids] = 0.
        self.nominal_commands[ids, 0] = speed
        # Small lateral/turn examples on flat ground; ceiling trials remain
        # straight, so turning cannot dilute the forward-speed measurement.
        turning = (self.family[ids] == 0) & ~self.at_ceiling[ids]
        self.nominal_commands[ids, 1] = (self._rand((len(ids),))-.5)*.3*turning
        self.nominal_commands[ids, 2] = (self._rand((len(ids),))-.5)*.6*turning
        if self.eval_scenario is not None:
            self.nominal_commands[ids, 1:] = 0.
        self.nominal_commands[ids[self.task[ids] == 2]] = 0.
        # Vary initial distance and stand on the flight itself in half of
        # standing terrain trials. This breaks a fixed approach/stride phase.
        x = 1.+self._rand((len(ids),))
        on_terrain = (self.task[ids] == 2) & (self.family[ids] > 0) & (self._rand((len(ids),)) < .5)
        x[on_terrain] = 4.2+self._rand((int(on_terrain.sum()),))*1.2
        pos = self.base_pos[ids].clone()
        pos[:, 0] = level*LENGTH+x
        pos[:, 1] = col*WIDTH+WIDTH/2+(self._rand((len(ids),))-.5)*.2
        pos[:, 2] = self.terrain_heights(pos[:, :2])+.34
        self.base_pos[ids] = pos
        self._env_spawn_xy[ids] = pos[:, :2]
        qpos = torch.zeros(len(ids), self.init_qpos.numel(), device=self.device)
        qpos[:, :3], qpos[:, 3:7], qpos[:, 7:] = pos, self.init_base_quat, self.init_dof_pos
        qpos[:, 7:] += (self._rand((len(ids), self.num_actions))-.5)*.06
        self.robot.set_qpos(qpos, envs_idx=ids.to(self._env_tile_idx.dtype), zero_velocity=True, skip_forward=True)
        walking_duration = torch.where(self.family[ids] == 0, 12., ((10.-x)/speed+2.).clamp(12., 30.))
        self.duration[ids] = torch.where(self.task[ids] == 2, 12., walking_duration)
        # Stop trials alternate mid-flight pauses and end-of-flight holds;
        # resume after a pause, including a final braking/quiet interval.
        self.stop_time[ids] = torch.where(self.family[ids] == 0, 6., (5.-x)/speed).clamp(3., 15.)
        self.stop_until[ids] = self.stop_time[ids]+3.
        self.duration[ids] += (self.task[ids] == 1)*7.
        self.duration[ids] = self.duration[ids].clamp(max=39.)
        smoke_seconds = self.env_cfg["fresh_experiment"].get("smoke_episode_seconds")
        if smoke_seconds is not None:
            self.duration[ids] = smoke_seconds
        self.pulse_family[ids] = (self._rand((len(ids),))*9).long()
        self.single_foot[ids] = (self._rand((len(ids),))*4).long()
        self.pulse_family[ids[(level == 0) | (self._rand((len(ids),)) < .35)]] = 0
        if self.eval_scenario is not None:
            self.pulse_family[ids] = self.eval_scenario.get("pulse_family", 0)
        theta = self._rand((len(ids),))*2*math.pi
        self.pulse_direction[ids] = torch.stack((theta.cos(), theta.sin(), torch.zeros_like(theta)), -1)
        self.pulse_start[ids] = torch.where(self.task[ids] == 1, self.stop_time[ids]+.8, 3.+self._rand((len(ids),))*2.)
        self.pulse_start[ids] = scheduled_tick(self.pulse_start[ids], self.dt)*self.dt
        repeat = self.env_cfg["fresh_experiment"].get("repeated_pushes", False) if self.eval_scenario is None else self.eval_scenario.get("repeat_pulses", False)
        self.repeated_pulses[ids] = repeat
        self.next_pulse_start[ids] = float("inf")
        self.pulse_latched[ids] = False
        self.last_pulse_active[ids] = False
        self.missed_recoveries[ids] = 0.
        self.worst_recovery_time[ids] = -1.
        self.applied_impulse[ids] = 0.
        self.clean_recovery_windows[ids] = 0.
        self.moving_recovery.reset(ids)
        if hasattr(self, "height_sensor"):
            self.height_sensor.reset(ids)
            self.height_observation[ids] = 0.
        for name in ("pulse_count", "metric_steps", "tracking_steps", "quiet_steps", "speed_error", "tracking_good",
                     "quiet_good", "slip_sum", "scuff_sum", "impact_sum", "max_roll", "max_pitch", "recovery_dwell"):
            getattr(self, name)[ids] = 0
        self.recovery_time[ids] = -1.
        for name in ("foot_valid", "contact_previous", "pair_seen", "ever_fall", "ever_strict"):
            getattr(self, name)[ids] = False
        for name in ("air_age", "support_age", "pair_duration", "normal_speed_previous"):
            getattr(self, name)[ids] = 0
        self.commands[ids] = self.nominal_commands[ids]
        self.reference_commands[ids] = self.nominal_commands[ids]
        # Native qpos writes defer derived kinematics until forward(). The
        # reset quaternion above is identity; do not read outgoing yaw here.
        self.reference_heading[ids] = 0.
        self.heading_error[ids] = 0.
        self.phase_contact_history[:, ids] = False
        self.phase_contact_steps[ids] = 0
        self.phase_contact_ema[ids] = 0.
        self.phase_contact_cost[ids] = 0.
        self.phase_contact_gate[ids] = False
        self.phase_contact_sum[ids] = 0.
        self.phase_contact_count[ids] = 0.
        self.hip_samples[ids] = 0.
        self.hip_error_sq[ids] = 0.
        self.hip_velocity_sq[ids] = 0.
        self._foot_cache = None

    def _out_of_bounds(self):
        row = self.terrain_level
        col = self._env_tile_idx.long()%len(COLUMN_FAMILY)
        x = self.base_pos[:, 0]-row*LENGTH
        y = self.base_pos[:, 1]-col*WIDTH
        return (x < .35) | (x > LENGTH-.6) | (y < .4) | (y > WIDTH-.4)

    def _measure(self):
        if self._foot_cache is not None:
            return self._foot_cache
        feet = self.get_feet_positions()
        z, normal = query_surface(feet[..., :2])
        height = feet[..., 2]-z
        velocity = (feet-self.foot_previous)/self.dt
        velocity = torch.where(self.foot_valid[:, None, None], velocity, 0.)
        forces = self.robot.get_links_net_contact_force()[:, self.feet_indices]
        force_normal = (forces*normal).sum(-1).clamp_min(0.)
        contact = (height > -.02) & (height < .055) & (force_normal > 2.)
        # Fit a support-height plane for posture shaping only, in world frame.
        # A little regularization makes nearly collinear feet well defined.
        relative_xy = feet[..., :2]-self.base_pos[:, None, :2]
        A = torch.cat((relative_xy, torch.ones_like(height[..., None])), -1)
        lhs = A.transpose(-1, -2)@A+torch.eye(3, device=self.device)*1e-4
        plane = torch.linalg.solve(lhs, A.transpose(-1, -2)@z[..., None]).squeeze(-1)
        support_normal = torch.cat((-plane[:, :2], torch.ones_like(plane[:, :1])), -1)
        support_normal = support_normal/support_normal.norm(dim=-1, keepdim=True)
        R = rotation_matrix(self.base_quat)
        orientation = (1.-(R[:, :, 2]*support_normal).sum(-1)).clamp(0., 2.)
        tilt = torch.acos((1.-orientation).clamp(-.99999, .99999))
        # A predictive heuristic, not an exact capturability boundary. Do not
        # use yaw rate to license stepping during a stand.
        tilt_direction = torch.cross(R[:, :, 2], support_normal, dim=-1)
        omega_world = torch.einsum("bij,bj->bi", R, self.base_ang_vel)
        outward = (-(omega_world*tilt_direction).sum(-1)/tilt_direction.norm(dim=-1).clamp_min(.02)).clamp_min(0.)
        predicted_tilt = tilt+.18*outward
        risk = ((predicted_tilt-math.radians(22))/math.radians(18)).clamp(0., 1.)
        world_velocity = torch.einsum("bij,bj->bi", R, self.base_lin_vel)
        margin = support_margin(feet[..., :2], contact, self.base_pos[:, :2]+.18*world_velocity[:, :2])
        support_risk = ((.015-margin.nan_to_num(nan=1.))/.08).clamp(0., 1.)
        support_risk *= (self.base_ang_vel[:, :2].norm(dim=-1)-.3).clamp(0., 1.)
        risk = torch.maximum(risk, support_risk)
        clearance = self.base_pos[:, 2]-plane[:, 2]
        # An internal yaw correction may remain active while the requested task
        # is standing. It must not disable landing/quiet/stand objectives.
        stopped = reference_is_stopped(self.reference_commands)
        costs = foot_costs(height, velocity, normal, contact, self.contact_previous,
            self.normal_speed_previous, self.air_age, stopped, risk, self.foot_valid)
        quiet = quiet_support(contact, velocity, self.base_lin_vel, self.base_ang_vel, clearance, risk)
        fall = (self.projected_gravity[:, 2] > 0.) | (clearance < .12)
        local_feet = torch.einsum("bij,bpj->bpi", R.transpose(-1, -2), feet-self.base_pos[:, None, :])
        crossing = ((.055-(local_feet[:, [0, 2], 1]-local_feet[:, [1, 3], 1])).clamp_min(0.)).square().sum(-1)
        self._foot_cache = dict(feet=feet, velocity=velocity, height=height, normal=normal,
            support_heights=z,
            normal_speed=(velocity*normal).sum(-1), force_normal=force_normal, contact=contact,
            costs=costs, risk=risk, orientation=orientation, clearance=clearance,
            quiet=quiet, stopped=stopped, fall=fall, crossing=crossing,
            calf_scuff=((forces.norm(dim=-1) > 5.) & ~contact).float().sum(-1))
        return self._foot_cache

    def _reward_fresh_alive(self):
        s = self._measure()
        elapsed = self.episode_length_buf*self.dt
        if self.env_cfg["fresh_experiment"].get("abduction_revision", False):
            stable = (elapsed > 2.) & (self.family == 0) & (s["risk"] < .1)
            stable &= (self.commands[:, 1].abs() < .02) & (self.commands[:, 2].abs() < .05)
            self.hip_samples += stable
            self.hip_error_sq += (self.dof_pos-self.default_dof_pos)[:, [0, 3, 6, 9]].square()*stable[:, None]
            self.hip_velocity_sq += self.dof_vel[:, [0, 3, 6, 9]].square()*stable[:, None]
        self.air_age = torch.where(s["contact"], 0., self.air_age+self.dt)
        self.support_age = torch.where(s["contact"], self.support_age+self.dt, 0.)
        pair = support_loss_pairs(s["height"], s["force_normal"]) & ~s["fall"][:, None]
        # Count during or just after an ACTUAL pulse, before strict failure.
        pulse_window = ((elapsed >= self.pulse_start) & (elapsed < self.pulse_start+1.) & (self.pulse_count > 0))
        pair &= pulse_window[:, None] & ~self.ever_strict[:, None]
        self.pair_duration = torch.where(pair, self.pair_duration+self.dt, 0.)
        self.pair_seen |= self.pair_duration >= .10-1e-6
        self.ever_fall |= s["fall"]
        self.ever_strict |= self.base_euler[:, :2].abs().amax(-1) > 35.
        self.max_roll = torch.maximum(self.max_roll, self.base_euler[:, 0].abs())
        self.max_pitch = torch.maximum(self.max_pitch, self.base_euler[:, 1].abs())
        valid = self.foot_valid & (elapsed > 1.)
        self.metric_steps += valid
        moving = valid & ~s["stopped"] & (elapsed > 2.)
        error = (self.base_lin_vel[:, :2]-self.commands[:, :2]).norm(dim=-1)
        self.speed_error += error*moving
        self.tracking_steps += moving
        self.tracking_good += (error < torch.maximum(.12*torch.ones_like(error), .1*self.commands[:, :2].norm(dim=-1)))*moving
        post_pulse = (self.pulse_count > 0) & (elapsed > self.pulse_start+.12)
        walking_recovered = self.moving_recovery.update(error,
            self.base_ang_vel[:, 2]-self.commands[:, 2], self.base_ang_vel[:, :2],
            torch.acos((1.-s["orientation"]).clamp(-1., 1.)), s["costs"]["slip"],
            (elapsed > self.pulse_start+.12) & ~s["stopped"],
            s["fall"] | self.ever_strict | (s["risk"] >= .25))
        # Counterfactual sanity check: does the same predicate accept ordinary
        # clean walking? This does not grant recovery credit before a real push.
        self.clean_recovery_windows += (walking_recovered & (self.pulse_count == 0)
            & (elapsed <= self.pulse_start+.12+RECOVERY.deadline_seconds))
        self.phase_contact_history[self.phase_contact_cursor].copy_(s["contact"])
        self.phase_contact_cursor = (self.phase_contact_cursor+1) % self.phase_contact_history.shape[0]
        self.phase_contact_steps += 1
        phase = phase_aligned_contact_cost(self.phase_contact_history, self.phase_contact_cursor,
            self.phase_contact_lags, self.phase_contact_ema, min(1., self.dt/.30))
        uneven = s["support_heights"].amax(-1)-s["support_heights"].amin(-1)
        recovering = self.last_pulse_active | ((self.pulse_count > 0)
            & (elapsed < self.pulse_start+RECOVERY.deadline_seconds+.5) & (self.recovery_time < 0.))
        ready_phase = self.phase_contact_steps >= max(self.phase_contact_lags)+1
        self.phase_contact_gate = (ready_phase & (elapsed > 2.) & ~s["stopped"]
            & (self.reference_commands[:, 0] > .2) & (self.reference_commands[:, 1].abs() < .03)
            & (self.reference_commands[:, 2].abs() < .05) & (self.commands[:, 2].abs() < .08)
            & (s["risk"] < .1) & (uneven < .04) & ~recovering)
        self.phase_contact_cost.copy_(phase*self.phase_contact_gate)
        self.phase_contact_sum += self.phase_contact_cost
        self.phase_contact_count += self.phase_contact_gate
        self.recovery_dwell = torch.where(s["quiet"] & s["stopped"] & post_pulse, self.recovery_dwell+self.dt, 0.)
        recovered = torch.where(s["stopped"], self.recovery_dwell >= .5-1e-6, walking_recovered)
        first_recovery = recovered & post_pulse & (self.recovery_time < 0.)
        self.recovery_time[first_recovery] = (elapsed-self.pulse_start-.12)[first_recovery]
        self.worst_recovery_time[first_recovery] = torch.maximum(self.worst_recovery_time, self.recovery_time)[first_recovery]
        # Sustained tail standing, not a briefly achieved quiet moment.
        tail = valid & s["stopped"] & (elapsed > self.duration-3.)
        self.quiet_steps += tail
        self.quiet_good += s["quiet"]*tail
        self.slip_sum += s["costs"]["slip"]*valid
        self.scuff_sum += s["costs"]["scuff"]*valid
        self.impact_sum += s["costs"]["impact"]*valid
        self.foot_previous.copy_(s["feet"])
        self.contact_previous.copy_(s["contact"])
        self.normal_speed_previous.copy_(s["normal_speed"])
        self.foot_valid[:] = True
        return (~s["fall"]).float()

    def _reward_fresh_slip(self): return self._measure()["costs"]["slip"]
    def _reward_fresh_scuff(self): return self._measure()["costs"]["scuff"]
    def _reward_fresh_impact(self): return self._measure()["costs"]["impact"]
    def _reward_fresh_hover(self): return self._measure()["costs"]["hover"]
    def _reward_fresh_over_lift(self): return self._measure()["costs"]["over_lift"]
    def _reward_fresh_quiet(self): return self._measure()["costs"]["quiet"]
    def _reward_fresh_risk(self): return self._measure()["risk"].square()
    def _reward_fresh_crossing(self): return self._measure()["crossing"]
    def _reward_fresh_failure(self): return self._measure()["fall"].float()
    def _reward_fresh_calf_scuff(self): return self._measure()["calf_scuff"]
    def _reward_fresh_bounce(self):
        return self.base_lin_vel[:, 2].square()*(self.family == 0)

    def _reward_fresh_ang_vel_xy(self):
        # Standard soft body-rate cost, not a hard walking-rate acceptance cap.
        return self.base_ang_vel[:, :2].square().sum(-1)*(1.-.8*self._measure()["risk"])

    def _reward_fresh_heading(self):
        return self.heading_error.square()

    def _reward_fresh_phase_contact_symmetry(self):
        return self.phase_contact_cost

    def _abduction_costs(self):
        s = self._measure()
        if "abduction_costs" not in s:
            s["abduction_costs"] = abduction_costs(self.dof_pos-self.default_dof_pos,
                self.dof_vel, self.commands, s["risk"], s["support_heights"])
        return s["abduction_costs"]

    def _reward_fresh_abduction_pose(self): return self._abduction_costs()[0]
    def _reward_fresh_abduction_motion(self): return self._abduction_costs()[1]

    def _reward_fresh_orientation(self):
        s = self._measure()
        scale = self.env_cfg["fresh_experiment"].get("posture_revision", {}).get("moving_orientation_scale", .2)
        return s["orientation"]*torch.where(s["stopped"], 1., scale)

    def _reward_fresh_stand(self):
        s = self._measure()
        supported = s["contact"].all(-1) & (s["clearance"] > .22) & (s["clearance"] < .42)
        motion = self.base_lin_vel.square().sum(-1)/.08**2+self.base_ang_vel.square().sum(-1)/.25**2
        return s["stopped"]*supported*torch.exp(-motion)*(1.-.8*s["risk"])

    def _reward_base_height(self):
        return (self._measure()["clearance"]-.32).square()

    def _reward_similar_to_default(self):
        return (self.dof_pos-self.default_dof_pos).abs().sum(-1)*(1.-.8*self._measure()["risk"])

    def _update_observation(self):
        elapsed = self.episode_length_buf*self.dt
        stopped = (self.task == 2) | ((self.task == 1) & (((elapsed >= self.stop_time) & (elapsed < self.stop_until))
                   | (elapsed >= self.duration-5.)))
        # Abrupt zero commands teach actual braking, not a supervisor secretly
        # replacing the user's request with a velocity-dependent ramp.
        moving_commands = self.nominal_commands.clone()
        # Sustained turning in a finite straight lane can make an otherwise
        # correct command physically leave the arena. Use bounded turn bouts.
        turn_window = (elapsed >= 3.) & (elapsed < 5.)
        moving_commands[:, 1:] *= turn_window[:, None]
        self.reference_commands.copy_(torch.where(stopped[:, None], 0., moving_commands))
        self.reference_heading += self.reference_commands[:, 2]*self.dt
        self.commands.copy_(self.reference_commands)
        revision = self.env_cfg["fresh_experiment"].get("heading_revision")
        if revision:
            corrected, error = heading_command(self.reference_commands[:, 2], self.reference_heading,
                self.base_quat, revision.get("stiffness", .5), revision.get("yaw_rate_limit", .5))
            self.commands[:, 2] = corrected
            self.heading_error.copy_(error)
        if hasattr(self, "height_sensor"):
            self.height_observation = self.height_sensor.update(self)

    def _apply_fresh_pulse(self):
        elapsed = self.episode_length_buf*self.dt
        next_event = self.repeated_pulses & (self.episode_length_buf >= scheduled_tick(self.next_pulse_start, self.dt)) & (elapsed < self.duration-3.2)
        if next_event.any():
            ids = next_event.nonzero().flatten()
            self.missed_recoveries[ids] += ((self.recovery_time[ids] < 0.) | (self.recovery_time[ids] > RECOVERY.deadline_seconds)).float()
            self.pulse_start[ids] = scheduled_tick(self.next_pulse_start[ids], self.dt)*self.dt
            self.pulse_latched[ids] = False
            self.recovery_time[ids] = -1.
            self.recovery_dwell[ids] = 0.
            self.moving_recovery.reset(ids)
            theta = self._rand((len(ids),))*2*math.pi
            self.pulse_direction[ids] = torch.stack((theta.cos(), theta.sin(), torch.zeros_like(theta)), -1)
        start_tick = scheduled_tick(self.pulse_start, self.dt)
        active = ((self.episode_length_buf >= start_tick)
                  & (self.episode_length_buf < start_tick+round(RECOVERY.pulse_seconds/self.dt))
                  & (self.pulse_family > 0))
        # A scheduled zero-force event (notably at level 0) is not an actual
        # disturbance and must never receive pulse/recovery credit.
        amplitude = torch.where(self.pulse_family == 2,
            self._push_torques[self.terrain_level]*torch.sign(self.pulse_direction[:, 1]).abs(),
            torch.where(self.pulse_family == 3,
                self._push_torques[self.terrain_level]*torch.sign(self.pulse_direction[:, 0]).abs(),
                self._push_strengths[self.terrain_level]))
        active &= amplitude > 0.
        start = active & ~self.pulse_latched
        self.pulse_count += start
        self.pulse_latched |= start
        if start.any():
            ids = start.nonzero().flatten()
            self.next_pulse_start[ids] = self.pulse_start[ids]+4.+2.*self._rand((len(ids),))
        strength = self._push_strengths[self.terrain_level]*active
        torque = self._push_torques[self.terrain_level]*active
        self.last_pulse_active.copy_(active)
        self.applied_impulse += strength*self.dt*(self.pulse_family == 1)
        force = torch.zeros(self.num_envs, 3, device=self.device)
        moment = torch.zeros_like(force)
        leg = torch.zeros(self.num_envs, 4, 3, device=self.device)
        family = self.pulse_family
        force[:] = self.pulse_direction*strength[:, None]*(family == 1)[:, None]
        moment[:, 0] = torch.sign(self.pulse_direction[:, 1])*torque*(family == 2)
        moment[:, 1] = torch.sign(self.pulse_direction[:, 0])*torque*(family == 3)
        for foot in range(4):
            leg[:, foot, 2] = strength*(family == 4)*(self.single_foot == foot)
        groups = ((0, 1), (2, 3), (0, 2), (1, 3))
        for group, legs in enumerate(groups, start=5):
            for foot in legs:
                leg[:, foot, 2] += strength*(family == group)
        self.apply_wrenches(force, moment, leg)

    def _record_episodes(self, ids, bounds):
        previous_records = len(self.completed_records)
        # Batch transfer once per reset group; no CPU sync per robot/metric.
        data = torch.stack([self.terrain_level, self.family, self.task, self.at_ceiling, self.pulse_count,
            self.ever_fall, self.ever_strict, self.speed_error/self.tracking_steps.clamp_min(1),
            self.tracking_good/self.tracking_steps.clamp_min(1), self.tracking_steps,
            self.quiet_good/self.quiet_steps.clamp_min(1), self.quiet_steps,
            self.slip_sum/self.metric_steps.clamp_min(1), self.scuff_sum/self.metric_steps.clamp_min(1),
            bounds, self.pair_seen.any(-1), self.base_pos[:, 0]-self.terrain_level*LENGTH,
            self.max_roll, self.max_pitch, self.impact_sum/self.metric_steps.clamp_min(1),
            self.episode_length_buf*self.dt, self.duration, self.recovery_time,
            self.missed_recoveries, self.worst_recovery_time, self.applied_impulse,
            self.reset_buf & ~(self.ever_fall | bounds | self.ever_strict |
                              (self.episode_length_buf*self.dt >= self.duration-.021)),
            self.clean_recovery_windows,
            self.phase_contact_sum/self.phase_contact_count.clamp_min(1),
            self.phase_contact_count], -1)[ids].detach().cpu().tolist()
        for env_id, values in zip(ids.detach().cpu().tolist(), data):
            (level, family, task, ceiling, pushes, fall, strict, error, track, track_n,
             quiet, quiet_n, slip, scuff, escaped, pair, x, roll, pitch, impact, elapsed, duration, recovery_time,
             missed, worst, impulse, sim_failure, clean_windows, phase_cost, phase_count) = values
            level, family, task = int(level), FAMILIES[int(family)], TASKS[int(task)]
            crossed = x > (8.6 if family.startswith("slope") else 7.2) if family != "flat" and task != "stand" else True
            row = dict(env_id=env_id, level=level, family=family, task=task,
                physical_fall=bool(fall), strict_angle_failure=bool(strict), ceiling=bool(ceiling),
                speed_error=error if track_n else None, tracking_fraction=track, quiet_fraction=quiet,
                slip_mean=slip, scuff_mean=scuff, impact_mean=impact, applied_pulses=int(pushes),
                verified_pair_event=bool(pair), max_roll_deg=roll, max_pitch_deg=pitch,
                crossed=bool(crossed), episode_seconds=elapsed,
                post_pulse_recovery_seconds=recovery_time if recovery_time >= 0 else None,
                tracking_steps=int(track_n), quiet_steps=int(quiet_n), out_of_bounds=bool(escaped),
                simulation_failure=bool(sim_failure),
                phase_aligned_contact_mismatch=phase_cost if phase_count else None,
                phase_aligned_contact_steps=int(phase_count),
                episode_complete=elapsed >= duration-.021, recovery_version=RECOVERY.version,
                missed_recoveries=int(missed), base_force_impulse_ns=impulse,
                clean_walking_recovery_compatible=bool(clean_windows) if task == "move" and not pushes else None)
            if recovery_time >= 0:
                row["post_pulse_recovery_seconds"] = max(recovery_time, worst)
            row["failure_reasons"] = failure_reasons(row)
            row["success"] = not row["failure_reasons"]
            self.ledger.add(level, family, task, row["success"], ceiling=ceiling, disturbed=pushes > 0,
                            pair_event=pair, physical_fall=fall, speed_error=error if track_n else None)
            self.completed_records.append(row)
        # Trainer drains the records to JSONL at every update.
        self.recent_episodes += len(ids)
        if self.env_cfg["fresh_experiment"].get("abduction_revision", False):
            counts = self.hip_samples[ids].cpu().tolist()
            rms = (self.hip_error_sq[ids]/self.hip_samples[ids, None].clamp_min(1)).sqrt().rad2deg().cpu().tolist()
            rates = (self.hip_velocity_sq[ids]/self.hip_samples[ids, None].clamp_min(1)).sqrt().cpu().tolist()
            for row, count, positions, velocities in zip(self.completed_records[previous_records:], counts, rms, rates):
                row.update(stable_straight_hip_samples=int(count), hip_rms_error_deg=positions,
                    hip_rms_velocity_rad_s=velocities, hip_order=["FR", "FL", "RR", "RL"])
