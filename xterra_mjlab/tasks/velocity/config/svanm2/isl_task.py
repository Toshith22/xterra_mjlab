"""Native mjlab adaptation of the ISL task; public robot/actuator unchanged.

Each invocation trains one explicitly selected, reviewed difficulty block.
This is a new simulator adaptation, not a numerically equivalent Genesis run.
"""
from dataclasses import dataclass
import math
import torch
from types import SimpleNamespace
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
from .env_cfgs import svanm2_rough_env_cfg
from .isl_terrain import LEVELS, FAMILIES, IslLaneCfg
from .reference_sensing import HeightSensorSurrogate, rotation_matrix
from .reference_rewards import foot_costs, quiet_support, support_margin, phase_aligned_contact_cost


class IslCommand(CommandTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.robot = env.scene["robot"]
        n, d = env.num_envs, env.device
        self.vel = torch.zeros(n, 3, device=d)
        self.reference = torch.zeros_like(self.vel)
        self.heading = torch.zeros(n, device=d)
        self.error = torch.zeros(n, device=d)
        self.task = torch.zeros(n, dtype=torch.long, device=d)
        self.speed = torch.zeros(n, device=d)
        self.elapsed = torch.zeros(n, device=d)
        self.next_push = torch.full((n,), 4., device=d)
        self.push_end = torch.zeros(n, device=d)
        self.push_direction = torch.zeros(n, 3, device=d)
        self.pushed = torch.zeros(n, dtype=torch.bool, device=d)
        self.feet_ids = self.robot.find_sites(tuple(f"{leg}_foot" for leg in ("FL", "FR", "RL", "RR")), preserve_order=True)[0]
        self.base_ids = self.robot.find_bodies("base")[0]
        self.hip_ids = self.robot.find_joints(".*_hip_joint")[0]
        self.previous_contacts = torch.zeros(n, 4, dtype=torch.bool, device=d)
        self.peak = torch.zeros(n, 4, device=d)
        self.air_age = torch.zeros(n, 4, device=d)
        self.previous_normal_speed = torch.zeros(n, 4, device=d)
        self.phase_history = torch.zeros(26, n, 4, dtype=torch.bool, device=d)
        self.phase_cursor = 0
        self.phase_ema = torch.zeros(n, len(range(8, 25, 2)), device=d)
        self.last_reward_step = -1
        self.cached_reward = torch.zeros(n, device=d)

    @property
    def command(self):
        return self.vel

    def _update_metrics(self):
        pass

    def _resample_command(self, ids):
        n = len(ids)
        draw = torch.rand(n, device=self.device)
        self.task[ids] = torch.where(draw < .2, 2, torch.where(draw < .5, 1, 0))
        family = self._env.scene.terrain.terrain_types[ids]
        limits = self.vel.new_tensor([LEVELS[self.cfg.level][0],
            LEVELS[self.cfg.level][2], LEVELS[self.cfg.level][2],
            LEVELS[self.cfg.level][4], LEVELS[self.cfg.level][4]])
        ceiling = torch.rand(n, device=self.device) < .5
        self.speed[ids] = limits[family]*torch.where(ceiling, 1., .5+.5*torch.rand(n, device=self.device))
        # Reset orientation is fixed to public nominal; derived world pose is
        # stale until mjlab's post-reset forward pass.
        self.heading[ids] = 0.
        self.elapsed[ids] = 0.
        self.next_push[ids] = 3.+2.*torch.rand(n, device=self.device)
        self.push_end[ids] = 0.
        self.pushed[ids] = (torch.rand(n, device=self.device) > .35) & (self.cfg.level > 0)
        self.previous_contacts[ids] = False
        self.peak[ids] = 0.
        self.air_age[ids] = 0.
        self.previous_normal_speed[ids] = 0.
        self.phase_history[:, ids] = False
        self.phase_ema[ids] = 0.
        self.reference[ids] = 0.
        self.reference[ids,0] = torch.where(self.task[ids]==2,0.,self.speed[ids])
        self.vel[ids] = self.reference[ids]
        self.error[ids] = 0.

    def _update_command(self):
        self.elapsed.copy_(self._env.episode_length_buf*self._env.step_dt)
        # Stop, resume, then brake to a final quiet hold.
        stopped = (self.task == 2) | ((self.task == 1) &
            (((self.elapsed >= 6.) & (self.elapsed < 9.)) | (self.elapsed >= 15.)))
        self.reference.zero_()
        self.reference[:, 0] = torch.where(stopped, 0., self.speed)
        self.error = torch.atan2(torch.sin(self.heading-self.robot.data.heading_w),
                                torch.cos(self.heading-self.robot.data.heading_w))
        self.vel.copy_(self.reference)
        self.vel[:, 2] = (.5*self.error).clamp(-.5, .5)
        start = (self.elapsed >= self.next_push) & self.pushed & (self.elapsed < 15.)
        if start.any():
            theta = 2*math.pi*torch.rand(int(start.sum()), device=self.device)
            self.push_direction[start] = torch.stack((theta.cos(), theta.sin(), torch.zeros_like(theta)), -1)
            self.push_end[start] = self.elapsed[start]+.12
            self.next_push[start] = self.elapsed[start]+4.+2.*torch.rand(int(start.sum()), device=self.device)
        active = self.elapsed < self.push_end
        force = self.push_direction*active[:, None]*LEVELS[self.cfg.level][5]
        torque = torch.zeros_like(force)
        torque[:, 0] = active*self.push_direction[:, 1]*LEVELS[self.cfg.level][6]
        self.robot.write_external_wrench_to_sim(force[:, None], torque[:, None], body_ids=self.base_ids)


@dataclass(kw_only=True)
class IslCommandCfg(CommandTermCfg):
    level: int = 0

    def build(self, env):
        return IslCommand(self, env)


def state(env):
    c = env.command_manager.get_term("twist")
    robot = env.scene["robot"]
    heights = env.scene["foot_height_scan"].data.heights
    contact = env.scene["feet_ground_contact"].data.found > 0
    feet_vel = robot.data.site_lin_vel_w[:, c.feet_ids]
    return c, robot, heights, contact, feet_vel


def isl_objective(env):
    c, r, height, contact, fv = state(env)
    if c.last_reward_step == env.common_step_counter:
        return c.cached_reward
    c.last_reward_step = env.common_step_counter
    vel, ang = r.data.root_link_lin_vel_b, r.data.root_link_ang_vel_b
    stopped = c.reference.norm(dim=-1) < .02
    error = (vel[:, :2]-c.vel[:, :2]).square().sum(-1)
    feet = r.data.site_pos_w[:, c.feet_ids]
    z = terrain_height(env, feet[..., :2])
    height = feet[..., 2]-z
    relative = feet[..., :2]-r.data.root_link_pos_w[:, None, :2]
    a = torch.cat((relative, torch.ones_like(height[..., None])), -1)
    plane = torch.linalg.solve(a.transpose(-1,-2)@a+torch.eye(3, device=env.device)*1e-4,
        a.transpose(-1,-2)@z[..., None]).squeeze(-1)
    normal = torch.cat((-plane[:, :2], torch.ones_like(plane[:, :1])), -1)
    normal = normal/normal.norm(dim=-1, keepdim=True)
    rotation = rotation_matrix(r.data.root_link_quat_w)
    tilt = (1.-(rotation[:, :, 2]*normal).sum(-1)).clamp(0., 2.)
    direction = torch.cross(rotation[:, :, 2], normal, dim=-1)
    omega = torch.einsum("bij,bj->bi", rotation, ang)
    outward = (-(omega*direction).sum(-1)/direction.norm(dim=-1).clamp_min(.02)).clamp_min(0.)
    predicted = torch.acos((1.-tilt).clamp(-.99999,.99999))+.18*outward
    risk = ((predicted-math.radians(22))/math.radians(18)).clamp(0.,1.)
    margin = support_margin(feet[..., :2], contact,
        r.data.root_link_pos_w[:, :2]+.18*r.data.root_link_lin_vel_w[:, :2])
    support_risk = ((.015-margin.nan_to_num(nan=1.))/.08).clamp(0.,1.)
    risk = torch.maximum(risk, support_risk*(ang[:,:2].norm(dim=-1)-.3).clamp(0.,1.))
    clearance = r.data.root_link_pos_w[:,2]-plane[:,2]
    costs = foot_costs(height, fv, normal[:,None].expand_as(fv), contact,
        c.previous_contacts, c.previous_normal_speed, c.air_age, stopped, risk,
        env.episode_length_buf > 1)
    slip, scuff, over_lift, quiet = (costs[k] for k in ("slip","scuff","over_lift","quiet"))
    dq = r.data.joint_pos-r.data.default_joint_pos
    hip_error = dq[:, c.hip_ids]
    uneven = (z.amax(-1)-z.amin(-1))/.12
    release = torch.maximum(risk, uneven.clamp(0., 1.))
    stand = stopped*contact.all(-1)*torch.exp(-vel.square().sum(-1)/.08**2-ang.square().sum(-1)/.25**2)
    local_feet = torch.einsum("bji,bpj->bpi", rotation, feet-r.data.root_link_pos_w[:,None])
    crossing = ((.055-(local_feet[:,[0,2],1]-local_feet[:,[1,3],1])).clamp_min(0.)).square().sum(-1)
    c.phase_history[c.phase_cursor].copy_(contact)
    c.phase_cursor = (c.phase_cursor+1)%26
    phase = phase_aligned_contact_cost(c.phase_history,c.phase_cursor,range(8,25,2),c.phase_ema,env.step_dt/.3)
    phase *= (env.episode_length_buf > 100)&~stopped&(risk<.1)&(uneven<.04/.12)&~c.pushed
    score = (.5+2.*torch.exp(-error/.25)+.5*torch.exp(-(ang[:, 2]-c.vel[:, 2]).square()/.25)
        -.4*slip-.3*scuff-2.*over_lift-.3*quiet+stand-.3*risk.square()
        -.08*costs["impact"]-.25*costs["hover"]-2.*crossing-.05*phase
        -12.*(clearance-.32).square()-2e-4*r.data.qfrc_actuator.square().sum(-1)
        -2.5e-7*r.data.joint_acc.square().sum(-1)
        -.5*tilt*torch.where(stopped, 1., .6)-.5*vel[:, 2].square()*(env.scene.terrain.terrain_types==0)
        -.05*ang[:, :2].square().sum(-1)*(1.-.8*risk)-.25*c.error.square()
        -.04*dq.square().sum(-1)*(1.-.8*risk)
        -.35*((hip_error.abs()-.06).clamp_min(0.)/.15).square().mean(-1)*(1.-.9*release)
        -.025*r.data.joint_vel[:, c.hip_ids].square().mean(-1)*(1.-.9*release))
    c.peak = torch.where(~contact, torch.maximum(c.peak, height), c.peak)
    landing = contact & ~c.previous_contacts
    env.extras["log"]["ISL/foot_clearance_mean"] = height.mean()
    env.extras["log"]["ISL/swing_peak_at_landing"] = (c.peak*landing).sum()/landing.sum().clamp_min(1)
    env.extras["log"]["ISL/slip"] = slip.mean()
    env.extras["log"]["ISL/scuff"] = scuff.mean()
    for i,leg in enumerate(("FL","FR","RL","RR")):
        env.extras["log"][f"ISL/{leg}_swing_fraction"] = (~contact[:,i]).float().mean()
        env.extras["log"][f"ISL/{leg}_swing_peak"] = (c.peak[:,i]*landing[:,i]).sum()/landing[:,i].sum().clamp_min(1)
    c.peak.masked_fill_(landing, 0.)
    c.previous_contacts.copy_(contact)
    c.air_age = torch.where(contact, 0., c.air_age+env.step_dt)
    c.previous_normal_speed.copy_((fv*normal[:,None]).sum(-1))
    c.cached_reward = score
    return score


def terrain_height(env, xy):
    # The generated lanes are translated copies of the exact checked profile.
    terrain = env.scene.terrain
    left = terrain.terrain_origins[0,:,0].min()-1.5
    bottom = terrain.terrain_origins[0,:,1].min()-3.
    family = ((xy[...,1]-bottom)/6.).floor().long().clamp(0, len(terrain.cfg.terrain_generator.sub_terrains)-1)
    x = xy[...,0]-left
    level = env.command_manager.get_term("twist").cfg.level
    slope = (x-4.).clamp(0.,4.)*math.tan(math.radians(LEVELS[level][1]))
    stairs = ((x-4.+1e-6)/.3).floor().add(1).clamp(0,6)*LEVELS[level][3]
    return torch.where(family==1,slope,torch.where(family==2,
        4.*math.tan(math.radians(LEVELS[level][1]))-slope,
        torch.where(family==3,stairs,torch.where(family==4,6*LEVELS[level][3]-stairs,0.))))


class DelayedHeight:
    """10 Hz measured ray heights, 40 ms latency, three masked frames."""
    def __init__(self, cfg, env):
        n, d = env.num_envs, env.device
        self.sensor = HeightSensorSurrogate(n,d,env.cfg.seed+71000)
        self.output = torch.zeros(n,381,device=d)
        self.last = -1

    def reset(self, env_ids):
        self.sensor.reset(env_ids)
        self.output[env_ids] = 0.

    def __call__(self, env):
        tick = env.common_step_counter
        if tick != self.last:
            self.last = tick
            terrain = env.scene.terrain
            origin = terrain.terrain_origins[0].amin(0)-torch.tensor([1.5,3.,0.],device=env.device)
            proxy = SimpleNamespace(base_pos=env.scene["robot"].data.root_link_pos_w,
                base_quat=env.scene["robot"].data.root_link_quat_w,
                terrain_heights=lambda xy: terrain_height(env,xy), _terrain_origin=origin,
                terrain_arena=SimpleNamespace(size=(24.,6.*len(terrain.cfg.terrain_generator.sub_terrains))))
            self.output = self.sensor.update(proxy)
        return self.output


def make_isl_cfg(level=0):
    cfg = svanm2_rough_env_cfg()
    allowed = FAMILIES[:1 if level < 2 else 3 if level < 3 else 5]
    cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(seed=98001,
        curriculum=True, size=(24., 6.), num_rows=1, border_width=2.,
        sub_terrains={f: IslLaneCfg(family=f, level=level) for f in allowed})
    cfg.scene.terrain.max_init_terrain_level = 0
    cfg.curriculum = {}
    cfg.episode_length_s = 20.
    cfg.commands["twist"] = IslCommandCfg(level=level, resampling_time_range=(1000., 1000.))
    cfg.events.pop("push_robot", None)
    cfg.events["reset_base"].params["pose_range"] = {"x": (-.3, .3), "y": (-.1, .1), "yaw": (0., 0.)}
    # Preserve the public robot, actuator, action scale, delay and DR parameters.
    cfg.observations["height"] = ObservationGroupCfg(
        terms={"delayed_height": ObservationTermCfg(func=DelayedHeight)}, concatenate_terms=True)
    from .isl_observations import Dynamics
    cfg.observations["dynamics"] = ObservationGroupCfg(
        terms={"actual_parameters": ObservationTermCfg(func=Dynamics)}, concatenate_terms=True)
    # Keep native collision/action/limit protection, replace locomotion shaping.
    cfg.rewards = {k: v for k, v in cfg.rewards.items() if k in
        ("dof_pos_limits", "action_rate_l2", "self_collisions", "shank_collision", "trunk_collision", "soft_landing")}
    cfg.rewards["action_rate_l2"].weight = -.015
    cfg.rewards["isl_objective"] = RewardTermCfg(func=isl_objective, weight=1.)
    from mjlab.tasks.velocity import mdp
    cfg.terminations["fell_over"] = TerminationTermCfg(func=mdp.bad_orientation,
        params={"limit_angle": math.radians(55.)})
    cfg.sim.njmax = 600
    cfg.sim.contact_sensor_maxmatch = 128
    return cfg
