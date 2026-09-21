"""WTW gait-conditioned M2 adaptation, not an exact Go1 reward reproduction.

Phase/clock and contact-shaping equations follow Improbable-AI/walk-these-ways
legged_robot.py and corl_rewards.py (license in the pinned upstream checkout).
Adaptations: native M2 hardware; terrain-relative foot clearance; friction and
base mass privileged targets instead of IsaacGym friction and restitution;
six gait controls, public proprioception and 30-frame history. Body geometry
commands and the original multidimensional command-bin curriculum are omitted.
"""
import math
import torch
from mjlab.managers import EventTermCfg, ObservationTermCfg, RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg


class GaitState:
    def __init__(self, env):
        self.commands = torch.zeros(env.num_envs, 6, device=env.device)
        self.commands[:, 0] = 2.5
        self.commands[:, 1] = .5
        self.commands[:, 4] = .5
        self.commands[:, 5] = .1
        self.phase_offset = torch.zeros(env.num_envs, device=env.device)
        feet = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
        self.feet = SceneEntityCfg("robot", site_names=feet, geom_names=feet, preserve_order=True)
        self.feet.resolve(env.scene)
        self.base = SceneEntityCfg("robot", body_names=("base",))
        self.base.resolve(env.scene)


def state(env):
    if not hasattr(env, "_wtw_gait"):
        env._wtw_gait = GaitState(env)
    return env._wtw_gait


def reset_gait(env, env_ids):
    s = state(env)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    n = len(env_ids)
    draw = torch.rand(n, device=env.device)
    gait = torch.where(draw < .7, 0, torch.where(draw < .8, 1, torch.where(draw < .9, 2, 3)))
    phase = torch.tensor([[.5, 0., 0.], [0., .5, 0.], [0., 0., .5], [0., 0., 0.]], device=env.device)
    s.commands[env_ids, 0] = 2. + 2.*torch.rand(n, device=env.device)
    s.commands[env_ids, 1:4] = phase[gait]
    s.commands[env_ids, 4] = .5 + .1*torch.rand(n, device=env.device)
    s.commands[env_ids, 5] = .06 + .10*torch.rand(n, device=env.device)
    s.phase_offset[env_ids] = torch.rand(n, device=env.device)


def phase_contacts(env):
    s = state(env)
    c = s.commands
    phase = (env.episode_length_buf*env.step_dt*c[:, 0] + s.phase_offset)[:, None]
    offsets = torch.stack((c[:, 1]+c[:, 2]+c[:, 3], c[:, 2], c[:, 3], c[:, 1]), -1)
    raw = (phase + offsets) % 1.
    duration = c[:, 4:5]
    warped = torch.where(raw < duration, raw*.5/duration, .5+(raw-duration)*.5/(1-duration))
    cdf = lambda x: .5*(1+torch.erf(x/(.07*math.sqrt(2.))))
    desired = cdf(warped)*(1-cdf(warped-.5)) + cdf(warped-1)*(1-cdf(warped-1.5))
    return raw, warped, desired


def gait_observation(env):
    _, warped, _ = phase_contacts(env)
    return torch.cat((state(env).commands, torch.sin(2*math.pi*warped)), -1)


def privileged(env):
    s = state(env)
    indexing = env.scene["robot"].indexing
    friction = env.sim.model.geom_friction[:, indexing.geom_ids[s.feet.geom_ids], 0].mean(-1)
    base_mass = env.sim.model.body_mass[:, indexing.body_ids[s.base.body_ids]].mean(-1)
    return torch.stack(((friction-.9)/.6, base_mass/10.), -1)


def gait_reward(env, component):
    s = state(env)
    raw, _, desired = phase_contacts(env)
    command = env.command_manager.get_command("twist")
    active = (command.norm(dim=-1) > .05).float()
    if component == "force":
        force = env.scene["feet_ground_contact"].data.force.norm(dim=-1)
        result = -((1-desired)*(1-torch.exp(-force.square()/100.))).mean(-1)
    elif component == "velocity":
        velocity = env.scene["robot"].data.site_lin_vel_w[:, s.feet.site_ids].square().sum(-1)
        result = -(desired*(1-torch.exp(-velocity/10.))).mean(-1)
    else:
        swing = (raw-s.commands[:, 4:5]).clamp(min=0)/(1-s.commands[:, 4:5])
        shape = 1-(1-2*swing).abs()
        height = env.scene["foot_height_scan"].data.heights
        result = -((height-s.commands[:, 5:6]*shape).square()*(1-desired)).sum(-1)
    return result*active


def configure(cfg):
    cfg.observations["actor"].terms["gait"] = ObservationTermCfg(func=gait_observation)
    cfg.events["wtw_gait_reset"] = EventTermCfg(func=reset_gait, mode="reset")
    cfg.events["wtw_gait_resample"] = EventTermCfg(func=reset_gait, mode="interval", interval_range_s=(10., 10.))
    for component, weight in (("force", 4.), ("velocity", 4.), ("clearance", 30.)):
        cfg.rewards["wtw_"+component] = RewardTermCfg(
            func=gait_reward, weight=weight, params={"component": component})
    cfg.rewards["foot_clearance"].weight = 0.
    cfg.rewards["foot_swing_height"].weight = 0.
    return cfg
