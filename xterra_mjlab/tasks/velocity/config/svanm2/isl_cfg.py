"""ISL V7 learning/task configuration with the public SvanM2 embodiment."""
import torch
from mjlab.managers import ObservationGroupCfg, ObservationTermCfg, RewardTermCfg, TerminationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensorCfg, ContactMatch
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
from .env_cfgs import svanm2_rough_env_cfg
from .isl_terrain import IslLaneCfg,FAMILIES
from .isl_observations import Dynamics
from . import isl_native as mdp

WEIGHTS = dict(fresh_alive=.5, tracking_lin_vel=2., tracking_ang_vel=.5,
    base_height=-12., action_rate=-.015, similar_to_default=-.04,
    dof_torques=-2e-4, dof_acc=-2.5e-7, collision=-1.,
    fresh_slip=-.4, fresh_scuff=-.3, fresh_impact=-.08, fresh_hover=-.25,
    fresh_over_lift=-2., fresh_quiet=-.3, fresh_stand=1., fresh_risk=-.3,
    fresh_orientation=-.5, fresh_crossing=-2., fresh_failure=-5., fresh_calf_scuff=-.2,
    fresh_bounce=-.5, fresh_ang_vel_xy=-.05, fresh_heading=-.25,
    fresh_phase_contact_symmetry=-.05, fresh_abduction_pose=-.35, fresh_abduction_motion=-.025)


def structured_gains(env,env_ids):
    ids = torch.arange(env.num_envs,device=env.device) if env_ids is None else env_ids
    for actuator in env.scene["robot"].actuators:
        n,k = len(ids),actuator.stiffness.shape[-1]
        def draw():
            return (.92+.16*torch.rand(n,1,device=env.device))*(.98+.04*torch.rand(n,k,device=env.device))
        actuator.set_gains(kp=actuator.default_stiffness[ids]*draw(),
                          kd=actuator.default_damping[ids]*draw(),env_ids=ids)


def make_isl_cfg(level=0):
    cfg = svanm2_rough_env_cfg()
    cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(seed=98001,
        curriculum=True,size=(24.,6.),num_rows=10,border_width=2.,
        sub_terrains={f"lane_{i}":IslLaneCfg(family=FAMILIES[f],column=i)
            for i,f in enumerate((0,1,2,3,4,3,4))})
    cfg.scene.terrain.max_init_terrain_level = 0
    cfg.curriculum = {}
    cfg.episode_length_s = 40.
    cfg.commands["twist"] = mdp.NativeCommandCfg(resampling_time_range=(1000.,1000.))
    # The task owns all reset poses, episode timing, disturbances and objectives.
    cfg.events = {k:v for k,v in cfg.events.items() if k in ("foot_friction","base_mass")}
    cfg.events["foot_friction"].params.update(ranges=(.6,1.3),operation="scale",shared_random=True)
    cfg.events["foot_friction"].params["asset_cfg"] = SceneEntityCfg("robot",geom_names=(".*",))
    cfg.events["base_mass"].params["ranges"] = (-1.,2.)
    cfg.events["pd_gains"] = EventTermCfg(func=structured_gains,mode="startup")
    cfg.scene.sensors = tuple(s for s in cfg.scene.sensors if s.name=="feet_ground_contact")+(
        ContactSensorCfg(name="isl_body_contact",
            primary=ContactMatch(mode="body",pattern=".*",entity="robot"),
            fields=("found","force"),reduce="netforce",num_slots=1),)
    # Same six-frame proprioception categories, with native joint/term ordering.
    cfg.observations = {"actor":cfg.observations["actor"]}
    cfg.observations["actor"].terms["command"].scale = (2.,2.,.25)
    for name,func in (("height",mdp.height_obs),("priv_vel",mdp.priv_vel),
                      ("priv_height",mdp.priv_height),("priv_feet",mdp.priv_feet),
                      ("dynamics",Dynamics)):
        cfg.observations[name] = ObservationGroupCfg(
            terms={name:ObservationTermCfg(func=func)},concatenate_terms=True)
    cfg.rewards = {name:RewardTermCfg(func=mdp.reward,weight=weight,params={"name":name})
        for name,weight in WEIGHTS.items()}
    cfg.terminations = {"failure":TerminationTermCfg(func=mdp.terminated),
        "task_timeout":TerminationTermCfg(func=mdp.timed_out,time_out=True)}
    cfg.sim.njmax = 600
    cfg.sim.contact_sensor_maxmatch = 128
    return cfg
