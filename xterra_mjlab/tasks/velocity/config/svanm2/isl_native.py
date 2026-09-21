"""Native MuJoCo bridge for the frozen ISL V7 tensor task.

Coordinates exposed to the reference task start at the arena corner. Native
world coordinates remain unchanged. No Genesis runtime, robot or terrain assets.
"""
from dataclasses import dataclass
from types import SimpleNamespace
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from .ref_task import ReferenceTask
from .ref_fresh_terrain import COLUMN_FAMILY, query_surface
from .ref_fresh_acceptance import AcceptanceMonitor, apply_acceptance
from .ref_fresh_abduction_gate import annotate_abduction
from .ref_fresh_privileged import clean_heights, terrain_offsets, encode_feet


class RobotBridge:
    def __init__(self, state):
        self.s = state

    def set_qpos(self, qpos, envs_idx, **kwargs):
        s, ids = self.s, envs_idx.long()
        root = torch.zeros(len(ids), 13, device=s.device)
        root[:, :7] = qpos[:, :7]
        root[:, :3] -= s.offset
        s.native.write_root_state_to_sim(root, env_ids=ids)
        s.native.write_joint_state_to_sim(qpos[:, 7:], torch.zeros_like(qpos[:, 7:]),
            joint_ids=s.joint_ids, env_ids=ids)
        s.last_dof_vel[ids] = 0.

    def get_links_net_contact_force(self):
        # Native contact sensor reports force exerted BY the primary body;
        # ISL expects force ON the robot. Calibrated with a resting 1 kg sphere.
        return -self.s.env.scene["isl_body_contact"].data.force


class NativeTask(ReferenceTask):
    def __init__(self, env):
        self.env, self.native = env, env.scene["robot"]
        self.num_envs, self.device, self.dt = env.num_envs, env.device, env.step_dt
        self.num_actions = 12
        self.env_cfg = {"fresh_experiment": dict(seed=env.cfg.seed,
            policy_backbone="rsl_height", repeated_pushes=True, abduction_revision=True,
            posture_revision=dict(moving_orientation_scale=.6),
            heading_revision=dict(stiffness=.5, yaw_rate_limit=.5))}
        self.joint_ids = self.native.find_joints(tuple(
            f"{leg}_{joint}_joint" for leg in ("FR","FL","RR","RL")
            for joint in ("hip","thigh","calf")), preserve_order=True)[0]
        self.feet_site_ids = self.native.find_sites(tuple(
            f"{leg}_foot" for leg in ("FL","FR","RL","RR")), preserve_order=True)[0]
        self.feet_geom_ids = self.native.find_geoms(tuple(
            f"{leg}_foot" for leg in ("FL","FR","RL","RR")), preserve_order=True)[0]
        self.feet_indices = self.native.find_bodies(tuple(
            f"{leg}_shank_link" for leg in ("FL","FR","RL","RR")), preserve_order=True)[0]
        self.base_ids = self.native.find_bodies("base")[0]
        origin = env.scene.terrain.terrain_origins[0,0]
        self.offset = origin.new_tensor([1.5,3.,0.])-origin
        self._terrain_origin = torch.zeros(3, device=self.device)
        self.terrain_arena = SimpleNamespace(size=(240.,42.))
        self.terrain_level = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._env_tile_idx = torch.zeros_like(self.terrain_level)
        self._env_spawn_xy = torch.zeros(self.num_envs,2,device=self.device)
        self._max_level = 9
        self.init_base_quat = torch.tensor([1.,0.,0.,0.],device=self.device)
        self.init_dof_pos = self.native.data.default_joint_pos[0,self.joint_ids].clone()
        self.init_qpos = torch.cat((torch.zeros(3,device=self.device),self.init_base_quat,self.init_dof_pos))
        self.robot = RobotBridge(self)
        self.last_dof_vel = torch.zeros(self.num_envs,12,device=self.device)
        self.commands = torch.zeros(self.num_envs,3,device=self.device)
        self._init_terrain_buffers()
        self.acceptance = AcceptanceMonitor(self.num_envs,self.dt,self.device)
        self.evaluation = False
        self.evaluation_batch = None
        self.trials_per_case = 64
        self.peak = torch.zeros(self.num_envs,4,device=self.device)
        self.swing_peak_sum = torch.zeros_like(self.peak)
        self.landings = torch.zeros_like(self.peak)
        self.edge_height_sum = torch.zeros_like(self.peak)
        self.edge_samples = torch.zeros_like(self.peak)
        self.previous_feet_x = torch.zeros_like(self.peak)
        self.speed_sums = torch.zeros(self.num_envs,6,device=self.device)
        self.privileged_offsets = terrain_offsets(self.device)

    @property
    def base_pos(self): return self.native.data.root_link_pos_w+self.offset
    @property
    def base_quat(self): return self.native.data.root_link_quat_w
    @property
    def base_lin_vel(self): return self.native.data.root_link_lin_vel_b
    @property
    def base_ang_vel(self): return self.native.data.root_link_ang_vel_b
    @property
    def projected_gravity(self): return self.native.data.projected_gravity_b
    @property
    def dof_pos(self): return self.native.data.joint_pos[:,self.joint_ids]
    @property
    def dof_vel(self): return self.native.data.joint_vel[:,self.joint_ids]
    @property
    def default_dof_pos(self): return self.init_dof_pos
    @property
    def episode_length_buf(self): return self.env.episode_length_buf
    @property
    def reset_buf(self): return self.env.reset_buf
    @property
    def extras(self): return self.env.extras
    @property
    def base_euler(self):
        w,x,y,z = self.base_quat.unbind(-1)
        return torch.stack((torch.atan2(2*(w*x+y*z),1-2*(x*x+y*y)),
            torch.asin((2*(w*y-z*x)).clamp(-1,1)),
            torch.atan2(2*(w*z+x*y),1-2*(y*y+z*z))),-1).rad2deg()

    def get_feet_positions(self):
        return self.native.data.site_pos_w[:,self.feet_site_ids]+self.offset

    def _env_indices(self, ids):
        if ids is None: return torch.arange(self.num_envs,device=self.device)
        return ids.nonzero().flatten() if ids.dtype == torch.bool else ids

    def native_reset(self, ids):
        self._env_tile_idx[ids] = self._sample_tiles_for_levels(self.terrain_level[ids])
        col = self._env_tile_idx[ids]%7
        terrain = self.env.scene.terrain
        terrain.terrain_levels[ids] = self.terrain_level[ids]
        terrain.terrain_types[ids] = col
        terrain.env_origins[ids] = terrain.terrain_origins[self.terrain_level[ids],col]
        # Reference reset reads y before writing qpos, so place the lane origin.
        root = self.native.data.default_root_state[ids].clone()
        root[:,:3] = terrain.env_origins[ids]
        root[:,2] += .34
        self.native.write_root_state_to_sim(root,env_ids=ids)

    def reset(self, ids):
        if self.evaluation_batch is None:
            self._reset_idx(ids)
        else:
            for index, case in enumerate(self.evaluation_batch):
                subset = ids[ids//self.trials_per_case == index]
                if len(subset):
                    self.eval_scenario = case
                    self._reset_idx(subset)
            self.eval_scenario = None
        self.acceptance.reset(ids)
        for name in ("peak","swing_peak_sum","landings","edge_height_sum","edge_samples","previous_feet_x","speed_sums"):
            getattr(self,name)[ids] = 0

    def apply_wrenches(self, force, moment, leg):
        forces = torch.cat((force[:,None],leg),1)
        moments = torch.cat((moment[:,None],torch.zeros_like(leg)),1)
        self.native.write_external_wrench_to_sim(forces,moments,
            body_ids=self.base_ids+self.feet_indices)

    def _reward_fresh_alive(self):
        s = self._measure()
        self.peak = torch.maximum(self.peak,s["height"])
        landing = s["contact"] & ~self.contact_previous & self.foot_valid[:,None]
        self.swing_peak_sum += self.peak*landing
        self.landings += landing
        self.peak.masked_fill_(s["contact"],0.)
        # Actual clearance over each riser's upper edge, not height above the
        # low approach floor. This is advisory and never alters the objective.
        centers = self.native.data.geom_pos_w[:,self.feet_geom_ids]+self.offset
        x = centers[...,0]-self.terrain_level[:,None]*24.
        col = self._env_tile_idx%7
        treads = x.new_tensor([.25,.30,.35,.40])[(self.terrain_level+col)%4]
        for edge in range(6):
            ex = 4.+edge*treads[:,None]
            passed = (self.previous_feet_x<ex)&(x>=ex)&self.foot_valid[:,None]&(self.family[:,None]>=3)
            edge_xy=centers[...,:2].clone()
            edge_xy[...,0]=ex+self.terrain_level[:,None]*24.
            before=edge_xy.clone()
            before[...,0]-=.001
            upper=torch.maximum(query_surface(edge_xy)[0],query_surface(before)[0])
            self.edge_height_sum += (centers[...,2]-.022-upper)*passed
            self.edge_samples += passed
        self.previous_feet_x.copy_(x)
        if self.evaluation:
            local_x = self.base_pos[:,0]-self.terrain_level*24.
            moving = (self.episode_length_buf*self.dt>2.)&(self.commands.norm(dim=-1)>=.02)
            obstacle = moving&(self.family>0)&(local_x>=4.)&(local_x<=8.)
            good = ((self.base_lin_vel[:,:2]-self.commands[:,:2]).norm(dim=-1)
                <= torch.maximum(local_x.new_full(local_x.shape,.12),.1*self.commands[:,:2].norm(dim=-1)))
            self.speed_sums[:,0] += self.base_lin_vel[:,0]*moving
            self.speed_sums[:,1] += self.commands[:,0]*moving
            self.speed_sums[:,2] += moving
            self.speed_sums[:,3] += obstacle
            self.speed_sums[:,4] += obstacle*good
            self.speed_sums[:,5] += self.base_lin_vel[:,0]*obstacle
            self.acceptance.update(self.episode_length_buf*self.dt,self.reference_commands,
                self.base_lin_vel,s["quiet"],self.base_quat,self.base_ang_vel,
                obstacle)
        result = super()._reward_fresh_alive()
        self.extras["log"]["ISL/scuff"] = s["costs"]["scuff"].mean()
        for i,leg in enumerate(("FL","FR","RL","RR")):
            self.extras["log"][f"ISL/{leg}_swing_peak"] = (self.swing_peak_sum[:,i].sum()/self.landings[:,i].sum().clamp_min(1))
        return result

    def _record_episodes(self, ids, bounds):
        first = len(self.completed_records)
        super()._record_episodes(ids,bounds)
        acceptance = self.acceptance.records(ids) if self.evaluation else [{} for _ in ids]
        peaks = (self.swing_peak_sum[ids]/self.landings[ids].clamp_min(1)).cpu().tolist()
        edges = (self.edge_height_sum[ids]/self.edge_samples[ids].clamp_min(1)).cpu().tolist()
        counts = self.edge_samples[ids].cpu().tolist()
        for row, metrics, peak, edge, count in zip(self.completed_records[first:],acceptance,peaks,edges,counts):
            row.update(metrics)
            row.update(swing_peak_m=peak,stair_edge_clearance_m=edge,stair_edge_samples=count,
                       clearance_leg_order=["FL","FR","RL","RR"])
            if self.evaluation:
                case = self.evaluation_batch[row["env_id"]//self.trials_per_case]
                row.update(pulse_family=case.get("pulse_family",0))
                from .ref_evaluation import assign_pair_evidence
                assign_pair_evidence([row],[self.pair_seen[row["env_id"]].cpu().tolist()],row["pulse_family"])
                actual,command,n,nt,good,actual_t = self.speed_sums[row["env_id"]].cpu().tolist()
                row.update(mean_actual_forward_mps=actual/n if n else None,
                    mean_commanded_forward_mps=command/n if n else None,
                    terrain_tracking_fraction=good/nt if nt else None,
                    terrain_mean_actual_forward_mps=actual_t/nt if nt else None)
                apply_acceptance(row)
                if row["family"]=="flat":
                    annotate_abduction(row,row["pulse_family"],self.dt)


class NativeCommand(CommandTerm):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        self.task = NativeTask(env)
    @property
    def command(self): return self.task.commands
    def _resample_command(self,ids): self.task.reset(ids)
    def _update_command(self): self.task._update_observation()
    def _update_metrics(self): pass


@dataclass(kw_only=True)
class NativeCommandCfg(CommandTermCfg):
    def build(self,env): return NativeCommand(self,env)


def task(env): return env.command_manager.get_term("twist").task


class IslEnv(ManagerBasedRlEnv):
    def step(self,action):
        s = task(self)
        s._foot_cache = None
        s._apply_fresh_pulse()
        result = super().step(action)
        s.last_dof_vel.copy_(s.dof_vel)
        return result

    def _reset_idx(self,env_ids=None):
        s = task(self)
        ids = s._env_indices(env_ids)
        outgoing = ids[self.episode_length_buf[ids]>0]
        if len(outgoing):
            s._record_episodes(outgoing,s._out_of_bounds())
        super()._reset_idx(ids)


def reward(env,name):
    s = task(env)
    if name.startswith("fresh_") or name in ("base_height","similar_to_default"):
        return getattr(s,"_reward_"+name)()
    if name == "tracking_lin_vel":
        return torch.exp(-(s.commands[:,:2]-s.base_lin_vel[:,:2]).square().sum(-1)/.25)
    if name == "tracking_ang_vel":
        return torch.exp(-(s.commands[:,2]-s.base_ang_vel[:,2]).square()/.25)
    if name == "action_rate":
        return (env.action_manager.action-env.action_manager.prev_action).square().sum(-1)
    if name == "dof_torques":
        return s.native.data.qfrc_actuator.square().sum(-1)
    if name == "dof_acc":
        return ((s.dof_vel-s.last_dof_vel)/s.dt).square().sum(-1)
    if name == "collision":
        force = s.robot.get_links_net_contact_force().clone()
        force[:,s.feet_indices] = 0
        return (force.norm(dim=-1)>1.).float().sum(-1)
    raise ValueError(name)


def terminated(env):
    s = task(env)
    return s._measure()["fall"] | (s.base_euler[:,0].abs()>55.) | (s.base_euler[:,1].abs()>60.)


def timed_out(env):
    s = task(env)
    return ((s.episode_length_buf*s.dt>=s.duration)|s._out_of_bounds()) & ~terminated(env)


def height_obs(env): return task(env).height_observation
def priv_vel(env): return task(env).base_lin_vel
def priv_height(env):
    s = task(env)
    return clean_heights(s.base_pos,s.base_quat,s.privileged_offsets,s.terrain_heights)
def priv_feet(env):
    s = task(env)
    feet = s.get_feet_positions()
    z,normal = query_surface(feet[...,:2])
    height = feet[...,2]-z
    forces = s.robot.get_links_net_contact_force()[:,s.feet_indices]
    contact = (height>-.02)&(height<.055)&((forces*normal).sum(-1)>2.)
    order = [1,0,3,2]
    return encode_feet(height[:,order],s.air_age[:,order],contact[:,order],
        forces[:,order],s.base_quat,s.episode_length_buf>0)
