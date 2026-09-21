"""Native observation-order-aware reflection and actual actuator DR readback."""
import math
import torch


def permutation(names):
    names = [n.split("/")[-1] for n in names]
    swap = {"FL":"FR", "FR":"FL", "RL":"RR", "RR":"RL"}
    return [names.index(swap[n[:2]]+n[2:]) for n in names]


class Dynamics:
    def __init__(self, cfg, env):
        self.robot = env.scene["robot"]
        self.base = self.robot.indexing.body_ids[self.robot.find_bodies("base")[0]]
        self.feet = self.robot.indexing.geom_ids[self.robot.find_geoms(
            tuple(f"{leg}_foot" for leg in ("FL","FR","RL","RR")), preserve_order=True)[0]]
        self.actuator = self.robot.actuators[0]
        native = [n.split("/")[-1] for n in self.actuator._target_names]
        self.order = [native.index(n.split("/")[-1]) for n in self.robot.joint_names]

    def __call__(self, env):
        friction = env.sim.model.geom_friction[:,self.feet,0]
        mass = env.sim.model.body_mass[:,self.base]
        kp = self.actuator.stiffness[:,self.order]/20.-1.
        kd = self.actuator.damping[:,self.order]/.7-1.
        return torch.cat((friction/.8-1.,(mass-4.35875)/2.,kp,kd),-1)


def reflect(env, obs):
    env = env.unwrapped if hasattr(env,"unwrapped") else env
    robot = env.scene["robot"]
    qperm = permutation(robot.joint_names)
    anames = env.action_manager.get_term("joint_pos").target_names
    aperm = permutation(anames)
    qsign = obs["actor"].new_tensor([-1. if "_hip_" in n else 1. for n in robot.joint_names])
    asign = obs["actor"].new_tensor([-1. if "_hip_" in n else 1. for n in anames])
    out = obs.clone()
    for group,x in obs.items():
        if group == "priv_vel":
            out[group]=x*x.new_tensor([1.,-1.,1.])
            continue
        if group == "priv_height":
            out[group]=x.reshape(*x.shape[:-1],9,7).flip(-1).reshape_as(x)
            continue
        if group == "priv_feet":
            from .ref_fresh_privileged import mirror_privileged
            out[group]=mirror_privileged(group,x)
            continue
        if group == "height":
            f=x.reshape(*x.shape[:-1],3,127).clone()
            for start in (0,63):
                f[...,start:start+63]=f[...,start:start+63].reshape(*f.shape[:-1],9,7).flip(-1).flatten(-2)
            out[group]=f.flatten(-2)
            continue
        if group == "dynamics":
            out[group]=torch.cat((x[...,:4][...,[1,0,3,2]],x[...,4:5],
                x[...,5:17][...,qperm],x[...,17:29][...,qperm]),-1)
            continue
        offset=0
        for name,shape in zip(env.observation_manager.active_terms[group],
                              env.observation_manager.group_obs_term_dim[group]):
            size=math.prod(shape)
            v=x[...,offset:offset+size]
            if name in ("command","base_ang_vel","projected_gravity","base_lin_vel"):
                signs={"command":[1.,-1.,-1.],"base_ang_vel":[-1.,1.,-1.],
                       "projected_gravity":[1.,-1.,1.],"base_lin_vel":[1.,-1.,1.]}[name]
                v=(v.reshape(*v.shape[:-1],-1,3)*v.new_tensor(signs)).reshape_as(v)
            elif name in ("joint_pos","joint_vel","actions"):
                perm,sign=(aperm,asign) if name=="actions" else (qperm,qsign)
                v=(v.reshape(*v.shape[:-1],-1,12)[...,perm]*sign).reshape_as(v)
            elif name=="height_scan":
                assert size==187
                v=v.reshape(*v.shape[:-1],11,17).flip(-2).reshape_as(v)
            elif name in ("foot_air_time","foot_contact"):
                v=v[...,[1,0,3,2]]
            elif name=="foot_contact_forces":
                v=(v.reshape(*v.shape[:-1],4,3)[...,[1,0,3,2],:]*v.new_tensor([1.,-1.,1.])).reshape_as(v)
            else:
                raise ValueError(f"Unmapped observation: {group}/{name}")
            out[group][...,offset:offset+size]=v
            offset+=size
        assert offset==x.shape[-1]
    return out


def augment(env, obs=None, actions=None):
    native = env.unwrapped if hasattr(env,"unwrapped") else env
    mirrored_actions=None
    if actions is not None:
        names=native.action_manager.get_term("joint_pos").target_names
        signs=actions.new_tensor([-1. if "_hip_" in n else 1. for n in names])
        mirrored_actions=torch.cat((actions,actions[...,permutation(names)]*signs),0)
    return None if obs is None else torch.cat((obs,reflect(native,obs)),0), mirrored_actions
