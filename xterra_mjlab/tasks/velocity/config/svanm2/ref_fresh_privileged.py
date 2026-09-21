"""Versioned critic-only observations; no extra randomization or reward changes.

Terrain/feet follow Unitree's asymmetric velocity-task observation categories.
Dynamics expose existing simulator parameters, not invented motor/COM estimates.
"""
import torch
from .ref_stair_sensing import rotation_matrix

GROUPS = ("priv_height", "priv_feet", "priv_dynamics")
LEG_SWAP = [1, 0, 3, 2]  # FR, FL, RR, RL
JOINT_SWAP = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]


def privileged_manifest():
    return dict(version=1, groups=dict(priv_vel=3, priv_height=63, priv_feet=24, priv_dynamics=29),
        height="yaw-aligned 9x7 grid x=[-0.6,1.0], y=[-0.45,0.45] m; (terrain_z-base_z)/1m clipped +/-1",
        feet="FR/FL/RR/RL: height/0.3m (4), airtime/1s (4), contact proxy (4), body-frame calf net contact force/100N (12)",
        dynamics="calf friction ratios minus 1 (4), trunk payload/2kg (1), kp/nominal-1 (12), kd/nominal-1 (12)",
        limitations="calf contact force and geometric foot-contact proxy, not isolated sole force; friction is robot-side ratio, not effective pair coefficient",
        excluded=["terrain ID", "curriculum level", "future push schedule", "unmodelled COM/motor strength/delay"],
        actor_access=False)


def terrain_offsets(device):
    x, y = torch.meshgrid(torch.linspace(-.6, 1., 9, device=device),
                          torch.linspace(-.45, .45, 7, device=device), indexing="ij")
    return torch.stack((x.flatten(), y.flatten()), -1)


def clean_heights(base_pos, base_quat, offsets, query):
    rotation = rotation_matrix(base_quat)
    forward = rotation[:, :2, 0]
    forward = forward/forward.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    left = torch.stack((-forward[:, 1], forward[:, 0]), -1)
    xy = (base_pos[:, None, :2]+offsets[None, :, :1]*forward[:, None]
          +offsets[None, :, 1:]*left[:, None])
    return (query(xy)-base_pos[:, 2:3]).clamp(-1., 1.)


def encode_feet(height, air_age, contact, forces_world, base_quat, valid):
    force_body = torch.einsum("bji,bpj->bpi", rotation_matrix(base_quat), forces_world)
    features = torch.cat(((height/.3).clamp(-1., 1.), air_age.clamp(0., 1.),
        contact.to(height.dtype), (force_body/100.).clamp(-5., 5.).flatten(-2)), -1)
    # A reset teleports with deferred forward kinematics. Never return the
    # outgoing episode's contacts/geometry as the new episode's observation.
    return torch.where(valid[:, None], features, 0.)


def encode_dynamics(friction_ratio, payload, kp_ratio, kd_ratio):
    if (friction_ratio.shape[-1], payload.shape[-1], kp_ratio.shape[-1], kd_ratio.shape[-1]) != (4, 1, 12, 12):
        raise ValueError("unexpected privileged dynamics layout")
    return torch.cat((friction_ratio-1., payload/2., kp_ratio-1., kd_ratio-1.), -1)


def mirror_privileged(key, x):
    if key == "priv_feet":
        if x.shape[-1] != 24:
            raise ValueError("expected 24 foot features")
        scalars = [x[..., start:start+4][..., LEG_SWAP] for start in (0, 4, 8)]
        forces = x[..., 12:].reshape(*x.shape[:-1], 4, 3)[..., LEG_SWAP, :]
        forces = forces*x.new_tensor([1., -1., 1.])  # force is polar
        return torch.cat((*scalars, forces.flatten(-2)), -1)
    if key == "priv_dynamics":
        if x.shape[-1] != 29:
            raise ValueError("expected 29 dynamics features")
        # Gain magnitudes swap joints WITHOUT the position/action hip sign flip.
        return torch.cat((x[..., :4][..., LEG_SWAP], x[..., 4:5],
            x[..., 5:17][..., JOINT_SWAP], x[..., 17:29][..., JOINT_SWAP]), -1)
    raise ValueError(f"unknown privileged group: {key}")
