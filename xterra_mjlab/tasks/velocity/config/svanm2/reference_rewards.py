"""Foot-state costs without touchdown bonuses or incentives to freeze a swing.

Functions are pure tensor operations for adversarial reward checks independent
of Genesis. The environment supplies FOOT POINT motion, not calf COM motion.
"""
import torch


def quaternion_yaw(quaternion):
    """World yaw for Genesis quaternions in w,x,y,z order."""
    q = quaternion/quaternion.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))


def heading_command(reference_yaw_rate, target_heading, quaternion, stiffness=.5, limit=.5):
    """Unitree/Isaac-Lab style heading outer loop, preserving the 3-D command API."""
    error = torch.atan2(torch.sin(target_heading-quaternion_yaw(quaternion)),
                        torch.cos(target_heading-quaternion_yaw(quaternion)))
    return (reference_yaw_rate+stiffness*error).clamp(-limit, limit), error


def reference_is_stopped(reference_command, tolerance=.02):
    """Task state follows the user's command, not an internal heading correction."""
    return reference_command.norm(dim=-1) < tolerance


def phase_aligned_contact_cost(history, cursor, lags, ema, alpha):
    """Update phase-lagged left/right contact mismatch and return its best lag.

    history is a circular (time, env, FL/FR/RL/RR) boolean buffer. Comparing
    current contacts with swapped delayed contacts avoids the invalid rule that
    left and right legs must contact at the same instant.
    """
    if history.ndim != 3 or history.shape[-1] != 4:
        raise ValueError("expected time x environment x four-foot contact history")
    current = history[(cursor-1) % history.shape[0]]
    swap = torch.tensor([1, 0, 3, 2], device=history.device)
    delayed = torch.stack([history[(cursor-1-int(lag)) % history.shape[0]][:, swap]
                           for lag in lags])
    mismatch = (delayed != current[None]).float().mean(-1).transpose(0, 1)
    ema.mul_(1-alpha).add_(mismatch*alpha)
    return ema.amin(-1)


def abduction_costs(joint_error, joint_velocity, commands, risk, support_heights):
    """Soft abduction preference; never lock joints or impose amplitude ratios."""
    hip_error = joint_error[:, [0, 3, 6, 9]]
    hip_velocity = joint_velocity[:, [0, 3, 6, 9]]
    maneuver = torch.maximum(commands[:, 1].abs()/.3, commands[:, 2].abs()/.6).clamp(0., 1.)
    uneven = ((support_heights.amax(-1)-support_heights.amin(-1))/.12).clamp(0., 1.)
    release = torch.maximum(risk, torch.maximum(maneuver, uneven))
    gate = 1.-.9*release
    # Independent costs cannot be satisfied by increasing thigh/knee movement.
    posture = ((hip_error.abs()-.06).clamp_min(0.)/.15).square().mean(-1)*gate
    motion = hip_velocity.square().mean(-1)*gate
    return posture, motion


def support_margin(feet_xy, contact, predicted_xy):
    """Signed distance inside the convex hull for >=3 loaded feet.

    Two-foot dynamic support is not a polygon: return NaN instead of inventing
    an area or treating ordinary two-contact trotting as automatic failure.
    """
    pairs = torch.tensor([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]], device=feet_xy.device)
    a, b = feet_xy[:, pairs[:, 0]], feet_xy[:, pairs[:, 1]]
    edge = b-a
    other = feet_xy[:, None]-a[:, :, None]
    cross = edge[:, :, None, 0]*other[..., 1]-edge[:, :, None, 1]*other[..., 0]
    positive = ((cross >= -1e-6) | ~contact[:, None]).all(-1)
    negative = ((cross <= 1e-6) | ~contact[:, None]).all(-1)
    length = edge.norm(dim=-1)
    boundary = contact[:, pairs[:, 0]] & contact[:, pairs[:, 1]] & (positive ^ negative) & (length > .02)
    point = predicted_xy[:, None]-a
    distance = (edge[..., 0]*point[..., 1]-edge[..., 1]*point[..., 0])/length.clamp_min(.02)
    distance *= torch.where(positive, 1., -1.)
    margin = distance.masked_fill(~boundary, float("inf")).amin(-1)
    return torch.where((contact.sum(-1) >= 3) & boundary.any(-1), margin, float("nan"))


def foot_costs(height, velocity, normal, contact, previous_contact, previous_normal_speed,
               air_age, stopped, recovery, valid):
    normal_speed = (velocity*normal).sum(-1)
    tangent = velocity-normal_speed[..., None]*normal
    tangent_speed2 = tangent.square().sum(-1)
    # Sliding costs never disappear during recovery or zero command. Vertical
    # lowering does not incur this cost; motion parallel to a slope does.
    slip = (tangent_speed2*contact).sum(-1)
    low = ((.065-height)/.065).clamp(0., 1.)
    # Near-ground horizontal travel is scraping even before the force sensor
    # crosses its contact threshold. No permanent height target during descent.
    scuff = (tangent_speed2*low.square()).sum(-1)
    touchdown = contact & ~previous_contact & valid[:, None]
    impact = ((-previous_normal_speed-.5).clamp_min(0.).square()*touchdown).sum(-1)
    # Delay gives normal swing/landing time; no reward for repeatedly touching.
    placement_gate = stopped*(1.-.8*recovery)
    grace = torch.where(stopped, .4, .8)
    use_gate = torch.where(stopped, 1., .5)*(1.-.8*recovery)
    hover = (((air_age-grace[:, None])/.8).clamp(0., 2.)*(~contact)).sum(-1)*use_gate
    over_lift = ((height-.22).clamp_min(0.).square()*(~contact)).sum(-1)
    # All-contact quietness is a separate objective. Unplaced feet remain free
    # to lower, but scraping/impact/overswing costs above still apply.
    quiet = (velocity.square().sum(-1)*contact).sum(-1)*placement_gate
    return {"slip": slip*valid, "scuff": scuff*valid, "impact": impact*valid,
            "hover": hover, "over_lift": over_lift, "quiet": quiet*valid}


def quiet_support(contact, velocity, base_velocity, angular_velocity, clearance, risk):
    return (contact.all(-1) & (velocity.square().sum(-1).mean(-1) < .10**2)
            & (base_velocity[:, :2].norm(dim=-1) < .08)
            & (angular_velocity.norm(dim=-1) < .25)
            & (clearance > .22) & (clearance < .42) & (risk < .25))


def support_loss_pairs(height, normal_force):
    # Foot order FL, FR, RL, RR. This is a conservative calf-force/foot-height
    # proxy, not a claim of exact contact sensing. Require opposite support.
    airborne = (height > .06) & (normal_force < 1.)
    supporting = (height.abs() < .055) & (normal_force > 2.)
    pairs = ((0, 1, 2, 3), (2, 3, 0, 1), (0, 2, 1, 3), (1, 3, 0, 2))
    return torch.stack([airborne[:, a] & airborne[:, b] & (supporting[:, c] | supporting[:, d])
                        for a, b, c, d in pairs], -1)
