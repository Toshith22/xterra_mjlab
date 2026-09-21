"""Analytic flat lanes, solid ramp prisms and vertical-riser stair flights.

Physics and training queries use the same profiles. Ramps are not staircased
heightfields. Importing this module does not require a simulator.
"""
import math
from functools import lru_cache
import numpy as np
from .ref_fresh_curriculum import LEVELS

LENGTH, WIDTH = 24., 6.
COLUMN_FAMILY = (0, 1, 2, 3, 4, 3, 4)


def tread_width(row, col):
    return (.25, .30, .35, .40)[(row+col)%4]


def profile(x, row, col):
    d = LEVELS[row]
    family = COLUMN_FAMILY[col]
    x = np.asarray(x)
    if family == 0:
        return np.zeros_like(x, dtype=float)
    if family in (1, 2):
        height = np.clip(x-4., 0., 4.)*math.tan(math.radians(d.slope_degrees))
        return height if family == 1 else 4.*math.tan(math.radians(d.slope_degrees))-height
    steps = np.clip(np.floor((x-4.+1e-9)/tread_width(row, col))+1, 0, 6)
    return d.riser*(steps if family == 3 else 6-steps)


@lru_cache(maxsize=16)
def surface_constants(device, dtype):
    import torch
    return (torch.tensor(COLUMN_FAMILY, device=device),
        torch.tensor([d.riser for d in LEVELS], device=device, dtype=dtype),
        torch.tensor([math.tan(math.radians(d.slope_degrees)) for d in LEVELS], device=device, dtype=dtype),
        torch.tensor([.25, .30, .35, .40], device=device, dtype=dtype))


def query_surface(xy):
    import torch
    x, y = xy[..., 0], xy[..., 1]
    row = (x/LENGTH).floor().long().clamp(0, len(LEVELS)-1)
    col = (y/WIDTH).floor().long().clamp(0, len(COLUMN_FAMILY)-1)
    local_x = x-row*LENGTH
    columns, risers, slopes, treads = surface_constants(xy.device, xy.dtype)
    family, riser, slope = columns[col], risers[row], slopes[row]
    tread = treads[(row+col)%4]
    steps = ((local_x-4.+1e-6)/tread).floor().add(1).clamp(0, 6)
    ramp = (local_x-4.).clamp(0, 4.)*slope
    z = torch.where(family == 1, ramp, torch.where(family == 2, 4.*slope-ramp,
        torch.where(family == 3, riser*steps, torch.where(family == 4, riser*(6-steps), 0.))))
    gradient = torch.where((local_x > 4.) & (local_x < 8.) & ((family == 1) | (family == 2)),
                           torch.where(family == 1, slope, -slope), 0.)
    normal = torch.stack((-gradient, torch.zeros_like(z), torch.ones_like(z)), -1)
    return z, normal/normal.norm(dim=-1, keepdim=True)
