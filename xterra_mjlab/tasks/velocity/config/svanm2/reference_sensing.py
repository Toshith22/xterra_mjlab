"""Finite-view height sensor surrogate plus explicit hardware-facing input format.

This ray/heightfield surrogate is NOT rendered RGB-D, calibrated Orbbec data, or
an implementation of Miki/StairMaster. Its latent interface can accept measured
elevation samples; unknown, delayed and occluded samples must remain marked.
"""
import math
import torch

POINTS = 63
FRAMES = 3
FRAME_DIM = 2*POINTS+1
FEATURE_DIM = FRAMES*FRAME_DIM


def perceptive_latent(base, features, encoder):
    if features.shape[-1] != FEATURE_DIM:
        raise ValueError("invalid stair observation shape")
    frames = features.reshape(*features.shape[:-1], FRAMES, FRAME_DIM)
    coverage = frames[..., POINTS:2*POINTS].mean(dim=(-1, -2)).clamp(0., 1.)
    return base+.3*torch.tanh(encoder(features))*coverage[..., None]


def scan_offsets(device="cpu"):
    x, y = torch.meshgrid(torch.linspace(.3, 1.5, 9, device=device),
                          torch.linspace(-.45, .45, 7, device=device), indexing="ij")
    return torch.stack((x.flatten(), y.flatten()), dim=-1)


def rotation_matrix(q):
    w, x, y, z = q.unbind(-1)
    return torch.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                        2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                        2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), -1).reshape(-1, 3, 3)


def encode_height_history(heights, valid, ages):
    """Pure tensor input contract: metres relative to capture-time body z, seconds.

    No simulator lookup occurs here. Real deployment must supply measured
    heights, observed masks and timestamp-derived ages in this exact layout.
    """
    if heights.shape[-2:] != (FRAMES, POINTS) or valid.shape != heights.shape:
        raise ValueError("requires three frames of 63 heights and masks")
    if ages.shape != heights.shape[:-1]:
        raise ValueError("requires one age per frame")
    good = (valid.bool() & torch.isfinite(heights) & torch.isfinite(ages[..., None])
            & (ages[..., None] >= 0) & (ages[..., None] <= .3))
    h = torch.where(good, heights.clamp(-1., 1.), 0.)
    age = torch.nan_to_num(ages/.3, nan=1., posinf=1., neginf=1.).clamp(0., 1.)
    return torch.cat((h, good.to(h.dtype), age[..., None]), -1).flatten(-2)


class HeightSensorSurrogate:
    def __init__(self, n, device, seed=67001):
        self.device, self.n, self.tick = device, n, 0
        self.offsets = scan_offsets(device)
        self.rng = torch.Generator(device=device).manual_seed(seed)
        self.h = torch.zeros((n, FRAMES, POINTS), device=device)
        self.mask = torch.zeros_like(self.h, dtype=torch.bool)
        self.stamps = torch.full((n, FRAMES), -100., device=device)
        self.pending_h = torch.zeros((n, POINTS), device=device)
        self.pending_mask = torch.zeros((n, POINTS), device=device, dtype=torch.bool)
        self.pending_due = torch.full((n,), -1, device=device, dtype=torch.long)
        self.pending_stamp = torch.zeros(n, device=device)

    def reset(self, ids):
        self.mask[ids] = False
        self.stamps[ids] = -100.
        self.pending_due[ids] = -1

    def sample_world(self, env):
        # Truth is used ONLY inside simulated sensor geometry, never returned
        # directly to the actor as a privileged map.
        R = rotation_matrix(env.base_quat)
        forward = R[:, :2, 0]
        forward = forward/forward.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        left = torch.stack((-forward[:, 1], forward[:, 0]), -1)
        xy = (env.base_pos[:, None, :2]+self.offsets[None, :, 0:1]*forward[:, None]
              +self.offsets[None, :, 1:2]*left[:, None])
        z = env.terrain_heights(xy)
        targets = torch.cat((xy, z[..., None]), -1)
        camera = env.base_pos+torch.einsum("bij,j->bi", R,
                    torch.tensor([.2, 0., .15], device=self.device))
        delta = targets-camera[:, None]
        body_delta = torch.einsum("bji,bpj->bpi", R, delta)
        c, s = math.cos(math.pi/6), math.sin(math.pi/6)
        optical_z = c*body_delta[..., 0]-s*body_delta[..., 2]
        optical_y = s*body_delta[..., 0]+c*body_delta[..., 2]
        visible = ((optical_z > .15) & (optical_z < 3.)
                   & (body_delta[..., 1].abs() < optical_z)
                   & (optical_y.abs() < optical_z*math.tan(math.radians(32.5))))
        # A finite set of intervening surface tests rejects blocked top samples.
        fraction = torch.linspace(.08, .96, 12, device=self.device)
        rays = camera[:, None, None, :]+delta[:, :, None, :]*fraction[None, None, :, None]
        visible &= (env.terrain_heights(rays[..., :2]) <= rays[..., 2]+.015).all(dim=-1)
        origin = env._terrain_origin
        extent = torch.tensor(env.terrain_arena.size, device=self.device)
        visible &= ((xy >= origin[:2]) & (xy <= origin[:2]+extent)).all(-1)
        # Correlated vertical pose error plus per-sample measurement error.
        noise = .01*torch.randn(z.shape, device=self.device, generator=self.rng)
        bias = .015*torch.randn((self.n, 1), device=self.device, generator=self.rng)
        visible &= torch.rand(z.shape, device=self.device, generator=self.rng) >= .1
        return z-env.base_pos[:, None, 2]+noise+bias, visible

    def update(self, env):
        now = self.tick*.02
        due = self.pending_due == self.tick
        if due.any():
            self.h[due, :-1] = self.h[due, 1:].clone()
            self.mask[due, :-1] = self.mask[due, 1:].clone()
            self.stamps[due, :-1] = self.stamps[due, 1:].clone()
            self.h[due, -1] = self.pending_h[due]
            self.mask[due, -1] = self.pending_mask[due]
            self.stamps[due, -1] = self.pending_stamp[due]
            self.pending_due[due] = -1
        if self.tick % 5 == 0:  # 10 Hz acquisition, 40 ms latency
            h, mask = self.sample_world(env)
            received = torch.rand((self.n,), device=self.device, generator=self.rng) >= .05
            self.pending_h[received] = h[received]
            self.pending_mask[received] = mask[received]
            self.pending_stamp[received] = now
            self.pending_due[received] = self.tick+2
        self.tick += 1
        return encode_height_history(self.h, self.mask, now-self.stamps)
