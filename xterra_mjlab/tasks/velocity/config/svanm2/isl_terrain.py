"""ISL's straight, solid stair/ramp lanes, implemented natively in mjlab.

Only terrain changes here. The public SvanM2 XML and actuator remain untouched.
"""
from dataclasses import dataclass
import math

import mujoco
import numpy as np
from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainGeometry, TerrainOutput

# flat speed, slope degrees, slope speed, riser, stair speed, force N, torque Nm
LEVELS = (
    (.40, 0., .40, 0., .30, 0., 0.),
    (.65, 0., .40, 0., .30, 15., 2.),
    (.85, 8., .50, 0., .30, 20., 3.),
    (1., 12., .65, .06, .40, 25., 4.),
    (1.20, 16., .80, .10, .50, 30., 5.),
    (1.50, 20., 1., .14, .60, 35., 6.),
    (1.50, 24., 1.20, .16, .80, 40., 8.),
    (1.50, 28., 1.50, .18, 1., 45., 10.),
    (1.50, 28., 1.50, .18, 1.25, 45., 10.),
    (1.50, 28., 1.50, .18, 1.50, 45., 10.),
)
FAMILIES = ("flat", "slope_up", "slope_down", "stairs_up", "stairs_down")


def profile(x, level, family, tread=.30):
    x = np.asarray(x)
    _, angle, _, riser, *_ = LEVELS[level]
    if family == "flat":
        return np.zeros_like(x, dtype=float)
    if family.startswith("slope"):
        z = np.clip(x-4., 0., 4.)*math.tan(math.radians(angle))
        return z if family == "slope_up" else 4.*math.tan(math.radians(angle))-z
    steps = np.clip(np.floor((x-4.+1e-9)/tread)+1, 0, 6)
    return riser*(steps if family == "stairs_up" else 6-steps)


@dataclass(kw_only=True)
class IslLaneCfg(SubTerrainCfg):
    family: str = "flat"
    level: int = 0
    tread: float = .30
    column: int | None = None

    def function(self, difficulty, spec, rng):
        del rng
        if self.column is not None:
            self.level = round(difficulty*9)
            self.tread = (.25, .30, .35, .40)[(self.level+self.column)%4]
        assert self.family in FAMILIES and 0 <= self.level < len(LEVELS)
        length, width = self.size
        assert length >= 12. and width >= 2.
        body = spec.body("terrain")
        geometries = []

        def box(left, right, top, bottom=-.10):
            geom = body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=((left+right)/2, width/2, (top+bottom)/2),
                size=((right-left)/2, width/2, (top-bottom)/2),
                contype=1, conaffinity=1, group=0,
                rgba=(.45, .48, .52, 1.))
            geometries.append(TerrainGeometry(geom=geom))

        if self.family.startswith("stairs"):
            cuts = [0., *[4.+i*self.tread for i in range(6)], length]
            for left, right in zip(cuts, cuts[1:]):
                box(left, right, float(profile((left+right)/2, self.level, self.family, self.tread)))
        elif self.family.startswith("slope") and LEVELS[self.level][1] > 0:
            low, high = float(profile(4., self.level, self.family)), float(profile(8., self.level, self.family))
            box(0., 4., low)
            box(8., length, high)
            vertices = np.array([[x, y, z] for y in (0., width)
                for x, z in ((4., -.1), (8., -.1), (4., low), (8., high))])
            # A convex mesh preserves an actual planar ramp, not a heightfield staircase.
            mesh = spec.add_mesh(name=f"isl_ramp_{len(list(spec.meshes))}", uservert=vertices.flatten())
            geom = body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mesh.name,
                contype=1, conaffinity=1, group=0, rgba=(.45, .48, .52, 1.))
            geometries.append(TerrainGeometry(geom=geom))
        else:
            box(0., length, 0.)
        return TerrainOutput(origin=np.array([1.5, width/2,
            float(profile(1.5, self.level, self.family, self.tread))]), geometries=geometries)
