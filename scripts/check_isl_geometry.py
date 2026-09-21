"""Check physical MuJoCo collision heights against every ISL lane profile."""
import json
from pathlib import Path
import mujoco
import numpy as np
from xterra_mjlab.tasks.velocity.config.svanm2.isl_terrain import FAMILIES, IslLaneCfg, profile

results = []
for level in range(10):
    for column, family in enumerate(("flat","slope_up","slope_down","stairs_up","stairs_down","stairs_up","stairs_down")):
        spec = mujoco.MjSpec()
        spec.worldbody.add_body(name="terrain")
        cfg = IslLaneCfg(family=family, column=column, size=(24., 6.))
        cfg.function(level/9., spec, np.random.default_rng(0))
        model = spec.compile()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        # Sample immediately before/after each riser as well as both landings.
        xs = [1.5, 3.99, 4.01, 7.9, 8.1, 20.]
        xs += [4.+i*cfg.tread+d for i in range(6) for d in (-.001, .001)]
        errors = []
        for x in xs:
            distance = mujoco.mj_ray(model, data, np.array([x, 3., 8.]),
                np.array([0., 0., -1.]), None, 1, -1, np.array([-1], dtype=np.int32))
            assert distance >= 0, (level, family, x, "missing collision")
            errors.append(abs(8.-distance-float(profile(x, level, family,cfg.tread))))
        assert max(errors) < 1e-5, (level, family, max(errors))
        results.append(dict(level=level, family=family, tread=cfg.tread,max_height_error=max(errors)))
print(json.dumps(dict(passed=True, collision_cases=len(results),max_height_error=max(r["max_height_error"] for r in results))))
