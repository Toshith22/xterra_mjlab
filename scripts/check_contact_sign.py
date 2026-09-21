"""Calibrate native contact sensor sign using a one-kilogram resting sphere."""
import mujoco
import numpy as np
spec=mujoco.MjSpec()
spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE,size=[1,1,.1])
body=spec.worldbody.add_body(name="ball",pos=[0,0,.1])
body.add_freejoint()
body.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE,size=[.05,0,0],mass=1.)
spec.add_sensor(name="force",type=mujoco.mjtSensor.mjSENS_CONTACT,
    objtype=mujoco.mjtObj.mjOBJ_BODY,objname="ball",intprm=[2,3,1])
model=spec.compile()
data=mujoco.MjData(model)
for _ in range(1000): mujoco.mj_step(model,data)
print("RESTING_BALL_SENSOR_FORCE",data.sensordata.tolist())
assert np.isclose(abs(data.sensordata[2]),9.81,rtol=.01)
assert data.sensordata[2]<0, "Bridge sign convention needs review"
