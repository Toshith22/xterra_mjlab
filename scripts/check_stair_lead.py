"""Measure front-foot order in a recorded trial; no gait/reward changes."""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--recording",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    model=mujoco.MjModel.from_xml_path(str(root/"xterra_mjlab/assets/svanm2/xml/svanm2_mjlab.xml"))
    data=mujoco.MjData(model)
    ids=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,leg+"_foot") for leg in ("FL","FR")]
    result={}
    for family in ("stairs_up","stairs_down"):
        trace=json.loads((a.recording/f"{family}_trace.json").read_text())
        x_origin=trace[0]["qpos"][0]-trace[0]["x"]
        row=3
        col=int((trace[0]["qpos"][1]+21.)//6.)
        tread=(.25,.30,.35,.40)[(row+col)%4]
        first=np.full((6,2),np.nan)
        for record in trace:
            data.qpos[:]=record["qpos"]
            mujoco.mj_forward(model,data)
            xs=data.site_xpos[ids,0]-x_origin
            for edge in range(6):
                for foot in range(2):
                    if (np.isnan(first[edge,foot]) and xs[foot]>=4.+edge*tread
                            and record["foot_contact"][foot]):
                        first[edge,foot]=record["t"]
        rows=[]
        for edge,(left,right) in enumerate(first):
            lead=("unknown" if np.isnan(left) or np.isnan(right) else
                  "together" if abs(left-right)<=.021 else "FL" if left<right else "FR")
            rows.append(dict(edge=edge+1,FL_seconds=None if np.isnan(left) else left,
                FR_seconds=None if np.isnan(right) else right,lead=lead))
        angles=np.asarray([r["qpos"][7:] for r in trace])
        hips={}
        for leg in ("FL","FR","RL","RR"):
            joint=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,leg+"_hip_joint")
            qindex=model.jnt_qposadr[joint]-7
            values=np.rad2deg(angles[:,qindex])
            hips[leg]=dict(min_deg=float(values.min()),max_deg=float(values.max()))
        result[family]=dict(tread_m=tread,edges=rows,hip_ranges=hips,
            definition="First loaded forefoot at or beyond each riser; skipped treads can share a landing")
    result["limitation"]="One recorded trial per direction; cannot establish population bias or its cause."
    a.out.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:{"tread_m":v["tread_m"],"lead":[r["lead"] for r in v["edges"]]}
        for k,v in result.items() if isinstance(v,dict)}))


if __name__=="__main__": main()
