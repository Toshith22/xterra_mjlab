"""Re-render recorded states with clearer terrain colors; never resimulate."""
import argparse
import json
from pathlib import Path
import time
import imageio.v2 as imageio
import mujoco
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont
from xterra_mjlab.tasks.velocity.config.svanm2.isl_cfg import make_isl_cfg
from xterra_mjlab.tasks.velocity.config.svanm2.isl_native import IslEnv


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--recording",type=Path,required=True)
    a=p.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(112345)
    cfg=make_isl_cfg()
    cfg.seed=112345
    cfg.scene.num_envs=1
    cfg.viewer.width,cfg.viewer.height=768,512
    cfg.viewer.origin_type=cfg.viewer.OriginType.WORLD
    cfg.viewer.azimuth=105.
    cfg.viewer.elevation=-23.
    cfg.viewer.distance=2.1
    cfg.viewer.max_extra_envs=0
    env=IslEnv(cfg,device="cuda:0",render_mode="rgb_array")
    off=env._offline_renderer
    model,data=off._model,off._data
    terrain=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,"terrain")
    # Visual color only. No collision geometry, qpos, timing or policy changes.
    for i in range(model.ngeom):
        if model.geom_bodyid[i]==terrain:
            band=round((model.geom_pos[i,2]+model.geom_size[i,2])/.06)%2
            model.geom_rgba[i]=(.36,.44,.53,1.) if band else (.68,.74,.80,1.)
    font_path=Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    font=ImageFont.truetype(str(font_path),19) if font_path.exists() else ImageFont.load_default()
    try:
        traces = sorted(a.recording.glob("*_trace.json"))
        if not traces:
            deadline = time.time() + 1200
            while not traces:
                if time.time() > deadline: raise TimeoutError(f"No traces found in {a.recording}")
                time.sleep(5)
                traces = sorted(a.recording.glob("*_trace.json"))
        for path in traces:
            file_stem = path.stem.replace("_trace", "")
            rows = json.loads(path.read_text())
            output = a.recording / f"{file_stem}_clear.mp4"
            parts = file_stem.split("_")
            family = "_".join(parts[:2])
            task_type = parts[2] if len(parts) > 2 else "move"
            title = f"{family.replace('_',' ').title()} ({task_type.capitalize()})" if task_type != "move" else family.replace("_"," ").title()
            difficulty = "6 cm risers" if family.startswith("stairs") else "12 degree slope"
            rec_tag = a.recording.name.split("_")[-1]
            with imageio.get_writer(str(output),fps=25,codec="libx264",quality=8,
                                    macro_block_size=16,ffmpeg_log_level="error") as writer:
                for index,row in enumerate(rows):
                    if index%2 and index!=len(rows)-1: continue
                    data.qpos[:]=row["qpos"]
                    data.qvel[:]=0
                    mujoco.mj_forward(model,data)
                    off._cam.lookat[:]=np.asarray(row["qpos"][:3])+np.array([.3,0.,.03])
                    off.renderer.update_scene(data,camera=off._cam)
                    frame=Image.fromarray(off.render())
                    draw=ImageDraw.Draw(frame)
                    draw.rectangle((0,0,768,79),fill=(15,20,28))
                    draw.text((12,6),f"M2 | {title} | {difficulty} | {rec_tag}",font=font,fill="white")
                    draw.text((12,31),f"First attempt | t={row['t']:.1f}s | v={row['velocity'][0]:.2f} m/s",font=font,fill="white")
                    draw.text((12,54),"Recorded motion unchanged; enhanced terrain contrast",font=font,fill=(255,211,115))
                    writer.append_data(np.asarray(frame))
                    if index==450: frame.save(a.recording/f"{file_stem}_clear_preview.jpg")
            print("REPLAY_COMPLETE",file_stem,flush=True)
    finally:
        env.close()


if __name__=="__main__": main()
