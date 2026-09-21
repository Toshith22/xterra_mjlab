"""Record first-attempt terrain rollouts, without altering any training run."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import torch
import imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.runner import MjlabOnPolicyRunner
from xterra_mjlab.tasks.velocity.config.svanm2.isl_cfg import make_isl_cfg
from xterra_mjlab.tasks.velocity.config.svanm2.isl_native import IslEnv,task
from xterra_mjlab.tasks.velocity.config.svanm2.rl_cfg import svanm2_ppo_runner_cfg


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--seed",type=int,default=112345)
    p.add_argument("--max-seconds",type=float,default=30.)
    p.add_argument("--cases",type=str,default="traverse",choices=["traverse","incline_stop_stand","all"])
    a=p.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    a.out.mkdir(parents=True,exist_ok=False)
    shutil.copy2(__file__,a.out/"record_isl.py")
    cfg=make_isl_cfg()
    cfg.seed=a.seed
    cfg.scene.num_envs=1
    cfg.auto_reset=False
    cfg.viewer.width,cfg.viewer.height=768,512
    cfg.viewer.origin_type=cfg.viewer.OriginType.WORLD
    cfg.viewer.distance=2.6
    cfg.viewer.elevation=-24.
    cfg.viewer.azimuth=125.
    cfg.viewer.max_extra_envs=0
    env=IslEnv(cfg,device="cuda:0",render_mode="rgb_array")
    wrapped=RslRlVecEnvWrapper(env,clip_actions=6.)
    agent=svanm2_ppo_runner_cfg()
    agent.obs_groups={"actor":("actor","height"),
        "critic":("actor","priv_vel","priv_height","priv_feet","dynamics","height")}
    runner=MjlabOnPolicyRunner(wrapped,asdict(agent),None,"cuda:0")
    saved=torch.load(a.checkpoint,map_location="cuda:0",weights_only=True)
    runner.alg.load(saved,{"actor":True,"critic":True,"optimizer":False,"iteration":False,"rnd":False},True)
    policy=runner.get_inference_policy(device="cuda:0")
    s=task(env)
    s.ledger.level=3
    s.evaluation=True
    s.trials_per_case=1
    font_path="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font=ImageFont.truetype(font_path,19) if Path(font_path).exists() else ImageFont.load_default()
    results=[]
    if a.cases == "traverse":
        scenarios = [("stairs_up","move"),("stairs_down","move"),("slope_up","move"),("slope_down","move")]
    elif a.cases == "incline_stop_stand":
        scenarios = [("slope_up","stop"),("slope_up","stand"),("stairs_up","stop"),("stairs_up","stand")]
    else:
        scenarios = [("stairs_up","move"),("stairs_down","move"),("slope_up","move"),("slope_down","move"),
                     ("slope_up","stop"),("slope_up","stand"),("stairs_up","stop"),("stairs_up","stand")]
    try:
        for family, task_type in scenarios:
            case=dict(level=3,family=family,task=task_type,pulse_family=0,repeat_pulses=False)
            s.evaluation_batch=[case]
            env.reset()
            s.completed_records.clear()
            obs=wrapped.get_observations()
            frames=0
            trace=[]
            done=False
            title=f"{family.replace('_',' ').title()} ({task_type.capitalize()})" if task_type != "move" else family.replace("_"," ").title()
            difficulty="6 cm risers" if family.startswith("stairs") else "12 degree slope"
            file_stem=f"{family}_{task_type}" if task_type != "move" else family
            video=a.out/f"{file_stem}.mp4"
            with imageio.get_writer(str(video),fps=25,codec="libx264",quality=8,
                                    macro_block_size=16,ffmpeg_log_level="error") as writer:
                for tick in range(round(a.max_seconds/env.step_dt)):
                    obs,reward,dones,_=wrapped.step(policy(obs))
                    done=bool(dones[0])
                    pos=s.base_pos[0].cpu().numpy()
                    local_x=float(pos[0]-s.terrain_level[0].item()*24.)
                    measured=s._measure()
                    trace.append(dict(t=(tick+1)*env.step_dt,x=local_x,
                        velocity=s.base_lin_vel[0].tolist(),
                        command=s.reference_commands[0].tolist(),
                        roll_pitch=s.base_euler[0,:2].tolist(),
                        foot_height=measured["height"][0].tolist(),
                        foot_contact=measured["contact"][0].tolist(),
                        qpos=env.sim.data.qpos[0].tolist()))
                    if tick%2==0 or done:
                        world=s.native.data.root_link_pos_w[0].cpu().numpy()
                        env._offline_renderer._cam.lookat[:]=world+np.array([.3,0.,.05])
                        pixels=env.render()
                        frame=Image.fromarray(pixels)
                        draw=ImageDraw.Draw(frame)
                        draw.rectangle((0,0,768,79),fill=(15,20,28))
                        ckpt_tag=a.checkpoint.stem.replace("checkpoint_00","update ").replace("checkpoint_","update ")
                        draw.text((12,6),f"M2 | {title} | {difficulty} | {ckpt_tag}",font=font,fill="white")
                        draw.text((12,31),f"First attempt | t={(tick+1)*env.step_dt:.1f}s | v={s.base_lin_vel[0,0].item():.2f} m/s",font=font,fill="white")
                        draw.text((12,54),"Diagnostic rollout: no pushes; failures retained",font=font,fill=(255,211,115))
                        writer.append_data(np.asarray(frame))
                        if frames==100: frame.save(a.out/f"{file_stem}_preview.jpg")
                        frames+=1
                    if done: break
            s._record_episodes(torch.tensor([0],device=env.device),s._out_of_bounds())
            row=s.completed_records[-1]
            row.update(video=video.name,frames=frames,last_x=local_x,termination=done,
                clipped_by_video_limit=not done)
            results.append(row)
            (a.out/f"{file_stem}_trace.json").write_text(json.dumps(trace,allow_nan=False))
            print("VIDEO_COMPLETE",file_stem,dict(seconds=frames/25.,crossed=row["crossed"],
                fall=row["physical_fall"],success=row["success"],reasons=row["failure_reasons"]),flush=True)
        report=dict(checkpoint=str(a.checkpoint),sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
            seed=a.seed,level=3,selection="first trial of each terrain, no reroll or success filtering",
            real_time_playback=True,robot="unchanged public SvanM2",episodes=results)
        (a.out/"summary.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    finally:
        env.close()


if __name__=="__main__": main()
