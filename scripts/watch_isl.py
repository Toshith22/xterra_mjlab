"""Low-cost five-minute status watcher; alerts queue back into this Codex chat."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time

ROOT="runs"
RUNS=[]


def remote_status(root=ROOT, runs=RUNS, trainer="continue_isl.py"):
    result=[]
    for name in runs:
        folder=Path(root)/"runs"/name
        path=folder/"progress.json"
        if not path.exists():
            campaign=folder.parent/(name.removesuffix("_continuous")+"_campaign.json")
            if campaign.exists():
                row=json.loads(campaign.read_text())
                cmd=Path("/proc")/str(row.get("pid",-1))/"cmdline"
                row["process_alive"]=cmd.exists() and b"start_isl_campaign.py" in cmd.read_bytes()
                row["run"]=name
                row["heartbeat"]=row["time"]
                if row["status"] in ("bootstrap","continuous"):
                    row["status"]="evaluating"  # Bootstrap includes its bounded acceptance evaluation.
                result.append(row)
                continue
            result.append(dict(run=name,status="starting"))
            continue
        row=json.loads(path.read_text())
        row["run"]=name
        cmd=Path("/proc")/str(row.get("pid",-1))/"cmdline"
        row["process_alive"]=cmd.exists() and trainer.encode() in cmd.read_bytes()
        if row["status"] == "completed": row["status"] = "complete"
        alert=folder/"ALERT.json"
        if alert.exists(): row["alert"]=json.loads(alert.read_text())
        result.append(row)
    return result


def alerts(rows,now,started):
    findings=[]
    for r in rows:
        name=r["run"]
        if r["status"]=="complete": continue
        if r.get("alert") or r["status"]=="stopped_error":
            findings.append(name+": "+str(r.get("alert",r.get("error","training error"))))
        elif r["status"]=="starting":
            if now-started>900: findings.append(name+": no progress file after 15 minutes")
        elif not r.get("process_alive",False):
            findings.append(name+": training process exited before completion")
        elif now-r.get("heartbeat",now)>(7800 if r["status"]=="evaluating" else 900):
            findings.append(name+": progress heartbeat stalled")
        elif r.get("completed_updates",0)>1000 and r.get("mean_episode_seconds",20)<2:
            findings.append(name+": episodes remain under 2 seconds after 1000 updates; inspect physical failures")
        elif r.get("elapsed_s",0)>7200 and r.get("mean_terrain_level",99)<1:
            findings.append(name+": terrain learning remains below mean level 1 after two hours; inspect learning progress")
    return findings


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--remote-status",action="store_true")
    p.add_argument("--thread")
    p.add_argument("--state",type=Path)
    p.add_argument("--once",action="store_true")
    p.add_argument("--no-notify",action="store_true")
    p.add_argument("--host",default="localhost")
    p.add_argument("--root",default=ROOT)
    p.add_argument("--runs",nargs="+",default=RUNS)
    p.add_argument("--python",default="python3")
    p.add_argument("--key",default="")
    p.add_argument("--trainer",default="continue_isl.py")
    p.add_argument("--duration-hours",type=float,default=48.)
    a=p.parse_args()
    if a.remote_status:
        print(json.dumps(remote_status(a.root,a.runs,a.trainer),allow_nan=False))
        return
    if not a.state or not a.thread: p.error("--state and --thread required")
    started=time.time()
    sent=set()
    failures=0
    while time.time()-started<a.duration_hours*3600:
        now=time.time()
        try:
            cmd=shlex.join([a.python,
                a.root+"/scripts/watch_isl.py","--remote-status","--root",a.root,"--trainer",a.trainer,"--runs",*a.runs])
            proc=subprocess.run(["ssh","-o","BatchMode=yes","-o","ConnectTimeout=10",
                *(["-i",a.key] if a.key else []),a.host,cmd],
                capture_output=True,text=True,timeout=45,check=True)
            rows=json.loads(proc.stdout)
            failures=0
            issues=alerts(rows,now,started)
        except Exception as exc:
            failures+=1
            rows=[]
            issues=[a.host+" monitoring unreachable for three checks: "+str(exc)] if failures>=3 else []
        done=bool(rows) and all(r["status"]=="complete" for r in rows)
        if done: issues.append(a.host+": public M2 runs reached their configured target; review results.")
        delivery=[]
        for issue in issues:
            if issue in sent: continue
            message=("Automated M2 watcher alert on "+a.host+" (user requested periodic monitoring): "+issue+
                ". Inspect the scoped runs under "+a.root+"/runs. Report important findings briefly. "
                "Do not restart, alter rewards, or broaden the training scope without user direction.")
            if not a.no_notify:
                proc=subprocess.run(["codex","queue","--thread",a.thread,
                    "--message",message],capture_output=True,text=True,timeout=45)
                delivery.append(dict(issue=issue,returncode=proc.returncode,
                    output=(proc.stdout+proc.stderr)[-2000:]))
                if proc.returncode==0: sent.add(issue)
            else: delivery.append(dict(issue=issue,dry_run=True))
        state=dict(last_check=now,rows=rows,issues=issues,delivery=delivery,
            next_check=now+300,expires=started+a.duration_hours*3600,notification_transport="codex queue")
        tmp=a.state.with_suffix(".tmp")
        tmp.write_text(json.dumps(state,indent=2)+"\n")
        tmp.replace(a.state)
        print(json.dumps(dict(time=now,issues=issues,states=[r["status"] for r in rows])),flush=True)
        if a.once or done: break
        time.sleep(300)


if __name__=="__main__": main()
