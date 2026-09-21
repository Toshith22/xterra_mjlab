"""Frozen ISL V7 mean-action, first-episode development suite on native M2."""
import torch
from xterra_mjlab.tasks.velocity.config.svanm2.isl_native import task
from xterra_mjlab.tasks.velocity.config.svanm2.ref_evaluation import summarize
from xterra_mjlab.tasks.velocity.config.svanm2.ref_fresh_curriculum import families
from xterra_mjlab.tasks.velocity.config.svanm2.ref_fresh_objective_review import assess_objectives
from xterra_mjlab.tasks.velocity.config.svanm2.ref_stair_sensing import HeightSensorSurrogate


@torch.inference_mode()
def evaluate(env,wrapped,policy,seed,model_hash,through=0,trials=64):
    s = task(env)
    s.evaluation = True
    s.trials_per_case = trials
    torch.manual_seed(seed)
    s.fresh_rng.manual_seed(seed+31000)
    s.height_sensor = HeightSensorSurrogate(env.num_envs,env.device,seed+71000)
    # Independent reset/dynamics seed; no optimizer or training-data update.
    env.event_manager.apply(mode="startup")
    scenarios = [dict(level=l,family=f,task=t,pulse_family=p,repeat_pulses=bool(p))
        for l in range(through+1) for f in families(l) for t in ("move","stop","stand")
        for p in ((0,) if l==0 else (0,1))]
    if through==9:
        scenarios += [dict(level=9,family="flat",task="stand",pulse_family=p,repeat_pulses=True)
                      for p in range(2,9)]
    batch_size = env.num_envs//trials
    assert batch_size>0 and env.num_envs%trials==0
    cases=[]
    for offset in range(0,len(scenarios),batch_size):
        batch=scenarios[offset:offset+batch_size]
        s.evaluation_batch=batch+[batch[-1]]*(batch_size-len(batch))
        env.reset()
        s.completed_records.clear()
        obs=wrapped.get_observations()
        first={}
        for _ in range(2050):
            obs,_,_,_=wrapped.step(policy(obs))
            for row in s.completed_records:
                if row["env_id"]<len(batch)*trials:
                    first.setdefault(row["env_id"],row)
            s.completed_records.clear()
            if len(first)==len(batch)*trials:
                break
        for i,scenario in enumerate(batch):
            rows=[row for eid,row in first.items() if eid//trials==i]
            cases.append(dict(scenario=scenario,requested_trials=trials,episodes=rows))
    result=dict(seed=seed,model_sha256=model_hash,smoke=False,diagnostic_only=False,
        evaluator_version="native_m2_ISL_V7_acceptance2",**summarize(cases))
    result["objective_review"]=assess_objectives(result,through,s.ledger.development_evaluations)
    result["episode_cases"]=cases
    s.evaluation=False
    s.evaluation_batch=None
    return result
