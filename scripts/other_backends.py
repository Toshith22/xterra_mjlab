"""Load pinned MoE-CTS / WTW core algorithms without legacy simulator imports.

Upstream source and licenses are preserved separately. WTW PrefixProto classes
are used only as static defaults, so the CLI dependency is removed at load time.
"""
from pathlib import Path
import sys
import types


def load(source, algorithm):
    if algorithm == "moe":
        root = Path(source)/"rsl_rl/rsl_rl"
        old, namespace = "rsl_rl", "m2_moe_upstream"
        specs = [("utils.utils", ["split_and_pad_trajectories"]),
                 ("modules.utils", []),
                 ("modules.actor_critic_cts", ["ActorCriticCTS"]),
                 ("modules.actor_critic_moe_cts", ["ActorCriticMoECTS"]),
                 ("storage.rollout_storage_cts", ["RolloutStorageCTS"]),
                 ("algorithms.cts", ["CTS"]),
                 ("algorithms.moe_cts", ["MoECTS"])]
        packages = ("", ".utils", ".modules", ".storage", ".algorithms")
    else:
        root = Path(source)/"go1_gym_learn"
        old, namespace = "go1_gym_learn", "m2_wtw_upstream"
        specs = [("utils.utils", ["split_and_pad_trajectories"]),
                 ("ppo_cse.actor_critic", ["ActorCritic"]),
                 ("ppo_cse.rollout_storage", ["RolloutStorage"]),
                 ("ppo_cse.ppo", ["PPO"])]
        packages = ("", ".utils", ".ppo_cse")
    for package in packages:
        module = types.ModuleType(namespace+package)
        module.__path__ = []
        sys.modules[module.__name__] = module
    for name, exports in specs:
        path = root/(name.replace(".", "/")+".py")
        code = path.read_text().replace(old, namespace)
        code = code.replace("from params_proto import PrefixProto", "")
        code = code.replace("class AC_Args(PrefixProto, cli=False):", "class AC_Args:")
        code = code.replace("class PPO_Args(PrefixProto):", "class PPO_Args:")
        code = code.replace(f"from {namespace}.ppo_cse import caches", "")
        module = types.ModuleType(namespace+"."+name)
        module.__file__ = str(path)
        module.__package__ = module.__name__.rsplit(".", 1)[0]
        sys.modules[module.__name__] = module
        exec(compile(code, str(path), "exec"), module.__dict__)
        for exported in exports:
            setattr(sys.modules[module.__package__], exported, getattr(module, exported))
    if algorithm == "moe":
        return (sys.modules[namespace+".modules"].ActorCriticMoECTS,
                sys.modules[namespace+".algorithms"].MoECTS)
    return (sys.modules[namespace+".ppo_cse"].ActorCritic,
            sys.modules[namespace+".ppo_cse"].PPO)
