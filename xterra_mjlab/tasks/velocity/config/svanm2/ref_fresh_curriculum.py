"""Explicit difficulty and evidence gates for the fresh locomotion experiment.

Training exposure, training promotion and held-out validation are different facts.
No checkpoint is called reliable merely because its command ceiling reached 1.5.
"""
from dataclasses import asdict, dataclass
from collections import defaultdict, deque
import math


@dataclass(frozen=True)
class Difficulty:
    level: int
    flat_speed: float
    slope_degrees: float
    slope_speed: float
    riser: float
    stair_speed: float
    push_newtons: float
    torque_nm: float


LEVELS = (
    Difficulty(0, .40, 0., .40, 0., .30, 0., 0.),
    Difficulty(1, .65, 0., .40, 0., .30, 15., 2.),
    Difficulty(2, .85, 8., .50, 0., .30, 20., 3.),
    Difficulty(3, 1.00, 12., .65, .06, .40, 25., 4.),
    Difficulty(4, 1.20, 16., .80, .10, .50, 30., 5.),
    Difficulty(5, 1.50, 20., 1.00, .14, .60, 35., 6.),
    Difficulty(6, 1.50, 24., 1.20, .16, .80, 40., 8.),
    Difficulty(7, 1.50, 28., 1.50, .18, 1.00, 45., 10.),
    Difficulty(8, 1.50, 28., 1.50, .18, 1.25, 45., 10.),
    Difficulty(9, 1.50, 28., 1.50, .18, 1.50, 45., 10.),
)
FAMILIES = ("flat", "slope_up", "slope_down", "stairs_up", "stairs_down")
TASKS = ("move", "stop", "stand")


def families(level):
    return FAMILIES[:1 if level < 2 else 3 if level < 3 else 5]


def speed_limit(level, family):
    d = LEVELS[level]
    return d.flat_speed if family == "flat" else d.slope_speed if family.startswith("slope") else d.stair_speed


class ProgressLedger:
    """Bounded recent episode windows, with gates for each task and terrain.

    Failures and missing exposure cannot disappear into a pooled mean. Older
    levels remain sampled and must also pass to permit another promotion.
    """
    def __init__(self, minimum=64, window=128, pass_rate=.90):
        if minimum < 1 or window < minimum or not 0 < pass_rate <= 1:
            raise ValueError("invalid evidence gate")
        self.minimum, self.window, self.pass_rate = minimum, window, pass_rate
        self.level = 0
        self.records = defaultdict(lambda: deque(maxlen=window))
        self.exposure = defaultdict(int)
        self.transitions = []
        self.development_evaluations = []

    def add(self, level, family, task, success, *, ceiling=False, disturbed=False,
            pair_event=False, physical_fall=False, speed_error=None):
        if level not in range(len(LEVELS)) or family not in families(level) or task not in TASKS:
            raise ValueError("unknown curriculum stratum")
        record = dict(success=bool(success), ceiling=bool(ceiling), disturbed=bool(disturbed),
                      pair_event=bool(pair_event), physical_fall=bool(physical_fall), speed_error=speed_error)
        if speed_error is not None and not math.isfinite(speed_error):
            record["success"] = False
        self.records[(level, family, task)].append(record)
        self.exposure[(level, family, task)] += 1

    def blockers(self):
        reasons = []
        for level in range(self.level+1):
            for family in families(level):
                for task in TASKS:
                    rows = list(self.records[(level, family, task)])
                    name = f"L{level}/{family}/{task}"
                    if len(rows) < self.minimum:
                        reasons.append(f"{name}: {len(rows)}/{self.minimum} episodes")
                    elif sum(r["success"] for r in rows)/len(rows) < self.pass_rate:
                        reasons.append(f"{name}: success below {self.pass_rate:.0%}")
                    if task == "move":
                        top = [r for r in rows if r["ceiling"]]
                        if len(top) < self.minimum//4 or not top or sum(r["success"] for r in top)/len(top) < self.pass_rate:
                            reasons.append(f"{name}: command-ceiling tracking unpassed")
                    if level > 0:
                        pushed = [r for r in rows if r["disturbed"]]
                        if len(pushed) < self.minimum//4 or not pushed or sum(r["success"] for r in pushed)/len(pushed) < self.pass_rate:
                            reasons.append(f"{name}: disturbance recovery unpassed")
        return reasons

    def try_promote(self, update):
        if self.level == len(LEVELS)-1 or self.blockers():
            return False
        old = self.level
        self.level += 1
        self.transitions.append(dict(update=int(update), from_level=old, to_level=self.level,
                                     basis="recent training episodes; not held-out certification"))
        return True

    def accept_evaluation(self, result, update, maximum=None):
        """Promotion uses mean-action development tests, never exploration noise.

        Final confirmation uses different seeds and includes the speed sweep
        and the old benchmark. A failed/missing/partial test cannot promote.
        """
        validated = result.get("highest_consecutively_validated_level")
        evidence = dict(update=int(update), current_level=self.level, validated_level=validated,
                        model_sha256=result.get("model_sha256"), seed=result.get("seed"),
                        objective_review=result.get("objective_review", {}))
        self.development_evaluations.append(evidence)
        if result.get("smoke", False) or result.get("diagnostic_only", False) or result.get("review_alerts") or result.get("objective_review", {}).get("pause_recommended", False) or not result.get("model_sha256") or validated is None or validated < self.level:
            return False
        if self.level >= (len(LEVELS)-1 if maximum is None else maximum):
            return False
        old = self.level
        self.level += 1
        self.transitions.append(dict(update=int(update), from_level=old, to_level=self.level,
            basis="mean-action development evaluation; all earlier levels retained; fresh confirmation still required"))
        return True

    def report(self, update):
        rows = []
        for key in sorted(self.exposure):
            values = list(self.records[key])
            errors = [r["speed_error"] for r in values if r["speed_error"] is not None and math.isfinite(r["speed_error"])]
            rows.append(dict(level=key[0], family=key[1], task=key[2], total_episodes=self.exposure[key],
                recent_episodes=len(values), recent_success_rate=sum(r["success"] for r in values)/len(values),
                mean_speed_error=sum(errors)/len(errors) if errors else None,
                verified_pair_events=sum(r["pair_event"] for r in values),
                physical_falls=sum(r["physical_fall"] for r in values)))
        last_validated = self.development_evaluations[-1]["validated_level"] if self.development_evaluations else None
        secondary_review = self.development_evaluations[-1].get("objective_review", {}) if self.development_evaluations else {}
        return dict(completed_updates=int(update), unlocked_level=self.level,
                    unlocked_difficulty=asdict(LEVELS[self.level]),
                    highest_exposed_level=max((k[0] for k in self.exposure), default=None),
                    levels=[asdict(d) for d in LEVELS], transitions=self.transitions,
                    development_evaluations=self.development_evaluations,
                    secondary_review=secondary_review,
                    promotion_blockers=(["Secondary issue needs visual/telemetry review before progression"]
                        if secondary_review.get("pause_recommended", False) else []
                        if last_validated is not None and last_validated >= self.level else
                        ["Awaiting passing mean-action development evaluation at all unlocked levels"]),
                    exploration_diagnostics=self.blockers(), strata=rows,
                    held_out_validated_level=None, reliable_1_5_mps=False,
                    note="Training evidence only. Held-out speed, braking, terrain and support-loss tests are required.")
