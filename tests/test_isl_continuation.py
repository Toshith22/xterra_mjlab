import importlib.util
from pathlib import Path

spec=importlib.util.spec_from_file_location("continuation",Path(__file__).resolve().parents[1]/"scripts/continue_isl.py")
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def case(level=0,pulse=0,falls=0):
    return dict(scenario=dict(level=level,pulse_family=pulse),trials=64,physical_falls=falls)


def test_severe_clean_regression_is_flagged():
    assert module.severe_regression(dict(previous_validated_level=0,cases=[case(falls=32)]))


def test_new_terrain_learning_is_not_a_catastrophic_regression():
    assert not module.severe_regression(dict(previous_validated_level=0,cases=[case(level=1,falls=64)]))


def test_disturbance_challenges_and_minor_falls_do_not_stop_training():
    assert not module.severe_regression(dict(previous_validated_level=0,
        cases=[case(pulse=1,falls=64),case(falls=1)]))
