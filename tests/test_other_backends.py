import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import torch

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/"scripts"/(name+".py"))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_moe_teacher_student_mapping_and_both_optimizers():
    Model, Algorithm = module("other_backends").load(os.environ["M2_MOE_SOURCE"], "moe")
    torch.manual_seed(22)
    model = Model(45, 235, 12, 16, 5, actor_hidden_dims=[32], critic_hidden_dims=[32],
        teacher_encoder_hidden_dims=[32], student_encoder_hidden_dims=[32, 32, 32])
    alg = Algorithm(model, 16, 5, num_learning_epochs=1, num_mini_batches=2)
    alg.init_storage(16, 4, [45], [235], [12])
    old_actor = next(model.actor.parameters()).clone()
    old_student = next(model.student_moe_encoder.parameters()).clone()
    with torch.no_grad():
        for _ in range(4):
            actions = alg.act(torch.randn(16, 45), torch.randn(16, 235), torch.randn(16, 225))
            assert torch.equal(actions[alg.teacher_env_idxs], alg.transition.actions[:12])
            assert torch.equal(actions[alg.student_env_idxs], alg.transition.actions[12:])
            alg.process_env_step(torch.arange(16).float(), torch.zeros(16), {})
        assert torch.equal(alg.storage.rewards[0, :12, 0], alg.teacher_env_idxs.float())
        assert torch.equal(alg.storage.rewards[0, 12:, 0], alg.student_env_idxs.float())
        alg.compute_returns(torch.randn(16, 235), torch.randn(16, 225))
    assert torch.isfinite(torch.tensor(alg.update())).all()
    assert not torch.equal(old_actor, next(model.actor.parameters()))
    assert not torch.equal(old_student, next(model.student_moe_encoder.parameters()))


def test_wtw_adaptation_and_policy_update():
    Model, Algorithm = module("other_backends").load(os.environ["M2_WTW_SOURCE"], "wtw")
    model = Model(55, 2, 1650, 12)
    alg = Algorithm(model)
    alg.init_storage(16, 4, [55], [2], [1650], [12])
    old = next(model.adaptation_module.parameters()).clone()
    with torch.no_grad():
        for _ in range(4):
            alg.act(torch.randn(16, 55), torch.randn(16, 2), torch.randn(16, 1650))
            alg.process_env_step(torch.ones(16), torch.zeros(16), {"env_bins": torch.zeros(16)})
        alg.compute_returns(torch.randn(16, 1650), torch.randn(16, 2))
    assert torch.isfinite(torch.tensor(alg.update())).all()
    assert not torch.equal(old, next(model.adaptation_module.parameters()))


def test_wtw_trot_clock_has_diagonal_pairs():
    wtw = module("wtw_m2")
    env = SimpleNamespace(episode_length_buf=torch.tensor([0, 7]), step_dt=.02,
        _wtw_gait=SimpleNamespace(commands=torch.tensor([[2.5, .5, 0, 0, .5, .1]]).repeat(2, 1),
                                 phase_offset=torch.zeros(2)))
    _, _, contact = wtw.phase_contacts(env)
    assert torch.allclose(contact[:, 0], contact[:, 3])
    assert torch.allclose(contact[:, 1], contact[:, 2])
    assert torch.allclose(contact[:, 0]+contact[:, 1], torch.ones(2), atol=1e-4)
