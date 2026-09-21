from pathlib import Path
from types import SimpleNamespace
import copy
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from training_recovery import install_optimizer_guards, require_finite, restore, timeout_bootstrap


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.std = torch.nn.Parameter(torch.tensor([.8, 1.]))
        self.weight = torch.nn.Parameter(torch.tensor([.4]))


def test_non_timeout_nan_does_not_poison_rewards():
    r = timeout_bootstrap(torch.tensor([1., 2.]), torch.tensor([float("nan"), 3.]), torch.tensor([False, True]))
    torch.testing.assert_close(r, torch.tensor([1., 4.97]))
    with pytest.raises(FloatingPointError):
        timeout_bootstrap(torch.zeros(1), torch.tensor([float("nan")]), torch.tensor([True]))


def test_nonfinite_gradient_stops_before_adam_mutation():
    model = Model()
    opt = torch.optim.Adam(model.parameters(), lr=.01)
    install_optimizer_guards(model, [opt], SimpleNamespace(learning_rate=.01), 3e-4, .05, 1.)
    before = copy.deepcopy(model.state_dict())
    model.weight.grad = torch.tensor([float("nan")])
    with pytest.raises(RuntimeError):
        opt.step()
    assert not opt.state
    for k, v in before.items():
        torch.testing.assert_close(v, model.state_dict()[k])


def test_noise_is_bounded_after_each_minibatch():
    model = Model()
    opt = torch.optim.Adam(model.parameters(), lr=.01)
    alg = SimpleNamespace(learning_rate=.01)
    stats = install_optimizer_guards(model, [opt], alg, 3e-4, .05, 1.)
    (-model.std.sum()).backward()
    opt.step()
    assert model.std.max() <= 1
    assert stats["std_bound_events"] == 1
    assert opt.state[model.std]["exp_avg"][1] == 0
    assert opt.param_groups[0]["lr"] == alg.learning_rate == 3e-4


def test_reject_diverged_noise_checkpoint():
    model = Model()
    model.std.data[0] = 44.
    with pytest.raises(ValueError, match="healthy exploration"):
        install_optimizer_guards(model, [torch.optim.Adam(model.parameters())], SimpleNamespace(learning_rate=.001), 3e-4, .05, 1.)


def test_resume_preserves_model_optimizer_and_counter():
    model = Model()
    opt = torch.optim.Adam(model.parameters(), lr=.0002)
    (model.std.sum() + model.weight.sum()).backward(); opt.step()
    args = SimpleNamespace(algorithm="wtw", num_envs=64, steps=8, seed=12)
    saved = copy.deepcopy(dict(model_state_dict=model.state_dict(), optimizer_states=[opt.state_dict()],
                              arguments=vars(args), learning_rate=.0002, update=15000))
    new_model = Model(); new_opt = torch.optim.Adam(new_model.parameters())
    alg = SimpleNamespace(learning_rate=.001)
    assert restore(saved, new_model, [new_opt], alg, args) == 15000
    for k, v in model.state_dict().items():
        torch.testing.assert_close(v, new_model.state_dict()[k])
    torch.testing.assert_close(opt.state[model.std]["exp_avg"], new_opt.state[new_model.std]["exp_avg"])
    assert alg.learning_rate == .0002
    args.algorithm = "moe"
    with pytest.raises(ValueError, match="mismatch"):
        restore(saved, new_model, [new_opt], alg, args)


def test_corrupt_optimizer_is_rejected():
    with pytest.raises(FloatingPointError, match="exp_avg"):
        require_finite({"optimizers": [{"exp_avg": torch.tensor([float("inf")])}]})
