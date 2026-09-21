"""CPU tests independent of MuJoCo and legacy IsaacGym."""
import importlib.util
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("him_backend", ROOT/"scripts/him_backend.py")
backend = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backend)


def test_him_real_estimator_and_ppo_update():
    import os
    Model, Algorithm = backend.load(os.environ["M2_HIM_SOURCE"])
    torch.manual_seed(17)
    model = Model(270, 235, 45, 12, actor_hidden_dims=[32], critic_hidden_dims=[32])
    algorithm = Algorithm(model, num_learning_epochs=1, num_mini_batches=2)
    algorithm.init_storage(16, 4, [270], [235], [12])
    old_actor = model.actor[0].weight.clone()
    old_estimator = model.estimator.encoder[0].weight.clone()
    with torch.no_grad():
        for _ in range(4):
            algorithm.act(torch.randn(16, 270), torch.randn(16, 235))
            successor = torch.randn(16, 235)
            successor[:, 45:48] = torch.tensor([1., 2., 3.])
            algorithm.process_env_step(torch.ones(16), torch.zeros(16), {}, successor)
        algorithm.compute_returns(torch.randn(16, 235))
    assert torch.equal(algorithm.storage.next_privileged_observations[-1, :, 45:48],
                       torch.tensor([1., 2., 3.]).expand(16, -1))
    losses = algorithm.update()
    assert torch.isfinite(torch.tensor(losses)).all()
    assert not torch.equal(old_actor, model.actor[0].weight)
    assert not torch.equal(old_estimator, model.estimator.encoder[0].weight)
    assert model.actor[0].in_features == 64  # 45 proprio + 3 velocity + 16 latent


def test_nonfinite_gradient_rejected():
    import pytest
    parameter = torch.nn.Parameter(torch.ones(1))
    parameter.grad = torch.tensor([float("nan")])
    with pytest.raises(RuntimeError):
        backend.clip([parameter], 1.)
