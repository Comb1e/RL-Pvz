"""Independent controls for structured exploration and spatial placement."""

from types import SimpleNamespace

import pytest
import torch
from pvz_game import Dig, Place

from pvz_rl.config import load_config
from pvz_rl.envs.env import PvZEnv
from pvz_rl.learning.curriculum import CurriculumState
from pvz_rl.learning.deadline import RunBudget


@pytest.fixture(autouse=True)
def one_cpu_thread():
    torch.set_num_threads(1)


def test_stage_residency_counts_only_matching_episode_starts():
    cfg = load_config()
    state = CurriculumState(stage=1)
    for _ in range(100):
        state.completed_episode(0)
    assert not state.observe({"easy": 100}, 500, cfg)
    assert not state.observe({"easy": 100}, 1000, cfg)
    for _ in range(99):
        state.completed_episode(1)
    restored = CurriculumState(**state.to_dict())
    assert not restored.observe({"easy": 100}, 1500, cfg)
    restored.completed_episode(1)
    assert restored.observe({"easy": 100}, 2000, cfg)
    assert restored.name == "standard" and restored.completed_stage_games == 0


def test_wall_budget_reserves_cleanup_and_restores_consumption():
    cfg = load_config()
    now = [100.0]
    budget = RunBudget(cfg, elapsed=6000, clock=lambda: now[0])
    assert budget.deadline == 1300
    assert budget.learning_deadline == 400
    assert budget.can_collect(200)
    assert not budget.can_collect(251)
    now[0] = 250
    restored = RunBudget(cfg, elapsed=budget.state()["elapsed_seconds"], clock=lambda: 500)
    assert restored.state()["remaining_seconds"] == 1050
    assert restored.state()["reserve_seconds"] == 900


def test_early_dig_metric_tracks_actual_accepted_actions(per_tick_cfg):
    env = PvZEnv(per_tick_cfg)
    env.reset(seed=1)
    env.step(env.codec.encode(Place("sunflower", 0, 0)))
    env.step(env.codec.encode(Dig(0, 0)))
    assert env.episode_metrics()["early_voluntary_digs"] == 1
    env.step(env.codec.encode(Dig(0, 0)))
    assert env.episode_metrics()["early_voluntary_digs"] == 1


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("categories,width", [(9, 8), (8, 4)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_categorical_embedding_matches_lookup_gradients_and_adam(device, categories, width, dtype):
    from pvz_rl.policy.spatial_policy import CategoricalEmbedding

    torch.manual_seed(102)
    actual = CategoricalEmbedding(categories, width, padding_idx=0).to(device=device, dtype=dtype)
    reference = torch.nn.Embedding(categories, width, padding_idx=0).to(device=device, dtype=dtype)
    reference.load_state_dict(actual.state_dict())
    optimizers = [torch.optim.Adam(m.parameters(), lr=3e-4, eps=1e-5) for m in (actual, reference)]
    indices = torch.randint(categories, (7, 48, 45), device=device)
    indices[:, :, :30] = 0  # Repeated empty tiles dominate real placement histories.
    targets = torch.randn(*indices.shape, width, device=device, dtype=dtype)
    atol, rtol = (2e-7, 2e-6) if dtype == torch.float32 else (1e-12, 1e-10)
    for _ in range(3):
        outputs = []
        for module, optimizer in zip((actual, reference), optimizers):
            optimizer.zero_grad(set_to_none=True)
            output = module(indices)
            outputs.append(output)
            (output - targets).square().mean().backward()
        torch.testing.assert_close(*outputs, atol=atol, rtol=rtol)
        torch.testing.assert_close(actual.weight.grad, reference.weight.grad, atol=atol, rtol=rtol)
        assert torch.count_nonzero(actual.weight.grad[0]) == 0
        for optimizer in optimizers:
            optimizer.step()
        torch.testing.assert_close(actual.weight, reference.weight, atol=atol, rtol=rtol)
        with torch.no_grad():
            torch.testing.assert_close(actual(indices), reference(indices), atol=atol, rtol=rtol)
        assert torch.count_nonzero(actual.weight[0]) == 0
    if device == "cpu":
        with pytest.raises((RuntimeError, IndexError)):
            actual(torch.tensor([categories]))


def test_missing_cuda_compiler_backend_never_enters_tracing(monkeypatch):
    from pvz_rl.policy.spatial_policy import SpatialGroupedPolicy

    monkeypatch.setattr("pvz_rl.policy.spatial_policy.importlib.util.find_spec", lambda _: None)

    def unexpected(*args, **kwargs):
        raise AssertionError("Unavailable compiler must not trace beside the collector")

    monkeypatch.setattr(torch, "compile", unexpected)
    policy = SimpleNamespace(device=torch.device("cuda"))
    SpatialGroupedPolicy.enable_compilation(policy)
    assert policy.compilation_status == "unavailable" and "Triton" in policy.compilation_error
