"""GPU evaluation must retain every case and trace when slots are reused."""

import copy
from types import SimpleNamespace

import pytest
import torch
from pvz_game import LevelSpec, Spawn

from pvz_rl.config import load_config, research_config, validate_config
from pvz_rl.cuda_evaluation import batched_games
from pvz_rl.env import PvZEnv
from pvz_rl.rewards import REWARD_METRICS


class WaitingPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("anchor", torch.zeros(1, device="cuda"))

    @property
    def device(self):
        return self.anchor.device

    def forward(self, obs, **kwargs):
        return torch.zeros(len(obs), dtype=torch.long, device=self.device), None, None


@pytest.mark.parametrize("record", [False, True])
def test_refill_matches_fixed_batches_and_cpu(record, monkeypatch):
    pytest.importorskip("cupy")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    cfg = load_config()
    cfg["simulation"] = {"backend": "cuda"}
    cfg["training"].update(
        n_envs=2, device="cuda", rollout_size=256, rollout_steps_per_env=128, batch_size=128
    )
    cfg["environment"]["cutoff_seconds"] = 3

    def scenario(level, family, seed, *args):
        # Slot 1 finishes first; later cases exercise reuse, loss, and cutoff.
        ticks = {1: 18, 2: 1, 3: 3, 4: 100, 5: 1}
        return LevelSpec(level, (Spawn(ticks[seed], "basic", 0, x=0),), mowers=seed != 5)

    monkeypatch.setattr("pvz_rl.cuda_env.scenario", scenario)
    policy = SimpleNamespace(policy=WaitingPolicy())
    results = []
    for refill in (False, True):
        local = copy.deepcopy(cfg)
        local["runtime"]["refill_evaluation"] = refill
        validate_config(local)
        assert research_config(local) == research_config(cfg)
        results.append(
            list(
                batched_games(
                    local, policy, "masked", [1, 2, 3, 4, 5], ["easy"], "preset", record=record
                )
            )
        )
    for fixed, refilled in zip(*results, strict=True):
        for key in ("inference_seconds", "wall_seconds"):
            fixed.pop(key)
            refilled.pop(key)
        assert fixed == refilled
        assert bool("action_trace" in refilled) == record
        env = PvZEnv(cfg)
        env.reset(
            seed=refilled["scenario_seed"],
            options={"scenario": scenario("easy", "preset", refilled["scenario_seed"])},
        )
        for _ in range(refilled["decisions"]):
            env.step(0)
        assert env.game.state_hash() == refilled["state_hash"]
        assert env.state == refilled["status"]
        for key in REWARD_METRICS:
            assert refilled[key] == pytest.approx(env.episode_metrics()[key])
        assert refilled["mower_kills"] == (1 if refilled["status"] == "won" else 0)
    assert [r["scenario_seed"] for r in results[1]] == [1, 2, 3, 4, 5]
    assert [r["status"] for r in results[1]] == ["won", "won", "won", "truncated", "lost"]


def test_plant_and_mower_kills_survive_cuda_episode_reset(monkeypatch):
    pytest.importorskip("cupy")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from pvz_game.config import InitialPlant

    cfg = load_config()
    cfg["simulation"] = {"backend": "cuda"}
    cfg["training"].update(
        n_envs=1, device="cuda", rollout_size=128, rollout_steps_per_env=128, batch_size=128
    )

    def scenario(*args):
        return LevelSpec(
            "kills",
            (Spawn(1, "basic", 0, x=1400), Spawn(25, "basic", 2, x=0)),
            initial_sun=500,
            plants=(InitialPlant("cherry_bomb", 0, 1),),
        )

    monkeypatch.setattr("pvz_rl.cuda_env.scenario", scenario)
    rows = list(
        batched_games(
            cfg, SimpleNamespace(policy=WaitingPolicy()), "masked", [1, 2], ["easy"], "preset"
        )
    )
    for row in rows:
        assert row["status"] == "won"
        assert row["plant_kills"] == row["mower_kills"] == 1
        assert row["defeated"] == 2
        assert row["mower_activation_penalty"] == pytest.approx(-0.2)
