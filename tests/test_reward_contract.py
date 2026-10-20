"""Asset conservation, reward diagnostics and supported training-mode boundaries."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from pvz_game import Dig, LevelSpec, Place, Rules, Spawn, Status
from pvz_game.config import PLANT_TYPES, InitialPlant
from pvz_game.types import Event

from pvz_rl.config import load_config
from pvz_rl.envs.env import PvZEnv
from pvz_rl.envs.rewards import HomeProximityLedger, asset_value, reward_parts
from pvz_rl.learning.objective import (
    components,
    endgame_terminal_rewards,
    training_rewards,
    victory_time,
)


@pytest.mark.parametrize(
    "total,spawned,ordered,expected",
    [
        (15, 4, [("sunflower", 25)], 25),
        (15, 5, [("sunflower", 25)], 0),
        (14, 4, [("sunflower", 25)], 25),
        (0, 0, [("sunflower", 25)], 0),
        (15, 4, [("spawn", 0), ("sunflower", 25)], 0),
        (15, 4, [("sunflower", 25), ("spawn", 0), ("sunflower", 25)], 25),
        (15, 0, [("sky", 25), ("sunflower", 10), ("sunflower", 0)], 10),
        (15, 0, [("sky", 25), ("sunflower", 0)], 0),
    ],
)
def test_early_sun_independent_event_order_cap_and_training_scale(
    total, spawned, ordered, expected
):
    cfg = load_config()
    env = PvZEnv(cfg)
    try:
        env.reset(seed=3)
        before = replace(
            env.public, counts=replace(env.public.counts, initial_total=total, spawned=spawned)
        )
        events = tuple(
            Event("ZombieSpawned", 1, 99)
            if source == "spawn"
            else Event(
                "SunProduced", 1, 1, (("source", source), ("amount", amount), ("produced", 25))
            )
            for source, amount in ordered
        )
        income = sum(amount for source, amount in ordered if source != "spawn")
        sky = sum(amount for source, amount in ordered if source == "sky")
        after = replace(before, sun=before.sun + income)
        parts = reward_parts(before, after, cfg, events=events)
        assert parts["early_sun"] == expected
        assert parts["net_value"] == income - sky
        assert parts["early_sun_bonus"] == pytest.approx(2 * expected / 30000)
        assert parts["development"] == pytest.approx((income - sky + 2 * expected) / 30000)
        # The current objective keeps the resource delta and the
        # source-action sun credit as separate physical channels.
        assert training_rewards(components(parts), cfg) == pytest.approx(
            (2 * (income - sky) + 2 * expected) / 3000
        )
    finally:
        env.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize(
    "sun,total,spawn_tick,expected",
    [(9990, 3, 99, 0), (9980, 3, 99, 10), (0, 3, 1, 0), (0, 4, 1, 25), (0, 0, 99, 0)],
)
def test_early_sun_cuda_controls_at_production(sun, total, spawn_tick, expected):
    from pvz_game import Game

    from pvz_rl.envs.cuda_accounting import AccountingCudaBatch
    from pvz_rl.envs.cuda_features import REWARD_FIELDS, CudaFeatures

    cfg = load_config()
    game = Game()
    game.reset(
        LevelSpec(
            "early-control",
            tuple(
                Spawn(spawn_tick if index == 0 else 9999, "basic", index % 5)
                for index in range(total)
            ),
            initial_sun=sun,
            plants=(InitialPlant("sunflower", 4, 0),),
        ),
        7,
    )
    snapshot = game.snapshot()
    snapshot["plants"][0]["due"] = 1
    game.restore(snapshot)
    before = game.observe()
    result = game.step(ticks=1)
    cpu = reward_parts(before, result.observation, cfg, events=result.events)
    batch = AccountingCudaBatch(1, zombie_capacity=max(1, total), max_step_ticks=1)
    batch.restore([snapshot])
    with batch.cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
        features = CudaFeatures(batch, cfg, "masked")
        features.encode()
        features.step(batch.cp.asarray([0], dtype=batch.cp.int64))
        parts = dict(zip(REWARD_FIELDS, features.parts.get()[0], strict=True))
        assert parts["early_sun"] == cpu["early_sun"] == expected
        for name in REWARD_FIELDS:
            assert parts[name] == pytest.approx(cpu[name], abs=1e-10), name
        assert not hasattr(features, "history") and not hasattr(features, "history_tensor")


@pytest.mark.parametrize("kind", PLANT_TYPES)
def test_purchase_conserves_assets_and_dig_removes_remaining_value(kind):
    cfg = load_config()
    env = PvZEnv(cfg)
    env.reset(
        seed=4,
        options={"scenario": LevelSpec("value", (Spawn(9999, "basic", 0),), initial_sun=900)},
    )
    for _ in range(3501):
        env.step(0)
    before_assets = asset_value(env.public)
    before_tick = env.public.tick
    _, buy, _, _, _ = env.step(env.codec.encode(Place(kind, 2, 3)))
    assert env.public.tick == before_tick and asset_value(env.public) == before_assets and buy == 0
    _, dig, _, _, _ = env.step(env.codec.encode(Dig(2, 3)))
    cost = env.rules.plants[kind]["cost"]
    assert asset_value(env.public) == before_assets - cost
    assert dig == pytest.approx(-cost / 30000)


def test_independent_damage_death_terminal_and_timeout_formula():
    cfg = load_config()
    env = PvZEnv(cfg)
    env.reset(
        seed=3,
        options={
            "scenario": LevelSpec(
                "values",
                (Spawn(9999, "basic", 0),),
                initial_sun=600,
                plants=(InitialPlant("peashooter", 0, 0),),
            )
        },
    )
    before = env.public
    assert asset_value(before) == 700
    damaged = replace(
        before, plants=(replace(before.plants[0], health=before.plants[0].max_health // 2),)
    )
    assert asset_value(damaged) == 650
    assert reward_parts(before, damaged, cfg)["total"] == pytest.approx(-50 / 30000)
    after = replace(damaged, plants=())
    assert reward_parts(damaged, after, cfg)["total"] == pytest.approx(-50 / 30000)
    assert asset_value(after) == 600
    for status, terminal in ((Status.WON, 1), (Status.LOST, -1)):
        end = replace(after, status=status)
        assert asset_value(end) == 600
        assert reward_parts(damaged, end, cfg)["total"] == pytest.approx(terminal - 50 / 30000)


@pytest.mark.parametrize("sun", [0, 50, 300, 600, 9990])
@pytest.mark.parametrize("kills", [0, 1, 4])
def test_mower_cost_once_independent_of_sun_and_kills(sun, kills):
    cfg = load_config()
    env = PvZEnv(cfg)
    env.reset(seed=1)
    before = replace(env.public, sun=sun)
    events = [Event("MowerActivated", 1, details=(("row", 0),))]
    for i in range(kills):
        events.extend(
            (Event("DamageApplied", 1, i, (("source", -1),)), Event("ZombieDefeated", 1, i))
        )
    parts = reward_parts(before, before, cfg, events=events)
    assert parts["mower_activation_penalty"] == pytest.approx(-500 / 30000)
    assert parts["mower_kills"] == kills
    assert parts["total"] == pytest.approx(-500 / 30000)
    assert reward_parts(before, before, cfg)["mower_activation_penalty"] == 0


def test_empty_explosion_and_nut_bite_have_signed_blocking_credit():
    cfg = load_config()
    env = PvZEnv(cfg)
    env.reset(
        seed=3,
        options={
            "scenario": LevelSpec(
                "nut", (Spawn(1000, "basic", 0),), plants=(InitialPlant("wall_nut", 0, 0),)
            )
        },
    )
    before = env.public
    events = (
        Event("PlantDamaged", 1, before.plants[0].id, (("damage", 25),)),
        Event("PlantExploded", 1, 99),
    )
    part = reward_parts(before, before, cfg, events=events)
    assert part["wall_nut_damage"] == 25 and part["empty_explosions"] == 1
    assert part["action_blocking"] == pytest.approx(25 * 0.025)
    assert part["total"] == pytest.approx(25 * 0.025 / 30000)


def test_house_entries_are_staged_once_and_victory_time_is_win_only():
    cfg = load_config()
    rules = Rules()
    before = SimpleNamespace(zombies=(SimpleNamespace(id=7, x=2500, health=100, headless=False),))
    ledger = HomeProximityLedger(before, rules)
    outer = SimpleNamespace(zombies=(SimpleNamespace(id=7, x=1500, health=100, headless=False),))
    inner = SimpleNamespace(zombies=(SimpleNamespace(id=7, x=500, health=100, headless=False),))
    assert ledger.advance(outer, cfg["reward"]["home_entry_penalties"]) == {
        "home_outer_entries": 1,
        "home_inner_entries": 0,
        "home_proximity": -0.005,
    }
    assert ledger.advance(inner, cfg["reward"]["home_entry_penalties"])["home_proximity"] == -0.015
    assert ledger.advance(inner, cfg["reward"]["home_entry_penalties"])["home_proximity"] == 0
    assert victory_time(
        [80, 160, 240], [160, 160, 160], [True, True, False], 0.1
    ).tolist() == pytest.approx([1 / 30, 0, 0])

    # Exact boundaries, retreat/re-entry, a spawn skipping both stages, and
    # a threat eliminated in the same transition are independent controls.
    def board(*entries):
        return SimpleNamespace(
            zombies=tuple(
                SimpleNamespace(id=i, x=x, health=health, headless=headless)
                for i, x, health, headless in entries
            )
        )

    ledger = HomeProximityLedger(board(), rules)
    assert ledger.advance(board((1, 2000, 100, False)), [0.005, 0.015])["home_proximity"] == 0
    assert ledger.advance(board((1, 1999, 100, False)), [0.005, 0.015])["home_proximity"] == -0.005
    assert ledger.advance(board((1, 1000, 100, False)), [0.005, 0.015])["home_proximity"] == 0
    assert ledger.advance(board((1, 999, 100, False)), [0.005, 0.015])["home_proximity"] == -0.015
    ledger.advance(board((1, 2500, 100, False)), [0.005, 0.015])
    assert ledger.advance(board((1, 500, 100, False)), [0.005, 0.015])["home_proximity"] == 0
    assert (
        ledger.advance(
            board((2, 500, 100, False), (3, 500, 0, False), (4, 500, 80, True)), [0.005, 0.015]
        )["home_proximity"]
        == -0.02
    )
    assert ledger.advance(board(), [0.005, 0.015])["home_proximity"] == 0
    initial = HomeProximityLedger(board((1, 500, 100, False)), rules)
    assert initial.advance(board((1, 400, 100, False)), [0.005, 0.015])["home_proximity"] == 0
    assert victory_time([0, 80, 160, 240, 320], 160, True, 0.1).tolist() == pytest.approx(
        [0.1, 1 / 30, 0, -0.02, -1 / 30]
    )
    assert victory_time(0, 0, True, 0.1) == 0
    assert victory_time([0, 10, 300], 160, False, 0.1).tolist() == [0, 0, 0]


def test_endgame_time_ranking_changes_only_terminal_outcomes():
    terminal, adjustment = endgame_terminal_rewards([80, 160, 240], [True, False, True], weight=0.5)
    assert terminal.tolist() == pytest.approx([1.5, -1.0, 0.5])
    assert adjustment.tolist() == pytest.approx([0.5, 0.0, -0.5])

    terminal, _ = endgame_terminal_rewards([1, 2, 3], [False, False, False], weight=0.5)
    assert terminal.tolist() == pytest.approx([-1.0, -1.0, -1.5])

    # Equal completion times share a rank and therefore receive no arbitrary
    # ordering bonus.
    terminal, adjustment = endgame_terminal_rewards([10, 10, 10], [True, False, True])
    assert terminal.tolist() == pytest.approx([1.0, -1.0, 1.0])
    assert adjustment.tolist() == pytest.approx([0.0, 0.0, 0.0])


def test_six_component_targets_scale_all_physical_channels_once():
    from pvz_rl.learning.objective import components, training_rewards

    cfg = load_config()
    values = {
        "global_terminal": 1.0,
        "global_asset_value": -2.0,
        "action_damage": 50 / 270,
        "action_blocking": 50.0,
        "action_sun": -200.0,
        "action_mower": -3.0,
    }
    expected = 1 + 10 * (-2 + 50 / 270 + 50 - 200 - 3) / 30000
    assert training_rewards(components(values), cfg) == pytest.approx(expected)
    # Diagnostic-only aliases never enter the learning vector.
    assert training_rewards(components({"terminal": 1, "development": 100}), cfg) == 0
    assert training_rewards(components({"global_terminal": 1}), cfg) == 1


def test_shipped_action_reward_equivalents():
    cfg = load_config()
    assert cfg["reward"]["basic_zombie_value"] == 30
    assert cfg["reward"]["mower_value"] == 500
    assert cfg["reward"]["action"]["block_value_by_plant"]["wall_nut"] == pytest.approx(100 / 4000)
    for kind, health in (
        ("sunflower", 300),
        ("peashooter", 300),
        ("cherry_bomb", 300),
        ("potato_mine", 300),
        ("snow_pea", 300),
        ("chomper", 300),
        ("repeater", 300),
    ):
        assert cfg["reward"]["action"]["block_value_by_plant"][kind] == pytest.approx(10 / health)


@pytest.mark.parametrize("setting", ["shaped", "curriculum", "masked", "fixed"])
def test_retired_training_modes_fail_before_creating_output(tmp_path, setting):
    from pvz_rl.learning.training import train

    cfg = load_config()
    if setting == "fixed":
        cfg["curriculum"]["mode"] = "fixed"
    else:
        cfg["conditions"]["masked"][setting] = False
    output = tmp_path / "absent"
    with pytest.raises(ValueError, match="Retired policy|single masked teaching method"):
        train(cfg, "masked", 101, output)
    assert not output.exists()


@pytest.mark.parametrize("retired", ["network", "observation", "reward", "clock"])
def test_retired_report_is_rejected_without_loading_model(tmp_path, monkeypatch, retired):
    from pvz_rl.presentation.visualization import visualize_run
    from pvz_rl.provenance import write_json

    cfg = load_config()
    if retired == "network":
        cfg["policy"]["kind"] = "spatial_grouped_v2"
        cfg["encoding"]["version"] = "tactical_v2"
    elif retired == "observation":
        cfg["encoding"]["version"] = "event_v5"
    elif retired == "reward":
        cfg["reward"]["version"] = "potential_mower_v1"
    else:
        cfg["training"].pop("discount_clock")
    # A file exists, but archived reporting must never deserialize its weights.
    (tmp_path / "best.zip").write_bytes(b"not a model")
    write_json(tmp_path / "metadata.json", {"config": cfg, "family": "preset"})
    write_json(tmp_path / "best.json", {"checkpoint_hash": "deleted-model"})

    def forbidden(*args, **kwargs):
        raise AssertionError("Archived report must not deserialize weights")

    monkeypatch.setattr("pvz_rl.learning.training.load_policy", forbidden)
    result = visualize_run(tmp_path, videos=False)
    assert result["state"] == "failed"
    assert "current engine and model protocol" in result["error"]


@pytest.mark.parametrize(
    "case", ["sun", "cooldown", "occupied", "empty_dig", "full", "terminal_tick"]
)
def test_invalid_wait_equivalence_cpu_and_cuda(case):
    from pvz_rl.envs.cuda_accounting import AccountingCudaBatch
    from pvz_rl.envs.cuda_features import REWARD_FIELDS, CudaFeatures

    cfg = load_config()
    if case == "terminal_tick":
        cfg["environment"]["cutoff_seconds"] = 1
    plants = ()
    if case == "occupied":
        plants = (InitialPlant("sunflower", 0, 0),)
    elif case == "full":
        plants = tuple(InitialPlant("sunflower", row, col) for row in range(5) for col in range(9))
    level = LevelSpec(
        "invalid-wait",
        (Spawn(9999, "basic", 0),),
        initial_sun=0 if case in ("sun", "terminal_tick") else 1000,
        plants=plants,
    )
    proposed, waiting = PvZEnv(cfg), PvZEnv(cfg)
    try:
        for env in (proposed, waiting):
            env.reset(seed=101, options={"scenario": level})
            if case == "terminal_tick":
                for _ in range(99):
                    env.step(0)
        snapshot = proposed.game.snapshot()
        command = (
            Dig(4, 8)
            if case == "empty_dig"
            else Place("wall_nut" if case == "cooldown" else "sunflower", 0, 0)
        )
        action = proposed.codec.encode(command)
        result = proposed.step(action)
        control = waiting.step(0)
        assert proposed.public == waiting.public
        assert result[1:4] == control[1:4]
        info, reference = result[-1], control[-1]
        assert not info["accepted"] and info["executed_action"] == 0
        assert info["proposal_action"] == action and info["ticks_advanced"] == 1
        assert info["reward_parts"] == reference["reward_parts"]
        assert (
            info["reward_parts"]["invalid_plant_penalty"]
            == info["reward_parts"]["empty_dig_penalty"]
            == 0
        )
        if torch.cuda.is_available():
            batch = AccountingCudaBatch(2, zombie_capacity=1, max_step_ticks=1)
            batch.restore([snapshot, snapshot])
            with batch.cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
                features = CudaFeatures(batch, cfg, "masked")
                features.encode()
                features.step(batch.cp.asarray([action, 0], dtype=batch.cp.int64))
                parts = features.parts.get()
                assert parts[0].tolist() == parts[1].tolist()
                for key, value in zip(REWARD_FIELDS, parts[0], strict=True):
                    assert value == pytest.approx(info["reward_parts"][key], abs=1e-10), key
                assert batch.state_hash(0) == batch.state_hash(1) == proposed.game.state_hash()
    finally:
        proposed.close()
        waiting.close()
