"""Asset conservation, reward diagnostics and supported training-mode boundaries."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
from pvz_game import Dig, LevelSpec, Place, Rules, Spawn, Status
from pvz_game.config import PLANT_TYPES, InitialPlant
from pvz_game.types import Event

from pvz_rl.config import load_config
from pvz_rl.envs.env import PvZEnv
from pvz_rl.envs.rewards import HomeProximityLedger, asset_value, reward_parts
from pvz_rl.learning.objective import victory_time


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
    for status, terminal in ((Status.WON, 1), (Status.LOST, -2)):
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
    assert parts["mower_activation_penalty"] == pytest.approx(-200 / 30000)
    assert parts["mower_kills"] == kills
    assert parts["total"] == pytest.approx(-200 / 30000)
    assert reward_parts(before, before, cfg)["mower_activation_penalty"] == 0


def test_empty_explosion_and_nut_bite_are_only_diagnostics():
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
    assert part["total"] == 0


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


def test_six_component_targets_only_scale_development():
    from pvz_rl.learning.objective import components, training_rewards

    cfg = load_config()
    # One effective HP, full basic HP, mower, invalid planting and empty digging.
    raw = [1.0, -2.0, 50 / 270 / 30000, 50 / 30000, -200 / 30000, -0.00001, -0.000003, -0.02, 0.03]
    keys = (
        ["terminal"] * 2
        + ["development"] * 3
        + ["invalid_plant_penalty", "empty_dig_penalty", "home_proximity", "victory_time"]
    )
    got = [float(training_rewards(components({k: v}), cfg)) for k, v in zip(keys, raw)]
    assert got == pytest.approx(
        [1, -2, 50 / 270 / 3000, 50 / 3000, -200 / 3000, -0.00001, -0.000003, -0.02, 0.03]
    )
    assert training_rewards(components({"terminal": 1, "development": 0.01}), cfg) == pytest.approx(
        1.1
    )
    assert training_rewards(components({"terminal": 1}), cfg) == 1


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
