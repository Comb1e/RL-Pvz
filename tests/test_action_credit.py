from types import SimpleNamespace

import pytest
from pvz_game import Rules
from pvz_game.types import Event

from pvz_rl.envs.action_credit import CausalCreditLedger


def plant(ident, kind):
    return SimpleNamespace(id=ident, plant_type=kind)


def ledger(cfg, plants=()):
    result = CausalCreditLedger(cfg, Rules())
    result.seed_observation(SimpleNamespace(plants=tuple(plants)))
    return result


def test_projectile_damage_is_credited_to_placement_origin(cfg):
    credit = ledger(cfg)
    credit.attribute([Event("PlantPlaced", 1, 10, (("plant_type", "peashooter"),))], action_id=7)
    values = credit.attribute(
        [
            Event("ProjectileFired", 2, 20, (("source", 10),)),
            Event(
                "DamageApplied",
                2,
                3,
                (("source", 20), ("health_damage", 20), ("armor_damage", 0)),
            ),
            Event("ZombieDefeated", 2, 3),
        ],
        action_id=8,
    )
    assert values[7]["damage"] == pytest.approx(20 * 30 / 270)
    assert values.get(8, {}).get("damage", 0.0) == 0


def test_sun_and_swallow_use_the_plant_origin_without_duplicate_kill_credit(cfg):
    credit = ledger(cfg)
    credit.attribute([Event("PlantPlaced", 1, 10, (("plant_type", "sunflower"),))], action_id=2)
    values = credit.attribute(
        [Event("SunProduced", 2, 99, (("source", "sunflower"), ("amount", 25)))],
        action_id=3,
    )
    assert values[2]["sun"] == pytest.approx(25)

    credit.attribute([Event("PlantPlaced", 3, 11, (("plant_type", "chomper"),))], action_id=4)
    values = credit.attribute(
        [
            Event(
                "DamageApplied",
                4,
                5,
                (("source", 11), ("health_damage", 270), ("armor_damage", 0)),
            ),
            Event("ZombieSwallowed", 4, 5, (("source", "chomper"),)),
        ],
        action_id=5,
    )
    assert values[4]["damage"] == pytest.approx(270 * 30 / 270)


@pytest.mark.parametrize(("kind", "expected_sign"), [("peashooter", 1), ("wall_nut", 1)])
def test_authoritative_bites_use_signed_per_plant_values(cfg, kind, expected_sign):
    credit = ledger(cfg)
    credit.attribute([Event("PlantPlaced", 1, 10, (("plant_type", kind),))], action_id=1)
    values = credit.attribute(
        [
            Event("PlantDamaged", 2, 10, (("damage", 100),)),
            # Removal follows the bite, so same-transition damage remains owned.
            Event("PlantRemoved", 2, 10),
        ],
        action_id=2,
    )
    assert values[1]["blocking"] * expected_sign > 0


def test_mower_cost_falls_back_to_current_transition(cfg):
    credit = ledger(cfg)
    values = credit.attribute([Event("MowerActivated", 1, 0)], action_id=9)
    assert values[9]["mower"] == pytest.approx(-cfg["reward"]["mower_value"])


def test_initial_unowned_plant_damage_is_diagnostic_only(cfg):
    credit = ledger(cfg, [plant(10, "wall_nut")])
    values = credit.attribute([Event("PlantDamaged", 1, 10, (("damage", 100),))], action_id=0)
    assert not values
    assert credit.last_diagnostics["unattributable_events"] == 1
