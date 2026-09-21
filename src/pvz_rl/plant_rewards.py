"""Optional plant-role shaping and public-event loss penalties.

Offensive investment is a potential, so planting/digging cannot farm bonuses.
Plant destruction is attributed to the engine's removal reason, not disappearance.
"""

from pvz_game import Status


def enabled(cfg):
    r = cfg["reward"]
    return r.get("offensive_plant_weight", 0) > 0 or r.get("eaten_plant_penalty", 0) > 0


def offensive_potential(obs, cfg):
    if obs.status != Status.RUNNING:
        return 0.0
    roles = cfg["reward"].get("offensive_plants", ())
    return cfg["reward"].get("offensive_plant_weight", 0.0) * sum(
        p.plant_type in roles for p in obs.plants
    )


def plant_reward_parts(before, after, cfg, shaped, events):
    settings = cfg["reward"]
    planted = eaten = 0
    if enabled(cfg):
        roles = settings.get("offensive_plants", ())
        plants = {p.id: p.plant_type for p in before.plants}
        for event in events:
            if event.kind == "PlantPlaced":
                kind = event.get("plant_type")
                plants[event.entity_id] = kind
                planted += kind in roles
            elif event.kind == "PlantRemoved":
                kind = plants.pop(event.entity_id, None)
                if event.get("reason") == "eaten":
                    if kind is None:
                        raise RuntimeError("Cannot attribute eaten plant to public plant type")
                    eaten += kind != "wall_nut"
    return {
        "offensive_plantings": planted,
        "plants_eaten": eaten,
        "offensive_shaping": (
            settings["gamma"] * offensive_potential(after, cfg) - offensive_potential(before, cfg)
            if shaped
            else 0.0
        ),
        "plant_eaten_penalty": -settings.get("eaten_plant_penalty", 0.0) * eaten,
    }
