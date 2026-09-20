"""Potential shaping, with true terminals separated from external time cutoffs."""

from pvz_game import Observation, Status


def potential(obs: Observation, cfg: dict) -> float:
    if obs.status != Status.RUNNING:
        return 0.0
    r = cfg["reward"]
    flowers = sum(p.plant_type == "sunflower" for p in obs.plants)
    return (
        r["defeated_weight"] * obs.counts.defeated / max(1, obs.counts.initial_total)
        + r["sun_weight"] * min(obs.sun / r["sun_target"], 1)
        + r["flower_weight"] * min(flowers / r["flower_target"], 1)
    )


def reward_parts(before: Observation, after: Observation, cfg: dict, shaped: bool) -> dict:
    terminal = 1.0 if after.status == Status.WON else -1.0 if after.status == Status.LOST else 0.0
    shaping = (
        cfg["reward"]["gamma"] * potential(after, cfg) - potential(before, cfg) if shaped else 0.0
    )
    return {"terminal": terminal, "shaping": shaping, "total": terminal + shaping}
