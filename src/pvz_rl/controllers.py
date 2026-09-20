"""Public-state controllers and a common strategy proposal interface."""

from dataclasses import dataclass

from pvz_game import ActionResult, Observation, Place, Wait

STRATEGIES = ("wait", "economy", "attack", "defense", "emergency")


@dataclass
class PublicBoard:
    observation: Observation
    legal: frozenset

    def observe(self):
        return self.observation

    def validate_action(self, action):
        return ActionResult(action in self.legal)


def strategy_candidates(board: PublicBoard) -> tuple:
    """At most one placement per strategy; deterministic ties use row then column.

    Placement preferences and threat triggers adapt the pinned acceptance controller.
    These routines never modify a game or inspect a spawn schedule.
    """
    obs = board.observation
    rows = range(obs.rows)
    units = obs.units_per_tile
    lanes = [[z for z in obs.zombies if z.row == row] for row in rows]
    nearest = [min((z.x / units for z in lane), default=20) for lane in lanes]
    firepower = [
        sum(
            {"peashooter": 1, "repeater": 2, "snow_pea": 1}.get(p.plant_type, 0)
            for p in obs.plants
            if p.row == row
        )
        for row in rows
    ]

    def place(kind, row, cols):
        return next((a for col in cols if (a := Place(kind, row, col)) in board.legal), None)

    economy = None
    if sum(p.plant_type == "sunflower" for p in obs.plants) < 8:
        economy = next(
            (
                a
                for col in (0, 1)
                for row in (2, 0, 4, 1, 3)
                if (a := place("sunflower", row, [col]))
            ),
            None,
        )
    attack = None
    for row in sorted(rows, key=lambda r: (not lanes[r], firepower[r], nearest[r], r)):
        if firepower[row] >= 5:
            continue
        kinds = (
            ("peashooter", "repeater", "snow_pea")
            if firepower[row] == 0
            else ("repeater", "snow_pea", "peashooter")
        )
        attack = next((a for kind in kinds if (a := place(kind, row, [2, 3, 4, 5]))), None)
        if attack:
            break
    defense, emergency = None, None
    for row in sorted(rows, key=lambda r: (nearest[r], r)):
        if not lanes[row]:
            continue
        if defense is None and not any(
            p.row == row and p.plant_type == "wall_nut" for p in obs.plants
        ):
            col = max(0, min(6, int(nearest[row] - 0.8)))
            defense = place("wall_nut", row, [col])
        if emergency is None:
            if nearest[row] > 5 and not any(
                p.row == row and p.plant_type == "potato_mine" for p in obs.plants
            ):
                emergency = place("potato_mine", row, [2, 1, 3])
            if emergency is None and nearest[row] < 4:
                col = max(0, min(8, int(nearest[row])))
                kinds = (
                    ("cherry_bomb", "chomper")
                    if len(lanes[row]) >= 2
                    else ("chomper", "cherry_bomb")
                )
                emergency = next(
                    (a for kind in kinds if (a := place(kind, row, [col, max(0, col - 1)]))), None
                )
                if emergency is None:
                    emergency = next(
                        (
                            a
                            for adjacent in (row - 1, row + 1)
                            if (a := place("cherry_bomb", adjacent, [col]))
                        ),
                        None,
                    )
    return Wait(), economy, attack, defense, emergency
