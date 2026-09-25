"""Unmodified acceptance-controller function from Leafy's own game repository.

Source: tools/record_playthroughs.py at b3cfbd886ab378313a1fdb57ee43a9a1b36a0793.
Kept frozen to avoid changing the comparison while developing learned strategies.
The argument is a PublicBoard facade exposing only observations and legal actions.
"""

from pvz_game import Place, Wait


def choose_action(game):
    obs = game.observe()
    tiles = {(p.row, p.col): p for p in obs.plants}
    unit = obs.units_per_tile
    lanes = [[z for z in obs.zombies if z.row == row] for row in range(5)]
    nearest = [min((z.x / unit for z in zs), default=20) for zs in lanes]
    firepower = [
        sum(
            {"peashooter": 1, "repeater": 2, "snow_pea": 1}.get(p.plant_type, 0)
            for p in obs.plants
            if p.row == row
        )
        for row in range(5)
    ]
    flowers = sum(p.plant_type == "sunflower" for p in obs.plants)
    urgency = [
        min(
            (z.x / (400 if z.has_pole else 240 if z.zombie_type == "flag" else 200) for z in zs),
            default=1000,
        )
        for zs in lanes
    ]

    def place(kind, row, columns):
        for col in columns:
            action = Place(kind, row, col)
            if game.validate_action(action).accepted:
                return action
        return None

    # Use inexpensive mines to buy time for the opening economy.
    for row in sorted(range(5), key=lambda row: urgency[row]):
        protected = any(p.plant_type == "potato_mine" and p.row == row for p in obs.plants)
        if lanes[row] and firepower[row] < 2 and not protected and urgency[row] > 25:
            action = place("potato_mine", row, [2, 1, 3])
            if action:
                return action
    # Recover threatened lanes or consume heavy armor before it reaches the economy.
    for row in sorted(range(5), key=lambda row: urgency[row]):
        heavy = any(z.armor or z.has_pole for z in lanes[row])
        if nearest[row] < 2.5 or (heavy and nearest[row] < 4 and firepower[row] < 2):
            col = max(0, min(8, int(nearest[row])))
            kinds = (
                ("cherry_bomb", "chomper") if len(lanes[row]) >= 2 else ("chomper", "cherry_bomb")
            )
            for kind in kinds:
                columns = [col, max(0, col - 1)]
                action = place(kind, row, columns)
                if action:
                    return action
            # An adjacent lane's bomb can clear a crowded or occupied tile too.
            if obs.sun >= 150:
                for adjacent in (row - 1, row + 1):
                    action = place("cherry_bomb", adjacent, [col, max(0, col - 1)])
                    if action:
                        return action
            if heavy and obs.sun < 150:
                # Buy time with a wall instead of letting armor walk through the shooters.
                if not any(p.plant_type == "wall_nut" and p.row == row for p in obs.plants):
                    action = place("wall_nut", row, [max(0, col - 1)])
                    if action:
                        return action
                if nearest[row] < 4:
                    return Wait()
    # Establish one shooter in each occupied lane before extending the economy.
    for row in sorted(range(5), key=lambda row: urgency[row]):
        protected = any(p.plant_type == "potato_mine" and p.row == row for p in obs.plants)
        if lanes[row] and firepower[row] == 0 and not (protected and len(lanes[row]) == 1):
            for kind in ("peashooter", "repeater", "snow_pea"):
                action = place(kind, row, [2, 3, 4])
                if action:
                    return action
            if obs.sun < 100:
                return Wait()
    for row in sorted(range(5), key=lambda row: urgency[row]):
        if (
            lanes[row]
            and firepower[row] > 0
            and nearest[row] < 7
            and sum(z.health + z.armor for z in lanes[row]) > 350
            and not any(p.plant_type == "wall_nut" and p.row == row for p in obs.plants)
        ):
            col = max(3, min(6, int(nearest[row] - 0.8)))
            action = place("wall_nut", row, [col])
            if action:
                return action
    if flowers < 8:
        for col in (0, 1):
            for row in (2, 0, 4, 1, 3):
                if (row, col) not in tiles:
                    action = place("sunflower", row, [col])
                    if action:
                        return action
    # Spread sustained damage across the board, prioritizing pressured lanes.
    for row in sorted(range(5), key=lambda row: (not lanes[row], firepower[row], nearest[row])):
        if firepower[row] < 5:
            kinds = (
                ("peashooter", "repeater", "snow_pea")
                if firepower[row] == 0
                else ("repeater", "snow_pea", "peashooter")
            )
            for kind in kinds:
                action = place(kind, row, [2, 3, 4, 5])
                if action:
                    return action
    for row in range(5):
        if not any(p.plant_type == "snow_pea" and p.row == row for p in obs.plants):
            action = place("snow_pea", row, [5, 4, 3, 2])
            if action:
                return action
    return Wait()
