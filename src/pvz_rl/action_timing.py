"""Research-local action phase for the pinned engine; combat ticks are unchanged.

PVZ 1.2.1 has no public zero-time action API. This subclass suppresses only the
advance hook during a zero-tick call, reusing the engine's validator and complete
action implementation. Installed source hashes are checked before experiments.
"""

from pvz_game import Dig, Game, Place, Wait


def per_tick_actions(cfg):
    return cfg["environment"].get("action_timing", "fixed") == "per_tick"


class ActionPhaseGame(Game):
    def step(self, action=Wait(), *, ticks=1):
        if type(ticks) is not int or ticks != 0:
            return super().step(action, ticks=ticks)
        if not isinstance(action, (Place, Dig)):
            raise ValueError("Zero-tick calls require a planting or digging action")
        self._applying_action = True
        try:
            return super().step(action, ticks=1)
        finally:
            self._applying_action = False

    def _advance(self):
        if not getattr(self, "_applying_action", False):
            super()._advance()
