"""Scenario-local sky timing, retaining the pinned rules and random draws."""

import math

from pvz_rl.envs.action_timing import ActionPhaseGame


def validate_sky_sun_rate(rate, rules, max_tick=0):
    if type(rate) not in (int, float) or not math.isfinite(rate) or rate <= 0:
        raise ValueError("peashooter_sky_sun_rate must be finite and positive")
    maximum = max(
        rules.game["sky_first_max_ticks"],
        rules.game["sky_interval_max_ticks"] + rules.game["sky_interval_jitter_ticks"],
    )
    delay = maximum / rate
    if not math.isfinite(delay) or math.floor(delay) > 2**63 - 1 - math.ceil(max_tick):
        raise ValueError("peashooter_sky_sun_rate produces an unrepresentable sky countdown")
    return float(rate)


def scale_sky_delay(ticks, rate):
    return max(1, math.floor(ticks / rate))


def peashooter_sky_sun_rate(cfg):
    # Saved configurations without this setting retain their original timing.
    return cfg["training"].get("tile_scenarios", {}).get("peashooter_sky_sun_rate", 1.0)


class SkySunGame(ActionPhaseGame):
    """Scale only newly scheduled sky delays; restore never rescales a snapshot."""

    def __init__(self, rules=None, *, sky_sun_rate=1.0):
        super().__init__(rules)
        self.sky_sun_rate = validate_sky_sun_rate(sky_sun_rate, self.rules)

    def reset(self, level="standard", seed=42):
        observation = super().reset(level, seed)
        self._sky_due = scale_sky_delay(self._sky_due, self.sky_sun_rate)
        return observation

    def _advance(self):
        previous = self._sky_drops
        super()._advance()
        if self._sky_drops != previous:
            self._sky_due = self._tick + scale_sky_delay(
                self._sky_due - self._tick, self.sky_sun_rate
            )
