"""Causal credit for delayed simulator effects.

The simulator reports entity IDs, while the learner stores decisions.  This
small host-side ledger bridges those identities without exposing IDs to the
policy.  CUDA collection may use the same event contract when the backend
publishes detailed event facts; aggregate reward values remain valid when a
backend cannot expose an origin for a particular environmental event.
"""

from __future__ import annotations

from collections import defaultdict


class CausalCreditLedger:
    """Map plant/projectile effects back to the decision that created them."""

    def __init__(self, cfg, rules):
        self.cfg = cfg
        self.rules = rules
        self.reset()

    def reset(self, episode_id=None):
        self.episode_id = episode_id
        self.plants = {}
        self.projectiles = {}
        self.projectile_kinds = {}
        self.plant_kinds = {}
        self.last_diagnostics = {
            "credited_events": 0,
            "unattributable_events": 0,
            "unattributable_by_kind": {},
            "damage_by_plant": defaultdict(float),
            "blocking_by_plant": defaultdict(float),
            "sun_by_plant": defaultdict(float),
        }

    def seed_observation(self, observation):
        """Register plants already present at episode start as unowned."""
        self.plants = {plant.id: None for plant in observation.plants}
        self.plant_kinds = {plant.id: plant.plant_type for plant in observation.plants}
        self.projectiles.clear()
        self.projectile_kinds.clear()

    def _origin(self, entity_id):
        if entity_id is None:
            return None
        if entity_id in self.plants:
            return self.plants[entity_id]
        return self.projectiles.get(entity_id)

    def _record_unattributed(self, kind):
        self.last_diagnostics["unattributable_events"] += 1
        counts = self.last_diagnostics["unattributable_by_kind"]
        counts[kind] = counts.get(kind, 0) + 1

    def _credit(self, values, origin, field, amount, *, plant_kind=None):
        if origin is None:
            return False
        values[origin][field] += float(amount)
        self.last_diagnostics["credited_events"] += 1
        if plant_kind is not None:
            key = f"{plant_kind}"
            self.last_diagnostics[f"{field}_by_plant"][key] += float(amount)
        return True

    def _plant_origin_for_event(self, event):
        """Resolve an engine event's plant identity before its removal.

        ``SunProduced`` and ``ZombieSwallowed`` carry the source plant in
        ``details`` and the emitting entity in ``entity_id``.  Older replay
        adapters sometimes omit the entity id, so the active plant of the
        requested kind is used as a deterministic fallback.
        """
        entity_id = event.entity_id
        origin = self._origin(entity_id)
        if origin is not None:
            return origin, self.plant_kinds.get(entity_id)
        source = event.get("source")
        if isinstance(source, int):
            return self._origin(source), self.plant_kinds.get(source)
        if isinstance(source, str):
            candidates = sorted(
                (pid, self.plants[pid])
                for pid, kind in self.plant_kinds.items()
                if kind == source and pid in self.plants and self.plants[pid] is not None
            )
            if candidates:
                pid, action = candidates[0]
                return action, source
        return None, source if isinstance(source, str) else None

    def attribute(self, events, *, action_id):
        """Return raw source credits grouped by origin action id.

        Event order is authoritative.  A removal is applied after earlier
        events in the same stream, so detonation, damage, and production are
        still attributed to the plant that emitted them.
        """
        values = defaultdict(lambda: {"damage": 0.0, "blocking": 0.0, "sun": 0.0, "mower": 0.0})
        self.last_diagnostics = {
            "credited_events": 0,
            "unattributable_events": 0,
            "unattributable_by_kind": {},
            "damage_by_plant": defaultdict(float),
            "blocking_by_plant": defaultdict(float),
            "sun_by_plant": defaultdict(float),
        }
        block_values = self.cfg.get("reward", {}).get("action", {}).get("block_value_by_plant", {})
        basic_hp = float(self.rules.zombies["basic"]["health"])
        basic_value = float(self.cfg["reward"]["basic_zombie_value"])
        sun_multiplier = float(
            self.cfg.get("reward", {}).get("action", {}).get("sun_credit_multiplier", 1.0)
        )

        def bucket(origin):
            # Effects from initial plants are intentionally reportable but have
            # no learner row to mutate.  Keep them under None for diagnostics.
            return values[origin]

        for event in events:
            kind = event.kind
            if kind == "PlantPlaced":
                self.plants[event.entity_id] = action_id
                self.plant_kinds[event.entity_id] = event.get("plant_type")
            elif kind == "ProjectileFired":
                source = event.get("source")
                self.projectiles[event.entity_id] = self._origin(source)
                self.projectile_kinds[event.entity_id] = self.plant_kinds.get(source)
            elif kind == "DamageApplied":
                origin = self._origin(event.get("source"))
                amount = float(event.get("health_damage", 0)) + float(
                    event.get("armor_damage", 0)
                )
                if amount > 0 and origin is not None:
                    self._credit(
                        values,
                        origin,
                        "damage",
                        basic_value * amount / basic_hp,
                        plant_kind=self.projectile_kinds.get(event.get("source"))
                        or self.plant_kinds.get(event.get("source")),
                    )
                elif amount > 0 and event.get("source", 0) > 0:
                    self._record_unattributed(kind)
            elif kind == "SunProduced":
                origin, plant_kind = self._plant_origin_for_event(event)
                amount = float(event.get("amount", 0))
                if amount > 0 and event.get("source") != "sky" and origin is not None:
                    self._credit(
                        values,
                        origin,
                        "sun",
                        sun_multiplier * amount,
                        plant_kind=plant_kind,
                    )
                elif amount > 0 and event.get("source") != "sky":
                    self._record_unattributed(kind)
            elif kind == "PlantDamaged":
                plant_id = event.entity_id
                origin = self._origin(plant_id)
                # The event stream does not include plant_type, so resolve it
                # from the public observation map populated by ``register``.
                plant_kind = self.plant_kinds.get(plant_id)
                amount = float(event.get("damage", 0))
                if origin is not None and plant_kind in block_values and amount > 0:
                    self._credit(
                        values,
                        origin,
                        "blocking",
                        float(block_values[plant_kind]) * amount,
                        plant_kind=plant_kind,
                    )
                elif amount > 0:
                    self._record_unattributed(kind)
            elif kind == "ZombieSwallowed":
                # The lethal DamageApplied event already carries the Chomper
                # source.  Keep this event as an attribution audit marker and
                # never add a second kill reward.
                origin, plant_kind = self._plant_origin_for_event(event)
                if origin is None:
                    self._record_unattributed(kind)
            elif kind == "MowerActivated":
                self._credit(
                    values,
                    action_id,
                    "mower",
                    -float(self.cfg["reward"]["mower_value"]),
                )
            elif kind == "PlantRemoved":
                # The engine emits PlantRemoved after the effect that caused
                # it.  Retain kind/ownership until this point, then retire.
                self.plants.pop(event.entity_id, None)
                self.plant_kinds.pop(event.entity_id, None)
            elif kind in ("ProjectileRemoved", "ProjectileExpired"):
                self.projectiles.pop(event.entity_id, None)
                self.projectile_kinds.pop(event.entity_id, None)

        self.last_diagnostics = {
            **self.last_diagnostics,
            "damage_by_plant": dict(self.last_diagnostics["damage_by_plant"]),
            "blocking_by_plant": dict(self.last_diagnostics["blocking_by_plant"]),
            "sun_by_plant": dict(self.last_diagnostics["sun_by_plant"]),
        }
        return {key: dict(value) for key, value in values.items()}

    def register_plant_kinds(self, observation):
        """Cache public plant kinds for PlantDamaged attribution."""
        self.plant_kinds = {plant.id: plant.plant_type for plant in observation.plants}

