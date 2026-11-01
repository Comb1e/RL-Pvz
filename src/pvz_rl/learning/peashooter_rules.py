"""Private prioritized lane labels and column-distribution cross-entropy."""

import copy
import math

import numpy as np
import torch

from pvz_rl.envs.actions import ActionSchema as A
from pvz_rl.envs.encoding import collate_observations
from pvz_rl.envs.scratch_game import EngineSnapshot, restored_game
from pvz_rl.policy.sequential_q import outcome_weights

RULE_OBJECTIVE = "prioritized_lane_cross_entropy_v1"
RULE_PROTOCOL = "visible_lane_oracle_v1"


def rule_settings(cfg):
    return cfg["training"].get("peashooter_rules", {})


def rules_enabled(cfg, phase):
    return phase == "peashooter" and rule_settings(cfg).get("fitting_objective") == RULE_OBJECTIVE


def validate_rule_settings(cfg):
    settings = rule_settings(cfg)
    if not settings:
        return  # Saved runs retain their original objective.
    if settings.get("fitting_objective") != RULE_OBJECTIVE:
        raise ValueError("Unsupported Peashooter fitting objective")
    for key in ("column_decay", "softmax_temperature"):
        value = settings.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Peashooter {key} must be finite and positive")
    if settings["column_decay"] >= 1:
        raise ValueError("Peashooter column_decay must be strictly less than one")
    for key in ("oracle_lanes", "oracle_vram_mib", "oracle_chunk_ticks"):
        if type(settings.get(key)) is not int or settings[key] < 1:
            raise ValueError(f"Peashooter {key} must be a positive integer")
    if cfg["training"].get("counterfactual", {}).get("peashooter", {}).get("enabled"):
        raise ValueError("Peashooter rule labels and counterfactual fitting are mutually exclusive")


def lane_facts(snapshot):
    counts = [0] * A.rows
    occupied = {p["row"] * A.cols + p["col"] for p in snapshot["plants"]}
    for plant in snapshot["plants"]:
        if plant["kind"] == "peashooter" and plant["health"] > 0:
            counts[plant["row"]] += 1
    threats = {}
    for zombie in snapshot["zombies"]:
        if zombie["health"] > 0 and not zombie["headless"]:
            row = zombie["row"]
            threats[row] = min(threats.get(row, zombie["x"]), zombie["x"])
    legal = [[r * A.cols + c for c in range(A.cols) if r * A.cols + c not in occupied]
             for r in range(A.rows)]
    return counts, threats, legal


def choose_rule(snapshot, covered):
    """Return a fixed preferred set, or the urgent rows requiring candidate tests."""
    counts, threats, legal = lane_facts(snapshot)
    blocked = sum(not legal[r] for r in threats if not covered.get(r, False))
    if not threats or all(covered.get(r, False) for r in threats):
        minimum = min(counts)
        tiles = [t for r in range(A.rows) if counts[r] == minimum for t in legal[r]]
        return dict(rule=1, tiles=tiles, rows=[], blocked=blocked)
    empty = [r for r in threats if counts[r] == 0]
    if empty:
        closest = min(threats[r] for r in empty)
        return dict(rule=2, tiles=[], rows=[r for r in empty if threats[r] == closest], blocked=blocked)
    endangered = [r for r in threats if not covered.get(r, False) and legal[r]]
    if endangered:
        closest = min(threats[r] for r in endangered)
        tiles = [legal[r][0] for r in endangered if threats[r] == closest]
    else:
        tiles = []
    return dict(rule=3, tiles=tiles, rows=[], blocked=blocked)


def candidate_targets(snapshot, rows, successful):
    legal = lane_facts(snapshot)[2]
    targets, fallback = [], []
    for row in rows:
        winners = [t for t in legal[row] if successful.get(t, False)]
        if winners:
            targets.extend(winners)
        elif legal[row]:
            targets.append(legal[row][0])
            fallback.append(row)
    return targets, fallback


def lane_snapshot(snapshot, row, rules):
    """A validated current lane, with no future spawns or other-lane RNG draws.

    Synthetic *past* spawn entries reconcile engine counters with the retained
    live entities and defeated bodies. Physical fields and RNG are copied, not
    reconstructed; the original sky countdown is never rescaled.
    """
    result = copy.deepcopy(snapshot)
    for name in ("plants", "zombies", "projectiles"):
        result[name] = [e for e in result[name] if e["row"] == row]
    zombies = result["zombies"]
    result["level"] = dict(
        name="peashooter-visible-lane", initial_sun=result["sun"], plants=[], mowers=False,
        spawns=[dict(tick=result["tick"], zombie_type=z["kind"], row=row, x=max(0, z["x"]), wave=1)
                for z in zombies],
    )
    result.update(spawn_index=len(zombies), defeated=sum(z["headless"] for z in zombies),
                  wave=int(bool(zombies)), status="running" if any(not z["headless"] for z in zombies) else "won")
    for mower in result["mowers"]:
        mower["state"] = "spent"
    game = restored_game(result, rules)
    return EngineSnapshot.capture(game)


def column_targets(preferred, *, decay, device=None):
    """Dense labels only; zeros never become an inference mask."""
    target = torch.zeros((len(preferred), A.tiles), device=device, dtype=torch.float32)
    for index, tiles in enumerate(preferred):
        if not tiles or len(set(tiles)) != len(tiles) or any(t < 0 or t >= A.tiles for t in tiles):
            raise ValueError("Peashooter preferred set must contain distinct valid tiles")
        ids = torch.as_tensor(tiles, device=device, dtype=torch.long)
        columns = ids.remainder(A.cols)
        weights = decay ** (columns - columns.min()).float()
        target[index, ids] = weights / weights.sum()
    return target


def column_cross_entropy(values, preferred, *, decay, temperature):
    target = column_targets(preferred, decay=decay, device=values.device)
    logits = values.float()
    logits = (logits - logits.max(-1, keepdim=True).values) / temperature
    logp = logits.log_softmax(-1)
    losses = -(target * logp).sum(-1)
    mode = target.max(-1, keepdim=True).values
    accuracy = (target.gather(1, logits.argmax(-1, keepdim=True)) == mode).float().flatten()
    return losses, accuracy


class PeashooterObjective:
    def __init__(self, teacher, spool, device):
        self.teacher = teacher
        counts = np.zeros((3, 2), np.int64)
        self.examples = 0
        for row in spool.rows:
            if self.label(row) is not None:
                counts[1, int(row["accepted"])] += 1
                self.examples += 1
        self.weights = outcome_weights(counts, device=device, dtype=torch.float32,
            accepted_share=teacher.cfg["training"]["objective"]["accepted_outcome_share"])

    def label(self, row):
        key = tuple(int(row[k]) for k in ("env", "episode", "decision"))
        label = self.teacher.labels.get(key)
        return label if label and label["tiles"] else None

    def loss(self, policy, rows, observations):
        selected, preferred = [], []
        for i, row in enumerate(rows):
            label = self.label(row)
            if label:
                selected.append(i)
                preferred.append(label["tiles"])
        loss = torch.zeros((), device=policy.device, requires_grad=True)
        metrics = torch.zeros(3, device=policy.device)
        if selected:
            at = torch.as_tensor(selected, device=policy.device)
            obs = collate_observations(observations, policy.device)
            branch = policy.cfg["environment"]["plants"].index("peashooter") + 1
            values = policy.fitting_tiles(obs, torch.full_like(at, branch), rows=at)
            losses, correct = column_cross_entropy(values, preferred,
                decay=self.teacher.settings["column_decay"], temperature=self.teacher.settings["softmax_temperature"])
            accepted = torch.as_tensor(np.array(rows["accepted"][selected]), device=policy.device).long()
            weights = self.weights[1, accepted]
            loss = (losses * weights).sum()
            metrics = torch.stack((loss.detach(), (correct * weights).sum(), weights.sum()))
        policy._last_species_consistency_loss = loss.detach() * 0
        policy._last_species_consistency_metrics = loss.new_zeros(4, dtype=torch.float64)
        return loss, loss.new_empty(0), loss.new_empty(0), metrics
