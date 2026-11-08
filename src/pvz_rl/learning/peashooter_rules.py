"""Private prioritized lane labels and column-distribution cross-entropy."""

import math

import numpy as np
import torch

from pvz_rl.envs.actions import ActionSchema as A
from pvz_rl.envs.encoding import collate_observations
from pvz_rl.envs.teacher_rules import (
    candidate_targets as candidate_targets,
)
from pvz_rl.envs.teacher_rules import (
    choose_rule as choose_rule,
)
from pvz_rl.envs.teacher_rules import (
    lane_facts as lane_facts,
)
from pvz_rl.envs.teacher_rules import (
    lane_snapshot as lane_snapshot,
)
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
    for key in ("prepare_workers",):
        if type(settings.get(key, 0)) is not int or settings.get(key, 0) < 0:
            raise ValueError(f"Peashooter {key} must be a nonnegative integer")
    if "n_envs" in settings and (type(settings["n_envs"]) is not int or settings["n_envs"] < 1):
        raise ValueError("Peashooter n_envs must be a positive integer")
    if "snapshot_cache_mib" in settings and (
        type(settings["snapshot_cache_mib"]) is not int or settings["snapshot_cache_mib"] < 1
    ):
        raise ValueError("Peashooter snapshot_cache_mib must be a positive integer")


def column_targets(preferred, *, decay, device=None):
    """Dense labels only; zeros never become an inference mask."""
    # These labels are fixed data, not differentiable model outputs. Build one
    # FP32 host panel and transfer it once instead of launching several CUDA
    # kernels and tiny host transfers separately for every board.
    target = np.zeros((len(preferred), A.tiles), dtype=np.float32)
    for index, tiles in enumerate(preferred):
        if not tiles or len(set(tiles)) != len(tiles) or any(t < 0 or t >= A.tiles for t in tiles):
            raise ValueError("Peashooter preferred set must contain distinct valid tiles")
        ids = np.asarray(tiles, dtype=np.int64)
        columns = ids % A.cols
        weights = np.power(np.float32(decay), columns - columns.min(), dtype=np.float32)
        target[index, ids] = weights / weights.sum(dtype=np.float32)
    return torch.from_numpy(target).to(device=device)


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
        self.weights = outcome_weights(
            counts,
            device=device,
            dtype=torch.float32,
            accepted_share=teacher.cfg["training"]["objective"]["accepted_outcome_share"],
        )

    def label(self, row):
        key = tuple(int(row[k]) for k in ("env", "episode", "decision"))
        label = self.teacher.labels.get(key)
        return label if label and label["tiles"] else None

    def loss(self, policy, rows, observations, *, features=None, memory=None):
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
            if features is None:
                values = policy.fitting_tiles(obs, torch.full_like(at, branch), rows=at)
            else:
                del memory
                values = policy.tile_values(features[at])
            losses, correct = column_cross_entropy(
                values,
                preferred,
                decay=self.teacher.settings["column_decay"],
                temperature=self.teacher.settings["softmax_temperature"],
            )
            accepted = torch.as_tensor(
                np.array(rows["accepted"][selected]), device=policy.device
            ).long()
            weights = self.weights[1, accepted]
            loss = (losses * weights).sum()
            metrics = torch.stack((loss.detach(), (correct * weights).sum(), weights.sum()))
        policy._last_species_consistency_loss = loss.detach() * 0
        policy._last_species_consistency_metrics = loss.new_zeros(4, dtype=torch.float64)
        return loss, loss.new_empty(0), loss.new_empty(0), metrics
