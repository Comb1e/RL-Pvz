"""Pairwise tile preferences and independent complete-return species/Dig fitting."""

import numpy as np
import torch
from torch.nn import functional as F

from pvz_rl.envs.actions import ActionSchema as A
from pvz_rl.envs.encoding import collate_observations
from pvz_rl.learning.action_routing import FitRoute
from pvz_rl.learning.objective import selected_action_targets
from pvz_rl.policy.sequential_q import action_groups, action_parts, outcome_weights


def pair_cross_entropy(values, pairs, temperature):
    """One root has unit weight, irrespective of how many alternatives survived."""
    q = values.float()
    if not pairs:
        return q.sum() * 0, q.new_zeros(())
    pairs = torch.as_tensor(pairs, device=q.device, dtype=torch.long)
    better, worse = q[pairs[:, 0]], q[pairs[:, 1]]
    return F.softplus((worse - better) / temperature).mean(), (better > worse).float().mean()


def row_identity(row):
    return tuple(int(row[k]) for k in ("env", "episode", "decision"))


class CounterfactualObjective:
    def __init__(self, runner, spool, device):
        self.runner = runner
        self.preferences = {tuple(r["identity"]): r["pairs"] for r in runner.records}
        self.feasibility = runner.feasibility
        self.temperature = runner.settings["temperature"]
        tile_counts, species_counts = np.zeros((3, 2), np.int64), np.zeros((3, 2), np.int64)
        for rows in np.array_split(spool.rows, max(1, len(spool.rows) // 8192 + 1)):
            for row in rows:
                action, accepted = int(row["action"]), bool(row["accepted"])
                group = 0 if action == 0 else (1 if action < A.dig_start else 2)
                identity = row_identity(row)
                if row["route"] != FitRoute.TILE_INVALID:
                    species_counts[group, int(accepted)] += 1
                if (
                    (accepted and action >= A.dig_start)
                    or identity in self.preferences
                    or identity in self.feasibility
                ):
                    tile_counts[group, int(accepted)] += 1
        share = runner.cfg["training"]["objective"]["accepted_outcome_share"]
        self.tile_weights = outcome_weights(
            tile_counts, device=device, dtype=torch.float32, accepted_share=share
        )
        self.species_weights = outcome_weights(
            species_counts, device=device, dtype=torch.float32, accepted_share=share
        )

    def loss(self, policy, rows, observations, stage):
        device = policy.device
        obs = collate_observations(observations, device).reshape(-1)
        actions = torch.as_tensor(np.array(rows["action"]), device=device, dtype=torch.long)
        accepted = torch.as_tensor(np.array(rows["accepted"]), device=device, dtype=torch.bool)
        targets = torch.as_tensor(np.array(rows["target"]), device=device, dtype=torch.float32)
        routes = torch.as_tensor(np.array(rows["route"]), device=device, dtype=torch.long)
        branches, positions = action_parts(actions)
        groups = action_groups(actions)
        indices, pairs, feasibility = [], [], []
        for i, row in enumerate(rows):
            identity = row_identity(row)
            if row["accepted"] and row["action"] >= A.dig_start:
                indices.append(i)
                pairs.append(None)
                feasibility.append(False)
            elif identity in self.preferences:
                indices.append(i)
                pairs.append(self.preferences[identity])
                feasibility.append(False)
            elif identity in self.feasibility and row["route"] == FitRoute.TILE_INVALID:
                indices.append(i)
                pairs.append([self.feasibility[identity]])
                feasibility.append(True)
        loss = torch.zeros((), device=device, requires_grad=True)
        metrics = torch.zeros(6, device=device)
        errors = []
        if indices:
            at = torch.tensor(indices, device=device)
            tables = policy.fitting_tiles(obs, branches[at], rows=at).float()
            weights = self.tile_weights[groups[at], accepted[at].long()]
            for j, (i, label, invalid) in enumerate(zip(indices, pairs, feasibility, strict=True)):
                if label is None:
                    value = (tables[j, positions[i]] - targets[i]).square()
                    errors.append(value.detach())
                else:
                    value, accuracy = pair_cross_entropy(tables[j], label, self.temperature)
                    metrics[0 if not invalid else 1] += value.detach() * weights[j]
                    metrics[4 if invalid else 2] += accuracy * weights[j]
                    metrics[5 if invalid else 3] += weights[j]
                loss = loss + value * weights[j]
        if stage == "joint":
            selected = torch.as_tensor(
                np.flatnonzero(rows["route"] != FitRoute.TILE_INVALID), device=device
            )
            if len(selected):
                values = policy.species_values(obs[selected]).float()
                prediction = values.gather(1, branches[selected, None]).flatten()
                y = selected_action_targets(
                    targets[selected],
                    routes[selected],
                    values.detach().amin(-1),
                    policy.cfg["training"]["objective"]["invalid_action_q_offset"],
                )
                error = (prediction - y).square()
                loss = (
                    loss
                    + (
                        error * self.species_weights[groups[selected], accepted[selected].long()]
                    ).sum()
                )
                errors.extend(error.detach().unbind())
        policy._last_species_consistency_loss = loss.detach() * 0
        policy._last_species_consistency_metrics = loss.new_zeros(4, dtype=torch.float64)
        return (
            loss,
            torch.stack(errors) if errors else loss.new_empty(0),
            loss.new_empty(0),
            metrics,
        )
