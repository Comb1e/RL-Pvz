"""Wait/dig/plant, conditional species, then tile; compact engine transport."""

import math

import torch
from sb3_contrib.common.maskable.distributions import MaskableCategorical, MaskableDistribution
from torch import nn

from .actions import ActionSchema as A


class GroupedDistribution(MaskableDistribution):
    action_dim, logit_dim = A.size, A.logit_size

    def __init__(self, epsilon=0.0):
        super().__init__()
        self.epsilon = epsilon

    def proba_distribution_net(self, latent_dim):
        return nn.Linear(latent_dim, self.logit_dim)

    def proba_distribution(self, action_logits, masks=None):
        self.logits = action_logits.reshape(-1, self.logit_dim)
        self.apply_masking(masks)
        return self

    def apply_masking(self, masks):
        mask = (
            torch.ones((len(self.logits), A.size), dtype=torch.bool, device=self.logits.device)
            if masks is None
            else torch.as_tensor(masks, dtype=torch.bool, device=self.logits.device).reshape(
                -1, A.size
            )
        )
        tile_mask = A.tile_masks(mask)
        self.legal_tile_counts = tile_mask.sum(-1)
        available = tile_mask.any(-1)
        self.available_plants = available[:, : A.plant_types]
        plant_available = self.available_plants.any(-1)
        kind_mask = torch.stack((mask[:, 0], available[:, -1], plant_available), -1)
        if not kind_mask.any(-1).all():
            raise ValueError("Grouped policy requires at least one legal action per observation")
        # Dummy conditionals have no joint mass and no action-likelihood gradient.
        plant_mask = self.available_plants.clone()
        plant_mask[:, 0] |= ~plant_available
        self.learned_types = MaskableCategorical(logits=self.logits[:, : A.kinds], masks=kind_mask)
        self.learned_plants = MaskableCategorical(
            logits=self.logits[:, A.kinds : A.tile_logits_start], masks=plant_mask
        )
        self.types, self.plants = self.learned_types, self.learned_plants
        if self.epsilon:
            # Preserve uniform exploration over wait and available species, never dig.
            # Factor the resulting joint mixture, rather than independently mixing heads.
            leaf_mask = torch.cat((mask[:, :1], self.available_plants, available[:, -1:]), -1)
            exploratory = leaf_mask.clone()
            exploratory[:, -1] = False
            count = exploratory.sum(-1, keepdim=True)
            learned = torch.cat(
                (
                    self.learned_types.logits[:, A.wait : A.wait + 1],
                    self.learned_types.logits[:, A.plant : A.plant + 1]
                    + self.learned_plants.logits,
                    self.learned_types.logits[:, A.dig : A.dig + 1],
                ),
                -1,
            )
            prior = torch.where(
                exploratory, -count.clamp_min(1).to(self.logits.dtype).log(), -torch.inf
            )
            weight = torch.where(count > 0, self.logits.new_tensor(math.log1p(-self.epsilon)), 0)
            mixed = torch.logaddexp(learned + weight, prior + math.log(self.epsilon))
            kind_logits = torch.stack((mixed[:, 0], mixed[:, -1], mixed[:, 1:-1].logsumexp(-1)), -1)
            self.types = MaskableCategorical(logits=kind_logits, masks=kind_mask)
            self.plants = MaskableCategorical(logits=mixed[:, 1:-1], masks=plant_mask)
        safe_mask = tile_mask.clone()
        safe_mask[:, :, 0] |= ~available
        self.locations = MaskableCategorical(
            logits=self.logits[:, A.tile_logits_start :].reshape(-1, A.tile_groups, A.tiles),
            masks=safe_mask,
        )

    @property
    def branch_probs(self):
        return torch.cat(
            (
                self.types.probs[:, A.plant : A.plant + 1] * self.plants.probs,
                self.types.probs[:, A.dig : A.dig + 1],
            ),
            -1,
        )

    @property
    def probs(self):
        tiles = self.branch_probs[:, :, None] * self.locations.probs
        return torch.cat((self.types.probs[:, A.wait : A.wait + 1], tiles.flatten(1)), -1)

    def log_prob(self, actions):
        kinds, plants, tiles = A.unpack(actions.long().flatten())
        groups = torch.where(kinds == A.dig, A.plant_types, plants)
        batch = torch.arange(len(kinds), device=kinds.device)
        return (
            self.types.log_prob(kinds)
            + torch.where(kinds == A.plant, self.plants.log_prob(plants), 0)
            + torch.where(kinds == A.wait, 0, self.locations.logits[batch, groups, tiles])
        )

    def entropy_parts(self):
        """The three additive contributions to true joint entropy."""
        return (
            self.types.entropy(),
            self.types.probs[:, A.plant] * self.plants.entropy(),
            (self.branch_probs * self.locations.entropy()).sum(-1),
        )

    def entropy(self):
        return sum(self.entropy_parts())

    def _actions(self, deterministic):
        # Deterministic evaluation excludes the injected exploration mixture.
        kinds = self.learned_types.probs.argmax(-1) if deterministic else self.types.sample()
        plants = self.learned_plants.probs.argmax(-1) if deterministic else self.plants.sample()
        groups = torch.where(kinds == A.dig, A.plant_types, plants)
        # Sample only the selected tile distribution, not all nine maps.
        batch = torch.arange(len(kinds), device=kinds.device)
        probs = self.locations.probs[batch, groups]
        tiles = probs.argmax(-1) if deterministic else torch.multinomial(probs, 1).squeeze(-1)
        return A.pack(kinds, plants, tiles)

    def mode(self):
        return self._actions(True)

    def sample(self):
        return self._actions(False)

    def actions_from_params(self, action_logits, deterministic=False):
        return self.proba_distribution(action_logits).get_actions(deterministic)

    def log_prob_from_params(self, action_logits):
        actions = self.actions_from_params(action_logits)
        return actions, self.log_prob(actions)
