"""Conditional planting probabilities; greedy choices have no likelihood."""

import torch
from sb3_contrib.common.maskable.distributions import MaskableDistribution
from torch import nn

from pvz_rl.envs.actions import ActionSchema as A

ACTION_DISTRIBUTION = "conditional_plant_v1"


class MaskedCategorical:
    def __init__(self, logits, mask):
        self.masks = mask
        self.logits = logits.masked_fill(~mask, -1e8).log_softmax(-1)
        self.probs = self.logits.exp() * mask

    def log_prob(self, value):
        return self.logits.gather(-1, value.long().unsqueeze(-1)).squeeze(-1)

    def entropy(self):
        return -(self.probs * self.logits).sum(-1)


class GroupedDistribution(MaskableDistribution):
    """Eight species followed by a legal tile, with one complete-action mixture.

    Non-plantable states have dummy conditionals with zero public mass. They
    never enter actor fitting and produce no likelihood gradient.
    """

    action_dim, logit_dim = A.size, A.plant_types * (A.tiles + 1)

    def __init__(self, epsilon=0.0, validate_args=True):
        super().__init__()
        if not 0 <= epsilon <= 1:
            raise ValueError("Exploration epsilon must be in [0, 1]")
        self.epsilon, self.validate_args = float(epsilon), validate_args

    def proba_distribution_net(self, latent_dim):
        return nn.Linear(latent_dim, self.logit_dim)

    def proba_distribution(self, action_logits, masks=None):
        self.logits = action_logits.reshape(-1, self.logit_dim)
        self.apply_masking(masks)
        return self

    def validate_inputs(self):
        if not bool(torch.isfinite(self.logits).all()):
            raise ValueError("Plant logits must be finite")
        if not bool(self.action_mask.any(-1).all()):
            raise ValueError("Controller requires at least one legal action")
        return self

    def apply_masking(self, masks):
        self.action_mask = (
            torch.ones(len(self.logits), A.size, dtype=torch.bool, device=self.logits.device)
            if masks is None
            else torch.as_tensor(masks, device=self.logits.device, dtype=torch.bool).reshape(
                -1, A.size
            )
        )
        if self.validate_args:
            self.validate_inputs()
        tile_mask = A.tile_masks(self.action_mask)[:, : A.plant_types]
        self.legal_tile_counts = tile_mask.sum(-1)
        self.available_plants = tile_mask.any(-1)
        self.plantable = self.available_plants.any(-1)
        species_mask = self.available_plants.clone()
        species_mask[:, 0] |= ~self.plantable
        safe_tiles = tile_mask.clone()
        safe_tiles[:, :, 0] |= ~self.available_plants
        self.learned_plants = MaskedCategorical(self.logits[:, : A.plant_types], species_mask)
        self.learned_locations = MaskedCategorical(
            self.logits[:, A.plant_types :].reshape(-1, A.plant_types, A.tiles), safe_tiles
        )
        learned = self.learned_plants.probs[:, :, None] * self.learned_locations.probs
        prior = (
            tile_mask.to(self.logits.dtype)
            / self.legal_tile_counts.clamp_min(1)[:, :, None]
            / self.available_plants.sum(-1).clamp_min(1)[:, None, None]
        )
        self._probs = ((1 - self.epsilon) * learned + self.epsilon * prior) * tile_mask
        floor = torch.finfo(self._probs.dtype).tiny
        species_mass = (1 - self.epsilon) * self.learned_plants.probs + self.epsilon * (
            self.available_plants.to(self.logits.dtype)
            / self.available_plants.sum(-1).clamp_min(1)[:, None]
        )
        self.plants = MaskedCategorical(species_mass.clamp_min(floor).log(), species_mask)
        self.locations = MaskedCategorical(self._probs.clamp_min(floor).log(), safe_tiles)

    @property
    def probs(self):
        return self._probs.flatten(1)

    def log_prob(self, actions):
        kinds, species, tiles = A.unpack(actions.flatten())
        row = torch.arange(len(actions), device=actions.device)
        logs = self.plants.log_prob(species) + self.locations.logits[row, species, tiles]
        return torch.where(kinds == A.plant, logs, torch.zeros_like(logs))

    def entropy_parts(self):
        return (
            self.plants.entropy() * self.plantable,
            (self.plants.probs * self.locations.entropy()).sum(-1) * self.plantable,
        )

    def entropy(self):
        return sum(self.entropy_parts())

    def kl_divergence(self, other):
        if self.validate_args and not torch.equal(self.action_mask, other.action_mask):
            raise ValueError("Exact planting KL requires identical legal masks")
        floor = torch.finfo(self.probs.dtype).tiny
        return (
            (self.probs * (self.probs.clamp_min(floor).log() - other.probs.clamp_min(floor).log()))
            .sum(-1)
            .clamp_min(0)
        )

    def _actions(self, deterministic):
        plants = self.learned_plants if deterministic else self.plants
        locations = self.learned_locations if deterministic else self.locations
        species = (
            plants.probs.argmax(-1)
            if deterministic
            else torch.multinomial(plants.probs, 1).squeeze(-1)
        )
        row = torch.arange(len(species), device=species.device)
        tiles = locations.probs[row, species]
        tile = tiles.argmax(-1) if deterministic else torch.multinomial(tiles, 1).squeeze(-1)
        return 1 + A.tiles * species + tile

    def mode(self):
        return self._actions(True)

    def sample(self):
        return self._actions(False)

    def actions_from_params(self, action_logits, deterministic=False):
        return self.proba_distribution(action_logits).get_actions(deterministic)

    def log_prob_from_params(self, action_logits):
        actions = self.actions_from_params(action_logits)
        return actions, self.log_prob(actions)


def controller_choice(values, masks):
    """Q columns: wait, plant, 45 row-major dig tiles; argmax breaks ties."""
    legal = torch.cat(
        (masks[:, :1], masks[:, 1 : A.dig_start].any(-1, keepdim=True), masks[:, A.dig_start :]), -1
    )
    return values.masked_fill(~legal, -torch.inf).argmax(-1)


def selected_value_indices(actions):
    return torch.where(
        actions == 0, 0, torch.where(actions < A.dig_start, 1, 2 + actions - A.dig_start)
    ).long()
