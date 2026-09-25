"""Wait/dig/plant, conditional species, then tile; compact engine transport."""

import math

import torch
from sb3_contrib.common.maskable.distributions import MaskableDistribution
from torch import nn
from torch.distributions import Categorical

from .actions import ActionSchema as A

ACTION_DISTRIBUTION = "balanced_species_tiles_v1"


class FastMaskedCategorical(Categorical):
    """Tensor-only categorical distribution for the fixed CUDA hot path.

    PyTorch's default argument and sample validation performs a device-to-host
    truth check on every minibatch. Inputs produced by the policy and masks are
    validated once at the policy boundary; this class preserves the resulting
    probabilities, gradients, entropy and log probabilities without repeating
    those checks.
    """

    def __init__(self, logits, masks):
        self.masks = masks
        self._original_logits = logits
        huge_negative = torch.tensor(-1e8, dtype=logits.dtype, device=logits.device)
        masked = torch.where(masks, logits, huge_negative)
        super().__init__(logits=masked, validate_args=False)

    def log_prob(self, value):
        return self.logits.gather(-1, value.long().unsqueeze(-1)).squeeze(-1)

    def entropy(self):
        p_log_p = self.logits.exp() * self.logits
        return -torch.where(self.masks, p_log_p, torch.zeros_like(p_log_p)).sum(-1)


class GroupedDistribution(MaskableDistribution):
    action_dim, logit_dim = A.size, A.logit_size

    def __init__(self, epsilon=0.0, validate_args=True, wait_weight=1.2):
        super().__init__()
        if not 0 <= epsilon <= 1 or not math.isfinite(wait_weight) or wait_weight <= 0:
            raise ValueError("Exploration epsilon must be in [0, 1] and wait weight positive")
        self.epsilon = epsilon
        self.validate_args = validate_args
        self.wait_weight = wait_weight

    def proba_distribution_net(self, latent_dim):
        return nn.Linear(latent_dim, self.logit_dim)

    def proba_distribution(self, action_logits, masks=None):
        self.logits = action_logits.reshape(-1, self.logit_dim)
        self.apply_masking(masks)
        return self

    def validate_inputs(self):
        """Explicit slow-path validation for diagnostics and boundary checks."""
        if not bool(torch.isfinite(self.logits).all()):
            raise ValueError("Grouped policy logits must be finite")
        if not bool(self.action_mask.any(-1).all()):
            raise ValueError("Grouped policy requires at least one legal action per observation")
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
        self.action_mask = mask
        self.legal_tile_counts = tile_mask.sum(-1)
        available = tile_mask.any(-1)
        self.available_plants = available[:, : A.plant_types]
        plant_available = self.available_plants.any(-1)
        kind_mask = torch.stack((mask[:, 0], available[:, -1], plant_available), -1)
        if self.validate_args and not bool(kind_mask.any(-1).all()):
            raise ValueError("Grouped policy requires at least one legal action per observation")
        # Dummy conditionals have no joint mass and no action-likelihood gradient.
        plant_mask = self.available_plants.clone()
        plant_mask[:, 0] |= ~plant_available
        # A species gets the same initial mass regardless of affordability of others.
        kind_logits = self.logits[:, : A.kinds].clone()
        kind_logits[:, A.plant] += (
            self.available_plants.sum(-1).to(self.logits.dtype).clamp_min(1).log()
        )
        self.learned_types = FastMaskedCategorical(kind_logits, kind_mask)
        self.learned_plants = FastMaskedCategorical(
            logits=self.logits[:, A.kinds : A.tile_logits_start], masks=plant_mask
        )
        safe_mask = tile_mask.clone()
        safe_mask[:, :, 0] |= ~available
        location_logits = self.logits[:, A.tile_logits_start :].reshape(-1, A.tile_groups, A.tiles)
        self.learned_locations = FastMaskedCategorical(
            location_logits,
            safe_mask,
        )
        self.types, self.plants = self.learned_types, self.learned_plants
        self.locations = self.learned_locations
        if self.epsilon:
            # Mix complete actions, including uniform exploratory tiles, then
            # factor the same joint distribution for sampling and PPO/entropy/KL.
            branch = self.branch_probs
            learned_tiles = branch[:, :, None] * self.locations.probs
            prior_branches = available.to(self.logits.dtype)
            prior_branches[:, -1] = 0
            prior_wait = mask[:, 0].to(self.logits.dtype) * self.wait_weight
            total = prior_wait + prior_branches.sum(-1)
            epsilon = (total > 0).to(self.logits.dtype) * self.epsilon
            denominator = total.clamp_min(torch.finfo(total.dtype).tiny)
            prior_tiles = (
                prior_branches[:, :, None]
                * tile_mask
                / self.legal_tile_counts.clamp_min(1)[:, :, None]
                / denominator[:, None, None]
            )
            mixed_tiles = (1 - epsilon[:, None, None]) * learned_tiles + epsilon[
                :, None, None
            ] * prior_tiles
            mixed_wait = (1 - epsilon) * self.types.probs[
                :, A.wait
            ] + epsilon * prior_wait / denominator
            mixed_branch = (1 - epsilon[:, None]) * branch + epsilon[
                :, None
            ] * prior_branches / denominator[:, None]

            def logs(probabilities):
                # Finite zeros keep entropy and the epsilon=1 boundary differentiable.
                return torch.where(
                    probabilities > 0,
                    probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log(),
                    -1e8,
                )

            mixed_plant = (1 - epsilon) * self.types.probs[
                :, A.plant
            ] + epsilon * self.available_plants.sum(-1) / denominator
            kind = torch.stack((mixed_wait, mixed_branch[:, -1], mixed_plant), -1)
            self.types = FastMaskedCategorical(logs(kind), kind_mask)
            self.plants = FastMaskedCategorical(logs(mixed_branch[:, :-1]), plant_mask)
            self.locations = FastMaskedCategorical(logs(mixed_tiles), safe_mask)

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

    def kl_divergence(self, other):
        """Exact KL(self || other) of the masked, mixed hierarchical policy."""
        if not torch.equal(self.action_mask, other.action_mask):
            raise ValueError("Exact hierarchical KL requires identical legal masks")

        def categorical(first, second):
            return torch.where(
                first.probs > 0, first.probs * (first.logits - second.logits), 0
            ).sum(-1)

        return (
            categorical(self.types, other.types)
            + self.types.probs[:, A.plant] * categorical(self.plants, other.plants)
            + (self.branch_probs * categorical(self.locations, other.locations)).sum(-1)
        ).clamp_min(0)

    def _actions(self, deterministic):
        # Deterministic evaluation excludes the injected exploration mixture.
        kinds = self.learned_types.probs.argmax(-1) if deterministic else self.types.sample()
        plants = self.learned_plants.probs.argmax(-1) if deterministic else self.plants.sample()
        groups = torch.where(kinds == A.dig, A.plant_types, plants)
        # Sample only the selected tile distribution, not all nine maps.
        batch = torch.arange(len(kinds), device=kinds.device)
        locations = self.learned_locations if deterministic else self.locations
        probs = locations.probs[batch, groups]
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
