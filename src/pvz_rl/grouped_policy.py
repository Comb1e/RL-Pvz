"""Wait/dig/plant, conditional species, then tile; compact engine transport."""

import math

import torch
from sb3_contrib.common.maskable.distributions import MaskableDistribution
from torch import nn
from torch.distributions import Categorical

from .actions import ActionSchema as A


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

    def __init__(self, epsilon=0.0, validate_args=True):
        super().__init__()
        self.epsilon = epsilon
        self.validate_args = validate_args

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
        self.learned_types = FastMaskedCategorical(self.logits[:, : A.kinds], kind_mask)
        self.learned_plants = FastMaskedCategorical(
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
            self.types = FastMaskedCategorical(kind_logits, kind_mask)
            self.plants = FastMaskedCategorical(mixed[:, 1:-1], plant_mask)
        safe_mask = tile_mask.clone()
        safe_mask[:, :, 0] |= ~available
        location_logits = self.logits[:, A.tile_logits_start :].reshape(
            -1, A.tile_groups, A.tiles
        )
        self.locations = FastMaskedCategorical(
            location_logits,
            safe_mask,
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
