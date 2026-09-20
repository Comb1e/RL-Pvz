"""A joint type/tile distribution over the unchanged Discrete(406) controls."""

import torch
from sb3_contrib.common.maskable.distributions import MaskableCategorical, MaskableDistribution
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from torch import nn


class GroupedDistribution(MaskableDistribution):
    # These are the versioned action codec dimensions, not tunable hyperparameters.
    groups, tiles = 10, 45
    action_dim, logit_dim = 406, 415

    def proba_distribution_net(self, latent_dim):
        return nn.Linear(latent_dim, self.logit_dim)

    def proba_distribution(self, action_logits):
        self.logits = action_logits.reshape(-1, self.logit_dim)
        self.apply_masking(None)
        return self

    def apply_masking(self, masks):
        mask = (
            torch.ones(
                (len(self.logits), self.action_dim), dtype=torch.bool, device=self.logits.device
            )
            if masks is None
            else torch.as_tensor(masks, dtype=torch.bool, device=self.logits.device).reshape(
                -1, self.action_dim
            )
        )
        tile_mask = mask[:, 1:].reshape(-1, self.groups - 1, self.tiles)
        available = tile_mask.any(-1)
        group_mask = torch.cat((mask[:, :1], available), dim=1)
        if not group_mask.any(-1).all():
            raise ValueError("Grouped policy requires at least one legal action per observation")
        self.types = MaskableCategorical(logits=self.logits[:, : self.groups], masks=group_mask)
        # Unavailable groups have zero type mass. A dummy tile makes their
        # conditional distribution well-defined without contributing joint mass.
        safe_mask = tile_mask.clone()
        safe_mask[:, :, 0] |= ~available
        self.locations = MaskableCategorical(
            logits=self.logits[:, self.groups :].reshape(-1, self.groups - 1, self.tiles),
            masks=safe_mask,
        )

    @property
    def probs(self):
        tiles = self.types.probs[:, 1:, None] * self.locations.probs
        return torch.cat((self.types.probs[:, :1], tiles.flatten(1)), dim=1)

    def log_prob(self, actions):
        actions = actions.long().flatten()
        groups = torch.where(actions == 0, 0, (actions - 1) // self.tiles + 1)
        tiles = ((actions - 1) % self.tiles).clamp_min(0)
        batch = torch.arange(len(actions), device=actions.device)
        tile_log = self.locations.logits[batch, (groups - 1).clamp_min(0), tiles]
        return self.types.log_prob(groups) + torch.where(groups == 0, 0, tile_log)

    def entropy_parts(self):
        return self.types.entropy(), (self.types.probs[:, 1:] * self.locations.entropy()).sum(-1)

    def entropy(self):
        types, tiles = self.entropy_parts()
        return types + tiles

    def _actions(self, deterministic):
        groups = self.types.probs.argmax(-1) if deterministic else self.types.sample()
        locations = self.locations.probs.argmax(-1) if deterministic else self.locations.sample()
        tiles = locations.gather(1, (groups - 1).clamp_min(0)[:, None]).flatten()
        return torch.where(groups == 0, 0, 1 + (groups - 1) * self.tiles + tiles)

    def mode(self):
        return self._actions(True)

    def sample(self):
        return self._actions(False)

    def actions_from_params(self, action_logits, deterministic=False):
        return self.proba_distribution(action_logits).get_actions(deterministic)

    def log_prob_from_params(self, action_logits):
        actions = self.actions_from_params(action_logits)
        return actions, self.log_prob(actions)


class GroupedPolicy(MaskableActorCriticPolicy):
    def _build(self, lr_schedule):
        if self.action_space.n != GroupedDistribution.action_dim:
            raise ValueError("Grouped policy requires direct Discrete(406) actions")
        self.action_dist = GroupedDistribution()
        super()._build(lr_schedule)

    def evaluate_actions(self, obs, actions, action_masks=None):
        result = super().evaluate_actions(obs, actions, action_masks)
        parts = torch.stack(self.action_dist.entropy_parts()).detach().sum(dim=1)
        self._entropy_totals = getattr(self, "_entropy_totals", 0) + parts
        self._entropy_count = getattr(self, "_entropy_count", 0) + len(actions)
        return result

    def pop_entropy_metrics(self):
        if not getattr(self, "_entropy_count", 0):
            return {}
        types, tiles = (self._entropy_totals / self._entropy_count).cpu().tolist()
        self._entropy_totals, self._entropy_count = 0, 0
        return {"type_entropy": types, "conditional_tile_entropy": tiles}
