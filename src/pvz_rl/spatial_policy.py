"""Small resolution-preserving placement policy using only compact_v3 inputs."""

import torch
from pvz_game import Rules
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from .encoding import ObservationEncoder
from .exploration import pooled_spatial
from .grouped_policy import GroupedDistribution


def mlp(input_size, widths):
    layers = []
    for width in widths:
        layers.extend((nn.Linear(input_size, width), nn.ReLU()))
        input_size = width
    return nn.Sequential(*layers)


class SpatialFeatures(BaseFeaturesExtractor):
    def __init__(self, observation_space, layout_cfg):
        layout = ObservationEncoder(layout_cfg, Rules())
        if observation_space.shape != (layout.size,):
            raise ValueError("Spatial policy requires the compact_v3 observation layout")
        spec = layout_cfg["policy"]
        self.channels, self.scalar_channels = spec["channels"][-1], spec["scalar_sizes"][-1]
        super().__init__(
            observation_space, self.scalar_channels + self.channels * layout.rows * layout.cols
        )
        self.layout = layout
        self.plant_types = nn.Embedding(
            len(layout.plants) + 1, spec["plant_embedding"], padding_idx=0
        )
        self.plant_states = nn.Embedding(
            len(layout.plant_states) + 1, spec["state_embedding"], padding_idx=0
        )
        self.global_encoder = mlp(layout.global_width, spec["scalar_sizes"])
        width = spec["plant_embedding"] + spec["state_embedding"] + 2
        width += layout.bins * (layout.zombie_width + 3) + self.scalar_channels + 1
        layers = []
        for channels in spec["channels"]:
            layers.extend((nn.Conv2d(width, channels, 3, padding=1), nn.ReLU()))
            width = channels
        self.board = nn.Sequential(*layers)
        self.register_buffer("columns", torch.linspace(0, 1, layout.cols)[None, None, None, :])

    def forward(self, observations):
        layout, batch = self.layout, len(observations)
        blocks = {k: observations[:, s] for k, s in layout.slices.items()}
        tiles = blocks["plants"].reshape(batch, layout.rows, layout.cols, 4)
        plants = torch.cat(
            (
                self.plant_types(tiles[..., 0].long()),
                tiles[..., 1:3],
                self.plant_states(tiles[..., 3].long()),
            ),
            dim=-1,
        ).permute(0, 3, 1, 2)
        lane = torch.cat(
            [blocks[k].reshape(batch, layout.rows, -1) for k in ("zombies", "projectiles")], dim=-1
        )
        lane = lane.transpose(1, 2).unsqueeze(-1).expand(-1, -1, -1, layout.cols)
        scalars = self.global_encoder(blocks["globals"])
        grid = scalars[:, :, None, None].expand(-1, -1, layout.rows, layout.cols)
        board = self.board(
            torch.cat((plants, lane, grid, self.columns.expand(batch, 1, layout.rows, -1)), 1)
        )
        return torch.cat((scalars, board.flatten(1)), 1)


class SpatialLatents(nn.Module):
    def __init__(self, channels, scalar_channels, hidden_sizes, features_dim):
        super().__init__()
        self.channels, self.scalar_channels = channels, scalar_channels
        self.latent_dim_pi, self.latent_dim_vf = features_dim, hidden_sizes[-1]
        self.critic = mlp(channels * 2 + scalar_channels, hidden_sizes)

    def forward_actor(self, features):
        return features

    def forward_critic(self, features):
        return self.critic(pooled_spatial(features, self.channels, self.scalar_channels)[1])

    def forward(self, features):
        return self.forward_actor(features), self.forward_critic(features)


class SpatialLogits(nn.Module):
    def __init__(self, channels, scalar_channels, hidden_sizes):
        super().__init__()
        self.channels, self.scalar_channels = channels, scalar_channels
        self.type_head = nn.Sequential(
            mlp(channels * 2 + scalar_channels, hidden_sizes), nn.Linear(hidden_sizes[-1], 10)
        )
        # A constant per-map bias cancels in each tile softmax. Omitting it
        # avoids optimizing an unidentifiable parameter on roundoff gradients.
        self.tiles = nn.Conv2d(channels, 9, 1, bias=False)

    def forward(self, features):
        board, pooled = pooled_spatial(features, self.channels, self.scalar_channels)
        return torch.cat((self.type_head(pooled), self.tiles(board).flatten(1)), 1)


class SpatialGroupedPolicy(MaskableActorCriticPolicy):
    def __init__(self, *args, critic_learning_rate=None, **kwargs):
        if kwargs.pop("share_features_extractor", False):
            raise ValueError("Shared encoders require weights-only conversion with --init-from")
        self.critic_learning_rate = critic_learning_rate
        super().__init__(*args, share_features_extractor=False, **kwargs)

    def actor_parameters(self):
        return (*self.pi_features_extractor.parameters(), *self.action_net.parameters())

    def critic_parameters(self):
        return (
            *self.vf_features_extractor.parameters(),
            *self.mlp_extractor.critic.parameters(),
            *self.value_net.parameters(),
        )

    def initialize_dig_logit(self, value):
        with torch.no_grad():
            self.action_net.type_head[-1].bias[self.action_dist.groups - 1] = value

    def _build_mlp_extractor(self):
        self.mlp_extractor = SpatialLatents(
            self.features_extractor.channels,
            self.features_extractor.scalar_channels,
            self.net_arch["vf"],
            self.features_dim,
        ).to(self.device)

    def _build(self, lr_schedule):
        if self.action_space.n != GroupedDistribution.action_dim:
            raise ValueError("Spatial policy requires direct Discrete(406) actions")
        self.action_dist = GroupedDistribution()
        super()._build(lr_schedule)
        self.action_net = SpatialLogits(
            self.features_extractor.channels,
            self.features_extractor.scalar_channels,
            self.net_arch["pi"],
        ).to(self.device)
        if self.ortho_init:
            self.action_net.apply(lambda module: self.init_weights(module, gain=2**0.5))
            self.init_weights(self.action_net.type_head[-1], gain=0.01)
            self.init_weights(self.action_net.tiles, gain=0.01)
        # Include the replacement spatial head, never the discarded linear head.
        self.optimizer = self.optimizer_class(
            self.actor_parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )
        self.critic_optimizer = self.optimizer_class(
            self.critic_parameters(),
            lr=self.critic_learning_rate or lr_schedule(1),
            **self.optimizer_kwargs,
        )

    def sample_actions(self, obs, action_masks, deterministic=False):
        distribution = self.get_distribution(obs, action_masks=action_masks)
        actions = distribution.get_actions(deterministic=deterministic)
        return actions, distribution.log_prob(actions)

    def evaluate_actor(self, obs, actions, action_masks):
        distribution = self.get_distribution(obs, action_masks=action_masks)
        self._record_entropy()
        return distribution.log_prob(actions), distribution.entropy()

    def _record_entropy(self):
        parts = torch.stack(self.action_dist.entropy_parts()).detach().sum(dim=1)
        self._entropy_totals = getattr(self, "_entropy_totals", 0) + parts
        self._entropy_count = getattr(self, "_entropy_count", 0) + len(self.action_dist.logits)

    def evaluate_actions(self, obs, actions, action_masks=None):
        result = super().evaluate_actions(obs, actions, action_masks)
        self._record_entropy()
        return result

    def pop_entropy_metrics(self):
        if not getattr(self, "_entropy_count", 0):
            return {}
        types, tiles = (self._entropy_totals / self._entropy_count).cpu().tolist()
        self._entropy_totals, self._entropy_count = 0, 0
        return {"type_entropy": types, "conditional_tile_entropy": tiles}
