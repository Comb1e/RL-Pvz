"""Small resolution-preserving placement policy using only tactical_v2 inputs."""

import torch
from pvz_game import Rules
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from .encoding import ObservationEncoder
from .exploration import pooled_spatial
from .grouped_policy import GroupedPolicy


def mlp(input_size, widths):
    layers = []
    for width in widths:
        layers.extend((nn.Linear(input_size, width), nn.ReLU()))
        input_size = width
    return nn.Sequential(*layers)


class SpatialFeatures(BaseFeaturesExtractor):
    def __init__(self, observation_space, layout_cfg, channels=64):
        layout = ObservationEncoder(layout_cfg, Rules())
        if not layout.tactical or observation_space.shape != (layout.size,):
            raise ValueError("spatial_grouped_v2 requires the tactical_v2 observation layout")
        super().__init__(observation_space, channels * (1 + layout.rows * layout.cols))
        self.layout, self.channels = layout, channels
        self.lane = mlp(layout.bins * (layout.zombie_width + 3) + 4, [channels])
        self.global_encoder = mlp(layout.global_width + 2 * len(layout.plants), [channels])
        self.board = nn.Sequential(
            nn.Conv2d(layout.plant_width + 2 * channels + 1, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
        )
        self.register_buffer("columns", torch.linspace(0, 1, layout.cols)[None, None, None, :])

    def forward(self, observations):
        layout, batch = self.layout, len(observations)
        blocks = {k: observations[:, s] for k, s in layout.slices.items()}
        lane = (
            self.lane(
                torch.cat(
                    [
                        blocks[k].reshape(batch, layout.rows, -1)
                        for k in ("zombies", "projectiles", "lanes")
                    ],
                    dim=-1,
                )
            )
            .transpose(1, 2)
            .unsqueeze(-1)
            .expand(-1, -1, -1, layout.cols)
        )
        global_features = self.global_encoder(torch.cat((blocks["globals"], blocks["economy"]), 1))
        global_grid = global_features[:, :, None, None].expand(-1, -1, layout.rows, layout.cols)
        plants = blocks["plants"].reshape(batch, layout.rows, layout.cols, -1).permute(0, 3, 1, 2)
        board = self.board(
            torch.cat(
                (plants, lane, global_grid, self.columns.expand(batch, 1, layout.rows, -1)), 1
            )
        )
        return torch.cat((global_features, board.flatten(1)), 1)


class SpatialLatents(nn.Module):
    def __init__(self, channels, hidden_sizes, features_dim):
        super().__init__()
        self.channels = channels
        self.latent_dim_pi, self.latent_dim_vf = features_dim, hidden_sizes[-1]
        self.critic = mlp(channels * 3, hidden_sizes)

    def forward_actor(self, features):
        return features

    def forward_critic(self, features):
        return self.critic(pooled_spatial(features, self.channels)[1])

    def forward(self, features):
        return self.forward_actor(features), self.forward_critic(features)


class SpatialLogits(nn.Module):
    def __init__(self, channels, hidden_sizes):
        super().__init__()
        self.channels = channels
        self.type_head = nn.Sequential(
            mlp(channels * 3, hidden_sizes), nn.Linear(hidden_sizes[-1], 10)
        )
        self.tiles = nn.Conv2d(channels, 9, 1)

    def forward(self, features):
        board, pooled = pooled_spatial(features, self.channels)
        return torch.cat((self.type_head(pooled), self.tiles(board).flatten(1)), 1)


class SpatialGroupedPolicy(GroupedPolicy):
    def _build_mlp_extractor(self):
        self.mlp_extractor = SpatialLatents(
            self.features_extractor.channels, self.net_arch["vf"], self.features_dim
        ).to(self.device)

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        self.action_net = SpatialLogits(self.features_extractor.channels, self.net_arch["pi"]).to(
            self.device
        )
        if self.ortho_init:
            self.action_net.apply(lambda module: self.init_weights(module, gain=2**0.5))
            self.init_weights(self.action_net.type_head[-1], gain=0.01)
            self.init_weights(self.action_net.tiles, gain=0.01)
        # Include the replacement spatial head, never the discarded linear head.
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )
