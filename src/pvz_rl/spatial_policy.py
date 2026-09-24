"""Small resolution-preserving placement policy using only event_v5 inputs."""

import torch
from pvz_game import Rules
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from .encoding import ObservationEncoder
from .event_memory import MemoryContext
from .exploration import pooled_spatial
from .grouped_policy import GroupedDistribution
from .temporal import TemporalEncoder


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
            raise ValueError("Spatial policy requires the event_v5 observation layout")
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
        width = spec["plant_embedding"] + spec["state_embedding"] + 1
        width += layout.bins * (layout.zombie_width + 3) + self.scalar_channels + 1
        layers = []
        for channels in spec["channels"]:
            layers.extend((nn.Conv2d(width, channels, 3, padding=1), nn.ReLU()))
            width = channels
        self.board = nn.Sequential(*layers)
        self.cfg = layout_cfg
        self.temporal = TemporalEncoder(
            layout, spec["memory"], spec["plant_embedding"], spec["state_embedding"]
        )
        self.memory_board = nn.Linear(spec["memory"]["model_width"], self.channels)
        self.memory_scalar = nn.Linear(spec["memory"]["model_width"], self.scalar_channels)
        self.register_buffer("columns", torch.linspace(0, 1, layout.cols)[None, None, None, :])

    def forward(self, observations, context=None):
        layout, batch = self.layout, len(observations)
        blocks = {k: observations[:, s] for k, s in layout.slices.items()}
        tiles = blocks["plants"].reshape(batch, layout.rows, layout.cols, 3)
        plants = torch.cat(
            (
                self.plant_types(tiles[..., 0].long()),
                tiles[..., 1:2],
                self.plant_states(tiles[..., 2].long()),
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
        if context is None:
            context = self.initial_context(observations)
        temporal = self.temporal(context, self.plant_types, self.plant_states)
        board = (board + self.memory_board(temporal)[:, :, None, None]).relu()
        scalars = scalars + self.memory_scalar(temporal)
        return torch.cat((scalars, board.flatten(1)), 1)

    def initial_context(self, observations):
        spec = self.cfg["policy"]["memory"]
        length = sum(spec[k] for k in ("local_tokens", "event_tokens", "summary_tokens"))
        b = len(observations)
        tokens = observations.new_zeros(b, length, self.layout.size + 3)
        valid = torch.zeros(b, length, dtype=torch.bool, device=observations.device)
        ix = spec["local_tokens"] - 1
        tokens[:, ix, : self.layout.size] = observations
        tokens[:, ix, -2] = 1
        ticks = (
            observations[:, self.layout.slices["globals"].start + 1]
            * self.cfg["environment"]["cutoff_seconds"]
            * self.layout.rules.game["tick_rate"]
        )
        tokens[:, ix, -1] = ticks
        valid[:, ix] = True
        return MemoryContext(tokens, valid, valid.float(), tokens[..., -1].clone())


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
    def __init__(self, *args, critic_learning_rate=None, exploration_epsilon=0.0, **kwargs):
        if kwargs.pop("share_features_extractor", False):
            raise ValueError("The event Transformer requires independent actor and critic encoders")
        self.critic_learning_rate = critic_learning_rate
        self.exploration_epsilon = exploration_epsilon
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
        self.action_dist = GroupedDistribution(self.exploration_epsilon)
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
        # SB3 recursively initializes Linear modules. Restore identity-biased gates
        # and zero padding embeddings after that pass.
        for extractor in (self.pi_features_extractor, self.vf_features_extractor):
            for block in extractor.temporal.blocks:
                for gate in (block.attn_gate, block.feed_gate):
                    nn.init.zeros_(gate.weight)
                    nn.init.constant_(gate.bias, -extractor.cfg["policy"]["memory"]["gate_bias"])
            with torch.no_grad():
                extractor.plant_types.weight[0].zero_()
                extractor.plant_states.weight[0].zero_()
        # Include the replacement spatial head, never the discarded linear head.
        self.optimizer = self.optimizer_class(
            self.actor_parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )
        self.critic_optimizer = self.optimizer_class(
            self.critic_parameters(),
            lr=self.critic_learning_rate or lr_schedule(1),
            **self.optimizer_kwargs,
        )

    def sample_actions(self, obs, action_masks, deterministic=False, context=None):
        distribution = self.get_distribution(obs, action_masks=action_masks, context=context)
        actions = distribution.get_actions(deterministic=deterministic)
        return actions, distribution.log_prob(actions)

    @torch.no_grad()
    def predict(
        self, observation, state=None, episode_start=None, deterministic=False, action_masks=None
    ):
        """Explicit episode history for external actor-only inference.

        Legal masked actions are executed as proposed. Callers that reject an
        action must set state.previous_actions to its executed action instead.
        """
        from .event_memory import EventMemory

        self.set_training_mode(False)
        obs, vectorized = self.obs_to_tensor(observation)
        cfg = self.features_extractor.cfg
        masks = (
            torch.as_tensor(action_masks, device=self.device, dtype=torch.bool)
            if action_masks is not None
            else torch.ones(len(obs), 406, device=self.device, dtype=torch.bool)
        )
        masks = masks.reshape(len(obs), 406)
        new = state is None
        if new:
            state = EventMemory(cfg, self.features_extractor.layout.rules, len(obs), self.device)
            state.previous_actions = torch.zeros(len(obs), device=self.device)
        resets = (
            torch.full((len(obs),), new, device=self.device, dtype=torch.bool)
            if episode_start is None
            else torch.as_tensor(episode_start, device=self.device).reshape(-1)
        )
        layout = self.features_extractor.layout
        ticks = (
            obs[:, layout.slices["globals"].start + 1]
            * cfg["environment"]["cutoff_seconds"]
            * layout.rules.game["tick_rate"]
        ).round()
        context = state.observe(
            obs, masks, state.previous_actions * (~resets.bool()), resets, ticks
        )
        actions, _ = self.sample_actions(obs, masks, deterministic, context)
        state.previous_actions = actions
        actions = actions.cpu().numpy()
        return (actions if vectorized else actions.squeeze(0)), state

    def evaluate_actor(self, obs, actions, action_masks, context=None):
        distribution = self.get_distribution(obs, action_masks=action_masks, context=context)
        self._record_entropy()
        return distribution.log_prob(actions), distribution.entropy()

    def _record_entropy(self):
        parts = torch.stack(self.action_dist.entropy_parts()).detach().sum(dim=1)
        self._entropy_totals = getattr(self, "_entropy_totals", 0) + parts
        self._entropy_count = getattr(self, "_entropy_count", 0) + len(self.action_dist.logits)

    def get_distribution(self, obs, action_masks=None, context=None):
        features = self.pi_features_extractor(obs, context)
        self.action_dist.proba_distribution(self.action_net(features))
        self.action_dist.apply_masking(action_masks)
        return self.action_dist

    def predict_values(self, obs, context=None):
        features = self.vf_features_extractor(obs, context)
        return self.value_net(self.mlp_extractor.forward_critic(features))

    def evaluate_actions(self, obs, actions, action_masks=None, context=None):
        logs, entropy = self.evaluate_actor(obs, actions, action_masks, context)
        return self.predict_values(obs, context), logs, entropy

    def forward(self, obs, deterministic=False, action_masks=None, context=None):
        actions, logs = self.sample_actions(obs, action_masks, deterministic, context)
        return actions, self.predict_values(obs, context), logs

    def pop_entropy_metrics(self):
        if not getattr(self, "_entropy_count", 0):
            return {}
        types, tiles = (self._entropy_totals / self._entropy_count).cpu().tolist()
        self._entropy_totals, self._entropy_count = 0, 0
        return {"type_entropy": types, "conditional_tile_entropy": tiles}
