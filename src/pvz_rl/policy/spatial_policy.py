"""Small resolution-preserving policy using only event_v7 inputs."""

import importlib.util

import torch
from pvz_game import Rules
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from pvz_rl.envs.actions import ActionSchema as A
from pvz_rl.envs.encoding import ObservationEncoder
from pvz_rl.learning.exploration import pooled_spatial
from pvz_rl.policy.event_memory import MemoryContext
from pvz_rl.policy.grouped_policy import (
    GroupedDistribution,
    controller_choice,
    masked_controller_values,
    selected_value_indices,
)
from pvz_rl.policy.temporal import TemporalEncoder


def mlp(input_size, widths):
    layers = []
    for width in widths:
        layers.extend((nn.Linear(input_size, width), nn.ReLU()))
        input_size = width
    return nn.Sequential(*layers)


class CategoricalEmbedding(nn.Embedding):
    """Small categorical tables with dense gradient accumulation.

    Repeated categories across history make CUDA embedding backward expensive.
    One-hot multiplication has the same learned weights and gradients, without
    sorting millions of repeated indices. Inference keeps the ordinary lookup.
    """

    def forward(self, indices):
        if not torch.is_grad_enabled() or not self.weight.requires_grad:
            return super().forward(indices)
        categories = self.weight.new_zeros((*indices.shape, self.num_embeddings))
        categories.scatter_(-1, indices.unsqueeze(-1), 1)
        if self.padding_idx is not None:
            categories[..., self.padding_idx] = 0
        return categories @ self.weight


class SpatialFeatures(BaseFeaturesExtractor):
    def __init__(self, observation_space, layout_cfg):
        layout = ObservationEncoder(layout_cfg, Rules())
        if observation_space.shape != (layout.size,):
            raise ValueError("Spatial policy requires the event_v7 observation layout")
        spec = layout_cfg["policy"]
        self.channels, self.scalar_channels = spec["channels"][-1], spec["scalar_sizes"][-1]
        super().__init__(
            observation_space, self.scalar_channels + self.channels * layout.rows * layout.cols
        )
        self.layout = layout
        self.plant_types = CategoricalEmbedding(
            len(layout.plants) + 1, spec["plant_embedding"], padding_idx=0
        )
        self.plant_states = CategoricalEmbedding(
            len(layout.plant_states) + 1, spec["state_embedding"], padding_idx=0
        )
        self.global_encoder = mlp(layout.global_width + layout.rows, spec["scalar_sizes"])
        width = spec["plant_embedding"] + spec["state_embedding"] + 1
        width += layout.bins * layout.zombie_width + self.scalar_channels + 1
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
        lane = blocks["zombies"].reshape(batch, layout.rows, -1)
        lane = lane.transpose(1, 2).unsqueeze(-1).expand(-1, -1, -1, layout.cols)
        scalars = self.global_encoder(torch.cat((blocks["globals"], blocks["headless"]), -1))
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
            observations[
                :, self.layout.slices["globals"].start + self.layout.global_fields["elapsed"]
            ]
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
        self.shared_head = mlp(channels * 2 + scalar_channels, hidden_sizes)
        self.plant_head = nn.Linear(hidden_sizes[-1], A.plant_types)
        # A constant per-map bias cancels in each tile softmax. Omitting it
        # avoids optimizing an unidentifiable parameter on roundoff gradients.
        self.tiles = nn.Conv2d(channels, A.plant_types, 1, bias=False)

    def forward(self, features):
        board, pooled = pooled_spatial(features, self.channels, self.scalar_channels)
        hidden = self.shared_head(pooled)
        return torch.cat((self.plant_head(hidden), self.tiles(board).flatten(1)), 1)


class SpatialGroupedPolicy(MaskableActorCriticPolicy):
    def __init__(self, *args, critic_learning_rate=None, exploration_epsilon=0.0, **kwargs):
        if kwargs.pop("share_features_extractor", False):
            raise ValueError("The event Transformer requires independent actor and critic encoders")
        self.critic_learning_rate = critic_learning_rate
        self.exploration_epsilon = exploration_epsilon
        self.compilation_status = "disabled"
        self.compilation_error = None
        super().__init__(*args, share_features_extractor=False, **kwargs)

    def enable_compilation(self):
        """Compile fixed-shape feature paths without changing module parameters."""
        if self.device.type != "cuda" or not hasattr(torch, "compile"):
            self.compilation_status = "unavailable"
            return
        if importlib.util.find_spec("triton") is None:
            # Avoid entering compiler tracing (and its RNG save/restore contexts)
            # beside a sampling collector when this CUDA backend cannot execute.
            self.compilation_status = "unavailable"
            self.compilation_error = "CUDA compilation requires an installed Triton backend"
            return
        try:
            # Keep compiled wrappers out of the module tree so checkpoint keys and
            # optimizer parameter ownership remain identical to eager execution.
            self.__dict__["_compiled_pi"] = torch.compile(
                self.pi_features_extractor, mode="reduce-overhead", dynamic=False
            )
            self.__dict__["_compiled_vf"] = torch.compile(
                self.vf_features_extractor, mode="reduce-overhead", dynamic=False
            )
            self.compilation_status = "enabled"
        except Exception as exc:  # pragma: no cover - depends on local compiler
            self.__dict__.pop("_compiled_pi", None)
            self.__dict__.pop("_compiled_vf", None)
            self.compilation_status = "fallback"
            self.compilation_error = repr(exc)

    def actor_parameters(self):
        return (*self.pi_features_extractor.parameters(), *self.action_net.parameters())

    def critic_parameters(self):
        return (
            *self.vf_features_extractor.parameters(),
            *self.mlp_extractor.critic.parameters(),
            *self.value_net.parameters(),
        )

    def initialize_action_heads(self):
        with torch.no_grad():
            for head in (self.action_net.plant_head, self.action_net.tiles):
                head.weight.zero_()
                if head.bias is not None:
                    head.bias.zero_()
            self.value_net.weight.zero_()
            self.value_net.bias.zero_()
            self.value_net.bias[2:] = -self.features_extractor.cfg["reward"]["loss_penalty"]

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
        # Environment masks are validated at the CUDA environment boundary;
        # avoid a device-to-host reduction on every policy forward.
        self.action_dist = GroupedDistribution(
            self.exploration_epsilon,
            validate_args=False,
        )
        super()._build(lr_schedule)
        self.action_net = SpatialLogits(
            self.features_extractor.channels,
            self.features_extractor.scalar_channels,
            self.net_arch["pi"],
        ).to(self.device)
        if self.ortho_init:
            self.action_net.apply(lambda module: self.init_weights(module, gain=2**0.5))
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 2 + A.tiles).to(self.device)
        self.initialize_action_heads()
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

    def decide(self, obs, action_masks, deterministic=False, context=None, diagnostic_indices=None):
        values = self.predict_values(obs, context)
        choices = controller_choice(values, action_masks)
        actions = torch.where(choices >= 2, A.dig_start + choices - 2, 0).long()
        logs = values.new_zeros(len(obs))
        diagnostics = None
        if diagnostic_indices is not None:
            diagnostics = values.new_zeros((len(obs), A.plant_types, A.tiles))
        planting = (choices == 1).nonzero(as_tuple=True)[0]
        if len(planting):
            distribution = self.get_distribution(
                obs[planting],
                action_masks[planting],
                None if context is None else context.select(planting),
            )
            selected = distribution.get_actions(deterministic=deterministic)
            actions[planting] = selected
            logs[planting] = distribution.log_prob(selected)
            if diagnostics is not None:
                diagnostics[planting] = distribution.probs.reshape(-1, A.plant_types, A.tiles)
        if diagnostics is not None:
            # Reuse sampled-row outputs; extra forwards are restricted to watched
            # non-planting rows. This path never draws random numbers.
            extra = diagnostic_indices[choices[diagnostic_indices] != 1]
            if len(extra):
                dist = self.get_distribution(
                    obs[extra],
                    action_masks[extra],
                    None if context is None else context.select(extra),
                )
                diagnostics[extra] = dist.probs.reshape(-1, A.plant_types, A.tiles)
            ix = diagnostic_indices
            joint = diagnostics[ix]
            plants = joint.sum(-1)
            tiles = joint / plants.clamp_min(torch.finfo(joint.dtype).tiny)[..., None]
            legal_values = masked_controller_values(values[ix], action_masks[ix])
            q = torch.stack(
                (
                    legal_values[:, 0],
                    legal_values[:, 1],
                    legal_values[:, 2:].max(-1).values,
                ),
                -1,
            )
            result = dict(q=q, plants=plants, tiles=tiles, actions=actions[ix])
            return actions, values.gather(1, choices[:, None]).flatten(), logs, result
        return actions, values.gather(1, choices[:, None]).flatten(), logs

    def sample_actions(self, obs, action_masks, deterministic=False, context=None):
        actions, _, logs = self.decide(obs, action_masks, deterministic, context)
        return actions, logs

    @torch.no_grad()
    def predict(
        self, observation, state=None, episode_start=None, deterministic=False, action_masks=None
    ):
        """Explicit public history for actor and critic playing inference.

        Legal masked actions are executed as proposed. Callers that reject an
        action must set state.previous_actions to its executed action instead.
        """
        from pvz_rl.policy.event_memory import EventMemory

        self.set_training_mode(False)
        obs, vectorized = self.obs_to_tensor(observation)
        cfg = self.features_extractor.cfg
        masks = (
            torch.as_tensor(action_masks, device=self.device, dtype=torch.bool)
            if action_masks is not None
            else torch.ones(len(obs), A.size, device=self.device, dtype=torch.bool)
        )
        masks = masks.reshape(len(obs), A.size)
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
            obs[:, layout.slices["globals"].start + layout.global_fields["elapsed"]]
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
        return distribution.log_prob(actions), distribution.entropy()

    def get_distribution(self, obs, action_masks=None, context=None):
        extractor = self.__dict__.get("_compiled_pi", self.pi_features_extractor)
        try:
            features = extractor(obs, context)
        except Exception as exc:  # pragma: no cover - compiler/backend dependent
            if extractor is self.pi_features_extractor:
                raise
            self.__dict__.pop("_compiled_pi", None)
            self.compilation_status = "fallback"
            self.compilation_error = repr(exc)
            features = self.pi_features_extractor(obs, context)
        self.action_dist.proba_distribution(self.action_net(features), masks=action_masks)
        return self.action_dist

    def predict_values(self, obs, context=None):
        extractor = self.__dict__.get("_compiled_vf", self.vf_features_extractor)
        try:
            features = extractor(obs, context)
        except Exception as exc:  # pragma: no cover - compiler/backend dependent
            if extractor is self.vf_features_extractor:
                raise
            self.__dict__.pop("_compiled_vf", None)
            self.compilation_status = "fallback"
            self.compilation_error = repr(exc)
            features = self.vf_features_extractor(obs, context)
        return self.value_net(self.mlp_extractor.forward_critic(features))

    def evaluate_actions(self, obs, actions, action_masks=None, context=None):
        logs, entropy = self.evaluate_actor(obs, actions, action_masks, context)
        indices = selected_value_indices(actions)
        values = self.predict_values(obs, context).gather(1, indices[:, None]).flatten()
        return values, logs, entropy

    def forward(self, obs, deterministic=False, action_masks=None, context=None):
        return self.decide(obs, action_masks, deterministic, context)
