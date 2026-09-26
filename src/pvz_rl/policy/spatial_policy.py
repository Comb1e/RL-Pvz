"""Small resolution-preserving policy using only event_v7 inputs."""

import importlib.util

import torch
from pvz_game import Rules
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from pvz_rl.envs.actions import ActionSchema as A
from pvz_rl.envs.encoding import ObservationEncoder
from pvz_rl.policy.event_memory import MemoryContext
from pvz_rl.policy.sequential_q import (
    action_parts,
    assemble,
    branch_masks,
    explore,
    greedy_choice,
    per_head_epsilon,
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


def pooled_spatial(features, channels, scalar_channels):
    spatial = features[:, scalar_channels:].reshape(-1, channels, A.rows, A.cols)
    pooled = torch.cat(
        (spatial.mean((2, 3)), spatial.amax((2, 3)), features[:, :scalar_channels]), dim=1
    )
    return spatial, pooled


class SequentialQPolicy(BasePolicy):
    """One encoder, a branch Q head and a shared branch-conditioned tile Q head."""

    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        *,
        features_extractor_class=SpatialFeatures,
        features_extractor_kwargs=None,
        hidden_sizes=(128, 128),
        exploration_epsilon=0.0,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
        )
        if action_space.n != A.size:
            raise ValueError("Sequential Q control requires Discrete(406) transport")
        self.features_extractor = self.make_features_extractor()
        encoder = self.features_extractor
        self.exploration_epsilon = exploration_epsilon
        self.compilation_status, self.compilation_error = "disabled", None
        pooled = 2 * encoder.channels + encoder.scalar_channels
        self.branch_head = nn.Sequential(
            mlp(pooled, hidden_sizes), nn.Linear(hidden_sizes[-1], A.tile_groups + 1)
        )
        self.tile_head = nn.Sequential(
            mlp(encoder.channels + pooled + A.tile_groups, hidden_sizes),
            nn.Linear(hidden_sizes[-1], 1),
        )
        self.tile_offsets = nn.Parameter(torch.zeros(A.tile_groups))
        with torch.no_grad():
            for head in (self.branch_head[-1], self.tile_head[-1]):
                head.weight.zero_()
                head.bias.zero_()
            defeat = -encoder.cfg["reward"]["loss_penalty"]
            self.branch_head[-1].bias[-1] = defeat
            self.tile_offsets[-1] = defeat
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr_schedule(1), eps=1e-5)

    def enable_compilation(self):
        if self.device.type != "cuda" or importlib.util.find_spec("triton") is None:
            self.compilation_status = "unavailable"
            self.compilation_error = "CUDA compilation requires an installed Triton backend"
            return
        try:
            self.__dict__["_compiled_encoder"] = torch.compile(
                self.features_extractor, mode="reduce-overhead", dynamic=False
            )
            self.compilation_status = "enabled"
        except Exception as exc:
            self.compilation_status, self.compilation_error = "fallback", repr(exc)

    def encode(self, observations, context=None):
        encoder = self.__dict__.get("_compiled_encoder", self.features_extractor)
        try:
            features = encoder(observations, context)
        except Exception as exc:
            if encoder is self.features_extractor:
                raise
            self.__dict__.pop("_compiled_encoder", None)
            self.compilation_status, self.compilation_error = "fallback", repr(exc)
            features = self.features_extractor(observations, context)
        return pooled_spatial(
            features, self.features_extractor.channels, self.features_extractor.scalar_channels
        )

    def tile_values(self, board, pooled, branches):
        # Non-wait branch IDs 1..9 condition a shared per-tile MLP.
        category = torch.nn.functional.one_hot(branches.long() - 1, A.tile_groups).to(pooled.dtype)
        local = board.flatten(2).transpose(1, 2)
        inputs = torch.cat(
            (
                local,
                pooled[:, None].expand(-1, A.tiles, -1),
                category[:, None].expand(-1, A.tiles, -1),
            ),
            -1,
        )
        return self.tile_head(inputs).squeeze(-1) + self.tile_offsets[branches - 1, None]

    def selected_values(self, obs, actions, context=None):
        board, pooled = self.encode(obs, context)
        branches, tiles = action_parts(actions)
        first = self.branch_head(pooled).gather(1, branches[:, None]).flatten()
        second = first.new_zeros(len(obs))
        ix = (branches != 0).nonzero(as_tuple=True)[0]
        if len(ix):
            second[ix] = (
                self.tile_values(board[ix], pooled[ix], branches[ix])
                .gather(1, tiles[ix, None])
                .flatten()
            )
        return first, second

    def predict_values(self, obs, context=None):
        return self.branch_head(self.encode(obs, context)[1])

    def decide(self, obs, action_masks, deterministic=False, context=None, diagnostic_indices=None):
        board, pooled = self.encode(obs, context)
        values = self.branch_head(pooled)
        legal = branch_masks(action_masks)
        greedy = greedy_choice(values, legal)
        branches = greedy.clone()
        coins = torch.zeros(len(obs), 2, device=obs.device, dtype=torch.bool)
        epsilon = per_head_epsilon(0.0 if deterministic else self.exploration_epsilon)
        planting = ((greedy > 0) & (greedy <= A.plant_types)).nonzero(as_tuple=True)[0]
        if len(planting):
            species, coins[planting, 0] = explore(
                greedy[planting] - 1, legal[planting, 1:-1], epsilon
            )
            branches[planting] = species + 1
        tiles = torch.zeros_like(branches)
        greedy_tiles = torch.zeros_like(branches)
        selected_tile_values = values.new_zeros(len(obs))
        nonwait = (branches > 0).nonzero(as_tuple=True)[0]
        if len(nonwait):
            tile_q = self.tile_values(board[nonwait], pooled[nonwait], branches[nonwait])
            tile_legal = A.tile_masks(action_masks[nonwait])[
                torch.arange(len(nonwait), device=obs.device), branches[nonwait] - 1
            ]
            preferred = greedy_choice(tile_q, tile_legal)
            tiles[nonwait] = preferred
            greedy_tiles[nonwait] = preferred
            plant_rows = (branches[nonwait] <= A.plant_types).nonzero(as_tuple=True)[0]
            if len(plant_rows):
                selected, fired = explore(preferred[plant_rows], tile_legal[plant_rows], epsilon)
                tiles[nonwait[plant_rows]] = selected
                coins[nonwait[plant_rows], 1] = fired
            selected_tile_values[nonwait] = tile_q.gather(1, tiles[nonwait, None]).flatten()
        # Recover the unmodified greedy full command when species exploration switched branches.
        changed = (branches != greedy).nonzero(as_tuple=True)[0]
        if len(changed):
            q = self.tile_values(board[changed], pooled[changed], greedy[changed])
            mask = A.tile_masks(action_masks[changed])[
                torch.arange(len(changed), device=obs.device), greedy[changed] - 1
            ]
            greedy_tiles[changed] = greedy_choice(q, mask)
        actions = assemble(branches, tiles)
        details = dict(greedy_actions=assemble(greedy, greedy_tiles), coins=coins)
        details["branch_q"] = values
        if diagnostic_indices is not None:
            ix = diagnostic_indices
            details["viewer"] = dict(
                q=values[ix],
                legal=legal[ix],
                actions=actions[ix],
                greedy_actions=details["greedy_actions"][ix],
                coins=coins[ix],
            )
        return actions, values.gather(1, branches[:, None]).flatten(), selected_tile_values, details

    def sample_actions(self, obs, action_masks, deterministic=False, context=None):
        return self.decide(obs, action_masks, deterministic, context)[0], None

    def _predict(self, observation, deterministic=False):
        masks = torch.ones(len(observation), A.size, device=observation.device, dtype=torch.bool)
        return self.decide(observation, masks, deterministic)[0]

    def forward(self, obs, deterministic=False, action_masks=None, context=None):
        return self.decide(obs, action_masks, deterministic, context)

    @torch.no_grad()
    def predict(
        self, observation, state=None, episode_start=None, deterministic=False, action_masks=None
    ):
        """Explicit public history for sequential Q playing inference.

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
