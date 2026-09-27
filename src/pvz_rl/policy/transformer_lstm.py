"""Entity Transformer + recurrent Q policy used by fresh demonstrations.

The module has a deliberately small public interface: callers carry a
``RecurrentState`` and call :meth:`forward_step` once for every decision.  A
sequence helper uses exactly the same step path, which makes collection,
evaluation and replay agree about waits and zero-time plant/dig operations.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from pvz_game import Rules
from torch import nn

from pvz_rl.envs.actions import ActionSchema as A
from pvz_rl.envs.encoding import ObservationEncoder
from pvz_rl.policy.sequential_q import greedy_choice, observation_tile_masks, selection_masks


@dataclass
class RecurrentState:
    """LSTM state owned by one batch of episodes."""

    hidden: torch.Tensor
    cell: torch.Tensor

    def detach(self) -> "RecurrentState":
        return RecurrentState(self.hidden.detach(), self.cell.detach())

    def clone(self) -> "RecurrentState":
        return RecurrentState(self.hidden.clone(), self.cell.clone())


@dataclass
class RecurrentOutput:
    branch_q: torch.Tensor
    tile_features: torch.Tensor
    context: torch.Tensor
    state: RecurrentState


class EntityTransformer(nn.Module):
    """Encode all 45 tiles and 15 zombie regions, including empty entities."""

    def __init__(self, layout: ObservationEncoder, spec: dict):
        super().__init__()
        width = spec["entity_width"]
        self.layout = layout
        self.width = width
        self.tile_type = nn.Embedding(len(layout.plants) + 1, width)
        self.tile_state = nn.Embedding(len(layout.plant_states) + 1, width)
        self.region_type = nn.Embedding(len(layout.zombies) + 1, width)
        self.tile_position = nn.Embedding(layout.rows * layout.cols, width)
        self.region_position = nn.Embedding(layout.rows * layout.bins, width)
        self.segment = nn.Embedding(2, width)
        self.tile_continuous = nn.Linear(1, width)
        self.region_continuous = nn.Linear(layout.zombie_width, width)
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=spec["transformer_heads"],
            dim_feedforward=spec["transformer_feedforward"],
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=spec["transformer_layers"])
        self.norm = nn.LayerNorm(width)

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b = observations.shape[0]
        tiles = observations[:, self.layout.slices["plants"]].reshape(b, -1, 3)
        regions = observations[:, self.layout.slices["zombies"]].reshape(
            b, self.layout.rows * self.layout.bins, self.layout.zombie_width
        )
        tile_ids = tiles[..., 0].long().clamp(0, self.tile_type.num_embeddings - 1)
        tile_states = tiles[..., 2].long().clamp(0, self.tile_state.num_embeddings - 1)
        tile_pos = torch.arange(tiles.shape[1], device=observations.device)
        region_pos = torch.arange(regions.shape[1], device=observations.device)
        tile = (
            self.tile_type(tile_ids)
            + self.tile_state(tile_states)
            + self.tile_position(tile_pos)[None]
            + self.segment.weight[0]
            + self.tile_continuous(tiles[..., 1:2])
        )
        region = (
            self.region_position(region_pos)[None]
            + self.segment.weight[1]
            + self.region_continuous(regions)
        )
        region_counts = regions[..., : len(self.layout.zombies)]
        region_ids = region_counts.argmax(-1) + 1
        region_ids = torch.where(
            region_counts.sum(-1) > 0, region_ids, torch.zeros_like(region_ids)
        )
        region = region + self.region_type(region_ids)
        entities = self.norm(self.transformer(torch.cat((tile, region), dim=1)))
        return entities[:, : tiles.shape[1]], entities[:, tiles.shape[1] :]


class TransformerLSTMPolicy(nn.Module):
    """Two-level Q controller with a persistent LSTM decision state."""

    protocol = "transformer_lstm_q_v1"

    def __init__(self, cfg: dict, *, action_count: int = A.size):
        super().__init__()
        self.cfg = cfg
        self.layout = ObservationEncoder(cfg, Rules())
        spec = cfg["policy"]
        self.action_count = action_count
        self.entity = EntityTransformer(self.layout, spec)
        self.scalar_width = spec["scalar_width"]
        scalar_input = self.layout.global_width + self.layout.rows + self.layout.cooldown_width
        self.scalar = nn.Sequential(nn.Linear(scalar_input, self.scalar_width), nn.ReLU())
        action_width = spec["action_embedding"]
        self.previous_action = nn.Embedding(action_count, action_width)
        self.outcome = nn.Sequential(nn.Linear(2, spec["outcome_width"]), nn.ReLU())
        self.elapsed = nn.Sequential(nn.Linear(1, 8), nn.Tanh())
        core_input = (
            spec["entity_width"] + self.scalar_width + action_width + spec["outcome_width"] + 8
        )
        self.lstm = nn.LSTM(core_input, spec["lstm_hidden"], num_layers=1, batch_first=True)
        hidden = spec["lstm_hidden"]
        self.branch_head = nn.Sequential(nn.Linear(hidden, 128), nn.ReLU(), nn.Linear(128, 10))
        self.tile_head = nn.Sequential(
            nn.Linear(spec["entity_width"] + hidden + A.tile_groups, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.tile_offsets = nn.Parameter(torch.zeros(A.tile_groups))
        self.register_buffer("zero_action", torch.zeros(1, dtype=torch.long), persistent=False)

    @property
    def hidden_size(self) -> int:
        return self.lstm.hidden_size

    def initial_state(self, batch_size: int, *, device=None, dtype=None) -> RecurrentState:
        parameter = next(self.parameters())
        device = parameter.device if device is None else device
        dtype = parameter.dtype if dtype is None else dtype
        shape = (1, int(batch_size), self.hidden_size)
        return RecurrentState(
            torch.zeros(shape, device=device, dtype=dtype),
            torch.zeros(shape, device=device, dtype=dtype),
        )

    def reset_state(
        self, state: RecurrentState, reset: torch.Tensor | None = None
    ) -> RecurrentState:
        """Clear selected batch entries at episode boundaries."""
        if reset is None:
            return self.initial_state(
                state.hidden.shape[1], device=state.hidden.device, dtype=state.hidden.dtype
            )
        reset = reset.to(device=state.hidden.device, dtype=torch.bool).view(1, -1, 1)
        return RecurrentState(
            torch.where(reset, torch.zeros_like(state.hidden), state.hidden),
            torch.where(reset, torch.zeros_like(state.cell), state.cell),
        )

    def _inputs(
        self,
        observations: torch.Tensor,
        previous_action: torch.Tensor | None,
        execution_outcome: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tiles, regions = self.entity(observations)
        blocks = {name: observations[:, sl] for name, sl in self.layout.slices.items()}
        scalar = self.scalar(
            torch.cat((blocks["globals"], blocks["headless"], blocks["cooldowns"]), -1)
        )
        batch = len(observations)
        if previous_action is None:
            previous_action = torch.zeros(batch, dtype=torch.long, device=observations.device)
        previous_action = previous_action.long().clamp(0, self.action_count - 1)
        if execution_outcome is None:
            execution_outcome = observations.new_zeros(batch, 2)
        if execution_outcome.ndim == 1:
            execution_outcome = execution_outcome[:, None]
        if execution_outcome.shape[-1] != 2:
            raise ValueError("execution_outcome must contain accepted and ticks_advanced")
        elapsed = blocks["globals"][
            :, self.layout.global_fields["elapsed"] : self.layout.global_fields["elapsed"] + 1
        ]
        combined = torch.cat(
            (
                tiles.mean(1),
                scalar,
                self.previous_action(previous_action),
                self.outcome(execution_outcome.to(observations.dtype)),
                self.elapsed(elapsed),
            ),
            -1,
        )
        return combined, tiles

    def forward_step(
        self,
        observations: torch.Tensor,
        state: RecurrentState | None = None,
        *,
        previous_action: torch.Tensor | None = None,
        execution_outcome: torch.Tensor | None = None,
    ) -> RecurrentOutput:
        if observations.ndim == 1:
            observations = observations.unsqueeze(0)
        if observations.ndim != 2 or observations.shape[-1] != self.layout.size:
            raise ValueError(f"expected observations with shape (batch, {self.layout.size})")
        inputs, tiles = self._inputs(observations, previous_action, execution_outcome)
        if state is None:
            state = self.initial_state(
                len(observations), device=observations.device, dtype=inputs.dtype
            )
        output, (hidden, cell) = self.lstm(inputs[:, None], (state.hidden, state.cell))
        context = output[:, 0]
        return RecurrentOutput(
            self.branch_head(context), tiles, context, RecurrentState(hidden, cell)
        )

    def forward_sequence(
        self,
        observations: torch.Tensor,
        state: RecurrentState | None = None,
        *,
        previous_actions: torch.Tensor | None = None,
        execution_outcomes: torch.Tensor | None = None,
        return_context: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, RecurrentState]:
        if observations.ndim != 3:
            raise ValueError("sequence observations must have shape (batch, time, features)")
        outputs, tiles, contexts = [], [], []
        current = state
        for t in range(observations.shape[1]):
            action = None if previous_actions is None else previous_actions[:, t]
            outcome = None if execution_outcomes is None else execution_outcomes[:, t]
            result = self.forward_step(
                observations[:, t], current, previous_action=action, execution_outcome=outcome
            )
            outputs.append(result.branch_q)
            tiles.append(result.tile_features)
            contexts.append(result.context)
            current = result.state
        result = (torch.stack(outputs, 1), torch.stack(tiles, 1), current)
        if return_context:
            return (*result[:2], torch.stack(contexts, 1), result[2])
        return result

    def tile_values(
        self, tile_features: torch.Tensor, context: torch.Tensor, branches: torch.Tensor
    ) -> torch.Tensor:
        if branches.ndim != 1:
            branches = branches.reshape(-1)
        category = torch.nn.functional.one_hot(
            (branches - 1).clamp(0, A.tile_groups - 1), A.tile_groups
        ).to(context.dtype)
        combined = torch.cat(
            (
                tile_features,
                context[:, None].expand(-1, tile_features.shape[1], -1),
                category[:, None].expand(-1, tile_features.shape[1], -1),
            ),
            -1,
        )
        return (
            self.tile_head(combined).squeeze(-1)
            + self.tile_offsets[(branches - 1).clamp(0, A.tile_groups - 1), None]
        )

    def decide(
        self,
        observations: torch.Tensor,
        state: RecurrentState | None = None,
        *,
        action_masks: torch.Tensor | None = None,
        previous_action: torch.Tensor | None = None,
        execution_outcome: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, RecurrentState, dict]:
        """Select one complete command and return the advanced recurrent state."""
        result = self.forward_step(
            observations,
            state,
            previous_action=previous_action,
            execution_outcome=execution_outcome,
        )
        batch = result.branch_q.shape[0]
        if action_masks is None:
            action_masks = observation_tile_masks(observations.reshape(batch, -1))
        if action_masks.shape != (batch, A.size):
            raise ValueError(f"action_masks must have shape ({batch}, {A.size})")
        # Geometry masks constrain only the second-level tile choice. Every
        # active decision still compares wait, all eight species and dig so a
        # rejected proposal remains observable and trainable.
        branches = greedy_choice(result.branch_q, selection_masks(action_masks))
        actions = torch.zeros(batch, dtype=torch.long, device=observations.device)
        tile_q = result.branch_q.new_zeros(batch, A.tiles)
        for branch in range(1, 10):
            ix = (branches == branch).nonzero(as_tuple=True)[0]
            if not len(ix):
                continue
            values = self.tile_values(result.tile_features[ix], result.context[ix], branches[ix])
            tile_q[ix] = values
            if branch == 9:
                legal_tiles = action_masks[ix, A.dig_start :]
                offset = A.dig_start
            else:
                offset = 1 + (branch - 1) * A.tiles
                legal_tiles = action_masks[ix, offset : offset + A.tiles]
            # A full board still needs a well-defined rejected proposal; zero is
            # the deterministic row-major fallback and remains simulator-owned.
            legal_tiles = legal_tiles.clone()
            legal_tiles[~legal_tiles.any(1), 0] = True
            actions[ix] = offset + greedy_choice(values, legal_tiles)
        return (
            actions,
            result.state,
            {
                "branch_q": result.branch_q,
                "branches": branches,
                "tile_q": tile_q,
                "context": result.context,
            },
        )
