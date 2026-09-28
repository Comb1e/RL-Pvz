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
from pvz_rl.envs.encoding import ObservationEncoder, collate_observations
from pvz_rl.policy.entity_attention import EntityTransformer
from pvz_rl.policy.sequential_q import observation_tile_masks, select_q_actions


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


class TransformerLSTMPolicy(nn.Module):
    """Two-level Q controller with a persistent LSTM decision state."""

    protocol = "transformer_lstm_q_v2"

    def __init__(self, cfg: dict, *, action_count: int = A.size):
        super().__init__()
        self.cfg = cfg
        self.layout = ObservationEncoder(cfg, Rules())
        spec = cfg["policy"]
        self.action_count = action_count
        self.entity = EntityTransformer(self.layout, spec)
        self.scalar_width = spec["scalar_width"]
        scalar_input = self.layout.global_width
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
        summary, tiles = self.entity(observations)
        scalar = self.scalar(observations.globals.to(self.scalar[0].weight.dtype))
        batch = len(observations)
        if previous_action is None:
            previous_action = torch.zeros(batch, dtype=torch.long, device=observations.device)
        previous_action = previous_action.long().clamp(0, self.action_count - 1)
        if execution_outcome is None:
            execution_outcome = observations.globals.new_zeros(batch, 2)
        if execution_outcome.ndim == 1:
            execution_outcome = execution_outcome[:, None]
        if execution_outcome.shape[-1] != 2:
            raise ValueError("execution_outcome must contain accepted and ticks_advanced")
        elapsed = observations.globals[:, 1:2]
        combined = torch.cat(
            (
                summary,
                scalar,
                self.previous_action(previous_action),
                self.outcome(execution_outcome.to(scalar.dtype)),
                self.elapsed(elapsed.to(scalar.dtype)),
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
        observations = collate_observations(observations, next(self.parameters()).device)
        if len(observations.shape) != 1:
            raise ValueError("expected a batch of entity observations")
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
        if len(observations.shape) != 2:
            raise ValueError("sequence observations require batch and time dimensions")
        batch, time = observations.shape
        inputs, tiles = self._inputs(
            observations.reshape(-1),
            None if previous_actions is None else previous_actions.reshape(-1),
            None if execution_outcomes is None else execution_outcomes.reshape(-1, 2),
        )
        if state is None:
            state = self.initial_state(batch, device=inputs.device, dtype=inputs.dtype)
        contexts, (hidden, cell) = self.lstm(
            inputs.reshape(batch, time, -1), (state.hidden, state.cell)
        )
        result = (
            self.branch_head(contexts),
            tiles.reshape(batch, time, A.tiles, -1),
            RecurrentState(hidden, cell),
        )
        if return_context:
            return (*result[:2], contexts, result[2])
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
        deterministic: bool = True,
        active: torch.Tensor | None = None,
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
            action_masks = observation_tile_masks(
                collate_observations(observations, result.branch_q.device)
            )
        if action_masks.shape != (batch, A.size):
            raise ValueError(f"action_masks must have shape ({batch}, {A.size})")
        actions, first, second, details = select_q_actions(
            result.branch_q,
            result.tile_features,
            result.context,
            self.tile_values,
            action_masks,
            deterministic=deterministic,
            exploration_epsilon=getattr(self, "exploration_epsilon", 0.0),
            tile_exploration_epsilon=getattr(self, "tile_exploration_epsilon", 0.0),
            active=active,
        )
        details.update(branch_value=first, tile_value=second, context=result.context)
        return actions, result.state, details
