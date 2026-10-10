"""Current-board Q inference with deterministic sparse public-event memory."""

from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch
from pvz_game import Rules
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from pvz_rl.envs.actions import ActionSchema as A
from pvz_rl.envs.encoding import ObservationEncoder, collate_observations
from pvz_rl.envs.history import EVENT_WIDTH, normalize_events
from pvz_rl.policy.entity_attention import EntityTransformer
from pvz_rl.policy.sequential_q import (
    ActionQValues,
    action_parts,
    observation_tile_masks,
    select_q_actions,
)


@dataclass
class EventMemoryState:
    hidden: torch.Tensor
    cell: torch.Tensor
    pending: torch.Tensor
    elapsed_ticks: torch.Tensor

    def detach(self):
        return EventMemoryState(*(value.detach() for value in self.tensors()))

    def clone(self):
        return EventMemoryState(*(value.clone() for value in self.tensors()))

    def tensors(self):
        return self.hidden, self.cell, self.pending, self.elapsed_ticks

    def inputs(self):
        return torch.cat((self.pending, self.elapsed_ticks[:, None]), -1)

    def consume(self, hidden, cell, writes, events=None):
        events = self.inputs() if events is None else events.to(self.pending.dtype)
        return EventMemoryState(
            hidden,
            cell,
            torch.where(writes[:, None], 0, events[:, :7]),
            torch.where(writes, 0, events[:, 7]),
        )

    def observe(self, facts, duration, active):
        return EventMemoryState(
            self.hidden,
            self.cell,
            torch.where(active[:, None], facts.to(self.pending.dtype), self.pending),
            self.elapsed_ticks + torch.where(active, duration.to(self.elapsed_ticks.dtype), 0),
        )


@dataclass
class RecurrentOutput:
    wait_q: torch.Tensor
    tile_features: torch.Tensor
    context: torch.Tensor
    state: EventMemoryState


class EventMemoryRead(torch.autograd.Function):
    @staticmethod
    def forward(ctx, memory, latest):
        ctx.save_for_backward(latest)
        ctx.width = memory.shape[1]
        return memory.gather(1, latest[..., None].expand(-1, -1, memory.shape[-1]))

    @staticmethod
    def backward(ctx, gradient):
        (latest,) = ctx.saved_tensors
        selection = torch.nn.functional.one_hot(latest, ctx.width).to(gradient.dtype)
        return torch.bmm(selection.transpose(1, 2), gradient), None


class TransformerLSTMPolicy(nn.Module):
    protocol = "transformer_lstm_q_v4"

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.fit_precision = "fp32"
        self.layout = ObservationEncoder(cfg, Rules())
        spec = cfg["policy"]
        self.entity = EntityTransformer(self.layout, spec)
        self.scalar = nn.Sequential(
            nn.Linear(self.layout.global_width, spec["scalar_width"]), nn.ReLU()
        )
        self.event = nn.Sequential(nn.Linear(EVENT_WIDTH, spec["event_width"]), nn.ReLU())
        current_width = spec["entity_width"] + spec["scalar_width"]
        hidden = spec["lstm_hidden"]
        self.lstm = nn.LSTM(current_width + spec["event_width"], hidden, batch_first=True)
        self.fusion = nn.Sequential(nn.Linear(current_width + hidden, hidden), nn.ReLU())
        self.wait_head = nn.Sequential(nn.Linear(hidden, 128), nn.ReLU(), nn.Linear(128, 1))
        self.tile_head = nn.Sequential(
            nn.Linear(spec["entity_width"] + hidden + A.tile_groups, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.tile_offsets = nn.Parameter(torch.zeros(A.tile_groups))

    @property
    def hidden_size(self):
        return self.lstm.hidden_size

    def initial_state(self, batch_size, *, device=None, dtype=None):
        parameter = next(self.parameters())
        device = parameter.device if device is None else device
        dtype = parameter.dtype if dtype is None else dtype
        shape = (1, int(batch_size), self.hidden_size)
        return EventMemoryState(
            torch.zeros(shape, device=device, dtype=dtype),
            torch.zeros(shape, device=device, dtype=dtype),
            torch.zeros(batch_size, 7, device=device, dtype=torch.int64),
            torch.zeros(batch_size, device=device, dtype=torch.int64),
        )

    def reset_state(self, state, reset=None):
        if reset is None:
            return self.initial_state(
                state.hidden.shape[1], device=state.hidden.device, dtype=state.hidden.dtype
            )
        reset = reset.to(device=state.hidden.device, dtype=torch.bool)
        return EventMemoryState(
            torch.where(reset[None, :, None], 0, state.hidden),
            torch.where(reset[None, :, None], 0, state.cell),
            torch.where(reset[:, None], 0, state.pending),
            torch.where(reset, 0, state.elapsed_ticks),
        )

    @contextmanager
    def fitting_precision(self, precision):
        previous = self.fit_precision
        self.fit_precision = precision
        try:
            yield
        finally:
            self.fit_precision = previous

    def _inputs(self, observations, events):
        mixed = self.training and torch.is_grad_enabled() and self.fit_precision == "features_bf16"
        with torch.autocast(observations.device.type, dtype=torch.bfloat16, enabled=mixed):
            summary, tiles = self.entity(observations)
            self.compilation_status = self.entity.compilation_status
            self.compilation_backend = self.entity.compilation_backend
            self.compilation_error = self.entity.compilation_error
            scalar = self.scalar(observations.globals.to(self.scalar[0].weight.dtype))
            current = torch.cat((summary, scalar), -1)
            event = self.event(normalize_events(events.to(self.event[0].weight.dtype), self.layout))
            recurrent = torch.cat((current, event), -1)
        dtype = self.lstm.weight_ih_l0.dtype
        return current.to(dtype), recurrent.to(dtype), tiles.to(dtype)

    def forward_step(self, observations, state=None, *, events=None, memory_write=None):
        observations = collate_observations(observations, next(self.parameters()).device)
        if len(observations.shape) != 1:
            raise ValueError("expected a batch of entity observations")
        if state is None:
            state = self.initial_state(len(observations), device=observations.device)
        events = state.inputs() if events is None else events
        writes = events[:, :7].ne(0).any(-1) if memory_write is None else memory_write
        current, inputs, tiles = self._inputs(observations, events)
        _, (hidden, cell) = self.lstm(inputs[:, None], (state.hidden, state.cell))
        hidden = torch.where(writes[None, :, None], hidden, state.hidden)
        cell = torch.where(writes[None, :, None], cell, state.cell)
        context = self.fusion(torch.cat((current, hidden[0]), -1))
        return RecurrentOutput(
            self.wait_head(context), tiles, context, state.consume(hidden, cell, writes, events)
        )

    def forward_sequence(
        self,
        observations,
        state=None,
        *,
        events=None,
        memory_write=None,
        write_plan=None,
        active_plan=None,
        return_context=False,
    ):
        if len(observations.shape) != 2:
            raise ValueError("sequence observations require batch and time dimensions")
        batch, time = observations.shape
        if state is None:
            state = self.initial_state(batch, device=observations.device)
        if events is None:
            events = observations.globals.new_zeros(batch, time, EVENT_WIDTH)
        if memory_write is None:
            memory_write = events[..., :7].ne(0).any(-1)
        plan = np.asarray(
            memory_write.detach().cpu() if write_plan is None else write_plan, dtype=bool
        )
        current, inputs, tiles = self._inputs(
            observations.reshape(-1), events.reshape(-1, EVENT_WIDTH)
        )
        inputs = inputs.reshape(batch, time, -1)
        lengths = plan.sum(-1)
        represented = np.flatnonzero(lengths)
        contexts = state.hidden[0][:, None].expand(-1, time, -1)
        hidden, cell = state.hidden, state.cell
        if len(represented):
            width = int(lengths.max())
            locations = np.zeros((len(represented), width), dtype=np.int64)
            for index, slot in enumerate(represented):
                locations[index, : lengths[slot]] = np.flatnonzero(plan[slot])
            slots = torch.as_tensor(represented, device=inputs.device)
            positions = torch.as_tensor(locations, device=inputs.device)
            packed = pack_padded_sequence(
                inputs[slots[:, None], positions],
                lengths[represented],
                batch_first=True,
                enforce_sorted=False,
            )
            packed_output, (next_hidden, next_cell) = self.lstm(
                packed, (hidden[:, slots], cell[:, slots])
            )
            outputs, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=width)
            latest = torch.as_tensor(np.cumsum(plan[represented], -1), device=inputs.device)
            available = torch.cat((hidden[0, slots, None], outputs), 1)
            selected = EventMemoryRead.apply(available, latest)
            contexts = contexts.index_copy(0, slots, selected)
            hidden = hidden.index_copy(1, slots, next_hidden)
            cell = cell.index_copy(1, slots, next_cell)
        contexts = self.fusion(torch.cat((current.reshape(batch, time, -1), contexts), -1))
        last = (
            np.full(batch, time - 1)
            if active_plan is None
            else np.where(active_plan, np.arange(time), -1).max(-1)
        )
        last_rows = torch.as_tensor(last.clip(min=0), device=events.device)
        slots = torch.arange(batch, device=events.device)
        consumed = state.consume(
            hidden, cell, memory_write[slots, last_rows], events[slots, last_rows]
        )
        live = torch.as_tensor(last >= 0, device=events.device)
        next_state = EventMemoryState(
            hidden,
            cell,
            torch.where(live[:, None], consumed.pending, state.pending),
            torch.where(live, consumed.elapsed_ticks, state.elapsed_ticks),
        )
        result = (self.wait_head(contexts), tiles.reshape(batch, time, A.tiles, -1), next_state)
        return (*result[:2], contexts, result[2]) if return_context else result

    def tile_values(self, tile_features, context, branches):
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

    def selected_values(self, wait, tile_features, context, actions):
        branches, locations = action_parts(actions)
        rows = torch.arange(len(actions), device=actions.device)
        category = torch.nn.functional.one_hot((branches - 1).clamp_min(0), A.tile_groups).to(
            context.dtype
        )
        combined = torch.cat((tile_features[rows, locations], context, category), -1)
        tiles = self.tile_head(combined).flatten() + self.tile_offsets[(branches - 1).clamp_min(0)]
        return torch.where(branches == 0, wait.flatten(), tiles)

    def action_values(self, output, masks=None, *, observations=None, plan=None):
        if masks is None:
            masks = observation_tile_masks(observations)
        count = len(output.wait_q)
        slots = (
            torch.arange(count, device=output.wait_q.device)
            if plan is None
            else torch.as_tensor(plan.original_slot_ids, device=output.wait_q.device)
        )
        tables = output.wait_q.new_zeros(count, A.tile_groups, A.tiles)
        for first in range(0, len(slots), self.entity.microbatch):
            selected = slots[first : first + self.entity.microbatch]
            tiles = output.tile_features[selected]
            context = output.context[selected]
            branches = torch.arange(1, A.tile_groups + 1, device=slots.device)
            tables[selected] = self.tile_values(
                tiles[:, None]
                .expand(-1, A.tile_groups, -1, -1)
                .reshape(-1, A.tiles, tiles.shape[-1]),
                context[:, None].expand(-1, A.tile_groups, -1).reshape(-1, context.shape[-1]),
                branches.expand(len(selected), -1).reshape(-1),
            ).reshape(len(selected), A.tile_groups, A.tiles)
        return ActionQValues.derive(output.wait_q, tables, masks)

    def decide(
        self,
        observations,
        state=None,
        *,
        action_masks=None,
        deterministic=True,
        active=None,
        events=None,
        memory_write=None,
        active_plan=None,
    ):
        if active_plan is None:
            result = self.forward_step(
                observations, state, events=events, memory_write=memory_write
            )
        else:
            from pvz_rl.policy.active_batch import forward_active

            result = forward_active(
                self, observations, state, active_plan, events=events, memory_write=memory_write
            )
        batch = result.wait_q.shape[0]
        if action_masks is None:
            action_masks = observation_tile_masks(
                collate_observations(observations, result.wait_q.device)
            )
        if action_masks.shape != (batch, A.size):
            raise ValueError(f"action_masks must have shape ({batch}, {A.size})")
        actions, first, second, details = select_q_actions(
            self.action_values(result, action_masks, plan=active_plan),
            action_masks,
            deterministic=deterministic,
            exploration_epsilon=getattr(self, "exploration_epsilon", 0.0),
            tile_exploration_epsilon=getattr(self, "tile_exploration_epsilon", 0.0),
            active=active,
        )
        details.update(
            branch_value=first,
            tile_value=second,
            context=result.context,
            tile_features=result.tile_features,
        )
        return actions, result.state, details
