"""Host-planned public inference buckets with canonical episode identities."""

from dataclasses import dataclass

import numpy as np
import torch

from pvz_rl.envs.actions import ActionSchema as A
from pvz_rl.envs.encoding import EntityBatch, width_bucket


@dataclass(frozen=True)
class ActiveBucket:
    slot_ids: np.ndarray
    compact_indices: np.ndarray
    padding_mask: np.ndarray
    padded_size: int
    entity_width: int


@dataclass(frozen=True)
class ActiveBatchPlan:
    original_slot_ids: np.ndarray
    compact_indices: np.ndarray
    buckets: tuple[ActiveBucket, ...]

    @classmethod
    def from_counts(cls, active, counts, entity_cap, microbatch=None, token_budget=None):
        if microbatch is None or token_budget is None:
            from pvz_rl.config import load_config

            settings = load_config()["policy"]
            microbatch = settings["encoder_microbatch"] if microbatch is None else microbatch
            token_budget = (
                settings["encoder_token_budget"] if token_budget is None else token_budget
            )
        active = np.asarray(active, dtype=bool)
        counts = np.asarray(counts, dtype=np.int64)
        if active.ndim != 1 or counts.shape != active.shape or np.any(counts < 0):
            raise ValueError("Active counts must be nonnegative canonical slot vectors")
        if entity_cap < 1 or microbatch < 1 or token_budget < 1:
            raise ValueError("Inference limits must be positive")
        if np.any(counts[active] > entity_cap):
            raise ValueError("Public inference counts exceed the retained entity cap")
        original = np.flatnonzero(active)
        compact = np.full(len(active), -1, dtype=np.int64)
        compact[original] = np.arange(len(original))
        widths = np.array([width_bucket(int(counts[slot]), entity_cap) for slot in original])
        buckets = []
        for width in np.unique(widths):
            slots = original[widths == width]
            capacity = max(1, min(microbatch, token_budget // (int(width) + 46)))
            capacity = 1 << (capacity.bit_length() - 1)
            for start in range(0, len(slots), capacity):
                selected = slots[start : start + capacity]
                padded = 1 << (len(selected) - 1).bit_length()
                buckets.append(
                    ActiveBucket(
                        selected,
                        compact[selected],
                        np.arange(padded) >= len(selected),
                        padded,
                        int(width),
                    )
                )
        return cls(original, compact, tuple(buckets))

    @property
    def logical_rows(self):
        return len(self.original_slot_ids)

    @property
    def padded_rows(self):
        return sum(bucket.padded_size for bucket in self.buckets)


def gather_padded(value, slots, size, *, dimension=0):
    selected = value.index_select(dimension, slots)
    if selected.shape[dimension] == size:
        return selected
    shape = list(value.shape)
    shape[dimension] = size - len(slots)
    return torch.cat((selected, value.new_zeros(shape)), dimension)


@torch.no_grad()
def forward_active(policy, observations, state, plan, *, events=None, memory_write=None):
    from pvz_rl.policy.transformer_lstm import EventMemoryState, RecurrentOutput

    events = state.inputs() if events is None else events
    memory_write = events[:, :7].ne(0).any(-1) if memory_write is None else memory_write
    count = len(observations)
    wait = state.hidden.new_zeros(count, 1)
    tiles = state.hidden.new_zeros(count, A.tiles, policy.entity.width)
    context = state.hidden.new_zeros(count, policy.hidden_size)
    next_state = state.clone()
    for bucket in plan.buckets:
        slots = torch.as_tensor(bucket.slot_ids, device=observations.device)
        size = bucket.padded_size
        width = min(bucket.entity_width, observations.entities.shape[1])
        batch = EntityBatch(
            gather_padded(observations.entities[:, :width], slots, size),
            gather_padded(observations.entity_mask[:, :width], slots, size),
            gather_padded(observations.globals, slots, size),
            observations.validated,
        )
        if width < bucket.entity_width:
            batch = EntityBatch(
                torch.nn.functional.pad(batch.entities, (0, 0, 0, bucket.entity_width - width)),
                torch.nn.functional.pad(batch.entity_mask, (0, bucket.entity_width - width)),
                batch.globals,
                batch.validated,
            )
        memory = EventMemoryState(
            gather_padded(state.hidden, slots, size, dimension=1),
            gather_padded(state.cell, slots, size, dimension=1),
            gather_padded(state.pending, slots, size),
            gather_padded(state.elapsed_ticks, slots, size),
        )
        output = policy.forward_step(
            batch,
            memory,
            events=gather_padded(events, slots, size),
            memory_write=gather_padded(memory_write, slots, size),
        )
        logical = len(bucket.slot_ids)
        wait.index_copy_(0, slots, output.wait_q[:logical])
        tiles.index_copy_(0, slots, output.tile_features[:logical])
        context.index_copy_(0, slots, output.context[:logical])
        for target, source, dimension in zip(
            next_state.tensors(), output.state.tensors(), (1, 1, 0, 0), strict=True
        ):
            target.index_copy_(dimension, slots, source.narrow(dimension, 0, logical))
    return RecurrentOutput(wait, tiles, context, next_state)
