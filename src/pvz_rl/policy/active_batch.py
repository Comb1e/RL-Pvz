"""Host-planned public inference buckets with canonical episode identities."""

from dataclasses import dataclass

import numpy as np
import torch

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
    def from_counts(
        cls, active, counts, entity_cap, microbatch=None, token_budget=None, *, token_overhead=1
    ):
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
        if entity_cap < 1 or microbatch < 1 or token_budget < 1 or token_overhead < 1:
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
            capacity = max(1, min(microbatch, token_budget // (int(width) + token_overhead)))
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

    def for_encoder(self, encoder):
        counts = np.zeros(len(self.compact_indices), dtype=np.int64)
        for bucket in self.buckets:
            counts[bucket.slot_ids] = bucket.entity_width
        return self.from_counts(
            self.compact_indices >= 0,
            counts,
            encoder.layout.max_entities,
            encoder.microbatch,
            encoder.token_budget,
            token_overhead=encoder.token_overhead,
        )


def gather_padded(value, slots, size, *, dimension=0):
    selected = value.index_select(dimension, slots)
    if selected.shape[dimension] == size:
        return selected
    shape = list(value.shape)
    shape[dimension] = size - len(slots)
    return torch.cat((selected, value.new_zeros(shape)), dimension)


def encode_active(encoder, observations, plan, branches=None):
    plan = plan.for_encoder(encoder)
    parameter = next(encoder.parameters())
    result = parameter.new_zeros(len(observations), encoder.token_overhead, encoder.width)
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
        requested = None if branches is None else gather_padded(branches, slots, size)
        output = encoder(batch, requested)
        logical = len(bucket.slot_ids)
        result = result.index_copy(0, slots, output[:logical].to(result.dtype))
    return result
