"""Ordered double-buffer preparation with explicit host-buffer ownership."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from time import perf_counter

import numpy as np
import torch

from pvz_rl.envs.encoding import ENTITY_WIDTH, GLOBAL_WIDTH, EntityBatch
from pvz_rl.learning.objective import MAX_PROBES

METADATA_WIDTH = 13 + MAX_PROBES * 5


@dataclass(frozen=True)
class SequenceMetadata:
    """Device metadata for one chronological fitting batch.

    Keeping these tensors together avoids allocating a new dictionary for every
    sequence chunk while ``__getitem__`` preserves the old test and replay API.
    """

    active: torch.Tensor
    action: torch.Tensor
    events: torch.Tensor
    memory_write: torch.Tensor
    accepted: torch.Tensor
    target: torch.Tensor
    probes: torch.Tensor

    def __getitem__(self, name):
        return getattr(self, name)


def sequence_rows(buffer, chunk_length, decision_budget):
    if decision_budget < 1 or decision_budget % chunk_length:
        raise ValueError("Recurrent batch_size must be a positive multiple of chunk_length")
    width = decision_budget // chunk_length
    steps = buffer.size // buffer.n_envs
    time_template = np.arange(chunk_length, dtype=np.int64)
    for first in range(0, buffer.n_envs, width):
        slots = np.arange(first, min(first + width, buffer.n_envs), dtype=np.int64)
        for start in range(0, steps, chunk_length):
            times = start + time_template[: min(chunk_length, steps - start)]
            rows = buffer.take(slots[:, None] + times[None] * buffer.n_envs)
            if not rows["active"].any():
                break
            yield start == 0, rows


class SequencePrefetch:
    """One CPU producer, two pinned slots, independent ordered CUDA transfers."""

    def __init__(self, buffer, chunk_length, batch_size, device, max_entities):
        self.buffer, self.device = buffer, torch.device(device)
        amount = (
            2
            * batch_size
            * (max_entities * (ENTITY_WIDTH * 4 + 1) + (GLOBAL_WIDTH + METADATA_WIDTH) * 4)
        )
        if not buffer.reserve_staging(amount):
            raise MemoryError("Trajectory RAM budget cannot accommodate pinned staging")
        self.source = iter(sequence_rows(buffer, chunk_length, batch_size))
        self.stream = torch.cuda.Stream(device=self.device)
        self.slots, self.events, self.timings = [], [None, None], []
        for _ in range(2):
            self.slots.append(
                (
                    EntityBatch(
                        torch.empty(
                            batch_size,
                            max_entities,
                            ENTITY_WIDTH,
                            dtype=torch.int32,
                            pin_memory=True,
                        ),
                        torch.empty(batch_size, max_entities, dtype=torch.bool, pin_memory=True),
                        torch.empty(batch_size, GLOBAL_WIDTH, pin_memory=True),
                        validated=True,
                    ),
                    torch.empty(batch_size, METADATA_WIDTH, pin_memory=True),
                )
            )
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pvz-sequences")
        self.futures = [self.worker.submit(self._prepare, i) for i in range(2)]
        self.turn = 0
        self.closed = False

    def _prepare(self, slot):
        event = self.events[slot]
        if event is not None:
            event.synchronize()  # D2H/H2D completion must precede host-buffer reuse.
        started = perf_counter()
        item = next(self.source, None)
        if item is None:
            return None
        reset, rows = item
        host, metadata = self.slots[slot]
        obs = self.buffer.observations(rows, destination=host)
        flat = rows.reshape(-1)
        fields = metadata[: len(flat)]
        array = fields.numpy()
        array[:, 0] = flat["active"]
        array[:, 1] = flat["action"]
        array[:, 2:10] = flat["events"]
        array[:, 10] = flat["memory_write"]
        array[:, 11] = flat["accepted"]
        array[:, 12] = flat["target"]
        probes = flat["probes"]
        array[:, 13:] = np.stack(
            [probes[k] for k in ("valid", "branch_role", "action", "target", "accepted")], -1
        ).reshape(len(flat), -1)
        self.buffer.transport_metrics["preparation_seconds"] += perf_counter() - started
        return reset, rows, obs, fields

    def __iter__(self):
        return self

    def __next__(self):
        if self.closed:
            raise StopIteration
        slot = self.turn % 2
        started = perf_counter()
        item = self.futures[slot].result()
        self.buffer.transport_metrics["transfer_wait_seconds"] += perf_counter() - started
        if item is None:
            self.close()
            raise StopIteration
        reset, rows, obs, host_fields = item
        with torch.cuda.stream(self.stream):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            obs = obs.to(self.device, non_blocking=True)
            fields = host_fields.to(self.device, non_blocking=True).reshape(
                *rows.shape, METADATA_WIDTH
            )
            end.record()
        self.events[slot] = end
        self.timings.append((start, end))
        torch.cuda.current_stream(self.device).wait_event(end)
        for tensor in (*obs.tensors(), fields):
            tensor.record_stream(torch.cuda.current_stream(self.device))
        self.turn += 1
        # Prepare the just-consumed slot while its successor is computed.
        # Submit in chronological order: there is exactly one producer.
        self.futures[slot] = self.worker.submit(self._prepare, slot)
        return (
            reset,
            rows,
            obs,
            SequenceMetadata(
                active=fields[..., 0].bool(),
                action=fields[..., 1].long(),
                events=fields[..., 2:10],
                memory_write=fields[..., 10].bool(),
                accepted=fields[..., 11].bool(),
                target=fields[..., 12],
                probes=fields[..., 13:].reshape(*rows.shape, MAX_PROBES, 5),
            ),
        )

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.worker.shutdown(wait=True, cancel_futures=True)
        self.stream.synchronize()
        self.buffer.transport_metrics["transfer_stream_seconds"] += sum(
            start.elapsed_time(end) / 1000 for start, end in self.timings
        )
        self.slots.clear()
        self.futures.clear()
        self.timings.clear()
        self.buffer.reserve_staging(0)
