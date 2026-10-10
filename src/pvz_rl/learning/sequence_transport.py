"""Ordered double-buffer preparation with explicit host-buffer ownership."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import torch

from pvz_rl.envs.encoding import ENTITY_WIDTH, GLOBAL_WIDTH, EntityBatch


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
    branch_value: torch.Tensor
    tile_value: torch.Tensor

    def __getitem__(self, name):
        return getattr(self, name)


def sequence_rows(buffer, chunk_length, decision_budget):
    if decision_budget < 1 or decision_budget % chunk_length:
        raise ValueError("Recurrent batch_size must be a positive multiple of chunk_length")
    width = decision_budget // chunk_length
    actual_count = sum(
        np.count_nonzero(
            block["active"][: min(buffer.block_rows, buffer.size - index * buffer.block_rows)]
        )
        for index, block in enumerate(buffer.blocks)
    )
    base_staging = buffer.staging_bytes
    map_size = actual_count + 3 * buffer.n_envs
    temporary = None
    mapping = None
    try:
        if buffer.reserve_staging(base_staging + map_size * np.dtype(np.int64).itemsize):
            mapping = np.zeros(map_size, np.int64)
        else:
            temporary = TemporaryDirectory(prefix="sequence-", dir=buffer.path)
            mapping = np.lib.format.open_memmap(
                Path(temporary.name) / "history.npy", mode="w+", dtype=np.int64, shape=(map_size,)
            )
        counts, offsets, cursors = np.split(mapping[: 3 * buffer.n_envs], 3)
        indices = mapping[3 * buffer.n_envs :]
        for index, block in enumerate(buffer.blocks):
            rows = block[: min(buffer.block_rows, buffer.size - index * buffer.block_rows)]
            counts += np.bincount(rows["env"][rows["active"]], minlength=buffer.n_envs)
        offsets[:] = np.cumsum(counts) - counts
        cursors[:] = offsets
        for index, block in enumerate(buffer.blocks):
            rows = block[: min(buffer.block_rows, buffer.size - index * buffer.block_rows)]
            selected = np.flatnonzero(rows["active"])
            slots = rows["env"][selected]
            per_slot = np.bincount(slots, minlength=buffer.n_envs)
            ordered = selected[np.argsort(slots, kind="stable")] + index * buffer.block_rows
            start = 0
            for slot in np.flatnonzero(per_slot):
                length = per_slot[slot]
                indices[cursors[slot] : cursors[slot] + length] = ordered[start : start + length]
                cursors[slot] += length
                start += length
        for first in range(0, buffer.n_envs, width):
            histories = [
                indices[offsets[slot] : offsets[slot] + counts[slot]]
                for slot in range(first, min(first + width, buffer.n_envs))
            ]
            maximum = max(map(len, histories), default=0)
            for start in range(0, maximum, chunk_length):
                length = min(chunk_length, maximum - start)
                rows = np.zeros((len(histories), length), dtype=buffer.dtype)
                for sequence, history in enumerate(histories):
                    selected = history[start : start + length]
                    rows[sequence, : len(selected)] = buffer.take(selected)
                yield start == 0, rows
    finally:
        if isinstance(mapping, np.memmap):
            mapping._mmap.close()
        if temporary is not None:
            temporary.cleanup()
        buffer.reserve_staging(base_staging)


class SequencePrefetch:
    """One CPU producer, two pinned slots, independent ordered CUDA transfers."""

    def __init__(self, buffer, chunk_length, batch_size, device, max_entities):
        self.buffer, self.device = buffer, torch.device(device)
        self.layout = buffer.layout
        amount = (
            2
            * batch_size
            * (
                max_entities * (ENTITY_WIDTH * 4 + 1)
                + (GLOBAL_WIDTH + self.layout.metadata_width) * 4
            )
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
                    torch.empty(batch_size, self.layout.metadata_width, pin_memory=True),
                )
            )
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pvz-sequences")
        self.futures = [self.worker.submit(self._prepare, i) for i in range(2)]
        self.turn = 0
        self.closed = False
        try:
            self.futures[0].result()
        except BaseException:
            self.close()
            raise

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
        array[:, 13] = flat["branch_value"]
        array[:, 14] = flat["tile_value"]
        array[:, 15:] = np.stack(
            [probes[k] for k in ("valid", "action", "target", "accepted")], -1
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
                *rows.shape, self.layout.metadata_width
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
                probes=fields[..., 15:].reshape(*rows.shape, self.layout.count, 4),
                branch_value=fields[..., 13],
                tile_value=fields[..., 14],
            ),
        )

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.worker.shutdown(wait=True, cancel_futures=True)
        self.source.close()
        self.stream.synchronize()
        self.buffer.transport_metrics["transfer_stream_seconds"] += sum(
            start.elapsed_time(end) / 1000 for start, end in self.timings
        )
        self.slots.clear()
        self.futures.clear()
        self.timings.clear()
        self.buffer.reserve_staging(0)
