"""Flat active-row batches and ordered, owned double-buffer transport."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import torch

from pvz_rl.envs.encoding import ENTITY_WIDTH, GLOBAL_WIDTH, EntityBatch


@dataclass(frozen=True)
class BatchMetadata:
    active: torch.Tensor
    action: torch.Tensor
    accepted: torch.Tensor
    target: torch.Tensor
    branch_value: torch.Tensor
    tile_value: torch.Tensor
    placement_gap: torch.Tensor
    probes: torch.Tensor

    def __getitem__(self, name):
        return getattr(self, name)


def batch_rows(buffer, batch_size):
    """Yield every genuine decision once, without slot gaps or recurrent padding."""
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("Flat batch_size must be a positive integer")
    base_staging = buffer.staging_bytes
    temporary = None
    pending = None
    try:
        if buffer.reserve_staging(base_staging + 2 * batch_size * np.dtype(np.int64).itemsize):
            pending = np.empty(batch_size, np.int64)
        else:
            temporary = TemporaryDirectory(prefix="flat-batches-", dir=buffer.path)
            pending = np.lib.format.open_memmap(
                Path(temporary.name) / "indices.npy", mode="w+", dtype=np.int64, shape=(batch_size,)
            )
        used = 0
        for block_index, block in enumerate(buffer.blocks):
            size = min(buffer.block_rows, buffer.size - block_index * buffer.block_rows)
            for start in range(0, size, batch_size):
                selected = np.flatnonzero(block["active"][start : min(start + batch_size, size)])
                selected += block_index * buffer.block_rows + start
                position = 0
                while position < len(selected):
                    amount = min(batch_size - used, len(selected) - position)
                    pending[used : used + amount] = selected[position : position + amount]
                    used += amount
                    position += amount
                    if used == batch_size:
                        yield buffer.take(pending)
                        used = 0
        if used:
            yield buffer.take(pending[:used])
    finally:
        if isinstance(pending, np.memmap):
            pending._mmap.close()
        if temporary is not None:
            temporary.cleanup()
        buffer.reserve_staging(base_staging)


class BatchPrefetch:
    """One ordered producer and two pinned slots with explicit reuse fences."""

    def __init__(self, buffer, batch_size, device, max_entities):
        self.buffer, self.device = buffer, torch.device(device)
        self.layout = buffer.layout
        self.base_staging = buffer.staging_bytes
        amount = (
            2
            * batch_size
            * (
                max_entities * (ENTITY_WIDTH * 4 + 1)
                + (GLOBAL_WIDTH + self.layout.metadata_width) * 4
                + buffer.dtype.itemsize
            )
        )
        if not buffer.reserve_staging(self.base_staging + amount):
            raise MemoryError("Trajectory RAM budget cannot accommodate pinned staging")
        self.closed = False
        self.worker = None
        self.stream = None
        self.source = iter(batch_rows(buffer, batch_size))
        self.slots, self.events, self.timings, self.futures = [], [None, None], [], []
        try:
            self.stream = torch.cuda.Stream(device=self.device)
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
                            torch.empty(
                                batch_size, max_entities, dtype=torch.bool, pin_memory=True
                            ),
                            torch.empty(batch_size, GLOBAL_WIDTH, pin_memory=True),
                            validated=True,
                        ),
                        torch.empty(batch_size, self.layout.metadata_width, pin_memory=True),
                    )
                )
            self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pvz-batches")
            self.futures = [self.worker.submit(self._prepare, slot) for slot in range(2)]
            self.turn = 0
            self.futures[0].result()
        except BaseException:
            self.close()
            raise

    def _prepare(self, slot):
        event = self.events[slot]
        if event is not None:
            event.synchronize()
        started = perf_counter()
        rows = next(self.source, None)
        if rows is None:
            return None
        host, metadata = self.slots[slot]
        observations = self.buffer.observations(rows, destination=host)
        fields = metadata[: len(rows)]
        array = fields.numpy()
        for column, name in enumerate(
            (
                "active",
                "action",
                "accepted",
                "target",
                "branch_value",
                "tile_value",
                "placement_gap",
            )
        ):
            array[:, column] = rows[name]
        probes = rows["probes"]
        array[:, 7:] = np.stack(
            [probes[name] for name in ("valid", "action", "target", "accepted")], -1
        ).reshape(len(rows), -1)
        self.buffer.transport_metrics["preparation_seconds"] += perf_counter() - started
        return rows, observations, fields

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
        rows, observations, host_fields = item
        with torch.cuda.stream(self.stream):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            observations = observations.to(self.device, non_blocking=True)
            fields = host_fields.to(self.device, non_blocking=True)
            end.record()
        self.events[slot] = end
        self.timings.append((start, end))
        current = torch.cuda.current_stream(self.device)
        current.wait_event(end)
        for tensor in (*observations.tensors(), fields):
            tensor.record_stream(current)
        self.turn += 1
        self.futures[slot] = self.worker.submit(self._prepare, slot)
        return (
            rows,
            observations,
            BatchMetadata(
                active=fields[:, 0].bool(),
                action=fields[:, 1].long(),
                accepted=fields[:, 2].bool(),
                target=fields[:, 3],
                branch_value=fields[:, 4],
                tile_value=fields[:, 5],
                placement_gap=fields[:, 6],
                probes=fields[:, 7:].reshape(len(rows), self.layout.count, 4),
            ),
        )

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.worker is not None:
            self.worker.shutdown(wait=True, cancel_futures=True)
        self.source.close()
        if self.stream is not None:
            self.stream.synchronize()
        self.buffer.transport_metrics["transfer_stream_seconds"] += sum(
            start.elapsed_time(end) / 1000 for start, end in self.timings
        )
        self.slots.clear()
        self.futures.clear()
        self.timings.clear()
        self.buffer.reserve_staging(self.base_staging)
