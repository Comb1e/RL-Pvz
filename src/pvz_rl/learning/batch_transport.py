"""Flat active-row batches and ordered, owned double-buffer transport."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import torch

from pvz_rl.envs.encoding import ENTITY_WIDTH, GLOBAL_WIDTH, EntityBatch, PackedEntityBatch


@dataclass(frozen=True)
class BatchMetadata:
    active: torch.Tensor
    action: torch.Tensor
    accepted: torch.Tensor
    target: torch.Tensor
    branch_value: torch.Tensor
    tile_value: torch.Tensor
    placement_gap: torch.Tensor

    def __getitem__(self, name):
        return getattr(self, name)


class FittingSpool:
    """Reusable active rows and ragged public inputs for every fitting pass."""

    def __init__(self, buffer):
        self.buffer = buffer
        indices = buffer.valid_indices()
        if len(indices):
            rows = buffer.take(indices)
            indices = indices[np.argsort(rows["entity_count"], kind="stable")]
        self.indices = np.asarray(indices, dtype=np.int64)
        self._ram_arrays = []
        self._temp_paths = []
        self.rows = self._allocate("rows", (len(self.indices),), buffer.dtype)
        if len(self.indices):
            self.rows[:] = buffer.take(self.indices)
        entity_count = int(self.rows["entity_count"].sum())
        self.entities = self._allocate("entities", (entity_count, ENTITY_WIDTH), np.dtype("<i4"))
        cursor = 0
        for row in self.rows:
            count = int(row["entity_count"])
            if count:
                source = int(row["entity_offset"])
                positions = np.arange(source, source + count, dtype=np.int64)
                blocks, offsets = np.divmod(positions, buffer.block_rows)
                for block in np.unique(blocks):
                    selected = blocks == block
                    start = cursor + int(np.flatnonzero(selected)[0])
                    stop = start + int(selected.sum())
                    self.entities[start:stop] = buffer.entity_blocks[block][offsets[selected]]
                row["entity_offset"] = cursor
                cursor += count
            else:
                row["entity_offset"] = cursor
        self.widths = self.rows["entity_count"].copy()
        self._closed = False

    def __len__(self):
        return len(self.rows)

    def _allocate(self, name, shape, dtype):
        amount = int(np.prod(shape, dtype=np.int64)) * np.dtype(dtype).itemsize
        if self.buffer.ram_used + self.buffer.staging_bytes + amount <= self.buffer.ram_bytes:
            value = np.empty(shape, dtype=dtype)
            self.buffer.ram_used += amount
            self._ram_arrays.append(value)
            return value
        path = self.buffer.path / f"fitting-spool-{name}.npy"
        path.unlink(missing_ok=True)
        value = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
        self._temp_paths.append(path)
        return value

    def iter_batches(self, batch_size):
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("Flat batch_size must be a positive integer")
        for start in range(0, len(self), batch_size):
            stop = min(start + batch_size, len(self))
            rows = self.rows[start:stop]
            counts = rows["entity_count"].astype(np.int64, copy=False)
            offsets = np.empty(len(rows), dtype=np.int64)
            if len(rows):
                offsets[0] = 0
                if len(rows) > 1:
                    offsets[1:] = np.cumsum(counts[:-1], dtype=np.int64)
            entity_start = int(rows[0]["entity_offset"]) if len(rows) else 0
            entity_stop = entity_start + int(counts.sum())
            yield (
                rows,
                PackedEntityBatch(
                    np.asarray(self.entities[entity_start:entity_stop]),
                    offsets,
                    counts,
                    np.asarray(rows["globals"]),
                ),
            )

    def close(self):
        if self._closed:
            return
        self._closed = True
        for value in self._ram_arrays:
            self.buffer.ram_used -= value.nbytes
        self._ram_arrays.clear()
        for value in (self.rows, self.entities):
            if isinstance(value, np.memmap):
                value.flush()
                value._mmap.close()
        for path in self._temp_paths:
            path.unlink(missing_ok=True)
        self._temp_paths.clear()
        self.rows = np.empty(0, dtype=self.buffer.dtype)
        self.entities = np.empty((0, ENTITY_WIDTH), dtype=np.int32)
        self.indices = np.empty(0, np.int64)
        self.widths = np.empty(0, np.uint32)
        self.widths = np.empty(0, np.uint32)


def batch_rows(buffer, batch_size, *, spool=None):
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
        if spool is not None:
            for rows, _ in spool.iter_batches(batch_size):
                yield rows
            return
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

    def __init__(self, buffer, batch_size, device, max_entities, *, spool=None):
        self.buffer, self.device = buffer, torch.device(device)
        self.spool = spool
        self.base_staging = buffer.staging_bytes
        amount = (
            2
            * batch_size
            * (
                max_entities * (ENTITY_WIDTH * 4 + 1)
                + (GLOBAL_WIDTH + 7) * 4
                + buffer.dtype.itemsize
            )
        )
        if not buffer.reserve_staging(self.base_staging + amount):
            raise MemoryError("Trajectory RAM budget cannot accommodate pinned staging")
        self.closed = False
        self.worker = None
        self.stream = None
        self.source = (
            iter(spool.iter_batches(batch_size))
            if spool is not None
            else iter(batch_rows(buffer, batch_size))
        )
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
                        torch.empty(batch_size, 7, pin_memory=True),
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
        item = next(self.source, None)
        if item is None:
            return None
        rows, packed = item if self.spool is not None else (item, None)
        host, metadata = self.slots[slot]
        if packed is None:
            observations = self.buffer.observations(rows, destination=host)
        else:
            observations = self.buffer.observations_from_packed(packed, destination=host)
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
