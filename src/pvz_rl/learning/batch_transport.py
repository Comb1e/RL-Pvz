"""Flat active-row batches and ordered, owned double-buffer transport."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import torch

from pvz_rl.envs.encoding import (
    ENTITY_WIDTH,
    GLOBAL_WIDTH,
    EntityBatch,
    PackedEntityBatch,
    collate_observations,
)
from pvz_rl.learning.action_routing import InvalidRowCompactor, route_indices, workload_counts


@dataclass(frozen=True)
class BatchMetadata:
    active: torch.Tensor
    action: torch.Tensor
    accepted: torch.Tensor
    target: torch.Tensor
    branch_value: torch.Tensor
    tile_value: torch.Tensor
    placement_gap: torch.Tensor
    route: torch.Tensor | None = None
    plan: dict | None = None
    nonwait_indices: torch.Tensor | None = None
    trusted: bool = False

    def __getitem__(self, name):
        return getattr(self, name)


def fitting_metadata(rows, device, *, plan=None):
    """CPU-owned selections for validated active fitting rows."""
    if not np.all(rows["active"]) or np.any(rows["action"] > 405):
        raise ValueError("Fitting plans require genuine active rows and valid proposals")
    dtypes = (
        torch.bool,
        torch.long,
        torch.bool,
        torch.float32,
        torch.float32,
        torch.float32,
        torch.float32,
    )
    names = (
        "active",
        "action",
        "accepted",
        "target",
        "branch_value",
        "tile_value",
        "placement_gap",
    )
    values = [
        torch.as_tensor(np.array(rows[name]), dtype=dtype, device=device)
        for name, dtype in zip(names, dtypes, strict=True)
    ]
    indices = torch.as_tensor(np.flatnonzero(rows["action"] != 0), dtype=torch.long, device=device)
    plan = fitting_plan(rows) if plan is None else plan
    plan = {k: torch.as_tensor(v, dtype=torch.long, device=device) for k, v in plan.items()}
    return BatchMetadata(
        *values,
        route=torch.as_tensor(np.array(rows["route"]), device=device),
        plan=plan,
        nonwait_indices=indices,
        trusted=True,
    )


def collate_fitting_observations(observations, cfg, device):
    """Shared masked width buckets for list-backed annotation/demo records."""
    batch = collate_observations(observations, "cpu")
    width = batch.entities.shape[1]
    buckets = cfg["training"]["performance"]["fit_entity_buckets"]
    at = int(np.searchsorted(buckets, width))
    if at >= len(buckets):
        raise ValueError("Fitting entities exceed the unchanged encoding cap")
    target = buckets[at]
    if width != target:
        entities = batch.entities.new_zeros(len(batch), target, ENTITY_WIDTH)
        mask = batch.entity_mask.new_zeros(len(batch), target)
        entities[:, :width] = batch.entities
        mask[:, :width] = batch.entity_mask
        batch = EntityBatch(entities, mask, batch.globals, validated=True)
    return batch.to(device)


def fitting_plan(rows):
    plan = route_indices(rows)
    plan["ema_species"] = np.searchsorted(plan["species"], plan["ema"])
    plan["ema_tiles"] = np.searchsorted(plan["tiles"], plan["ema"])
    return plan


class PreparedPlans:
    def plan_for(self, rows):
        if not hasattr(self, "_plans"):
            self._plans = {}
            self._plan_bytes = 0
        key = (int(rows["env"][0]), int(rows["episode"][0]), int(rows["decision"][0]), len(rows))
        if key in self._plans:
            return self._plans[key]
        plan = fitting_plan(rows)
        amount = sum(value.nbytes for value in plan.values())
        if self.buffer.ram_used + self.buffer.staging_bytes + amount <= self.buffer.ram_bytes:
            self.buffer.ram_used += amount
            self._plan_bytes += amount
            self._plans[key] = plan
        return plan

    def clear_plans(self):
        self.buffer.ram_used -= getattr(self, "_plan_bytes", 0)
        self._plan_bytes = 0
        getattr(self, "_plans", {}).clear()


class FittingSpool(PreparedPlans):
    """Reusable active rows and ragged public inputs for every fitting pass."""

    def __init__(self, buffer):
        self.buffer = buffer
        self._ram_arrays, self._temp_paths = [], []
        self._mapped_arrays = []
        self._closed = False
        started = perf_counter()
        try:
            self._build(buffer)
            self.preparation_seconds = perf_counter() - started
        except BaseException:
            self.close()
            raise

    def _build(self, buffer):
        from pvz_rl.config import load_config

        cfg = buffer.cfg or load_config()
        self.buckets = cfg["training"]["performance"]["fit_entity_buckets"]
        compactor = InvalidRowCompactor(cfg["training"]["objective"]["invalid_fit_group_size"])
        self.workloads = dict(
            routes={},
            online_species_rows=0,
            online_tile_rows=0,
            ema_tile_rows=0,
            skipped_ema_rows=0,
        )
        candidates = self._allocate("selection", (buffer.size,), np.dtype("<i8"))
        keys = self._allocate("partition", (buffer.size,), np.dtype("u1"))
        used, entities = 0, 0
        populations = np.zeros(len(self.buckets) * 4, np.int64)
        fields = (
            "active",
            "action",
            "accepted",
            "route",
            "branch_usable",
            "rejection_reason",
            "env",
            "episode",
            "decision",
            "reset",
            "done",
            "run_break",
        )
        slim_dtype = np.dtype([(name, buffer.dtype.fields[name][0]) for name in fields])
        for block_index, block in enumerate(buffer.blocks):
            rows = block[: min(buffer.block_rows, buffer.size - block_index * buffer.block_rows)]
            slim = np.empty(len(rows), dtype=slim_dtype)
            for field in fields:
                slim[field] = rows[field]
            selected = compactor.select(slim)
            workload = workload_counts(slim[selected])
            for key, value in workload.items():
                if key == "routes":
                    for route, count in value.items():
                        self.workloads["routes"][route] = (
                            self.workloads["routes"].get(route, 0) + count
                        )
                else:
                    self.workloads[key] += value
            counts = rows["entity_count"][selected]
            if np.any(counts > self.buckets[-1]):
                raise ValueError("Fitting entities exceed the unchanged encoding cap")
            part = np.searchsorted(self.buckets, counts) * 4 + rows["route"][selected]
            candidates[used : used + len(selected)] = block_index * buffer.block_rows + selected
            keys[used : used + len(selected)] = part
            populations += np.bincount(part, minlength=len(populations))
            entities += int(counts.sum())
            used += len(selected)
        self.compaction, self.outcome_counts = compactor.diagnostics, compactor.counts
        self.indices = self._allocate("indices", (used,), np.dtype("<i8"))
        starts = np.r_[0, np.cumsum(populations)]
        cursor = starts[:-1].copy()
        # Stable counting partition; scratch is bounded by a storage block.
        for at in range(0, used, buffer.block_rows):
            source = candidates[at : min(used, at + buffer.block_rows)]
            partition = keys[at : min(used, at + buffer.block_rows)]
            for key in np.unique(partition):
                selected = source[partition == key]
                begin = int(cursor[key])
                self.indices[begin : begin + len(selected)] = selected
                cursor[key] += len(selected)
        self._release(candidates)
        self._release(keys)
        self.rows = self._allocate("rows", (used,), buffer.dtype)
        for at in range(0, used, buffer.block_rows):
            self.rows[at : at + buffer.block_rows] = buffer.take(
                self.indices[at : at + buffer.block_rows]
            )
        self.entities = self._allocate("entities", (entities, ENTITY_WIDTH), np.dtype("<i4"))
        chunk = max(1, buffer.block_rows // self.buckets[-1])
        destination = 0
        for at in range(0, used, chunk):
            rows = self.rows[at : min(at + chunk, used)]
            counts = rows["entity_count"].astype(np.int64)
            offsets = rows["entity_offset"].astype(np.int64)
            if np.any(offsets + counts > buffer.entity_size):
                raise ValueError("Corrupt fitting entity offset/count")
            local = np.cumsum(counts) - counts
            positions = np.repeat(offsets, counts)
            positions += np.arange(len(positions)) - np.repeat(local, counts)
            blocks, indices = np.divmod(positions, buffer.block_rows)
            for block in np.unique(blocks):
                selected = np.flatnonzero(blocks == block)
                self.entities[destination + selected] = buffer.entity_blocks[block][
                    indices[selected]
                ]
            rows["entity_offset"] = destination + local
            destination += len(positions)
        self.widths = self.rows["entity_count"]
        self.boundaries = np.unique(starts)

    def _release(self, value):
        if isinstance(value, np.memmap):
            path = Path(value.filename)
            value._mmap.close()
            path.unlink(missing_ok=True)
            self._temp_paths.remove(path)
            self._mapped_arrays = [array for array in self._mapped_arrays if array is not value]
        else:
            self.buffer.ram_used -= value.nbytes
            self._ram_arrays = [array for array in self._ram_arrays if array is not value]

    @property
    def consistency_counts(self):
        result = self.outcome_counts[1:].copy()
        result[:, 0] = 0
        return result

    def __len__(self):
        return len(self.rows)

    def _allocate(self, name, shape, dtype):
        amount = int(np.prod(shape, dtype=np.int64)) * np.dtype(dtype).itemsize
        if (
            not amount
            or self.buffer.ram_used + self.buffer.staging_bytes + amount <= self.buffer.ram_bytes
        ):
            value = np.empty(shape, dtype=dtype)
            self.buffer.ram_used += amount
            self._ram_arrays.append(value)
            return value
        path = self.buffer.path / f"fitting-spool-{name}.npy"
        path.unlink(missing_ok=True)
        value = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
        self._temp_paths.append(path)
        self._mapped_arrays.append(value)
        return value

    def iter_batches(self, batch_size):
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("Flat batch_size must be a positive integer")
        for lower, upper in zip(self.boundaries[:-1], self.boundaries[1:], strict=True):
            for start in range(int(lower), int(upper), batch_size):
                stop = min(start + batch_size, int(upper))
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
        self.clear_plans()
        for value in self._ram_arrays:
            self.buffer.ram_used -= value.nbytes
        self._ram_arrays.clear()
        for value in self._mapped_arrays:
            value.flush()
            value._mmap.close()
        self._mapped_arrays.clear()
        for path in self._temp_paths:
            path.unlink(missing_ok=True)
        self._temp_paths.clear()
        self.rows = np.empty(0, dtype=self.buffer.dtype)
        self.entities = np.empty((0, ENTITY_WIDTH), dtype=np.int32)
        self.indices = np.empty(0, np.int64)
        self.widths = np.empty(0, np.uint32)


class PreparedDataSource(PreparedPlans):
    """One-time canonical preparation for list-backed demo/annotation inputs.

    The same ragged storage budget, padding and fenced transport serve all
    fitters. Order is caller-owned; preparation never shuffles or compacts.
    """

    def __init__(self, observations, cfg, *, rows=None):
        from pvz_rl.learning.cuda_buffer import CompleteGameBuffer

        self._workspace = TemporaryDirectory(prefix="pvz-prepared-")
        spec = cfg["training"]["storage"]
        self.buffer = CompleteGameBuffer(
            self._workspace.name,
            1,
            ram_bytes=spec["ram_gib"] * 1024**3,
            block_rows=spec["block_rows"],
            cfg=cfg,
        )
        self.buckets = cfg["training"]["performance"]["fit_entity_buckets"]
        self.original_rows = rows
        try:
            for start in range(0, len(observations), cfg["training"]["batch_size"]):
                current = observations[start : start + cfg["training"]["batch_size"]]
                metadata = np.zeros(len(current), dtype=self.buffer.dtype)
                if rows is not None:
                    metadata[:] = rows[start : start + len(current)]
                else:
                    metadata["accepted"] = True
                metadata["active"] = True
                # This cache has its own physical order. Original identities are
                # restored on read, after caller-owned demo partitioning/shuffling.
                metadata["env"], metadata["episode"] = 0, 0
                self.buffer.append(metadata, current)
        except BaseException:
            self.close()
            raise

    def __len__(self):
        return self.buffer.size

    def batch_ranges(self, batch_size):
        boundaries = getattr(self, "boundaries", [0, len(self)])
        for low, high in zip(boundaries[:-1], boundaries[1:], strict=True):
            for start in range(int(low), int(high), batch_size):
                yield start, min(start + batch_size, int(high))

    def observation_bytes(self, batch_size):
        """Exact padded input footprint, scanned without gathering entity slabs."""
        amount = 0
        for start, stop in self.batch_ranges(batch_size):
            rows = self.buffer.take(np.arange(start, stop))
            width = self.buckets[
                np.searchsorted(self.buckets, int(rows["entity_count"].max(initial=0)))
            ]
            amount += (stop - start) * (int(width) * (ENTITY_WIDTH * 4 + 1) + GLOBAL_WIDTH * 4)
        return amount

    def iter_batches(self, batch_size):
        for start, stop in self.batch_ranges(batch_size):
            rows = self.buffer.take(np.arange(start, stop))
            if self.original_rows is not None:
                for field in ("env", "episode", "decision", "run_break"):
                    rows[field] = self.original_rows[field][start:stop]
            counts = rows["entity_count"].astype(np.int64)
            source_offsets = rows["entity_offset"].astype(np.int64)
            offsets = np.cumsum(counts) - counts
            positions = np.repeat(source_offsets, counts)
            positions += np.arange(len(positions)) - np.repeat(offsets, counts)
            entities = np.empty((len(positions), ENTITY_WIDTH), np.int32)
            blocks, indices = np.divmod(positions, self.buffer.block_rows)
            for block in np.unique(blocks):
                selected = np.flatnonzero(blocks == block)
                entities[selected] = self.buffer.entity_blocks[block][indices[selected]]
            yield rows, PackedEntityBatch(entities, offsets, counts, rows["globals"])

    def batches(self, batch_size, device):
        if (
            torch.device(device).type == "cuda"
            and self.buffer.cfg["training"]["performance"]["prefetch"]
        ):
            return BatchPrefetch(self.buffer, batch_size, device, self.buckets[-1], spool=self)

        def eager():
            for rows, packed in self.iter_batches(batch_size):
                width = self.buckets[
                    np.searchsorted(self.buckets, int(packed.counts.max(initial=0)))
                ]
                yield (
                    rows,
                    self.buffer.observations_from_packed(packed, width=width, device=device),
                    fitting_metadata(rows, device, plan=self.plan_for(rows)),
                )

        return eager()

    def close(self):
        self.clear_plans()
        self.buffer.close()
        self._workspace.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class PreparedDeviceBatches:
    """Owned immutable raw inputs with the source's exact batching and padding.

    Admission is caller-owned across all related sources. No learned features,
    fitting metadata or autograd graphs are retained. Construction is exception
    safe: a failed caller can drop the object and keep using streaming transport.
    """

    @staticmethod
    def required_bytes(source, batch_size, auxiliary):
        return (
            source.observation_bytes(batch_size)
            + len(source) * 8
            + sum(value.numel() * value.element_size() for value in auxiliary.values())
        )

    def __init__(self, source, batch_size, device, auxiliary):
        self.items = []
        self.bytes = self.required_bytes(source, batch_size, auxiliary)
        self.batch_size = batch_size
        with torch.no_grad():
            labels = {key: value.to(device, copy=True) for key, value in auxiliary.items()}
            start = 0
            for rows, packed in source.iter_batches(batch_size):
                stop = start + len(rows)
                width = source.buckets[
                    np.searchsorted(source.buckets, int(packed.counts.max(initial=0)))
                ]
                observations = source.buffer.observations_from_packed(
                    packed, width=width, device=device
                )
                fields = {key: value[start:stop] for key, value in labels.items()}
                fields["rows"] = torch.arange(len(rows), device=device)
                self.items.append((start, stop, observations, fields))
                start = stop

    def close(self):
        self.items.clear()


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
        from pvz_rl.config import load_config

        self.buffer, self.device = buffer, torch.device(device)
        self.buckets = (buffer.cfg or load_config())["training"]["performance"][
            "fit_entity_buckets"
        ]
        self.spool = spool
        self.base_staging = buffer.staging_bytes
        self.batch_size = batch_size
        self.max_entities = max_entities
        amount = 2 * batch_size * ((GLOBAL_WIDTH + 8) * 4 + buffer.dtype.itemsize + 5 * 8)
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
                                0,
                                ENTITY_WIDTH,
                                dtype=torch.int32,
                                pin_memory=True,
                            ),
                            torch.empty(batch_size, 0, dtype=torch.bool, pin_memory=True),
                            torch.empty(batch_size, GLOBAL_WIDTH, pin_memory=True),
                            validated=True,
                        ),
                        torch.empty(batch_size, 8, pin_memory=True),
                        torch.empty(5, batch_size, dtype=torch.long, pin_memory=True),
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
        reuse_started = perf_counter()
        event = self.events[slot]
        if event is not None:
            event.synchronize()
        self.buffer.transport_metrics["producer_wait_seconds"] += perf_counter() - reuse_started
        started = perf_counter()
        item = next(self.source, None)
        if item is None:
            return None
        rows, packed = item if self.spool is not None else (item, None)
        host, metadata, plan = self.slots[slot]
        required = int((rows["entity_count"] if packed is None else packed.counts).max(initial=0))
        buckets = self.buckets
        width = int(buckets[np.searchsorted(buckets, required)])
        if host.entities.shape[1] != width:
            # The prior copy event was fenced above; no borrowed pinned view survives reuse.
            widths = [entry[0].entities.shape[1] for entry in self.slots]
            widths[slot] = width
            amount = sum(
                self.batch_size
                * (
                    w * (ENTITY_WIDTH * 8 + 1 + 3 * 8)
                    + (GLOBAL_WIDTH + 8) * 4
                    + self.buffer.dtype.itemsize
                    + 5 * 8
                )
                for w in widths
            )
            if not self.buffer.reserve_staging(self.base_staging + amount):
                raise MemoryError("Pinned bucket exceeds the shared storage budget")
            host = EntityBatch(
                torch.empty(
                    self.batch_size, width, ENTITY_WIDTH, dtype=torch.int32, pin_memory=True
                ),
                torch.empty(self.batch_size, width, dtype=torch.bool, pin_memory=True),
                host.globals,
                validated=True,
            )
            self.slots[slot] = host, metadata, plan
        if packed is None:
            observations = self.buffer.observations(rows, destination=host)
        else:
            observations = self.buffer.observations_from_packed(
                packed,
                destination=host,
                validated=True,
                width=self.spool.buckets[
                    np.searchsorted(self.spool.buckets, int(packed.counts.max(initial=0)))
                ],
            )
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
                "route",
            )
        ):
            array[:, column] = rows[name]
        self.buffer.transport_metrics["preparation_seconds"] += perf_counter() - started
        selections = self.spool.plan_for(rows) if self.spool is not None else fitting_plan(rows)
        pinned = {}
        for index, (name, indices) in enumerate(selections.items()):
            pinned[name] = plan[index, : len(indices)]
            pinned[name].numpy()[:] = indices
        return rows, observations, fields, pinned

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
        rows, observations, host_fields, host_plan = item
        self.buffer.transport_metrics["transferred_bytes"] += sum(
            value.numel() * value.element_size()
            for value in (*observations.tensors(), host_fields, *host_plan.values())
        )
        with torch.cuda.stream(self.stream):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            observations = observations.to(self.device, non_blocking=True)
            fields = host_fields.to(self.device, non_blocking=True)
            plan = {k: v.to(self.device, non_blocking=True) for k, v in host_plan.items()}
            end.record()
        self.events[slot] = end
        self.timings.append((start, end))
        current = torch.cuda.current_stream(self.device)
        current.wait_event(end)
        for tensor in (*observations.tensors(), fields, *plan.values()):
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
                route=fields[:, 7].long(),
                plan=plan,
                nonwait_indices=plan["tiles"],
                trusted=True,
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
