"""Bounded device-resident lane roots and batched scratch-slot transfers."""

from collections import OrderedDict

import numpy as np

from pvz_rl.envs.cuda_snapshots import pack_owned_snapshots, pack_snapshots
from pvz_rl.envs.snapshot_packet import SnapshotRoot

_COPY = r"""
extern "C" __global__ void copy_roots(const unsigned long long *sources,
    const long long *slots, const long long *widths, long long *destination,
    long long stride, long long n) {
  long long row = blockIdx.x;
  if (row >= n) return;
  const long long *source = reinterpret_cast<const long long *>(sources[row]);
  for (long long i = threadIdx.x; i < widths[row]; i += blockDim.x)
    destination[slots[row] * stride + i] = source[i];
}
"""


class LaneTransfers:
    """Cache immutable packed roots; mutable slots always receive owned copies.

    Tails beyond the engine's entity counts are never read. They need neither
    host construction nor clearing when a scratch slot is reused.
    """

    def __init__(self, cp, limit, *, batched_owned=False):
        self.cp, self.limit = cp, limit
        self.cache, self.nbytes = OrderedDict(), 0
        self.hits = self.misses = self.uploaded_bytes = 0
        self.inflight = []
        self.copy = cp.RawKernel(_COPY, "copy_roots", options=("--std=c++11",))
        self.batched_owned = batched_owned
        self.bundles = OrderedDict()
        self.submits = self.refills = 0

    def fill(self, batch, entries):
        if self.batched_owned:
            return self._fill_owned(batch, entries)
        cp = self.cp
        roots, templates = [], []
        for slot, key, snapshot in entries:
            direct = (
                snapshot.device_slabs(cp.cuda.get_current_stream())
                if isinstance(snapshot, SnapshotRoot)
                else None
            )
            if direct is not None:
                templates.append(direct[0])
                roots.append(direct[1])
                continue
            if isinstance(snapshot, SnapshotRoot):
                snapshot = snapshot.resolve(wait=True)
            cached = self.cache.get(key)
            if cached is None:
                self.misses += 1
                raw, packed = pack_snapshots(batch, [snapshot])
                slabs = {name: cp.asarray(value) for name, value in packed.items()}
                cached = (raw[0], slabs)
                size = sum(v.nbytes for v in slabs.values())
                self.uploaded_bytes += size
                while self.cache and self.nbytes + size > self.limit:
                    _, (_, old) = self.cache.popitem(last=False)
                    self.nbytes -= sum(v.nbytes for v in old.values())
                if size <= self.limit:
                    self.cache[key] = cached
                    self.nbytes += size
            else:
                self.hits += 1
                self.cache.move_to_end(key)
            templates.append(cached[0])
            roots.append(cached[1])
        if not entries:
            return
        slots = cp.asarray([e[0] for e in entries], dtype=cp.int64)
        for name in roots[0]:
            target = getattr(batch, name)
            # One pointer/size upload and one parallel copy kernel per slab,
            # independent of how many roots or candidates share this refill.
            pointers = cp.asarray([r[name].data.ptr for r in roots], dtype=cp.uint64)
            widths = cp.asarray([r[name].size for r in roots], dtype=cp.int64)
            self.copy(
                (len(entries),),
                (128,),
                (pointers, slots, widths, target, np.int64(target[0].size), np.int64(len(entries))),
            )
        batch.facts[slots] = 0
        batch._event_counts[slots] = 0
        if hasattr(batch, "accounting"):
            batch.accounting[slots] = 0
        self.inflight = (roots, [entry[2] for entry in entries])
        for (slot, _, _), template in zip(entries, templates, strict=True):
            batch._templates[slot] = template

    def _fill_owned(self, batch, entries):
        """One upload per slab/refill, not per root; charge whole shared bundles."""
        if not entries:
            return
        cp = self.cp
        self.refills += 1
        missing, roots, templates = {}, [], []
        retained = {key: self.cache[key] for _, key, _ in entries if key in self.cache}
        for _, key, snapshot in entries:
            if key in self.cache:
                self.hits += 1
            elif key not in missing:
                missing[key] = snapshot
                self.misses += 1
        fresh = {}
        if missing:
            keys = list(missing)
            raw, packed = pack_owned_snapshots(batch, list(missing.values()))
            slabs = {name: cp.asarray(value) for name, value in packed.items()}
            self.submits += len(slabs)
            amount = sum(v.nbytes for v in slabs.values())
            self.uploaded_bytes += amount
            for i, key in enumerate(keys):
                fresh[key] = (raw[i], {name: slab[i] for name, slab in slabs.items()})
            # Cache ownership is per allocation bundle. Per-row views cannot
            # claim freed bytes while another row retains their shared base.
            if amount <= self.limit:
                while self.bundles and self.nbytes + amount > self.limit:
                    _, (old_keys, size) = self.bundles.popitem(last=False)
                    for key in old_keys:
                        self.cache.pop(key, None)
                    self.nbytes -= size
                identity = object()
                self.bundles[identity] = (keys, amount)
                self.cache.update(fresh)
                self.nbytes += amount
        for _, key, _ in entries:
            template, slabs = fresh[key] if key in fresh else retained[key]
            templates.append(template)
            roots.append(slabs)
        slots = cp.asarray([entry[0] for entry in entries], dtype=cp.int64)
        for name in roots[0]:
            target = getattr(batch, name)
            pointers = cp.asarray([r[name].data.ptr for r in roots], dtype=cp.uint64)
            widths = cp.asarray([r[name].size for r in roots], dtype=cp.int64)
            self.copy(
                (len(entries),),
                (128,),
                (pointers, slots, widths, target, np.int64(target[0].size), np.int64(len(entries))),
            )
        batch.facts[slots] = 0
        batch._event_counts[slots] = 0
        if hasattr(batch, "accounting"):
            batch.accounting[slots] = 0
        self.inflight = (roots, [entry[2] for entry in entries])
        for (slot, _, _), template in zip(entries, templates, strict=True):
            batch._templates[slot] = template

    def close(self):
        self.inflight = []
        self.cache.clear()
        self.bundles.clear()
        self.nbytes = 0
