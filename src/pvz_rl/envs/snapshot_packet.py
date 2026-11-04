"""Exact occupied-prefix snapshots with asynchronous, owned host transport."""

from copy import copy

import numpy as np
from pvz_game.cuda import schema as s

from pvz_rl.envs.cuda_snapshots import unpack_snapshots

PACK = r"""
extern "C" __global__ void pack_physical(const long long *h, const long long *p,
 const long long *z, const long long *q, const long long *m, const long long *cd,
 const long long *slots, const long long *offsets, long long *out, long long n,
 long long hw, long long pw, long long zw, long long qw, long long mw,
 long long zcap, long long qcap, long long np_at, long long nz_at, long long nq_at,
 long long hs, long long ps, long long zs, long long qs, long long ms, long long cs) {
 long long r = blockIdx.x; if (r >= n) return;
 long long slot = slots[r]; const long long *header = h + slot * hs;
 const long long *parts[] = {header, p + slot*ps, z + slot*zs,
    q + slot*qs, m + slot*ms, cd + slot*cs};
 long long live[] = {hw, header[np_at]*pw, header[nz_at]*zw, header[nq_at]*qw, 5*mw, 8};
 long long offset = offsets[r];
 for (int part=0; part<6; part++) {
   for (long long k=threadIdx.x; k<live[part]; k+=blockDim.x) out[offset+k]=parts[part][k];
   offset += live[part];
 }
}
"""


class SnapshotPacket:
    """Freeze before mutation; decoding and hashing happen on the coordinator."""

    def __init__(self, batch, indices, headers, *, stream, pool=None, device_only=False):
        self.headers = np.array(headers, dtype=np.int64, copy=True)
        self.templates = [copy(batch._templates[i]) for i in indices]
        self.indices = list(indices)
        self.decoded = None
        self.cp, self.stream = batch.cp, stream
        self.device_only = device_only
        self.event = None
        self.counts = [self.headers[:, s.HEADER.index(k)] for k in ("np", "nz", "nq")]
        lengths = (
            len(s.HEADER)
            + self.counts[0] * len(s.PLANT)
            + self.counts[1] * len(s.ZOMBIE)
            + self.counts[2] * len(s.PROJECTILE)
            + 5 * len(s.MOWER)
            + 8
        )
        self.offsets = np.r_[0, np.cumsum(lengths)].astype(np.int64)
        cp = batch.cp
        self.device = cp.empty(int(self.offsets[-1]), cp.int64)
        kernel = getattr(batch, "_snapshot_pack", None)
        if kernel is None:
            kernel = batch._snapshot_pack = cp.RawKernel(PACK, "pack_physical")
        slots, offsets = cp.asarray(indices, cp.int64), cp.asarray(self.offsets[:-1])
        kernel(
            (len(indices),),
            (128,),
            (
                *[
                    getattr(batch, name)
                    for name in (
                        "header",
                        "plants",
                        "zombies",
                        "projectiles",
                        "mowers",
                        "cooldowns",
                    )
                ],
                slots,
                offsets,
                self.device,
                len(indices),
                len(s.HEADER),
                len(s.PLANT),
                len(s.ZOMBIE),
                len(s.PROJECTILE),
                len(s.MOWER),
                batch.zcap,
                batch.qcap,
                s.HEADER.index("np"),
                s.HEADER.index("nz"),
                s.HEADER.index("nq"),
                *[
                    getattr(batch, name).strides[0] // 8
                    for name in (
                        "header",
                        "plants",
                        "zombies",
                        "projectiles",
                        "mowers",
                        "cooldowns",
                    )
                ],
            ),
        )
        self.references = (slots, offsets)
        self.ready = cp.cuda.Event(disable_timing=True)
        self.ready.record()
        self.size_bytes = self.device.nbytes
        if not device_only:
            self.download()

    def download(self):
        if self.event is not None:
            return
        cp, stream = self.cp, self.stream
        self.memory = cp.cuda.alloc_pinned_memory(self.device.nbytes)
        self.host = np.frombuffer(self.memory, np.int64, count=self.device.size)
        self.event = cp.cuda.Event(disable_timing=True)
        with stream:
            stream.wait_event(self.ready)
            self.device.get(out=self.host, stream=stream, blocking=False)
            self.event.record()

    @property
    def nbytes(self):
        return self.size_bytes

    def resolve(self, *, wait=False):
        if self.decoded is not None:
            return self.decoded
        self.download()
        if wait:
            self.event.synchronize()
        elif not self.event.done:
            return None
        decoded = []
        schemas = (s.PLANT, s.ZOMBIE, s.PROJECTILE, s.MOWER)
        for row, start in enumerate(self.offsets[:-1]):
            at = int(start)
            header = self.host[at : at + len(s.HEADER)][None, :]
            at += len(s.HEADER)
            slabs, counts = [], []
            for i, schema in enumerate(schemas):
                count = int(self.counts[i][row]) if i < 3 else 5
                size = count * len(schema)
                slabs.append(self.host[at : at + size].reshape(1, count, len(schema)).tolist())
                counts.append(np.array([count]))
                at += size
            cooldowns = self.host[at : at + 8].reshape(1, 8).tolist()
            decoded.extend(
                unpack_snapshots(
                    [self.templates[row]], header, cooldowns, slabs, counts, share_metadata=True
                )
            )
        self.decoded = decoded
        self.device = self.references = self.memory = self.host = None
        return decoded

    def roots(self):
        return [SnapshotRoot(self, i) for i in range(len(self.indices))]


class SnapshotRoot:
    def __init__(self, packet, index):
        self.packet, self.index = packet, index

    @property
    def estimated_bytes(self):
        return int(self.packet.offsets[self.index + 1] - self.packet.offsets[self.index]) * 8 + 8192

    def resolve(self, *, wait=False):
        decoded = self.packet.resolve(wait=wait)
        return decoded[self.index] if decoded is not None else None

    def device_slabs(self, stream):
        """Immutable occupied records; worker copies never alias the real game."""
        packet, row = self.packet, self.index
        if packet.device is None:
            return None
        stream.wait_event(packet.ready)
        at = int(packet.offsets[row])
        slabs = {}
        for name, count, width in (
            ("header", 1, len(s.HEADER)),
            ("plants", packet.counts[0][row], len(s.PLANT)),
            ("zombies", packet.counts[1][row], len(s.ZOMBIE)),
            ("projectiles", packet.counts[2][row], len(s.PROJECTILE)),
            ("mowers", 5, len(s.MOWER)),
            ("cooldowns", 1, 8),
        ):
            size = int(count) * width
            slabs[name] = packet.device[at : at + size]
            at += size
        return packet.templates[row], slabs

    def __getitem__(self, key):
        if key == "tick":
            return int(self.packet.headers[self.index, s.HEADER.index("tick")])
        return self.resolve(wait=True)[key]
