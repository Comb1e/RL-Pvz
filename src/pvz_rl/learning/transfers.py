"""Bounded raw-token caching and one-batch-ahead, event-ordered CUDA transfers."""

from concurrent.futures import ThreadPoolExecutor
from time import perf_counter

import numpy as np
import torch

VRAM_RESERVE = 2 * 1024**3


class TokenCache:
    """A disposable prefix of raw history, allocated in trajectory-sized blocks."""

    def __init__(self, device, block_rows, width, budget):
        self.device = torch.device(device)
        self.block_rows, self.width, self.budget = block_rows, width, int(budget)
        self.blocks = []
        self.size = 0
        self.full = self.budget == 0

    def append(self, tokens):
        if self.full:
            return
        offset = 0
        while offset < len(tokens):
            block, row = divmod(self.size, self.block_rows)
            if block == len(self.blocks):
                amount = self.block_rows * self.width * 4
                free, _ = torch.cuda.mem_get_info(self.device)
                if (block + 1) * amount > self.budget or free - amount < VRAM_RESERVE:
                    self.full = True
                    return
                try:
                    allocated = torch.empty(
                        (self.block_rows, self.width), device=self.device, dtype=torch.float32
                    )
                    # The caching allocator may reserve more than tensor bytes.
                    if torch.cuda.mem_get_info(self.device)[0] < VRAM_RESERVE:
                        del allocated
                        torch.cuda.empty_cache()
                        self.full = True
                        return
                    self.blocks.append(allocated)
                except torch.OutOfMemoryError:
                    self.full = True
                    return
            count = min(len(tokens) - offset, self.block_rows - row)
            self.blocks[block][row : row + count].copy_(tokens[offset : offset + count])
            self.size += count
            offset += count


class StagingSlot:
    def __init__(self, metrics):
        self.metrics = metrics
        self.host = {}
        self.ready = torch.cuda.Event()
        self.started = torch.cuda.Event(enable_timing=True)
        self.finished = torch.cuda.Event(enable_timing=True)
        self.pending = False

    def drain(self):
        if self.pending:
            self.ready.synchronize()
            self.metrics["transfer_stream_seconds"] += (
                self.started.elapsed_time(self.finished) / 1000
            )
            self.pending = False

    def tensor(self, key, array, device):
        array = np.ascontiguousarray(array)
        source = torch.from_numpy(array)
        saved = self.host.get(key)
        if saved is None or saved.dtype != source.dtype or saved.numel() < source.numel():
            saved = torch.empty(source.numel(), dtype=source.dtype, pin_memory=True)
            self.host[key] = saved
        host = saved[: source.numel()].view(source.shape)
        host.copy_(source)
        return host.to(device, non_blocking=True)


class BatchPrefetch:
    """Exactly two staging slots, one pending batch, no sampling or optimizer work.

    Host sources cannot be reused before their copy event. Device allocations are
    retained by the consumer and recorded on its stream before the next transfer.
    """

    def __init__(self, buffer, indices, batch_size, device):
        self.buffer, self.indices, self.batch_size = buffer, indices, batch_size
        self.device = torch.device(device)
        self.slots = [StagingSlot(buffer.transport_metrics), StagingSlot(buffer.transport_metrics)]
        self.stream = torch.cuda.Stream(device=self.device)
        self.stream.wait_stream(torch.cuda.current_stream(self.device))
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pvz-transfer")
        self.cursor = self.sequence = 0
        self.future = None
        self.closed = False
        self._submit()

    def _prepare(self, indices, slot):
        with torch.cuda.device(self.device), torch.cuda.stream(self.stream), torch.no_grad():
            slot.drain()
            slot.started.record(self.stream)
            data = self.buffer.batch(indices, self.device, staging=slot)
            slot.finished.record(self.stream)
            slot.ready.record(self.stream)
            slot.pending = True
            return data, slot

    def _submit(self):
        if self.cursor < len(self.indices):
            indices = self.indices[self.cursor : self.cursor + self.batch_size]
            slot = self.slots[self.sequence % 2]
            self.cursor += len(indices)
            self.sequence += 1
            self.future = self.worker.submit(self._prepare, indices, slot)
        else:
            self.future = None

    def __iter__(self):
        return self

    def __next__(self):
        if self.future is None:
            raise StopIteration
        start = perf_counter()
        data, slot = self.future.result()
        self.buffer.transport_metrics["transfer_wait_seconds"] += perf_counter() - start
        stream = torch.cuda.current_stream(self.device)
        stream.wait_event(slot.ready)
        tensors = [v for v in data.values() if isinstance(v, torch.Tensor)]
        tensors.extend(getattr(data["context"], k) for k in ("tokens", "valid", "counts", "starts"))
        for tensor in tensors:
            tensor.record_stream(stream)
        self._submit()
        return data

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.worker.shutdown(wait=True)
        try:
            if self.future is not None:
                self.future.result()  # Never hide a failed preparation task.
        finally:
            self.stream.synchronize()
            for slot in self.slots:
                slot.drain()
            self.slots.clear()
