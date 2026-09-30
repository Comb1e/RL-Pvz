"""The single per-step host handoff: pinned copies completed by one stream wait."""

from time import perf_counter

import torch


class HostHandoff:
    """Reusable pinned buffers for every device-to-host copy of one decision step.

    Callers queue asynchronous copies on the collection stream; :meth:`wait`
    is the only synchronization. Buffers grow to the largest request and are
    reused, so a returned view is valid until the same name is queued again.
    """

    def __init__(self, stream):
        self.stream = stream
        self.buffers = {}
        self.wait_seconds = 0.0

    def enqueue(self, name, tensor):
        host = self.buffers.get(name)
        if host is None or host.dtype != tensor.dtype or host.numel() < tensor.numel():
            host = torch.empty(tensor.numel(), dtype=tensor.dtype, pin_memory=True)
            self.buffers[name] = host
        view = host[: tensor.numel()].view(tensor.shape)
        view.copy_(tensor, non_blocking=True)
        return view

    def wait(self):
        started = perf_counter()
        self.stream.synchronize()
        self.wait_seconds += perf_counter() - started

    def enqueue_fields(self, prefix, tensors):
        return {name: self.enqueue(f"{prefix}_{name}", value) for name, value in tensors.items()}

    @property
    def nbytes(self):
        return sum(buffer.numel() * buffer.element_size() for buffer in self.buffers.values())

    def clear(self):
        self.buffers.clear()
