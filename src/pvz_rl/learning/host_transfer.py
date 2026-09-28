"""Reusable pinned copies completed by the collector's existing stream boundary."""

import torch


class HostTransfer:
    def __init__(self):
        self.buffers = {}

    def enqueue(self, tensors):
        result = {}
        for name, tensor in tensors.items():
            host = self.buffers.get(name)
            if host is None or host.shape != tensor.shape or host.dtype != tensor.dtype:
                host = torch.empty_like(tensor, device="cpu", pin_memory=True)
                self.buffers[name] = host
            host.copy_(tensor, non_blocking=True)
            result[name] = host
        return result
