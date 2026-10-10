"""Fixed-shape AOT CUDA graphs with caller-owned activations and gradients."""

from collections import OrderedDict
from time import perf_counter

import torch
from torch.utils._pytree import tree_map


class EncoderCompilationError(RuntimeError):
    """An optional compiled encoder failed before the pass could commit."""


def _copy(destination, source):
    # Broadcast views contain repeated addresses. Copy each distinct location
    # once, retaining the stride/alignment expected by SDPA's backward kernel.
    strides = destination.stride()
    if 0 not in strides:
        destination.copy_(source)
    else:
        index = tuple(slice(0, 1) if stride == 0 else slice(None) for stride in strides)
        destination[index].copy_(source[index])


def _owned_tensor(value):
    if not isinstance(value, torch.Tensor):
        return value
    result = torch.empty_strided(
        value.shape, value.stride(), dtype=value.dtype, device=value.device
    )
    _copy(result, value)
    return result


class GraphCapture:
    """One capture with explicit ownership and quiescent release."""

    def __init__(self, graph_module):
        self.graph_module = graph_module
        self.graph = self.static_inputs = self.outputs = self.stream = self.replay_stream = None
        self.device = None

    def __call__(self, inputs):
        with torch.inference_mode(False), torch.no_grad():
            if self.graph is None:
                if not inputs or not all(
                    isinstance(value, torch.Tensor) and value.is_cuda for value in inputs
                ):
                    raise RuntimeError("CUDA encoder graphs require only CUDA tensor inputs")
                self.device = inputs[0].device
                if any(value.device != self.device for value in inputs):
                    raise RuntimeError("CUDA encoder graph inputs must share one device")
                with torch.cuda.device(self.device):
                    self.static_inputs = [_owned_tensor(value) for value in inputs]
                    self.stream = torch.cuda.Stream(device=self.device)
                    current = torch.cuda.current_stream(self.device)
                    self.stream.wait_stream(current)
                    with torch.cuda.stream(self.stream):
                        for _ in range(3):
                            self.graph_module(*self.static_inputs)
                    current.wait_stream(self.stream)
                    self.graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(self.graph, stream=self.stream):
                        self.outputs = self.graph_module(*self.static_inputs)
                    current.wait_stream(self.stream)
            for destination, source in zip(self.static_inputs, inputs, strict=True):
                _copy(destination, source)
            self.graph.replay()
            # Copy ALL AOT outputs, including saved forward activations and
            # backward gradients. Cloning only the public readouts is too late
            # to protect tensors held internally by AOT/autograd from replay.
            value = tree_map(_owned_tensor, self.outputs)
            self.replay_stream = torch.cuda.current_stream(self.device)
            return value

    def retire(self):
        """Fence the last replay and owned-output copies without a host wait."""
        completed = torch.cuda.Event()
        completed.record(self.replay_stream)
        return completed

    def release(self):
        """Drop buffers after the owning phase has synchronized their device."""
        if self.graph is not None:
            self.graph.reset()
        self.graph = self.static_inputs = self.outputs = self.stream = self.replay_stream = None


def _release_captures(captures):
    devices = {capture.device for capture in captures if capture.device is not None}
    for device in devices:
        torch.cuda.synchronize(device)
    for capture in captures:
        capture.release()


class CollectionGraphCache:
    """Bounded LRU with event-fenced retirement and no host wait on eviction."""

    def __init__(self, entries, headroom_mib):
        self.limit = entries
        self.headroom_bytes = headroom_mib * 1024**2
        self.captures = OrderedDict()
        self.retired = []
        self.counters = dict(
            hits=0,
            misses=0,
            captures=0,
            evictions=0,
            retired_releases=0,
            capacity_fallbacks=0,
            headroom_fallbacks=0,
            oom_fallbacks=0,
            capture_failures=0,
            capture_setup_seconds=0.0,
            releases=0,
        )
        self.last_free_bytes = None

    @property
    def metrics(self):
        return dict(
            self.counters,
            entries=len(self.captures),
            retired_entries=len(self.retired),
            resident_entries=len(self.captures) + len(self.retired),
            cache_entries=self.limit,
            vram_headroom_mib=self.headroom_bytes // 1024**2,
            last_free_mib=(
                None if self.last_free_bytes is None else self.last_free_bytes / 1024**2
            ),
        )

    def _key(self, inputs):
        device = inputs[0].device
        return (
            tuple(
                (tuple(value.shape), value.stride(), value.dtype, value.device) for value in inputs
            ),
            torch.is_autocast_enabled(device.type),
            torch.get_autocast_dtype(device.type),
            torch.is_inference_mode_enabled(),
        )

    def __call__(self, encode, inputs):
        key = self._key(inputs)
        capture = self.captures.get(key)
        admitting = capture is None
        if capture is not None:
            self.counters["hits"] += 1
            self.captures.move_to_end(key)
        else:
            self.counters["misses"] += 1
            self.release_retired()
            if not self.limit:
                self.counters["capacity_fallbacks"] += 1
                return encode(*inputs)
            if len(self.captures) >= self.limit:
                if len(self.retired) >= self.limit:
                    self.counters["capacity_fallbacks"] += 1
                    return encode(*inputs)
                _, victim = self.captures.popitem(last=False)
                self.retired.append((victim, victim.retire()))
                self.counters["evictions"] += 1
            self.last_free_bytes, _ = torch.cuda.mem_get_info(inputs[0].device)
            if self.last_free_bytes < self.headroom_bytes:
                self.counters["headroom_fallbacks"] += 1
                return encode(*inputs)
            capture = GraphCapture(encode)
        out_of_memory = False
        setup_started = perf_counter() if admitting else None
        try:
            value = capture(inputs)
        except torch.cuda.OutOfMemoryError:
            self.counters["oom_fallbacks"] += 1
            _release_captures([capture])
            self.release()
            out_of_memory = True
        except Exception as exc:
            self.counters["capture_failures"] += 1
            _release_captures([capture])
            self.release()
            raise EncoderCompilationError(str(exc)) from exc
        finally:
            if setup_started is not None:
                self.counters["capture_setup_seconds"] += perf_counter() - setup_started
        if out_of_memory:
            with torch.cuda.device(inputs[0].device):
                torch.cuda.empty_cache()
            return encode(*inputs)
        if key not in self.captures:
            self.last_free_bytes, _ = torch.cuda.mem_get_info(inputs[0].device)
            if self.last_free_bytes < self.headroom_bytes:
                self.counters["headroom_fallbacks"] += 1
                _release_captures([capture])
                del value
                return encode(*inputs)
            self.captures[key] = capture
            self.counters["captures"] += 1
        return value

    def release_retired(self):
        """Poll fences after an existing handoff; never synchronize the device."""
        pending = []
        released = 0
        for capture, completed in self.retired:
            if completed.query():
                capture.release()
                released += 1
            else:
                pending.append((capture, completed))
        self.retired = pending
        self.counters["retired_releases"] += released
        return released

    def release(self):
        """Caller must be outside collection replay and fitting autograd work."""
        captures = list(self.captures.values()) + [capture for capture, _ in self.retired]
        if captures:
            _release_captures(captures)
            self.captures.clear()
            self.counters["retired_releases"] += len(self.retired)
            self.retired.clear()
        self.counters["releases"] += 1


class FittingGraphBackend:
    """AOT captures belong to one encoder, not the compiler's global cache."""

    def __init__(self):
        self.captures = []

    def __call__(self, graph_module, example_inputs):
        return cudagraph_backend(graph_module, example_inputs, owner=self)

    def release(self):
        """Release only after every delayed backward in the phase has finished."""
        _release_captures(self.captures)
        self.captures.clear()


def _capture(graph_module, _example_inputs, *, owner=None):
    capture = GraphCapture(graph_module)

    def run(inputs):
        if owner is not None and capture not in owner.captures:
            owner.captures.append(capture)
        try:
            return list(capture(inputs))
        except torch.cuda.OutOfMemoryError:
            raise
        except Exception as exc:
            raise EncoderCompilationError(str(exc)) from exc

    run._boxed_call = True
    return run


def cudagraph_backend(graph_module, example_inputs, *, owner=None):
    # This is the custom-backend interface documented by PyTorch 2.8. Capture
    # forward and backward separately, outside the recurrent/optimizer loops.
    from torch._dynamo.backends.common import aot_autograd

    def compiler(module, inputs):
        return _capture(module, inputs, owner=owner)

    return aot_autograd(fw_compiler=compiler, bw_compiler=compiler)(graph_module, example_inputs)
