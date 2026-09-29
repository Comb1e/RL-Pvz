"""Fixed-shape AOT CUDA graphs with caller-owned activations and gradients."""

import torch


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


def _capture(graph_module, _example_inputs):
    graph = None
    static_inputs = outputs = None

    def replay(inputs):
        nonlocal graph, static_inputs, outputs
        with torch.no_grad():
            if graph is None:
                if not all(isinstance(x, torch.Tensor) and x.is_cuda for x in inputs):
                    raise RuntimeError("CUDA encoder graphs require only CUDA tensor inputs")
                device = inputs[0].device
                with torch.cuda.device(device):
                    static_inputs = [_owned_tensor(x) for x in inputs]
                    stream = torch.cuda.Stream(device=device)
                    stream.wait_stream(torch.cuda.current_stream(device))
                    with torch.cuda.stream(stream):
                        for _ in range(3):
                            graph_module(*static_inputs)
                    torch.cuda.current_stream(device).wait_stream(stream)
                    captured = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(captured, stream=stream):
                        outputs = graph_module(*static_inputs)
                    graph = captured
            for destination, source in zip(static_inputs, inputs, strict=True):
                _copy(destination, source)
            graph.replay()
            # Copy ALL AOT outputs, including saved forward activations and
            # backward gradients. Cloning only the public readouts is too late
            # to protect tensors held internally by AOT/autograd from replay.
            return [_owned_tensor(value) for value in outputs]

    def run(inputs):
        try:
            return replay(inputs)
        except torch.cuda.OutOfMemoryError:
            raise
        except Exception as exc:
            raise EncoderCompilationError(str(exc)) from exc

    run._boxed_call = True
    return run


def cudagraph_backend(graph_module, example_inputs):
    # This is the custom-backend interface documented by PyTorch 2.8. Capture
    # forward and backward separately, outside the recurrent/optimizer loops.
    from torch._dynamo.backends.common import aot_autograd

    return aot_autograd(fw_compiler=_capture, bw_compiler=_capture)(graph_module, example_inputs)
