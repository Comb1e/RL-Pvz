"""Availability probes and optional phase instrumentation; never formal training."""

from contextlib import contextmanager


class DeviceProfiler:
    """CUDA events measure execution on the shared stream, not enqueue latency."""

    def __init__(self, cp, enabled=False):
        self.cp, self.enabled = cp, enabled
        self.pending, self.seconds = [], {}

    @contextmanager
    def track(self, phase):
        if not self.enabled:
            yield
            return
        start, end = self.cp.cuda.Event(), self.cp.cuda.Event()
        start.record()
        try:
            yield
        finally:
            end.record()
            self.pending.append((phase, start, end))

    def flush(self):
        for phase, start, end in self.pending:
            end.synchronize()
            self.seconds[phase] = (
                self.seconds.get(phase, 0.0) + self.cp.cuda.get_elapsed_time(start, end) / 1000
            )
        self.pending.clear()
        return dict(self.seconds)


def cuda_doctor():
    """Compile, run a game, verify hash parity, and test bidirectional sharing."""
    import torch
    from pvz_game import Game, LevelSpec
    from pvz_game.cuda import CudaBatch

    if not torch.cuda.is_available():
        return {"available": False, "error": "PyTorch CUDA is unavailable"}
    try:
        batch = CudaBatch(1, zombie_capacity=1, max_step_ticks=1)
        cp = batch.cp
        stream = torch.cuda.current_stream()
        with cp.cuda.ExternalStream(stream.cuda_stream, device_id=stream.device_index):
            batch.reset([LevelSpec("cuda-doctor")], [0])
            tensor = torch.from_dlpack(batch.header)
            assert tensor.data_ptr() == batch.header.data.ptr
            action = torch.zeros(1, device="cuda", dtype=torch.long)
            batch.step_device(cp.from_dlpack(action))
            oracle = Game()
            oracle.reset(LevelSpec("cuda-doctor"), 0)
            oracle.step()
            assert batch.state_hash(0) == oracle.state_hash()
            probe = torch.ones(3, device="cuda")
            shared = cp.from_dlpack(probe)
            shared += 2
            assert probe.cpu().tolist() == [3, 3, 3]
        free, total = cp.cuda.runtime.memGetInfo()
        return {
            "available": True,
            "cupy": cp.__version__,
            "kernel_compilation": "passed",
            "dlpack_shared_stream": "passed",
            "simulation_hash": oracle.state_hash(),
            "device": torch.cuda.get_device_name(),
            "free_mib": free / 2**20,
            "total_mib": total / 2**20,
            "projectile_capacity": batch.qcap,
        }
    except Exception as exc:
        return {"available": False, "error": repr(exc)}
