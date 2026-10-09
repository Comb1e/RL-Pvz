"""Availability probes and optional phase instrumentation; never formal training."""

from contextlib import contextmanager


class DeviceProfiler:
    """Stream-ordered device phase spans plus host phase timings.

    CUDA events bracket device phases on the recording stream. :meth:`flush`
    reads only completed spans, normally after the caller's existing host
    handoff. Pending events stay queued rather than adding a wait. Event objects
    are pooled; disabled profilers record no events. ``snapshot`` reads only
    cached host totals and never queries the device.
    """

    def __init__(self, cp, enabled=False):
        self.cp, self.enabled = cp, enabled
        self.pending, self.seconds, self._pool = [], {}, []

    def _event(self):
        return self._pool.pop() if self._pool else self.cp.cuda.Event()

    @contextmanager
    def track(self, phase):
        if not self.enabled:
            yield
            return
        start, end = self._event(), self._event()
        start.record()
        try:
            yield
        finally:
            end.record()
            self.pending.append((phase, start, end))

    def host(self, phase, seconds):
        self.seconds[phase] = self.seconds.get(phase, 0.0) + seconds

    def flush(self):
        pending = []
        for phase, start, end in self.pending:
            if not end.done:
                pending.append((phase, start, end))
                continue
            self.seconds[phase] = (
                self.seconds.get(phase, 0.0) + self.cp.cuda.get_elapsed_time(start, end) / 1000
            )
            self._pool.extend((start, end))
        self.pending[:] = pending
        return dict(self.seconds)

    def snapshot(self):
        """Cached cohort-cumulative timings for ``model.collection_execution``.

        CuPy 13.6's Event.done uses nonblocking cudaEventQuery; flush must never
        call synchronize. Source: https://github.com/cupy/cupy/blob/v13.6.0/cupy/cuda/stream.pyx
        """
        return dict(
            scope="current_cohort",
            completed_phase_seconds=dict(self.seconds),
            pending_device_spans=len(self.pending),
        )


def cuda_doctor():
    """Compile, run a game, verify hash parity, and test bidirectional sharing."""
    import torch
    from pvz_game import Game, LevelSpec

    from pvz_rl.config import load_config
    from pvz_rl.envs.cuda_accounting import ACCOUNTING_WIDTH, AccountingCudaBatch
    from pvz_rl.envs.cuda_features import CudaFeatures

    if not torch.cuda.is_available():
        return {"available": False, "error": "PyTorch CUDA is unavailable"}
    try:
        batch = AccountingCudaBatch(1, zombie_capacity=1, max_step_ticks=1)
        cp = batch.cp
        stream = torch.cuda.current_stream()
        with cp.cuda.ExternalStream(stream.cuda_stream, device_id=stream.device_index):
            batch.reset([LevelSpec("cuda-doctor")], [0])
            tensor = torch.from_dlpack(batch.header)
            assert tensor.data_ptr() == batch.header.data.ptr
            action = torch.zeros(1, device="cuda", dtype=torch.long)
            features = CudaFeatures(batch, load_config(), "masked")
            features.encode()
            features.step(cp.from_dlpack(action))
            assert features.rewards.get().tolist() == [1.0]
            assert batch.accounting.get().tolist() == [[0] * ACCOUNTING_WIDTH]
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
            "research_accounting": "passed",
            "dlpack_shared_stream": "passed",
            "simulation_hash": oracle.state_hash(),
            "device": torch.cuda.get_device_name(),
            "free_mib": free / 2**20,
            "total_mib": total / 2**20,
            "projectile_capacity": batch.qcap,
        }
    except Exception as exc:
        return {"available": False, "error": repr(exc)}
