"""Spawn bootstrap: configure CPU libraries before importing the simulator."""

import os


def run(connection):
    os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    if os.name == "nt":
        import ctypes

        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    from pvz_rl.envs.auxiliary_cpu import worker

    worker(connection)
