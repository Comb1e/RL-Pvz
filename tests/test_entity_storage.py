"""Actual-decision storage and reusable fitting-spool controls."""

import numpy as np

from pvz_rl.config import load_config
from pvz_rl.envs.encoding import ENTITY_WIDTH, GLOBAL_WIDTH, PackedEntityBatch
from pvz_rl.learning.batch_transport import FittingSpool, batch_rows
from pvz_rl.learning.cuda_buffer import TRAJECTORY_PROTOCOL, CompleteGameBuffer


def _buffer(tmp_path, rows=5):
    cfg = load_config()
    buffer = CompleteGameBuffer(
        tmp_path,
        1,
        ram_bytes=10**8,
        block_rows=rows,
        schema={"max_entities": 256},
        cfg=cfg,
    )
    records = np.zeros(rows, dtype=buffer.dtype)
    records["active"] = True
    records["env"] = 0
    records["action"] = np.arange(rows) % 2
    records["accepted"] = True
    observations = PackedEntityBatch(
        np.zeros((0, ENTITY_WIDTH), np.int32),
        np.zeros(rows, np.int64),
        np.zeros(rows, np.int64),
        np.zeros((rows, GLOBAL_WIDTH), np.float32),
    )
    buffer.append(records, observations)
    return buffer


def test_actual_storage_has_no_counterfactual_fields(tmp_path):
    buffer = _buffer(tmp_path)
    assert buffer.metadata()["protocol"] == TRAJECTORY_PROTOCOL
    assert "probes" not in buffer.dtype.names
    assert "probe_pending" not in buffer.dtype.names
    buffer.close()


def test_fitting_spool_reuses_width_order_and_rows(tmp_path):
    buffer = _buffer(tmp_path)
    spool = FittingSpool(buffer)
    batches = list(batch_rows(buffer, 2, spool=spool))
    assert len(spool) == 5
    assert sum(len(batch) for batch in batches) == 5
    assert all(np.all(batch["active"]) for batch in batches)
    spool.close()
    buffer.close()
