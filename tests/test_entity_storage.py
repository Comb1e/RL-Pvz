"""Actual-decision storage and reusable fitting-spool controls."""

import numpy as np
import pytest

from pvz_rl.config import load_config
from pvz_rl.envs.encoding import ENTITY_WIDTH, GLOBAL_WIDTH, PackedEntityBatch
from pvz_rl.learning.action_routing import InvalidRowCompactor, compact_invalid_rows
from pvz_rl.learning.batch_transport import FittingSpool, batch_rows
from pvz_rl.learning.cuda_buffer import TRAJECTORY_PROTOCOL, CompleteGameBuffer
from pvz_rl.learning.source_actions import SourceActionRef


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
    records["route"] = (records["action"] != 0).astype(np.uint8)
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


@pytest.mark.parametrize("ram_bytes", [1, 10**7])
def test_compact_identity_boundaries_credit_atomicity_and_restore(tmp_path, ram_bytes):
    cfg = load_config()
    buffer = CompleteGameBuffer(tmp_path / "live", 2, cfg=cfg, block_rows=3, ram_bytes=ram_bytes)
    receipts = []
    for active in ([True, True], [False, True], [True, True]):
        rows = np.zeros(2, buffer.dtype)
        rows["env"] = [0, 1]
        rows["active"] = active
        rows["action"] = 1
        rows["route"] = 2
        rows["rejection_reason"] = 2
        observations = [dict(entities=np.zeros((0, 11), np.int32), globals=np.zeros(18))] * 2
        receipts.append(buffer.append(rows, observations))
    assert buffer.size == 5
    assert receipts[-1].reference(0) == SourceActionRef(0, 0, 1)
    rows = buffer.take(np.arange(buffer.size))
    selected, _, _ = compact_invalid_rows(rows)
    np.testing.assert_array_equal(selected, [0, 1, 3])
    before = rows.copy()
    with pytest.raises(ValueError, match="identity"):
        buffer.apply_causal_credits({receipts[0].reference(0): {"damage": 1},
                                    SourceActionRef(0, 99, 0): {"damage": 1}})
    np.testing.assert_array_equal(buffer.take(np.arange(buffer.size)), before)
    buffer.apply_causal_credits({receipts[0].reference(0): {"damage": 30}})
    assert buffer._row_view(0)["components"][2] == 30
    state = buffer.save(tmp_path / "saved")
    restored = CompleteGameBuffer.restore(tmp_path / "saved", state, tmp_path / "restored")
    assert restored.source_row(receipts[-1].reference(0)) == 3
    np.testing.assert_array_equal(restored.take(np.arange(5)), buffer.take(np.arange(5)))
    restored.close()
    buffer.close()


def test_segmented_compaction_is_partition_invariant(tmp_path):
    buffer = _buffer(tmp_path, 401)
    rows = buffer.take(np.arange(buffer.size))
    rows["action"] = 1
    rows["accepted"] = False
    rows["route"] = 2
    rows["rejection_reason"] = 2
    rows["run_break"] = False
    rows["run_break"][[0, 199]] = True
    rows["decision"][300:] += 1  # A sampled gap also starts a new run.
    expected = [0, 100, 199, 299, 300, 400]
    single, counts, _ = compact_invalid_rows(rows)
    np.testing.assert_array_equal(single, expected)
    for width in (1, 17, 100, 128, 201):
        compactor = InvalidRowCompactor()
        selected = [compactor.select(rows[start:start + width]) + start
                    for start in range(0, len(rows), width)]
        np.testing.assert_array_equal(np.concatenate(selected), expected)
        np.testing.assert_array_equal(compactor.counts, counts)
    buffer.close()
