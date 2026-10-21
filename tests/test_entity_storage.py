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
        buffer.apply_causal_credits(
            {receipts[0].reference(0): {"damage": 1}, SourceActionRef(0, 99, 0): {"damage": 1}}
        )
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
        selected = [
            compactor.select(rows[start : start + width]) + start
            for start in range(0, len(rows), width)
        ]
        np.testing.assert_array_equal(np.concatenate(selected), expected)
        np.testing.assert_array_equal(compactor.counts, counts)
    buffer.close()


@pytest.mark.parametrize("spool_budget", [1, 10**8])
def test_failed_spool_build_releases_partial_ram_and_disk_allocations(
    tmp_path, monkeypatch, spool_budget
):
    buffer = _buffer(tmp_path)
    buffer.ram_bytes = spool_budget
    before = buffer.ram_used
    original = FittingSpool._allocate

    def fail(self, name, shape, dtype):
        if name == "entities":
            raise MemoryError("injected spool preparation failure")
        return original(self, name, shape, dtype)

    monkeypatch.setattr(FittingSpool, "_allocate", fail)
    with pytest.raises(MemoryError, match="injected"):
        FittingSpool(buffer)
    assert buffer.ram_used == before
    assert not list(tmp_path.glob("fitting-spool-*.npy"))
    buffer.close()


def test_prepared_source_preserves_reordered_episode_identities(tmp_path):
    from pvz_rl.learning.batch_transport import PreparedDataSource

    buffer = _buffer(tmp_path)
    rows = buffer.take(np.arange(5))
    rows["episode"] = [7, 1, 7, 3, 1]
    rows["decision"] = [5, 2, 4, 0, 1]
    observations = [dict(entities=[], globals=[index] + [0] * 17) for index in range(5)]
    with PreparedDataSource(observations, load_config(), rows=rows) as source:
        actual = list(source.batches(2, "cpu"))
        gathered = np.concatenate([item[0] for item in actual])
        np.testing.assert_array_equal(gathered["episode"], rows["episode"])
        np.testing.assert_array_equal(gathered["decision"], rows["decision"])
        assert [int(value) for _, boards, _ in actual for value in boards.globals[:, 0]] == list(
            range(5)
        )
    buffer.close()


def test_episode_returns_empty_execution_gap_and_corrupt_identity_recovery(tmp_path):
    buffer = CompleteGameBuffer(tmp_path / "live", 2, block_rows=3)
    observations = [dict(entities=[], globals=[0] * 18)] * 2
    rows = np.zeros(2, buffer.dtype)
    rows["env"], rows["active"], rows["accepted"] = [0, 1], True, True
    rows["reward"], rows["done"] = [1, 5], [False, True]
    first = buffer.append(rows, observations, identities=([10, 20], [0, 0]))
    rows["active"] = False
    assert len(buffer.append(rows, observations).rows) == 0
    rows["active"], rows["reward"], rows["done"] = True, [2, 7], True
    second = buffer.append(rows, observations, identities=([10, 21], [1, 0]))
    assert second.reference(1) == SourceActionRef(1, 21, 0)
    assert buffer.source_row(first.reference(1)) == 1
    buffer.finalize()
    stored = buffer.take(np.arange(4))
    np.testing.assert_array_equal(stored["target"], [3, 5, 2, 7])
    assert stored["run_break"].all()
    saved = tmp_path / "saved"
    state = buffer.save(saved)
    block = np.load(saved / "block-000000.npy")
    block["decision"][0] = 9
    np.save(saved / "block-000000.npy", block)
    with pytest.raises(ValueError, match="identity"):
        CompleteGameBuffer.restore(saved, state, tmp_path / "restored")
    buffer.close()
