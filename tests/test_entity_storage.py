"""Ragged entity storage budgets, recovery and corruption boundaries."""

from zipfile import ZipFile

import numpy as np
import pytest
from pvz_game import Rules

from pvz_rl.config import load_config
from pvz_rl.envs.encoding import ObservationEncoder, observations_equal
from pvz_rl.learning.cuda_buffer import CompleteGameBuffer


@pytest.mark.parametrize("ram_bytes", [0, 500, 10000])
def test_ragged_round_trip_shared_budget_and_empty_rows(tmp_path, ram_bytes):
    schema = ObservationEncoder(load_config(), Rules()).schema()
    buf = CompleteGameBuffer(
        tmp_path / "collect", 1, block_rows=3, ram_bytes=ram_bytes, schema=schema
    )
    restored = None
    observations = [
        dict(
            entities=np.tile([16, 17, 0, -500, 0, 0, 0, 0, 0, 0, 0], (n, 1)),
            globals=np.arange(18, dtype=np.float32) + i,
        )
        for i, n in enumerate([0, 1, 7, 0, 12])
    ]
    try:
        rows = np.zeros(5, buf.dtype)
        rows["active"] = True
        rows["reward"] = [1, 2, 3, 4, 5]
        buf.append(rows[:2], observations[:2])
        buf.append(rows[2:], observations[2:])
        assert buf.entity_size == 20 and buf.ram_used <= ram_bytes
        real = buf.take(np.arange(5))
        assert real["entity_offset"].tolist() == [0, 0, 1, 8, 8]
        assert all(
            observations_equal(a, b)
            for a, b in zip(observations, buf.observations(real).observations())
        )
        with ZipFile(tmp_path / "saved.zip", "w") as archive:
            buf.write_archive(archive)
        workspace = tmp_path / "recovered"
        workspace.mkdir()
        with ZipFile(tmp_path / "saved.zip") as archive:
            restored = CompleteGameBuffer.restore_archive(archive, buf.metadata(), workspace)
        assert restored.entity_size == 20 and restored.ram_used <= ram_bytes
        restored.finalize()
        np.testing.assert_array_equal(restored.take(np.arange(5))["target"], [15, 14, 12, 9, 5])
        assert all(
            observations_equal(a, b)
            for a, b in zip(observations, restored.observations(real).observations())
        )
    finally:
        buf.close()
        if restored:
            restored.close()


@pytest.mark.parametrize("field,value", [("entity_offset", 2), ("entity_count", 999)])
def test_corrupt_offsets_and_counts_rejected_before_recovery(tmp_path, field, value):
    buf = CompleteGameBuffer(tmp_path / "collect", 1, block_rows=2)
    try:
        buf.append(np.zeros(1, buf.dtype), [dict(entities=[], globals=[0] * 18)])
        state = buf.save(tmp_path / "saved")
        path = tmp_path / "saved/block-000000.npy"
        data = np.load(path)
        data[field] = value
        np.save(path, data)
        with pytest.raises(ValueError, match="offset/count"):
            CompleteGameBuffer.restore(tmp_path / "saved", state, tmp_path / "recover")
    finally:
        buf.close()


@pytest.mark.parametrize("fault", ["missing", "fields", "cap", "retired"])
def test_checkpoint_schema_rejected_without_reading_weights(tmp_path, fault):
    import json

    from pvz_rl.learning.checkpoints import inspect_checkpoint, protocol_for

    cfg = load_config()
    schema = ObservationEncoder(cfg, Rules()).schema()
    if fault == "fields":
        schema["fields"][4] = "aggregated_health"
    elif fault == "cap":
        schema["max_entities"] += 1
    identity = protocol_for(cfg["policy"]["kind"])
    if fault == "retired":
        identity["policy"] = "transformer_lstm_q_v1"
    path = tmp_path / "incompatible.zip"
    with ZipFile(path, "w") as archive:
        archive.writestr("protocol.json", json.dumps(identity))
        archive.writestr("run.json", json.dumps(dict(config=cfg)))
        for name in ("data", "policy.pth", "policy.optimizer.pth", "cohort-state.pt"):
            archive.writestr(name, "invalid weights must not be deserialized")
        if fault != "missing":
            archive.writestr("observation-schema.json", json.dumps(schema))
    with pytest.raises(ValueError, match="schema|fresh initialization"):
        inspect_checkpoint(path)
