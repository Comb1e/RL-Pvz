"""Ragged entity storage budgets, recovery and corruption boundaries."""

from zipfile import ZipFile

import numpy as np
import pytest
import torch
from pvz_game import Rules

from pvz_rl.config import load_config
from pvz_rl.envs.encoding import ObservationEncoder, observations_equal
from pvz_rl.learning.cuda_buffer import CompleteGameBuffer
from pvz_rl.learning.recurrent_q import sequence_batches
from pvz_rl.learning.sequence_transport import SequencePrefetch
from pvz_rl.monitoring.entity_benchmark import observation


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


@pytest.mark.parametrize("interrupt", [False, True])
def test_prefetch_order_spill_and_buffer_reuse(tmp_path, interrupt):
    cfg = load_config()
    raw, _ = observation(cfg, 5)
    buffer = CompleteGameBuffer(tmp_path / "buffer", 2, ram_bytes=24000, block_rows=8)
    prefetch = None
    try:
        rows = np.zeros(40, buffer.dtype)
        rows["active"] = True
        rows["env"] = np.arange(40) % 2
        rows["action"] = np.arange(40)
        rows["reward"] = np.arange(40) / 10
        buffer.append(rows, [raw] * 40)
        buffer.finalize()
        reference = list(sequence_batches(buffer, 4, 8))
        prefetch = SequencePrefetch(buffer, 4, 8, "cuda", 5)
        assert buffer.ram_used + buffer.staging_bytes <= buffer.ram_bytes
        collected = []
        for i, (reset, actual, obs, fields) in enumerate(prefetch):
            exp_reset, exp_rows, exp_obs = reference[i]
            assert reset == exp_reset
            np.testing.assert_array_equal(actual, exp_rows)
            for a, b in zip(obs.tensors(), exp_obs.tensors()):
                torch.testing.assert_close(a.cpu(), b)
            np.testing.assert_array_equal(fields["action"].cpu(), actual["action"])
            collected.append(obs)
            if interrupt and i == 1:
                break
        prefetch.close()
        assert buffer.staging_bytes == 0
        # Old device batches still own their values after repeated host-slot reuse.
        for actual, (_, _, expected) in zip(collected, reference):
            torch.testing.assert_close(actual.entities.cpu(), expected.entities)
        assert len(collected) == (2 if interrupt else len(reference))
    finally:
        if prefetch:
            prefetch.close()
        buffer.close()
