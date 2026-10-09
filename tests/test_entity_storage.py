"""Ragged entity storage budgets, recovery and corruption boundaries."""

from zipfile import ZipFile

import numpy as np
import pytest
import torch
from pvz_game import Rules

from pvz_rl.config import load_config
from pvz_rl.envs.encoding import ObservationEncoder, PackedEntityBatch, observations_equal
from pvz_rl.learning.cuda_buffer import CompleteGameBuffer
from pvz_rl.learning.recurrent_q import sequence_batches
from pvz_rl.learning.sequence_transport import SequencePrefetch
from pvz_rl.monitoring.entity_benchmark import observation


@pytest.mark.parametrize("ram_bytes", [0, 2**20])
def test_pending_bootstraps_update_canonical_spilled_rows_before_finalization(tmp_path, ram_bytes):
    buffer = CompleteGameBuffer(tmp_path / "pending", 3, block_rows=2, ram_bytes=ram_bytes)
    try:
        rows = np.zeros(3, buffer.dtype)
        rows["active"] = True
        rows["env"] = np.arange(3)
        rows["probes"]["valid"] = True
        buffer.append(rows, [dict(entities=[], globals=[0] * 18)] * 3)
        expected = np.arange(12, dtype=np.float32).reshape(3, 4) / 10
        buffer.update_probe_bootstrap([2, 0, 1], expected[[2, 0, 1]])
        np.testing.assert_array_equal(buffer.take([0, 1, 2])["probes"]["bootstrap"], expected)
        with pytest.raises(ValueError, match="pending"):
            buffer.update_probe_bootstrap([3], expected[:1])
        with pytest.raises(ValueError, match="pending"):
            buffer.update_probe_bootstrap([0], np.full((1, 4), np.nan))
        buffer.rewards_finalized = True
        with pytest.raises(RuntimeError, match="finalization"):
            buffer.update_probe_bootstrap([0], expected[:1])
    finally:
        buffer.close()


def test_committed_metadata_spill_and_corruption(tmp_path):
    buffer = CompleteGameBuffer(tmp_path / "collect", 1, block_rows=1, ram_bytes=0)
    recovered = None
    try:
        buffer.exploration = dict(plant_epsilon=0.1, tile_epsilon=0.5)
        rows = np.zeros(1, buffer.dtype)
        rows["active"] = rows["accepted"] = True
        rows["policy_action"] = 46
        rows["exploration_mode"] = 1
        rows["pending_species"] = rows["next_species"] = 1
        buffer.append(rows, [dict(entities=[], globals=[0] * 18)])
        buffer.finalize()
        state = buffer.save(tmp_path / "saved")
        recovered = CompleteGameBuffer.restore(tmp_path / "saved", state, tmp_path / "restored")
        np.testing.assert_array_equal(recovered.take([0]), buffer.take([0]))
        assert recovered.exploration == buffer.exploration
        path = tmp_path / "saved/block-000000.npy"
        corrupted = np.load(path)
        corrupted["exploration_mode"] = 2
        np.save(path, corrupted)
        with pytest.raises(ValueError, match="exploration"):
            CompleteGameBuffer.restore(tmp_path / "saved", state, tmp_path / "bad")
    finally:
        if recovered:
            recovered.close()
        buffer.close()


@pytest.mark.parametrize("corruption", ["negative", "write_mask", "counts", "counts_mismatch"])
def test_event_storage_corruption_is_rejected(tmp_path, corruption):
    buffer = CompleteGameBuffer(tmp_path / "collect", 1, block_rows=1, ram_bytes=0)
    try:
        rows = np.zeros(1, buffer.dtype)
        rows["active"] = True
        rows["events"][0, 0] = 25
        rows["memory_write"] = True
        buffer.append(rows, [dict(entities=[], globals=[0] * 18)])
        buffer.finalize()
        state = buffer.save(tmp_path / "saved")
        path = tmp_path / "saved/block-000000.npy"
        stored = np.load(path)
        if corruption == "counts":
            state["outcome_counts"] = np.zeros((3, 1), dtype=np.int64)
        elif corruption == "counts_mismatch":
            state["outcome_counts"] = state["outcome_counts"].copy()
            state["outcome_counts"][0, 0] += 1
        elif corruption == "negative":
            stored["events"][0, 7] = -1
        else:
            stored["memory_write"] = False
        np.save(path, stored)
        with pytest.raises(ValueError, match="event|acceptance"):
            CompleteGameBuffer.restore(tmp_path / "saved", state, tmp_path / "bad")
    finally:
        buffer.close()


def test_probe_dedup_median_finalization_and_spilled_recovery(tmp_path):
    cfg = load_config()
    raw, _ = observation(cfg, 5)
    buf = CompleteGameBuffer(tmp_path / "collect", 3, block_rows=2, ram_bytes=0, cfg=cfg)
    recovered = None
    try:
        rows = np.zeros(3, buf.dtype)
        rows["active"] = rows["components_valid"] = rows["done"] = True
        rows["env"] = [0, 1, 2]
        rows["tick"] = [8000, 16000, 24000]
        rows["won"] = [True, False, True]
        rows["components"][:, 0] = [1, -2, 1]
        rows["components"][:, 1] = [0.01, 0, 0.02]
        p = rows["probes"][:, 0]
        p["valid"] = p["branch_role"] = p["done"] = p["won"] = True
        p["action"] = 46
        p["tick"] = 100  # Must not enter the duration median.
        p["components"][:, 0] = 1
        p["bootstrap"] = 99  # Terminal target excludes this.
        rows["probes"][:, 1] = p
        from pvz_rl.envs.encoding import collate_observations

        obs = collate_observations([raw] * 3)
        entities, mask, globals_ = (value.numpy() for value in obs.tensors())
        counts = np.zeros((3, 4), np.int64)
        counts[:, :2] = mask.sum(-1)[:, None]
        offsets = np.zeros_like(counts)
        offsets[:, :2] = (np.cumsum(mask.sum(-1)) - mask.sum(-1))[:, None]
        probes = PackedEntityBatch(
            entities[mask], offsets, counts, np.repeat(globals_[:, None], 4, axis=1)
        )
        buf.append(rows, obs, probes)
        stored = buf.take(np.arange(3))
        assert buf.entity_size == 30  # Three actual + three unique probe boards.
        np.testing.assert_array_equal(
            stored["probes"][:, 0]["entity_offset"], stored["probes"][:, 1]["entity_offset"]
        )
        state = buf.save(tmp_path / "saved")
        recovered = CompleteGameBuffer.restore(tmp_path / "saved", state, tmp_path / "restored")
        recovered.finalize()
        summary = recovered.reward_summary
        assert summary["duration_median_seconds"] == 160
        assert summary["duration_reference_count"] == 3
        assert summary["victory_time_adjustments"] == pytest.approx([1 / 30, -0.02])
        actual = recovered.take(np.arange(3))
        assert actual["target"].tolist() == pytest.approx([1.1 + 1 / 30, -2, 1.2 - 0.02])
        assert actual["probes"][:, 0]["target"].tolist() == pytest.approx([1 + 0.1 * 159 / 161] * 3)
        before = actual.copy()
        recovered.finalize_rewards()
        np.testing.assert_array_equal(before, recovered.take(np.arange(3)))
        saved = recovered.save(tmp_path / "finalized")
        pth = tmp_path / "finalized/block-000000.npy"
        broken = np.load(pth)
        broken["probes"][0, 0]["entity_offset"] = 100000
        np.save(pth, broken)
        with pytest.raises(ValueError, match="offset/count"):
            CompleteGameBuffer.restore(tmp_path / "finalized", saved, tmp_path / "bad")
    finally:
        buf.close()
        if recovered:
            recovered.close()


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
        rows["events"][1, [0, 1, 7]] = [25, 25, 1000]
        rows["events"][3, [5, 6]] = [1, 1]
        rows["memory_write"][[1, 3]] = True
        rows["accepted"] = [1, 0, 0, 1, 1]
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
        np.testing.assert_array_equal(restored.take(np.arange(5))["events"], rows["events"])
        restored.finalize()
        np.testing.assert_array_equal(restored.outcome_counts, [[2, 3], [0, 0], [0, 0]])
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
