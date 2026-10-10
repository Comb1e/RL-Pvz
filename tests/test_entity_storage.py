"""Ragged entity storage budgets, recovery and corruption boundaries."""

from zipfile import ZipFile

import numpy as np
import pytest
import torch
from pvz_game import Rules

from pvz_rl.config import load_config
from pvz_rl.envs.encoding import ObservationEncoder, PackedEntityBatch, observations_equal
from pvz_rl.learning.cuda_buffer import CompleteGameBuffer, probe_dtype
from pvz_rl.learning.recurrent_q import sequence_batches
from pvz_rl.learning.sequence_transport import SequencePrefetch
from pvz_rl.monitoring.entity_benchmark import observation


@pytest.mark.parametrize("ram_bytes", [0, 2**20])
def test_endpoint_values_patch_canonical_spilled_rows_before_finalization(tmp_path, ram_bytes):
    buffer = CompleteGameBuffer(tmp_path / "pending", 3, block_rows=2, ram_bytes=ram_bytes)
    try:
        rows = np.zeros(3, buffer.dtype)
        rows["active"] = True
        rows["env"] = np.arange(3)
        rows["probe_expected"] = rows["probe_pending"] = 1
        buffer.append(rows, [dict(entities=[], globals=[0] * 18)] * 3)
        records = np.zeros(3, probe_dtype())
        records["valid"] = True
        records["bootstrap"] = np.arange(3, dtype=np.float32) / 10
        endpoints = PackedEntityBatch(
            np.zeros((0, 11), np.int32),
            np.zeros(3, np.int64),
            np.zeros(3, np.int64),
            np.zeros((3, 18), np.float32),
        )
        invalid = records.copy()
        invalid["bootstrap"] = np.nan
        with pytest.raises(ValueError, match="Non-finite"):
            buffer.patch_probe_endpoints([0, 1, 2], [0, 0, 0], invalid, endpoints)
        with pytest.raises(IndexError, match="outside"):
            buffer.patch_probe_endpoints([3, 1, 2], [0, 0, 0], records, endpoints)
        buffer.patch_probe_endpoints([2, 0, 1], [0, 0, 0], records[[2, 0, 1]], endpoints)
        np.testing.assert_array_equal(
            buffer.take([0, 1, 2])["probes"][:, 0]["bootstrap"], records["bootstrap"]
        )
        buffer.rewards_finalized = True
        with pytest.raises(RuntimeError, match="finalized"):
            buffer.patch_probe_endpoints([0, 1, 2], [0, 0, 0], records, endpoints)
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
        p["valid"] = p["done"] = p["won"] = True
        p["action"] = 46
        p["tick"] = 100  # Must not enter the duration median.
        p["components"][:, 0] = 1
        p["bootstrap"] = 99  # Terminal target excludes this.
        p["transitions"] = p["stop_reason"] = 1
        rows["probes"][:, 1] = p
        from pvz_rl.envs.encoding import collate_observations

        obs = collate_observations([raw] * 3)
        entities, mask, globals_ = (value.numpy() for value in obs.tensors())
        counts = np.zeros((3, buf.layout.count), np.int64)
        counts[:, :2] = mask.sum(-1)[:, None]
        offsets = np.zeros_like(counts)
        offsets[:, :2] = (np.cumsum(mask.sum(-1)) - mask.sum(-1))[:, None]
        rows["probe_expected"] = rows["probes"]["valid"].sum(-1)
        rows["probe_pending"] = rows["probe_expected"]
        endpoints = rows["probes"][:, :2].copy().reshape(-1)
        rows["probes"] = 0
        probes = PackedEntityBatch(
            entities[mask],
            offsets[:, :2].reshape(-1),
            counts[:, :2].reshape(-1),
            np.repeat(globals_, 2, axis=0),
        )
        buf.append(rows, obs)
        buf.patch_probe_endpoints(np.repeat(np.arange(3), 2), np.tile([0, 1], 3), endpoints, probes)
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


@pytest.mark.parametrize("tiles, interrupt", [(1, False), (4, True), (44, False)])
def test_prefetch_order_spill_and_buffer_reuse(tmp_path, tiles, interrupt):
    cfg = load_config()
    cfg["training"]["objective"].update(tile_probes=tiles)
    raw, _ = observation(cfg, 5)
    buffer = CompleteGameBuffer(tmp_path / "buffer", 2, ram_bytes=24000, block_rows=8, cfg=cfg)
    prefetch = None
    recovered = None
    try:
        rows = np.zeros(40, buffer.dtype)
        rows["active"] = True
        rows["env"] = np.arange(40) % 2
        rows["action"] = np.arange(40)
        rows["reward"] = np.arange(40) / 10
        probes = rows["probes"]
        probes["valid"] = True
        probes["action"] = np.arange(buffer.layout.count)
        probes["accepted"] = probes["action"] % 2 == 0
        probes["executed_action"] = np.where(probes["accepted"], probes["action"], 0)
        probes["transitions"] = 1
        probes["stop_reason"] = 3
        probes["bootstrap"] = np.arange(probes.size).reshape(probes.shape) / 100
        rows["probe_expected"] = probes["valid"].sum(-1)
        buffer.append(rows, [raw] * 40)
        buffer.finalize()
        state = buffer.save(tmp_path / "saved")
        recovered = CompleteGameBuffer.restore(tmp_path / "saved", state, tmp_path / "restored")
        assert recovered.layout == buffer.layout
        np.testing.assert_array_equal(recovered.take(np.arange(40)), buffer.take(np.arange(40)))
        reference = list(sequence_batches(recovered, 4, 8))
        prefetch = SequencePrefetch(recovered, 4, 8, "cuda", 5)
        assert recovered.ram_used + recovered.staging_bytes <= recovered.ram_bytes
        assert any(isinstance(block, np.memmap) for block in recovered.blocks)
        collected = []
        for i, (reset, actual, obs, fields) in enumerate(prefetch):
            exp_reset, exp_rows, exp_obs = reference[i]
            assert reset == exp_reset
            np.testing.assert_array_equal(actual, exp_rows)
            for a, b in zip(obs.tensors(), exp_obs.tensors()):
                torch.testing.assert_close(a.cpu(), b)
            np.testing.assert_array_equal(fields["action"].cpu(), actual["action"])
            expected_probes = np.stack(
                [actual["probes"][name] for name in ("valid", "action", "target", "accepted")],
                -1,
            )
            np.testing.assert_array_equal(fields.probes.cpu(), expected_probes)
            collected.append(obs)
            if interrupt and i == 1:
                break
        prefetch.close()
        assert recovered.staging_bytes == 0
        # Old device batches still own their values after repeated host-slot reuse.
        for actual, (_, _, expected) in zip(collected, reference):
            torch.testing.assert_close(actual.entities.cpu(), expected.entities)
        assert len(collected) == (2 if interrupt else len(reference))
    finally:
        if prefetch:
            prefetch.close()
        if recovered:
            recovered.close()
        buffer.close()


def test_out_of_order_endpoint_patches_are_owned_exact_once_and_recover_spill(tmp_path):
    from pvz_rl.learning.cuda_buffer import probe_dtype

    cfg = load_config()
    raw, _ = observation(cfg, 5)
    buffer = CompleteGameBuffer(tmp_path / "async", 3, ram_bytes=0, block_rows=2, cfg=cfg)
    recovered = None
    try:
        rows = np.zeros(6, buffer.dtype)
        rows["env"] = np.arange(6) % 3
        rows["active"][[0, 2, 4]] = True
        rows["action"][[0, 4]] = 1
        rows["accepted"] = True
        rows["probe_expected"][[0, 4]] = 2
        rows["probe_pending"] = rows["probe_expected"]
        buffer.append(rows, [raw] * 6)
        records = np.zeros(2, probe_dtype())
        records["valid"] = records["accepted"] = True
        records["action"] = records["executed_action"] = 2
        records["transitions"] = 1
        records["stop_reason"] = 3
        records["bootstrap"] = 0.5
        entities = np.tile(raw["entities"], (2, 1))
        endpoint = PackedEntityBatch(
            entities, np.array([0, 5]), np.array([5, 5]), np.tile(raw["globals"], (2, 1))
        )
        buffer.patch_probe_endpoints([4, 0], [1, 0], records, endpoint)
        assert buffer.pending_probes == 2
        with pytest.raises(RuntimeError, match="Pending probe"):
            buffer.save(tmp_path / "incomplete")
        before = buffer.entity_size
        with pytest.raises(ValueError, match="already-patched"):
            buffer.patch_probe_endpoints([4, 0], [1, 0], records, endpoint)
        assert buffer.entity_size == before
        buffer.patch_probe_endpoints([0, 4], [1, 0], records, endpoint)
        entities[:] = 0
        assert buffer.pending_probes == 0
        state = buffer.save(tmp_path / "saved-async")
        recovered = CompleteGameBuffer.restore(
            tmp_path / "saved-async", state, tmp_path / "restored-async"
        )
        expected = buffer.take(np.arange(6))
        np.testing.assert_array_equal(recovered.take(np.arange(6)), expected)
        assert expected["probes"]["valid"].sum() == 4 and not expected["probe_pending"].any()
        assert recovered.observations(expected["probes"][[0, 4]]).entities.any()
    finally:
        if recovered:
            recovered.close()
        buffer.close()


@pytest.mark.parametrize("budget", [256, 512])
def test_actual_decision_sequences_ignore_long_capacity_gaps_and_keep_256_boundaries(
    tmp_path, budget
):
    from pvz_rl.learning.sequence_transport import sequence_rows

    cfg = load_config()
    raw, _ = observation(cfg, 5)
    buffer = CompleteGameBuffer(tmp_path / "gaps", 2, ram_bytes=0, block_rows=64, cfg=cfg)
    try:
        rows = np.zeros(1800, buffer.dtype)
        rows["env"] = np.arange(len(rows)) % 2
        rows["active"][np.arange(0, 1800, 6)] = True
        rows["active"][[1001, 1401]] = True
        rows["accepted"] = True
        rows["reward"] = np.where(rows["env"] == 0, 1, 2)
        rows["reset"][[0, 1001]] = True
        buffer.append(rows, [raw] * len(rows))
        buffer.finalize()
        chunks = list(sequence_rows(buffer, 256, budget))
        histories = {0: [], 1: []}
        counts = []
        for reset, chunk in chunks:
            counts.append(chunk["active"].sum(-1).tolist())
            for sequence in chunk:
                actual = sequence[sequence["active"]]
                if len(actual):
                    histories[int(actual["env"][0])].extend(actual["target"].tolist())
                    assert len(actual) <= 256
            if reset:
                assert all(
                    sequence[sequence["active"]][0]["reset"]
                    for sequence in chunk
                    if sequence["active"].any()
                )
        np.testing.assert_array_equal(histories[0], np.arange(300, 0, -1))
        np.testing.assert_array_equal(histories[1], [4, 2])
        assert counts == ([[256], [44], [2]] if budget == 256 else [[256, 2], [44, 0]])
    finally:
        buffer.close()


@pytest.mark.parametrize("ram_bytes", [0, 4096])
def test_sequence_maps_share_budget_spill_and_release_on_early_close(tmp_path, ram_bytes):
    from pvz_rl.learning.sequence_transport import sequence_rows

    buffer = CompleteGameBuffer(tmp_path / "maps", 2, ram_bytes=ram_bytes, block_rows=2)
    source = None
    try:
        rows = np.zeros(6, buffer.dtype)
        rows["active"] = True
        rows["env"] = [1, 0, 1, 0, 1, 0]
        rows["action"] = np.arange(6)
        buffer.append(rows, [dict(entities=[], globals=[0] * 18)] * 6)
        source = sequence_rows(buffer, 2, 4)
        reset, actual = next(source)
        assert reset and actual["action"].tolist() == [[1, 3], [0, 2]]
        if ram_bytes:
            assert buffer.staging_bytes == (6 + 3 * 2) * 8
            assert buffer.ram_used + buffer.staging_bytes <= ram_bytes
        else:
            assert buffer.staging_bytes == 0 and list(buffer.path.glob("sequence-*/history.npy"))
        source.close()
        assert buffer.staging_bytes == 0 and not list(buffer.path.glob("sequence-*"))
    finally:
        if source is not None:
            source.close()
        buffer.close()
