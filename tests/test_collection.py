"""Batched scratch isolation, endpoint deduplication and collection boundaries."""

import numpy as np
import pytest
import torch
from pvz_game import Game, InitialPlant, LevelSpec, Spawn

from pvz_rl.config import load_config
from pvz_rl.envs.cuda_env import CudaVecEnv
from pvz_rl.envs.encoding import collate_observations
from pvz_rl.envs.probes import ProbePool, endpoint_representatives
from pvz_rl.monitoring.entity_benchmark import observation
from pvz_rl.policy.two_stage_transformer import TwoStageTransformerPolicy

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")


@pytest.fixture
def collection_env():
    cfg = load_config()
    cfg["training"]["n_envs"] = 3
    cfg["training"]["performance"].update(compile_kernels=False, telemetry=True)
    cfg["visualization"].update(enabled=False, live_enabled=False)
    cfg["policy"].update(
        species_width=16,
        species_heads=1,
        species_layers=1,
        species_feedforward=32,
        tile_width=16,
        tile_heads=1,
        tile_layers=1,
        tile_feedforward=32,
    )
    env = CudaVecEnv(cfg, "masked", 101)
    env.reset()
    snapshots = []
    for index, count in enumerate((0, 30, 60)):
        game = Game()
        game.reset(
            LevelSpec(
                f"probes-{index}",
                tuple(Spawn(1, "basic", offset % 5, x=9000 - offset) for offset in range(count))
                + (Spawn(10000, "basic", 0),),
                initial_sun=50 if index == 0 else 500,
                plants=(InitialPlant("peashooter", 0, 0), InitialPlant("repeater", 1, 0)),
            ),
            index,
        )
        game.step(ticks=1)
        snapshots.append(game.snapshot())
    with env.device_context():
        env.batch.restore(snapshots)
        env._refresh_growth_bounds(range(3))
        env.features.initialize_home(range(3))
        env.features.encode()
    try:
        yield env
    finally:
        env.close()


@torch.no_grad()
def test_compiled_inference_buckets_and_owned_outputs(collection_env):
    torch._dynamo.reset()
    env = collection_env
    policy = TwoStageTransformerPolicy(env.cfg).cuda().eval()
    eager = TwoStageTransformerPolicy(env.cfg).cuda().eval()
    eager.load_state_dict(policy.state_dict())
    policy.species_encoder.enable_compilation()
    owned = original = None
    for index, count in enumerate((5, 40, 100, 256, 5, 5)):
        batch = collate_observations([observation(env.cfg, count)[0]], "cuda")
        expected = eager.species_values(batch)
        saved = expected.clone()
        with torch.inference_mode(index == 0), torch.no_grad():
            compiled = policy.species_values(batch)
        if index == 0:
            owned, original = compiled, compiled.clone()
        torch.testing.assert_close(compiled, saved, rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(expected, saved, rtol=0, atol=0)
    torch.testing.assert_close(owned, original, rtol=0, atol=0)
    assert policy.species_encoder._compiled_shapes == {}
    assert policy.species_encoder.collection_metrics["entries"] == 5
    assert policy.species_encoder.collection_metrics["hits"] == 1
    policy.species_encoder.release_collection()
    assert policy.species_encoder.collection_metrics["entries"] == 0
    assert policy.species_encoder.compilation_status == "compiled", (
        policy.species_encoder.compilation_error
    )


@pytest.mark.parametrize("different", [None, "entity", "globals", "version", "terminal", "count"])
def test_stateless_endpoint_dedup_uses_exact_public_inputs_and_ema_version(different):
    entities = np.tile([[[16, 17, 0, -500, 0, 0, 0, 0, 0, 0, 0]]], (3, 2, 1)).astype(np.int32)
    globals_ = np.zeros((3, 18), np.float32)
    counts = np.ones(3, np.int64)
    versions = np.ones(3, np.int64)
    done = np.zeros(3, bool)
    if different == "entity":
        entities[1, 0, 3] += 1
    elif different == "globals":
        globals_[1, 0] = np.nextafter(np.float32(0), np.float32(1))
    elif different == "version":
        versions[1] += 1
    elif different == "terminal":
        done[1] = True
    elif different == "count":
        counts[1] += 1
    entities[2, 1, 3] += 1
    actual = endpoint_representatives(entities, globals_, counts, versions, done)
    np.testing.assert_array_equal(actual, [0, 0 if different is None else 1, 0])


@torch.no_grad()
@pytest.mark.parametrize("count", [1, 4, 44])
def test_tile_schedule_shapes_distinct_candidates_and_inactive_rows(collection_env, count):
    from pvz_rl.policy.sequential_q import action_parts

    env = collection_env
    env.cfg["training"]["objective"]["tile_probes"] = count
    with env.device_context():
        collector = ProbePool(env)
        actions = torch.full((3,), 41, device="cuda", dtype=torch.long)
        eligible = torch.tensor([True, False, True], device="cuda")
        rng = torch.cuda.get_rng_state().clone()
        proposals, valid = collector.schedule(actions, env.action_masks(), eligible)
        assert proposals.shape == (3, count)
        assert not valid[1].any()
        branch, tile = action_parts(proposals)
        assert (branch[valid] == 1).all() and (tile[valid] != 40).all()
        for source in (0, 2):
            selected = tile[source][valid[source]]
            assert len(selected.unique()) == len(selected)
        collector.tile_cursor[0, 0] += int(valid[0].sum())
        next_proposals, _ = collector.schedule(actions, env.action_masks(), eligible)
        if count == 4:
            assert not torch.equal(next_proposals[0], proposals[0])
        torch.testing.assert_close(rng, torch.cuda.get_rng_state(), atol=0, rtol=0)
        restored = ProbePool(env)
        restored.restore(collector.snapshot())
        expected = collector.schedule(actions, env.action_masks(), eligible)
        actual = restored.schedule(actions, env.action_masks(), eligible)
        for expected_values, actual_values in zip(expected, actual, strict=True):
            torch.testing.assert_close(expected_values, actual_values)


@torch.no_grad()
def test_tile_schedule_limited_candidates_and_nonplant_exclusion(collection_env):
    env = collection_env
    with env.device_context():
        collector = ProbePool(env)
        masks = env.action_masks().clone()
        masks[:, 1:361] = False
        masks[:, [41, 21, 22]] = True
        actions = torch.tensor([41, 0, 361], device="cuda")
        proposals, valid = collector.schedule(
            actions, masks, torch.ones(3, dtype=torch.bool, device="cuda")
        )
        assert valid[0].tolist() == [True, True, False, False]
        assert proposals[0, :2].tolist() == [21, 22]
        assert not valid[1:].any()
        assert not collector.plant_cursor.any()


@torch.no_grad()
def test_inference_compilation_failure_is_warned_once_and_keeps_eager_math(collection_env):
    torch._dynamo.reset()
    env = collection_env
    policy = TwoStageTransformerPolicy(env.cfg).cuda().eval()
    expected = policy.species_values(env.features.obs_tensor).clone()
    calls = []

    def failed(self, *args):
        calls.append(True)
        raise RuntimeError("forced inference compiler failure")

    policy.species_encoder.enable_compilation()
    with pytest.MonkeyPatch.context() as patch:
        from pvz_rl.policy.cudagraph_backend import GraphCapture

        patch.setattr(GraphCapture, "__call__", failed)
        with pytest.warns(RuntimeWarning, match="using eager execution") as captured:
            first = policy.species_values(env.features.obs_tensor)
            second = policy.species_values(env.features.obs_tensor)
    assert len(captured) == 1 and len(calls) == 1
    assert policy.species_encoder.compilation_status == "fallback"
    torch.testing.assert_close(first, expected, rtol=0, atol=0)
    torch.testing.assert_close(second, expected, rtol=0, atol=0)
