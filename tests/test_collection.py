"""Batched scratch isolation, device deduplication and collection boundaries."""

import numpy as np
import pytest
import torch
from pvz_game import Game, InitialPlant, LevelSpec, Spawn

from pvz_rl.config import load_config
from pvz_rl.envs.cuda_env import CudaVecEnv
from pvz_rl.envs.encoding import collate_observations
from pvz_rl.envs.probes import ProbePool
from pvz_rl.monitoring.entity_benchmark import observation
from pvz_rl.policy.runner import PolicyRunner
from pvz_rl.policy.transformer_lstm import TransformerLSTMPolicy

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")


@pytest.fixture
def collection_env():
    cfg = load_config()
    cfg["training"]["n_envs"] = 3
    cfg["training"]["performance"].update(compile_kernels=False, telemetry=True)
    cfg["visualization"].update(enabled=False, live_enabled=False)
    cfg["policy"].update(
        entity_width=8,
        transformer_heads=2,
        transformer_layers=1,
        transformer_feedforward=16,
        scalar_width=8,
        lstm_hidden=8,
        event_width=4,
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


@pytest.mark.parametrize(
    "case, expected",
    [
        ("cap", [0, 0, 0, 0, 0, 0, 0]),
        ("gross_sun", [50, 50, 0, 0, 0, 1, 0]),
        ("spawn_remove", [0, 0, 1, 1, 1, 0, 1]),
        ("head_loss", [0, 0, 1, 1, 0, 0, 0]),
        ("plant_death", [0, 0, 0, 0, 0, 0, 1]),
        ("plant", [0, 50, 0, 0, 0, 1, 0]),
        ("dig", [0, 0, 0, 0, 0, 0, 1]),
        ("rejected_spawn", [0, 0, 1, 0, 0, 0, 0]),
    ],
)
def test_independent_cpu_cuda_public_event_accounting(case, expected):
    from pvz_game import Dig, Place, Rules, Wait

    from pvz_rl.envs.action_timing import ActionPhaseGame
    from pvz_rl.envs.actions import ActionCodec
    from pvz_rl.envs.cuda_accounting import AccountingCudaBatch
    from pvz_rl.envs.cuda_features import CudaFeatures
    from pvz_rl.envs.history import public_history_event

    action, per_tick, plants = Wait(), True, ()
    spawns = (Spawn(1000, "basic", 4),)
    if case == "gross_sun":
        action, per_tick = Place("sunflower", 0, 0), False
        plants = (InitialPlant("sunflower", 0, 1),)
    elif case == "spawn_remove":
        plants = (InitialPlant("cherry_bomb", 2, 1),)
        spawns = (Spawn(1, "basic", 2, x=1500), *spawns)
    elif case == "head_loss":
        spawns = (Spawn(1, "basic", 0, x=1000), *spawns)
    elif case == "plant_death":
        plants = (InitialPlant("sunflower", 0, 0),)
        spawns = (Spawn(1, "basic", 0, x=100), *spawns)
    elif case == "plant":
        action = Place("sunflower", 0, 0)
    elif case == "dig":
        plants, action = (InitialPlant("sunflower", 0, 0),), Dig(0, 0)
    elif case == "rejected_spawn":
        plants, action = (InitialPlant("sunflower", 0, 0),), Place("sunflower", 0, 0)
        spawns = (Spawn(1, "basic", 4), *spawns)
    rules = Rules()
    scenario = LevelSpec(case, spawns, initial_sun=100, plants=plants, mowers=False)
    cpu = ActionPhaseGame(rules)
    cpu.reset(scenario, seed=101)
    if case == "plant_death":
        cpu.step(ticks=3)
    snapshot = cpu.snapshot()
    snapshot["sky_due"] = 9999
    if case == "cap":
        snapshot["sun"], snapshot["sky_due"] = rules.game["sun_cap"], 1
    elif case == "gross_sun":
        snapshot["sky_due"] = snapshot["plants"][0]["due"] = 1
    elif case == "spawn_remove":
        snapshot["plants"][0]["due"] = 1
    elif case == "head_loss":
        snapshot["projectiles"] = [
            dict(id=1, row=0, x=1000, damage=200, icy=False, move_remainder=0)
        ]
        snapshot["next_id"] = 2
    elif case == "plant_death":
        snapshot["plants"][0]["health"] = 1
    cpu.restore(snapshot)
    duration = (
        0
        if per_tick and not isinstance(action, Wait) and cpu.validate_action(action).accepted
        else 1
    )
    result = cpu.step(action, ticks=duration)
    facts = public_history_event(result.events, rules)
    np.testing.assert_array_equal(facts.array(), expected)
    assert facts.writes == any(expected)
    batch = AccountingCudaBatch(1, zombie_capacity=2)
    with batch.cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
        batch.restore([snapshot])
        features = CudaFeatures(batch, load_config(), "masked")
        features.encode()
        proposal = ActionCodec(load_config()).encode(action)
        features.step(batch.cp.asarray([proposal], batch.cp.int64), per_tick=per_tick)
        np.testing.assert_array_equal(features.history.get()[0], expected)
        assert batch.state_hash(0) == cpu.state_hash()
        assert not batch.diagnostic


def inputs(env):
    torch.manual_seed(9)
    policy = TransformerLSTMPolicy(env.cfg).cuda().eval()
    runner = PolicyRunner(policy, env.cfg, env.batch.rules, 3, "cuda")
    runner.state.hidden.normal_()
    runner.state.cell.normal_()
    runner.state.pending[:, 0] = torch.tensor([0, 50, 0], device="cuda")
    runner.state.pending[2, 6] = 1
    runner.state.elapsed_ticks[:] = 1
    output = policy.forward_step(
        env.features.obs_tensor,
        runner.state,
    )
    details = dict(
        tile_features=output.tile_features,
        context=output.context,
        action_q=policy.action_values(output, env.action_masks()),
    )
    return policy, runner, details


@pytest.mark.parametrize(
    "different", [None, "entity", "globals", "gross", "timing", "hidden", "cell"]
)
def test_device_dedup_counterexamples(collection_env, different):
    env = collection_env
    with env.device_context():
        from types import SimpleNamespace

        from pvz_rl.envs.cuda_features import CudaFeatures
        from pvz_rl.learning.probe_layout import ProbeLayout

        layout = ProbeLayout.from_config(env.cfg)
        collector = SimpleNamespace(
            layout=layout, source=torch.empty(3 * layout.count, dtype=torch.long, device="cuda")
        )
        features = CudaFeatures(env.batch, env.cfg, env.condition, store_rows=3 * layout.count)
        features.entity_tensor[:] = env.features.entity_tensor.repeat(collector.layout.count, 1, 1)
        features.globals_tensor[:] = env.features.globals_tensor.repeat(collector.layout.count, 1)
        features.summary_tensor[:] = env.features.summary_tensor.repeat(collector.layout.count, 1)
        events = torch.zeros(3 * collector.layout.count, 8, dtype=torch.long, device="cuda")
        valid = torch.ones(3 * collector.layout.count, dtype=torch.bool, device="cuda")
        state = (
            TransformerLSTMPolicy(env.cfg)
            .cuda()
            .initial_state(3 * collector.layout.count, device="cuda")
        )
        if different == "entity":
            features.entity_tensor[3:6, 0, 3] += 1
        elif different == "globals":
            features.globals_tensor[3:6, 0] += 1
        elif different in ("hidden", "cell"):
            getattr(state, different)[:, 3:6] += 1
        elif different is not None:
            events[3:6, 0 if different == "gross" else 7] += 1
        features.dedup(events, valid, collector.source, collector.layout.count, state=state)
        result = collector.source.cpu().reshape(collector.layout.count, 3)
        assert result[0].tolist() == [0, 0, 0]
        assert result[1].tolist() == ([0, 0, 0] if different is None else [1, 1, 1])
        assert result[2:].tolist() == [[0, 0, 0]] * (collector.layout.count - 2)


@torch.no_grad()
def test_compiled_inference_buckets_and_owned_outputs(collection_env):
    torch._dynamo.reset()
    env = collection_env
    policy = TransformerLSTMPolicy(env.cfg).cuda().eval()
    eager = TransformerLSTMPolicy(env.cfg).cuda().eval()
    eager.load_state_dict(policy.state_dict())
    policy.entity.enable_compilation()
    owned = original = None
    for index, count in enumerate((5, 40, 100, 256, 5, 5)):
        batch = collate_observations([observation(env.cfg, count)[0]], "cuda")
        expected = eager.forward_step(batch)
        saved = expected.wait_q.clone()
        with torch.inference_mode(index == 0), torch.no_grad():
            compiled = policy.forward_step(batch)
        if index == 0:
            owned, original = compiled.wait_q, compiled.wait_q.clone()
        torch.testing.assert_close(compiled.wait_q, saved, rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(expected.wait_q, saved, rtol=0, atol=0)
    torch.testing.assert_close(owned, original, rtol=0, atol=0)
    assert policy.entity._compiled_shapes == {}
    assert policy.entity.collection_metrics["entries"] == 5
    assert policy.entity.collection_metrics["hits"] == 1
    policy.entity.release_collection()
    assert policy.entity.collection_metrics["entries"] == 0
    assert policy.entity.compilation_status == "compiled", policy.entity.compilation_error


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
    policy = TransformerLSTMPolicy(env.cfg).cuda().eval()
    expected = policy.forward_step(env.features.obs_tensor).wait_q.clone()
    calls = []

    def failed(self, *args):
        calls.append(True)
        raise RuntimeError("forced inference compiler failure")

    policy.entity.enable_compilation()
    with pytest.MonkeyPatch.context() as patch:
        from pvz_rl.policy.cudagraph_backend import GraphCapture

        patch.setattr(GraphCapture, "__call__", failed)
        with pytest.warns(RuntimeWarning, match="using eager execution") as captured:
            first = policy.forward_step(env.features.obs_tensor)
            second = policy.forward_step(env.features.obs_tensor)
    assert len(captured) == 1 and len(calls) == 1
    assert policy.entity.compilation_status == "fallback"
    torch.testing.assert_close(first.wait_q, expected, rtol=0, atol=0)
    torch.testing.assert_close(second.wait_q, expected, rtol=0, atol=0)
