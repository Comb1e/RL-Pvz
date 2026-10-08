"""Batched scratch isolation, device deduplication and collection boundaries."""

import copy

import numpy as np
import pytest
import torch
from pvz_game import Game, InitialPlant, LevelSpec, Spawn

from pvz_rl.config import load_config
from pvz_rl.envs.cuda_env import CudaVecEnv
from pvz_rl.envs.encoding import collate_observations
from pvz_rl.envs.probes import CounterfactualCollector
from pvz_rl.monitoring.entity_benchmark import observation
from pvz_rl.policy.runner import PolicyRunner
from pvz_rl.policy.transformer_lstm import EventMemoryState, TransformerLSTMPolicy

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
    details = dict(tile_features=output.tile_features, context=output.context)
    return policy, runner, details


@torch.no_grad()
def test_serial_batched_rng_rewards_histories_and_memory_fallback(collection_env, monkeypatch):
    env = collection_env
    with env.device_context():
        policy, memory, details = inputs(env)
        memory_before = memory.snapshot()
        source_before = [env.batch.snapshot(index) for index in range(3)]
        ledger_before = env.features.home_ledger.get()
        collector = CounterfactualCollector(env)
        schedule = collector.snapshot()
        actions = torch.tensor([46, 1, 361], device="cuda")
        active = torch.ones(3, device="cuda", dtype=torch.bool)
        collector.schedule(policy, actions, details, env.action_masks(), active)
        collector.restore(schedule)
        host = collector.collect(
            policy, memory, env.features.obs_tensor, actions, details, env.action_masks(), active
        )
        env.handoff.wait()
        batched, boards = collector.materialize(host)
        batched = batched.copy()
        boards = copy.deepcopy(boards)
        memory_after = memory.snapshot()
        final_headers = collector.batch.header[3:].get()
        final_ledger = collector.features.home_ledger[3:].get()
        allocate = CounterfactualCollector._allocate

        def constrained(instance):
            if instance.lanes == 2:
                raise torch.cuda.OutOfMemoryError("forced scratch pressure")
            allocate(instance)

        monkeypatch.setattr(CounterfactualCollector, "_allocate", constrained)
        with pytest.warns(RuntimeWarning, match="all scheduled probes"):
            serial = CounterfactualCollector(env)
        memory.restore(memory_before)
        serial.restore(schedule)
        host = serial.collect(
            policy, memory, env.features.obs_tensor, actions, details, env.action_masks(), active
        )
        env.handoff.wait()
        reference, reference_boards = serial.materialize(host)
        for name in batched.dtype.names:
            if name == "bootstrap":
                np.testing.assert_allclose(batched[name], reference[name], rtol=2e-5, atol=2e-6)
            else:
                np.testing.assert_array_equal(batched[name], reference[name])
        for name in ("entities", "offsets", "counts", "globals"):
            np.testing.assert_array_equal(getattr(boards, name), getattr(reference_boards, name))
        for slot in range(4):
            independent = []
            for game_index in range(3):
                offset, count = boards.offsets[game_index, slot], boards.counts[game_index, slot]
                independent.append(
                    dict(
                        entities=boards.entities[offset : offset + count],
                        globals=boards.globals[game_index, slot],
                    )
                )
            next_output = policy.forward_step(
                collate_observations(independent, "cuda"),
                EventMemoryState(
                    *(
                        memory_after[key].cuda()
                        for key in ("hidden", "cell", "pending", "elapsed_ticks")
                    )
                ),
                events=torch.as_tensor(batched["events"][:, slot].copy(), device="cuda"),
                memory_write=torch.as_tensor(
                    batched["memory_write"][:, slot].copy(), device="cuda"
                ),
            )
            targets = (
                next_output.branch_q.max(-1).values.cpu().numpy()
                * env.cfg["training"]["gamma"] ** batched["duration"][:, slot]
            )
            np.testing.assert_allclose(
                batched["bootstrap"][:, slot],
                np.where(batched["done"][:, slot], 0, targets),
                rtol=2e-5,
                atol=2e-6,
            )
        np.testing.assert_array_equal(final_headers, serial.batch.header.get())
        np.testing.assert_array_equal(final_ledger, serial.features.home_ledger.get())
        for key in memory_after:
            if isinstance(memory_after[key], torch.Tensor):
                torch.testing.assert_close(
                    memory_after[key], memory.snapshot()[key], rtol=0, atol=0
                )
            else:
                assert memory_after[key] == memory.snapshot()[key]
        np.testing.assert_array_equal(ledger_before, env.features.home_ledger.get())
        assert source_before == [env.batch.snapshot(index) for index in range(3)]


@pytest.mark.parametrize("different", [None, "entity", "globals", "gross", "timing"])
def test_device_dedup_counterexamples(collection_env, different):
    env = collection_env
    with env.device_context():
        collector = CounterfactualCollector(env)
        features = collector.features
        features.entity_tensor[:] = env.features.entity_tensor.repeat(4, 1, 1)
        features.globals_tensor[:] = env.features.globals_tensor.repeat(4, 1)
        features.summary_tensor[:] = env.features.summary_tensor.repeat(4, 1)
        events = torch.zeros(12, 8, dtype=torch.long, device="cuda")
        valid = torch.ones(12, dtype=torch.bool, device="cuda")
        if different == "entity":
            features.entity_tensor[3:6, 0, 3] += 1
        elif different == "globals":
            features.globals_tensor[3:6, 0] += 1
        elif different is not None:
            events[3:6, 0 if different == "gross" else 7] += 1
        features.dedup(events, valid, collector.source, 4)
        result = collector.source.cpu().reshape(4, 3)
        assert result[0].tolist() == [0, 0, 0]
        assert result[1].tolist() == ([0, 0, 0] if different is None else [1, 1, 1])
        assert result[2:].tolist() == [[0, 0, 0], [0, 0, 0]]


@torch.no_grad()
def test_compiled_inference_buckets_and_owned_outputs(collection_env):
    torch._dynamo.reset()
    env = collection_env
    policy = TransformerLSTMPolicy(env.cfg).cuda().eval()
    eager = TransformerLSTMPolicy(env.cfg).cuda().eval()
    eager.load_state_dict(policy.state_dict())
    policy.entity.enable_compilation()
    for count in (5, 40, 100, 256, 5):
        batch = collate_observations([observation(env.cfg, count)[0]], "cuda")
        expected = eager.forward_step(batch)
        saved = expected.branch_q.clone()
        compiled = policy.forward_step(batch)
        torch.testing.assert_close(compiled.branch_q, saved, rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(expected.branch_q, saved, rtol=0, atol=0)
    assert len(policy.entity._compiled_shapes) <= 4
    assert policy.entity.compilation_status == "compiled", policy.entity.compilation_error


@torch.no_grad()
def test_probe_schedule_disabled_roles_interval_and_inactive_rows(collection_env):
    env = collection_env
    env.cfg["training"]["objective"].update(
        branch_probes=0, tile_probes=1, probe_interval_decisions=2
    )
    with env.device_context():
        policy, _, details = inputs(env)
        collector = CounterfactualCollector(env)
        actions = torch.ones(3, device="cuda", dtype=torch.long)
        active = torch.tensor([True, False, True], device="cuda")
        rng = torch.cuda.get_rng_state().clone()
        _, valid = collector.schedule(policy, actions, details, env.action_masks(), active)
        assert valid.cpu().tolist() == [
            [False, False, True, False],
            [False] * 4,
            [False, False, True, False],
        ]
        _, valid = collector.schedule(policy, actions, details, env.action_masks(), active)
        assert not valid.any()
        assert collector.decision_cursor.cpu().tolist() == [2, 0, 2]
        torch.testing.assert_close(rng, torch.cuda.get_rng_state())
        assert collector.branch_cursor.cpu().tolist() == [0, 0, 0]


@torch.no_grad()
def test_inference_compilation_failure_is_warned_once_and_keeps_eager_math(collection_env):
    torch._dynamo.reset()
    env = collection_env
    policy = TransformerLSTMPolicy(env.cfg).cuda().eval()
    expected = policy.forward_step(env.features.obs_tensor).branch_q.clone()
    calls = []

    def failed(*args):
        calls.append(True)
        raise RuntimeError("forced inference compiler failure")

    policy.entity._compiled_encode = failed
    policy.entity.compilation_status = "requested"
    with pytest.warns(RuntimeWarning, match="using eager execution") as captured:
        first = policy.forward_step(env.features.obs_tensor)
        second = policy.forward_step(env.features.obs_tensor)
    assert len(captured) == 1 and len(calls) == 1
    assert policy.entity.compilation_status == "fallback"
    torch.testing.assert_close(first.branch_q, expected, rtol=0, atol=0)
    torch.testing.assert_close(second.branch_q, expected, rtol=0, atol=0)
