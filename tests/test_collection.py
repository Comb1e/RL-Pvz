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
        bootstrap = collector.bootstrap(host)
        env.handoff.wait()
        batched, boards = collector.materialize(host)
        batched["bootstrap"] = bootstrap.numpy().reshape(4, 3).T
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
        bootstrap = serial.bootstrap(host)
        env.handoff.wait()
        reference, reference_boards = serial.materialize(host)
        reference["bootstrap"] = bootstrap.numpy().reshape(4, 3).T
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
    assert policy.entity._compiled_shapes == {}
    assert policy.entity.collection_metrics["entries"] == 4
    assert policy.entity.collection_metrics["hits"] == 1
    policy.entity.release_collection()
    assert policy.entity.collection_metrics["entries"] == 0
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
    torch.testing.assert_close(first.branch_q, expected, rtol=0, atol=0)
    torch.testing.assert_close(second.branch_q, expected, rtol=0, atol=0)


@torch.no_grad()
def test_valid_probe_copy_dedup_evidence_and_no_sync_staging(collection_env, monkeypatch):
    from pvz_rl.policy.active_batch import ActiveBatchPlan

    env = collection_env
    with env.device_context():
        policy, memory, details = inputs(env)
        collector = CounterfactualCollector(env)
        active_host = np.array([True, False, True])
        active = torch.as_tensor(active_host, device="cuda")
        plan = ActiveBatchPlan.from_counts(
            active_host, env.features.summary_host[:, 0], env.features.limit
        )
        proposals = torch.tensor([[0, 46, 362, 91]] * 3, device="cuda")
        valid = torch.tensor(
            [[True, True, False, False], [False] * 4, [True, False, True, False]],
            device="cuda",
        )
        monkeypatch.setattr(collector, "schedule", lambda *args, **kwargs: (proposals, valid))
        copied = []
        copy_kernel = collector.copy_kernel

        def copy_checked(grid, block, arguments):
            count = grid[0]
            before = [value[:count].copy() for value in arguments[10:17]]
            copy_kernel(grid, block, arguments)
            copied.append(
                (arguments[-1].copy(), before, [value[:count].copy() for value in arguments[10:17]])
            )

        monkeypatch.setattr(collector, "copy_kernel", copy_checked)
        calls = []
        forward = policy.forward_step

        def observed_forward(observations, state, **kwargs):
            calls.append(observations.clone())
            return forward(observations, state, **kwargs)

        monkeypatch.setattr(policy, "forward_step", observed_forward)

        def forbidden(*args, **kwargs):
            raise AssertionError("Probe staging introduced a device/host synchronization")

        with monkeypatch.context() as guarded:
            guarded.setattr(torch.Tensor, "cpu", forbidden)
            guarded.setattr(torch.Tensor, "item", forbidden)
            guarded.setattr(torch.Tensor, "nonzero", forbidden)
            guarded.setattr(torch, "nonzero", forbidden)
            guarded.setattr(env.handoff, "wait", forbidden)
            host = collector.collect(
                policy,
                memory,
                env.features.obs_tensor,
                torch.zeros(3, dtype=torch.long, device="cuda"),
                details,
                env.action_masks(),
                active,
                plan=plan,
            )
        assert [batch.entities.shape[:2] for batch in calls] == [(1, 32), (1, 128)]
        env.handoff.wait()
        assert len(host["entities"]) == 4 * (13 + 73)
        np.testing.assert_array_equal(host["active_ids"], [0, 2])
        assert host["globals"].shape == (8, 18)
        for selected, before, after in copied:
            inactive = ~selected.get()
            for previous, current in zip(before, after, strict=True):
                np.testing.assert_array_equal(previous.get()[inactive], current.get()[inactive])
        records, boards = collector.materialize(host)
        np.testing.assert_array_equal(records["valid"], valid.cpu().numpy())
        assert not records["bootstrap"].any()
        assert records[0, 0]["accepted"] and not records[0, 1]["accepted"]
        assert records[0, 1]["action"] == 46 and records[0, 1]["executed_action"] == 0
        assert records[0, 0]["components"][2] == 0
        assert records[0, 1]["components"][2] == -env.cfg["reward"]["invalid_plant_penalty"]
        assert records[2, 0]["components"][3] == 0
        assert records[2, 2]["components"][3] == -env.cfg["reward"]["empty_dig_penalty"]
        assert boards.offsets[0, 0] == boards.offsets[0, 1]
        assert boards.offsets[2, 0] == boards.offsets[2, 2]
        inactive = ~records["valid"]
        assert not host["metadata"].numpy().reshape(4, 3, 17).transpose(1, 0, 2)[inactive].any()
        assert not boards.counts[inactive].any() and not boards.globals[inactive].any()
        assert not records["events"][inactive].any() and not records["memory_write"][inactive].any()
        pending = collector._pending_bootstrap
        expected = (
            forward(
                collector.features.view(0, 12, collector.features.limit),
                pending["state"],
                events=pending["events"],
                memory_write=pending["memory_write"],
            )
            .branch_q.max(-1)
            .values
        )
        expected *= env.cfg["training"]["gamma"] ** pending["duration"]
        expected = torch.where(pending["valid"] & ~pending["done"], expected, 0)
        with monkeypatch.context() as guarded:
            guarded.setattr(torch.Tensor, "cpu", forbidden)
            guarded.setattr(torch.Tensor, "item", forbidden)
            guarded.setattr(torch.Tensor, "nonzero", forbidden)
            guarded.setattr(torch, "nonzero", forbidden)
            guarded.setattr(env.handoff, "wait", forbidden)
            bootstrap = collector.bootstrap(host)
        assert collector.inference_metrics == dict(
            logical_rows=4,
            unique_rows=2,
            padded_rows=2,
            bucket_widths=(32, 128),
            scratch_rows=8,
            scratch_capacity_rows=12,
        )
        assert [batch.entities.shape[:2] for batch in calls[2:]] == [(1, 32), (1, 128)]
        env.handoff.wait()
        torch.testing.assert_close(bootstrap, expected.cpu(), rtol=2e-5, atol=2e-6)
        assert not memory.state.pending[0].any()
        assert memory.state.pending[1, 0] == 50
        assert memory.state.pending[2, 6] == 0


@pytest.mark.parametrize("terminal", ["none", "representative", "all"])
@torch.no_grad()
def test_probe_bootstrap_terminal_aliases_public_widths_and_discount(
    collection_env, monkeypatch, terminal
):
    env = collection_env
    env.cfg["training"]["gamma"] = 0.5
    with env.device_context():
        policy, memory, details = inputs(env)
        collector = CounterfactualCollector(env)
        active = torch.ones(3, dtype=torch.bool, device="cuda")
        proposals = torch.tensor([[0, 46, 91, 362]] * 3, device="cuda")
        monkeypatch.setattr(
            collector, "schedule", lambda *args: (proposals, active[:, None].expand(-1, 4))
        )
        host = collector.collect(
            policy,
            memory,
            env.features.obs_tensor,
            torch.zeros(3, dtype=torch.long, device="cuda"),
            details,
            env.action_masks(),
            active,
        )
        env.handoff.wait()
        np.testing.assert_array_equal(host["source"].numpy(), np.zeros(12, dtype=np.int64))
        np.testing.assert_array_equal(
            collector._pending_bootstrap["entity_bounds"], np.tile([13, 43, 73], 4)
        )
        if terminal != "none":
            selected = slice(0, 3) if terminal == "representative" else slice(None)
            host["metadata"][selected, 8] = 1
            collector._pending_bootstrap["done"][selected] = True
        forward = policy.forward_step
        pending = collector._pending_bootstrap
        expected = (
            forward(
                collector.features.view(0, 12, collector.features.limit),
                pending["state"],
                events=pending["events"],
                memory_write=pending["memory_write"],
            )
            .branch_q.max(-1)
            .values
            * 0.5
        )
        expected = torch.where(pending["done"], 0, expected)
        calls = []

        def observed_forward(observations, state, **kwargs):
            calls.append(observations.clone())
            return forward(observations, state, **kwargs)

        monkeypatch.setattr(policy, "forward_step", observed_forward)
        if terminal == "none":
            bounds = pending["entity_bounds"].copy()
            pending["entity_bounds"][0] = 0
            with pytest.raises(RuntimeError, match="public-count bound"):
                collector.bootstrap(host)
            pending["entity_bounds"] = bounds
            assert not calls
        env._pending_spawns[:] = 0
        env._plant_counts[:] = 0
        bootstrap = collector.bootstrap(host)
        env.handoff.wait()
        assert [batch.entities.shape[:2] for batch in calls] == (
            [] if terminal == "all" else [(1, 32), (1, 64), (1, 128)]
        )
        assert (
            collector.inference_metrics["logical_rows"]
            == {"none": 12, "representative": 9, "all": 0}[terminal]
        )
        assert collector.inference_metrics["unique_rows"] == (0 if terminal == "all" else 3)
        torch.testing.assert_close(bootstrap, expected.cpu(), rtol=2e-5, atol=2e-6)
        assert not host["metadata"][:, 16].any()


@torch.no_grad()
def test_probe_workspace_release_preserves_schedule_and_lazy_fallback(collection_env, monkeypatch):
    env = collection_env
    with env.device_context():
        policy, memory, details = inputs(env)
        collector = CounterfactualCollector(env)
        collector.branch_cursor[:] = 7
        collector.tile_cursor[:] = 3
        collector.decision_cursor[:] = 5
        schedule = collector.snapshot()
        collector.release_workspaces()
        collector.release_workspaces()
        assert all(
            getattr(collector, name) is None
            for name in ("batch", "features", "packed", "metadata", "source", "slots", "games")
        )
        collector.restore(schedule)
        for name in ("branch_cursor", "tile_cursor", "decision_cursor"):
            torch.testing.assert_close(
                getattr(collector, name).cpu(), schedule[name], rtol=0, atol=0
            )
        allocate = CounterfactualCollector._allocate

        def constrained(instance):
            if instance.lanes == 2:
                raise torch.cuda.OutOfMemoryError("forced reallocation pressure")
            allocate(instance)

        monkeypatch.setattr(CounterfactualCollector, "_allocate", constrained)
        with pytest.warns(RuntimeWarning, match="all scheduled probes"):
            host = collector.collect(
                policy,
                memory,
                env.features.obs_tensor,
                torch.ones(3, dtype=torch.long, device="cuda"),
                details,
                env.action_masks(),
                torch.ones(3, dtype=torch.bool, device="cuda"),
            )
        with pytest.raises(RuntimeError, match="pending probe bootstrap"):
            collector.release_workspaces()
        env.handoff.wait()
        bootstrap = collector.bootstrap(host)
        env.handoff.wait()
        records, _ = collector.materialize(host)
        records["bootstrap"] = bootstrap.numpy().reshape(4, 3).T
        assert records["valid"].all() and collector.lanes == 1
        assert collector.decision_cursor.cpu().tolist() == [6, 6, 6]
        assert collector.branch_cursor.cpu().tolist() == [9, 9, 9]
        released_schedule = collector.snapshot()
        collector.release_workspaces()
        collector.restore(released_schedule)
        assert collector.decision_cursor.cpu().tolist() == [6, 6, 6]
        assert collector.features is None


@torch.no_grad()
def test_zero_scheduled_probes_skip_bootstrap_and_clear_all_evidence(collection_env, monkeypatch):
    env = collection_env
    env.cfg["training"]["objective"].update(branch_probes=0, tile_probes=0)
    with env.device_context():
        policy, memory, details = inputs(env)
        collector = CounterfactualCollector(env)
        arguments = (
            policy,
            memory,
            env.features.obs_tensor,
            torch.zeros(3, dtype=torch.long, device="cuda"),
            details,
            env.action_masks(),
            torch.ones(3, dtype=torch.bool, device="cuda"),
        )
        host = collector.collect(*arguments)
        with pytest.raises(RuntimeError, match="previous probe bootstrap"):
            collector.collect(*arguments)
        env.handoff.wait()

        def forbidden(*args, **kwargs):
            raise AssertionError("Empty probe scheduling must not run bootstrap inference")

        monkeypatch.setattr(policy, "forward_step", forbidden)
        bootstrap = collector.bootstrap(host)
        env.handoff.wait()
        records, boards = collector.materialize(host)
        assert not any(records[name].any() for name in records.dtype.names)
        assert not bootstrap.any()
        assert not boards.entities.size and not boards.counts.any() and not boards.globals.any()
        assert collector.inference_metrics == dict(
            logical_rows=0,
            unique_rows=0,
            padded_rows=0,
            bucket_widths=(),
            scratch_rows=12,
            scratch_capacity_rows=12,
        )
        assert not collector.features.entity_mask_tensor.any()
        np.testing.assert_array_equal(host["source"].numpy(), -np.ones(12, dtype=np.int64))
        with pytest.raises(RuntimeError, match="current staged probe handoff"):
            collector.bootstrap(host)
        collector.release_workspaces()


def test_compact_probe_scratch_views_contract(collection_env):
    env = collection_env
    with env.device_context():
        collector = CounterfactualCollector(env)
        owner = collector.batch
        batch = owner.compact_view(3)
        assert batch.n == 3 and owner.n == 6
        assert batch._step.func is type(batch)._step_with_accounting
        assert batch._step.args[0].accounting is batch.accounting
        assert batch._step.args[0].n == batch.n
        assert batch._step_kernel is owner._step_kernel and batch.module is owner.module
        assert batch.sun_spawn_fraction == owner.sun_spawn_fraction
        for name in (
            "header",
            "plants",
            "zombies",
            "projectiles",
            "mowers",
            "cooldowns",
            "_schedules",
            "_events",
            "_event_counts",
            "facts",
            "masks",
            "accounting",
        ):
            selected, original = getattr(batch, name), getattr(owner, name)
            assert selected.shape[0] == 3 and original.shape[0] == 6
            assert selected.data.ptr == original.data.ptr
        nested = batch.compact_view(1)
        assert nested.n == 1 and nested._step.args[0].accounting is nested.accounting
        assert nested._step.args[0].n == nested.n
        for invalid in (0, -1, 7, True, 2.0):
            with pytest.raises(ValueError, match="Compact batch size"):
                owner.compact_view(invalid)
        with pytest.raises(ValueError, match="Compact batch size"):
            batch.compact_view(4)
        features = collector.features.compact_view(batch)
        assert features.batch is batch and collector.features.batch is owner
        for name in (
            "entities",
            "entity_mask",
            "globals",
            "summary",
            "history",
            "entity_tensor",
            "entity_mask_tensor",
            "globals_tensor",
            "summary_tensor",
            "history_tensor",
        ):
            assert getattr(features, name) is getattr(collector.features, name)
        for name in (
            "records",
            "truncation_counts",
            "home_ledger",
            "home_entries",
            "assets",
            "rewards",
            "parts",
            "totals",
            "before_header",
            "before_cooldowns",
            "before_assets",
            "enabled",
            "reward_tensor",
            "mask_tensor",
        ):
            assert getattr(features, name).shape[0] == 3
            assert getattr(collector.features, name).shape[0] == 6
        assert len(features.entities) == 12
        with pytest.raises(ValueError, match="Compact feature batch"):
            features.compact_view(owner)


@pytest.mark.parametrize(
    "active_host",
    [[False, True, True], [False, False, True], [False, False, False]],
    ids=["noncontiguous-original-rows", "one-active", "zero-active"],
)
@torch.no_grad()
def test_compact_probe_scratch_noncontiguous_serial_equivalence(
    collection_env, monkeypatch, active_host
):
    from pvz_rl.policy.active_batch import ActiveBatchPlan

    env = collection_env
    active_host = np.asarray(active_host, dtype=bool)
    active_ids = np.flatnonzero(active_host)
    count = len(active_ids)
    with env.device_context():
        policy, memory, details = inputs(env)
        initial_memory = memory.snapshot()
        source_before = [env.batch.snapshot(index) for index in range(3)]
        ledger_before = env.features.home_ledger.get()
        rng = torch.cuda.get_rng_state().clone()
        plan = ActiveBatchPlan.from_counts(
            active_host, env.features.summary_host[:, 0], env.features.limit
        )
        collector = CounterfactualCollector(env)
        schedule = collector.snapshot()
        collector.metadata.fill_(41)
        collector.features.entity_mask_tensor.fill_(True)
        collector.features.globals_tensor.fill_(43)
        collector.features.summary_tensor.fill_(45)
        collector.features.history_tensor.fill_(47)
        collector.batch.accounting.fill(49)
        collector.batch.facts.fill(51)
        collector.batch._event_counts.fill(53)
        launches = []
        step_kernel = collector.batch._step_kernel

        def checked_step(grid, block, arguments):
            launches.append((grid, arguments[11], arguments[10].shape, arguments[14].shape))
            step_kernel(grid, block, arguments)

        monkeypatch.setattr(collector.batch, "_step_kernel", checked_step)
        batch_views = []
        compact_batch = collector.batch.compact_view

        def checked_batch(size):
            view = compact_batch(size)
            batch_views.append(view)
            assert view._step.func is type(view)._step_with_accounting
            assert view._step.args[0].accounting is view.accounting
            assert view._step.args[0].n == view.n
            return view

        monkeypatch.setattr(collector.batch, "compact_view", checked_batch)
        feature_views = []
        compact_features = collector.features.compact_view

        def checked_features(batch):
            view = compact_features(batch)
            feature_views.append(view)
            assert view.entity_tensor is collector.features.entity_tensor
            assert view.globals_tensor is collector.features.globals_tensor
            return view

        monkeypatch.setattr(collector.features, "compact_view", checked_features)
        arguments = (
            policy,
            memory,
            env.features.obs_tensor,
            torch.tensor([46, 1, 361], device="cuda"),
            details,
            env.action_masks(),
            torch.as_tensor(active_host, device="cuda"),
        )
        host = collector.collect(*arguments, plan=plan)
        env.handoff.wait()
        bootstrap = collector.bootstrap(host)
        env.handoff.wait()
        records, boards = collector.materialize(host)
        records["bootstrap"] = bootstrap.numpy().reshape(4, 3).T
        records = records.copy()
        boards = copy.deepcopy(boards)
        np.testing.assert_array_equal(host["active_ids"], active_ids)
        assert host["globals"].shape == (4 * count, 18)
        assert len(host["entities"]) == 4 * int(np.array([13, 43, 73])[active_host].sum())
        assert [view.n for view in batch_views] == ([2 * count] if count else [])
        assert [view.batch.n for view in feature_views] == ([2 * count] if count else [])
        assert launches == (
            [((2 * count,), 2 * count, (2 * count,), (2 * count, 11))] * 2 if count else []
        )
        assert collector.batch.n == 6 and collector.features.batch.n == 6
        assert collector.inference_metrics["scratch_rows"] == 4 * count
        assert collector.inference_metrics["scratch_capacity_rows"] == 12
        np.testing.assert_array_equal(collector.batch.accounting[2 * count :].get(), 49)
        np.testing.assert_array_equal(collector.batch.facts[2 * count :].get(), 51)
        np.testing.assert_array_equal(collector.batch._event_counts[2 * count :].get(), 53)
        assert not any(records[name][~active_host].any() for name in records.dtype.names)
        assert not boards.counts[~active_host].any() and not boards.globals[~active_host].any()
        if not count:
            assert not collector.features.summary_tensor.any()
            assert not collector.features.entity_mask_tensor.any()
            assert not collector.features.globals_tensor.any()
            assert not collector.features.history_tensor.any()
            assert collector.inference_metrics["logical_rows"] == 0
            assert collector.inference_metrics["unique_rows"] == 0
            assert collector.inference_metrics["padded_rows"] == 0
        allocate = CounterfactualCollector._allocate

        def constrained(instance):
            if instance.lanes == 2:
                raise torch.cuda.OutOfMemoryError("forced compact scratch fallback")
            allocate(instance)

        monkeypatch.setattr(CounterfactualCollector, "_allocate", constrained)
        with pytest.warns(RuntimeWarning, match="all scheduled probes"):
            serial = CounterfactualCollector(env)
        memory.restore(initial_memory)
        serial.restore(schedule)
        host = serial.collect(*arguments, plan=plan)
        env.handoff.wait()
        bootstrap = serial.bootstrap(host)
        env.handoff.wait()
        expected, expected_boards = serial.materialize(host)
        expected["bootstrap"] = bootstrap.numpy().reshape(4, 3).T
        for name in records.dtype.names:
            if name == "bootstrap":
                np.testing.assert_allclose(records[name], expected[name], rtol=2e-5, atol=2e-6)
            else:
                np.testing.assert_array_equal(records[name], expected[name])
        for name in ("entities", "offsets", "counts", "globals"):
            np.testing.assert_array_equal(getattr(boards, name), getattr(expected_boards, name))
        assert source_before == [env.batch.snapshot(index) for index in range(3)]
        np.testing.assert_array_equal(ledger_before, env.features.home_ledger.get())
        torch.testing.assert_close(rng, torch.cuda.get_rng_state(), rtol=0, atol=0)


def test_compact_probe_scratch_release_collects_bound_view_cycles(collection_env):
    import weakref

    env = collection_env
    with env.device_context():
        collector = CounterfactualCollector(env)
        collector.decision_cursor[:] = 3
        schedule = collector.snapshot()
        batch = collector.batch.compact_view(2)
        scratch_reference = weakref.ref(batch)
        owner_reference = weakref.ref(collector.batch)
        features_reference = weakref.ref(collector.features)
        del batch
        collector.release_workspaces()
        assert scratch_reference() is None
        assert owner_reference() is None
        assert features_reference() is None
        collector.restore(schedule)
        assert collector.decision_cursor.cpu().tolist() == [3, 3, 3]
        assert collector.batch is None and collector.features is None


@torch.no_grad()
def test_unique_bootstrap_respects_current_encoder_microbatch(collection_env, monkeypatch):
    env = collection_env
    with env.device_context():
        policy, memory, details = inputs(env)
        policy.entity.microbatch = 1
        collector = CounterfactualCollector(env)
        proposals = torch.zeros(3, 4, device="cuda", dtype=torch.long)
        valid = torch.ones(3, 4, device="cuda", dtype=torch.bool)
        monkeypatch.setattr(collector, "schedule", lambda *args: (proposals, valid))
        host = collector.collect(
            policy,
            memory,
            env.features.obs_tensor,
            torch.zeros(3, device="cuda", dtype=torch.long),
            details,
            env.action_masks(),
            torch.ones(3, device="cuda", dtype=torch.bool),
        )
        env.handoff.wait()
        sizes = []
        handle = policy.entity.register_forward_pre_hook(
            lambda module, args: sizes.append(len(args[0]))
        )
        try:
            bootstrap = collector.bootstrap(host)
            env.handoff.wait()
            assert sizes == [1, 1, 1]
            assert collector.inference_metrics["unique_rows"] == 3
            assert collector.inference_metrics["padded_rows"] == 3
            assert torch.isfinite(bootstrap).all()
        finally:
            handle.remove()
