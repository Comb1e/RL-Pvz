"""Serial backend, bounded consequences, compaction and transactional replay controls."""

import copy
import signal
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from pvz_game import InitialPlant, LevelSpec, Spawn
from pvz_game.cuda.schema import REASONS
from torch import nn

from pvz_rl.config import load_config
from pvz_rl.envs.action_timing import ActionPhaseGame
from pvz_rl.envs.cuda_accounting import AccountingCudaBatch
from pvz_rl.envs.cuda_features import CudaFeatures
from pvz_rl.envs.encoding import observations_equal
from pvz_rl.envs.probes import ProbePool, ProbeStop, cpu_probe
from pvz_rl.envs.rewards import HomeProximityLedger
from pvz_rl.learning.host_transfer import HostHandoff
from pvz_rl.learning.objective import components
from pvz_rl.monitoring.cuda_diagnostics import DeviceProfiler
from pvz_rl.policy.runner import PolicyRunner
from pvz_rl.policy.sequential_q import ActionQValues, select_q_actions
from pvz_rl.policy.transformer_lstm import EventMemoryState, RecurrentOutput


class FrozenControl(nn.Module):
    def __init__(self, cfg, *, dig=False):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()), requires_grad=False)
        self.entity = SimpleNamespace(width=1, microbatch=128, token_budget=65536)
        self.hidden_size = 2
        self.dig = dig
        self.cfg = cfg

    def initial_state(self, batch_size, *, device=None):
        return EventMemoryState(
            torch.zeros(1, batch_size, 2, device=device),
            torch.zeros(1, batch_size, 2, device=device),
            torch.zeros(batch_size, 7, dtype=torch.long, device=device),
            torch.zeros(batch_size, dtype=torch.long, device=device),
        )

    def forward_step(self, observations, state=None, *, events=None, memory_write=None):
        if state is None:
            state = self.initial_state(len(observations), device=observations.device)
        events = state.inputs() if events is None else events
        memory_write = events[:, :7].ne(0).any(-1) if memory_write is None else memory_write
        increment = events[:, :7].sum(-1)[None, :, None] * memory_write[None, :, None]
        next_state = state.consume(state.hidden + increment, state.cell + increment, memory_write)
        return RecurrentOutput(
            torch.full((len(observations), 1), 0.5, device=observations.device),
            torch.zeros(len(observations), 45, 1, device=observations.device),
            torch.zeros(len(observations), 2, device=observations.device),
            next_state,
        )

    def action_values(self, output, masks=None, *, observations=None, plan=None):
        from pvz_rl.policy.sequential_q import observation_tile_masks

        masks = observation_tile_masks(observations) if masks is None else masks
        tiles = torch.full((len(output.wait_q), 9, 45), -1.0, device=output.wait_q.device)
        if self.dig:
            tiles[:, 8] = 2
        return ActionQValues.derive(output.wait_q, tiles, masks)

    def decide(self, observations, state=None):
        from pvz_rl.policy.sequential_q import observation_tile_masks

        output = self.forward_step(observations, state)
        masks = observation_tile_masks(observations)
        actions, _, _, details = select_q_actions(
            self.action_values(output, masks), masks, deterministic=True
        )
        return actions, output.state, details


def probe_fixture(case, cfg):
    plants = ()
    initial_sun = 100
    spawns = (Spawn(100000, "basic", 4),)
    if case == "sunflower":
        plants = (InitialPlant("sunflower", 0, 0),)
    elif case == "mine":
        plants = (InitialPlant("potato_mine", 0, 0),)
        spawns = (Spawn(1600, "basic", 0, x=100), *spawns)
    elif case == "shooting":
        plants = (InitialPlant("peashooter", 0, 0),)
        spawns = (Spawn(500, "basic", 0, x=2800), *spawns)
    elif case == "terminal":
        spawns = ()
    elif case == "cutoff":
        cfg["environment"]["cutoff_seconds"] = 1
    elif case == "cap":
        cfg["training"]["objective"]["probe_rollout_max_decisions"] = 2
    elif case == "rejection":
        initial_sun = 50
    game = ActionPhaseGame()
    game.reset(LevelSpec(case, spawns, initial_sun=initial_sun, plants=plants), 101)
    snapshot = game.snapshot()
    snapshot["sky_due"] = 99999
    if case == "sunflower":
        snapshot["plants"][0]["due"] = 600
    if case == "mine":
        snapshot["plants"][0]["due"] = 1500
    game.restore(snapshot)
    return game


def service(pool, buffer):
    pool.advance()
    pool.env.handoff.wait()
    pool.consume_transfers(buffer)
    pool.finish_slice()


@pytest.mark.parametrize(
    "case", ["sunflower", "mine", "shooting", "terminal", "cutoff", "cap", "rejection"]
)
def test_cpu_cuda_background_endpoints_match_independent_serial(case, monkeypatch, tmp_path):
    from pvz_rl.learning.cuda_buffer import CompleteGameBuffer

    cfg = load_config()
    cfg["training"]["objective"]["tile_probes"] = 1
    cfg["training"]["performance"].update(probe_active_capacity=1, probe_pending_capacity=1)
    game = probe_fixture(case, cfg)
    cpu_teacher = FrozenControl(cfg, dig=case == "cap")
    proposal = 1 if case == "cap" else 46 if case == "rejection" else 0
    ledger = HomeProximityLedger(game.observe(), game.rules)
    before = copy.deepcopy(game.snapshot())
    expected = cpu_probe(game, ledger, proposal, cfg, cpu_teacher)
    assert game.snapshot() == before
    if case == "sunflower":
        assert expected["components"]["produced_sun"] >= 25
        assert expected["components"]["early_sun"] >= 25
    if case in ("mine", "shooting"):
        assert expected["components"]["combat_value"] > 0
    if case in ("sunflower", "mine", "shooting", "rejection"):
        assert expected["stop_reason"] == ProbeStop.HORIZON and expected["duration"] == 3000
    elif case == "cap":
        assert expected["stop_reason"] == ProbeStop.DECISION_CAP
        assert expected["transitions"] == 2 and expected["duration"] == 0
    elif case == "cutoff":
        assert expected["stop_reason"] == ProbeStop.CUTOFF and expected["tick"] == 100
    else:
        assert expected["stop_reason"] == ProbeStop.TERMINAL and expected["won"]
    teacher = FrozenControl(cfg, dig=case == "cap").cuda()
    batch = AccountingCudaBatch(1, zombie_capacity=2, max_step_ticks=1)
    buffer = CompleteGameBuffer(tmp_path / "buffer", 1, cfg=cfg)
    try:
        with batch.cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
            batch.restore([before])
            features = CudaFeatures(batch, cfg, "masked")
            obs = features.encode()
            source_count = features.summary_host[:, 0].copy()
            features.initialize_home([0])
            env = SimpleNamespace(
                batch=batch,
                features=features,
                cfg=cfg,
                condition="masked",
                enabled_envs=np.ones(1, dtype=bool),
                profiler=DeviceProfiler(batch.cp),
                handoff=HostHandoff(torch.cuda.current_stream()),
            )
            pool = ProbePool(env)
            monkeypatch.setattr(
                pool,
                "schedule",
                lambda *args: (
                    torch.tensor([[proposal]], device="cuda"),
                    torch.ones(1, 1, dtype=torch.bool, device="cuda"),
                ),
            )
            memory = PolicyRunner(teacher, cfg, batch.rules, 1, "cuda")
            memory.advance(obs, torch.ones(1, dtype=torch.bool, device="cuda"))
            rng = torch.cuda.get_rng_state().clone()
            executing = pool.reserve(
                teacher,
                memory.state,
                torch.tensor([45], device="cuda"),
                features.mask_tensor,
                torch.ones(1, dtype=torch.bool, device="cuda"),
                torch.zeros(1, dtype=torch.long, device="cuda"),
            )
            assert executing.all()
            features.step(batch.cp.asarray([45], batch.cp.int64))
            source_hash = batch.state_hash(0)
            env.handoff.wait()
            expected_count = pool.commit(
                np.ones(1, bool), np.ones(1, bool), np.array([45]), source_count, np.array([0])
            )
            rows = np.zeros(1, buffer.dtype)
            rows["active"] = True
            rows["probe_expected"] = rows["probe_pending"] = expected_count
            buffer.append(rows, [features.encoder.encode(game.observe())])
            for _ in range(cfg["training"]["objective"]["probe_rollout_max_decisions"] + 4):
                if not pool.has_work:
                    break
                service(pool, buffer)
            assert not pool.has_work and buffer.pending_probes == 0
            actual = buffer.take([0])["probes"][0, 0]
            for name in (
                "duration",
                "tick",
                "initial_duration",
                "accepted",
                "executed_action",
                "done",
                "won",
                "transitions",
                "event_writes",
                "stop_reason",
            ):
                assert actual[name] == expected[name], name
            assert actual["reason"] == (
                0 if expected["accepted"] else REASONS.index(expected["reason"])
            )
            np.testing.assert_allclose(
                actual["components"], components(expected["components"]), atol=2e-12
            )
            assert actual["bootstrap"] == expected["bootstrap"]
            np.testing.assert_array_equal(actual["events"], expected["events"])
            endpoint = buffer.observations(np.array([actual]))[0]
            assert observations_equal(
                dict(entities=endpoint.entities.numpy(), globals=endpoint.globals.numpy()),
                expected["observation"],
            )
            assert source_hash == batch.state_hash(0)
            torch.testing.assert_close(rng, torch.cuda.get_rng_state(), atol=0, rtol=0)
            assert not memory.state.hidden.any() and not memory.state.pending.any()
            pool.release_workspaces()
    finally:
        buffer.close()


def make_model(tmp_path, *, count=3, sources=2, lanes=4, cap=12, cutoff=600):
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import configure

    from pvz_rl.learning.training import build_model, vector_env

    class Continue(BaseCallback):
        def _on_step(self):
            return True

    cfg = load_config()
    cfg["training"].update(n_envs=count, batch_size=16)
    cfg["policy"].update(
        chunk_length=8,
        entity_width=8,
        scalar_width=8,
        transformer_heads=2,
        transformer_layers=1,
        transformer_feedforward=16,
        lstm_hidden=8,
        event_width=4,
    )
    cfg["training"]["performance"].update(
        compile_kernels=False,
        fit_precision="fp32",
        prefetch=False,
        probe_active_capacity=lanes,
        probe_pending_capacity=sources,
    )
    cfg["training"]["objective"]["probe_rollout_max_decisions"] = cap
    cfg["environment"]["cutoff_seconds"] = cutoff
    cfg["training"]["exploration"].update(plant_epsilon_start=0.01, plant_epsilon_floor=0.01)
    cfg["visualization"].update(enabled=False, live_enabled=False)
    env = vector_env(cfg, "masked", 101)
    model = build_model(cfg, "masked", env, 101)
    model.trajectory_root = tmp_path
    model.set_logger(configure(folder=None, format_strings=[]))
    callback = Continue()
    callback.init_callback(model)
    with torch.no_grad():
        model.policy.wait_head[-1].weight.zero_()
        model.policy.wait_head[-1].bias.zero_()
        model.policy.tile_head[-1].weight.zero_()
        model.policy.tile_head[-1].bias.zero_()
        model.policy.tile_offsets.zero_()
        model.policy.tile_offsets[0] = 1
    model._begin(callback)
    with env.device_context():
        env.batch.cooldowns[:, 0] = 0
        env.header_tensor[:, 2] = 200
        env.features.encode()
        model._last_obs = env.features.obs_tensor
    return model, env, callback


def close_model(model, env):
    model._drain()
    if model._buffer:
        model._buffer.close()
    env.close()


def test_real_and_planting_slots_continue_while_auxiliary_games_run(tmp_path):
    model, env, callback = make_model(tmp_path, count=2, sources=2, cap=20)
    try:
        model._collect_step(callback)
        assert model._buffer.pending_probes == 8 and len(model._probes.jobs) == 2
        first_ticks = env.header_tensor[:, 0].clone()
        for _ in range(5):
            model._collect_step(callback)
        assert model.num_timesteps == 12
        assert (env.header_tensor[:, 0] > first_ticks).all()
        assert model._buffer.pending_probes == 8
        assert not model._scheduler.pending.any()
        assert len(model._probes.lanes) == 4
        assert model.collection_execution["active_auxiliary_lanes"] == 4
        with pytest.raises(RuntimeError, match="Pending probe"):
            model._buffer.finalize()
        before = torch.cuda.get_rng_state().clone()
        model._drain()
        torch.testing.assert_close(torch.cuda.get_rng_state(), before, atol=0, rtol=0)
        assert not model._probes.has_work
        assert not model._buffer.take(model._buffer.valid_indices())["probe_pending"].any()
    finally:
        close_model(model, env)


def test_capacity_pauses_only_eligible_slots_and_preserves_fifo_tickets(tmp_path):
    model, env, callback = make_model(tmp_path, sources=1, lanes=1, cap=3)
    try:
        before = env.header_tensor[:, 0].clone()
        model._collect_step(callback)
        assert model._scheduler.pending.tolist() == [False, True, True]
        tickets = model._scheduler.columns.clone()
        states = model._memory.state.clone()
        for _ in range(2):
            model._collect_step(callback)
        assert model._scheduler.pending.tolist() == [False, True, True]
        torch.testing.assert_close(model._scheduler.columns[1:], tickets[1:], atol=0, rtol=0)
        torch.testing.assert_close(
            model._memory.state.hidden[:, 1:], states.hidden[:, 1:], atol=0, rtol=0
        )
        torch.testing.assert_close(env.header_tensor[1:, 0], before[1:], atol=0, rtol=0)
        assert env.header_tensor[0, 0] > before[0]
        rng = torch.cuda.get_rng_state().clone()
        model._drain()
        torch.testing.assert_close(torch.cuda.get_rng_state(), rng, atol=0, rtol=0)
        actual = model._buffer.take(model._buffer.valid_indices())
        plants = actual[(actual["action"] > 0) & (actual["action"] < 361) & actual["accepted"]]
        assert plants["env"].tolist() == [0, 1, 2]
        for slot in (1, 2):
            first = actual[actual["env"] == slot][0]
            assert first["action"] == tickets[slot, 10]
            assert first["tick"] == 0 and first["duration"] == 0 and first["reset"]
            assert first["tile_coin"] == tickets[slot, 15]
        assert len(plants) == 3 and plants["probes"]["valid"].sum() == 12
        assert model._probes.plant_cursor.cpu().tolist() == [1, 1, 1]
    finally:
        close_model(model, env)


@pytest.mark.parametrize("interrupt_at", ["before", "auxiliary"])
def test_interruption_drains_then_fresh_reload_continues_exactly(
    tmp_path, monkeypatch, interrupt_at
):
    from pvz_rl.learning.cohort import atomic_transition
    from pvz_rl.learning.training import load_policy, vector_env

    model, env, callback = make_model(tmp_path, count=2, sources=1, lanes=1, cap=2)
    path = tmp_path / "interrupted.zip"
    try:
        hashes = [env.batch.state_hash(index) for index in range(2)]
        if interrupt_at == "before":
            with pytest.raises(KeyboardInterrupt), atomic_transition():
                signal.raise_signal(signal.SIGINT)
                model._collect_step(callback)
            assert model._buffer.size == 0
            assert hashes == [env.batch.state_hash(index) for index in range(2)]
            model._collect_step(callback)
        else:
            model._collect_step(callback)
            advance = model._probes.advance
            fired = []

            def interrupted():
                advance()
                if not fired:
                    fired.append(True)
                    signal.raise_signal(signal.SIGINT)

            with monkeypatch.context() as patch:
                patch.setattr(model._probes, "advance", interrupted)
                with pytest.raises(KeyboardInterrupt), atomic_transition():
                    model._collect_step(callback)
        model.save(path)
        assert not model._probes.has_work and not model._scheduler.pending.any()
        assert model._buffer.pending_probes == 0
        model._collect_step(callback)
        model._drain()
        expected = model._buffer.take(np.arange(model._buffer.size))
        memory = model._memory.snapshot()
        cursor = model._probes.snapshot()
        cfg = copy.deepcopy(model.cfg)
    finally:
        close_model(model, env)
    restored, _ = load_policy(path, "cuda")
    resumed_env = vector_env(cfg, "masked", 101)
    try:
        restored.set_env(resumed_env)
        restored._restore_runtime()
        callback.init_callback(restored)
        restored._collect_step(callback)
        restored._drain()
        np.testing.assert_array_equal(
            restored._buffer.take(np.arange(restored._buffer.size)), expected
        )
        for key, value in memory.items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(restored._memory.snapshot()[key], value, atol=0, rtol=0)
        for key in ("tile_cursor", "plant_cursor"):
            torch.testing.assert_close(restored._probes.snapshot()[key], cursor[key])
    finally:
        close_model(restored, resumed_env)


def test_multiple_zero_time_jobs_same_slot_and_no_recursive_triggering(tmp_path):
    model, env, callback = make_model(tmp_path, count=1, sources=4, lanes=4, cap=4)
    try:
        for _ in range(3):
            with env.device_context():
                env.batch.cooldowns[:, 0] = 0
                env.features.encode()
                model._last_obs = env.features.obs_tensor
            model._collect_step(callback)
        rows = model._buffer.take(model._buffer.valid_indices())
        assert rows["accepted"].all() and not rows["duration"].any()
        assert model._probes.plant_cursor.item() == 3 and len(model._probes.jobs) == 3
        model._drain()
        rows = model._buffer.take(model._buffer.valid_indices())
        assert rows["probes"]["valid"].sum() == 12
        assert model._probes.plant_cursor.item() == 3
        assert model.num_timesteps == 3
    finally:
        close_model(model, env)


def test_allocation_fallback_keeps_all_alternatives(tmp_path, monkeypatch):
    import pvz_rl.envs.probes as implementation

    model, env, callback = make_model(tmp_path, count=1, sources=2, lanes=4, cap=2)
    original = implementation.AccountingCudaBatch
    allocations = []

    def allocate(count, **kwargs):
        allocations.append(count)
        if count > 1:
            raise MemoryError("controlled pool pressure")
        return original(count, **kwargs)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(implementation, "AccountingCudaBatch", allocate)
            with pytest.warns(RuntimeWarning, match="allocation fallback"):
                model._collect_step(callback)
        assert model._probes.capacity == model._probes.source_capacity == 1
        model._drain()
        rows = model._buffer.take(model._buffer.valid_indices())
        assert rows["probes"]["valid"].sum() == 4
        assert allocations[0] == 2 and allocations[-1] == 1
    finally:
        close_model(model, env)


def test_minimum_pool_failure_preserves_real_board_rng_and_memory(tmp_path, monkeypatch):
    model, env, callback = make_model(tmp_path, count=1, sources=1, lanes=1, cap=2)

    def unavailable(*args, **kwargs):
        raise MemoryError("controlled minimum allocation failure")

    try:
        before = env.batch.state_hash(0)
        rng = torch.cuda.get_rng_state().clone()
        memory = model._memory.snapshot()
        monkeypatch.setattr("pvz_rl.envs.probes.AccountingCudaBatch", unavailable)
        with pytest.raises(MemoryError, match="minimum isolated"):
            model._collect_step(callback)
        assert before == env.batch.state_hash(0) and model._buffer.size == model.num_timesteps == 0
        torch.testing.assert_close(torch.cuda.get_rng_state(), rng, atol=0, rtol=0)
        for name in ("hidden", "cell", "pending", "elapsed_ticks"):
            torch.testing.assert_close(model._memory.snapshot()[name], memory[name], atol=0, rtol=0)
    finally:
        close_model(model, env)


def test_queue_age_excludes_running_jobs(monkeypatch):
    from collections import deque

    monkeypatch.setattr("pvz_rl.envs.probes.perf_counter", lambda: 100.0)
    pool = SimpleNamespace(
        lanes=[(0, 0)],
        queue=deque([(1, 0), (1, 1)]),
        jobs={0: SimpleNamespace(created_at=10.0), 1: SimpleNamespace(created_at=80.0)},
        capacity=4,
        source_capacity=2,
        workspace_bytes=32,
    )
    metrics = ProbePool.telemetry(pool)
    assert metrics["oldest_queue_seconds"] == 20.0
    assert metrics["oldest_pending_job_seconds"] == 90.0
    pool.queue.clear()
    assert ProbePool.telemetry(pool)["oldest_queue_seconds"] == 0.0


def test_auxiliary_batching_does_not_change_real_actions_rewards_memory_or_rng(tmp_path):
    traces = []
    for lanes in (1, 4):
        model, env, callback = make_model(tmp_path, count=3, sources=16, lanes=lanes, cap=3)
        try:
            for _ in range(8):
                model._collect_step(callback)
            model._drain()
            records = model._buffer.take(model._buffer.valid_indices())
            traces.append(
                (
                    records,
                    model._memory.snapshot(),
                    torch.cuda.get_rng_state().clone(),
                    [env.batch.state_hash(index) for index in range(3)],
                )
            )
        finally:
            close_model(model, env)
    first, second = traces
    for name in (
        "env",
        "action",
        "accepted",
        "reward",
        "duration",
        "events",
        "memory_write",
        "exploration_mode",
        "species_coin",
        "tile_coin",
        "pending_species",
        "next_species",
    ):
        np.testing.assert_array_equal(first[0][name], second[0][name])
    for name in ("hidden", "cell", "pending", "elapsed_ticks"):
        torch.testing.assert_close(first[1][name], second[1][name], atol=0, rtol=0)
    torch.testing.assert_close(first[2], second[2], atol=0, rtol=0)
    assert first[3] == second[3]
    for name in ("components", "duration", "transitions", "events", "accepted", "bootstrap"):
        np.testing.assert_allclose(first[0]["probes"][name], second[0]["probes"][name], atol=2e-6)


def test_noncontiguous_slots_simultaneous_finishes_and_exploratory_sources(tmp_path):
    from pvz_rl.learning.cohort import CohortPhase

    model, env, callback = make_model(tmp_path, count=128, sources=3, lanes=4, cap=3, cutoff=1)
    slots = np.array([5, 17, 90])
    try:
        env.enabled_envs[:] = False
        env.enabled_envs[slots] = True
        with env.device_context():
            env.header_tensor[:, 17].copy_(torch.as_tensor(env.enabled_envs, device="cuda").long())
            env.header_tensor[slots, 0] = 99
            env.features.encode()
            model._last_obs = env.features.obs_tensor
        model._plant_exploration.pending[slots] = 1
        initial = model._memory.state.hidden.clone()
        model._collect_step(callback)
        first = model._buffer.take(model._buffer.valid_indices())
        assert first["env"].tolist() == slots.tolist()
        assert first["exploration_mode"].tolist() == [2, 2, 2]
        assert first["probe_pending"].tolist() == [4, 4, 4]
        model._collect_step(callback)
        assert not env.enabled_envs.any() and model._probes.has_work
        model._drain()
        assert model.phase == CohortPhase.FINALIZE_REWARDS
        assert model.num_timesteps == 6 and model._buffer.size == 256
        rows = model._buffer.take(model._buffer.valid_indices())
        assert rows[rows["done"]]["env"].tolist() == slots.tolist()
        assert rows["probes"]["valid"].sum() == 12
        inactive = np.ones(128, bool)
        inactive[slots] = False
        torch.testing.assert_close(
            model._memory.state.hidden[:, inactive], initial[:, inactive], atol=0, rtol=0
        )
        assert set(model._pending_episodes) == set(slots.tolist())
    finally:
        close_model(model, env)


def test_interrupt_during_drain_finishes_evidence_before_rethrow(tmp_path, monkeypatch):
    model, env, callback = make_model(tmp_path, count=1, sources=1, lanes=1, cap=2)
    try:
        model._collect_step(callback)
        advance = model._probes.advance
        fired = []

        def interrupted():
            advance()
            if not fired:
                fired.append(True)
                signal.raise_signal(signal.SIGINT)

        with monkeypatch.context() as patch:
            patch.setattr(model._probes, "advance", interrupted)
            with pytest.raises(KeyboardInterrupt):
                model._drain()
        assert model._buffer.pending_probes == 0 and not model._probes.has_work
        assert model.num_timesteps == 1
    finally:
        close_model(model, env)
