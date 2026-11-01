"""Tensor vector environment; only compact completion information crosses to CPU."""

from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from functools import partial
from time import perf_counter

import numpy as np
import torch
from gymnasium import spaces
from pvz_game import Game, LevelSpec, Rules
from pvz_game.config import PLANT_TYPES
from pvz_game.cuda.backend import projectile_bound
from pvz_game.cuda.schema import HEADER
from stable_baselines3.common.vec_env import VecEnv

from pvz_rl.config import bomb_max_illegal_distance, training_env_capacity
from pvz_rl.envs.actions import ActionSchema as A
from pvz_rl.envs.actions import dig_branch_available
from pvz_rl.envs.cuda_accounting import AccountingCudaBatch
from pvz_rl.envs.cuda_features import LEDGER_INDICES, METRIC_INDICES, CudaFeatures
from pvz_rl.envs.rewards import REWARD_METRICS
from pvz_rl.envs.scenarios import difficulty_weights, scenario
from pvz_rl.envs.sky_sun import peashooter_sky_sun_rate
from pvz_rl.envs.step_result import PackedStepResult
from pvz_rl.envs.tile_scenarios import (
    BombGenerationFailure,
    bomb_tile_distances,
    is_placement_skill,
    peashooter_level,
    preparation_settings,
    skill_plant,
)
from pvz_rl.learning.budget import budget_target
from pvz_rl.learning.curriculum import stage_distribution, teaching_enabled
from pvz_rl.learning.host_transfer import HostHandoff
from pvz_rl.learning.training_requirements import require_supported_policy
from pvz_rl.monitoring.cuda_diagnostics import DeviceProfiler
from pvz_rl.monitoring.metrics import task_name

# Host step record: done, timed_out, ticks, reward | accepted, reason | proposal,
# executed | kept entities, omitted plants/zombies/projectiles | reward parts.
RECORD_OUTCOME, RECORD_RESULT, RECORD_ACTIONS, RECORD_SUMMARY, RECORD_PARTS = (
    slice(0, 4),
    slice(4, 6),
    slice(6, 8),
    slice(8, 12),
    slice(12, None),
)


class ScenarioQueue:
    """Bounded CPU staging. Selection observes progress at reset, never mid-game.

    At most one pending reset per lane. This avoids selecting a future task with
    stale curriculum weights and never places a seed or schedule in policy input.
    """

    def __init__(self, cfg, condition, seed, family, n):
        self.cfg, self.condition, self.family = cfg, condition, family
        self.rngs = [np.random.default_rng(seed * 1000 + i) for i in range(n)]
        self.progress, self.stage = 0, 0
        self.rules = Rules()
        self.tile_scenario = "complete"
        self.tile_boards = {}
        self.generation_failures = 0
        self.deadline = None
        self._preparation = None
        self._prefetched = None
        self.preparation_metrics = {}
        self.validation_cases = None

    def _draft(self, indices, rngs, seeds=None):
        cfg = self.cfg
        staged = []
        for offset, index in enumerate(indices):
            rng = rngs[index]
            lo, hi = cfg["splits"]["train"]
            seed = int(rng.integers(lo, hi + 1)) if seeds is None else int(seeds[offset])
            weights = difficulty_weights(
                cfg,
                self.progress,
                budget_target(cfg),
                cfg["conditions"][self.condition]["curriculum"],
            )
            level = str(rng.choice(cfg["evaluation"]["levels"], p=weights))
            family = self.family
            if family == "preset" and teaching_enabled(cfg):
                tasks, weights = stage_distribution(cfg, self.stage)
                task = str(rng.choice(tasks, p=weights))
                family, level = ("preset", task)
            spec = None
            if self.tile_scenario == "peashooter":
                level, family = "easy", "tile_peashooter"
                spec = peashooter_level(self.rules, seed)
            elif is_placement_skill(self.tile_scenario):
                level, family = "easy", "tile_" + self.tile_scenario
            else:
                spec = scenario(level, family, seed, self.rules, cfg)
            staged.append((index, level, family, seed, spec))
        return staged

    def _bomb_preparation(self):
        if self._preparation is not None and (
            (self._preparation.skill == "attacker_pre") != (self.tile_scenario == "attacker_pre")
        ):
            self.discard_prefetch()
            self._preparation.close()
            self._preparation = None
        if self._preparation is None:
            from pvz_rl.envs.bomb_preparation import BombPreparation

            self._preparation = BombPreparation(self.cfg, self.tile_scenario)
        return self._preparation

    def prefetch(self, count):
        """Speculate one CPU cohort without committing RNGs, progress or labels."""
        if not is_placement_skill(self.tile_scenario) or self._prefetched is not None:
            return
        if not preparation_settings(self.cfg, self.tile_scenario).get("prefetch", False):
            return
        if self.deadline is not None and perf_counter() >= self.deadline:
            return
        rngs = deepcopy(self.rngs)
        staged = self._draft(list(range(count)), rngs)
        tasks = self._bomb_preparation().submit(
            [s[3] for s in staged], self.deadline, skill=self.tile_scenario
        )
        self._prefetched = (staged, rngs, tasks, self.tile_scenario)

    def discard_prefetch(self):
        if self._prefetched is not None:
            self._preparation.cancel(self._prefetched[2])
            self._prefetched = None

    def prepare(self, indices, seeds=None):
        from pvz_rl.learning.deadline import check_deadline

        check_deadline(self.deadline)
        indices = list(indices)
        rngs = deepcopy(self.rngs)
        tasks = None
        prefetched = False
        if self._prefetched is not None:
            staged, saved_rngs, saved_tasks, saved_skill = self._prefetched
            if (
                is_placement_skill(self.tile_scenario)
                and self.tile_scenario == saved_skill
                and seeds is None
                and indices == [s[0] for s in staged[: len(indices)]]
            ):
                staged, tasks = staged[: len(indices)], saved_tasks[: len(indices)]
                self._preparation.cancel(saved_tasks[len(indices) :])
                for i in indices:
                    rngs[i] = saved_rngs[i]
                self._prefetched = None
                prefetched = True
            else:
                self.discard_prefetch()
        if tasks is None:
            staged = self._draft(indices, rngs, seeds)
        if is_placement_skill(self.tile_scenario):
            prep = self._bomb_preparation()
            selected_seeds = [s[3] for s in staged]
            if tasks is None:
                tasks = prep.submit(selected_seeds, self.deadline, skill=self.tile_scenario,
                                    cases=self.validation_cases)
            try:
                boards = prep.collect(
                    tasks, selected_seeds, deadline=self.deadline, skill=self.tile_scenario,
                    cases=self.validation_cases,
                )
            except BombGenerationFailure:
                self.generation_failures += self.cfg["training"]["annotation"][
                    "sampling_max_attempts"
                ]
                raise
            staged = [
                (*s[:4], LevelSpec.from_dict(board["snapshot"]["level"]))
                for s, board in zip(staged, boards, strict=True)
            ]
            check_deadline(self.deadline)
            self.tile_boards.update(zip(indices, boards, strict=True))
            self.generation_failures += sum(b["generation_failures"] for b in boards)
            self.preparation_metrics = dict(
                workers=prep.workers,
                oracle=prep.oracle,
                warmup_backend=(
                    "cuda"
                    if prep.oracle in ("cuda", "distance")
                    and preparation_settings(self.cfg, self.tile_scenario).get("cuda_warmup", False)
                    else "cpu"
                ),
                prefetched=prefetched,
                scratch_lanes=prep.cuda.batch.n if prep.cuda is not None else 0,
                host_reserved_bytes=self.reserved_bytes,
            )
        else:
            self.preparation_metrics = {}
        # Atomic consumption: unfinished preparation can be regenerated on resume.
        for i in indices:
            self.rngs[i].bit_generator.state = rngs[i].bit_generator.state
        return staged

    def release_scratch(self):
        if self._preparation is not None:
            self._preparation.release_scratch()

    def close(self):
        self.discard_prefetch()
        if self._preparation is not None:
            self._preparation.close()
            self._preparation = None

    @property
    def reserved_bytes(self):
        if self._preparation is None:
            return 0
        return self._preparation.reserved_bytes + len(self.rngs) * 65536


class CudaVecEnv(VecEnv):
    """SB3 control interface plus device-native complete-game collection.

    One compact step record per decision is the synchronization boundary; it
    also publishes the next observation's padding width. Detailed episode totals
    transfer only for completed games, as one batch. Rendering and canonical
    snapshots are explicit diagnostic operations.
    """

    def __init__(self, cfg, condition, learner_seed, family="preset", *, training=True, cases=None):
        if cfg["conditions"][condition]["hybrid"]:
            raise ValueError("Hybrid training is no longer supported; use direct CUDA training")
        if cfg["training"]["device"] != "cuda":
            raise ValueError("CUDA simulation requires --device cuda")
        require_supported_policy(cfg, condition)
        self.cfg, self.condition, self.family, self.training = cfg, condition, family, training
        self._per_tick = cfg["environment"].get("action_timing") == "per_tick"
        self._decision_ticks = int(cfg["environment"]["decision_ticks"])
        count = training_env_capacity(cfg, family) if training else cfg["training"]["n_envs"]
        self.queue = ScenarioQueue(cfg, condition, learner_seed, family, count)
        self.stream = torch.cuda.current_stream()
        self.handoff = HostHandoff(self.stream)
        self.cases = cases
        self._episode = [None] * count
        self._episode_serial = [0] * count
        self._level_names = [None] * count
        self.live_view = None
        from pvz_rl.config import output_settings
        from pvz_rl.presentation.action_journal import ActionJournal

        self.action_journal = ActionJournal(
            count,
            ram_bytes=output_settings(cfg)["visualization"]["live_history_ram_mib"] * 1024**2,
            reward_settings=cfg["reward"],
        )
        self._record = None
        self.finished_outcomes = {}
        self._episode_stages = [0] * count
        self.task_started, self.task_transitions, self.task_completed = (
            Counter(),
            Counter(),
            Counter(),
        )
        self.active_tasks, self._tasks = Counter(), {}
        self.enabled_envs = np.ones(count, dtype=bool)
        self._pending_spawns = np.zeros(count, dtype=np.int64)
        self.tile_scenario = "complete"
        self.tile_boards = {}
        self.last_preparation_seconds = 0.0
        # All shipped scenario families preserve roster size. Lessons use smaller
        # rosters. Custom batches derive their capacity from the supplied cases.
        game = Game()
        counts = []
        for level in cfg["evaluation"]["levels"]:
            game.reset(level, 0)
            counts.append(game.observe().counts.initial_total)
        if cases:
            for level, fam, seed in cases:
                game.reset(scenario(level, fam, seed, self.queue.rules, cfg), seed)
                counts.append(game.observe().counts.initial_total)
        self.batch = AccountingCudaBatch(
            count,
            zombie_capacity=max(1, *counts),
            # A mid-game recovery snapshot already owns some projectile slots;
            # the pinned backend also reserves its full reachable-shot bound for
            # future simulation.  Allocate both portions up front so resume does
            # not fail merely because the checkpoint was saved after firing.
            projectile_capacity=2 * projectile_bound(self.queue.rules),
            max_step_ticks=self._decision_ticks,
            diagnostic=False,
        )
        self.cp = self.batch.cp
        # Collection telemetry is shared by the environment and its features;
        # evaluation leaves it disabled.
        self.profiler = DeviceProfiler(
            self.cp,
            training and cfg["training"].get("performance", {}).get("telemetry", False),
            span_limit=cfg["training"]["performance"].get("telemetry_span_limit"),
        )
        with self.device_context():
            self.features = CudaFeatures(self.batch, cfg, condition)
            self.features.profiler = self.profiler
            self.header_tensor = torch.from_dlpack(self.batch.header)
        self.render_mode = None
        super().__init__(self.batch.n, self.features.encoder.space, spaces.Discrete(A.size))
        self._branch_costs = torch.tensor(
            [self.batch.rules.plants[kind]["cost"] for kind in cfg["environment"]["plants"]],
            device=torch.device("cuda"),
            dtype=self.header_tensor.dtype,
        )
        self._card_bits = 1 << torch.arange(
            len(self._branch_costs), device="cuda", dtype=torch.int64
        )
        self._closed = False
        self.transition_ticks = torch.zeros(self.num_envs, dtype=torch.int64, device="cuda")
        # Keep proposal and execution channels separate.  Trajectories and
        # journals retain the selected proposal even when the simulator rejects
        # it and advances a one-tick wait; policy history consumes execution.
        self.proposed_actions = torch.zeros(self.num_envs, dtype=torch.long, device="cuda")
        self.executed_actions = torch.zeros(self.num_envs, dtype=torch.long, device="cuda")

    @contextmanager
    def device_context(self):
        with (
            torch.cuda.stream(self.stream),
            self.cp.cuda.ExternalStream(self.stream.cuda_stream, device_id=self.batch.device),
        ):
            yield

    def reset_indices(self, indices, cases=None, *, tile_seeds=None):
        started = perf_counter()
        if self.live_view is not None:
            try:
                self.live_view.drain(wait=True)
            except Exception as exc:
                self.live_view.fail(exc)
        self.enabled_envs[indices] = True
        if cases is None:
            staged = self.queue.prepare(indices, seeds=tile_seeds)
        else:
            staged = [
                (i, level, family, seed, scenario(level, family, seed, self.queue.rules, self.cfg))
                for i, (level, family, seed) in zip(indices, cases)
            ]
        # Branch restrictions are intentionally absent from policy masks and
        # simulator setup.  Invalid plant proposals are retained as actions
        # and execute as wait with zero rejection reward penalties.
        for index, level, family, seed, spec in staged:
            self.finished_outcomes.pop(index, None)
            task = task_name(level, family)
            if index in self._tasks:
                self.active_tasks[self._tasks[index]] -= 1
            self._tasks[index] = task
            self.active_tasks[task] += 1
            self.task_started[task] += 1
            self._episode[index] = (level, family, seed)
            self._episode_serial[index] += 1
            self.action_journal.reset(index, self._episode_serial[index])
            self._level_names[index] = spec if isinstance(spec, str) else spec.name
            self._episode_stages[index] = self.queue.stage
        if self.live_view is not None:
            self.live_view.reset_episodes(indices)
        self.batch.set_sky_sun_rate(
            peashooter_sky_sun_rate(self.cfg) if self.tile_scenario == "peashooter" else 1.0
        )
        if is_placement_skill(self.tile_scenario):
            self.tile_boards = deepcopy(self.queue.tile_boards)
            from pvz_rl.envs.cuda_snapshots import upload_snapshots

            upload_snapshots(
                self.batch,
                [self.tile_boards[i]["snapshot"] for i in indices],
                indices=indices,
            )
        else:
            from pvz_rl.envs.cuda_snapshots import reset_owned
            reset = partial(reset_owned, self.batch) if self.tile_scenario == "peashooter" else self.batch.reset
            reset(
                [s[4] for s in staged],
                [s[3] for s in staged],
                indices=indices,
            )
        self._refresh_growth_bounds(indices)
        self.features.initialize_home(indices)
        self.proposed_actions[indices] = 0
        self.executed_actions[indices] = 0
        ix = self.cp.asarray(indices)
        self.features.totals[ix] = 0
        self.features.totals[ix, 8] = self.batch.header[ix, 2]
        self.features.totals[ix, 11] = -1
        self.features.encode()
        self.last_preparation_seconds = perf_counter() - started
        self.profiler.host("scenario_preparation", self.last_preparation_seconds)

    def supplied_branches(self, skill, availability):
        """Public exercise instruction, independently supplied for each board."""
        if skill == "attacker_pre":
            branches = torch.tensor([
                self.cfg["environment"]["plants"].index(board["supplied_plant"]) + 1
                if (board := self.tile_boards.get(i)) is not None else 0
                for i in range(self.num_envs)
            ], device=availability.device, dtype=torch.long)
        else:
            branches = torch.full((self.num_envs,),
                self.cfg["environment"]["plants"].index(skill_plant(skill)) + 1,
                device=availability.device, dtype=torch.long)
        usable = availability.gather(1, branches[:, None]).squeeze(1)
        return torch.where(usable, branches, 0)

    def reset(self):
        for index, seed in enumerate(self._seeds):
            if seed is not None and self.training:
                self.queue.rngs[index] = np.random.default_rng(seed)
        with self.device_context():
            self.reset_indices(list(range(self.num_envs)), self.cases)
        self._reset_seeds()
        return self.features.obs_tensor

    def reset_cohort(self, count):
        """Initialize only the remaining requested games; other slots stay inactive."""
        for index, seed in enumerate(self._seeds):
            if seed is not None and self.training:
                self.queue.rngs[index] = np.random.default_rng(seed)
        with self.device_context():
            unused = [i for i in range(count, self.num_envs) if self._episode[i] is None]
            if unused:
                # Initialize private storage for atomic snapshots, without scheduling,
                # observing, counting or advancing these inactive slots.
                self.batch.reset(["easy"] * len(unused), [0] * len(unused), indices=unused)
            self.enabled_envs[:] = False
            self.batch.header[:, 17] = 0
            self.reset_indices(list(range(count)))
        self._reset_seeds()
        return self.features.obs_tensor

    def action_masks(self):
        return self.features.mask_tensor

    def branch_availability(self, active=None):
        """Return pre-action card availability and empty-board Dig rejection.

        Policy masks intentionally expose all tile outputs.  This separate
        tensor lets the runner skip the conditional tile encoder when a plant
        card cannot execute because it is absent, cooling down, unaffordable,
        or task-restricted. Dig requires permission and at least one plant;
        on populated boards its selected tile is checked after inference.
        """
        header = self.header_tensor
        running = (header[:, HEADER.index("status")] == 0) & header[
            :, HEADER.index("enabled")
        ].bool()
        sun = header[:, HEADER.index("sun")]
        cooldowns = torch.from_dlpack(self.batch.cooldowns)
        device = self.header_tensor.device
        costs = self._branch_costs
        allowed = (
            header[:, HEADER.index("allowed")][:, None].bitwise_and(self._card_bits[None, :]) != 0
        )
        plants = (cooldowns == 0) & (sun[:, None] >= costs[None, :]) & allowed
        digging = running & dig_branch_available(
            header[:, HEADER.index("np")], header[:, HEADER.index("dig")].bool()
        )
        result = torch.cat((running[:, None], plants & running[:, None], digging[:, None]), dim=1)
        if active is not None:
            active = torch.as_tensor(active, device=device, dtype=torch.bool).reshape(-1)
            if active.shape != (self.num_envs,):
                raise ValueError("active must match the CUDA environment size")
            result &= active[:, None]
        return result

    def _refresh_growth_bounds(self, indices):
        header = self.batch.header.get()
        self._update_public_counts(header, indices)

    def _update_public_counts(self, header, indices):
        self._pending_spawns[indices] = (
            header[indices, HEADER.index("total_spawns")]
            - header[indices, HEADER.index("spawn_index")]
        )

    def step_device(self, actions, *, active=None):
        """Queue one decision's simulation, encoding and host record without waiting.

        Callers add their own copies to :attr:`handoff`, wait once, then call
        :meth:`step_host`. The record also carries the next observation's kept
        counts, so publishing its padding width needs no extra device read.
        """
        viewer = self.live_view
        if viewer is not None:
            presentation_started = perf_counter()
            try:
                if not viewer.prepared:
                    viewer.prepare()
                viewer.prepared = False
            except Exception as exc:
                viewer.fail(exc)
            self.profiler.host("presentation", perf_counter() - presentation_started)
        self._step_active = self.enabled_envs.copy()
        self._device_active = (
            torch.as_tensor(self.enabled_envs, device=actions.device) if active is None else active
        )
        self.header_tensor[:, 17].copy_(self._device_active.long())
        _, reward = self.features.step(
            self.cp.from_dlpack(actions.detach().contiguous()),
            ticks=self._decision_ticks,
            per_tick=self._per_tick,
            publish=False,
            observation_active=self.cp.asarray(self.enabled_envs),
        )
        self._device_active = torch.from_dlpack(self.features.enabled)
        h = self.header_tensor
        self.proposed_actions.copy_(actions)
        self.transition_ticks.copy_(h[:, 14])
        # Preserve the concrete proposal only when the simulator accepted it.
        # In fixed-cadence mode accepted actions advance the decision interval;
        # in action-phase mode they advance zero ticks. Both use the same
        # acceptance bit, while rejected proposals execute as waits.
        self.executed_actions.copy_(torch.where(self._device_active & h[:, 12].bool(), actions, 0))
        done = (
            (h[:, 1] != 0)
            | (
                h[:, 0]
                >= self.cfg["environment"]["cutoff_seconds"] * self.batch.rules.game["tick_rate"]
            )
        ) & self._device_active
        timed_out = done & (h[:, 1] == 0)
        with self.profiler.track("transfer"):
            record = torch.cat(
                (
                    torch.stack((done, timed_out), 1).double(),
                    h[:, 14:15].double(),
                    reward[:, None].double(),
                    h[:, 12:14].double(),
                    torch.stack((self.proposed_actions, self.executed_actions), 1).double(),
                    self.features.summary_tensor[: self.num_envs].double(),
                    torch.from_dlpack(self.features.parts),
                ),
                dim=1,
            )
            self._record = self.handoff.enqueue("transition", record)
            self._step_state_host = self.handoff.enqueue_fields(
                "step_state",
                dict(
                    executing=self._device_active,
                    pending_spawns=h[:, HEADER.index("total_spawns")]
                    - h[:, HEADER.index("spawn_index")],
                ),
            )
        self._step_device_result = reward, done, timed_out

    def step_host(self, *, autoreset=True, active=None, packed=False):
        """Interpret the completed step record after the handoff wait."""
        host = self._record.numpy()
        executed = self._step_state_host["executing"].numpy()
        self._step_active = executed.copy()
        if active is not None and not np.array_equal(np.asarray(active, dtype=bool), executed):
            raise ValueError("Host execution flags disagree with the completed simulator step")
        if self.training:
            self.task_transitions.update(self._tasks[i] for i in np.flatnonzero(self._step_active))
        self._pending_spawns[:] = self._step_state_host["pending_spawns"].numpy()
        compact = host[:, RECORD_OUTCOME]
        if is_placement_skill(self.tile_scenario):
            # A supervised exercise ends after its one proposal. Its success
            # is its private target label, not a whole-game outcome.
            compact[self._step_active, 0] = 1
            compact[self._step_active, 1] = 0
        self.last_transition_host = compact
        self.last_action_result_host = host[:, RECORD_RESULT]
        actions = host[:, RECORD_ACTIONS]
        self.last_reward_parts_host = host[:, RECORD_PARTS]
        if not np.isfinite(self.last_reward_parts_host).all():
            raise RuntimeError("Invalid CUDA reward accounting or exhausted proximity ledger")
        summary = host[:, RECORD_SUMMARY].astype(np.int64)
        # Every game active during this step keeps its complete terminal record.
        obs = self.features.publish(summary, self._step_active | self.enabled_envs)
        truncation = summary[:, 1:]
        completed = {}
        step = PackedStepResult(
            compact, actions, self.last_action_result_host, truncation, self._step_active, completed
        )
        reward, done, timed_out = self._step_device_result
        if is_placement_skill(self.tile_scenario):
            done = self._device_active.clone()
            timed_out = torch.zeros_like(done)
        reward = reward.clone()
        indices = np.flatnonzero(compact[:, 0]).tolist()
        terminal_observations = None
        if indices:
            terminal_observations = None if packed else obs.clone()
            selected = torch.as_tensor(indices, device=self.header_tensor.device)
            self._episode_host = self.handoff.enqueue_fields(
                "completed_episode",
                dict(
                    header=self.header_tensor.index_select(0, selected),
                    totals=torch.from_dlpack(self.features.totals).index_select(0, selected),
                ),
            )
            self.handoff.wait()
            headers = self._episode_host["header"].numpy()
            totals = self._episode_host["totals"].numpy()
            for index, header, total in zip(indices, headers, totals):
                self.task_completed[self._tasks[index]] += 1
                completed[index] = dict(
                    episode_metrics=self.episode_metrics(index, header, total),
                    **{"TimeLimit.truncated": bool(compact[index, 1])},
                )
                if is_placement_skill(self.tile_scenario):
                    action = int(actions[index, 0])
                    branch = 1 + (action - 1) // 45 if action else 0
                    board = self.tile_boards[index]
                    expected = self.cfg["environment"]["plants"].index(skill_plant(self.tile_scenario, board)) + 1
                    settings = self.cfg["training"]["tile_scenarios"]
                    success = bool(
                        self.last_action_result_host[index, 0]
                        and branch == expected
                        and (action - 1) % 45 in board["feasible_tiles"]
                    )
                    metrics = completed[index]["episode_metrics"]
                    metrics.update(
                        engine_status=metrics["status"],
                        status="won" if success else "lost",
                        win=int(success),
                        tile_scenario=self.tile_scenario,
                        selected_tile_distance=bomb_tile_distances(
                            board["feasible_tiles"],
                            board["absolute_infeasible_tiles"]
                            if self.tile_scenario == "attacker_pre" or bomb_max_illegal_distance(settings) else [],
                            rows=self.cfg["environment"]["rows"],
                            cols=self.cfg["environment"]["cols"],
                        )[(action - 1) % 45] if action > 0 and branch == expected else None,
                    )
                    if self.tile_scenario == "attacker_pre":
                        metrics.update(
                            target_correct=success, fallback=board["fallback"],
                            supplied_plant=board["supplied_plant"], zombie_count=board["zombie_count"],
                            actual_kill=None,
                        )
                self.finished_outcomes[index] = completed[index]["episode_metrics"]["status"]
        # Supervised outcomes are authoritative only after attribution above.
        # Capture before autoreset, including all one-shot cohort results.
        viewer = self.live_view
        if viewer is not None and viewer.enabled:
            presentation_started = perf_counter()
            try:
                viewer.after_step(compact)
            except Exception as exc:
                viewer.fail(exc)
            self.profiler.host("presentation", perf_counter() - presentation_started)
        if indices:
            if autoreset:
                self.reset_indices(indices)
                obs = self.features.obs_tensor
            else:
                self.batch.header[self.cp.asarray(indices), 17] = 0
                self.enabled_envs[indices] = False
        return (
            obs,
            reward,
            done,
            timed_out,
            terminal_observations,
            (step if packed else list(step.infos())),
        )

    def step_tensors(self, actions, *, autoreset=True):
        with self.device_context():
            self.step_device(actions)
            self.handoff.wait()
            return self.step_host(autoreset=autoreset)

    def snapshot_training(self):
        """Complete private simulator/queue state, never part of model inputs."""
        header = self.batch.header.get()
        return {
            "games": [self.batch.snapshot(i) for i in range(self.num_envs)],
            "allowed": header[:, 15].tolist(),
            "digging": header[:, 16].astype(bool).tolist(),
            "enabled": header[:, 17].tolist(),
            "totals": self.features.totals.get(),
            "home_ledger": self.features.home_ledger.get(),
            "queue_rng": [deepcopy(r.bit_generator.state) for r in self.queue.rngs],
            "queue_progress": self.queue.progress,
            "queue_stage": self.queue.stage,
            "tile_scenario": self.tile_scenario,
            "sky_sun_rate": self.batch.sky_sun_rate,
            "tile_boards": deepcopy(
                {
                    i: dict(board, snapshot=dict(board["snapshot"]))
                    for i, board in self.tile_boards.items()
                }
            ),
            "tile_generation_failures": self.queue.generation_failures,
            "tile_preparation_metrics": deepcopy(self.queue.preparation_metrics),
            "attributes": {
                k: deepcopy(getattr(self, k))
                for k in (
                    "_episode",
                    "_episode_serial",
                    "_level_names",
                    "_episode_stages",
                    "_tasks",
                    "task_started",
                    "task_transitions",
                    "task_completed",
                    "active_tasks",
                    "finished_outcomes",
                    "last_preparation_seconds",
                )
            },
        }

    def restore_training(self, state):
        if state.get("tile_scenario") == "attacker_pre":
            from pvz_rl.envs.attacker_scenarios import label_attacker

            for board in state.get("tile_boards", {}).values():
                if label_attacker(board, self.queue.rules) != board:
                    raise ValueError("Saved attacker labels or conservative bounds changed")
        ledger = state.get("home_ledger")
        if ledger is not None:
            ledger = np.asarray(ledger)
            if ledger.shape != self.features.home_ledger.shape or ledger.dtype != np.int64:
                raise ValueError("Invalid proximity ledger shape or dtype")
            ids, stages = ledger[..., 0], ledger[..., 1]
            if (
                np.any(ids < 0)
                or np.any((stages < 0) | (stages > 2))
                or np.any((ids == 0) & (stages != 0))
            ):
                raise ValueError("Invalid proximity ledger ID or stage")
            for row in ids:
                occupied = row[row != 0]
                if len(np.unique(occupied)) != len(occupied) or np.any(row[: len(occupied)] == 0):
                    raise ValueError("Invalid proximity ledger duplicate ID or gap")
        with self.device_context():
            skill = state.get("tile_scenario", "complete")
            rate = state.get("sky_sun_rate", 1.0)
            expected = peashooter_sky_sun_rate(self.cfg) if skill == "peashooter" else 1.0
            if rate != expected:
                raise ValueError("Saved sky-sun rate differs from scenario configuration")
            self.batch.set_sky_sun_rate(rate)
            self.batch.restore(state["games"], allowed=state["allowed"], digging=state["digging"])
            self._refresh_growth_bounds(range(self.num_envs))
            self.batch.header[:, 17] = self.cp.asarray(state["enabled"])
            self.enabled_envs[:] = np.asarray(state["enabled"], dtype=bool)
            self.features.totals[:] = self.cp.asarray(state["totals"])
            if ledger is not None:
                self.features.home_ledger[:] = self.cp.asarray(ledger)
            else:
                self.features.initialize_home(list(range(self.num_envs)))
            self.features.encode()
            for rng, saved in zip(self.queue.rngs, state["queue_rng"], strict=True):
                rng.bit_generator.state = deepcopy(saved)
            self.queue.progress, self.queue.stage = state["queue_progress"], state["queue_stage"]
            self.tile_scenario = self.queue.tile_scenario = state.get("tile_scenario", "complete")
            self.tile_boards = deepcopy(state.get("tile_boards", {}))
            self.queue.tile_boards = deepcopy(self.tile_boards)
            self.queue.generation_failures = state.get("tile_generation_failures", 0)
            self.queue.preparation_metrics = deepcopy(state.get("tile_preparation_metrics", {}))
            for key, value in state["attributes"].items():
                setattr(self, key, deepcopy(value))
            self._reset_seeds()  # Saved per-worker RNG replaces constructor seed requests.
        return self.features.obs_tensor

    def task_counts(self):
        paused = Counter(self._tasks[i] for i in self._tasks if not self.enabled_envs[i])
        return {
            task: {
                "started_games": self.task_started[task],
                "active_games": self.active_tasks[task] - paused[task],
                "completed_games": self.task_completed[task],
                "transitions": self.task_transitions[task],
            }
            for task in sorted(self.task_started)
        }

    def episode_metrics(self, index, h=None, t=None):
        if h is None:
            h = self.batch.header[index].get()
        if t is None:
            t = self.features.totals[index].get()
        level, family, seed = self._episode[index]
        status = ("truncated", "won", "lost")[int(h[1])]
        usage = {k: int(t[12 + j]) for j, k in enumerate(PLANT_TYPES) if t[12 + j]}
        return dict(
            level=level,
            family=family,
            episode_start_stage=self._episode_stages[index],
            early_voluntary_digs=int(t[35]),
            scenario_seed=seed,
            status=status,
            win=int(status == "won"),
            tick=int(h[0]),
            remaining_sun=float(h[HEADER.index("sun")]),
            simulated_seconds=float(h[0] / self.batch.rules.game["tick_rate"]),
            **{"return": float(t[0])},
            decisions=int(t[1]),
            # Accepted actions advance zero ticks in action-phase mode and the
            # configured decision interval in fixed-cadence mode. Waits and
            # rejected actions advance the configured interval.
            agent_actions=int(t[34]),
            simulation_ticks=int(t[2]),
            instant_actions=int(t[3]),
            max_actions_per_tick=int(t[5]),
            action_timing=self.cfg["environment"]["action_timing"],
            invalid_actions=int(t[6]),
            wait_actions=int(t[7]),
            maximum_sun=int(t[8]),
            affordable_attacker_opportunities=int(t[9]),
            attacker_purchases=int(t[10]),
            first_attacker_seconds=None
            if t[11] < 0
            else float(t[11] / self.batch.rules.game["tick_rate"]),
            plant_usage=usage,
            plant_spending={k: v * self.batch.rules.plants[k]["cost"] for k, v in usage.items()},
            **{k: float(t[j]) for k, j in zip(REWARD_METRICS, METRIC_INDICES, strict=True)},
            **{k: float(t[j]) for k, j in LEDGER_INDICES.items()},
            mowers_used=int(t[24]),
            defeated=int(h[5]),
            total_zombies=int(h[8]),
            failure_category="house_breach"
            if status == "lost"
            else "time_cutoff"
            if status == "truncated"
            else None,
        )

    def step_async(self, actions):
        self._actions = torch.as_tensor(actions, device="cuda", dtype=torch.long)

    def step_wait(self):
        obs, reward, done, _, terminal, infos = self.step_tensors(self._actions)
        for i, info in enumerate(infos):
            if "episode_metrics" in info:
                info["terminal_observation"] = terminal[i : i + 1].observations()[0]
        return obs.observations(), reward.cpu().numpy(), done.cpu().numpy(), infos

    def get_attr(self, attr_name, indices=None):
        return [getattr(self, attr_name)] * len(list(self._get_indices(indices)))

    def set_attr(self, attr_name, value, indices=None):
        raise ValueError("CUDA environment attributes are configured at construction")

    def env_method(self, method_name, *args, indices=None, **kwargs):
        ids = list(self._get_indices(indices))
        if method_name == "set_progress":
            self.queue.progress = int(args[0])
        elif method_name == "set_curriculum_stage":
            stage_distribution(self.cfg, args[0])
            self.queue.stage = int(args[0])
        elif method_name == "action_masks":
            return [self.features.mask_tensor[i] for i in ids]
        else:
            raise ValueError(f"Unsupported CUDA environment method: {method_name}")
        return [None] * len(ids)

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * len(list(self._get_indices(indices)))

    def close(self):
        self.queue.close()
        if getattr(self, "lane_preparation", None) is not None:
            self.lane_preparation.close()
            self.lane_preparation = None
        if self.live_view is not None:
            try:
                self.live_view.close()
            except Exception as exc:
                self.live_view.fail(exc)
                self.live_view.session.close()
            self.live_view = None
        self.action_journal.close()
        self._closed = True

    def start_live_view(self, *, notify=None):
        from pvz_rl.config import output_settings
        from pvz_rl.presentation.live_capture import CudaLiveCapture
        from pvz_rl.presentation.live_view import LiveSession

        settings = output_settings(self.cfg)["visualization"]
        if self.live_view is not None or not settings["live_enabled"]:
            return
        session = LiveSession(self.num_envs, settings, cfg=self.cfg, notify=notify)
        if session.viewer in ("web", "auto"):
            session.ready.wait(timeout=2)
            if session.proxy_url:
                print(f"Jupyter viewer: {session.proxy_url}", flush=True)
        try:
            with self.device_context():
                self.live_view = CudaLiveCapture(self, settings, session=session)
        except Exception as exc:
            session.fail(exc)
            session.close()
