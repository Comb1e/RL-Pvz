"""Tensor vector environment; only compact completion information crosses to CPU."""

from collections import Counter, deque
from contextlib import contextmanager
from copy import deepcopy
from time import perf_counter

import numpy as np
import torch
from gymnasium import spaces
from pvz_game import Game, Rules
from pvz_game.config import PLANT_TYPES
from pvz_game.cuda.backend import projectile_bound
from pvz_game.cuda.schema import HEADER, REASONS
from stable_baselines3.common.vec_env import VecEnv

from pvz_rl.envs.actions import ActionSchema as A
from pvz_rl.envs.cuda_accounting import AccountingCudaBatch
from pvz_rl.envs.cuda_features import LEDGER_INDICES, METRIC_INDICES, CudaFeatures
from pvz_rl.envs.encoding import width_bucket
from pvz_rl.envs.rewards import REWARD_METRICS
from pvz_rl.envs.scenarios import difficulty_weights, scenario
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
        self.pending = deque(maxlen=n)
        self.progress, self.stage = 0, 0
        self.rules = Rules()

    def prepare(self, indices):
        cfg = self.cfg
        for index in indices:
            rng = self.rngs[index]
            lo, hi = cfg["splits"]["train"]
            seed = int(rng.integers(lo, hi + 1))
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
            self.pending.append(
                (index, level, family, seed, scenario(level, family, seed, self.rules, cfg))
            )
        result = list(self.pending)
        self.pending.clear()
        return result


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
        if cfg["environment"].get("action_timing") != "per_tick":
            raise ValueError(
                "CUDA training requires per_tick actions; legacy timing is inference-only"
            )
        require_supported_policy(cfg, condition)
        self.cfg, self.condition, self.family, self.training = cfg, condition, family, training
        self.queue = ScenarioQueue(cfg, condition, learner_seed, family, cfg["training"]["n_envs"])
        self.stream = torch.cuda.current_stream()
        self.handoff = HostHandoff(self.stream)
        self.cases = cases
        self._episode = [None] * cfg["training"]["n_envs"]
        self._episode_serial = [0] * cfg["training"]["n_envs"]
        self._level_names = [None] * cfg["training"]["n_envs"]
        self.live_view = None
        from pvz_rl.config import output_settings
        from pvz_rl.presentation.action_journal import ActionJournal

        self.action_journal = ActionJournal(
            cfg["training"]["n_envs"],
            ram_bytes=output_settings(cfg)["visualization"]["live_history_ram_mib"] * 1024**2,
            reward_settings=cfg["reward"],
        )
        self._record = None
        self.finished_outcomes = {}
        self._episode_stages = [0] * cfg["training"]["n_envs"]
        self.task_started, self.task_transitions, self.task_completed = (
            Counter(),
            Counter(),
            Counter(),
        )
        self.active_tasks, self._tasks = Counter(), {}
        self.enabled_envs = np.ones(cfg["training"]["n_envs"], dtype=bool)
        self._pending_spawns = np.zeros(cfg["training"]["n_envs"], dtype=np.int64)
        self._plant_counts = np.zeros_like(self._pending_spawns)
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
            cfg["training"]["n_envs"],
            zombie_capacity=max(1, *counts),
            # A mid-game recovery snapshot already owns some projectile slots;
            # the pinned backend also reserves its full reachable-shot bound for
            # future simulation.  Allocate both portions up front so resume does
            # not fail merely because the checkpoint was saved after firing.
            projectile_capacity=2 * projectile_bound(self.queue.rules),
            max_step_ticks=1,
            diagnostic=False,
        )
        self.cp = self.batch.cp
        # Collection telemetry is shared by the environment, its features and
        # the counterfactual collector; evaluation leaves it disabled.
        self.profiler = DeviceProfiler(
            self.cp,
            training and cfg["training"].get("performance", {}).get("telemetry", False),
        )
        with self.device_context():
            self.features = CudaFeatures(self.batch, cfg, condition)
            self.features.profiler = self.profiler
            self.header_tensor = torch.from_dlpack(self.batch.header)
        self.render_mode = None
        super().__init__(self.batch.n, self.features.encoder.space, spaces.Discrete(A.size))
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

    def reset_indices(self, indices, cases=None):
        started = perf_counter()
        if self.live_view is not None:
            try:
                self.live_view.drain(wait=True)
            except Exception as exc:
                self.live_view.fail(exc)
        self.enabled_envs[indices] = True
        if cases is None:
            staged = self.queue.prepare(indices)
        else:
            staged = [
                (i, level, family, seed, scenario(level, family, seed, self.queue.rules, self.cfg))
                for i, (level, family, seed) in zip(indices, cases)
            ]
        # Branch restrictions are intentionally absent from policy masks and
        # simulator setup.  Invalid plant proposals are retained as actions
        # and receive the configured rejection penalty.
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
        self.batch.reset(
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
        self.profiler.host("scenario_preparation", perf_counter() - started)

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

    def _refresh_growth_bounds(self, indices):
        header = self.batch.header.get()
        self._update_public_counts(header, indices)

    def _update_public_counts(self, header, indices):
        self._pending_spawns[indices] = (
            header[indices, HEADER.index("total_spawns")]
            - header[indices, HEADER.index("spawn_index")]
        )
        self._plant_counts[indices] = header[indices, HEADER.index("np")]

    def probe_width(self):
        """Public-count bound: one new plant, pending zombies and two shots per plant."""
        kept = self.features.summary_host[:, 0]
        bound = kept + self._pending_spawns + 2 * self._plant_counts + 1
        return width_bucket(int(bound[self.enabled_envs].max(initial=0)), self.features.limit)

    def step_device(self, actions):
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
        if self.training:
            self.task_transitions.update(self._tasks[i] for i in np.flatnonzero(self.enabled_envs))
        self._step_active = self.enabled_envs.copy()
        _, reward = self.features.step(
            self.cp.from_dlpack(actions.detach().contiguous()), publish=False
        )
        h = self.header_tensor
        self.proposed_actions.copy_(actions)
        self.transition_ticks.copy_(h[:, 14])
        # Instantaneous accepted plant/dig proposals execute themselves;
        # waits and every rejected proposal execute action zero.
        self.executed_actions.copy_(torch.where(h[:, 14] == 0, actions, 0))
        done = (
            (h[:, 1] != 0)
            | (
                h[:, 0]
                >= self.cfg["environment"]["cutoff_seconds"] * self.batch.rules.game["tick_rate"]
            )
        ) & (h[:, 17] != 0)
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
            self._episode_host = self.handoff.enqueue_fields(
                "episode",
                dict(
                    header=h,
                    totals=torch.from_dlpack(self.features.totals),
                ),
            )
        self._step_device_result = reward, done, timed_out

    def step_host(self, *, autoreset=True):
        """Interpret the completed step record after the handoff wait."""
        host = self._record.numpy()
        self._update_public_counts(self._episode_host["header"].numpy(), slice(None))
        compact = host[:, RECORD_OUTCOME]
        self.last_transition_host = compact
        self.last_action_result_host = host[:, RECORD_RESULT]
        actions = host[:, RECORD_ACTIONS]
        self.last_reward_parts_host = host[:, RECORD_PARTS]
        viewer = self.live_view
        if viewer is not None and viewer.enabled:
            presentation_started = perf_counter()
            try:
                viewer.after_step(compact)
            except Exception as exc:
                viewer.fail(exc)
            self.profiler.host("presentation", perf_counter() - presentation_started)
        if not np.isfinite(self.last_reward_parts_host).all():
            raise RuntimeError("Invalid CUDA reward accounting or exhausted proximity ledger")
        summary = host[:, RECORD_SUMMARY].astype(np.int64)
        # Every game active during this step keeps its complete terminal record.
        obs = self.features.publish(summary, self._step_active)
        truncation = summary[:, 1:]
        infos = []
        for index, row in enumerate(compact):
            accepted = bool(self.last_action_result_host[index, 0])
            reason_code = int(self.last_action_result_host[index, 1])
            reason = (
                None if accepted or not 0 <= reason_code < len(REASONS) else REASONS[reason_code]
            )
            infos.append(
                {
                    "entity_truncation": {
                        "total": int(truncation[index].sum()),
                        **dict(
                            zip(("plants", "zombies", "projectiles"), map(int, truncation[index]))
                        ),
                    },
                    "proposal_action": int(actions[index, 0]),
                    "executed_action": int(actions[index, 1]),
                    "accepted": accepted,
                    "rejection_reason": reason,
                    "ticks_advanced": int(row[2]),
                }
            )
        reward, done, timed_out = self._step_device_result
        reward = reward.clone()
        indices = np.flatnonzero(compact[:, 0]).tolist()
        terminal_observations = None
        if indices:
            terminal_observations = obs.clone()
            headers = self._episode_host["header"].numpy()[indices]
            totals = self._episode_host["totals"].numpy()[indices]
            for index, header, total in zip(indices, headers, totals):
                self.task_completed[self._tasks[index]] += 1
                infos[index].update(
                    episode_metrics=self.episode_metrics(index, header, total),
                    **{"TimeLimit.truncated": bool(compact[index, 1])},
                )
                self.finished_outcomes[index] = infos[index]["episode_metrics"]["status"]
            if autoreset:
                self.reset_indices(indices)
                obs = self.features.obs_tensor
            else:
                self.batch.header[self.cp.asarray(indices), 17] = 0
                self.enabled_envs[indices] = False
        return obs, reward, done, timed_out, terminal_observations, infos

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
                )
            },
        }

    def restore_training(self, state):
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
            simulated_seconds=float(h[0] / self.batch.rules.game["tick_rate"]),
            **{"return": float(t[0])},
            decisions=int(t[1]),
            # In the required per-tick mode, only accepted plant/dig actions
            # advance zero ticks. Waits and rejected actions always advance one.
            agent_actions=int(t[3]),
            simulation_ticks=int(t[2]),
            instant_actions=int(t[3]),
            max_actions_per_tick=int(t[5]),
            action_timing="per_tick",
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
        try:
            with self.device_context():
                self.live_view = CudaLiveCapture(self, settings, session=session)
        except Exception as exc:
            session.fail(exc)
            session.close()
