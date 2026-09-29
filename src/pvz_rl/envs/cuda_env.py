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
from pvz_game.cuda.schema import REASONS
from stable_baselines3.common.vec_env import VecEnv

from pvz_rl.envs.actions import ActionSchema as A
from pvz_rl.envs.cuda_accounting import AccountingCudaBatch
from pvz_rl.envs.cuda_features import LEDGER_INDICES, METRIC_INDICES, CudaFeatures
from pvz_rl.envs.rewards import REWARD_METRICS
from pvz_rl.envs.scenarios import difficulty_weights, scenario
from pvz_rl.learning.budget import budget_target
from pvz_rl.learning.curriculum import stage_distribution, teaching_enabled
from pvz_rl.learning.training_requirements import require_supported_policy
from pvz_rl.monitoring.metrics import task_name


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

    Completion flags (three integers/game) are the synchronization boundary.
    Detailed episode totals transfer only for completed games, as one batch.
    Rendering and canonical snapshots are explicit diagnostic operations.
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
        self.phases = dict(simulation_features=0.0, scenario_preparation=0.0, transfers=0.0)
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
        self._transition_host = None
        self.finished_outcomes = {}
        self._episode_stages = [0] * cfg["training"]["n_envs"]
        self.task_started, self.task_transitions, self.task_completed = (
            Counter(),
            Counter(),
            Counter(),
        )
        self.active_tasks, self._tasks = Counter(), {}
        self.enabled_envs = np.ones(cfg["training"]["n_envs"], dtype=bool)
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
        with self.device_context():
            self.features = CudaFeatures(self.batch, cfg, condition)
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
        self.features.initialize_home(indices)
        self.proposed_actions[indices] = 0
        self.executed_actions[indices] = 0
        ix = self.cp.asarray(indices)
        self.features.totals[ix] = 0
        self.features.totals[ix, 8] = self.batch.header[ix, 2]
        self.features.totals[ix, 11] = -1
        self.features.encode()
        self.phases["scenario_preparation"] += perf_counter() - started

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

    def step_tensors(self, actions, *, autoreset=True):
        with self.device_context():
            viewer = self.live_view
            if viewer is not None:
                try:
                    if not viewer.prepared:
                        viewer.prepare()
                    viewer.prepared = False
                except Exception as exc:
                    viewer.fail(exc)
                if not viewer.enabled:
                    viewer = None
            if self.training:
                self.task_transitions.update(
                    self._tasks[i] for i in np.flatnonzero(self.enabled_envs)
                )
            started = perf_counter()
            obs, reward = self.features.step(self.cp.from_dlpack(actions.detach().contiguous()))
            self.phases["simulation_features"] += perf_counter() - started
            h = self.header_tensor
            self.proposed_actions.copy_(actions)
            self.transition_ticks.copy_(h[:, 14])
            # Instantaneous accepted plant/dig proposals execute themselves;
            # waits and every rejected proposal execute action zero.
            self.executed_actions.copy_(torch.where(h[:, 14] == 0, actions, 0))
            self.terminal_ticks = h[:, 0].clone()
            self.terminal_masks = self.action_masks().clone()
            done = (
                (h[:, 1] != 0)
                | (
                    h[:, 0]
                    >= self.cfg["environment"]["cutoff_seconds"]
                    * self.batch.rules.game["tick_rate"]
                )
            ) & (h[:, 17] != 0)
            timed_out = done & (h[:, 1] == 0)
            # One compact transfer per decision; no entity state/observations.
            started = perf_counter()
            compact_tensor = torch.stack(
                (done.double(), timed_out.double(), h[:, 14].double(), reward.double()), dim=1
            )
            # Include proposal/execution/omissions: never extract per-environment GPU scalars.
            extra = torch.cat(
                (
                    self.proposed_actions[:, None].double(),
                    self.executed_actions[:, None].double(),
                    torch.from_dlpack(self.features.truncation_counts).double(),
                ),
                dim=1,
            )
            packed = torch.cat(
                (
                    compact_tensor.flatten(),
                    h[:, 12:14].double().flatten(),
                    extra.flatten(),
                    torch.from_dlpack(self.features.parts).flatten(),
                )
            )
            if self._transition_host is None or self._transition_host.shape != packed.shape:
                self._transition_host = torch.empty_like(packed, device="cpu", pin_memory=True)
            self._transition_host.copy_(packed, non_blocking=True)
            self.stream.synchronize()
            host = self._transition_host.numpy()
            compact = host[: self.num_envs * 4].reshape(self.num_envs, 4)
            self.last_transition_host = compact
            self.last_action_result_host = host[self.num_envs * 4 : self.num_envs * 6].reshape(
                self.num_envs, 2
            )
            if viewer is not None:
                try:
                    viewer.after_step(compact)
                except Exception as exc:
                    viewer.fail(exc)
            self.phases["transfers"] += perf_counter() - started
            indices = np.flatnonzero(compact[:, 0]).tolist()
            extra_host = host[self.num_envs * 6 : self.num_envs * 11].reshape(self.num_envs, 5)
            self.last_reward_parts_host = host[self.num_envs * 11 :].reshape(self.num_envs, -1)
            if not np.isfinite(self.last_reward_parts_host).all():
                raise RuntimeError("Invalid CUDA reward accounting or exhausted proximity ledger")
            truncation = extra_host[:, 2:]
            infos = []
            for index, row in enumerate(compact):
                accepted = bool(self.last_action_result_host[index, 0])
                reason_code = int(self.last_action_result_host[index, 1])
                reason = (
                    None
                    if accepted or not 0 <= reason_code < len(REASONS)
                    else REASONS[reason_code]
                )
                infos.append(
                    {
                        "entity_truncation": {
                            "total": int(truncation[index].sum()),
                            **dict(
                                zip(
                                    ("plants", "zombies", "projectiles"),
                                    map(int, truncation[index]),
                                )
                            ),
                        },
                        "proposal_action": int(extra_host[index, 0]),
                        "executed_action": int(extra_host[index, 1]),
                        "accepted": accepted,
                        "rejection_reason": reason,
                        "ticks_advanced": int(row[2]),
                    }
                )
            reward = reward.clone()
            terminal_observations = None
            if indices:
                terminal_observations = obs.clone()
                headers = self.batch.header[self.cp.asarray(indices)].get()
                totals = self.features.totals[self.cp.asarray(indices)].get()
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
