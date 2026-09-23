"""Tensor vector environment; only compact completion information crosses to CPU."""

from collections import Counter, deque
from contextlib import contextmanager
from time import perf_counter

import numpy as np
import torch
from gymnasium import spaces
from pvz_game import Game, Rules
from pvz_game.config import PLANT_TYPES
from pvz_game.cuda import CudaBatch
from stable_baselines3.common.vec_env import VecEnv

from .budget import budget_target
from .config import lesson_settings
from .cuda_features import METRIC_INDICES, CudaFeatures
from .curriculum import LESSONS, stage_distribution, teaching_enabled
from .metrics import task_name
from .rewards import REWARD_METRICS
from .scenarios import difficulty_weights, scenario


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
                family, level = (task, "easy") if task in LESSONS else ("preset", task)
            self.pending.append(
                (index, level, family, seed, scenario(level, family, seed, self.rules, cfg))
            )
        result = list(self.pending)
        self.pending.clear()
        return result


class CudaVecEnv(VecEnv):
    """SB3 control interface plus device-native reset/step used by TensorPPO.

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
        self.cfg, self.condition, self.family, self.training = cfg, condition, family, training
        self.queue = ScenarioQueue(cfg, condition, learner_seed, family, cfg["training"]["n_envs"])
        self.stream = torch.cuda.current_stream()
        self.phases = dict(simulation_features=0.0, scenario_preparation=0.0, transfers=0.0)
        self.cases = cases
        self._episode = [None] * cfg["training"]["n_envs"]
        self._episode_stages = [0] * cfg["training"]["n_envs"]
        self.task_started, self.task_transitions, self.task_completed = (
            Counter(),
            Counter(),
            Counter(),
        )
        self.active_tasks, self._tasks = Counter(), {}
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
        self.batch = CudaBatch(
            cfg["training"]["n_envs"],
            zombie_capacity=max(1, *counts),
            max_step_ticks=1,
            diagnostic=False,
        )
        self.cp = self.batch.cp
        with self.device_context():
            self.features = CudaFeatures(self.batch, cfg, condition)
            self.header_tensor = torch.from_dlpack(self.batch.header)
        self.render_mode = None
        super().__init__(self.batch.n, self.features.encoder.space, spaces.Discrete(406))
        self._closed = False

    @contextmanager
    def device_context(self):
        with (
            torch.cuda.stream(self.stream),
            self.cp.cuda.ExternalStream(self.stream.cuda_stream, device_id=self.batch.device),
        ):
            yield

    def reset_indices(self, indices, cases=None):
        started = perf_counter()
        if cases is None:
            staged = self.queue.prepare(indices)
        else:
            staged = [
                (i, level, family, seed, scenario(level, family, seed, self.queue.rules, self.cfg))
                for i, (level, family, seed) in zip(indices, cases)
            ]
        allowed, digging = [], []
        for index, level, family, seed, _ in staged:
            task = task_name(level, family)
            if index in self._tasks:
                self.active_tasks[self._tasks[index]] -= 1
            self._tasks[index] = task
            self.active_tasks[task] += 1
            self.task_started[task] += 1
            self._episode[index] = (level, family, seed)
            self._episode_stages[index] = self.queue.stage
            types = PLANT_TYPES
            if family == "diagnostic":
                types = ("peashooter",)
            elif family in LESSONS:
                types = lesson_settings(self.cfg)[family]["allowed_plants"]
            allowed.append(sum(1 << PLANT_TYPES.index(t) for t in types))
            digging.append(family != "diagnostic")
        self.batch.reset(
            [s[4] for s in staged],
            [s[3] for s in staged],
            indices=indices,
            allowed=allowed,
            digging=digging,
        )
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

    def action_masks(self):
        return self.features.mask_tensor

    def step_tensors(self, actions, *, autoreset=True):
        with self.device_context():
            if self.training:
                self.task_transitions.update(self.active_tasks)
            started = perf_counter()
            obs, reward = self.features.step(self.cp.from_dlpack(actions.detach().contiguous()))
            self.phases["simulation_features"] += perf_counter() - started
            h = self.header_tensor
            done = (
                (h[:, 1] != 0) | (h[:, 0] >= self.cfg["environment"]["cutoff_seconds"] * 20)
            ) & (h[:, 17] != 0)
            timed_out = done & (h[:, 1] == 0)
            # One compact transfer per decision; no entity state/observations.
            started = perf_counter()
            compact = torch.stack((done.long(), timed_out.long(), h[:, 14]), dim=1).cpu().numpy()
            self.phases["transfers"] += perf_counter() - started
            indices = np.flatnonzero(compact[:, 0]).tolist()
            infos = [{"ticks_advanced": int(row[2])} for row in compact]
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
                if autoreset:
                    self.reset_indices(indices)
                else:
                    self.batch.header[self.cp.asarray(indices), 17] = 0
            return obs, reward, done, timed_out, terminal_observations, infos

    def task_counts(self):
        return {
            task: {
                "started_games": self.task_started[task],
                "active_games": self.active_tasks[task],
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
            simulated_seconds=float(h[0] / 20),
            **{"return": float(t[0])},
            decisions=int(t[1]),
            simulation_ticks=int(t[2]),
            instant_actions=int(t[3]),
            max_actions_per_tick=int(t[5]),
            action_timing="per_tick",
            invalid_actions=int(t[6]),
            wait_actions=int(t[7]),
            maximum_sun=int(t[8]),
            affordable_attacker_opportunities=int(t[9]),
            attacker_purchases=int(t[10]),
            first_attacker_seconds=None if t[11] < 0 else float(t[11] / 20),
            plant_usage=usage,
            plant_spending={k: v * self.batch.rules.plants[k]["cost"] for k, v in usage.items()},
            **{k: float(t[j]) for k, j in zip(REWARD_METRICS, METRIC_INDICES, strict=True)},
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
                info["terminal_observation"] = terminal[i].cpu().numpy()
        return obs.cpu().numpy(), reward.cpu().numpy(), done.cpu().numpy(), infos

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
        self._closed = True
