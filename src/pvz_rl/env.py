"""Gymnasium adapter with explicit outcomes and no automatic reset."""

from __future__ import annotations

import copy
from collections import Counter
from enum import StrEnum

import gymnasium as gym
import numpy as np
from pvz_game import Game, LevelSpec, Place, Rules, Status, Wait, WaveSpec
from pvz_game.replay import Recorder

from .actions import ActionCodec
from .config import load_config, validate_config
from .controllers import PublicBoard, strategy_candidates
from .encoding import ObservationEncoder
from .rewards import reward_parts
from .scenarios import difficulty_weights, scenario


class EpisodeState(StrEnum):
    NEEDS_RESET = "needs_reset"
    RUNNING = "running"
    WON = "won"
    LOST = "lost"
    TRUNCATED = "truncated"


class PvZEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 2}

    def __init__(
        self,
        cfg: dict | None = None,
        *,
        condition="masked",
        level="standard",
        family="preset",
        training=False,
        worker_seed=0,
        record=False,
        replay_metadata=None,
        render_mode=None,
        rules=None,
    ):
        super().__init__()
        self.cfg = copy.deepcopy(cfg or load_config())
        validate_config(self.cfg)
        self.condition = condition
        self.options = self.cfg["conditions"][condition]
        self.level, self.family, self.training = level, family, training
        self.record, self.render_mode = record, render_mode
        self.replay_metadata = copy.deepcopy(replay_metadata or {})
        if render_mode not in (None, "rgb_array"):
            raise ValueError("Only rgb_array rendering is supported")
        self.rules = rules or Rules()
        self.game = Game(self.rules)
        self.codec = ActionCodec(self.cfg)
        self.encoder = ObservationEncoder(self.cfg, self.rules)
        self.observation_space = self.encoder.space
        self.action_space = gym.spaces.Discrete(5 if self.options["hybrid"] else self.codec.size)
        self.selection_rng = np.random.default_rng(worker_seed)
        self.progress = 0
        self.state = EpisodeState.NEEDS_RESET
        self.public = None
        self.recorder = None
        self._legal = None
        self._candidates = None

    def set_progress(self, decisions: int):
        self.progress = int(decisions)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        if self.training:
            if seed is not None:
                self.selection_rng = np.random.default_rng(seed)
            lo, hi = self.cfg["splits"]["train"]
            game_seed = int(self.selection_rng.integers(lo, hi + 1))
            weights = difficulty_weights(
                self.cfg,
                self.progress,
                self.cfg["training"]["total_steps"],
                self.options["curriculum"],
            )
            level = str(self.selection_rng.choice(self.cfg["evaluation"]["levels"], p=weights))
        else:
            game_seed = int(seed) if seed is not None else int(self.np_random.integers(2**31))
            level = options.get("level", self.level)
        family = options.get("family", self.family)
        if family == "diagnostic" and (self.options["hybrid"] or not self.options["masked"]):
            raise ValueError("The diagnostic requires a direct-placement masked condition")
        resolved = options.get("scenario")
        if resolved is None:
            resolved = scenario(level, family, game_seed, self.rules)
        if not isinstance(resolved, (str, LevelSpec, WaveSpec)):
            raise TypeError("Scenario must be a preset, LevelSpec or WaveSpec")
        self.public = self.game.reset(resolved, game_seed)
        self.episode_level, self.episode_family, self.episode_seed = level, family, game_seed
        self.state = EpisodeState.RUNNING
        self._legal = self._candidates = None
        self.recorder = (
            Recorder(
                self.game,
                metadata={
                    **self.replay_metadata,
                    "outcome": "running",
                    "experiment": {
                        **self.replay_metadata.get("experiment", {}),
                        "level": level,
                        "family": family,
                        "scenario_seed": game_seed,
                    },
                },
            )
            if self.record
            else None
        )
        self.metrics = Counter()
        self.plant_usage = Counter()
        self.episode_reward = 0.0
        self.cutoff_ticks = self.cfg["environment"]["cutoff_seconds"] * self.public.tick_rate
        return self.encoder.encode(self.public), {"status": self.state.value}

    def public_board(self):
        if self._legal is None:
            self._legal = frozenset(self.game.legal_actions())
        return PublicBoard(self.public, self._legal)

    def candidates(self):
        if self._candidates is None:
            self._candidates = strategy_candidates(self.public_board())
        return self._candidates

    def action_masks(self):
        if self.state != EpisodeState.RUNNING:
            return np.zeros(self.action_space.n, dtype=np.bool_)
        if self.options["hybrid"]:
            return np.array([a is not None for a in self.candidates()], dtype=np.bool_)
        mask = np.zeros(self.codec.size, dtype=np.bool_)
        for action in self.public_board().legal:
            mask[self.codec.encode(action)] = True
        if self.episode_family == "diagnostic":
            # An explicitly restricted learning diagnostic, never a formal game condition.
            allowed = np.array(
                [
                    isinstance(a, Wait) or isinstance(a, Place) and a.plant_type == "peashooter"
                    for a in self.codec.actions
                ]
            )
            mask &= allowed
        return mask

    def step(self, action):
        if self.state != EpisodeState.RUNNING:
            raise RuntimeError("Reset after an episode finishes")
        if not self.action_space.contains(action) or isinstance(action, (bool, np.bool_)):
            raise ValueError("Action outside the discrete space")
        rejected_strategy = False
        if self.options["hybrid"]:
            concrete = self.candidates()[int(action)]
            rejected_strategy = concrete is None
            concrete = Wait() if concrete is None else concrete
        else:
            concrete = self.codec.decode(action)
        before = self.public
        ticks = min(self.cfg["environment"]["decision_ticks"], self.cutoff_ticks - before.tick)
        result = (self.recorder or self.game).step(concrete, ticks=ticks)
        self.public = result.observation
        self._legal = self._candidates = None
        terminated = result.status != Status.RUNNING
        truncated = not terminated and self.public.tick >= self.cutoff_ticks
        self.state = (
            EpisodeState(result.status.value)
            if terminated
            else EpisodeState.TRUNCATED
            if truncated
            else EpisodeState.RUNNING
        )
        parts = reward_parts(before, self.public, self.cfg, self.options["shaped"])
        self.episode_reward += parts["total"]
        self.metrics["decisions"] += 1
        if self.training:
            # Every vector worker advances once per collection step. Updating here
            # makes the new stage visible before VecEnv's automatic episode reset.
            self.progress += self.cfg["training"]["n_envs"]
        self.metrics["invalid_actions"] += int(
            not result.action_result.accepted or rejected_strategy
        )
        self.metrics["wait_actions"] += int(isinstance(concrete, Wait))
        if isinstance(concrete, Place) and result.action_result.accepted:
            self.plant_usage[concrete.plant_type] += 1
        for event in result.events:
            self.metrics[event.kind] += 1
        info = {
            "status": self.state.value,
            "reward_parts": parts,
            "accepted": result.action_result.accepted and not rejected_strategy,
            "ticks_advanced": result.ticks_advanced,
        }
        if terminated or truncated:
            if self.recorder:
                self.recorder.update_metadata(
                    {
                        "outcome": self.state.value,
                        "termination_reason": "time_limit"
                        if truncated
                        else ("victory" if self.state == EpisodeState.WON else "house_breach"),
                    }
                )
            info["episode_metrics"] = self.episode_metrics()
        return self.encoder.encode(self.public), parts["total"], terminated, truncated, info

    def episode_metrics(self):
        obs = self.public
        return {
            "level": self.episode_level,
            "family": self.episode_family,
            "scenario_seed": self.episode_seed,
            "status": self.state.value,
            "win": int(self.state == EpisodeState.WON),
            "tick": obs.tick,
            "simulated_seconds": obs.elapsed_seconds,
            "return": self.episode_reward,
            "decisions": self.metrics["decisions"],
            "invalid_actions": self.metrics["invalid_actions"],
            "wait_actions": self.metrics["wait_actions"],
            "mowers_used": self.metrics["MowerActivated"],
            "defeated": obs.counts.defeated,
            "total_zombies": obs.counts.initial_total,
            "plant_usage": dict(self.plant_usage),
            "failure_category": (
                "house_breach"
                if self.state == EpisodeState.LOST
                else "time_cutoff"
                if self.state == EpisodeState.TRUNCATED
                else None
            ),
        }

    def render(self):
        if self.public is None:
            raise RuntimeError("Reset before rendering")
        from .rendering import render_observation

        return render_observation(self.public)
