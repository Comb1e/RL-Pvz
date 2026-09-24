"""Gymnasium adapter with explicit outcomes and no automatic reset."""

from __future__ import annotations

import copy
from collections import Counter
from enum import StrEnum

import gymnasium as gym
import numpy as np
from pvz_game import Dig, Game, LevelSpec, Place, Rules, Status, Wait, WaveSpec
from pvz_game.replay import Recorder

from .action_timing import ActionPhaseGame, per_tick_actions
from .actions import ActionCodec
from .budget import budget_target, uses_games
from .config import lesson_settings, load_config, runtime_settings, validate_config
from .controllers import PublicBoard, strategy_candidates
from .curriculum import LESSONS, stage_distribution, teaching_enabled
from .encoding import ObservationEncoder
from .lesson_rules import natural_sun, sky_rules
from .recordings import ActionPhaseRecorder
from .rewards import LEDGER_METRICS, REWARD_METRICS, reward_parts
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
        self.options = (
            {"masked": True, "shaped": True, "curriculum": True, "hybrid": True}
            if condition == "hybrid"
            else self.cfg["conditions"][condition]
        )
        self.level, self.family, self.training = level, family, training
        self.record, self.render_mode = record, render_mode
        self.replay_metadata = copy.deepcopy(replay_metadata or {})
        if render_mode not in (None, "rgb_array"):
            raise ValueError("Only rgb_array rendering is supported")
        self.rules = rules or Rules()
        self.per_tick = per_tick_actions(self.cfg)
        self.game = ActionPhaseGame(self.rules) if self.per_tick else Game(self.rules)
        self.metadata = {
            **self.metadata,
            "render_fps": self.rules.game["tick_rate"]
            if self.per_tick
            else self.rules.game["tick_rate"] / self.cfg["environment"]["decision_ticks"],
        }
        self.codec = ActionCodec(self.cfg)
        self.encoder = ObservationEncoder(self.cfg, self.rules)
        self.observation_space = self.encoder.space
        self.action_space = gym.spaces.Discrete(5 if self.options["hybrid"] else self.codec.size)
        self.selection_rng = np.random.default_rng(worker_seed)
        self.progress = 0
        self.curriculum_stage = 0
        self.state = EpisodeState.NEEDS_RESET
        self.public = None
        self.recorder = None
        self._legal = None
        self._legal_key = None
        self._direct_mask = None
        self.cache_legal_actions = runtime_settings(self.cfg)["cache_legal_actions"]
        self._candidates = None

    def set_progress(self, decisions: int):
        self.progress = int(decisions)

    def set_curriculum_stage(self, stage):
        # Active episode_family and masks are deliberately unchanged.
        stage_distribution(self.cfg, stage)
        self.curriculum_stage = stage

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
                budget_target(self.cfg),
                self.options["curriculum"],
            )
            level = str(self.selection_rng.choice(self.cfg["evaluation"]["levels"], p=weights))
        else:
            game_seed = int(seed) if seed is not None else int(self.np_random.integers(2**31))
            level = options.get("level", self.level)
        family = options.get("family", self.family)
        if self.training and self.family == "preset" and teaching_enabled(self.cfg):
            tasks, weights = stage_distribution(self.cfg, self.curriculum_stage)
            task = str(self.selection_rng.choice(tasks, p=weights))
            family, level = (task, "easy") if task in LESSONS else ("preset", task)
        if family == "diagnostic" and (self.options["hybrid"] or not self.options["masked"]):
            raise ValueError("The diagnostic requires a direct-placement masked condition")
        resolved = options.get("scenario")
        if resolved is None:
            resolved = scenario(level, family, game_seed, self.rules, self.cfg)
        if not isinstance(resolved, (str, LevelSpec, WaveSpec)):
            raise TypeError("Scenario must be a preset, LevelSpec or WaveSpec")
        rules = sky_rules(self.rules, natural_sun(self.cfg, family))
        if self.game.rules.digest != rules.digest:
            self.game = ActionPhaseGame(rules) if self.per_tick else Game(rules)
        self.public = self.game.reset(resolved, game_seed)
        self.episode_level, self.episode_family, self.episode_seed = level, family, game_seed
        self.episode_stage = self.curriculum_stage
        self.planted_ticks = {}
        self.state = EpisodeState.RUNNING
        self._legal = self._candidates = None
        self._legal_key = self._direct_mask = None
        self.recorder = (
            (ActionPhaseRecorder if self.per_tick else Recorder)(
                self.game,
                metadata={
                    **self.replay_metadata,
                    "outcome": "running",
                    "experiment": {
                        **self.replay_metadata.get("experiment", {}),
                        "level": level,
                        "family": family,
                        "scenario_seed": game_seed,
                        "action_timing": "per_tick" if self.per_tick else "fixed",
                    },
                },
            )
            if self.record
            else None
        )
        self.metrics = Counter()
        self.actions_this_tick = 0
        self.plant_usage = Counter()
        self.plant_spending = Counter()
        self.maximum_sun = self.public.sun
        self.first_attacker_tick = None
        self.episode_reward = 0.0
        self.cutoff_ticks = self.cfg["environment"]["cutoff_seconds"] * self.public.tick_rate
        return self.encoder.encode(self.public), {"status": self.state.value}

    def public_board(self):
        obs = self.public
        # In the pinned engine legality depends only on status, occupied tiles,
        # affordability, and whether recharge has completed. Other public changes
        # (health, projectiles, zombie movement) cannot change legal actions.
        key = (
            (
                obs.status,
                frozenset((p.row, p.col) for p in obs.plants),
                tuple(
                    (c.plant_type, c.cooldown_ticks == 0 and obs.sun >= c.cost) for c in obs.cards
                ),
            )
            if self.cache_legal_actions
            else None
        )
        if self._legal is None or self.cache_legal_actions and key != self._legal_key:
            self._legal = frozenset(self.game.legal_actions())
            self._legal_key = key
            self._direct_mask = None
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
        legal = self.public_board().legal
        if self._direct_mask is None or not self.cache_legal_actions:
            mask = np.zeros(self.codec.size, dtype=np.bool_)
            for action in legal:
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
            if self.episode_family in LESSONS:
                allowed_plants = lesson_settings(self.cfg)[self.episode_family]["allowed_plants"]
                mask &= np.array(
                    [
                        isinstance(a, (Wait, Dig))
                        or isinstance(a, Place)
                        and a.plant_type in allowed_plants
                        for a in self.codec.actions
                    ]
                )
            self._direct_mask = mask
        return self._direct_mask.copy()

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
        restricted = (
            self.episode_family in (*LESSONS, "diagnostic") and not self.action_masks()[int(action)]
        )
        if restricted:
            # Task restrictions apply even to callers that ignore the mask.
            concrete = Wait()
            rejected_strategy = True
        before = self.public
        attackers = ("peashooter", "snow_pea", "repeater")
        self.metrics["affordable_attacker_opportunities"] += int(
            any(
                c.plant_type in attackers and before.sun >= c.cost and c.cooldown_ticks == 0
                for c in before.cards
            )
            and len(before.plants)
            < self.cfg["environment"]["rows"] * self.cfg["environment"]["cols"]
        )
        if self.per_tick:
            # Accepted operations are immediate; Wait or rejection ends this tick's
            # action phase. Revalidation observes sun/cooldown/board changes.
            ticks = int(
                isinstance(concrete, Wait) or not self.game.validate_action(concrete).accepted
            )
        else:
            ticks = min(self.cfg["environment"]["decision_ticks"], self.cutoff_ticks - before.tick)
        result = (self.recorder or self.game).step(concrete, ticks=ticks)
        self.public = result.observation
        self.maximum_sun = max(self.maximum_sun, before.sun, self.public.sun)
        if not self.cache_legal_actions:
            self._legal = None
        self._candidates = None
        terminated = result.status != Status.RUNNING
        truncated = not terminated and self.public.tick >= self.cutoff_ticks
        self.state = (
            EpisodeState(result.status.value)
            if terminated
            else EpisodeState.TRUNCATED
            if truncated
            else EpisodeState.RUNNING
        )
        parts = reward_parts(
            before,
            self.public,
            self.cfg,
            events=result.events,
            rules=self.rules,
        )
        for key in REWARD_METRICS:
            self.metrics[key] += parts[key]
        self.episode_reward += parts["total"]
        self.metrics["discounted_return"] += (
            self.cfg["training"]["gamma"] ** self.metrics["simulation_ticks"] * parts["total"]
        )
        self.metrics["cumulative_net_value"] = self.metrics["net_value"]
        self.metrics["maximum_net_value"] = max(
            self.metrics["maximum_net_value"], self.metrics["net_value"]
        )
        self.metrics["value_drawdown"] = (
            self.metrics["maximum_net_value"] - self.metrics["net_value"]
        )
        self.metrics["decisions"] += 1
        self.metrics["simulation_ticks"] += result.ticks_advanced
        self.metrics["instant_actions"] += int(result.ticks_advanced == 0)
        if result.action_result.accepted and not isinstance(concrete, Wait):
            self.metrics["agent_actions"] += 1
            self.actions_this_tick += 1
            self.metrics["max_actions_per_tick"] = max(
                self.metrics["max_actions_per_tick"], self.actions_this_tick
            )
        if result.ticks_advanced:
            self.actions_this_tick = 0
        if self.training and not uses_games(self.cfg):
            # Every vector worker advances once per collection step. Updating here
            # makes the new stage visible before VecEnv's automatic episode reset.
            self.progress += self.cfg["training"]["n_envs"]
        self.metrics["invalid_actions"] += int(
            not result.action_result.accepted or rejected_strategy
        )
        self.metrics["wait_actions"] += int(isinstance(concrete, Wait))
        if isinstance(concrete, Place) and result.action_result.accepted:
            self.planted_ticks[concrete.row, concrete.col] = before.tick
            self.plant_usage[concrete.plant_type] += 1
            self.plant_spending[concrete.plant_type] += self.rules.plants[concrete.plant_type][
                "cost"
            ]
            if concrete.plant_type in attackers:
                self.metrics["attacker_purchases"] += 1
                if self.first_attacker_tick is None:
                    self.first_attacker_tick = before.tick
        if isinstance(concrete, Dig) and result.action_result.accepted:
            planted_at = self.planted_ticks.pop((concrete.row, concrete.col), None)
            self.metrics["early_voluntary_digs"] += int(
                planted_at is not None
                and before.tick - planted_at
                <= self.cfg.get("diagnostics", {}).get("early_dig_seconds", 5) * before.tick_rate
            )
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
            "episode_start_stage": self.episode_stage,
            "early_voluntary_digs": self.metrics["early_voluntary_digs"],
            "scenario_seed": self.episode_seed,
            "status": self.state.value,
            "win": int(self.state == EpisodeState.WON),
            "tick": obs.tick,
            "simulated_seconds": obs.elapsed_seconds,
            "return": self.episode_reward,
            "decisions": self.metrics["decisions"],
            "agent_actions": self.metrics["agent_actions"],
            "simulation_ticks": self.metrics["simulation_ticks"],
            "instant_actions": self.metrics["instant_actions"],
            "max_actions_per_tick": self.metrics["max_actions_per_tick"],
            "action_timing": "per_tick" if self.per_tick else "fixed",
            "attacker_purchases": self.metrics["attacker_purchases"],
            **{key: self.metrics[key] for key in (*REWARD_METRICS, *LEDGER_METRICS)},
            "first_attacker_seconds": None
            if self.first_attacker_tick is None
            else self.first_attacker_tick / obs.tick_rate,
            "maximum_sun": self.maximum_sun,
            "plant_spending": dict(self.plant_spending),
            "affordable_attacker_opportunities": self.metrics["affordable_attacker_opportunities"],
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
