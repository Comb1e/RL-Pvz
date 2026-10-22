"""Gymnasium adapter with explicit outcomes and no automatic reset."""

from __future__ import annotations

import copy
from collections import Counter
from enum import StrEnum

import gymnasium as gym
import numpy as np
from pvz_game import Dig, Game, LevelSpec, Place, Rules, Status, Wait, WaveSpec
from pvz_game.replay import Recorder

from pvz_rl.config import (
    load_config,
    runtime_settings,
    validate_config,
)
from pvz_rl.envs.action_credit import CausalCreditLedger
from pvz_rl.envs.action_timing import ActionPhaseGame, per_tick_actions
from pvz_rl.envs.actions import ActionCodec
from pvz_rl.envs.encoding import ObservationEncoder
from pvz_rl.envs.rewards import LEDGER_METRICS, REWARD_METRICS, HomeProximityLedger, reward_parts
from pvz_rl.envs.scenarios import difficulty_weights, scenario
from pvz_rl.evaluation.controllers import PublicBoard, strategy_candidates
from pvz_rl.learning.action_routing import branch_availability
from pvz_rl.learning.budget import budget_target, uses_games
from pvz_rl.learning.curriculum import stage_distribution, teaching_enabled
from pvz_rl.presentation.recordings import ActionPhaseRecorder


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
        self.causal_credit = CausalCreditLedger(self.cfg, self.rules)

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
            family, level = ("preset", task)
        if family == "diagnostic" and (self.options["hybrid"] or not self.options["masked"]):
            raise ValueError("The diagnostic requires a direct-placement masked condition")
        resolved = options.get("scenario")
        if resolved is None:
            resolved = scenario(level, family, game_seed, self.rules, self.cfg)
        if not isinstance(resolved, (str, LevelSpec, WaveSpec)):
            raise TypeError("Scenario must be a preset, LevelSpec or WaveSpec")
        rules = self.rules
        if self.game.rules.digest != rules.digest:
            self.game = ActionPhaseGame(rules) if self.per_tick else Game(rules)
        self.public = self.game.reset(resolved, game_seed)
        self.causal_credit.reset((level, family, game_seed))
        self.causal_credit.seed_observation(self.public)
        self.causal_summary = {
            "source_action_credited_rows": 0,
            "unattributable_events": 0,
            "damage_by_source_plant": Counter(),
            "blocking_by_plant": Counter(),
            "produced_sun_by_source_plant": Counter(),
        }
        self.proximity = HomeProximityLedger(self.public, self.rules)
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
        encoded = self.encoder.encode(self.public)
        return encoded, {
            "status": self.state.value,
            "entity_truncation": self.encoder.last_truncation,
        }

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
        if self._direct_mask is None or not self.cache_legal_actions:
            # The policy deliberately scores every tile. Card availability,
            # cooldown, sunlight and occupancy are simulator outcomes; the
            # resulting rejection executes as one ordinary wait tick.
            mask = np.zeros(self.codec.size, dtype=np.bool_)
            mask[0] = True
            for plant_index in range(len(self.codec.plants)):
                offset = 1 + plant_index * (self.codec.rows * self.codec.cols)
                mask[offset : offset + self.codec.rows * self.codec.cols] = True
            dig_start = 1 + len(self.codec.plants) * self.codec.rows * self.codec.cols
            mask[dig_start:] = True
            self._direct_mask = mask
        return self._direct_mask.copy()

    def branch_availability(self):
        """Return pre-action branch availability without tile geometry."""
        cards = {card.plant_type: card for card in self.public.cards}
        return branch_availability(
            [self.public.sun if self.state == EpisodeState.RUNNING else 0],
            [[cards[kind].cooldown_ticks if kind in cards else -1 for kind in self.codec.plants]],
            [self.rules.plants[kind]["cost"] for kind in self.codec.plants],
            active=[self.state == EpisodeState.RUNNING],
            card_present=[[kind in cards for kind in self.codec.plants]],
        )[0]

    def engine_action_masks(self):
        """Return authoritative engine legality for non-learning baselines."""
        if self.state != EpisodeState.RUNNING:
            return np.zeros(self.action_space.n, dtype=np.bool_)
        if self.options["hybrid"]:
            return np.array([a is not None for a in self.candidates()], dtype=np.bool_)
        mask = np.zeros(self.codec.size, dtype=np.bool_)
        for action in self.public_board().legal:
            mask[self.codec.encode(action)] = True
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
        causal_credits = self.causal_credit.attribute(
            result.events, action_id=self.metrics["decisions"]
        )
        self.causal_credit.register_plant_kinds(self.public)
        current_action_id = self.metrics["decisions"]
        delayed = {key: value for key, value in causal_credits.items() if key != current_action_id}
        self.causal_summary["source_action_credited_rows"] += len(delayed)
        diagnostics = self.causal_credit.last_diagnostics
        self.causal_summary["unattributable_events"] += int(
            diagnostics.get("unattributable_events", 0)
        )
        for name, target in (
            ("damage_by_plant", "damage_by_source_plant"),
            ("blocking_by_plant", "blocking_by_plant"),
            ("sun_by_plant", "produced_sun_by_source_plant"),
        ):
            self.causal_summary[target].update(diagnostics.get(name, {}))
        self.maximum_sun = max(self.maximum_sun, before.sun, self.public.sun)
        # Policy masks are all-45 and only depend on running status. Keep the
        # cache invalidation for active/inactive transitions and preserve the
        # engine legal cache for baselines and diagnostics.
        self._direct_mask = None
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
            action=concrete,
            action_result=result.action_result,
            proximity=self.proximity,
            # Keep aggregate event accounting for effects from unowned
            # initial plants.  Only an origin row present in this transition
            # switches the action fields to the causal ledger values.
            # An explicitly empty mapping means the transition has no
            # source-owned action effects (for example an initial plant).
            # Do not fall back to aggregate event shaping in that case.
            causal_credits=causal_credits.get(self.metrics["decisions"], {}),
        )
        # ``parts`` is the reward returned for the current transition.  A
        # delayed event may belong to an earlier source row; include that
        # source credit in episode totals without changing this tick's return.
        for origin, credit in causal_credits.items():
            if origin == current_action_id:
                continue
            for field, metric in (
                ("damage", "action_damage"),
                ("blocking", "action_blocking"),
                ("sun", "action_sun"),
                ("mower", "action_mower"),
            ):
                self.metrics[metric] += float(credit.get(field, 0.0))
        if truncated:
            parts["terminal"] = -self.cfg["reward"]["loss_penalty"]
            parts["global_terminal"] = parts["terminal"]
            parts["total"] += parts["terminal"]
        for key in REWARD_METRICS:
            self.metrics[key] += parts[key]
        self.episode_reward += parts["total"]
        self.metrics["discounted_return"] += (
            self.cfg["training"]["gamma"] ** self.metrics["simulation_ticks"] * parts["total"]
        )
        for name, part in (("outcome", "terminal"), ("development", "development")):
            self.metrics[f"discounted_{name}_return"] += (
                self.cfg["training"]["gamma"] ** self.metrics["simulation_ticks"] * parts[part]
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
        # Rejected proposals execute as waits.  Count the resolved execution
        # action, while keeping the submitted proposal available for penalties
        # and Q fitting below.
        self.metrics["wait_actions"] += int(
            isinstance(concrete, Wait) or not result.action_result.accepted
        )
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
            "proposal_action": int(action),
            "executed_action": int(action)
            if result.action_result.accepted and not rejected_strategy
            else 0,
            "reward_parts": parts,
            "accepted": result.action_result.accepted and not rejected_strategy,
            "rejection_reason": result.action_result.reason,
            "ticks_advanced": result.ticks_advanced,
            "causal_credits": causal_credits,
            "causal_diagnostics": self.causal_credit.last_diagnostics,
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
        encoded = self.encoder.encode(self.public)
        info["entity_truncation"] = self.encoder.last_truncation
        return encoded, parts["total"], terminated, truncated, info

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
            **self.causal_summary,
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
        from pvz_rl.presentation.rendering import render_observation

        return render_observation(self.public)
