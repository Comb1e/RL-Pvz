"""Interruptible, bounded simulator search with executable public-state policies.

The online model is frozen by the cohort lifecycle. Discovery and confirmation
never train it, and only completed confirmation panels publish fitting records.
"""

from copy import deepcopy
from enum import StrEnum
from sys import getsizeof
from time import perf_counter

import numpy as np
import torch
from pvz_game import Status
from pvz_game.cuda.backend import projectile_bound

from pvz_rl.envs.actions import ActionCodec
from pvz_rl.envs.cuda_accounting import AccountingCudaBatch
from pvz_rl.envs.cuda_features import REWARD_FIELDS, CudaFeatures
from pvz_rl.envs.encoding import ObservationEncoder, collate_observations
from pvz_rl.envs.search_worlds import (
    PublicWorldSampler,
    WorldGenerationError,
    legal_actions,
    public_observations,
)
from pvz_rl.evaluation.controllers import PublicBoard, strategy_candidates
from pvz_rl.evaluation.frozen_baseline import choose_action
from pvz_rl.learning.action_routing import FitRoute
from pvz_rl.learning.objective import (
    COMPONENTS,
    residual_sun_adjustment,
    residual_sun_wait_penalties,
    training_rewards,
)
from pvz_rl.learning.search_settings import (
    SEARCH_PROTOCOL,
    TARGET_PROTOCOL,
    validate_search_recovery,
)
from pvz_rl.learning.search_targets import (
    frozen_time_adjustment,
    make_teacher_record,
    paired_targets,
)
from pvz_rl.policy.runner import PolicyRunner


class SearchPhase(StrEnum):
    QUEUED = "queued"
    PILOT = "pilot"
    REFERENCE = "reference_frozen"
    DISCOVERY = "discovery"
    CONFIRMATION = "confirmation"
    READY = "ready"
    COMMITTED = "action_committed"


def frontier_facts(obs):
    """Public events which make a fresh decision useful, independent of time."""
    return (
        obs.wave,
        len(obs.plants),
        tuple((c.cooldown_ticks == 0, obs.sun >= c.cost) for c in obs.cards),
        tuple((z.row, z.zombie_type, z.x // obs.units_per_tile) for z in obs.zombies),
    )


def owned_bytes(value, seen=None):
    """Measure immutable search artifacts once when admitted, never each tick."""
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    size = getsizeof(value)
    if isinstance(value, dict):
        size += sum(owned_bytes(k, seen) + owned_bytes(v, seen) for k, v in value.items())
    elif isinstance(value, (tuple, list)):
        size += sum(owned_bytes(v, seen) for v in value)
    elif hasattr(value, "__dict__"):
        size += owned_bytes(vars(value), seen)
    elif isinstance(value, np.ndarray) and not value.flags.owndata:
        size += value.nbytes
    return size


def public_key(obs, root_tick, cfg, allowed, digging):
    """Search abstraction only. Complete occupancy/readiness remain exact."""
    rate, unit = obs.tick_rate, obs.units_per_tile
    return (
        allowed,
        digging,
        obs.wave,
        obs.sun // 25,
        (obs.tick - root_tick) // round(cfg["training"]["search"]["frontier_seconds"] * rate),
        tuple(
            (c.cooldown_ticks == 0, obs.sun >= c.cost, c.cooldown_ticks // rate) for c in obs.cards
        ),
        tuple(
            sorted(
                (
                    p.row,
                    p.col,
                    p.plant_type,
                    p.state,
                    4 * p.health // p.max_health,
                    p.timer_ticks // rate,
                )
                for p in obs.plants
            )
        ),
        tuple(
            sorted(
                (
                    z.row,
                    z.zombie_type,
                    z.x // unit,
                    4 * z.health // max(1, cfg["_search_hp"][z.zombie_type]),
                    z.armor // 100,
                    z.state,
                    z.timer_ticks // rate,
                    z.slow_ticks // rate,
                    z.has_pole,
                    z.headless,
                )
                for z in obs.zombies
            )
        ),
        tuple(sorted((p.row, p.x // unit, p.icy) for p in obs.projectiles)),
        tuple((m.row, m.x // unit, m.state) for m in obs.mowers),
    )


class PolicyOracle:
    def __init__(self, policy, cfg):
        self.policy, self.cfg = policy, cfg
        self.runner = PolicyRunner(policy)
        self.codec = ActionCodec(cfg)
        self.encoder = ObservationEncoder(cfg, __import__("pvz_game").Rules())

    @torch.no_grad()
    def proposals(self, observations, *, pea=False, allowed=255, digging=True):
        boards = collate_observations(
            [self.encoder.encode(o) for o in observations], self.policy.device
        )
        usable = np.ones((len(observations), 10), bool)
        for i, o in enumerate(observations):
            usable[i, 1:9] = [
                bool(allowed & (1 << n)) and not c.cooldown_ticks and o.sun >= c.cost
                for n, c in enumerate(o.cards)
            ]
            usable[i, 9] = digging and bool(o.plants)
        supplied = (
            torch.tensor(
                [2 if usable[i, 2] else 0 for i in range(len(observations))],
                device=self.policy.device,
            )
            if pea
            else None
        )
        result = self.runner.decide(
            boards,
            torch.ones((len(observations), 406), dtype=torch.bool, device=self.policy.device),
            supplied_branches=supplied,
            branch_usable=torch.tensor(usable, device=self.policy.device),
        )
        return result[0].cpu().numpy(), result[3]["branch_q"].float().cpu().numpy(), boards

    @torch.no_grad()
    def candidates(self, obs, rng, limit, *, pea, allowed, digging, rotation=0):
        raw, species, board = self.proposals([obs], pea=pea, allowed=allowed, digging=digging)
        legal = legal_actions(obs, self.cfg, allowed=allowed, digging=digging, peashooter=pea)
        legal_set = set(map(int, legal))
        branches = sorted({(int(a) - 1) // 45 + 1 for a in legal if a})
        values = {}
        if branches:
            tables = (
                self.policy.conditional_tiles(board, branches, rows=[0] * len(branches))
                .float()
                .cpu()
                .numpy()
            )
            values = dict(zip(branches, tables))
        selected = [0]

        def add(a):
            if a is not None and a in legal_set and a not in selected and len(selected) < limit:
                selected.append(int(a))

        def top(b, n):
            if b not in values:
                return
            tiles = [t for t in range(45) if 1 + (b - 1) * 45 + t in legal_set]
            # Randomized ties keep BF16 quantization from fixing proposal diversity.
            rng.shuffle(tiles)
            for t in sorted(tiles, key=lambda t: -float(values[b][t]))[:n]:
                add(1 + (b - 1) * 45 + t)

        if pea:
            top(2, 2)
        else:
            top(9, 1)
            for b in sorted((b for b in branches if b < 9), key=lambda b: -species[0, b])[:2]:
                top(b, 2)
        public = PublicBoard(obs, frozenset(self.codec.decode(a) for a in legal))
        proposals = list(strategy_candidates(public)[1:])
        proposals = proposals[rotation % 4 :] + proposals[: rotation % 4]
        for proposal in proposals:
            if proposal is not None:
                add(self.codec.encode(proposal))
                break
        remaining = [int(a) for a in legal if a not in selected]
        rng.shuffle(remaining)
        for a in remaining:
            add(a)
        return selected, legal, int(raw[0])

    def fallback(self, obs, raw, family, *, pea, allowed, digging):
        if family == 0:
            return int(raw)
        legal = legal_actions(obs, self.cfg, allowed=allowed, digging=digging, peashooter=pea)
        board = PublicBoard(obs, frozenset(self.codec.decode(a) for a in legal))
        if pea:
            attack = strategy_candidates(board)[2]
            if attack is not None:
                return self.codec.encode(attack)
            return next((int(a) for a in legal if a), 0)
        return self.codec.encode(choose_action(board))


def uct_choice(sums, counts, rng, c):
    counts = np.asarray(counts)
    unvisited = np.flatnonzero(counts == 0)
    if len(unvisited):
        return int(rng.choice(unvisited))
    mean = np.asarray(sums) / counts
    span = float(np.ptp(mean))
    normalized = (mean - mean.min()) / span if span > 1e-12 else np.zeros_like(mean)
    score = normalized + c * np.sqrt(np.log(counts.sum() + 1) / counts)
    return int(rng.choice(np.flatnonzero(np.isclose(score, score.max(), rtol=0, atol=1e-12))))


class ScratchPool:
    """Owned CUDA state. Only active public prefixes cross the frontier boundary."""

    def __init__(self, cfg, rules, capacity, rate):
        import cupy as cp

        settings = cfg["training"]["search"]
        self.cfg = cfg
        qcap = 2 * projectile_bound(rules)
        # Exact major storage plus a conservative allowance for feature temporaries.
        per_lane = qcap * 6 * 8 + capacity * 32 * 8 + cfg["encoding"]["max_entities"] * 64 + 65536
        free, _ = cp.cuda.runtime.memGetInfo()
        free += cp.get_default_memory_pool().free_bytes()
        headroom = int(
            cfg["training"]["performance"].get("collection_vram_headroom_mib", 0) * 1024**2
        )
        budget = min(settings["scratch_vram_mib"] * 1024**2, max(0, free - headroom))
        n = min(
            settings["scratch_envs"],
            max(1, settings["root_candidates"] * settings["discovery_worlds"]),
        )
        while n and per_lane * n > budget:
            n //= 2
        if not n:
            raise MemoryError(
                "Search scratch pool cannot fit one lane; reduce main environments or free GPU memory"
            )
        while n:
            try:
                self.batch = AccountingCudaBatch(
                    n,
                    zombie_capacity=capacity,
                    projectile_capacity=qcap,
                    max_step_ticks=cfg["environment"]["decision_ticks"],
                    rules=rules,
                    device=torch.cuda.current_device(),
                )
                self.batch.set_sky_sun_rate(rate)
                self.features = CudaFeatures(self.batch, cfg, "masked")
                self.n = n
                self.jobs = [None] * n
                self.bytes = per_lane * n
                return
            except (cp.cuda.memory.OutOfMemoryError, torch.cuda.OutOfMemoryError):
                self.__dict__.pop("features", None)
                self.__dict__.pop("batch", None)
                cp.get_default_memory_pool().free_all_blocks()
                n //= 2
        raise MemoryError("Search scratch allocation failed down to one lane")

    def restore_jobs(self, snapshots, query):
        indices = sorted(snapshots)
        if indices:
            self.batch.restore(
                [snapshots[i] for i in indices],
                indices=indices,
                allowed=[query["allowed"]] * len(indices),
                digging=[query["digging"]] * len(indices),
            )
            self.features.encode()


class SearchTeacher:
    """One serial query coordinator; many independent rollout lanes per query."""

    def __init__(self, policy, cfg, seed, state=None):
        self.cfg = deepcopy(cfg)
        self.rules = __import__("pvz_game").Rules()
        self.cfg["_search_hp"] = {k: r["health"] for k, r in self.rules.zombies.items()}
        self.settings = cfg["training"]["search"]
        self.oracle = PolicyOracle(policy, self.cfg)
        self.sampler = PublicWorldSampler(cfg, self.rules)
        self.rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0x53454152]))
        self.query = None
        self.pool = None
        self.records = []
        self.record_bytes = 0
        self.plans = {}
        self.slots = {}
        self.prefix = {}
        self.behavior = []
        self.prepared = None
        self.policy_identity = None
        self.metrics = dict(
            queries=0,
            generation_failures=0,
            rollouts=0,
            simulated_ticks=0,
            wall_seconds=0.0,
            root_actions=0,
            continuation_actions=0,
            fallback_actions=0,
            overrides=0,
            duration_reversals=0,
        )
        if state is not None:
            self.restore(state)

    def key(self, obs, root_tick, allowed, digging):
        return public_key(obs, root_tick, self.cfg, allowed, digging)

    def begin_cohort(self, cfg=None, identity=None):
        if self.query:
            raise RuntimeError("Cannot begin a cohort with pending search")
        self.records.clear()
        self.record_bytes = 0
        self.plans.clear()
        self.slots.clear()
        self.prefix.clear()
        self.behavior.clear()
        self.prepared = None
        if cfg is not None:
            self.cfg["reward"] = deepcopy(cfg["reward"])
        self.policy_identity = identity

    def _new_query(self, slot, obs, level, *, pea, allowed, digging, rate):
        candidates, legal, raw = self.oracle.candidates(
            obs,
            self.rng,
            self.settings["root_candidates"],
            pea=pea,
            allowed=allowed,
            digging=digging,
            rotation=self.metrics["queries"],
        )
        if len(candidates) < 2 or pea and not any(candidates):
            return False
        query = dict(
            slot=slot,
            observation=obs,
            level=level,
            pea=pea,
            allowed=allowed,
            digging=digging,
            rate=rate,
            candidates=candidates,
            legal=legal,
            raw=raw,
            phase=SearchPhase.QUEUED,
            worlds=[],
            confirmation_worlds=[],
            tree={},
            reference=[],
            results=[],
            jobs_issued=0,
            root_sums=np.zeros(len(candidates)),
            root_counts=np.zeros(len(candidates), np.int64),
            prefix=deepcopy(self.prefix.get(slot, [])),
            representatives={},
            frozen={},
            samples=np.full((len(candidates), self.settings["confirmation_worlds"]), np.nan),
            snapshots={},
            jobs=[],
            parked=[],
            tree_bytes=0,
            world_bytes=0,
            spent_ticks=0,
        )
        self.query = query
        return True

    def prepare(self, env, ready, *, pea=False):
        """Called outside collection's atomic action transaction; one bounded unit."""
        if self.query is not None and self.query["phase"] != SearchPhase.READY:
            self.advance()
            return False
        if self.prepared is None:
            indices = list(map(int, np.flatnonzero(ready)))
            observations = public_observations(env.batch, env._level_names, indices)
            raw, _, _ = (
                self.oracle.proposals(
                    observations, pea=pea, allowed=(2 if pea else 255), digging=not pea
                )
                if indices
                else ([], None, None)
            )
            self.prepared = dict(
                indices=indices,
                observations=observations,
                raw=list(map(int, raw)),
                cursor=0,
                actions={},
                sources={},
            )
        prepared = self.prepared
        while prepared["cursor"] < len(prepared["indices"]):
            i = prepared["cursor"]
            slot, obs, raw = prepared["indices"][i], prepared["observations"][i], prepared["raw"][i]
            if self.query is not None:
                q = self.query
                assert q["slot"] == slot and q["phase"] == SearchPhase.READY
                plan = dict(
                    root_tick=obs.tick,
                    tree=q["frozen"],
                    family=int(self.rng.integers(2)),
                    allowed=q["allowed"],
                    digging=q["digging"],
                    pea=pea,
                    estimated_bytes=q["tree_bytes"],
                )
                self.plans[slot] = plan
                prepared["actions"][slot] = q["chosen"]
                prepared["sources"][slot] = "searched_root"
                self.records.append(q["record"])
                self.record_bytes += owned_bytes(q["record"])
                self.metrics["queries"] += 1
                self.metrics["root_actions"] += 1
                data = self.slots.setdefault(slot, dict(count=0, last=-(10**12), checked=-(10**12)))
                data.update(count=data["count"] + 1, last=obs.tick, checked=obs.tick)
                q["phase"] = SearchPhase.COMMITTED
                self.metrics["active_phase"] = "following_plan"
                self.query = None
                self.pool = None
                prepared["cursor"] += 1
                continue
            data = self.slots.setdefault(slot, dict(count=0, last=-(10**12), checked=-(10**12)))
            allowed, digging = (2, False) if pea else (255, True)
            legal = legal_actions(obs, self.cfg, allowed=allowed, digging=digging, peashooter=pea)
            interval = round(self.settings["minimum_interval_seconds"] * obs.tick_rate)
            public = frontier_facts(obs)
            frontier = (
                raw != 0
                and raw in legal
                or public != data.get("public")
                or obs.tick - data["checked"]
                >= round(self.settings["frontier_seconds"] * obs.tick_rate)
            )
            data["public"] = public
            eligible = (
                len(legal) > 1
                and data["count"] < self.settings["queries_per_episode"]
                and obs.tick - data["last"] >= interval
            )
            ask = False
            if eligible and (frontier or raw not in legal):
                data["checked"] = obs.tick
                probability = self.settings[
                    "peashooter_query_probability" if pea else "normal_query_probability"
                ]
                ask = data["count"] == 0 or raw not in legal or self.rng.random() < probability
            if ask:
                level = "easy" if pea else env._level_names[slot]
                family = env._episode[slot][1] if env._episode[slot] is not None else env.family
                if not pea and family != "preset":
                    raise ValueError(
                        "Search requires a supported public scenario generator; use preset family"
                    )
                if self._new_query(
                    slot,
                    obs,
                    level,
                    pea=pea,
                    allowed=allowed,
                    digging=digging,
                    rate=env.batch.sky_sun_rate,
                ):
                    self.advance()
                    return False
            if slot in self.plans:
                plan = self.plans[slot]
                key = self.key(obs, plan["root_tick"], allowed, digging)
                proposed = plan["tree"].get(key)
                source = "planned_continuation"
                if proposed not in legal:
                    proposed = self.oracle.fallback(
                        obs, raw, plan["family"], pea=pea, allowed=allowed, digging=digging
                    )
                    source = "public_fallback" if plan["family"] else "network"
                prepared["actions"][slot] = proposed
                prepared["sources"][slot] = source
                self.metrics[
                    "continuation_actions"
                    if source == "planned_continuation"
                    else "fallback_actions"
                ] += 1
            prepared["cursor"] += 1
        return True

    def consume(self):
        if self.prepared is None:
            return {}, {}
        result = self.prepared["actions"], self.prepared["sources"]
        self.prepared = None
        return result

    def observe(self, rows, sources):
        for row in rows[rows["active"]]:
            slot = int(row["env"])
            eligible = bool(row["accepted"] and row["action"] == 0 and row["exploration_mode"] == 0)
            self.prefix.setdefault(slot, []).append((int(row["tick"]), eligible))
            self.behavior.append(
                (
                    slot,
                    int(row["episode"]),
                    int(row["tick"]),
                    sources.get(slot, "exploration" if row["exploration_mode"] else "network"),
                    int(row["policy_action"]),
                    int(row["action"]),
                    bool(row["accepted"]),
                    int(row["rejection_reason"]),
                )
            )
            if row["done"]:
                self.plans.pop(slot, None)
                self.slots.pop(slot, None)
                self.prefix.pop(slot, None)
        if not self.plans and self.query is None:
            self.metrics["active_phase"] = "idle"

    def _ensure_pool(self):
        q = self.query
        if self.pool is None:
            self.pool = ScratchPool(
                self.cfg, self.rules, max(1, q["observation"].counts.initial_total), q["rate"]
            )
            if q["snapshots"]:
                # Running paths can wait in owned host storage when recovery has
                # less VRAM. Keep their progress, rewards and original world IDs.
                q["parked"].extend(
                    (job, q["snapshots"][i]) for i, job in enumerate(q["jobs"]) if job is not None
                )
            q["snapshots"] = {}
            q["jobs"] = []

    def _job(self, candidate, world, family, phase):
        return dict(
            candidate=candidate,
            world=world,
            family=family,
            phase=phase,
            first=True,
            reward=0.0,
            waits=[],
            path=[],
            depth=0,
            expanded=0,
            last_key=None,
            last_frontier=-(10**12),
            last_public=None,
        )

    def _fill(self):
        q, pool = self.query, self.pool
        phase = q["phase"]
        k = len(q["candidates"])
        worlds = (
            self.settings["confirmation_worlds"]
            if phase == SearchPhase.CONFIRMATION
            else self.settings["discovery_worlds"]
        )
        total = (
            k * worlds
            if phase != SearchPhase.DISCOVERY
            else self.settings["rollouts"]
            - k * (self.settings["discovery_worlds"] + self.settings["confirmation_worlds"])
        )
        snapshots, indices = [], []
        virtual = q["root_counts"].copy()
        estimates = q["root_sums"] / np.maximum(1, q["root_counts"])
        parked_snapshots = {}
        for slot, job in enumerate(pool.jobs):
            if job is not None:
                virtual[job["candidate"]] += 1
                continue
            if q["parked"]:
                job, snapshot = q["parked"].pop(0)
                pool.jobs[slot] = job
                parked_snapshots[slot] = snapshot
                virtual[job["candidate"]] += 1
                continue
            if q["jobs_issued"] >= total:
                continue
            ordinal = q["jobs_issued"]
            if phase == SearchPhase.DISCOVERY:
                candidate = uct_choice(
                    estimates * virtual, virtual, self.rng, self.settings["uct_c"]
                )
                world = ordinal % worlds
                virtual[candidate] += 1
            else:
                candidate, world = divmod(ordinal, worlds)
            pool.jobs[slot] = self._job(candidate, world, world % 2, phase)
            snapshots.append(
                (q["confirmation_worlds"] if phase == SearchPhase.CONFIRMATION else q["worlds"])[
                    world
                ]
            )
            indices.append(slot)
            q["jobs_issued"] += 1
        pool.restore_jobs(parked_snapshots, q)
        for slot in parked_snapshots:
            accounting = pool.jobs[slot].pop("accounting")
            pool.features.totals[slot] = pool.batch.cp.asarray(accounting["totals"])
            pool.features.home_ledger[slot] = pool.batch.cp.asarray(accounting["home_ledger"])
        if indices:
            pool.batch.restore(
                snapshots,
                indices=indices,
                allowed=[q["allowed"]] * len(indices),
                digging=[q["digging"]] * len(indices),
            )
            pool.features.totals[indices] = 0
            pool.features.initialize_home(indices)
            pool.features.encode()

    def _continuation(self, obs, raw, job):
        q = self.query
        fallback = self.oracle.fallback(
            obs, raw, job["family"], pea=q["pea"], allowed=q["allowed"], digging=q["digging"]
        )
        legal = legal_actions(
            obs, self.cfg, allowed=q["allowed"], digging=q["digging"], peashooter=q["pea"]
        )
        key = self.key(obs, q["observation"].tick, q["allowed"], q["digging"])
        if job["phase"] == SearchPhase.CONFIRMATION:
            action = q["frozen"].get(key)
            return action if action in legal else fallback
        if job["phase"] == SearchPhase.PILOT:
            return fallback
        public = frontier_facts(obs)
        frontier = (
            (fallback != 0 and fallback in legal)
            or public != job["last_public"]
            or obs.tick - job["last_frontier"] >= self.settings["frontier_seconds"] * obs.tick_rate
        )
        job["last_public"] = public
        if not frontier or job["depth"] >= self.settings["max_depth"]:
            return fallback
        job["last_frontier"] = obs.tick
        job["depth"] += 1
        if key not in q["tree"]:
            if (
                len(q["tree"]) >= self.settings["max_nodes"]
                or job["expanded"] >= self.settings["expansions_per_rollout"]
            ):
                return fallback
            actions, _, _ = self.oracle.candidates(
                obs,
                self.rng,
                self.settings["node_candidates"],
                pea=q["pea"],
                allowed=q["allowed"],
                digging=q["digging"],
            )
            if q["pea"] and any(actions):
                actions = [a for a in actions if a]
            q["tree"][key] = dict(
                actions=actions,
                sums=np.zeros(len(actions)),
                counts=np.zeros(len(actions), np.int64),
            )
            q["tree_bytes"] += owned_bytes(key) + owned_bytes(q["tree"][key])
            job["expanded"] += 1
        node = q["tree"][key]
        choice = uct_choice(node["sums"], node["counts"], self.rng, self.settings["uct_c"])
        action = node["actions"][choice]
        if action not in legal:
            return fallback
        job["path"].append((key, choice))
        return action

    def _utility(self, job, obs):
        q = self.query
        gamma = self.cfg["training"]["gamma"]
        root_tick = q["observation"].tick
        end_factor = gamma ** max(0, job["waits"][-1][0] - root_tick)
        value = job["reward"]
        won = obs.status == Status.WON
        if won:
            value += end_factor * residual_sun_adjustment(obs.sun, True, self.cfg)
        else:
            # Reuse the exact per-episode raw wait-run definition, including prefix.
            amount = -residual_sun_adjustment(obs.sun, False, self.cfg)
            waits = q["prefix"] + job["waits"]
            if waits and amount:
                rows = np.zeros(
                    len(waits),
                    dtype=[
                        ("active", "?"),
                        ("accepted", "?"),
                        ("action", "i8"),
                        ("route", "i1"),
                        ("exploration_mode", "i1"),
                        ("decision", "i8"),
                    ],
                )
                rows["active"] = True
                rows["accepted"] = [eligible for _, eligible in waits]
                rows["route"] = int(FitRoute.WAIT_RETURN)
                rows["decision"] = np.arange(len(rows))
                penalties, _ = residual_sun_wait_penalties(rows, amount)
                discounts = np.power(
                    gamma, np.array([tick - root_tick for tick, _ in waits], dtype=np.float64)
                )
                value += float(np.sum(penalties * discounts))
        adjustment = (
            frozen_time_adjustment(
                q["reference"], obs.tick, won, self.cfg["reward"]["win_time_weight"]
            )
            if q["reference"]
            else 0.0
        )
        return value + end_factor * adjustment, adjustment

    def advance(self):
        started = perf_counter()
        try:
            self._advance()
        finally:
            self.metrics["wall_seconds"] += perf_counter() - started
            if self.query is not None:
                self.metrics.update(
                    active_phase=str(self.query["phase"]),
                    query_rollouts=len(self.query["results"]),
                    query_ticks=self.query["spent_ticks"],
                    scratch_lanes=self.pool.n if self.pool else 0,
                    scratch_reserved_bytes=self.pool.bytes if self.pool else 0,
                    rollout_budget_fraction=len(self.query["results"]) / self.settings["rollouts"],
                )

    @torch.no_grad()
    def _advance(self):
        q = self.query
        if q["phase"] == SearchPhase.QUEUED:
            try:
                for name, count in (
                    ("worlds", self.settings["discovery_worlds"]),
                    ("confirmation_worlds", self.settings["confirmation_worlds"]),
                ):
                    if len(q[name]) < count:
                        failures = self.sampler.failures
                        try:
                            world = self.sampler.sample(
                                q["observation"], q["level"], self.rng, rate=q["rate"]
                            )
                        finally:
                            self.metrics["generation_failures"] += self.sampler.failures - failures
                        q[name].append(world)
                        q["world_bytes"] += owned_bytes(world)
                        return
                q["phase"] = SearchPhase.PILOT
            except WorldGenerationError:
                # Mark this attempted query, leave the game/previous plan intact.
                data = self.slots.setdefault(q["slot"], dict(count=0))
                data.update(
                    count=data["count"] + 1,
                    last=q["observation"].tick,
                    checked=q["observation"].tick,
                )
                self.query = None
                return
        if q["phase"] == SearchPhase.REFERENCE:
            q["reference"] = [
                r["duration"] for r in q["results"] if r["phase"] == SearchPhase.PILOT
            ]
            for r in q["results"]:
                score = r["base"] + r["factor"] * frozen_time_adjustment(
                    q["reference"], r["duration"], r["won"], self.cfg["reward"]["win_time_weight"]
                )
                q["root_sums"][r["candidate"]] += score
                q["root_counts"][r["candidate"]] += 1
            q["phase"], q["jobs_issued"] = SearchPhase.DISCOVERY, 0
        self._ensure_pool()
        self._fill()
        if not any(job is not None for job in self.pool.jobs):
            if q["phase"] == SearchPhase.PILOT:
                q["phase"] = SearchPhase.REFERENCE
            elif q["phase"] == SearchPhase.DISCOVERY:
                for key, node in q["tree"].items():
                    visited = np.flatnonzero(node["counts"])
                    if len(visited):
                        best = max(visited, key=lambda i: node["sums"][i] / node["counts"][i])
                        q["frozen"][key] = node["actions"][best]
                means = q["root_sums"] / np.maximum(1, q["root_counts"])
                for j, a in enumerate(q["candidates"]):
                    b = (a - 1) // 45 + 1 if a else 0
                    old = q["representatives"].get(b)
                    if old is None or means[j] > means[old]:
                        q["representatives"][b] = j
                q["phase"], q["jobs_issued"] = SearchPhase.CONFIRMATION, 0
            elif q["phase"] == SearchPhase.CONFIRMATION:
                self._finish()
            return
        pool = self.pool
        active = [i for i, j in enumerate(pool.jobs) if j is not None]
        observations = public_observations(pool.batch, [q["observation"].level] * pool.n, active)
        raw, _, _ = self.oracle.proposals(
            observations, pea=q["pea"], allowed=q["allowed"], digging=q["digging"]
        )
        actions = np.zeros(pool.n, np.int64)
        for i, obs, proposal in zip(active, observations, raw):
            job = pool.jobs[i]
            actions[i] = (
                q["candidates"][job["candidate"]]
                if job["first"]
                else self._continuation(obs, int(proposal), job)
            )
            job["first"] = False
        cp = pool.batch.cp
        pool.batch.header[:, 17] = cp.asarray([j is not None for j in pool.jobs], dtype=cp.int64)
        pool.features.step(
            cp.asarray(actions),
            ticks=self.cfg["environment"]["decision_ticks"],
            per_tick=self.cfg["environment"]["action_timing"] == "per_tick",
            publish=False,
        )
        header = pool.batch.header.get()
        parts = pool.features.parts[:, [REWARD_FIELDS.index(k) for k in COMPONENTS]].get()
        fitted = training_rewards(parts, self.cfg)
        done = []
        for i, before in zip(active, observations):
            h, job = header[i], pool.jobs[i]
            advanced = int(h[14])
            q["spent_ticks"] += advanced
            self.metrics["simulated_ticks"] += advanced
            job["reward"] += self.cfg["training"]["gamma"] ** (
                before.tick - q["observation"].tick
            ) * float(fitted[i])
            job["waits"].append((before.tick, bool(h[12] and actions[i] == 0)))
            if h[1] != 0 or h[0] >= self.cfg["environment"]["cutoff_seconds"] * before.tick_rate:
                done.append(i)
        cap = self.settings["rollouts"] * max(
            0,
            self.cfg["environment"]["cutoff_seconds"] * self.rules.game["tick_rate"]
            - q["observation"].tick,
        )
        if q["spent_ticks"] > cap:
            raise RuntimeError("Search exceeded its global simulated-tick budget")
        endings = (
            public_observations(pool.batch, [q["observation"].level] * pool.n, done) if done else []
        )
        for i, obs in zip(done, endings):
            job = pool.jobs[i]
            score, adjustment = self._utility(job, obs)
            factor = self.cfg["training"]["gamma"] ** (job["waits"][-1][0] - q["observation"].tick)
            q["results"].append(
                dict(
                    phase=job["phase"],
                    candidate=job["candidate"],
                    duration=obs.tick,
                    won=obs.status == Status.WON,
                    base=score - factor * adjustment,
                    factor=factor,
                    adjustment=adjustment,
                )
            )
            if job["phase"] == SearchPhase.CONFIRMATION:
                q["samples"][job["candidate"], job["world"]] = score
            elif job["phase"] == SearchPhase.DISCOVERY:
                q["root_sums"][job["candidate"]] += score
                q["root_counts"][job["candidate"]] += 1
                for key, choice in job["path"]:
                    q["tree"][key]["sums"][choice] += score
                    q["tree"][key]["counts"][choice] += 1
            pool.jobs[i] = None
            self.metrics["rollouts"] += 1

    def _finish(self):
        q = self.query
        means, _, _, _ = paired_targets(q["samples"])
        eligible = [j for j, a in enumerate(q["candidates"]) if not q["pea"] or a]
        best = max(means[j] for j in eligible)
        ties = [j for j in eligible if best - means[j] <= self.settings["tie_tolerance"]]
        j = next((j for j in ties if q["candidates"][j] == q["raw"]), None)
        if j is None:
            j = int(self.rng.choice(ties))
        q["chosen"] = q["candidates"][j]
        self.metrics["overrides"] += q["chosen"] != q["raw"]
        encoded = self.oracle.encoder.encode(q["observation"])
        obs = q["observation"]
        usable = np.array(
            [True]
            + [
                bool(q["allowed"] & (1 << n)) and not c.cooldown_ticks and obs.sun >= c.cost
                for n, c in enumerate(obs.cards)
            ]
            + [q["digging"] and bool(obs.plants)]
        )
        q["record"] = make_teacher_record(
            encoded,
            q["candidates"],
            q["samples"],
            q["legal"],
            q["representatives"],
            peashooter=q["pea"],
            branch_usable=usable,
        )
        q["record"]["provenance"] = dict(
            protocol=TARGET_PROTOCOL,
            reference=list(q["reference"]),
            raw=q["raw"],
            chosen=q["chosen"],
            rate=q["rate"],
            results=q["results"],
        )
        confirmation = [r for r in q["results"] if r["phase"] == SearchPhase.CONFIRMATION]
        base = np.array(
            [
                np.mean([r["base"] for r in confirmation if r["candidate"] == i])
                for i in range(len(q["candidates"]))
            ]
        )
        self.metrics["duration_reversals"] += int(
            sum(
                (base[a] - base[b]) * (means[a] - means[b]) < 0
                for a in range(len(means))
                for b in range(a + 1, len(means))
            )
        )
        self.metrics.update(
            discovery_score=float(np.max(q["root_sums"] / np.maximum(1, q["root_counts"]))),
            confirmation_score=float(means[j]),
            confirmation_duration_adjustment=float(
                np.mean([r["adjustment"] for r in confirmation if r["candidate"] == j])
            ),
            confirmation_standard_error=float(
                q["samples"][j].std(ddof=1) / np.sqrt(q["samples"].shape[1])
            ),
        )
        q["phase"] = SearchPhase.READY

    @property
    def estimated_bytes(self):
        decisions = sum(len(v) for v in self.prefix.values()) + len(self.behavior)
        artifacts = self.record_bytes + sum(p["estimated_bytes"] for p in self.plans.values())
        if self.query:
            jobs = self.pool.jobs if self.pool else self.query["jobs"]
            decisions += len(self.query["prefix"]) + sum(len(j["waits"]) for j in jobs if j)
            decisions += sum(len(job["waits"]) for job, _ in self.query["parked"])
            artifacts += self.query["world_bytes"] + self.query["tree_bytes"]
            artifacts += sum(owned_bytes(raw) for _, raw in self.query["parked"])
        return artifacts + decisions * 384 + (self.pool.bytes if self.pool else 0)

    def snapshot(self):
        state = dict(
            protocol=SEARCH_PROTOCOL,
            target_protocol=TARGET_PROTOCOL,
            settings=deepcopy(self.settings),
            policy_identity=self.policy_identity,
            rng=deepcopy(self.rng.bit_generator.state),
            query=deepcopy(self.query),
            records=self.records,
            plans=self.plans,
            slots=self.slots,
            prefix=self.prefix,
            behavior=self.behavior,
            prepared=self.prepared,
            metrics=dict(self.metrics),
        )
        if self.pool is not None and state["query"] is not None:
            state["query"]["jobs"] = deepcopy(self.pool.jobs)
            for i, job in enumerate(state["query"]["jobs"]):
                if job is not None:
                    job["accounting"] = dict(
                        totals=self.pool.features.totals[i].get(),
                        home_ledger=self.pool.features.home_ledger[i].get(),
                    )
            state["query"]["snapshots"] = {
                i: self.pool.batch.snapshot(i)
                for i, j in enumerate(self.pool.jobs)
                if j is not None
            }
        return state

    def restore(self, state):
        validate_search_recovery(state, self.cfg, required=True)
        if (
            state["protocol"] != SEARCH_PROTOCOL
            or state["target_protocol"] != TARGET_PROTOCOL
            or state["settings"] != self.settings
        ):
            raise ValueError("Search recovery protocol or effective settings differ")
        for key in (
            "query",
            "records",
            "plans",
            "slots",
            "prefix",
            "behavior",
            "prepared",
            "metrics",
            "policy_identity",
        ):
            setattr(self, key, deepcopy(state[key]))
        self.rng.bit_generator.state = deepcopy(state["rng"])
        self.record_bytes = owned_bytes(self.records)
        self.pool = None

    def close(self):
        self.pool = None
