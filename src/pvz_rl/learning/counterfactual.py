"""Bounded factual forks, shared command logs and private completion records.

There is deliberately no policy object in this module. Only the main collector
supplies tile values; scratch lanes execute physics and reward accounting.
"""

import copy
import io
import math
from collections import deque
from enum import IntEnum
from importlib.resources import files
from time import perf_counter

import numpy as np

from pvz_rl.envs.actions import ActionSchema as A
from pvz_rl.envs.cuda_accounting import AccountingCudaBatch
from pvz_rl.envs.cuda_features import REWARD_FIELDS, CudaFeatures
from pvz_rl.learning.counterfactual_settings import OBJECTIVE, PROTOCOL, counterfactual_settings
from pvz_rl.learning.objective import (
    COMPONENTS,
    endgame_terminal_rewards,
    residual_sun_adjustment,
    training_rewards,
)


class BranchState(IntEnum):
    queued = 0
    running = 1
    waiting_resources = 2
    waiting_commands = 3
    completed = 4
    discarded = 5


PHYSICAL = ("header", "plants", "zombies", "projectiles", "mowers", "cooldowns", "_schedules")
ACCOUNTING = ("assets", "home_ledger", "totals")


def select_alternatives(values, legal, original, identity, seed):
    """Keyed RNG is independent of lane admission, spilling and main RNG draws."""
    candidates = np.flatnonzero(legal)
    candidates = candidates[candidates != original]
    if not len(candidates):
        return []
    best = int(candidates[np.argmax(np.asarray(values)[candidates])])
    remaining = candidates[candidates != best]
    if not len(remaining):
        return [best]
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), *map(int, identity)]))
    return [best, int(rng.choice(remaining))]


def wait_discount(stats):
    """Sufficient statistics for the existing quadratic wait-run shaping."""
    s = np.array(stats, dtype=np.float64, copy=True)
    if s[1]:
        length, discount = s[1:3]
        s[3] = min(s[3], length) if s[3] else length
        s[4] = max(s[4], length)
        s[5:8] += discount * np.array([1, length, length * length])
    if s[4] == s[3]:
        return float(s[5])
    minimum, span = s[3], s[4] - s[3]
    return float(0.2 * s[5] + 0.8 * (s[7] - 2 * minimum * s[6] + minimum**2 * s[5]) / span**2)


def continuation_value(stats, header, prefix, root_tick, cfg):
    gamma = cfg["training"]["gamma"]
    won = int(header[1]) == 1
    shaping = residual_sun_adjustment(header[2], won, cfg)
    shaping *= gamma ** stats[8] if won else wait_discount(stats)
    # Prefix economic costs occur in every factual alternative and cancel once.
    return float((stats[0] - prefix + shaping) / gamma**root_tick)


def completed_preferences(tiles, outcomes, prefix, root_tick, cfg):
    """Duration ranks are computed once, only across completed same-board paths."""
    _, adjustments = endgame_terminal_rewards(
        [int(h[0]) for s, h in outcomes],
        [int(h[1]) == 1 for s, h in outcomes],
        weight=cfg["reward"]["win_time_weight"],
    )
    gamma = cfg["training"]["gamma"]
    values = [
        continuation_value(s, h, prefix, root_tick, cfg) + gamma ** (s[8] - root_tick) * a
        for (s, h), a in zip(outcomes, adjustments, strict=True)
    ]
    pairs = []
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            if abs(values[i] - values[j]) > 1e-6:
                pairs.append(
                    (tiles[i], tiles[j]) if values[i] > values[j] else (tiles[j], tiles[i])
                )
    return pairs, values, adjustments.tolist()


class CounterfactualRunner:
    def __init__(self, env, buffer, seed, phase, identity, state=None, archive=None):
        self.env, self.buffer, self.seed = env, buffer, int(seed or 0)
        self.cfg = buffer.cfg
        self.settings = counterfactual_settings(self.cfg, phase)
        self.phase = phase
        self.identity = tuple(identity)
        self.cp = cp = env.cp
        self.path = buffer.path / "factual"
        self.path.mkdir(exist_ok=True)
        self.cache, self.cache_bytes = {}, 0
        self.roots, self.jobs, self.records, self.feasibility = [], [], [], {}
        self.queue = deque()
        self.logs = [[] for _ in range(env.num_envs)]
        self.main_stats = cp.zeros((env.num_envs, 12), cp.float64)
        self.main_header = env.batch.header.copy()
        self.main_ended = cp.zeros(env.num_envs, cp.bool_)
        self.legal_masks = cp.empty_like(env.batch.masks)
        self.component_indices = cp.asarray([REWARD_FIELDS.index(k) for k in COMPONENTS], cp.int64)
        self.component_weights = cp.asarray(training_rewards(np.eye(len(COMPONENTS)), self.cfg))
        self.cutoff = math.ceil(
            self.cfg["environment"]["cutoff_seconds"] * env.batch.rules.game["tick_rate"]
        )
        self.capacity = self.cutoff // self.cfg["environment"]["decision_ticks"] + 2
        if env._per_tick:
            # Instant accepted actions consume no simulator ticks. Bound their
            # extra log entries by real card cooldowns, plus initial Dig assets.
            self.capacity += 45 + 2 * sum(
                1 + self.cutoff // (rule["recharge_ticks"] + 1)
                for rule in env.batch.rules.plants.values()
            )
        self.commands = cp.zeros((env.num_envs, self.capacity, 4), cp.int64)
        self.lengths = cp.zeros(env.num_envs, cp.int64)
        self.published = cp.zeros(env.num_envs, cp.int64)
        self.pool = None
        self.metrics = dict(
            roots=0,
            branches=0,
            shortages=0,
            completed=0,
            discarded=0,
            simulated_ticks=0,
            resource_waits=0,
            queue_lag_ticks=0,
            spill_bytes=0,
            drain_seconds=0.0,
            pending=0,
            resident=0,
            branch_wins=0,
            branch_failures=0,
            branch_timeouts=0,
        )
        self._pending_roots = []
        if state is not None:
            self._restore(state, archive)

    @property
    def estimated_bytes(self):
        # Charge cached roots and conservative Python metadata to trajectory RAM.
        return self.cache_bytes + sum(len(log) * 256 for log in self.logs) + len(self.jobs) * 4096

    @property
    def settled(self):
        return all(j["state"] in ("completed", "discarded") for j in self.jobs)

    def _ensure_pool(self):
        if self.pool is not None:
            return
        env, cp = self.env, self.cp
        from pvz_rl.learning.performance import phase_memory

        memory = phase_memory(env.header_tensor.device)
        per_lane = sum(getattr(env.batch, k).nbytes // env.num_envs for k in PHYSICAL)
        per_lane += env.features.totals.shape[1] * 8 + env.batch.zcap * 16 + 8192
        headroom = self.cfg["training"]["performance"].get("collection_vram_headroom_mib", 0)
        available = max(0, memory["free_mib"] - headroom) * 2**20
        workspace = sum(
            array.nbytes
            for array in (
                self.commands,
                self.lengths,
                self.published,
                self.main_ended,
                self.main_stats,
                self.main_header,
                self.legal_masks,
                self.component_indices,
                self.component_weights,
            )
        )
        budget = min(available, max(0, self.settings["scratch_vram_mib"] * 2**20 - workspace))
        n = min(self.settings["resident_envs"] - env.num_envs, int(budget // per_lane))
        if n < 1:
            raise MemoryError(
                "Factual replay cannot allocate one scratch lane: increase resident_envs "
                "above main n_envs, increase scratch_vram_mib or free VRAM"
            )
        while n:
            try:
                b = AccountingCudaBatch(
                    n,
                    zombie_capacity=env.batch.zcap,
                    projectile_capacity=env.batch.qcap,
                    max_step_ticks=env.batch.max_step_ticks,
                    rules=env.batch.rules,
                    device=env.batch.device,
                )
                b.set_sky_sun_rate(env.batch.sky_sun_rate)
                f = CudaFeatures(b, self.cfg, "masked", accounting_only=True)
                break
            except cp.cuda.memory.OutOfMemoryError:
                b = f = None
                cp.get_default_memory_pool().free_all_blocks()
                n //= 2
        if not n:
            raise MemoryError("Factual replay scratch allocation failed down to one lane")
        source = files("pvz_rl.envs").joinpath("counterfactual_replay.cu").read_text("utf-8")
        # Use the reward module's identical constants/functions, without policy kernels.
        # CudaFeatures keeps its compiled source to avoid regenerating rule definitions.
        module = cp.RawModule(
            code=f.source + "\n" + source, options=("--std=c++11", "--fmad=false")
        )
        self.select = module.get_function("replay_select")
        self.accumulate = module.get_function("replay_accumulate")
        self.clone = module.get_function("clone_lanes")
        self.pool = dict(
            batch=b,
            features=f,
            module=module,
            slots=[None] * n,
            meta=cp.zeros((n, 5), cp.int64),
            stats=cp.zeros((n, 12), cp.float64),
            actions=cp.zeros(n, cp.int64),
            eligible=cp.zeros(n, cp.bool_),
        )
        self.pool["meta"][:, 0] = -1
        self.metrics.update(
            resident_limit=env.num_envs + n,
            scratch_lanes=n,
            scratch_mib=(per_lane * n + workspace) / 2**20,
            command_storage_mib=workspace / 2**20,
        )

    def _save_state(self, key, batch, features, slot, stats):
        """Store compact occupied prefixes; both alternatives share this root."""
        h = batch.header[slot].get()
        state = {
            k: getattr(batch, k)[slot].get()
            for k in PHYSICAL
            if k not in ("zombies", "projectiles", "_schedules")
        }
        state["zombies"] = batch.zombies[slot, : int(h[10])].get()
        state["projectiles"] = batch.projectiles[slot, : int(h[11])].get()
        state["_schedules"] = batch._schedules[slot, : int(h[8])].get()
        state.update({k: getattr(features, k)[slot].get() for k in ACCOUNTING})
        state["stats"] = np.asarray(stats).copy()
        self._store(key, state)

    def _save_roots(self):
        """One transfer per occupied field slab, rather than per root or tile."""
        if not self._pending_roots:
            return
        slots, roots = zip(*self._pending_roots, strict=True)
        ix = self.cp.asarray(slots)
        b, f = self.env.batch, self.env.features
        headers = b.header[ix].get()
        stats = self.main_stats[ix].get()
        data = {"header": headers, "stats": stats}
        counts = {"plants": 9, "zombies": 10, "projectiles": 11, "_schedules": 8}
        for name in PHYSICAL[1:]:
            array = getattr(b, name)
            if name in counts:
                array = array[:, : int(headers[:, counts[name]].max(initial=0))]
            data[name] = array[ix].get()
        for name in ACCOUNTING:
            data[name] = getattr(f, name)[ix].get()
        for j, index in enumerate(roots):
            self.roots[index].update(tick=int(headers[j, 0]), prefix=float(stats[j, 0]))
            state = {
                name: (
                    value[j, : int(headers[j, counts[name]])].copy()
                    if name in counts
                    else np.array(value[j], copy=True)
                )
                for name, value in data.items()
            }
            self._store(f"root-{index}", state)

    def _store(self, key, state):
        amount = sum(a.nbytes for a in state.values())
        limit = min(64 * 2**20, self.buffer.ram_bytes // 16)
        old = self.cache.pop(key, None)
        if old is not None:
            self.cache_bytes -= sum(a.nbytes for a in old.values())
        if self.cache_bytes + amount <= limit:
            self.cache[key] = state
            self.cache_bytes += amount
        else:
            np.savez(self.path / f"{key}.npz", **state)
            self.metrics["spill_bytes"] += amount

    def _load(self, key):
        if key in self.cache:
            return self.cache[key]
        with np.load(self.path / f"{key}.npz", allow_pickle=False) as data:
            return {k: data[k] for k in data.files}

    def _upload(self, key, slot):
        p, cp = self.pool, self.cp
        data = self._load(key)
        for name in PHYSICAL:
            target = getattr(p["batch"], name)
            target[slot] = 0
            value = data[name]
            if value.ndim > 0 and value.shape != target.shape[1:]:
                target[slot, : len(value)] = cp.asarray(value)
            else:
                target[slot] = cp.asarray(value)
        for name in ACCOUNTING:
            getattr(p["features"], name)[slot] = cp.asarray(data[name])
        p["stats"][slot] = cp.asarray(data["stats"])

    def before(self, actions, tile_values, masks, active, episodes, decisions):
        """Capture before the main mutation; accepted legality is checked again later."""
        self._ensure_pool()
        cp = self.cp
        action = cp.from_dlpack(actions.detach().contiguous())
        live = cp.from_dlpack(active)
        # Policy masks intentionally expose all 45 cells. This separate engine
        # legality table is used only by the training oracle, never inference.
        b = self.env.batch
        b._mask(
            ((b.n * A.size + 255) // 256,),
            (256,),
            (b.header, b.plants, b.cooldowns, self.legal_masks, b.n),
        )
        legal = self.legal_masks
        slots = cp.flatnonzero(live & (action > 0) & (action < A.dig_start))
        # Compact action evidence is the only selection transfer. No private fields
        # participate in selecting alternatives.
        evidence = cp.stack((slots, action[slots]), axis=1).get()
        values = cp.from_dlpack(tile_values.detach().float().contiguous())
        tables = values[slots].get()
        legality = legal[slots].get()
        self._pending_roots = []
        for evidence_index, (slot, chosen) in enumerate(evidence):
            slot, chosen = int(slot), int(chosen)
            branch, tile = (chosen - 1) // 45 + 1, (chosen - 1) % 45
            allowed = legality[evidence_index, 1 + (branch - 1) * 45 : 1 + branch * 45]
            identity = (slot, int(episodes[slot]), int(decisions[slot]))
            alternatives = select_alternatives(
                tables[evidence_index], allowed, tile, identity, self.seed
            )
            if not allowed[tile]:
                # Branch availability is checked by collector routing after execution.
                if allowed.any():
                    self.feasibility[identity] = (
                        int(np.flatnonzero(allowed)[np.argmax(tables[evidence_index][allowed])]),
                        tile,
                    )
                continue
            self.metrics["shortages"] += 2 - len(alternatives)
            if not alternatives:
                continue
            index = len(self.roots)
            root = dict(identity=identity, branch=branch, tile=tile, jobs=[], finished=False)
            self.roots.append(root)
            for alternative in alternatives:
                job_id = len(self.jobs)
                root["jobs"].append(job_id)
                self.jobs.append(
                    dict(
                        root=index,
                        tile=alternative,
                        state="queued",
                        cursor=len(self.logs[slot]) + 1,
                        initial=1 + (branch - 1) * 45 + alternative,
                    )
                )
                self.queue.append(job_id)
            self.metrics["roots"] += 1
            self.metrics["branches"] += len(alternatives)
            self._pending_roots.append((slot, index))
        self._save_roots()
        # Dig feasibility uses exactly the same public legality reference.
        digs = cp.flatnonzero(live & (action >= A.dig_start)).get()
        for slot in digs:
            tile = int(action[slot].get()) - A.dig_start
            allowed = legal[slot, A.dig_start :].get()
            if not allowed[tile] and allowed.any():
                q = values[slot].get()
                reference = int(np.flatnonzero(allowed)[np.argmax(q[allowed])])
                self.feasibility[(int(slot), int(episodes[slot]), int(decisions[slot]))] = (
                    reference,
                    tile,
                )
        # Direct device cloning for resident forks. Queued alternatives retain a
        # single compact physical root in the shared-budget host/disk store.
        self._fill(direct=True)

    def _fill(self, direct=False):
        p, cp = self.pool, self.cp
        sources, destinations = [], []
        metadata = []
        for slot, old in enumerate(p["slots"]):
            if old is not None or not self.queue:
                continue
            job_id = self.queue.popleft()
            job = self.jobs[job_id]
            root = self.roots[job["root"]]
            if "parked" in job:
                self._upload(job.pop("parked"), slot)
            elif direct and any(index == job["root"] for _, index in self._pending_roots):
                source = root["identity"][0]
                sources.append(source)
                destinations.append(slot)
            else:
                self._upload(f"root-{job['root']}", slot)
            metadata.append(
                (slot, [root["identity"][0], job["cursor"], job.get("initial", 0), 0, root["tick"]])
            )
            p["slots"][slot] = job_id
            job["state"] = "running"
        if sources:
            source_indices, target_indices = cp.asarray(sources), cp.asarray(destinations)
            for owner, target, names in (
                (self.env.batch, p["batch"], PHYSICAL),
                (self.env.features, p["features"], ACCOUNTING),
            ):
                for name in names:
                    array = getattr(owner, name)
                    width = array.size // len(array)
                    self.clone(
                        (min(4096, (width * len(sources) + 255) // 256),),
                        (256,),
                        (
                            array,
                            getattr(target, name),
                            source_indices,
                            target_indices,
                            np.int64(width),
                            np.int64(len(sources)),
                        ),
                    )
            p["stats"][target_indices] = self.main_stats[source_indices]
        if metadata:
            slots, values = zip(*metadata, strict=True)
            p["meta"][cp.asarray(slots)] = cp.asarray(values, dtype=cp.int64)

    def observe(self, rows, *, wait_available=None):
        cp = self.cp
        voluntary = (
            np.ones(len(rows), bool) if wait_available is None else np.asarray(wait_available, bool)
        )
        if voluntary.shape != (len(rows),):
            raise ValueError("Main wait eligibility must match the episode slots")
        eligible_rows = (
            rows["active"]
            & rows["accepted"]
            & (rows["action"] == 0)
            & (rows["exploration_mode"] == 0)
            & voluntary
            & (self.phase != "peashooter")
        )
        # Main episode logs are published exactly once, after the action transaction.
        for slot in np.flatnonzero(rows["active"]):
            row, log = rows[slot], self.logs[slot]
            action = int(row["executed_action"])
            tick, end = int(row["tick"]), int(row["tick"] + row["duration"])
            eligible = bool(eligible_rows[slot])
            if (
                not action
                and log
                and log[-1][2] == 0
                and log[-1][1] == tick
                and log[-1][3] == eligible
            ):
                log[-1][1] = end
            else:
                log.append([tick, end, action, int(eligible)])
            if len(log) > self.capacity:
                raise RuntimeError("Main replay log exceeded the absolute decision cutoff")
            self.commands[slot, len(log) - 1] = cp.asarray(log[-1], dtype=cp.int64)
            self.lengths[slot] = len(log)
            self.published[slot] = end
            self.main_ended[slot] = bool(row["done"])
        eligible = cp.asarray(eligible_rows)
        self._accumulate(self.env.batch, self.env.features, eligible, self.main_stats, None)
        self.main_header[:] = self.env.batch.header
        for slot, index in self._pending_roots:
            if not rows["active"][slot] or not rows["accepted"][slot]:
                raise RuntimeError(
                    "Prevalidated factual root did not execute as an accepted main planting"
                )
        self._pending_roots = []

    def _accumulate(self, batch, features, eligible, stats, meta):
        self.accumulate(
            ((batch.n + 63) // 64,),
            (64,),
            (
                batch.header,
                features.before_header,
                features.parts,
                eligible,
                stats,
                meta if meta is not None else np.uint64(0),
                self.component_indices,
                self.component_weights,
                np.int64(len(COMPONENTS)),
                np.int64(self.cutoff),
                np.int64(batch.n),
            ),
        )

    def advance(self, *, draining=False):
        if self.settled:
            if draining:
                self.finish_records()
            return True
        self._ensure_pool()
        self._fill()
        started = perf_counter()
        p, cp = self.pool, self.cp
        b, f = p["batch"], p["features"]
        for _ in range(self.settings["chunk_decisions"]):
            self.select(
                ((b.n + 63) // 64,),
                (64,),
                (
                    b.header,
                    b.plants,
                    b.cooldowns,
                    self.commands,
                    self.lengths,
                    self.published,
                    self.main_ended,
                    p["meta"],
                    p["actions"],
                    p["eligible"],
                    p["stats"],
                    np.int64(self.capacity),
                    np.int64(self.cutoff),
                    np.int64(b.n),
                ),
            )
            f.step(
                p["actions"],
                ticks=self.cfg["environment"]["decision_ticks"],
                per_tick=self.env._per_tick,
                publish=False,
            )
            self._accumulate(b, f, p["eligible"], p["stats"], p["meta"])
        meta = p["meta"].get()
        stats = p["stats"].get()
        headers = b.header.get()
        if not bool(cp.isfinite(f.parts).all()):
            raise RuntimeError("Factual replay produced invalid accounting; recovery required")
        for slot, job_id in enumerate(p["slots"]):
            if job_id is None:
                continue
            job = self.jobs[job_id]
            state = int(meta[slot, 3])
            if state == 6 or not np.isfinite(stats[slot]).all():
                raise RuntimeError(
                    "Unexpected factual replay engine/accounting failure; recovery required"
                )
            job.update(
                state=BranchState(state).name, cursor=int(meta[slot, 1]), initial=int(meta[slot, 2])
            )
            if state in (4, 5):
                if state == 4:
                    job["outcome"] = stats[slot].copy(), headers[slot].copy()
                    self.metrics["branch_wins"] += int(headers[slot, 1] == 1)
                    self.metrics["branch_failures"] += int(headers[slot, 1] != 1)
                    self.metrics["branch_timeouts"] += int(headers[slot, 1] == 0)
                self.metrics["completed" if state == 4 else "discarded"] += 1
                self.metrics["simulated_ticks"] += max(
                    0, int(headers[slot, 0]) - self.roots[job["root"]]["tick"]
                )
                self.metrics["resource_waits"] += int(stats[slot, 10])
                self.metrics["queue_lag_ticks"] = max(
                    self.metrics["queue_lag_ticks"], int(stats[slot, 11])
                )
                p["slots"][slot] = None
                p["meta"][slot, 0] = -1
            elif state == 3 and self.queue:
                # Evict a parked follower so pending roots cannot block main progress.
                key = f"job-{job_id}"
                self._save_state(key, b, f, slot, stats[slot])
                job["parked"] = key
                self.queue.append(job_id)
                p["slots"][slot] = None
                p["meta"][slot, 0] = -1
        self.metrics["pending"] = sum(
            j["state"] not in ("completed", "discarded") for j in self.jobs
        )
        self.metrics["resident"] = sum(j is not None for j in p["slots"])
        if draining:
            self.metrics["drain_seconds"] += perf_counter() - started
        if self.settled and draining:
            self.finish_records()
        return self.settled

    def finish_records(self):
        if not self.settled:
            raise RuntimeError("Cannot publish unfinished factual comparisons")
        main_stats, main_header = self.main_stats.get(), self.main_header.get()
        for root in self.roots:
            if root["finished"]:
                continue
            slot = root["identity"][0]
            outcomes = [(main_stats[slot], main_header[slot])]
            tiles = [root["tile"]]
            for job_id in root["jobs"]:
                job = self.jobs[job_id]
                if job["state"] == "completed":
                    tiles.append(job["tile"])
                    outcomes.append(job["outcome"])
            pairs, values, adjustments = completed_preferences(
                tiles, outcomes, root["prefix"], root["tick"], self.cfg
            )
            if pairs:
                self.records.append(
                    dict(
                        identity=root["identity"],
                        branch=root["branch"],
                        pairs=pairs,
                        values=values,
                        tiles=tiles,
                        duration_adjustments=adjustments,
                    )
                )
            root["finished"] = True

    def snapshot(self):
        if self._pending_roots:
            raise RuntimeError("Cannot checkpoint a partially committed main action")
        if self.pool is not None:
            p = self.pool
            meta, stats = p["meta"].get(), p["stats"].get()
            for slot, job_id in enumerate(p["slots"]):
                if job_id is not None:
                    key = f"job-{job_id}"
                    self._save_state(key, p["batch"], p["features"], slot, stats[slot])
                    self.jobs[job_id].update(
                        parked=key, cursor=int(meta[slot, 1]), initial=int(meta[slot, 2])
                    )
        return dict(
            protocol=PROTOCOL,
            objective=OBJECTIVE,
            settings=self.settings,
            identity=self.identity,
            seed=self.seed,
            roots=copy.deepcopy(self.roots),
            jobs=copy.deepcopy(self.jobs),
            logs=copy.deepcopy(self.logs),
            main_stats=self.main_stats.get(),
            main_header=self.main_header.get(),
            main_ended=self.main_ended.get(),
            records=copy.deepcopy(self.records),
            feasibility=copy.deepcopy(self.feasibility),
            metrics=dict(self.metrics),
        )

    def write_archive(self, archive):
        for key, state in self.cache.items():
            stream = io.BytesIO()
            np.savez(stream, **state)
            archive.writestr(f"factual/{key}.npz", stream.getvalue())
        for path in self.path.glob("*.npz"):
            if path.stem not in self.cache:
                archive.write(path, f"factual/{path.name}")

    def _restore(self, state, archive):
        if state["seed"] != self.seed:
            raise ValueError("Factual alternative-selection seed differs from saved recovery")
        if tuple(state["identity"]) != self.identity:
            raise ValueError("Factual cohort policy identity differs from saved optimizer commits")
        self.roots, self.jobs = copy.deepcopy(state["roots"]), copy.deepcopy(state["jobs"])
        self.records, self.feasibility = (
            copy.deepcopy(state["records"]),
            copy.deepcopy(state["feasibility"]),
        )
        self.logs, self.metrics = copy.deepcopy(state["logs"]), dict(state["metrics"])
        self.main_stats[:] = self.cp.asarray(state["main_stats"])
        self.main_header[:] = self.cp.asarray(state["main_header"])
        self.main_ended[:] = self.cp.asarray(state["main_ended"])
        for slot, log in enumerate(self.logs):
            self.lengths[slot] = len(log)
            if log:
                self.commands[slot, : len(log)] = self.cp.asarray(log)
                self.published[slot] = log[-1][1]
        for name in archive.namelist():
            if name.startswith("factual/") and name.endswith(".npz"):
                # Only owned flat filenames; never extract user-controlled paths.
                leaf = name.removeprefix("factual/")
                if "/" in leaf or "\\" in leaf:
                    raise ValueError("Invalid factual archive path")
                with archive.open(name) as source, (self.path / leaf).open("wb") as target:
                    import shutil

                    shutil.copyfileobj(source, target)
        for index, job in enumerate(self.jobs):
            if job["state"] not in ("completed", "discarded"):
                key = job.get("parked", f"root-{job['root']}")
                if not (self.path / f"{key}.npz").is_file():
                    raise ValueError(f"Missing factual recovery state for branch {index}: {key}")
                self.queue.append(index)

    def release(self):
        if self.pool is not None and not self.settled:
            raise RuntimeError("Cannot release outstanding factual lanes")
        self.pool = None

    def close(self):
        self.pool = None
        self.cache.clear()
        self.cache_bytes = 0
        # Only our flat, owned spill files; recovery archives already contain them.
        for path in self.path.glob("*.npz"):
            path.unlink()
        self.path.rmdir()
        try:
            self.path.parent.rmdir()
        except OSError:
            pass  # The trajectory owner still has live blocks here.
