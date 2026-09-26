"""Current-game decision evidence, independent of visibility and training inputs."""

import json
import tempfile
import warnings
from pathlib import Path

import numpy as np

from pvz_rl.envs.actions import ActionSchema as A

PAGE_ROWS = 64
RECORD = np.dtype(
    [
        ("sequence", "<u8"),
        ("tick", "<u4"),
        ("action", "<u2"),
        ("greedy_action", "<u2"),
        ("accepted", "?"),
        ("reason", "u1"),
        ("penalty", "<f8"),
        ("coins", "?", 2),
        ("legal", "?", 10),
        ("q", "<f4", 10),
    ]
)


def public_record(row):
    return {name: row[name].tolist() for name in RECORD.names}


class ActionJournal:
    """Bounded resident blocks; disk spill retains every non-wait decision.

    One latest record per environment also retains waiting decisions. Every
    retained row contains the immutable ten-way Q vector and selection metadata.
    Reset only replaces that environment's current-game journal. Checkpoint blocks are
    streamed, so neither long games nor saving require a whole-game RAM copy.
    """

    def __init__(self, count, *, ram_bytes=16 * 1024**2, block_rows=256, reward_settings=None):
        self.workspace = self.root = None
        self.count, self.ram_bytes, self.block_rows = count, int(ram_bytes), block_rows
        self.disabled = False
        self.reward_settings = reward_settings or {}
        self.ram_used = 0
        self.episodes = [-1] * count
        self.sizes = [0] * count
        self.sequences = [0] * count
        self.blocks = [[] for _ in range(count)]
        self.latest = [None] * count
        try:
            self.workspace = tempfile.TemporaryDirectory(prefix="pvz-action-history-")
            self.root = Path(self.workspace.name)
        except OSError as exc:
            self.fail(exc)

    def fail(self, error):
        if not self.disabled:
            warnings.warn(
                f"Action history disabled; training continues: {error}",
                RuntimeWarning,
                stacklevel=2,
            )
        self.disabled = True

    def reset(self, env, episode):
        blocks, self.blocks[env] = self.blocks[env], []
        self.sizes[env] = self.sequences[env] = 0
        self.episodes[env], self.latest[env] = int(episode), None
        for block in blocks:
            if isinstance(block, np.memmap):
                try:
                    path = Path(block.filename)
                    block._mmap.close()
                    path.unlink(missing_ok=True)
                except OSError as exc:
                    self.fail(exc)
            else:
                self.ram_used -= block.nbytes

    def _allocate(self, env):
        amount = self.block_rows * RECORD.itemsize
        if self.ram_used + amount <= self.ram_bytes:
            block = np.zeros(self.block_rows, dtype=RECORD)
            self.ram_used += amount
        else:
            block = np.lib.format.open_memmap(
                self.root / f"{env}-{len(self.blocks[env])}.npy",
                mode="w+",
                dtype=RECORD,
                shape=(self.block_rows,),
            )
        self.blocks[env].append(block)
        return block

    def record_batch_safe(self, rows, scores, results, episodes):
        if self.disabled:
            return
        try:
            self.record_batch(rows, scores, results, episodes)
        except (OSError, MemoryError, ValueError) as exc:
            self.fail(exc)

    def record_batch(self, rows, scores, results, episodes):
        records = np.zeros(self.count, dtype=RECORD)
        for name in ("tick", "action", "greedy_action"):
            records[name] = rows[name]
        records["coins"][:, 0] = rows["species_coin"]
        records["coins"][:, 1] = rows["tile_coin"]
        records["q"] = scores
        records["accepted"], records["reason"] = results[:, 0], results[:, 1]
        records["legal"][:] = rows["active"][:, None]
        rejected = ~records["accepted"]
        records["penalty"][
            rejected & (rows["action"] > 0) & (rows["action"] < A.dig_start)
        ] = -self.reward_settings.get("invalid_plant_penalty", 0)
        from pvz_game.cuda.schema import REASONS

        records["penalty"][
            rejected
            & (rows["action"] >= A.dig_start)
            & (records["reason"] == REASONS.index("empty_tile"))
        ] = -self.reward_settings.get("empty_dig_penalty", 0)
        for env in np.flatnonzero(rows["active"]):
            if self.episodes[env] != episodes[env]:
                self.reset(env, episodes[env])
            self.sequences[env] += 1
            row = records[env]
            row["sequence"] = self.sequences[env]
            self.latest[env] = row.copy()
            if row["action"]:
                index, offset = divmod(self.sizes[env], self.block_rows)
                if index == len(self.blocks[env]):
                    self._allocate(env)
                self.blocks[env][index][offset] = row
                self.sizes[env] += 1

    def page(self, env, episode, start=None, limit=PAGE_ROWS):
        if self.disabled or not 0 <= env < self.count or self.episodes[env] != episode:
            return None
        limit = max(1, min(PAGE_ROWS, int(limit)))
        size = self.sizes[env]
        start = max(0, size - limit) if start is None else max(0, min(int(start), size))
        rows = [
            public_record(self.blocks[env][i // self.block_rows][i % self.block_rows])
            for i in range(start, min(start + limit, size))
        ]
        return dict(
            env=env,
            episode=episode,
            start=start,
            total=size,
            rows=rows,
            latest=None if self.latest[env] is None else public_record(self.latest[env]),
        )

    def write_archive(self, archive):
        if self.disabled:
            return
        for env, blocks in enumerate(self.blocks):
            for index, block in enumerate(blocks):
                length = min(self.block_rows, self.sizes[env] - index * self.block_rows)
                with archive.open(f"journal/{env}-{index}.npy", "w", force_zip64=True) as stream:
                    np.save(stream, block[:length], allow_pickle=False)

        archive.writestr(
            "journal/state.json",
            json.dumps(
                dict(
                    count=self.count,
                    block_rows=self.block_rows,
                    episodes=self.episodes,
                    sizes=self.sizes,
                    sequences=self.sequences,
                    latest=[None if x is None else public_record(x) for x in self.latest],
                )
            ),
        )

    def restore_archive_safe(self, archive):
        if self.disabled:
            return
        try:
            self.restore_archive(archive)
        except Exception as exc:
            # Optional diagnostics cannot prevent restoration of valid training state.
            self.fail(exc)

    def restore_archive(self, archive):
        if "journal/state.json" not in archive.namelist():
            return  # Existing compatible files have no diagnostic history.
        meta = json.loads(archive.read("journal/state.json"))
        if meta["count"] != self.count:
            raise ValueError("Journal environment count differs from checkpoint")
        self.block_rows = meta["block_rows"]
        for env in range(self.count):
            self.reset(env, meta["episodes"][env])
            self.sizes[env], self.sequences[env] = meta["sizes"][env], meta["sequences"][env]
            latest = meta["latest"][env]
            if latest is not None:
                row = np.zeros((), dtype=RECORD)
                for key, value in latest.items():
                    row[key] = value
                self.latest[env] = row
            for index in range((self.sizes[env] + self.block_rows - 1) // self.block_rows):
                with archive.open(f"journal/{env}-{index}.npy") as stream:
                    block = np.load(stream, allow_pickle=False)
                self._allocate(env)[: len(block)] = block

    def close(self):
        for env in range(self.count):
            self.reset(env, -1)
        if self.workspace is not None:
            try:
                self.workspace.cleanup()
            except OSError as exc:
                self.fail(exc)
