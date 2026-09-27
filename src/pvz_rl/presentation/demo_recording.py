"""Human-game transition archives and the interactive ``record-demo`` flow."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from pvz_game import Dig, Place, Rules, Status, Wait
from pvz_game.ui import App, Screen

from pvz_rl.config import digest, load_demo_config
from pvz_rl.envs.action_timing import ActionPhaseGame
from pvz_rl.envs.actions import ActionCodec
from pvz_rl.envs.actions import ActionSchema as A
from pvz_rl.envs.encoding import ObservationEncoder
from pvz_rl.envs.rewards import reward_parts
from pvz_rl.presentation.recordings import ActionPhaseRecorder
from pvz_rl.provenance import append_jsonl, file_hash, write_json

ARCHIVE_PROTOCOL = "pvz-rl/transition-archive-v1"
HISTORY_PROTOCOL = "pvz-rl/compact-history-v1"


def _observation_vector(encoder: ObservationEncoder, observation) -> list[float]:
    return encoder.encode(observation).astype(np.float32).tolist()


class CompactHistoryBuilder:
    """Build a derived event index without changing the complete sequence."""

    def __init__(self, *, cooldown_start: int = 286):
        self.cooldown_start = cooldown_start

    def build(
        self, records: list[dict], final_observation: list[float] | None = None
    ) -> list[dict]:
        history: list[dict] = []
        previous = None
        previous_dig = None
        for row in records:
            obs = np.asarray(row["observation"], dtype=np.float32)
            action = int(row["action"])
            reasons: list[str] = []
            if action and action < A.dig_start:
                reasons.append("plant_attempt")
            is_dig = action >= A.dig_start
            if is_dig and (previous_dig != action or previous is None):
                reasons.append("dig_attempt")
            if previous is not None:
                diff = np.flatnonzero(obs != previous)
                if np.any((diff < 135)):
                    reasons.append("plant_state")
                if np.any((135 <= diff) & (diff < 270)):
                    reasons.append("zombie_state")
                if np.any((diff == 270) | ((272 <= diff) & (diff < 281))):
                    reasons.append("sun_or_global")
                if np.any((281 <= diff) & (diff < 286)):
                    reasons.append("mower_state")
                if np.any(diff >= self.cooldown_start):
                    reasons.append("cooldown_decrement")
            if reasons:
                history.append({"decision_index": row["decision_index"], "reasons": reasons})
            previous = obs
            previous_dig = action if is_dig else None
        if final_observation is not None and records:
            final = np.asarray(final_observation, dtype=np.float32)
            if not np.array_equal(final, previous):
                diff = np.flatnonzero(final != previous)
                reasons = ["final_observation"]
                if np.any(diff < 135):
                    reasons.append("plant_state")
                if np.any((135 <= diff) & (diff < 270)):
                    reasons.append("zombie_state")
                if np.any((diff == 270) | ((272 <= diff) & (diff < 281))):
                    reasons.append("sun_or_global")
                if np.any((281 <= diff) & (diff < 286)):
                    reasons.append("mower_state")
                if np.any(diff >= self.cooldown_start):
                    reasons.append("cooldown_decrement")
                history.append(
                    {"decision_index": records[-1]["decision_index"] + 1, "reasons": reasons}
                )
        return history


class TransitionArchive:
    """Append-only JSONL decisions with atomic completion metadata.

    Records are written as they arrive, so a killed window leaves a usable
    incomplete archive and never creates a fake terminal return.
    """

    def __init__(self, path: str | Path, *, cfg: dict, episode_id: str = "demo-1"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.path.with_suffix(self.path.suffix + ".manifest.json")
        self.history_path = self.path.with_suffix(self.path.suffix + ".history.json")
        self.cfg = cfg
        self.codec = ActionCodec(cfg)
        self.encoder = ObservationEncoder(cfg, Rules())
        self.episode_id = episode_id
        self.records: list[dict] = []
        self.final_observation: list[float] | None = None
        self.closed = False
        if self.path.exists():
            raise FileExistsError(f"recording output already exists: {self.path}")
        self._stream = self.path.open("x", encoding="utf-8")
        self._manifest(complete=False)

    def _manifest(self, *, complete: bool, outcome: str | None = None, replay: str | None = None):
        payload = {
            "protocol": ARCHIVE_PROTOCOL,
            "history_protocol": HISTORY_PROTOCOL,
            "complete": complete,
            "outcome": outcome,
            "episode_id": self.episode_id,
            "decisions": len(self.records),
            "observation_size": self.encoder.size,
            "config_digest": digest(self.cfg),
            "engine": {
                "commit": self.cfg["engine_commit"],
                "version": self.cfg["engine_version"],
                "package_version": self.cfg["engine_package_version"],
            },
            "replay": replay,
            "storage": "streamed_jsonl",
        }
        write_json(self.manifest_path, payload)

    def append(
        self,
        observation,
        *,
        action: int,
        accepted: bool,
        rejection_reason: str | None,
        reward: dict,
        terminal: str,
        tick: int,
        ticks_advanced: int,
    ) -> dict:
        if self.closed:
            raise RuntimeError("archive is closed")
        if type(action) is not int or action < 0:
            raise ValueError("action must be a nonnegative integer")
        if (
            type(tick) is not int
            or tick < 0
            or type(ticks_advanced) is not int
            or ticks_advanced < 0
        ):
            raise ValueError("tick and ticks_advanced must be nonnegative integers")
        if self.records and tick < self.records[-1]["tick"]:
            raise ValueError("simulation ticks must be monotonic")
        index = len(self.records)
        row = {
            "record_type": "transition",
            "episode_id": self.episode_id,
            "decision_index": index,
            "tick": int(tick),
            "ticks_advanced": int(ticks_advanced),
            "observation": _observation_vector(self.encoder, observation),
            "action": action,
            "accepted": bool(accepted),
            "rejection_reason": rejection_reason,
            "reward_parts": dict(reward),
            "terminal": terminal,
        }
        if len(row["observation"]) != self.encoder.size:
            raise ValueError("encoded observation has the wrong protocol size")
        append_jsonl(self._stream, row)
        self.records.append(row)
        return row

    def finalize(self, observation, *, outcome: str, replay_path: str | Path | None = None):
        if self.closed:
            return
        if self.records and observation.tick < self.records[-1]["tick"]:
            raise ValueError("final observation tick precedes the last decision")
        if outcome not in ("won", "lost", "truncated", "interrupted", "running"):
            raise ValueError("unsupported archive outcome")
        self.final_observation = _observation_vector(self.encoder, observation)
        append_jsonl(
            self._stream,
            {
                "record_type": "final_observation",
                "episode_id": self.episode_id,
                "decision_index": len(self.records),
                "tick": int(observation.tick),
                "observation": self.final_observation,
                "terminal": outcome,
            },
        )
        self._stream.flush()
        self._stream.close()
        history = CompactHistoryBuilder(
            cooldown_start=self.encoder.slices["cooldowns"].start
        ).build(self.records, self.final_observation)
        write_json(self.history_path, {"protocol": HISTORY_PROTOCOL, "events": history})
        replay_hash = file_hash(replay_path) if replay_path and Path(replay_path).exists() else None
        self._manifest(complete=True, outcome=outcome, replay=replay_hash)
        self.closed = True

    def close(self):
        if self.closed:
            return
        self._stream.flush()
        self._stream.close()
        self._manifest(complete=False)
        self.closed = True

    def verify(self) -> dict:
        rows = [
            json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line
        ]
        transitions = [row for row in rows if row.get("record_type") == "transition"]
        finals = [row for row in rows if row.get("record_type") == "final_observation"]
        if any(row.get("decision_index") != i for i, row in enumerate(transitions)):
            raise ValueError("decision indices are not monotonic")
        for row in transitions + finals:
            if len(row["observation"]) != self.encoder.size:
                raise ValueError("archive observation size mismatch")
        if finals and finals[-1]["decision_index"] != len(transitions):
            raise ValueError("final observation index mismatch")
        return {
            "decisions": len(transitions),
            "complete": bool(finals),
            "outcome": finals[-1].get("terminal") if finals else None,
        }


class DemoRecordingApp(App):
    """Native UI with per-decision archive hooks and zero-tick operations."""

    def __init__(self, cfg, *, replay_path: Path, archive_path: Path, seed=1000, speed=1):
        self.research_cfg = cfg
        self.codec = ActionCodec(cfg)
        self._session_started = False
        # ``cfg`` is the research configuration.  App loads its own native UI
        # TOML; passing the research dictionary as a UI config is a startup bug.
        validate_recording_outputs(replay_path, archive_path)
        self.transition_archive = TransitionArchive(archive_path, cfg=cfg)
        self._archive_finalized = False
        self._replay_path = Path(replay_path)
        try:
            super().__init__(level="easy", seed=seed, record_path=self._replay_path, speed=speed)
        except BaseException:
            self.transition_archive.close()
            import pygame

            pygame.quit()
            raise

    def start(self, level):
        if self._session_started:
            raise RuntimeError("a human recording has one fixed easy-stage attempt")
        if level != "easy":
            raise ValueError("human recordings use the easy stage")
        self._session_started = True
        self._autosave()
        self.level = level
        # The native UI owns presentation settings, while the research recorder
        # needs the action-phase adapter for zero-tick plant/dig operations.
        self.game = ActionPhaseGame(self.rules)
        self.game.reset(level, self.seed)
        metadata = {
            "protocol": ARCHIVE_PROTOCOL,
            "config_digest": digest(self.research_cfg),
            "observation_schema": self.research_cfg["encoding"]["version"],
            "engine_commit": self.research_cfg["engine_commit"],
            "seed": self.seed,
        }
        self.recorder = ActionPhaseRecorder(self.game, metadata=metadata)
        self.saved_tick = -1
        self.mode = Screen.PLAYING
        self.accumulator = 0
        self.pending.clear()
        self.effects.clear()
        self.sun_flash_until = 0
        self.selected = None
        self.message = "Plant sunflowers early. Cover every lane."

    def restart(self):
        self.message = "Restart and stage switching are disabled during a recording."

    def activate(self, name):
        if name in ("restart", "menu") or name.startswith("level:"):
            self.message = "Restart and stage switching are disabled during a recording."
            return
        super().activate(name)

    def button(self, name, label, rect, active=False):
        if name in ("restart", "menu") or name.startswith("level:"):
            if name == "menu":
                self.text(
                    "One attempt per recording. Close the window to finish.",
                    rect[0],
                    rect[1] + 10,
                    15,
                )
            return
        super().button(name, label, rect, active)

    def text(self, label, *args, **kwargs):
        """Keep the pinned native help consistent with this session's controls."""
        return super().text(label.replace("   ·   R restart", ""), *args, **kwargs)

    def toggle_pause(self):
        # Pausing must freeze simulation while preserving queued operations.
        if self._dragging:
            return
        if self.mode == Screen.PLAYING:
            self.mode = Screen.PAUSED
        elif self.mode == Screen.PAUSED:
            self.mode = Screen.PLAYING
        self.accumulator = 0

    def advance(self):
        if self.game.observe().status != Status.RUNNING:
            self.mode = Screen.ENDED
            return
        action = self.pending.popleft() if self.pending else Wait()
        before = self.game.observe()
        proposed = self.game.validate_action(action)
        ticks = 0 if isinstance(action, (Place, Dig)) and proposed.accepted else 1
        result = self.recorder.step(action, ticks=ticks)
        parts = reward_parts(
            before,
            result.observation,
            self.research_cfg,
            events=result.events,
            rules=self.game.rules,
            action=action,
            action_result=result.action_result,
        )
        self.transition_archive.append(
            before,
            action=self._action_index(action),
            accepted=result.action_result.accepted,
            rejection_reason=result.action_result.reason,
            reward=parts,
            terminal=result.status.value,
            tick=before.tick,
            ticks_advanced=result.ticks_advanced,
        )
        if result.status != Status.RUNNING:
            self.mode = Screen.ENDED
            self.pending.clear()
            self._autosave()
        self.effects = [effect for effect in self.effects if effect[0] > result.observation.tick]
        for event in result.events:
            if event.kind == "PlantExploded":
                self.effects.append(
                    (
                        result.observation.tick + 12,
                        event.get("row"),
                        event.get("col"),
                        event.get("radius"),
                    )
                )
            elif event.kind == "SunProduced" and event.get("amount"):
                self.sun_flash_until = result.observation.tick + 8

    def _action_index(self, action):
        return self.codec.encode(action)

    def save_recording(self, path: Path | None = None):
        if self.recorder and not self._archive_finalized:
            destination = path or self._replay_path
            self.recorder.update_metadata({"outcome": self.game.observe().status.value})
            self.recorder.save(destination, overwrite=True)
            from pvz_rl.presentation.recordings import verify_replay

            verify_replay(destination)
            if self.game.observe().status != Status.RUNNING:
                self.transition_archive.finalize(
                    self.game.observe(),
                    outcome=self.game.observe().status.value,
                    replay_path=destination,
                )
                self._archive_finalized = True

    def _autosave(self):
        if self.recorder:
            self.save_recording(self._replay_path)

    def run(self):
        try:
            return super().run()
        finally:
            try:
                if not self._archive_finalized:
                    self.transition_archive.close()
            finally:
                import pygame

                pygame.quit()


def record_demo(
    output: str | Path,
    *,
    archive: str | Path | None = None,
    seed: int = 1000,
    speed: float = 1,
    cfg: dict | None = None,
) -> dict:
    cfg = cfg or load_demo_config()
    output = Path(output)
    archive = Path(archive) if archive else output.with_suffix(".jsonl")
    validate_recording_outputs(output, archive)
    app = DemoRecordingApp(cfg, replay_path=output, archive_path=archive, seed=seed, speed=speed)
    app.run()
    return {
        "replay": str(output),
        "archive": str(archive),
        "seed": seed,
        "verified": output.exists(),
    }


def validate_recording_outputs(replay_path: str | Path, archive_path: str | Path):
    """Reject every prior or overlapping artifact before opening the UI."""
    replay = Path(replay_path).resolve()
    archive = Path(archive_path).resolve()
    paths = {
        "replay": replay,
        "archive": archive,
        "manifest": archive.with_suffix(archive.suffix + ".manifest.json"),
        "history": archive.with_suffix(archive.suffix + ".history.json"),
    }
    from itertools import combinations

    for (name, path), (other_name, other) in combinations(paths.items(), 2):
        if path == other or path in other.parents or other in path.parents:
            raise ValueError(f"recording outputs overlap ({name}, {other_name}): {path}, {other}")
    for name, path in paths.items():
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"recording {name} output already exists: {path}")
