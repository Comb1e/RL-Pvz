"""Read native replay annotations with a fallback for research 0.2 sidecars."""

import json
from dataclasses import replace
from pathlib import Path

from pvz_game import Dig, Place, Rules
from pvz_game.config import integer
from pvz_game.replay import Playback, RecordedOperation, Recorder, decode_action, read_recording

from pvz_rl.envs.action_timing import ActionPhaseGame

ACTION_PHASE_REPLAY_VERSION = "pvz-rl/actions-v1"


class ActionPhaseRecorder(Recorder):
    """Compact recording with explicitly versioned zero-time operations."""

    def step(self, action, *, ticks=1):
        tick = self.game.observe().tick
        result = self.game.step(action, ticks=ticks)
        self._append(tick, action, result.ticks_advanced)
        return result

    def to_dict(self):
        return {**super().to_dict(), "replay_version": ACTION_PHASE_REPLAY_VERSION}


class ActionPhasePlayback(Playback):
    """Reuse native seeking and hash checks, applying operations between ticks.

    A displayed tick includes all instantaneous operations recorded at that tick.
    This also defines seeking to tick zero and recordings ending mid-action-phase.
    Playback.step still advances exactly one simulation tick for native UI/video.
    """

    def __init__(self, source):
        data = read_recording(source)
        if data.get("replay_version") != ACTION_PHASE_REPLAY_VERSION:
            raise ValueError("Incompatible research action-phase replay")
        game = ActionPhaseGame(Rules(data["initial"]["rules"]))
        game.restore(data["initial"])
        end = game.observe().tick
        for entry in data["entries"]:
            integer(entry["tick"], "replay tick")
            integer(entry["ticks"], "replay ticks")
            if entry["tick"] != end:
                raise ValueError("replay tick mismatch")
            action = decode_action(entry["action"])
            game.validate_action(action)
            if entry["ticks"] == 0 and not isinstance(action, (Place, Dig)):
                raise ValueError("Zero-tick replay entries must plant or dig")
            end += entry["ticks"]
        # Native setup validates the positive-time timeline, metadata, and cache.
        # Instant operations are validated above and executed by our pinned adapter.
        skeleton = {
            **data,
            "replay_version": 1,
            "entries": [e for e in data["entries"] if e["ticks"]],
        }
        if not skeleton["entries"]:
            skeleton.update(final_hash=game.state_hash(), final_status=game.observe().status.value)
        super().__init__(skeleton)
        self.game, self.data, self.done = game, data, False
        self._drain_instant()
        self._initial_cursor = self._checkpoint()

    def _drain_instant(self):
        events = []
        while self.index < len(self.data["entries"]):
            entry = self.data["entries"][self.index]
            if entry["ticks"]:
                break
            if entry["tick"] != self.current_tick:
                raise ValueError("replay tick mismatch")
            action = decode_action(entry["action"])
            result = self.game.step(action, ticks=0)
            self.last_operation = RecordedOperation(entry["tick"], action, result.action_result)
            events.extend(result.events)
            if "state_hash" in entry and entry["state_hash"] != self.game.state_hash():
                raise ValueError(f"replay state mismatch at entry {self.index}")
            self.index += 1
        if self.index == len(self.data["entries"]):
            self._finish()
        return events

    def step(self):
        result = super().step()
        events = self._drain_instant()
        if self.current_tick in self._cache:
            self._cache[self.current_tick] = self._checkpoint()
        return replace(result, observation=self.game.observe(), events=(*result.events, *events))


def open_playback(source, *, fallback=None):
    source = source if isinstance(source, dict) else Path(source)
    data = read_recording(source)
    legacy = dict(fallback or {})
    sidecar = None if isinstance(source, dict) else source.with_suffix(".metadata.json")
    if sidecar is not None and sidecar.exists():
        legacy.update(json.loads(sidecar.read_text("utf-8")))
    annotations = {}
    for key in ("policy_id", "checkpoint_sha256", "outcome", "termination_reason"):
        value = legacy.get(key)
        if value is not None:
            annotations[key] = value
    for old, new in (
        ("policy", "policy_id"),
        ("checkpoint_hash", "checkpoint_sha256"),
        ("status", "outcome"),
    ):
        if legacy.get(old) is not None:
            annotations.setdefault(new, legacy[old])
    annotations["experiment"] = {
        key: legacy[key]
        for key in ("level", "family", "scenario_seed", "learner_seed")
        if key in legacy
    }
    # Embedded metadata is authoritative. Legacy annotations only fill absent fields.
    annotations.update(data.get("metadata", {}))
    if annotations:
        data["metadata"] = annotations
    playback_type = (
        ActionPhasePlayback
        if data.get("replay_version") == ACTION_PHASE_REPLAY_VERSION
        else Playback
    )
    return playback_type(data)


def verify_replay(source):
    return open_playback(source).verify()


def watch_recording(source, *, speed=1):
    """Supply our playback to the unchanged native seekable UI."""
    from pvz_game.ui import App, Screen

    playback = open_playback(source)
    if not isinstance(playback, ActionPhasePlayback):
        App(replay_path=playback.data, speed=speed).run()
        return
    app = App(speed=speed)
    app.playback = playback
    app.replay_path = playback.data
    app.game = playback.game
    app.rules = playback.game.rules
    app.mode = Screen.ENDED if playback.done else Screen.PLAYING
    app.message = "REPLAY / per-tick actions / controls are read-only"
    app.run()
