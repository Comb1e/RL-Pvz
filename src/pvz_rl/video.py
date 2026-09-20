"""Stream hash-checked engine playback to portable MP4, without a display window."""

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from time import perf_counter

from pvz_game.replay import Playback

from .config import output_settings
from .progress import Phase, ProgressReporter
from .provenance import file_hash, write_json
from .rendering import render_observation


def _process_options():
    return {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}


def ffmpeg_info(cfg):
    configured = output_settings(cfg)["visualization"]["ffmpeg"]
    executable = shutil.which(configured or "ffmpeg")
    if not executable:
        raise RuntimeError("FFmpeg unavailable: install it on PATH or set visualization.ffmpeg")
    version = subprocess.run(
        [executable, "-hide_banner", "-version"],
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
        **_process_options(),
    ).stdout.splitlines()[0]
    encoders = subprocess.run(
        [executable, "-hide_banner", "-encoders"],
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
        **_process_options(),
    ).stdout
    if not any("libx264" in line.split() for line in encoders.splitlines()):
        raise RuntimeError("FFmpeg must include the libx264 H.264 encoder")
    return {"path": executable, "version": version, "encoder": "libx264"}


def export_replay(source, destination, cfg, *, context=None, progress=None):
    source, destination = Path(source), Path(destination)
    if destination.suffix.lower() != ".mp4":
        raise ValueError("Video destination must have an .mp4 extension")
    settings = output_settings(cfg)
    encoder = ffmpeg_info(cfg)
    playback = Playback(source)
    sidecar = source.with_suffix(".metadata.json")
    details = json.loads(sidecar.read_text("utf-8")) if sidecar.exists() else {}
    details.update(context or {})
    # The engine remains running at a wrapper cutoff. Never infer a loss from it.
    outcome = details.get("outcome", details.get("status", playback.data["final_status"]))
    fps = playback.game.observe().tick_rate
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.stem + ".tmp.mp4")
    owns_progress = progress is None
    progress = progress or ProgressReporter(
        destination.with_suffix(".log"), settings["logging"]["progress_seconds"]
    )
    progress.phase(Phase.EXPORTING)
    progress.emit(f"Video export: {destination.name}, {fps} frames/s", force=True)
    started = perf_counter()
    process = None
    frames = 0
    last_action = "Waiting for the next decision"
    try:
        with tempfile.TemporaryFile() as errors:
            command = [
                encoder["path"],
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pixel_format",
                "rgb24",
                "-video_size",
                "1000x600",
                "-framerate",
                str(fps),
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                str(settings["visualization"]["crf"]),
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(temporary),
            ]
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=errors,
                **_process_options(),
            )

            def write_frame(final=False):
                nonlocal frames
                observation = playback.game.observe()
                frame = render_observation(
                    observation,
                    context={
                        **details,
                        "action": last_action,
                        "outcome": outcome if final else observation.status.value,
                    },
                )
                process.stdin.write(frame.tobytes())
                frames += 1

            try:
                write_frame(final=playback.done)
                while not playback.done:
                    entry = playback.data["entries"][playback.index]
                    starts_action = playback.offset == 0
                    result = playback.step()  # Verifies intermediate and final hashes.
                    action = entry["action"]
                    if starts_action and action["kind"] != "wait":
                        name = action.get("plant_type", action["kind"]).replace("_", " ")
                        accepted = "" if result.action_result.accepted else " (rejected)"
                        last_action = (
                            f"{entry['tick'] / fps:.1f}s: {action['kind']} {name}, "
                            f"row {action['row'] + 1}, column {action['col'] + 1}{accepted}"
                        )
                    write_frame(final=playback.done)
                    progress.emit(
                        f"Encoding {destination.name}: game time "
                        f"{playback.game.observe().elapsed_seconds:.1f}s, {frames} frames"
                    )
                for _ in range(round(settings["visualization"]["final_hold_seconds"] * fps)):
                    write_frame(final=True)
                process.stdin.close()
                code = process.wait(timeout=60)
            except BrokenPipeError as exc:
                process.wait(timeout=15)
                errors.seek(0)
                raise RuntimeError(
                    "FFmpeg pipe failed: " + errors.read()[-4096:].decode(errors="replace")
                ) from exc
            if code:
                errors.seek(0)
                raise RuntimeError(
                    "FFmpeg failed: " + errors.read()[-4096:].decode(errors="replace")
                )
        temporary.replace(destination)
        record = {
            **details,
            "outcome": outcome,
            "frames": frames,
            "fps": fps,
            "duration_seconds": frames / fps,
            "export_seconds": perf_counter() - started,
            "replay_hash": file_hash(source),
            "final_state_hash": playback.game.state_hash(),
            "video_hash": file_hash(destination),
            "encoder": encoder,
            "settings": {
                key: settings["visualization"][key] for key in ("crf", "final_hold_seconds")
            },
            "verified": True,
        }
        write_json(destination.with_suffix(".video.json"), record)
        progress.emit(f"Video ready: {destination} ({frames / fps:.1f}s)", force=True)
        return record
    finally:
        if process:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=15)
            if process.stdin and not process.stdin.closed:
                try:
                    process.stdin.close()
                except BrokenPipeError:
                    pass
        temporary.unlink(missing_ok=True)
        if owns_progress:
            progress.close()
