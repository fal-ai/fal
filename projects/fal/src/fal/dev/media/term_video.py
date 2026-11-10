"""Video previews for `fal dev`: a poster frame, a contact sheet, optional looping.

Video frames come from ffmpeg when it is on PATH. Animated GIF and WebP go
through Pillow and need no ffmpeg. Without ffmpeg, a video prints its path and
a hint.
"""

from __future__ import annotations

import io
import json
import os
import random
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import IO

from PIL import Image

from fal.console.icons import get_inline_separator_icon
from fal.dev.media import term_media
from fal.dev.media.common import (
    ANIMATED_SUFFIXES,
    IMAGE_FORMATS,
    OFF,
    VIDEO_SUFFIXES,
    clean,
)
from fal.dev.media.term_media import ESC, Renderer, Terminal, Tile

SHEET_FRAMES = 6
POSTER_MAX_WIDTH = 1280
SHEET_MAX_WIDTH = 640
PLAY_FPS = 10
PLAY_MAX_FRAMES = 60
PLAY_MAX_WIDTH = 480
# Animated images are decoded frame by frame in memory: a tighter budget.
ANIMATED_MAX_PIXELS = 16_000_000
FFMPEG_HINT = "install ffmpeg to preview video (brew install ffmpeg)"


@dataclass
class VideoInfo:
    width: int
    height: int
    duration: float
    fps: float


def has_ffmpeg() -> bool:
    return bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


def _input(path: str) -> list[str]:
    # Local files only: no playlists or other protocols, and an absolute path
    # cannot be read as an option or a protocol prefix.
    return ["-protocol_whitelist", "file", "-i", os.path.abspath(path)]


def _run(cmd: list[str], timeout: float) -> bytes:
    # stdin closed: ffmpeg otherwise reads keystrokes and can leave the tty raw.
    try:
        return subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=timeout,
            check=True,
        ).stdout
    except subprocess.TimeoutExpired:
        raise ValueError(f"{cmd[0]} timed out after {timeout:g}s") from None
    except subprocess.CalledProcessError as exc:
        lines = (exc.stderr or b"").decode(errors="replace").strip().splitlines()
        raise ValueError(f"{cmd[0]} failed: {lines[-1] if lines else exc.returncode}")


def _float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):  # ffprobe says "N/A" for some containers
        return 0.0


def probe(path: str) -> VideoInfo | None:
    cmd = [
        "ffprobe", "-v", "error", "-protocol_whitelist", "file",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate,duration:format=duration",
        "-of", "json", os.path.abspath(path),
    ]  # fmt: skip
    try:
        data = json.loads(_run(cmd, timeout=30))
        stream = data["streams"][0]
        num, _, den = stream.get("avg_frame_rate", "0/1").partition("/")
        fps = _float(num) / _float(den) if _float(den) else 0.0
        # The video stream's own duration: audio can run longer than the picture.
        duration = _float(stream.get("duration")) or _float(
            data.get("format", {}).get("duration")
        )
        return VideoInfo(int(stream["width"]), int(stream["height"]), duration, fps)
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, IndexError):
        return None


def _fit_filter(side: int) -> str:
    # Both sides capped: a very tall video must not decode into a huge frame.
    return (
        f"scale=w='min({side},iw)':h='min({side},ih)':"
        "force_original_aspect_ratio=decrease"
    )


def frame_at(path: str, seconds: float, max_width: int) -> Image.Image:
    png = _run(
        [
            "ffmpeg", "-nostdin", "-v", "error", "-ss", f"{seconds:.3f}",
            *_input(path), "-frames:v", "1",
            "-vf", _fit_filter(max_width),
            "-f", "image2pipe", "-vcodec", "png", "-",
        ],
        timeout=60,
    )  # fmt: skip
    image = Image.open(io.BytesIO(png), formats=["PNG"])
    image.load()
    return image


def sample_times(duration: float, count: int) -> list[float]:
    """Evenly spaced times, each in the middle of its slice of the video."""
    if duration <= 0 or count < 1:
        return [0.0]
    return [duration * (i + 0.5) / count for i in range(count)]


def _stamp(seconds: float) -> str:
    minutes, secs = divmod(seconds, 60)
    return f"{int(minutes)}:{secs:04.1f}"


def is_video(path: str | Path, content_type: str = "") -> bool:
    return (
        content_type.startswith("video/") or Path(path).suffix.lower() in VIDEO_SUFFIXES
    )


def is_animated(path: str | Path) -> bool:
    if Path(path).suffix.lower() not in ANIMATED_SUFFIXES:
        return False
    try:
        with Image.open(path, formats=IMAGE_FORMATS) as image:
            return getattr(image, "n_frames", 1) > 1
    except Exception:
        return False


@dataclass
class Animation:
    frames: list[Image.Image]
    indexes: list[int]
    total: int
    delay_ms: int  # between the sampled frames, so playback keeps its speed


def _animated_frames(path: str, count: int, max_side: int) -> Animation:
    """Up to `count` evenly spaced frames, each shrunk to fit max_side."""
    with Image.open(path, formats=IMAGE_FORMATS) as image:
        width, height = image.size
        if width * height > ANIMATED_MAX_PIXELS:
            raise ValueError(f"{width}x{height} animation is too large to preview")
        total = getattr(image, "n_frames", 1)
        picked = min(count, total)
        indexes = [total * i // picked for i in range(picked)]
        frames = []
        for index in indexes:
            image.seek(index)
            frame = image.convert("RGBA")
            frame.thumbnail((max_side, max_side))
            frames.append(frame)
        delay = int(image.info.get("duration") or 100) * total // picked
        return Animation(frames, indexes, total, delay)


def _sheet(path: str, count: int) -> tuple[Image.Image, list[Tile], str]:
    """Poster, contact-sheet tiles and caption details for a video or animation."""
    if is_animated(path):
        anim = _animated_frames(path, count, SHEET_MAX_WIDTH)
        tiles = [Tile(f, f"frame {i}") for f, i in zip(anim.frames, anim.indexes)]
        return anim.frames[0], tiles, f"{anim.total} frames"
    info = probe(path)
    if info is None:
        raise ValueError("ffprobe could not read it")
    poster = frame_at(path, 0, POSTER_MAX_WIDTH)
    tiles = []
    for t in sample_times(info.duration, count):
        try:
            tiles.append(Tile(frame_at(path, t, SHEET_MAX_WIDTH), _stamp(t)))
        except (OSError, subprocess.SubprocessError, ValueError):
            continue  # e.g. past the last frame; keep the rest of the sheet
    sep = get_inline_separator_icon()
    return poster, tiles, f"{info.duration:.1f}s {sep} {info.fps:.0f} fps"


def show_video(
    path: str,
    extra: str = "",
    *,
    term: Terminal | None = None,
    play: bool = False,
    frames: int = SHEET_FRAMES,
    out: IO | None = None,
) -> None:
    term = term or term_media.detect()
    renderer = Renderer(term, out)
    write = renderer.out.write
    if term.protocol == OFF:
        write(clean(" | ".join(p for p in (extra, path) if p)) + "\n")
        return
    if not is_animated(path) and not has_ffmpeg():
        write(f"{clean(path)} ({FFMPEG_HINT})\n")
        return
    try:
        poster, tiles, meta = _sheet(path, frames)
    except Exception as exc:
        write(f"{clean(path)} (no preview: {clean(str(exc))})\n")
        return
    caption = term_media.describe(poster, meta, extra, term_media.short_path(path))
    renderer.show([Tile(poster, caption, path)])
    if len(tiles) > 1:
        renderer.show(tiles)
    if play:
        play_video(path, renderer)


def _play_frames(path: str) -> tuple[list[Image.Image], int]:
    """Downscaled frames and the delay between them in ms."""
    if is_animated(path):
        anim = _animated_frames(path, PLAY_MAX_FRAMES, PLAY_MAX_WIDTH)
        return anim.frames, anim.delay_ms
    info = probe(path)
    if info is None:
        return [], 0
    fps = min(PLAY_FPS, info.fps or PLAY_FPS)
    count = max(1, min(PLAY_MAX_FRAMES, int(info.duration * fps)))
    raw = _run(
        [
            "ffmpeg", "-nostdin", "-v", "error", *_input(path),
            "-t", f"{count / fps:.3f}", "-frames:v", str(count),
            "-vf", f"fps={fps},{_fit_filter(PLAY_MAX_WIDTH)}",
            "-f", "image2pipe", "-vcodec", "png", "-",
        ],
        timeout=120,
    )  # fmt: skip
    return _split_pngs(raw), int(1000 / fps)


def _split_pngs(raw: bytes) -> list[Image.Image]:
    magic = b"\x89PNG\r\n\x1a\n"
    frames = []
    for chunk in raw.split(magic)[1:]:
        image = Image.open(io.BytesIO(magic + chunk), formats=["PNG"])
        image.load()
        frames.append(image)
    return frames


def gif_bytes(frames: list[Image.Image], delay_ms: int) -> bytes:
    buf = io.BytesIO()
    rgb = [term_media.flatten(f) for f in frames]  # transparency on dark gray
    rgb[0].save(
        buf,
        format="GIF",
        save_all=True,
        append_images=rgb[1:],
        duration=delay_ms,
        loop=0,
        optimize=True,
    )
    return buf.getvalue()


def kitty_animation(
    frames: list[Image.Image], delay_ms: int, cols: int, rows: int, image_id: int
) -> str:
    """Kitty animation: first frame displayed, the rest added, then looped."""
    seqs = [
        term_media.kitty_sequence(
            term_media.png_bytes(frames[0]), cols, rows, image_id=image_id
        )
    ]
    for frame in frames[1:]:
        head = f"a=f,i={image_id},f=100,z={delay_ms},q=2"
        seqs.append(term_media.kitty_chunks(term_media.png_bytes(frame), head))
    # Gap of the first frame, then loop forever (v=1).
    seqs.append(f"{ESC}_Ga=a,i={image_id},r=1,z={delay_ms},q=2{term_media.ST}")
    seqs.append(f"{ESC}_Ga=a,i={image_id},s=3,v=1,q=2{term_media.ST}")
    return "".join(seqs)


def play_video(path: str, renderer: Renderer) -> None:
    term = renderer.term
    if term.protocol not in ("kitty", "iterm"):
        renderer.out.write("(--play needs kitty, Ghostty, iTerm2 or WezTerm)\n")
        return
    try:
        frames, delay = _play_frames(path)
    except Exception as exc:
        renderer.out.write(f"(no looping preview: {clean(str(exc))})\n")
        return
    if not frames:
        return
    w, h = frames[0].size
    cols, rows, _, _ = term_media.fit(
        w, h, term.cols, term_media.max_rows_for(term), term.cell
    )
    if term.protocol == "kitty":
        boxed = [term_media.letterbox(f, cols, rows, term.cell) for f in frames]
        image_id = random.randint(1, 2**31 - 1)
        seq = kitty_animation(boxed, delay, cols, rows, image_id=image_id)
    else:
        seq = term_media.iterm_sequence(
            gif_bytes(frames, delay), cols, rows, "play.gif"
        )
    # Encoded before anything is written, so a failure leaves the cursor alone.
    left = max(0, (term.cols - cols) // 2)
    caption = "looping preview".center(term.cols).rstrip()
    renderer.out.write(
        "\n" * rows
        + f"{ESC}[{rows}A\r"
        + (f"{ESC}[{left}C" if left else "")
        + f"{ESC}7{renderer.graphic(seq)}{ESC}8\r{ESC}[{rows}B"
        + f"{ESC}[2m{caption}{ESC}[0m\n"
    )
    renderer.out.flush()
