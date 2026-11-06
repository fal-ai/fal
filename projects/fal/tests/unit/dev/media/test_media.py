import base64
import io
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from fal.dev import hot
from fal.dev.media import common, preview, term_media, term_video
from fal.dev.media.term_media import (
    OFF,
    Renderer,
    Terminal,
    Tile,
    blocks_lines,
    detect,
    fit,
    grid_shape,
    iterm_sequence,
    kitty_sequence,
    sixel_sequence,
    tmux_wrap,
)

ESC = "\x1b"


class FakeTTY(io.StringIO):
    def isatty(self):
        return True


@pytest.fixture(autouse=True)
def _media_dir(tmp_path, monkeypatch):
    # Never write into the real ~/.fal/dev/media, and never download for real.
    monkeypatch.setattr(preview, "MEDIA_DIR", tmp_path / "media")

    def no_network(req, timeout=None):
        raise AssertionError(f"network access in a unit test: {req.full_url}")

    monkeypatch.setattr(preview._OPENER, "open", no_network)


def _detect(env, *, tty=True, passthrough=None):
    return detect(env, FakeTTY(), isatty=tty, passthrough=passthrough)


# ------------------------------------------------------------ detection


@pytest.mark.parametrize(
    "env, protocol",
    [
        ({"KITTY_WINDOW_ID": "1", "TERM": "xterm-kitty"}, "kitty"),
        ({"TERM": "xterm-kitty"}, "kitty"),
        ({"TERM_PROGRAM": "ghostty", "TERM": "xterm-ghostty"}, "kitty"),
        ({"TERM_PROGRAM": "WezTerm"}, "iterm"),
        ({"TERM_PROGRAM": "iTerm.app"}, "iterm"),
        ({"LC_TERMINAL": "iTerm2", "TERM_PROGRAM": "vscode"}, "iterm"),
        ({"TERM": "foot"}, "sixel"),
        ({"TERM": "mlterm"}, "sixel"),
        ({"WT_SESSION": "abc"}, "sixel"),
        ({"TERM_PROGRAM": "Apple_Terminal", "TERM": "xterm-256color"}, "blocks"),
        ({"TERM": "xterm-256color"}, "blocks"),
        ({}, "blocks"),
    ],
)
def test_detect_protocol(env, protocol):
    assert _detect(env).protocol == protocol


def test_detect_not_a_tty_prints_paths_only():
    assert _detect({"TERM_PROGRAM": "iTerm.app"}, tty=False).protocol == OFF


def test_detect_env_override():
    env = {"TERM_PROGRAM": "iTerm.app", "FAL_DEV_PREVIEW": "kitty"}
    assert _detect(env).protocol == "kitty"
    # Forcing works when piped too, so captures can record the bytes.
    assert _detect(env, tty=False).protocol == "kitty"
    env["FAL_DEV_PREVIEW"] = "off"
    assert _detect(env).protocol == OFF


def test_detect_tmux():
    kitty = _detect({"TMUX": "/tmp/x", "TERM": "tmux-256color", "KITTY_WINDOW_ID": "3"})
    assert (kitty.protocol, kitty.tmux) == ("blocks", True)
    assert "tmux" in kitty.note

    iterm = {"TMUX": "/tmp/x", "TERM": "screen-256color", "LC_TERMINAL": "iTerm2"}
    assert _detect(iterm, passthrough=True).protocol == "iterm"
    blocked = _detect(iterm, passthrough=False)
    assert blocked.protocol == "blocks"
    assert "allow-passthrough" in blocked.note


def test_detect_truecolor():
    assert _detect({"COLORTERM": "truecolor"}).truecolor
    assert not _detect({"TERM": "xterm-256color"}).truecolor
    assert _detect({"TERM_PROGRAM": "iTerm.app"}).truecolor


# --------------------------------------------------------------- sizing


def test_fit_never_upscales_tiny_images():
    assert fit(48, 48, 200, 60, (10, 20)) == (5, 3, 48, 48)


def test_fit_landscape_is_width_bound():
    cols, rows, px_w, px_h = fit(1344, 768, 80, 40, (10, 20))
    assert (cols, px_w) == (80, 800)
    assert px_h == 457
    assert rows == 23


def test_fit_portrait_is_height_bound():
    cols, rows, px_w, px_h = fit(768, 1344, 80, 20, (10, 20))
    assert (rows, px_h) == (20, 400)
    assert (px_w, cols) == (229, 23)


def test_fit_very_wide_never_overflows():
    for width in (3000, 10_000, 100_000):
        cols, rows, px_w, _ = fit(width, 300, 80, 40, (10, 20))
        assert cols <= 80 and px_w <= 800 and rows >= 1


def test_fit_keeps_aspect():
    _, _, px_w, px_h = fit(1000, 500, 30, 30, (9, 18))
    assert abs(px_w / px_h - 2) < 0.02


def test_fit_blocks_units():
    # Half blocks: one cell is 1x2 pixels.
    assert fit(1024, 1024, 80, 30, (1, 2)) == (60, 30, 60, 60)


def test_grid_shape():
    assert grid_shape(1, 100) == (1, 100)
    assert grid_shape(2, 100) == (2, 49)
    assert grid_shape(4, 100) == (4, 23)
    # Too narrow for 8 tiles of MIN_TILE_COLS: wraps.
    assert grid_shape(8, 100) == (4, 23)  # 4 + 4, not 5 + 3
    assert grid_shape(6, 100) == (3, 32)
    assert grid_shape(3, 20) == (1, 20)


def test_max_rows_override():
    term = Terminal("kitty", rows=50)
    assert term_media.max_rows_for(term, env={}) == 45
    assert term_media.max_rows_for(term, 2, env={}) == 22
    assert term_media.max_rows_for(term, env={"FAL_DEV_PREVIEW_ROWS": "12"}) == 11


# ------------------------------------------------------------- encoders


def test_kitty_sequence_golden():
    assert kitty_sequence(b"PNGDATA", 4, 2, image_id=7) == (
        f"{ESC}_Ga=T,f=100,t=d,q=2,C=1,c=4,r=2,i=7,m=0;UE5HREFUQQ=={ESC}\\"
    )


def test_kitty_sequence_chunks():
    data = bytes(range(256)) * 20  # 5120 bytes, 6828 base64 chars
    seq = kitty_sequence(data, 10, 5)
    parts = seq.split(f"{ESC}\\")[:-1]
    assert len(parts) == 2
    assert parts[0].startswith(f"{ESC}_Ga=T,f=100,t=d,q=2,C=1,c=10,r=5,m=1;")
    assert parts[1].startswith(f"{ESC}_Gm=0;")
    payload = "".join(p.split(";", 1)[1] for p in parts)
    assert base64.b64decode(payload) == data
    assert len(parts[0].split(";", 1)[1]) == term_media.KITTY_CHUNK


def test_iterm_sequence_golden():
    assert iterm_sequence(b"PNGDATA", 4, 2, "a.png") == (
        f"{ESC}]1337;File=inline=1;size=7;name=YS5wbmc=;width=4;height=2;"
        "preserveAspectRatio=1:UE5HREFUQQ==\x07"
    )


def test_tmux_wrap_doubles_escapes():
    assert tmux_wrap(f"{ESC}_Gx{ESC}\\") == (
        f"{ESC}Ptmux;{ESC}{ESC}_Gx{ESC}{ESC}\\{ESC}\\"
    )


def test_blocks_truecolor_golden():
    image = Image.new("RGB", (2, 2))
    image.putpixel((0, 0), (255, 0, 0))
    image.putpixel((1, 0), (255, 0, 0))
    image.putpixel((0, 1), (0, 0, 255))
    image.putpixel((1, 1), (0, 255, 0))
    assert blocks_lines(image) == [
        f"{ESC}[38;2;255;0;0;48;2;0;0;255m▀"
        f"{ESC}[38;2;255;0;0;48;2;0;255;0m▀{ESC}[0m"
    ]


def test_blocks_repeats_skip_color_codes_and_pad_odd_height():
    image = Image.new("RGB", (3, 1), (10, 20, 30))
    (line,) = blocks_lines(image, truecolor=False)
    assert line == f"{ESC}[38;5;233;49m▀▀▀{ESC}[0m"


def test_xterm256():
    assert term_media._xterm256(0, 0, 0) == 16
    assert term_media._xterm256(255, 255, 255) == 231
    assert term_media._xterm256(128, 128, 128) == 244
    assert term_media._xterm256(255, 0, 0) == 196


def test_sixel_golden():
    image = Image.new("RGB", (2, 6), (255, 255, 255))
    for y in range(3):
        image.putpixel((0, y), (0, 0, 0))
    # White (#0) covers rows 3-5 of column 0 (0b111000 -> "w") and all of
    # column 1 ("~"); black (#1) covers rows 0-2 of column 0 (0b000111 -> "F").
    assert sixel_sequence(image) == (
        f'{ESC}Pq"1;1;2;6#0;2;100;100;100#1;2;0;0;0#0w~$#1F-{ESC}\\'
    )


def test_sixel_rle():
    assert term_media._sixel_rle(bytearray([1] * 5 + [2, 0, 0])) == "!5@A"


# --------------------------------------------------------------- layout


def _tile(w, h, caption="cap"):
    return Tile(Image.new("RGB", (w, h), (200, 100, 50)), caption)


def test_renderer_off_prints_paths():
    out = io.StringIO()
    tile = Tile(Image.new("RGB", (1, 1)), "512x512 · infer 9 ms", "out/a.png")
    Renderer(Terminal(OFF), out).show([tile])
    assert out.getvalue() == "512x512 · infer 9 ms | out/a.png\n"


def test_renderer_kitty_layout():
    out = io.StringIO()
    term = Terminal("kitty", cols=40, rows=20, cell=(10, 20))
    Renderer(term, out).show([_tile(100, 40)])
    text = out.getvalue()
    # 100x40 px is 10x2 cells, centered in 40 cols (15 cols margin).
    assert text.startswith(f"\n\n{ESC}[2A\r{ESC}[15C{ESC}7{ESC}_Ga=T,f=100")
    assert ",c=10,r=2,m=0;" in text
    assert text.endswith(
        f"{ESC}8\r{ESC}[2B{ESC}[2m{'cap'.center(40).rstrip()}{ESC}[0m\n"
    )


def test_renderer_tmux_wraps_graphics():
    out = io.StringIO()
    term = Terminal("iterm", cols=40, rows=20, tmux=True)
    Renderer(term, out).show([_tile(100, 40)])
    assert f"{ESC}Ptmux;{ESC}{ESC}]1337;File=" in out.getvalue()


def test_renderer_grid_side_by_side_and_never_overflows():
    out = io.StringIO()
    term = Terminal("blocks", cols=60, rows=30)
    tiles = [_tile(300, 500, "before v2"), _tile(500, 300, "after v3")]
    Renderer(term, out).show(tiles)
    lines = out.getvalue().splitlines()
    plain = [_strip_ansi(line) for line in lines]
    assert all(len(line) <= 60 for line in plain)
    caption = plain[-1]
    assert "before v2" in caption and "after v3" in caption
    assert caption.index("before") < caption.index("after")


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text)


def test_caption_truncates():
    assert term_media._caption("abcdefghij", 5) == "abcd…"
    assert term_media._caption("ab", 6) == "  ab"


def test_show_images_off_skips_decoding(tmp_path):
    out = io.StringIO()
    missing = tmp_path / "never-read.png"
    term_media.show_images([(missing, "")], Terminal(OFF), out)
    assert out.getvalue() == f"{missing}\n"


# --------------------------------------------------------------- fetch


FILE_OUTPUT = {
    "images": [
        {
            "url": "https://v3.fal.media/files/x/a.png",
            "content_type": "image/png",
            "width": 1024,
        },
        {"url": "https://v3.fal.media/files/x/b", "content_type": "image/jpeg"},
    ],
    "video": {"url": "https://v3.fal.media/files/x/c.mp4"},
    "seed": 3,
    "thumb": "https://example.com/t.webp?sig=1",
    "doc": "https://example.com/readme.txt",
    "inline": "data:image/png;base64,UE5H",
}


def test_find_media():
    found = [(m.label, m.content_type) for m in preview.find_media(FILE_OUTPUT)]
    assert found == [
        ("images[0]", "image/png"),
        ("images[1]", "image/jpeg"),
        ("video", ""),
        ("thumb", ""),
        ("inline", "image/png"),
    ]


def test_fetch_data_uri_respects_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(preview, "MEDIA_DIR", tmp_path)
    media = preview.Media("data:image/png;base64,UE5HREFUQQ==", "image/png")
    path = preview.fetch(media)
    assert path.read_bytes() == b"PNGDATA" and path.suffix == ".png"
    with pytest.raises(preview.TooLarge):
        preview.fetch(media, limit=3)


class _Resp(io.BytesIO):
    def __init__(self, data, headers):
        super().__init__(data)
        self.headers = headers

    def geturl(self):
        return "https://v3.fal.media/f"

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_fetch_url_size_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(preview, "MEDIA_DIR", tmp_path)
    calls = []

    def urlopen(req, timeout):
        calls.append(req.full_url)
        return _Resp(b"x" * 100, {"content-type": "image/webp"})

    monkeypatch.setattr(preview._OPENER, "open", urlopen)
    path = preview.fetch(preview.Media("https://v3.fal.media/f/out"), limit=200)
    assert path.suffix == ".webp" and path.stat().st_size == 100
    with pytest.raises(preview.TooLarge):
        preview.fetch(preview.Media("https://v3.fal.media/f/out"), limit=50)

    def declared(req, timeout):
        return _Resp(b"", {"content-length": "999999"})

    monkeypatch.setattr(preview._OPENER, "open", declared)
    with pytest.raises(preview.TooLarge):
        preview.fetch(preview.Media("https://v3.fal.media/f/big.png"), limit=50)


def test_max_bytes_env():
    assert preview.max_bytes({}) == preview.DEFAULT_MAX_BYTES
    assert preview.max_bytes({"FAL_DEV_PREVIEW_MAX_MB": "1.5"}) == 1572864


def _png_bytes(w=8, h=4):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (1, 2, 3)).save(buf, format="PNG")
    return buf.getvalue()


def test_show_app_result_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(preview, "MEDIA_DIR", tmp_path)
    monkeypatch.setenv("FAL_DEV_PREVIEW", "blocks")
    preview.configure(type("A", (), {"preview": None, "play": False})())
    out = io.StringIO()
    result = {
        "content_type": "image/png",
        "body_base64": base64.b64encode(_png_bytes()).decode(),
    }
    preview.show_app_result(result, out=out)
    assert "8x4" in out.getvalue() and "▀" in out.getvalue()


def test_show_app_result_json_downloads(tmp_path, monkeypatch):
    monkeypatch.setattr(preview, "MEDIA_DIR", tmp_path)
    monkeypatch.setenv("FAL_DEV_PREVIEW", "blocks")
    png = _png_bytes(6, 6)
    monkeypatch.setattr(
        preview._OPENER,
        "open",
        lambda req, timeout: _Resp(png, {"content-type": "image/png"}),
    )
    out = io.StringIO()
    body = {"images": [{"url": "https://v3.fal.media/a.png"}] * 2}
    preview.show_app_result({"body": json.dumps(body)}, out=out)
    text = out.getvalue()
    assert "images[0]" in text and "images[1]" in text


def test_show_app_result_off_prints_nothing(monkeypatch):
    monkeypatch.setenv("FAL_DEV_PREVIEW", "off")
    out = io.StringIO()
    preview.show_app_result({"body": json.dumps(FILE_OUTPUT)}, out=out)
    assert out.getvalue() == ""


def test_show_app_result_never_raises(monkeypatch, capsys):
    monkeypatch.setenv("FAL_DEV_PREVIEW", "blocks")

    def save_bytes(*args):
        raise OSError("read-only home")

    monkeypatch.setattr(preview, "save_bytes", save_bytes)
    preview.show_app_result({"content_type": "image/png", "body_base64": "!!"})
    assert "preview failed" in capsys.readouterr().err


def test_hot_preview_caption_and_compare(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("FAL_DEV_PREVIEW", "blocks")
    monkeypatch.setenv("COLUMNS", "200")
    monkeypatch.chdir(tmp_path)
    first, second = Path("hot-out/1.png"), Path("hot-out/2.png")
    first.parent.mkdir()
    first.write_bytes(_png_bytes(16, 16))
    second.write_bytes(_png_bytes(16, 16))
    args = type("A", (), {"preview": None, "play": False, "compare": True})()
    hook = preview.HotPreview(args)
    hook(first, {"x-hot-infer-ms": "570", "x-hot-code-version": "2"})
    hook(second, {"x-hot-infer-ms": "561", "x-hot-code-version": "3"})
    last_caption = capsys.readouterr().out.strip().splitlines()[-1]
    assert "infer 570 ms · code v2 · hot-out/1.png" in last_caption
    assert "code v3 · hot-out/2.png" in last_caption


# ---------------------------------------------------------------- video


def test_sample_times():
    assert term_video.sample_times(6.0, 3) == [1.0, 3.0, 5.0]
    assert term_video.sample_times(0, 3) == [0.0]


def test_is_video():
    assert term_video.is_video("a.MP4")
    assert term_video.is_video("blob", "video/webm")
    assert not term_video.is_video("a.png", "image/png")


def test_video_without_ffmpeg_hints(tmp_path, monkeypatch):
    monkeypatch.setattr(term_video, "has_ffmpeg", lambda: False)
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"\0" * 10)
    out = io.StringIO()
    term_video.show_video(str(clip), term=Terminal("blocks"), out=out)
    assert "install ffmpeg" in out.getvalue()


def test_animated_gif_without_ffmpeg(tmp_path, monkeypatch):
    monkeypatch.setattr(term_video, "has_ffmpeg", lambda: False)
    frames = [Image.new("RGB", (20, 10), (i * 30, 0, 0)) for i in range(8)]
    gif = tmp_path / "a.gif"
    frames[0].save(gif, save_all=True, append_images=frames[1:], duration=50)
    out = io.StringIO()
    term_video.show_video(str(gif), term=Terminal("blocks", cols=120), out=out)
    text = out.getvalue()
    assert "8 frames" in text and "frame 0" in text


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")
def test_video_contact_sheet_with_ffmpeg(tmp_path):
    clip = tmp_path / "clip.mp4"
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-f", "lavfi",
            "-i", "testsrc2=size=160x90:rate=10:duration=2",
            "-pix_fmt", "yuv420p", str(clip),
        ],
        check=True,
    )  # fmt: skip
    out = io.StringIO()
    term_video.show_video(
        str(clip), term=Terminal("blocks", cols=120, rows=40), frames=4, out=out
    )
    text = out.getvalue()
    assert "160x90 · 2.0s · 10 fps" in text
    assert "0:00.2" in text and "0:01.8" in text


def test_kitty_animation_structure():
    frames = [Image.new("RGB", (4, 4), (i, i, i)) for i in range(3)]
    seq = term_video.kitty_animation(frames, 100, 2, 1, image_id=5)
    assert seq.count("a=f,i=5,f=100,z=100") == 2
    assert seq.endswith(f"{ESC}_Ga=a,i=5,s=3,v=1,q=2{ESC}\\")


def test_play_needs_animation_support():
    out = io.StringIO()
    renderer = Renderer(Terminal("blocks"), out)
    term_video.play_video("x.mp4", renderer)
    assert "--play needs" in out.getvalue()


def test_hot_request_calls_on_image(tmp_path, monkeypatch):
    box = hot.HotBox.__new__(hot.HotBox)
    headers = {"content-type": "image/png", "x-hot-code-version": "4"}
    monkeypatch.setattr(box, "post", lambda *a, **k: (200, headers, b"PNG"))
    seen = []
    box.on_image = lambda path, hdrs: seen.append((path, hdrs))
    out = tmp_path / "o.png"
    assert box.request({}, str(out))
    assert seen == [(out, headers)] and out.read_bytes() == b"PNG"


# ------------------------------------------------------------- hardening


def test_captions_cannot_send_escape_sequences(tmp_path, monkeypatch):
    monkeypatch.setattr(preview, "MEDIA_DIR", tmp_path)
    monkeypatch.setenv("FAL_DEV_PREVIEW", "blocks")
    monkeypatch.setenv("COLUMNS", "200")
    png = _png_bytes(4, 4)
    monkeypatch.setattr(
        preview._OPENER,
        "open",
        lambda req, timeout: _Resp(png, {"content-type": "image/png"}),
    )
    out = io.StringIO()
    key = "x\x1b]52;c;aGk=\x07"
    body = {key: {"url": "https://v3.fal.media/a.png"}}
    preview.show_app_result({"body": json.dumps(body)}, out=out)
    captions = [line for line in out.getvalue().splitlines() if "4x4" in line]
    assert captions and "x?]52;c;aGk=?" in captions[0]
    assert "\x1b]52" not in out.getvalue() and "\x07" not in out.getvalue()


def test_clean():
    assert common.clean("a\x1b[2Jb\x9bc\n") == "a?[2Jb?c?"


def test_save_bytes_is_atomic(tmp_path, monkeypatch):
    monkeypatch.setattr(preview, "MEDIA_DIR", tmp_path)
    path = preview.save_bytes(b"data", "image/png")
    assert path.read_bytes() == b"data"
    assert [p.name for p in tmp_path.iterdir()] == [path.name]


def test_data_uri_cap_checked_before_decoding(monkeypatch):
    def decode(data):
        raise AssertionError("decoded before the size check")

    monkeypatch.setattr(preview.base64, "b64decode", decode)
    with pytest.raises(preview.TooLarge):
        preview.fetch(preview.Media("data:image/png;base64," + "A" * 400), limit=10)


def test_video_content_type_without_suffix_routes_to_video(tmp_path, monkeypatch):
    monkeypatch.setattr(preview, "MEDIA_DIR", tmp_path)
    path = preview.save_bytes(b"\0" * 8, "video/mp4", "https://x/out")
    assert path.suffix == ".mp4" and term_video.is_video(path)


def test_animated_gif_routes_to_video_path(tmp_path, monkeypatch):
    monkeypatch.setattr(term_video, "has_ffmpeg", lambda: False)
    frames = [Image.new("RGB", (20, 10), (i * 60, 255 - i * 60, 0)) for i in range(4)]
    gif = tmp_path / "a.gif"
    frames[0].save(gif, save_all=True, append_images=frames[1:], duration=50)
    still = tmp_path / "s.gif"
    frames[0].save(still)
    assert term_video.is_animated(gif) and not term_video.is_animated(still)
    out = io.StringIO()
    preview.show_paths(
        [(str(gif), ""), (str(still), "")], Terminal("blocks", cols=120), out=out
    )
    assert "4 frames" in out.getvalue() and "frame 3" in out.getvalue()


def test_animated_frames_are_capped(tmp_path):
    frames = [Image.new("RGB", (4, 4), (i, 0, 0)) for i in range(200)]
    gif = tmp_path / "long.gif"
    frames[0].save(gif, save_all=True, append_images=frames[1:], duration=20)
    anim = term_video._animated_frames(str(gif), 6, 64)
    assert len(anim.frames) == 6 and anim.indexes[0] == 0 and anim.indexes[-1] < 200
    # 200 frames of 20 ms shown as 6 keep the same total duration.
    assert (anim.total, anim.delay_ms) == (200, 20 * 200 // 6)


def test_too_many_pixels_refused(monkeypatch, tmp_path):
    monkeypatch.setattr(term_media, "MAX_IMAGE_PIXELS", 100)
    path = tmp_path / "big.png"
    Image.new("RGB", (20, 20)).save(path)
    with pytest.raises(ValueError, match="too large"):
        term_media.load(path)


def test_ffmpeg_runs_hardened(monkeypatch):
    calls = []

    def run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(term_video.subprocess, "run", run)
    with pytest.raises(ValueError, match="ffmpeg failed"):
        term_video.frame_at("-evil.mp4", 0, 64)
    assert term_video.probe("-evil.mp4") is None
    for cmd, kwargs in calls:
        assert kwargs["stdin"] is subprocess.DEVNULL
        assert cmd[cmd.index("-protocol_whitelist") + 1] == "file"
        assert not any(arg.startswith("-evil") for arg in cmd)
    assert "-nostdin" in calls[0][0]


def test_sheet_survives_missing_frames(monkeypatch):
    monkeypatch.setattr(term_video, "has_ffmpeg", lambda: True)
    monkeypatch.setattr(
        term_video, "probe", lambda path: term_video.VideoInfo(64, 32, 8.0, 24)
    )

    def frame_at(path, t, max_width):
        if t > 1:  # audio runs past the last video frame
            raise OSError("cannot identify image file")
        return Image.new("RGB", (64, 32))

    monkeypatch.setattr(term_video, "frame_at", frame_at)
    out = io.StringIO()
    term_video.show_video("a.mp4", term=Terminal("blocks", cols=120), out=out)
    assert "64x32 · 8.0s · 24 fps" in out.getvalue()


def test_broken_video_prints_path(monkeypatch):
    monkeypatch.setattr(term_video, "has_ffmpeg", lambda: True)
    monkeypatch.setattr(term_video, "probe", lambda path: None)
    out = io.StringIO()
    term_video.show_video("bad.mp4", term=Terminal("blocks"), out=out)
    assert out.getvalue() == "bad.mp4 (no preview: ffprobe could not read it)\n"


def test_encode_failure_leaves_cursor_alone(monkeypatch):
    def boom(*args):
        raise RuntimeError("enc")

    out = io.StringIO()
    renderer = Renderer(Terminal("kitty", cols=40, rows=20), out)
    monkeypatch.setattr(renderer, "_encode", boom)
    with pytest.raises(RuntimeError):
        renderer.show([_tile(100, 40)])
    assert out.getvalue() == ""


def test_kitty_play_frames_are_letterboxed(monkeypatch):
    frames = [Image.new("RGB", (480, 270)) for _ in range(2)]
    monkeypatch.setattr(term_video, "_play_frames", lambda path: (frames, 100))
    captured = {}

    def animation(boxed, delay, cols, rows, image_id):
        captured["size"], captured["cells"] = boxed[0].size, (cols, rows)
        return ""

    monkeypatch.setattr(term_video, "kitty_animation", animation)
    term = Terminal("kitty", cols=60, rows=24, cell=(10, 20))
    term_video.play_video("a.mp4", Renderer(term, io.StringIO()))
    cols, rows = captured["cells"]
    assert captured["size"] == (cols * 10, rows * 20)


def test_terminal_for_keeps_detected_fields(monkeypatch):
    detected = Terminal("kitty", cell=(7, 15), truecolor=False)
    monkeypatch.setattr(term_media, "detect", lambda: detected)
    off = term_media.terminal_for(type("A", (), {"preview": False})())
    assert (off.protocol, off.cell, off.truecolor) == (OFF, (7, 15), False)


def test_view_reports_bad_items(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("FAL_DEV_PREVIEW", "off")
    good = tmp_path / "g.png"
    Image.new("RGB", (4, 4)).save(good)
    args = type(
        "A", (), {"files": [str(good), "missing.png"], "play": False, "preview": None}
    )()
    assert preview.view(args) == 1
    captured = capsys.readouterr()
    assert "missing.png: no such file" in captured.err
    assert str(good) in captured.out


def _hot_args(**kwargs):
    return type("A", (), {"preview": None, "play": False, "compare": True, **kwargs})()


def test_compare_keeps_previous_image_when_path_is_reused(tmp_path, monkeypatch):
    monkeypatch.setenv("FAL_DEV_PREVIEW", "blocks")
    monkeypatch.setenv("COLUMNS", "120")
    monkeypatch.chdir(tmp_path)
    out = Path("o.png")
    hook = preview.HotPreview(_hot_args())
    shown = []
    monkeypatch.setattr(
        term_media, "show_images", lambda items, term: shown.append(items)
    )
    Image.new("RGB", (8, 8), (255, 0, 0)).save(out)
    hook(out, {})
    Image.new("RGB", (8, 8), (0, 0, 255)).save(out)
    hook(out, {})
    before, after = shown[-1]
    assert before[0].getpixel((0, 0)) == (255, 0, 0)
    assert after[0].getpixel((0, 0)) == (0, 0, 255)


def test_compare_recovers_after_a_bad_result(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("FAL_DEV_PREVIEW", "blocks")
    monkeypatch.chdir(tmp_path)
    hook = preview.HotPreview(_hot_args())
    hook(Path("missing.png"), {})
    good = Path("g.png")
    Image.new("RGB", (8, 8)).save(good)
    hook(good, {})
    captured = capsys.readouterr()
    assert "preview failed" in captured.err and "8x8 · g.png" in captured.out


def test_one_bad_image_keeps_the_rest_of_the_grid(tmp_path):
    good, bad = tmp_path / "g.png", tmp_path / "b.png"
    Image.new("RGB", (8, 8)).save(good)
    bad.write_bytes(b"<html>not an image</html>")
    out = io.StringIO()
    term_media.show_images(
        [(good, ""), (bad, "")], Terminal("blocks", cols=80), out=out
    )
    text = out.getvalue()
    assert "b.png: not previewed" in text and "8x8" in text


def test_only_known_formats_are_opened(tmp_path):
    eps = tmp_path / "x.png"
    eps.write_bytes(b"%!PS-Adobe-3.0 EPSF-3.0\n%%BoundingBox: 0 0 10 10\n")
    with pytest.raises(Exception, match="cannot identify"):
        term_media.load(eps)


def test_media_per_response_are_capped(tmp_path, monkeypatch):
    monkeypatch.setattr(preview, "MEDIA_DIR", tmp_path)
    monkeypatch.setenv("FAL_DEV_PREVIEW", "blocks")
    fetched = []

    def fetch(media):
        fetched.append(media)
        raise preview.TooLarge("skip")

    monkeypatch.setattr(preview, "fetch", fetch)
    body = {"images": [{"url": f"https://v3.fal.media/{i}.png"} for i in range(20)]}
    out = io.StringIO()
    preview.show_app_result({"body": json.dumps(body)}, out=out)
    assert len(fetched) == preview.MAX_MEDIA
    assert f"first {preview.MAX_MEDIA} of 20" in out.getvalue()


def test_truncated_download_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(preview, "MEDIA_DIR", tmp_path)
    monkeypatch.setattr(
        preview._OPENER,
        "open",
        lambda req, timeout: _Resp(b"x" * 10, {"content-length": "100"}),
    )
    with pytest.raises(ValueError, match="cut short"):
        preview.fetch(preview.Media("https://v3.fal.media/a.png"))
    assert list(tmp_path.iterdir()) == []


def test_large_animations_are_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(term_video, "ANIMATED_MAX_PIXELS", 50)
    frames = [Image.new("RGB", (10, 10), (i * 80, 0, 0)) for i in range(3)]
    gif = tmp_path / "a.gif"
    frames[0].save(gif, save_all=True, append_images=frames[1:])
    with pytest.raises(ValueError, match="too large"):
        term_video._animated_frames(str(gif), 6, 64)


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://host/a.png"])
def test_fetch_refuses_non_http_urls(url, monkeypatch):
    def urlopen(req, timeout):
        raise AssertionError("opened")

    monkeypatch.setattr(preview._OPENER, "open", urlopen)
    media = next(preview.find_media({"url": url, "content_type": "image/png"}))
    with pytest.raises(ValueError, match="only http"):
        preview.fetch(media)


def test_fetch_refuses_redirect_off_http(monkeypatch):
    resp = _Resp(b"x", {})
    resp.geturl = lambda: "ftp://host/a.png"
    monkeypatch.setattr(preview._OPENER, "open", lambda req, timeout: resp)
    with pytest.raises(ValueError, match="redirected"):
        preview.fetch(preview.Media("https://v3.fal.media/a.png"))


def test_max_bytes_ignores_infinite():
    assert (
        preview.max_bytes({"FAL_DEV_PREVIEW_MAX_MB": "inf"})
        == preview.DEFAULT_MAX_BYTES
    )


def test_redirect_to_ftp_is_refused_before_connecting():
    handler = preview._HttpOnlyRedirects()
    req = preview.urllib.request.Request("https://v3.fal.media/a.png")
    with pytest.raises(ValueError, match="redirected"):
        handler.redirect_request(req, None, 302, "Found", {}, "ftp://host/x.png")
    follow = handler.redirect_request(
        req, None, 302, "Found", {}, "https://cdn.example/x.png"
    )
    assert follow.full_url == "https://cdn.example/x.png"


def test_cache_suffix_is_always_media(tmp_path):
    assert preview._suffix("application/x-sh", "https://x/out") == ".bin"
    assert preview._suffix("text/html", "https://x/out") == ".bin"
    assert preview._suffix("image/jpeg", "https://x/out") == ".jpg"


def test_find_media_keeps_nested_media_next_to_a_file():
    body = {"url": "https://x/a.png", "thumbnail": {"url": "https://x/t.png"}}
    assert [m.label for m in preview.find_media(body)] == ["url", "thumbnail"]


def test_16_bit_images_are_stretched_not_clipped(tmp_path):
    path = tmp_path / "depth.png"
    depth = Image.new("I;16", (64, 1))
    for x in range(64):
        depth.putpixel((x, 0), x * 1000)
    depth.save(path)
    image = term_media.load(path)
    assert image.mode == "L"
    assert image.getpixel((0, 0)) == 0 and image.getpixel((63, 0)) == 255
    assert 100 < image.getpixel((32, 0)) < 160


def test_fallback_note_is_printed_once(monkeypatch):
    monkeypatch.setattr(term_media, "_NOTES_SHOWN", set())
    out = io.StringIO()
    renderer = Renderer(Terminal("blocks", cols=40, note="hint"), out)
    renderer.show([_tile(4, 4)])
    renderer.show([_tile(4, 4)])
    assert out.getvalue().count("(hint)") == 1


def test_ffmpeg_caps_both_sides():
    assert "h='min(640,ih)'" in term_video._fit_filter(640)
    assert "force_original_aspect_ratio=decrease" in term_video._fit_filter(640)


@pytest.mark.parametrize("mode", ["CMYK", "YCbCr", "LAB", "PA"])
def test_unusual_modes_render_on_every_protocol(tmp_path, mode):
    path = tmp_path / "x.tif"
    source = Image.new(mode, (8, 8))
    try:
        source.save(path)
    except OSError:
        pytest.skip(f"Pillow cannot write {mode} TIFF")
    image = term_media.load(path)
    for protocol in ("kitty", "iterm", "sixel", "blocks"):
        Renderer(Terminal(protocol, cols=40), io.StringIO()).show([Tile(image, "x")])


def test_non_media_data_uri_in_file_object_is_skipped():
    body = {"audio": {"url": "data:audio/wav;base64,UklGRg==", "content_type": ""}}
    assert list(preview.find_media(body)) == []
    image = {"url": "data:image/png;base64,UE5H"}
    assert [m.content_type for m in preview.find_media(image)] == ["image/png"]


def test_ffmpeg_errors_show_ffmpeg_message(monkeypatch):
    def run(cmd, **kwargs):
        raise subprocess.CalledProcessError(
            1, cmd, stderr=b"warning\nclip.mp4: Invalid data found\n"
        )

    monkeypatch.setattr(term_video.subprocess, "run", run)
    with pytest.raises(ValueError) as info:
        term_video.frame_at("clip.mp4", 0, 64)
    assert str(info.value) == "ffmpeg failed: clip.mp4: Invalid data found"

    def slow(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, 60)

    monkeypatch.setattr(term_video.subprocess, "run", slow)
    with pytest.raises(ValueError, match="ffmpeg timed out after 60s"):
        term_video.frame_at("clip.mp4", 0, 64)


def test_preview_flag_forces_rendering_when_piped(monkeypatch):
    monkeypatch.setenv("FAL_DEV_PREVIEW", "off")
    monkeypatch.setenv("TERM_PROGRAM", "iTerm.app")
    monkeypatch.delenv("TMUX", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    forced = term_media.terminal_for(type("A", (), {"preview": True})())
    assert forced.protocol == "iterm"
    default = term_media.terminal_for(type("A", (), {"preview": None})())
    assert default.protocol == OFF


def test_preview_flag_keeps_tmux_rules(monkeypatch):
    monkeypatch.delenv("FAL_DEV_PREVIEW", raising=False)
    monkeypatch.setenv("TMUX", "/tmp/tmux-1/default")
    monkeypatch.setenv("KITTY_WINDOW_ID", "1")
    forced = term_media.terminal_for(type("A", (), {"preview": True})())
    assert forced.protocol == "blocks"


@pytest.mark.parametrize(
    "argv",
    [
        ["dev", "hot", "run", "--no-preview", "--compare", "--play"],
        ["dev", "app", "request", "/", "--preview", "--play"],
        ["dev", "app", "sync", "--no-preview"],
        ["dev", "view", "a.png", "--play"],
    ],
)
def test_preview_flags_are_registered(argv):
    from fal.cli.main import parse_args

    args = parse_args(argv)
    expected = {"--no-preview": False, "--preview": True}
    assert args.preview is next((v for k, v in expected.items() if k in argv), None)
    assert args.play is ("--play" in argv)


def test_glue_imports_no_pillow():
    code = (
        "import sys; import fal.dev.cli.parser, fal.dev.media.preview; "
        "from fal.dev.media import preview; preview.show_app_result({'body': '{}'}); "
        "assert 'PIL' not in sys.modules, 'PIL imported'"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
