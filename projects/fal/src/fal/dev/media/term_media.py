"""Inline image previews in the terminal for `fal dev` results.

Picks the best graphics protocol the terminal speaks (kitty graphics, iTerm2
inline images, sixel, else Unicode half blocks), fits images to the visible cell
area without upscaling, centers them, and writes a one-line caption under each.
Output that is not a terminal gets file paths only.

Force a protocol with FAL_DEV_PREVIEW=kitty|iterm|sixel|blocks|off.
"""

from __future__ import annotations

import base64
import io
import math
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, replace
from typing import IO, Iterable, Mapping, Sequence

from PIL import Image

from fal.console.icons import (
    UPPER_HALF_BLOCK,
    get_ellipsis_icon,
    get_inline_separator_icon,
)
from fal.dev.media.common import ENV_MAX_ROWS, ENV_PROTOCOL, IMAGE_FORMATS, OFF, clean

PROTOCOLS = ("kitty", "iterm", "sixel", "blocks")
# Typical cell size in pixels when the terminal does not report one.
DEFAULT_CELL = (10, 20)
GAP_COLS = 2
MIN_TILE_COLS = 16
KITTY_CHUNK = 4096
SIXEL_COLORS = 128
# Pixel budget for sixel, which is encoded in pure Python.
SIXEL_MAX_PIXELS = 900_000
# Larger images are refused rather than decoded (about 200 MB as RGBA).
MAX_IMAGE_PIXELS = 50_000_000
ESC = "\x1b"
ST = ESC + "\\"


@dataclass(frozen=True)
class Terminal:
    protocol: str  # one of PROTOCOLS, or OFF
    cols: int = 80
    rows: int = 24
    cell: tuple[int, int] = DEFAULT_CELL
    truecolor: bool = True
    tmux: bool = False
    note: str = ""  # why we fell back, shown once


@dataclass
class Tile:
    """One image to show, with its caption."""

    image: Image.Image
    caption: str = ""
    path: str | None = None


# ---------------------------------------------------------------- detection


def _in_tmux(env: Mapping[str, str]) -> bool:
    term = env.get("TERM", "")
    return bool(env.get("TMUX")) or term.startswith(("tmux", "screen"))


def outer_protocol(env: Mapping[str, str]) -> str:
    """The protocol of the terminal emulator itself, from its environment."""
    term = env.get("TERM", "")
    program = env.get("TERM_PROGRAM", "")
    lc_terminal = env.get("LC_TERMINAL", "")
    if env.get("KITTY_WINDOW_ID") or term == "xterm-kitty":
        return "kitty"
    if program.lower() == "ghostty" or term == "xterm-ghostty":
        return "kitty"
    # WezTerm speaks both; its iTerm2 support is the more complete one.
    if program == "WezTerm" or env.get("WEZTERM_PANE"):
        return "iterm"
    if program == "iTerm.app" or lc_terminal == "iTerm2":
        return "iterm"
    if program == "mintty":
        return "iterm"
    if term.startswith(("foot", "mlterm", "contour")) or env.get("WT_SESSION"):
        return "sixel"
    return "blocks"


def _tmux_passthrough() -> bool:
    try:
        out = subprocess.run(
            ["tmux", "show", "-gv", "allow-passthrough"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return False
    return out in ("on", "all")


def _cell_pixels(stream: IO) -> tuple[int, int] | None:
    """Cell size from TIOCGWINSZ pixel fields, when the terminal fills them."""
    try:
        import fcntl
        import struct
        import termios

        packed = fcntl.ioctl(stream.fileno(), termios.TIOCGWINSZ, b"\0" * 8)
        rows, cols, xpix, ypix = struct.unpack("HHHH", packed)
    except Exception:
        return None
    if not (rows and cols and xpix and ypix):
        return None
    return max(1, xpix // cols), max(1, ypix // rows)


def detect(
    env: Mapping[str, str] | None = None,
    stream: IO | None = None,
    *,
    isatty: bool | None = None,
    passthrough: bool | None = None,
) -> Terminal:
    env = os.environ if env is None else env
    stream = sys.stdout if stream is None else stream
    tty = stream.isatty() if isatty is None else isatty
    size = shutil.get_terminal_size((80, 24))
    cell = _cell_pixels(stream) if tty else None
    base = Terminal(
        OFF,
        cols=size.columns,
        rows=size.lines,
        cell=cell or DEFAULT_CELL,
        truecolor=env.get("COLORTERM", "").lower() in ("truecolor", "24bit")
        or outer_protocol(env) != "blocks",
        tmux=_in_tmux(env),
    )
    forced = env.get(ENV_PROTOCOL, "").strip().lower()
    if forced in ("off", "none", "0", "false", "no"):
        return base
    if forced in PROTOCOLS:  # also when piped, to capture the bytes
        return replace(base, protocol=forced)
    if not tty:
        return base
    protocol = outer_protocol(env)
    if base.tmux and protocol == "kitty":
        return replace(
            base, protocol="blocks", note="kitty graphics do not survive tmux"
        )
    if base.tmux and protocol in ("iterm", "sixel"):
        ok = _tmux_passthrough() if passthrough is None else passthrough
        if not ok:
            note = "for sharp images in tmux: set -g allow-passthrough on"
            return replace(base, protocol="blocks", note=note)
    return replace(base, protocol=protocol)


# -------------------------------------------------------------------- sizing


def fit(
    width: int,
    height: int,
    max_cols: int,
    max_rows: int,
    cell: tuple[int, int],
) -> tuple[int, int, int, int]:
    """(cols, rows, px_w, px_h) for an image fit into a cell box.

    Keeps the aspect ratio and never upscales. cols and rows cover the image.
    """
    cell_w, cell_h = cell
    max_cols, max_rows = max(1, max_cols), max(1, max_rows)
    scale = min(1.0, max_cols * cell_w / width, max_rows * cell_h / height)
    px_w = max(1, round(width * scale))
    px_h = max(1, round(height * scale))
    cols = min(max_cols, max(1, math.ceil(px_w / cell_w)))
    rows = min(max_rows, max(1, math.ceil(px_h / cell_h)))
    return cols, rows, px_w, px_h


def grid_shape(count: int, cols: int, gap: int = GAP_COLS) -> tuple[int, int]:
    """(tiles per row, tile width in cells) for `count` images side by side."""
    per_row = max(1, min(count, (cols + gap) // (MIN_TILE_COLS + gap)))
    per_row = math.ceil(count / math.ceil(count / per_row))
    tile_cols = (cols - gap * (per_row - 1)) // per_row
    return per_row, max(1, tile_cols)


def max_rows_for(term: Terminal, grid_rows: int = 1, env=None) -> int:
    env = os.environ if env is None else env
    try:
        budget = int(env.get(ENV_MAX_ROWS, ""))
    except ValueError:
        budget = term.rows - 4  # room for a caption and the prompt
    # Each grid row of tiles also needs its caption line.
    return max(4, budget // grid_rows - 1)


# ----------------------------------------------------------------- encoders


def flatten(image: Image.Image, background=(24, 24, 24)) -> Image.Image:
    if image.mode in ("RGBA", "LA") or "transparency" in image.info:
        rgba = image.convert("RGBA")
        canvas = Image.new("RGB", rgba.size, background)
        canvas.paste(rgba, mask=rgba.split()[3])
        return canvas
    return image.convert("RGB")


def png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def tmux_wrap(seq: str) -> str:
    """DCS passthrough so tmux forwards a sequence to the outer terminal."""
    return f"{ESC}Ptmux;{seq.replace(ESC, ESC + ESC)}{ST}"


def kitty_sequence(
    png: bytes, cols: int, rows: int, *, image_id: int | None = None
) -> str:
    """Transmit and display a PNG in a cols x rows cell box; cursor stays put."""
    ident = f",i={image_id}" if image_id else ""
    return kitty_chunks(png, f"a=T,f=100,t=d,q=2,C=1,c={cols},r={rows}{ident}")


def kitty_chunks(png: bytes, head: str) -> str:
    """A kitty graphics command with its base64 payload split into chunks."""
    data = base64.standard_b64encode(png).decode()
    chunks = [data[i : i + KITTY_CHUNK] for i in range(0, len(data), KITTY_CHUNK)]
    chunks = chunks or [""]
    out = []
    for index, chunk in enumerate(chunks):
        more = 1 if index < len(chunks) - 1 else 0
        keys = f"{head},m={more}" if index == 0 else f"m={more}"
        out.append(f"{ESC}_G{keys};{chunk}{ST}")
    return "".join(out)


def letterbox(image: Image.Image, cols: int, rows: int, cell: tuple[int, int]):
    """Center an image on a transparent cols x rows cell box.

    Kitty stretches an image to fill the cells it is given; at the box's own
    aspect the scaling is uniform.
    """
    box = Image.new("RGBA", (cols * cell[0], rows * cell[1]), (0, 0, 0, 0))
    offset = ((box.width - image.width) // 2, (box.height - image.height) // 2)
    box.paste(image.convert("RGBA"), offset)
    return box


def iterm_sequence(data: bytes, cols: int, rows: int, name: str = "preview") -> str:
    """OSC 1337 inline file, fit into cols x rows cells preserving aspect."""
    b64name = base64.b64encode(name.encode()).decode()
    payload = base64.b64encode(data).decode()
    return (
        f"{ESC}]1337;File=inline=1;size={len(data)};name={b64name};"
        f"width={cols};height={rows};preserveAspectRatio=1:{payload}\x07"
    )


def sixel_sequence(image: Image.Image, colors: int = SIXEL_COLORS) -> str:
    """Encode an RGB image as sixel with an adaptive palette."""
    rgb = flatten(image)
    quant = rgb.quantize(colors=colors, method=Image.Quantize.MEDIANCUT, dither=0)
    palette = quant.getpalette() or []
    width, height = quant.size
    pixels = quant.tobytes()
    used = sorted(set(pixels))
    out = [f"{ESC}Pq", f'"1;1;{width};{height}']
    for idx in used:
        r, g, b = palette[idx * 3 : idx * 3 + 3]
        out.append(f"#{idx};2;{r * 100 // 255};{g * 100 // 255};{b * 100 // 255}")
    for top in range(0, height, 6):
        masks: dict[int, bytearray] = {}
        for bit in range(min(6, height - top)):
            row = pixels[(top + bit) * width : (top + bit + 1) * width]
            weight = 1 << bit
            for x, idx in enumerate(row):
                band = masks.get(idx)
                if band is None:
                    band = masks[idx] = bytearray(width)
                band[x] |= weight
        first = True
        for idx in sorted(masks):
            if not first:
                out.append("$")  # back to the start of the band
            first = False
            out.append(f"#{idx}")
            out.append(_sixel_rle(masks[idx]))
        out.append("-")
    out.append(ST)
    return "".join(out)


def _sixel_rle(band: bytearray) -> str:
    out = []
    data = bytes(band).rstrip(b"\0")
    i = 0
    while i < len(data):
        value = data[i]
        j = i
        while j < len(data) and data[j] == value:
            j += 1
        char = chr(63 + value)
        run = j - i
        out.append(f"!{run}{char}" if run > 3 else char * run)
        i = j
    return "".join(out)


def _xterm256(r: int, g: int, b: int) -> int:
    def level(v: int) -> int:
        return 0 if v < 48 else 1 if v < 115 else (v - 35) // 40

    cube = 16 + 36 * level(r) + 6 * level(g) + level(b)
    gray = round((r + g + b) / 3)
    gray_idx = 232 + min(23, max(0, (gray - 8) // 10))
    steps = [0, 95, 135, 175, 215, 255]
    cube_rgb = (steps[level(r)], steps[level(g)], steps[level(b)])
    gray_rgb = (8 + 10 * (gray_idx - 232),) * 3

    def dist(c):
        return (c[0] - r) ** 2 + (c[1] - g) ** 2 + (c[2] - b) ** 2

    return cube if dist(cube_rgb) <= dist(gray_rgb) else gray_idx


def blocks_lines(image: Image.Image, truecolor: bool = True) -> list[str]:
    """Upper-half-block art: each cell shows two pixels (fg above, bg below)."""
    rgb = flatten(image)
    width, height = rgb.size
    px = rgb.load()

    def color(kind: int, c) -> str:
        if c is None:
            return f"{kind + 1}"  # 49: the terminal's own background
        if truecolor:
            return f"{kind};2;{c[0]};{c[1]};{c[2]}"
        return f"{kind};5;{_xterm256(*c)}"

    lines = []
    for y in range(0, height, 2):
        parts, last = [], None
        for x in range(width):
            # An odd last row leaves the lower half to the terminal background.
            pair = (px[x, y], px[x, y + 1] if y + 1 < height else None)
            if pair != last:
                parts.append(f"{ESC}[{color(38, pair[0])};{color(48, pair[1])}m")
                last = pair
            parts.append(UPPER_HALF_BLOCK)
        parts.append(f"{ESC}[0m")
        lines.append("".join(parts))
    return lines


# ------------------------------------------------------------------ layout


def _caption(text: str, width: int) -> str:
    text = clean(text)
    if len(text) > width:
        text = text[: max(1, width - 1)] + get_ellipsis_icon()
    return text.center(width).rstrip()


# Fallback hints are printed once per process, not after every result.
_NOTES_SHOWN: set[str] = set()


class Renderer:
    """Writes tiles to a stream for one detected terminal."""

    def __init__(self, term: Terminal, out: IO | None = None) -> None:
        self.term = term
        self.out = out or sys.stdout

    def _write(self, text: str) -> None:
        self.out.write(text)

    def graphic(self, seq: str) -> str:
        return tmux_wrap(seq) if self.term.tmux else seq

    def show(self, tiles: Sequence[Tile]) -> None:
        if not tiles:
            return
        term = self.term
        if term.protocol == OFF:
            for tile in tiles:
                line = tile.caption
                if tile.path and tile.path not in line:
                    line = f"{line} | {tile.path}" if line else tile.path
                self._write(clean(line) + "\n")
            self.out.flush()
            return
        per_row, tile_cols = grid_shape(len(tiles), term.cols)
        grid_rows = math.ceil(len(tiles) / per_row)
        max_rows = max_rows_for(term, min(grid_rows, 2))
        for start in range(0, len(tiles), per_row):
            self._show_row(list(tiles[start : start + per_row]), tile_cols, max_rows)
        if term.note and term.note not in _NOTES_SHOWN:
            _NOTES_SHOWN.add(term.note)
            self._write(f"({term.note})\n")
        self.out.flush()

    def _cell(self) -> tuple[int, int]:
        return (1, 2) if self.term.protocol == "blocks" else self.term.cell

    def _show_row(self, tiles: list[Tile], tile_cols: int, max_rows: int) -> None:
        term, cell = self.term, self._cell()
        sized = []
        for tile in tiles:
            w, h = tile.image.size
            sized.append(fit(w, h, tile_cols, max_rows, cell))
        row_rows = max(s[1] for s in sized)
        used = len(tiles) * tile_cols + GAP_COLS * (len(tiles) - 1)
        margin = max(0, (term.cols - used) // 2)
        if term.protocol == "blocks":
            self._row_blocks(tiles, sized, tile_cols, row_rows, margin)
        else:
            self._row_graphics(tiles, sized, tile_cols, row_rows, margin)
        captions = []
        for tile in tiles:
            captions.append(_caption(tile.caption, tile_cols).ljust(tile_cols))
        line = " " * margin + (" " * GAP_COLS).join(captions)
        self._write(f"{ESC}[2m{line.rstrip()}{ESC}[0m\n")

    def _row_blocks(self, tiles, sized, tile_cols, row_rows, margin) -> None:
        columns = []
        for tile, (cols, rows, px_w, px_h) in zip(tiles, sized):
            small = tile.image.resize((px_w, px_h), Image.Resampling.LANCZOS)
            lines = blocks_lines(small, self.term.truecolor)
            pad_left = (tile_cols - px_w) // 2
            top = row_rows - len(lines)  # bottom-aligned: captions sit under images
            blank = " " * tile_cols
            body = [blank] * top
            body += [
                " " * pad_left + ln + " " * (tile_cols - pad_left - px_w)
                for ln in lines
            ]
            body += [blank] * (row_rows - len(body))
            columns.append(body)
        for y in range(row_rows):
            parts = [col[y] for col in columns]
            self._write((" " * margin + (" " * GAP_COLS).join(parts)).rstrip() + "\n")

    def _row_graphics(self, tiles, sized, tile_cols, row_rows, margin) -> None:
        # Reserve the rows first so drawing near the bottom scrolls once, then draw
        # each image at its column with the cursor parked, and step past the row.
        # Everything is encoded before the first write, so a failure (or Ctrl-C)
        # while encoding leaves the cursor where it was.
        parts = ["\n" * row_rows + f"{ESC}[{row_rows}A"]
        for index, (tile, (cols, rows, px_w, px_h)) in enumerate(zip(tiles, sized)):
            left = margin + index * (tile_cols + GAP_COLS) + (tile_cols - cols) // 2
            down = row_rows - rows  # bottom-aligned, like the blocks path
            parts.append("\r" + (f"{ESC}[{left}C" if left else ""))
            if down:
                parts.append(f"{ESC}[{down}B")
            seq = self._encode(tile.image, cols, rows, px_w, px_h)
            # Save and restore the cursor: protocols disagree on where it ends up.
            parts.append(f"{ESC}7{self.graphic(seq)}{ESC}8")
            if down:
                parts.append(f"{ESC}[{down}A")
        parts.append(f"\r{ESC}[{row_rows}B")
        self._write("".join(parts))

    def _encode(self, image, cols, rows, px_w, px_h) -> str:
        protocol = self.term.protocol
        if protocol == "sixel":
            if px_w * px_h > SIXEL_MAX_PIXELS:
                scale = math.sqrt(SIXEL_MAX_PIXELS / (px_w * px_h))
                px_w, px_h = max(1, int(px_w * scale)), max(1, int(px_h * scale))
            small = image.resize((px_w, px_h), Image.Resampling.LANCZOS)
            return sixel_sequence(small)
        small = (
            image
            if image.size == (px_w, px_h)
            else image.resize((px_w, px_h), Image.Resampling.LANCZOS)
        )
        if protocol == "kitty":
            box = letterbox(small, cols, rows, self.term.cell)
            # No image id: reusing one would repaint earlier results in scrollback.
            return kitty_sequence(png_bytes(box), cols, rows)
        return iterm_sequence(png_bytes(small), cols, rows)


# ------------------------------------------------------------------ helpers


def describe(image: Image.Image, *parts: str | None) -> str:
    """Caption text: WxH plus whatever else is known, joined with dots."""
    w, h = image.size
    return f" {get_inline_separator_icon()} ".join(
        [f"{w}x{h}", *(p for p in parts if p)]
    )


def short_path(path: str) -> str:
    """Relative to the current folder when inside it, else the file name."""
    try:
        rel = os.path.relpath(path)
    except ValueError:
        return os.path.basename(path)
    return os.path.basename(path) if rel.startswith("..") else rel


def check_pixels(image: Image.Image) -> None:
    width, height = image.size
    if width * height > MAX_IMAGE_PIXELS:
        raise ValueError(f"{width}x{height} is too large to preview")


def _to_8bit(image: Image.Image) -> Image.Image:
    """8-bit modes every encoder can write.

    16-bit and float images (depth maps, masks) are stretched to gray: converting
    them directly to RGB clips everything above 255 to white.
    """
    if image.mode in ("1", "L", "LA", "P", "RGB", "RGBA"):
        return image
    if not image.mode.startswith(("I", "F")):
        return image.convert("RGBA")  # CMYK, YCbCr, LAB, PA ...
    image = image.convert("F")
    low, high = image.getextrema()
    scale = 255 / (high - low) if high > low else 0
    return image.point(lambda v: (v - low) * scale).convert("L")


def load(path: str | os.PathLike) -> Image.Image:
    image = Image.open(path, formats=IMAGE_FORMATS)
    check_pixels(image)
    image.load()
    return _to_8bit(image)


def show_images(
    items: Iterable[tuple[str | os.PathLike | Image.Image, str]],
    term: Terminal | None = None,
    out: IO | None = None,
) -> None:
    """Show (path or image, extra caption) pairs as one grid."""
    term = term or detect()
    renderer = Renderer(term, out)
    tiles = []
    for item, extra in items:
        if isinstance(item, Image.Image):
            image, path = item, None
        else:
            path = str(item)
            if term.protocol == OFF:
                tiles.append(Tile(Image.new("RGB", (1, 1)), extra or path, path))
                continue
            try:
                image = load(item)
            except Exception as exc:  # show the others
                renderer.out.write(clean(f"{path}: not previewed ({exc})") + "\n")
                continue
        name = short_path(path) if path else None
        tiles.append(Tile(image, describe(image, extra, name), path))
    renderer.show(tiles)


# ------------------------------------------------------------- CLI glue


def terminal_for(args) -> Terminal:
    """The terminal to render to, honoring --preview / --no-preview."""
    preview = getattr(args, "preview", None)
    term = detect()
    if preview is False:
        return replace(term, protocol=OFF)
    if preview and term.protocol == OFF:
        # Asked for explicitly: render even when piped, and over
        # FAL_DEV_PREVIEW=off (the flag is the more specific choice).
        env = {k: v for k, v in os.environ.items() if k != ENV_PROTOCOL}
        return detect(env, isatty=True)
    return term
