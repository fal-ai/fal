"""Glue between `fal dev` results and the terminal previews in term_media.

`hot run` (and `hot sync --watch --run`) saves an image and calls HotPreview.
`app request` returns JSON with media URLs (fal File objects, data URIs) or raw
bytes; show_app_result finds the media, downloads it with a size cap, and shows
it. `fal dev view` shows local files or URLs. term_media draws images and
term_video videos; both load Pillow and are imported only to show something.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import os
import re
import sys
import tempfile
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

from fal.dev.media.common import (
    ENV_PROTOCOL,
    OFF,
    VIDEO_SUFFIXES,
    clean,
)
from fal.dev.state import STATE_DIR

if TYPE_CHECKING:
    from fal.dev.media.term_media import Terminal

ENV_MAX_BYTES = "FAL_DEV_PREVIEW_MAX_MB"
DEFAULT_MAX_BYTES = 64 * 1024 * 1024
# Media items previewed per response; the rest are listed in the JSON anyway.
MAX_MEDIA = 12
MEDIA_DIR = STATE_DIR / "media"
MEDIA_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".gif",
    ".bmp",
    ".tif",
    ".tiff",
    *VIDEO_SUFFIXES,
}
DATA_URI = re.compile(
    r"^data:(?P<type>[\w.+-]+/[\w.+-]+)?(;[^,]*)?;base64,(?P<data>.*)$", re.S
)


@dataclass
class Media:
    source: str  # URL, data URI marker, or path
    content_type: str = ""
    label: str = ""  # JSON path, e.g. images[0]


def _is_media_type(content_type: str) -> bool:
    return content_type.startswith(("image/", "video/"))


def _looks_like_media_url(value: str) -> bool:
    if not value.startswith(("http://", "https://")):
        return False
    path = value.split("?", 1)[0].split("#", 1)[0]
    return Path(path).suffix.lower() in MEDIA_SUFFIXES


def find_media(body: Any, label: str = "") -> Iterator[Media]:
    """Media in a JSON response, in document order.

    Understands fal File objects ({"url", "content_type"}), bare media URLs,
    and base64 data URIs.
    """
    if isinstance(body, dict):
        url = body.get("url")
        content_type = str(body.get("content_type") or "")
        if isinstance(url, str) and url.startswith("data:"):
            match = DATA_URI.match(url)
            content_type = content_type or (match and match["type"]) or ""
            if _is_media_type(content_type):
                yield Media(url, content_type, label or "url")
        elif isinstance(url, str) and (
            _is_media_type(content_type) or _looks_like_media_url(url)
        ):
            yield Media(url, content_type, label or "url")
        for key, value in body.items():
            if key == "url":
                continue
            yield from find_media(value, f"{label}.{key}" if label else str(key))
    elif isinstance(body, list):
        for index, value in enumerate(body):
            yield from find_media(value, f"{label}[{index}]")
    elif isinstance(body, str):
        match = DATA_URI.match(body) if body.startswith("data:") else None
        if match and _is_media_type(match["type"] or ""):
            yield Media(body, match["type"], label)
        elif _looks_like_media_url(body):
            yield Media(body, "", label)


def max_bytes(env=None) -> int:
    env = os.environ if env is None else env
    try:
        return int(float(env[ENV_MAX_BYTES]) * 1024 * 1024)
    except (KeyError, ValueError, OverflowError):
        return DEFAULT_MAX_BYTES


class TooLarge(Exception):
    pass


class _HttpOnlyRedirects(urllib.request.HTTPRedirectHandler):
    """Refuses a redirect to anything but http(s) before connecting to it."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urllib.parse.urlsplit(newurl).scheme not in ("http", "https"):
            raise ValueError("redirected away from http(s)")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_OPENER = urllib.request.build_opener(_HttpOnlyRedirects)


def _suffix(content_type: str, source: str) -> str:
    path_suffix = Path(source.split("?", 1)[0]).suffix.lower()
    if path_suffix in MEDIA_SUFFIXES:
        return path_suffix
    guessed = mimetypes.guess_extension(content_type.split(";")[0].strip() or "")
    guessed = ".jpg" if guessed == ".jpe" else guessed
    # Only media names: the response must not choose names like x.sh or x.html.
    return guessed if guessed in MEDIA_SUFFIXES else ".bin"


def save_bytes(data: bytes, content_type: str, source: str = "") -> Path:
    MEDIA_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    digest = hashlib.sha256(data).hexdigest()[:16]
    path = MEDIA_DIR / f"{digest}{_suffix(content_type, source)}"
    if not path.exists():
        # Atomic: an interrupted write must not leave a truncated file under the
        # name of the complete content.
        fd, tmp = tempfile.mkstemp(dir=MEDIA_DIR, prefix=".", suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            os.replace(tmp, path)
        except BaseException:
            os.unlink(tmp)
            raise
    return path


def fetch(media: Media, limit: int | None = None, timeout: float = 60) -> Path:
    """Download (or decode) media into the local cache, refusing big files."""
    limit = max_bytes() if limit is None else limit
    if media.source.startswith("data:"):
        match = DATA_URI.match(media.source)
        if not match:
            raise ValueError("malformed data URI")
        if len(match["data"]) * 3 // 4 > limit:
            raise TooLarge(f"about {len(match['data']) * 3 // 4} bytes")
        data = base64.b64decode(match["data"])
        if len(data) > limit:
            raise TooLarge(f"{len(data)} bytes")
        return save_bytes(data, media.content_type or match["type"] or "")
    # The URL comes from the app's response: never file://, ftp:// and the like,
    # neither directly nor through a redirect.
    if urllib.parse.urlsplit(media.source).scheme not in ("http", "https"):
        raise ValueError("only http and https URLs are previewed")
    req = urllib.request.Request(media.source, headers={"User-Agent": "fal-dev"})
    with _OPENER.open(req, timeout=timeout) as resp:
        if urllib.parse.urlsplit(resp.geturl()).scheme not in ("http", "https"):
            raise ValueError("redirected away from http(s)")
        declared = int(resp.headers.get("content-length") or 0)
        if declared > limit:
            raise TooLarge(f"{declared} bytes")
        data = resp.read(limit + 1)
        if len(data) > limit:
            raise TooLarge(f"over {limit} bytes")
        if declared and len(data) < declared:
            raise ValueError(f"download cut short ({len(data)} of {declared} bytes)")
        content_type = media.content_type or resp.headers.get("content-type", "")
    return save_bytes(data, content_type, media.source)


def show_paths(
    paths: list[tuple[str, str]], term: Terminal, *, play: bool = False, out=None
) -> None:
    """Show (path, caption extra) pairs: images as one grid, videos one by one."""
    from fal.dev.media import term_media, term_video

    moving = {
        p for p, _ in paths if term_video.is_video(p) or term_video.is_animated(p)
    }
    images = [(p, extra) for p, extra in paths if p not in moving]
    if images:
        term_media.show_images(images, term, out)
    for path, extra in paths:
        if path in moving:
            term_video.show_video(path, extra, term=term, play=play, out=out)


# ------------------------------------------------------------- app request


# Set once per command by configure(): it keeps the `app` hooks to one line each.
_settings = argparse.Namespace(preview=None, play=False)


def configure(args) -> None:
    """Remember the preview flags of this command for show_app_result."""
    for key in ("preview", "play"):
        setattr(_settings, key, getattr(args, key, getattr(_settings, key)))


def show_app_result(result: dict, *, out=None) -> None:
    """Preview media in an `app request` result. Never raises."""
    try:
        _show_app_result(result, out or sys.stdout)
    except Exception as exc:  # a preview must never fail the request
        print(f"(preview failed: {clean(str(exc))})", file=sys.stderr)


def _app_terminal() -> Terminal:
    # Pillow (term_media) loads only once a response holds media.
    from fal.dev.media import term_media

    return term_media.terminal_for(_settings)


def _show_app_result(result: dict, out) -> None:
    if "body_base64" in result:
        content_type = result.get("content_type", "")
        if not _is_media_type(content_type):
            return
        term = _app_terminal()
        if term.protocol == OFF:
            return  # the size line was printed already
        path = save_bytes(base64.b64decode(result["body_base64"]), content_type)
        show_paths([(str(path), "")], term, play=_settings.play, out=out)
        return
    try:
        body = json.loads(result.get("body") or "null")
    except ValueError:
        return
    found = list(find_media(body))
    if not found:
        return
    term = _app_terminal()
    if term.protocol == OFF:
        return  # the URLs are in the JSON printed above
    if len(found) > MAX_MEDIA:
        out.write(f"(previewing the first {MAX_MEDIA} of {len(found)} media)\n")
    paths = []
    for media in found[:MAX_MEDIA]:
        try:
            paths.append((str(fetch(media)), media.label))
        except TooLarge as exc:
            out.write(clean(f"{media.label}: not previewed, too large ({exc})") + "\n")
        except Exception as exc:
            out.write(clean(f"{media.label}: could not download ({exc})") + "\n")
    show_paths(paths, term, play=_settings.play, out=out)


# ---------------------------------------------------------------- hot run


class HotPreview:
    """Called by HotBox.request with each saved image."""

    def __init__(self, args) -> None:
        self.args = args
        # A copy of the last image, not its path: -o reuses the file name.
        self.previous: tuple[Any, str] | None = None

    def __call__(self, path: Path, headers: dict) -> None:
        try:
            self._show(path, headers)
        except Exception as exc:
            print(f"(preview failed: {clean(str(exc))})", file=sys.stderr)

    def _show(self, path: Path, headers: dict) -> None:
        from fal.console.icons import get_inline_separator_icon
        from fal.dev.media import term_media

        term = term_media.terminal_for(self.args)
        if term.protocol == OFF:
            return  # HotBox printed the path
        sep = f" {get_inline_separator_icon()} "
        parts = []
        if headers.get("x-hot-infer-ms"):
            parts.append(f"infer {headers['x-hot-infer-ms']} ms")
        if headers.get("x-hot-code-version"):
            parts.append(f"code v{headers['x-hot-code-version']}")
        if not getattr(self.args, "compare", False):
            show_paths([(str(path), sep.join(parts))], term, play=self.args.play)
            return
        # Advance first, so one unreadable result cannot stall the comparison.
        previous, self.previous = self.previous, None
        image = term_media.load(path)  # fully read: survives the file being rewritten
        current = (image, sep.join([*parts, term_media.short_path(str(path))]))
        self.previous = current
        term_media.show_images([previous, current] if previous else [current], term)


# ------------------------------------------------------------ fal dev view


def view(args) -> int:
    """`fal dev view`: show local files or URLs inline."""
    from fal.dev.media import term_media

    term = term_media.terminal_for(args)
    paths, failed = [], 0
    for item in args.files:
        try:
            if item.startswith(("http://", "https://", "data:")):
                path = str(fetch(Media(item)))
            elif Path(item).is_file():
                path = item
            else:
                raise OSError("no such file")
        except Exception as exc:
            print(clean(f"{item[:200]}: {exc}"), file=sys.stderr)
            failed += 1
            continue
        paths.append((path, ""))
    try:
        show_paths(paths, term, play=args.play)
    except Exception as exc:
        print(f"preview failed: {clean(str(exc))}", file=sys.stderr)
        return 1
    opted_out = getattr(args, "preview", None) is False or os.environ.get(ENV_PROTOCOL)
    if paths and term.protocol == OFF and not opted_out:
        print(
            "no preview: this is not a terminal that can show images. Set "
            f"{ENV_PROTOCOL}=blocks for a text preview, or copy the file to your "
            "machine and run `fal dev view` there.",
            file=sys.stderr,
        )
    return 1 if failed else 0
