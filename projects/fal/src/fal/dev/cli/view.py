"""`fal dev view`: show image and video files or URLs inline in the terminal."""

from __future__ import annotations

from fal.dev.media import preview


def cmd_view(args):
    return preview.view(args)
