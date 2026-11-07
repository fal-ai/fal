import io
import subprocess
import sys

from rich.console import Console

from fal.console.icons import DUST_SQUARES
from fal.dev import loading
from fal.dev.loading import Loading


class _Tty(io.StringIO):
    encoding = "utf-8"

    def isatty(self):
        return True


def _text(renderable, width=80) -> str:
    console = Console(record=True, width=width, force_terminal=False, color_system=None)
    console.print(renderable)
    return console.export_text()


def test_import_is_free():
    # Only what `fal` itself loads: no rich.live or rich.panel for a wait.
    code = (
        "import sys, fal; before = set(sys.modules); import fal.dev.loading; "
        "new = set(sys.modules) - before - {'fal.dev', 'fal.dev.loading'}; "
        "assert not new, sorted(new)"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_frames_are_deterministic():
    a = _text(loading.dust_panel_body(12.5, "model load", expected=35))
    b = _text(loading.dust_panel_body(12.5, "model load", expected=35))
    assert a == b
    assert a != _text(loading.dust_panel_body(13.5, "model load", expected=35))


def test_dust_is_squares_on_a_grid_and_fits_the_width():
    for width in (40, 41):
        rows = loading.dust_rows(3.0, width, 4, 0.5)
        assert len(rows) == 4
        for row in rows:
            assert len(row.plain) == width
            assert set(row.plain) <= set(DUST_SQUARES + " ")
            # One square per two columns, the second always a gap.
            assert not row.plain[1::2].strip()
    assert any(ch in DUST_SQUARES for row in rows for ch in row.plain)


def test_dust_has_big_squares_in_its_clusters():
    frames = [loading.dust_rows(t / 4, 80, 4, 0.8) for t in range(40)]
    assert any(DUST_SQUARES[2] in row.plain for rows in frames for row in rows)


def test_a_jumping_row_moves_its_squares_sideways_and_keeps_its_width():
    # Bursts start on multiples of 3.1 s; find one that fires.
    t = next(
        3.1 * i + 0.05
        for i in range(50)
        if loading._burst(3.1 * i + 0.05, 3.1, 0.2, 0.7, 31)[0]
    )
    width = 41
    jumped = [row.plain for row in loading.dust_rows(t, width, 4, 0.8)]
    _, g = loading._burst(t, 3.1, 0.2, 0.7, 31)
    y = int(loading._rand(g, 32) * 4)
    still = [
        loading._square(x, y, t, loading._clusters(t, 20, 4, 0.8), DUST_SQUARES)
        for x in range(20)
    ]
    plain = "".join((cell[0] if cell else " ") + " " for cell in still)
    assert len(jumped[y]) == width
    doubled = plain + plain
    assert jumped[y][:40] in doubled


def test_more_clusters_burn_as_progress_grows():
    def squares(progress):
        return sum(
            ch != " "
            for t in range(40)
            for row in loading.dust_rows(t / 4, 120, 4, progress)
            for ch in row.plain
        )

    assert squares(0.9) > squares(0.1) * 1.5


def test_panel_body_shows_status_elapsed_and_typical_time():
    text = _text(
        loading.dust_panel_body(64, "waiting for a GPU runner: requested", expected=100)
    )
    assert "waiting for a GPU runner: requested  1m04s  typically ~1m40s" in text


def test_hot_restart_body_has_the_headline_and_model_load():
    headline, expected = loading.STATES["hot-restart"]
    text = _text(
        loading.dust_panel_body(9, "model load", expected=expected, headline=headline)
    )
    lines = text.splitlines()
    assert lines[0].strip() == "restarting the hot server: loading weights"
    assert "model load  9s  typically ~35s" in text


def test_status_text_is_never_scrambled():
    for t in (0.0, 1.3, 3.15, 6.2, 9.31):
        text = _text(loading.dust_panel_body(t, "model load", headline="keep me"))
        assert "keep me" in text and "model load" in text


def test_narrow_terminals_get_one_line():
    wait = Loading("waiting for a GPU runner", expected=100, animate=False)
    text = _text(wait.renderable(50, 40), width=50)
    assert len(text.strip().splitlines()) == 1
    assert "waiting for a GPU runner" in text


def test_wide_terminals_get_the_panel_with_title_and_subtitle():
    wait = Loading(
        "waiting", title="GPU", subtitle="runner pending", hint="hint", animate=False
    )
    text = _text(wait.renderable(100, 40), width=100)
    assert " GPU " in text.splitlines()[0]
    assert "runner pending" in text and "hint" in text


def test_animation_switches():
    env = {"TERM": "xterm-256color"}
    assert loading.animation_enabled(_Tty(), env)
    assert not loading.animation_enabled(io.StringIO(), env)
    assert not loading.animation_enabled(_Tty(), {**env, "NO_COLOR": "1"})
    assert not loading.animation_enabled(_Tty(), {"TERM": "dumb"})
    assert not loading.animation_enabled(_Tty(), {**env, "FAL_DEV_NO_ANIMATION": "1"})
    latin = _Tty()
    latin.encoding = "latin-1"
    assert not loading.animation_enabled(latin, env)


def test_plain_mode_prints_status_changes():
    out = io.StringIO()
    wait = Loading("waiting", stream=out, animate=False)
    wait.update("not shown before start")
    with wait:
        wait.update("waiting: provisioning")
        wait.say("  a log line")
    assert out.getvalue().splitlines() == [
        "not shown before start ...",
        "waiting: provisioning ...",
        "  a log line",
    ]


def test_plain_and_narrow_modes_keep_the_headline():
    headline = "restarting the hot server: loading weights"
    out = io.StringIO()
    with Loading("model load", headline=headline, stream=out, animate=False) as wait:
        assert headline in _text(wait.renderable(59, 40), width=59)
    assert out.getvalue() == f"{headline} ...\n"


def test_narrow_line_drops_the_hint_instead_of_cutting_it():
    text = _text(loading.one_line(0, 50, "waiting for a GPU runner: pending", 100), 50)
    assert "typically" not in text and "\u2026" not in text


def test_start_is_idempotent_and_close_is_safe():
    out = io.StringIO()
    wait = Loading("waiting", stream=out, animate=False)
    wait.start()
    wait.start()
    wait.close()
    wait.close()
    assert out.getvalue() == "waiting ...\n"


def test_animated_wait_clears_itself():
    out = _Tty()
    with Loading("waiting", stream=out, animate=True) as wait:
        wait.say("above the panel")
    assert "above the panel" in out.getvalue()


def test_estimate_progress_never_reaches_one():
    assert loading.estimate_progress(0, 35) == 0
    assert 0.9 < loading.estimate_progress(1e6, 35) < 1


def test_panel_drops_the_hint_rather_than_cut_it():
    body = loading.dust_panel_body(
        42, "waiting for a GPU runner: pending", expected=100
    )
    assert "typically" not in _text(body, width=48)
    assert "typically ~1m40s" in _text(body, width=80)


def test_without_animation_the_body_is_text_only():
    body = loading.dust_panel_body(
        9, "model load", expected=35, headline="restarting", animate=False
    )
    text = _text(body)
    assert text.splitlines() == ["restarting", "model load  9s  typically ~35s"]
    assert all(ord(ch) < 0x2800 and ch not in DUST_SQUARES for ch in text)


def test_plain_start_line_says_how_long_it_usually_takes():
    out = io.StringIO()
    with Loading("waiting for a GPU runner", expected=100, stream=out, animate=False):
        pass
    assert out.getvalue() == "waiting for a GPU runner ... (typically ~1m40s)\n"
