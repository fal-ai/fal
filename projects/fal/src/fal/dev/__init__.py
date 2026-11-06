"""Personal GPU dev boxes on fal, driven by `fal dev` (experimental).

Layout:

- cli/: the `fal dev` command line. parser.py defines every command and
  imports nothing heavy; a command's handler is imported only when it runs.
- box.py, hot.py, state.py: the laptop side of a box and of hot reload.
- app/: the laptop side of `fal dev app` (a box built from an existing App).
- dash/: the `fal dev dash` terminal dashboard.
- remote/: what runs on the box (its fal App, the hot server, the dashboard
  probe, the `fal dev app` holder and child process). It never imports the
  laptop side.

The package is self-contained and only imported when `fal dev` runs, so it can
move to a separate plugin package without code changes. Optional heavy
dependencies belong behind the `fal[devbox]` extra.

A box is one trust domain: everything on it runs as root, including the idle
watchdog that holds the team's DEVBOX_ADMIN_KEY, so code you run there (your
model, the hot server, Claude) can read that key.
"""
