"""Personal GPU dev boxes on fal, driven by `fal dev` (experimental).

- box.py: deploy, start, stop and connect to a box (laptop side).
- hot.py: hot reload client (laptop side).
- box_app.py, hot_server.py, Dockerfile: what runs on the box.

A box is one trust domain: everything on it runs as root, including the idle
watchdog that holds the team's DEVBOX_ADMIN_KEY, so code you run there (your
model, the hot server, Claude) can read that key.
"""
