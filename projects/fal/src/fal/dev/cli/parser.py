"""Every `fal dev` argument definition.

`fal` builds all parsers on each run, so this module imports nothing heavy. A
command's handler (`cmd_<name>` in box.py, hot.py, app.py, dash.py or view.py)
is only imported once that command runs.
"""

from __future__ import annotations

import argparse
import functools
import importlib
from typing import Callable

from fal.cli.parser import FalClientParser, RefAction, add_env_argument
from fal.dev import config
from fal.dev.errors import DevError
from fal.dev.remote.protocol import MAX_PROFILE_WARMUPS, MAX_REPEAT, MAX_WARMUPS

EXPERIMENTAL = "(experimental) "

DEFAULT_KEEP_ALIVE = 1800
PROXY_PORT = 8899

TEAM_HELP = (
    "The team the box runs as. Changes the team remembered for the box; without "
    "it, the remembered team, else your personal account."
)
SSH_CONFIG_TEAM_HELP = (
    "Write --team into the entry's ProxyCommand. Nothing is remembered until "
    "ssh-proxy runs."
)
YES_HELP = (
    "Deploy into a team that has production apps without asking (needed when "
    "there is no terminal to ask on)."
)

# Each dashboard refresh is a runner exec; faster only loads the box and the API.
MIN_INTERVAL = 1.0

APP_DESCRIPTION = """\
A GPU dev box for an existing fal App, with hot reload.

`start` deploys a private box that inherits the app's machine, requirements,
image and secrets, uploads the app's source tree, and runs ``setup()`` once in a
resident process. `sync` uploads changed files: endpoint edits reload in place
with the weights still loaded; edits to ``setup()`` (or what it uses) restart
the process; requirement or image changes need `start` again.

The app process runs with ``FAL_DEV_SESSION`` set to the box name. Anything
the app shares with its deployed runners, such as a compile cache on /data,
should be keyed by it so dev runs stay apart. An app that already reads
another variable for this can have it set too, with `start --session-env NAME`.
"""

START_DESCRIPTION = (
    "Deploy a dev box from an app and run its setup() once. Synced source files "
    "are also stored on your team's shared /data volume (under "
    "/data/.fal-dev/objects, deleted after 30 days unused), so a new box copies "
    "the files it already has instead of uploading them. Everyone on the team "
    "can read that store, so do not sync a tree that holds anything teammates "
    "should not see. Key, credential and .env files are never synced."
)

HOT_PROJECT = """\
A project is a folder with:

  load.py          def load() -> state
                   Runs once per server process; load weights here. Whatever it
                   returns is passed to infer(). Saving it restarts the server
                   and reloads the weights.
  infer.py         def infer(state, **params)
                   Runs per request with the keys of --prompt / -p key=value
                   (JSON values). Reloaded on every save, weights stay loaded.
                   Return a PIL image (sent as a PNG and saved), bytes, or any
                   JSON-serializable value. An error comes back with its
                   traceback and the server keeps running.
  requirements.txt (optional) extra packages, see below.

Preinstalled on the stock box image (the Python you deployed from, CUDA 12.8): torch 2.8.0
and torchvision 0.23.0 (cu128), transformers 4.56.1, diffusers 0.35.1,
accelerate 1.10.1, safetensors 0.6.2, huggingface_hub 0.34.4, and uv. A custom
image (`fal dev deploy --dockerfile`) has what its Dockerfile installs.

requirements.txt is installed with uv into <project>/.venv, which sees the
preinstalled packages and only adds what is missing or newer. The torch stack
(torch, torchvision, torchaudio, triton, nvidia-*) is pinned to the base image's
versions, so listing diffusers never swaps torch and breaks torchvision; a
requirement that needs a different torch fails with the resolver's message.
Anything already preinstalled at a compatible version is not downloaded again.
Saving requirements.txt reinstalls and restarts the server; the .venv is rebuilt
whenever the file changes, and removed when the file is emptied or deleted.
"""

EXEC_DESCRIPTION = """\
Run one command on the box: fal dev exec -- nvidia-smi

The words after -- run as one program, not through a shell. A single quoted string
is taken as the program's name, so pipes, && and ; need a shell:

  fal dev exec -- bash -c 'df -h /work && nvidia-smi'

Output is passed through as it arrives, not buffered until the end (a program
that buffers its own stdout, such as python, still prints in chunks: use python
-u). exec returns when the command does, so start a background job with its output
redirected and read the log later (for a job that must outlive the call, run it in
tmux instead):

  fal dev exec -- bash -c 'nohup python train.py > /work/train.log 2>&1 &'
  fal dev exec -- tail -n 20 /work/train.log
"""

HOT_DESCRIPTION = (
    "Keep a model loaded on the box and reload your inference code on save.\n\n"
    + HOT_PROJECT
    + "\nExample: fal dev hot start ./my-model, then "
    "fal dev hot sync --watch --run --prompt 'a red fox'"
)

HOT_START_DESCRIPTION = (
    "Push the project folder, start the hot server and wait for load() to finish. "
    "If load() fails, the log ends with the innermost exception.\n\n" + HOT_PROJECT
)

DASH_DESCRIPTION = (
    "Live dashboard for a running dev box. Shows GPU use, idle shutdown, "
    "connections, the /work snapshot and recent logs. Keys: s shell, h hold, "
    "d down, r hot restart, q quit. Watching the box does not keep it awake. "
    "Idle shutdown counts terminal input, a connected SSH client or editor, GPU "
    "use of 10% or more, Claude sessions, /hot requests and holds as activity. "
    "The cost line comes from the platform's usage API and needs a key with "
    "billing access; it is hidden otherwise."
)


def box_name(value: str) -> str:
    if not config.NAME_PATTERN.fullmatch(value):
        raise argparse.ArgumentTypeError(
            f"{value!r} is not a valid box name: use lowercase letters, digits and "
            f"dashes (set one with --name or ${config.ENV_DEFAULT_NAME})"
        )
    return value


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be 1 or more")
    return number


NETWORK_NOTE = (
    "Corporate networks: set HTTPS_PROXY as an http:// URL (and NO_PROXY) and fal "
    "dev uses it for API, REST and box calls; either letter case works. For a "
    "custom CA set SSL_CERT_FILE (REST, auth) and "
    "GRPC_DEFAULT_SSL_ROOTS_FILE_PATH (deploys). "
    "Allow these hosts: api.alpha.fal.ai, rest.alpha.fal.ai, fal.run, and, for "
    "the box image build, download.pytorch.org and claude.ai. Raise "
    "FAL_DEV_CONNECT_TIMEOUT (seconds, default 3) if a slow proxy times out."
)


def bounded(low: int, high: int) -> Callable[[str], int]:
    def parse(value: str) -> int:
        try:
            number = int(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f"invalid integer {value!r}") from None
        if not low <= number <= high:
            raise argparse.ArgumentTypeError(f"must be between {low} and {high}")
        return number

    return parse


def interval(value: str) -> float:
    seconds = float(value)
    if seconds < MIN_INTERVAL:
        raise argparse.ArgumentTypeError(f"must be at least {MIN_INTERVAL:g} seconds")
    return seconds


def _command(module: str, handler: str) -> Callable[[argparse.Namespace], object]:
    """A command that imports its handler `cmd_<handler>` when it runs."""

    def run(args: argparse.Namespace) -> object:
        implementation = importlib.import_module(f"fal.dev.cli.{module}")
        try:
            return getattr(implementation, f"cmd_{handler}")(args)
        except DevError as exc:
            _show(exc)
            return 1
        except Exception as exc:
            unreachable = _unreachable(exc)
            if unreachable is None:
                raise
            unreachable.__cause__ = exc
            _show(unreachable)
            return 1

    run.__name__ = f"{module}.{handler}"
    return run


def _add_take_over(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--take-over",
        action="store_true",
        help=(
            "Deploy over a box that another owner created. Without it a deploy "
            "refuses; owners are recorded on each deploy."
        ),
    )


def _show(exc: DevError) -> None:
    from rich.console import Console

    from fal.dev.cli import error_panel

    error_panel.show(exc, Console(stderr=True))


def _unreachable(exc: Exception) -> DevError | None:
    """A connect failure that no `fal dev` call wrapped, with host and proxy."""
    import httpx

    from fal._proxy import https_proxy_for, unreachable_message

    if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout, httpx.ProxyError)):
        try:
            host = exc.request.url.host
        except RuntimeError:
            host = "fal"
    else:
        return None
    proxied = https_proxy_for(host) is not None
    return DevError(
        unreachable_message(host, exc),
        title="the proxy is not reachable" if proxied else "the network is unreachable",
    )


def _add_yes(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--yes", action="store_true", help=YES_HELP)


def add(
    subparsers,
    module: str,
    parents: list,
    cmd: str,
    help_text: str,
    *,
    handler: str | None = None,
    extra_parents: tuple = (),
    team_help: str | None = None,
    **kwargs,
) -> argparse.ArgumentParser:
    """Add command `cmd`, run by `cmd_<handler>` (default: the command's name)."""
    kwargs.setdefault("description", help_text)
    if team_help is None:
        # Commands after `start` use the team their box was started with.
        after_start = module == "app" and cmd != "start"
        team_help = argparse.SUPPRESS if after_start else TEAM_HELP
    parser = subparsers.add_parser(
        cmd,
        help=help_text,
        parents=[*parents, *extra_parents],
        team_help=team_help,
        **kwargs,
    )
    parser.set_defaults(func=_command(module, handler or cmd.replace("-", "_")))
    return parser


def _group(subparsers, cmd: str, help_text: str, parents: list, **kwargs):
    """Add a command that only holds subcommands; returns their subparsers."""
    kwargs.setdefault("description", help_text)
    parser = subparsers.add_parser(
        cmd, help=help_text, parents=parents, team_help=TEAM_HELP, **kwargs
    )
    return parser.add_subparsers(
        title="Commands",
        metavar="command",
        dest=f"{cmd}_cmd",
        required=True,
        parser_class=FalClientParser,
    )


def _name_parser(help_text: str, type_=None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--name", type=type_, help=help_text)
    return parser


def _preview_parser(*, compare: bool = False) -> argparse.ArgumentParser:
    """Flags for showing image and video results inline (fal.dev.media.preview)."""
    parser = argparse.ArgumentParser(add_help=False)
    group = parser.add_argument_group("Preview")
    group.add_argument(
        "--preview",
        action="store_true",
        default=None,
        help=(
            "Show image and video results inline in the terminal (default: on "
            "when it is a terminal; $FAL_DEV_PREVIEW=kitty|iterm|sixel|blocks|off "
            "forces a protocol)."
        ),
    )
    group.add_argument(
        "--no-preview",
        dest="preview",
        action="store_false",
        help="Only print where results were saved.",
    )
    group.add_argument(
        "--play",
        action="store_true",
        help="Loop a small preview of video results (kitty, iTerm2, WezTerm).",
    )
    if compare:
        group.add_argument(
            "--compare",
            action="store_true",
            help="Show each new image next to the previous one.",
        )
    return parser


def _add_box_commands(subparsers, parents) -> None:
    box = _name_parser(
        "Box name; the app is devbox-<name>. Remembered for the current folder "
        f"(default: the folder's last box, else ${config.ENV_DEFAULT_NAME} or "
        "your OS username).",
        box_name,
    )
    add_box = functools.partial(add, subparsers, "box", [*parents, box])
    no_save_help = (
        f"Skip saving {config.LOCAL_WORK} first; changes since the last save are lost."
    )

    p = add_box(
        "deploy",
        "Build the image and (re)deploy your box as a private app. Ends open shells.",
    )
    p.add_argument(
        "--machine-type",
        help=(
            "Machine type, e.g. GPU-H100 "
            f"(default: the current one, else {config.DEFAULT_GPU}). "
            f"Limited by ${config.ENV_ALLOWED_MACHINES} (comma list); "
            "a client-side guardrail, not enforcement."
        ),
    )
    p.add_argument(
        "--num-gpus",
        type=positive_int,
        default=config.DEFAULT_NUM_GPUS,
        help=(
            "GPUs per box (default: %(default)s). Capped by "
            f"${config.ENV_MAX_GPUS}; a client-side guardrail, not enforcement."
        ),
    )
    p.add_argument(
        "--idle-minutes",
        type=positive_int,
        default=config.DEFAULT_IDLE_MINUTES,
        help=(
            "Minutes without input or GPU use before the box stops itself "
            "(default: %(default)s). A box whose GPU cannot be queried, such as a "
            "CPU machine type, never stops itself: stop it with `fal dev down`."
        ),
    )
    p.add_argument("--no-save", action="store_true", help=no_save_help)
    p.add_argument(
        "--claude",
        action="store_true",
        help=(
            "Run `claude remote-control` on the box, to use it from the Claude app "
            "(not remembered)."
        ),
    )
    p.add_argument(
        "--starship",
        action="store_true",
        help=(
            "Use your local starship prompt on the box: copies only your "
            "starship.toml ($STARSHIP_CONFIG or ~/.config/starship.toml, stored in "
            "the box's settings: keep secrets out of it) and "
            "installs starship there (not remembered)."
        ),
    )
    p.add_argument(
        "--dockerfile",
        help="Your own image, based on the default (see fal/dev/remote/Dockerfile).",
    )
    _add_yes(p)
    _add_take_over(p)
    p.add_argument(
        "--team-data",
        action="store_true",
        help=(
            "Keep the box on the team-shared /data (shared model weights, other "
            "apps' files) instead of its own volume. By default /data is the box's "
            "volume, devbox-<name>, and the team /data is not mounted, so a box "
            "whose files are on the team /data needs this flag to keep them. Not "
            "remembered: pass it on every deploy."
        ),
    )

    p = add_box(
        "up", "Start the box (deploying it the first time) and wait until ready."
    )
    p.add_argument(
        "--machine-type",
        help=(
            "Switch the machine type of a stopped box; it stays until changed "
            f"again. Limited by ${config.ENV_ALLOWED_MACHINES}."
        ),
    )
    p.add_argument(
        "--dash",
        action="store_true",
        help="Open the live dashboard (fal dev dash) once the box is ready.",
    )
    p.add_argument(
        "--starship",
        action="store_true",
        help=(
            "Use your local starship prompt on the box: copies only your "
            "starship.toml ($STARSHIP_CONFIG or ~/.config/starship.toml, stored in "
            "the box's settings: keep secrets out of it) and "
            "installs starship there (applies when the box is first created; "
            "later use `fal dev deploy --starship`)."
        ),
    )
    _add_yes(p)
    _add_take_over(p)
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Show the whole deploy log after the run, not its last lines on failure.",
    )
    p.add_argument(
        "--timeout",
        type=positive_int,
        default=1800,
        help="Seconds to wait (default: %(default)s).",
    )

    p = add_box("down", f"Save {config.LOCAL_WORK}, then stop the box. /data is kept.")
    p.add_argument("--no-save", action="store_true", help=no_save_help)
    p.add_argument(
        "--yes",
        action="store_true",
        help=(
            "Stop without asking when the GPU is busy or a job is running "
            "(needed when there is no terminal to ask on)."
        ),
    )
    p = add_box("status", "Show whether the box is running.")
    p.add_argument(
        "--json",
        action="store_true",
        help="Print the same snapshot as `fal dev dash --json` and exit.",
    )
    add(
        subparsers,
        "box",
        parents,
        "list",
        "List the dev boxes of your account or team.",
    ).add_argument(
        "--json", action="store_true", help="Print a JSON list, one object per box."
    )
    p = add_box(
        "destroy",
        "Delete the box's app. Its /data volume is kept: there is no delete API yet.",
    )
    p.add_argument(
        "--yes",
        action="store_true",
        help="Skip typing the box name to confirm (needed without a terminal).",
    )
    add_box(
        "shell",
        "Open an interactive shell on the box.",
        description="Open an interactive shell on the box. Your own shell setup "
        f"goes in ~/.devboxrc on the box (kept at {config.WORKSPACE_ROOT}/<name>/"
        ".devboxrc, so it survives restarts). ssh sessions get HISTFILE, DEVBOX_* "
        "and the uv warning too.",
    )

    p = add_box(
        "exec",
        "Run one command on the box: fal dev exec -- nvidia-smi",
        description=EXEC_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "command", nargs=argparse.REMAINDER, help="The command to run, after --."
    )

    p = add_box("logs", "Follow the box's logs.")
    p.add_argument(
        "--all",
        action="store_true",
        help="Show the full log of the latest runner, even if it stopped.",
    )

    p = add_box(
        "hold",
        f"Suspend idle shutdown for N hours (at most {config.MAX_HOLD_HOURS}); "
        "--off (or 0) clears the hold.",
    )
    p.add_argument(
        "hours",
        type=float,
        nargs="?",
        help="How long to keep the box up regardless of activity; 0 clears the hold.",
    )
    p.add_argument(
        "--off",
        action="store_true",
        help="Clear the hold so idle shutdown applies again. The box keeps running.",
    )

    add_box(
        "ssh-config",
        "Print an ~/.ssh/config entry for Cursor / VS Code Remote-SSH.",
        team_help=SSH_CONFIG_TEAM_HELP,
    )
    p = add_box(
        "cp",
        "Copy files between this laptop and the box; ':' marks the box side. "
        "Starts a stopped box.",
        epilog=(
            "examples:\n"
            "  fal dev cp :/work/out.mp4 .      download a file\n"
            "  fal dev cp -r ./data :/data/x    upload a folder\n"
            "Relative box paths start in the box's home folder. Uses scp over the "
            "ssh-proxy with your default keys in ~/.ssh (your ~/.ssh/config is "
            "not used); ~/.ssh/known_hosts is not touched."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("src", help="Source: a local path, or :path on the box.")
    p.add_argument("dst", help="Destination: a local path, or :path on the box.")
    p.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Copy a folder and what is in it.",
    )
    p = add_box(
        "ssh-proxy",
        "SSH ProxyCommand used by the ssh-config entry; starts a stopped box.",
    )
    p.add_argument(
        "--timeout",
        type=positive_int,
        default=900,
        help="Seconds to wait for a stopped box (default: %(default)s).",
    )

    hot = _group(
        subparsers,
        "hot",
        "Hot reload: keep a model loaded on the box and reload your inference code "
        "on save.",
        parents,
        description=HOT_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_hot = functools.partial(add, hot, "hot", [*parents, box])
    project = argparse.ArgumentParser(add_help=False)
    project.add_argument(
        "local",
        nargs="?",
        help=(
            "A local project folder to push to the box first "
            "(default: the one last used from this folder)."
        ),
    )
    project.add_argument(
        "--dir",
        help=(
            "Project folder on the box "
            f"(default: {config.LOCAL_WORK}/<local folder name>)."
        ),
    )
    request = argparse.ArgumentParser(add_help=False)
    request.add_argument("--prompt", help="Set the prompt parameter.")
    request.add_argument(
        "-p",
        "--param",
        action="append",
        help="A parameter for infer(), key=value (JSON values).",
    )
    request.add_argument(
        "-o",
        "--output",
        help="Where to save an image result (default: hot-out/<time>.png).",
    )
    hot_preview = _preview_parser(compare=True)
    add_hot(
        "start",
        "Start the hot server and wait for the model to load.",
        extra_parents=(project,),
        description=HOT_START_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p = add_hot(
        "sync",
        "Push a project folder to the box; reloads the server.",
        extra_parents=(project, request, hot_preview),
    )
    p.add_argument("--watch", action="store_true", help="Keep pushing as you save.")
    p.add_argument(
        "--run",
        action="store_true",
        help="With --watch: send a request after each push.",
    )
    add_hot(
        "run",
        "Send one request to the hot server.",
        extra_parents=(request, hot_preview),
    )
    add_hot("status", "Show the hot server's state and recent log.")
    add_hot("logs", "Show the hot server's log.")
    add_hot("restart", "Restart the hot server; the weights reload.")
    add_hot("stop", "Stop the hot server.")


def _add_app_commands(subparsers, parents) -> None:
    app = _group(
        subparsers,
        "app",
        "A dev box for an existing fal App, with hot reload.",
        parents,
        description=APP_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    name = _name_parser(
        "Box name, always prefixed dev- (default: the only box; for start, "
        "<app>-<your username>)."
    )
    add_app = functools.partial(add, app, "app", [*parents, name])

    p = add_app(
        "start",
        "Deploy a dev box from an app and run its setup() once.",
        description=START_DESCRIPTION,
    )
    p.add_argument(
        "app_ref",
        action=RefAction,
        help="path/to/app.py, path/to/app.py::App, or the name of an app in "
        "pyproject.toml's tool.fal.apps table (as with `fal deploy`).",
    )
    p.add_argument("--machine-type", help="Override the app's machine type.")
    _add_yes(p)
    p.add_argument(
        "--root",
        help="Source tree to sync and import from (defaults to the folder holding "
        "the app's top-level package, widened to its app_files folder).",
    )
    add_env_argument(p)
    p.add_argument(
        "--session-env",
        action="append",
        metavar="NAME",
        help="Also set environment variable NAME to the box name in the app "
        "process (repeatable). FAL_DEV_SESSION is always set.",
    )
    p.add_argument(
        "--replace",
        action="store_true",
        help="Take over an existing app with the same dev- name.",
    )
    p.add_argument(
        "--keep-alive",
        type=int,
        default=DEFAULT_KEEP_ALIVE,
        help="Seconds without requests before the box shuts down "
        f"(default {DEFAULT_KEEP_ALIVE}).",
    )

    app_preview = _preview_parser()
    p = add_app(
        "sync",
        "Upload changed files; endpoints reload in place.",
        extra_parents=(app_preview,),
    )
    p.add_argument("--watch", action="store_true", help="Keep syncing on save.")
    p.add_argument("--full", action="store_true", help="Upload every file.")
    p.add_argument(
        "--request",
        metavar="PATH",
        help="Send a request to PATH after each sync (with --data).",
    )
    p.add_argument("--data", help="JSON body, or @file.json.")

    p = add_app(
        "request",
        "Send a request to the app on the box.",
        extra_parents=(app_preview,),
    )
    p.add_argument("path", help="Endpoint path, e.g. /")
    p.add_argument("--data", help="JSON body, or @file.json.")
    p.add_argument("--output-file", help="Write the response body here.")
    p.add_argument(
        "--warmups",
        type=bounded(0, MAX_WARMUPS),
        default=0,
        help="Requests to send first and leave out of the timings.",
    )
    p.add_argument(
        "--repeat",
        type=bounded(1, MAX_REPEAT),
        default=1,
        help="Time this many requests (run back to back on the box) and report "
        "median, p90, min and max.",
    )
    p.add_argument(
        "--save",
        metavar="NAME",
        help="Save the timings as a named benchmark, for `benchmark compare`.",
    )

    p = add_app(
        "dash",
        "Live dashboard for the box: GPU, keep-alive, app and engine.",
        description="Live dashboard for a fal dev app box: the GPU, when the "
        "box scales to zero, the app process and its last sync, the inference "
        "engine of an LLM app, and the runner's logs. Watching it is not a "
        "request, so it never keeps the box up.",
    )
    p.add_argument(
        "--interval",
        type=interval,
        default=3.0,
        help="Seconds between refreshes (default: %(default)s).",
    )
    p.add_argument("--once", action="store_true", help="Print one frame and exit.")
    p.add_argument(
        "--wait",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="With no box yet, show one waiting frame (refreshed in place) and "
        "attach when a box appears, also after the box is stopped. Default: on "
        "when stdout is a terminal, so it can be started before the box. "
        "Ignored with --once, which prints one frame (the waiting text if there "
        "is no box) and exits.",
    )

    p = add_app(
        "split",
        "Open the dash and the sync watcher in panes next to this one.",
        description="Split the current terminal: the live dash on the right and "
        "`sync --watch` under it, both waiting for a box if none exists yet, so "
        "an agent in this pane can start one. Uses tmux when $TMUX is set, "
        "iTerm via AppleScript in iTerm, otherwise prints the two commands.",
    )

    p = add_app(
        "proxy",
        "Serve the box's app on a local port, for any HTTP client.",
        description="Serve the app on the box at http://BIND:PORT, so a local "
        "client (an OpenAI SDK, a coding agent, curl) can call it like a local "
        "server: every request and streamed response goes through the box, "
        "signed with your fal login. The client needs no fal key. Every "
        "request counts as activity, so the box stays up while it is used.",
    )
    p.add_argument("--port", type=int, default=PROXY_PORT, help="Local port.")
    p.add_argument(
        "--bind",
        default="127.0.0.1",
        help="Local address to listen on (default 127.0.0.1; anyone who can "
        "reach it can call the box as you).",
    )

    benchmark = _group(
        app,
        "benchmark",
        "Compare saved benchmark runs (`request --repeat N --save NAME`).",
        parents,
    )
    add_benchmark = functools.partial(add, benchmark, "app", [*parents, name])
    p = add_benchmark(
        "compare", "Show two saved runs side by side.", handler="benchmark_compare"
    )
    p.add_argument("baseline", help="Benchmark name or JSON file.")
    p.add_argument("candidate", help="Benchmark name or JSON file.")
    add_benchmark("list", "List saved runs.", handler="benchmark_list")

    p = add_app(
        "profile",
        "Run one request under the torch profiler and download the trace.",
        description="Warm the endpoint up, run one request under torch.profiler, "
        "print the top ops and save a Perfetto/Chrome trace locally. Without "
        "--data the request body is {}. CPU ops are recorded in the thread that runs "
        "the endpoint: work an async endpoint hands to other threads shows up "
        "as GPU kernels only.",
    )
    p.add_argument("path", help="Endpoint path, e.g. /")
    p.add_argument("--data", help="JSON body, or @file.json.")
    depth = p.add_mutually_exclusive_group()
    depth.add_argument(
        "--light",
        dest="mode",
        action="store_const",
        const="light",
        help="GPU kernels only: the smallest trace.",
    )
    depth.add_argument(
        "--deep",
        dest="mode",
        action="store_const",
        const="deep",
        help="Add shapes, memory, stacks and FLOPs (large traces).",
    )
    p.set_defaults(mode="standard")
    p.add_argument(
        "--warmups",
        type=bounded(0, MAX_PROFILE_WARMUPS),
        default=1,
        help="Unprofiled requests to send first (default 1).",
    )
    p.add_argument(
        "--top", type=bounded(1, 100), default=15, help="Ops to print (default 15)."
    )
    p.add_argument(
        "--output-dir",
        help="Where to save profiles (default ~/.fal/dev/profiles/<name>).",
    )
    p.add_argument(
        "--max-trace-mb",
        type=bounded(1, 100_000),
        default=200,
        help="Skip downloading compressed traces over this size (default 200).",
    )

    deps = _group(
        app,
        "deps",
        "Try other package versions without rebuilding the box.",
        parents,
        description="Install packages into an overlay that the app imports ahead "
        "of its own environment (which is never modified), then restart the app. "
        "Only what the app's environment does not already satisfy is installed. "
        "The overlay is kept with the box and installed again on a new runner.",
    )
    add_deps = functools.partial(add, deps, "app", [*parents, name])
    no_restart = argparse.ArgumentParser(add_help=False)
    no_restart.add_argument(
        "--no-restart",
        dest="restart",
        action="store_false",
        help="Change the overlay without restarting the app.",
    )
    p = add_deps(
        "install",
        "Install packages into the overlay.",
        handler="deps_install",
        extra_parents=(no_restart,),
    )
    p.add_argument("packages", nargs="+", help="Requirement specs.")
    p.add_argument(
        "--no-deps", action="store_true", help="Install exactly these packages."
    )
    p.add_argument("--pre", action="store_true", help="Allow pre-releases.")
    p.add_argument("--index-url", help="Package index to use.")
    p.add_argument(
        "--extra-index-url",
        action="append",
        help="Another index (repeatable). The highest version across indexes "
        "wins, so a public package can shadow a private one of the same name.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Allow changing packages the box's app server runs on "
        "(fastapi, pydantic, uvicorn, ...).",
    )
    p = add_deps(
        "uninstall",
        "Remove packages from the overlay (their dependencies stay until reset).",
        handler="deps_uninstall",
        extra_parents=(no_restart,),
    )
    p.add_argument("packages", nargs="+")
    add_deps(
        "reset",
        "Delete the overlay.",
        handler="deps_reset",
        extra_parents=(no_restart,),
    )
    p = add_deps(
        "status",
        "Show the overlay and the versions the app imports.",
        handler="deps_status",
    )
    p.add_argument(
        "packages", nargs="*", help="Also show which version the app imports."
    )
    p.add_argument("--json", action="store_true")

    p = add_app("restart", "Restart the app process; setup() runs again.")
    p.add_argument(
        "--no-sync", dest="sync", action="store_false", help="Skip syncing first."
    )
    p = add_app("logs", "Show the app process's output.")
    p.add_argument("--lines", type=int, default=200)
    p.add_argument("--follow", "-f", action="store_true")
    add_app("status", "Show the box and app state.")
    add_app("stop", "Shut the box down and delete it.")


def _add_dash_command(subparsers, parents) -> None:
    box = _name_parser(
        "Box name (default: the folder's last box, else "
        f"${config.ENV_DEFAULT_NAME} or your OS username).",
        box_name,
    )
    p = add(
        subparsers,
        "dash",
        parents,
        "dash",
        "Live dashboard for a running dev box.",
        description=DASH_DESCRIPTION,
        extra_parents=(box,),
    )
    p.add_argument(
        "--interval",
        type=interval,
        default=3.0,
        help="Seconds between refreshes (default: %(default)s).",
    )
    p.add_argument("--once", action="store_true", help="Print one frame and exit.")
    p.add_argument(
        "--json", action="store_true", help="Print the raw status as JSON and exit."
    )


def _add_view_command(subparsers, parents) -> None:
    p = add(
        subparsers,
        "view",
        parents,
        "view",
        "Show images or videos (files or URLs) inline in the terminal.",
        extra_parents=(_preview_parser(),),
        team_help=argparse.SUPPRESS,
    )
    p.add_argument("files", nargs="+", help="Image or video files, or URLs.")
    p.add_argument(
        "--name",
        help="Accepted for consistency with the other commands; view is local "
        "and ignores it.",
    )


def add_parser(main_subparsers, parents) -> None:
    dev_help = (
        f"{EXPERIMENTAL}Personal GPU dev boxes: a persistent box you shell or SSH "
        "into, with hot reload of inference code."
    )
    parser = main_subparsers.add_parser(
        "dev",
        description=f"{dev_help} Waits animate on a terminal; set "
        "FAL_DEV_NO_ANIMATION=1 for plain status lines.\n\n" + NETWORK_NOTE,
        help=dev_help,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=parents,
    )
    subparsers = parser.add_subparsers(
        title="Commands",
        metavar="command",
        required=True,
        parser_class=FalClientParser,
    )
    _add_box_commands(subparsers, parents)
    _add_app_commands(subparsers, parents)
    _add_dash_command(subparsers, parents)
    _add_view_command(subparsers, parents)
