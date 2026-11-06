"""Every `fal dev` argument definition.

`fal` builds all parsers on each run, so this module imports nothing heavy. A
command's handler (`cmd_<name>` in box.py, hot.py, app.py or dash.py) is only
imported once that command runs.
"""

from __future__ import annotations

import argparse
import functools
import importlib
from typing import Callable

from fal.cli.parser import FalClientParser, RefAction, add_env_argument
from fal.dev import config
from fal.dev.remote.protocol import MAX_PROFILE_WARMUPS, MAX_REPEAT, MAX_WARMUPS

EXPERIMENTAL = "(experimental) "

DEFAULT_KEEP_ALIVE = 1800

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

HOT_DESCRIPTION = (
    "A project folder holds load.py (load() -> state, run once) and infer.py "
    "(infer(state, **params), reloaded on every save). Saving load.py or "
    "requirements.txt restarts the server, which reloads the weights. Example: "
    "fal dev hot start ./my-model, then "
    "fal dev hot sync --watch --run --prompt 'a red fox'"
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
        return getattr(implementation, f"cmd_{handler}")(args)

    run.__name__ = f"{module}.{handler}"
    return run


def add(
    subparsers,
    module: str,
    parents: list,
    cmd: str,
    help_text: str,
    *,
    handler: str | None = None,
    extra_parents: tuple = (),
    **kwargs,
) -> argparse.ArgumentParser:
    """Add command `cmd`, run by `cmd_<handler>` (default: the command's name)."""
    kwargs.setdefault("description", help_text)
    parser = subparsers.add_parser(
        cmd, help=help_text, parents=[*parents, *extra_parents], **kwargs
    )
    parser.set_defaults(func=_command(module, handler or cmd.replace("-", "_")))
    return parser


def _group(subparsers, cmd: str, help_text: str, parents: list, **kwargs):
    """Add a command that only holds subcommands; returns their subparsers."""
    kwargs.setdefault("description", help_text)
    parser = subparsers.add_parser(cmd, help=help_text, parents=parents, **kwargs)
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
            f"(default: the current one, else {config.DEFAULT_GPU})."
        ),
    )
    p.add_argument(
        "--num-gpus",
        type=positive_int,
        default=config.DEFAULT_NUM_GPUS,
        help="GPUs per box (default: %(default)s).",
    )
    p.add_argument(
        "--idle-minutes",
        type=positive_int,
        default=config.DEFAULT_IDLE_MINUTES,
        help=(
            "Minutes without input or GPU use before the box stops itself "
            "(default: %(default)s)."
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
        "--dockerfile",
        help="Your own image, based on the default (see fal/dev/remote/Dockerfile).",
    )
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
        help="Switch the machine type of a stopped box; it stays until changed again.",
    )
    p.add_argument(
        "--dash",
        action="store_true",
        help="Open the live dashboard (fal dev dash) once the box is ready.",
    )
    p.add_argument(
        "--timeout",
        type=positive_int,
        default=1800,
        help="Seconds to wait (default: %(default)s).",
    )

    p = add_box("down", f"Save {config.LOCAL_WORK}, then stop the box. /data is kept.")
    p.add_argument("--no-save", action="store_true", help=no_save_help)
    add_box("status", "Show whether the box is running.")
    add_box("shell", "Open an interactive shell on the box.")

    p = add_box("exec", "Run one command on the box: fal dev exec -- nvidia-smi")
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
        f"Suspend idle shutdown for N hours (at most {config.MAX_HOLD_HOURS}).",
    )
    p.add_argument(
        "hours", type=float, help="How long to keep the box up regardless of activity."
    )

    add_box(
        "ssh-config", "Print an ~/.ssh/config entry for Cursor / VS Code Remote-SSH."
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
    add_hot(
        "start",
        "Start the hot server and wait for the model to load.",
        extra_parents=(project,),
    )
    p = add_hot(
        "sync",
        "Push a project folder to the box; reloads the server.",
        extra_parents=(project, request),
    )
    p.add_argument("--watch", action="store_true", help="Keep pushing as you save.")
    p.add_argument(
        "--run",
        action="store_true",
        help="With --watch: send a request after each push.",
    )
    add_hot("run", "Send one request to the hot server.", extra_parents=(request,))
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

    p = add_app("sync", "Upload changed files; endpoints reload in place.")
    p.add_argument("--watch", action="store_true", help="Keep syncing on save.")
    p.add_argument("--full", action="store_true", help="Upload every file.")
    p.add_argument(
        "--request",
        metavar="PATH",
        help="Send a request to PATH after each sync (with --data).",
    )
    p.add_argument("--data", help="JSON body, or @file.json.")

    p = add_app("request", "Send a request to the app on the box.")
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
        "--data the request is built from the endpoint's input model "
        "(examples and defaults). CPU ops are recorded in the thread that runs "
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


def add_parser(main_subparsers, parents) -> None:
    dev_help = (
        f"{EXPERIMENTAL}Personal GPU dev boxes: a persistent box you shell or SSH "
        "into, with hot reload of inference code."
    )
    parser = main_subparsers.add_parser(
        "dev", description=dev_help, help=dev_help, parents=parents
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
