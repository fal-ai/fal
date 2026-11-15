"""Every `fal dev` argument definition.

`fal` builds all parsers on each run, so this module imports nothing heavy. A
command's handler (`cmd_<name>` in box.py, hot.py, dash.py or view.py)
is only imported once that command runs.
"""

from __future__ import annotations

import argparse
import functools
import importlib
from typing import Callable

from fal.cli.parser import FalClientParser
from fal.dev import config
from fal.dev.errors import DevError

EXPERIMENTAL = "(experimental) "

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

Preinstalled on the stock box image (the Python you deployed from, CUDA 12.8):
torch 2.8.0 and torchvision 0.23.0 (cu128), transformers 4.56.1, diffusers 0.35.1,
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
    "Live dashboard for a running dev box. Shows GPU and CPU use, idle shutdown, "
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


def env_name(value: str) -> str:
    # Never echo the argument: `--secret NAME=value` is an easy slip, and the
    # value would land in terminals and CI logs.
    if not config.ENV_VAR_PATTERN.fullmatch(value):
        raise argparse.ArgumentTypeError(
            "expected a NAME of letters, digits and underscores (no '=' or value)"
        )
    try:
        return config.check_env_name(value, "name")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def env_pair(value: str) -> tuple[str, str]:
    name, sep, text = value.partition("=")
    if not sep:
        raise argparse.ArgumentTypeError("expected KEY=VALUE")
    return env_name(name), text


def add_user_env_arguments(p: argparse.ArgumentParser, *, removable: bool) -> None:
    """--secret and --set-env (and, on deploy, their --unset-* counterparts)."""
    p.add_argument(
        "--secret",
        action="append",
        default=[],
        metavar="NAME",
        type=env_name,
        help=(
            "Give the box a fal secret, by name (repeatable), e.g. HF_TOKEN. Secrets "
            "are team-level: create them with `fal secrets set NAME=value`; the box "
            "sees only the names you list, as environment variables in shells, "
            "`fal dev exec`, ssh, `hot` and Claude sessions. `fal dev status` shows "
            "the names, never the values. "
            + (
                "Remembered on this machine for later deploys; remove one with "
                "--unset-secret."
                if removable
                else "Applies when the box is first created; later use "
                "`fal dev deploy --secret`."
            )
        ),
    )
    p.add_argument(
        "--set-env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        type=env_pair,
        help=(
            "Set an environment variable on the box (repeatable). Not for secrets: "
            "the value is stored in the box's settings; use --secret for those. "
            + (
                "Remembered on this machine for later deploys; remove one with "
                "--unset-env."
                if removable
                else "Applies when the box is first created; later use "
                "`fal dev deploy --set-env`."
            )
        ),
    )
    if not removable:
        return
    p.add_argument(
        "--unset-secret",
        action="append",
        default=[],
        metavar="NAME",
        help="Stop giving the box this secret (repeatable). The secret itself stays.",
    )
    p.add_argument(
        "--unset-env",
        action="append",
        default=[],
        metavar="KEY",
        help="Stop setting this environment variable on the box (repeatable).",
    )


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
    from urllib.error import HTTPError, URLError
    from urllib.parse import urlsplit

    import grpc
    import httpx

    from fal import flags
    from fal._proxy import https_proxy_for, unreachable_message
    from fal.exceptions import FalServerlessException

    if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout, httpx.ProxyError)):
        try:
            host = exc.request.url.host
        except RuntimeError:
            host = "fal"
        message = unreachable_message(host, exc)
    elif isinstance(exc, grpc.RpcError) and exc.code() == grpc.StatusCode.UNAVAILABLE:
        host = flags.GRPC_HOST.partition(":")[0]
        if https_proxy_for(host) is None:
            # Not proxied: `fal`'s own handler already explains UNAVAILABLE.
            return None
        message = unreachable_message(host, exc.details())
    elif (
        isinstance(exc, FalServerlessException)
        and isinstance(exc.__context__, URLError)
        and not isinstance(exc.__context__, HTTPError)
    ):
        # The account lookup could not connect and already wrote the message
        # (fal.auth); an HTTP error answer is not a network problem.
        host = urlsplit(flags.REST_URL).hostname or flags.REST_URL
        message = str(exc)
    else:
        return None
    proxied = https_proxy_for(host) is not None
    return DevError(
        message,
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
        team_help = TEAM_HELP
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
        type=config.machine_type_name,
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
            "(default: %(default)s); CPU machine types count input only. A GPU box "
            "whose GPUs cannot be queried never stops itself: stop it with "
            "`fal dev down`."
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
    add_user_env_arguments(p, removable=True)
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
        type=config.machine_type_name,
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
    add_user_env_arguments(p, removable=False)
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
    _add_dash_command(subparsers, parents)
    _add_view_command(subparsers, parents)
