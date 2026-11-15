"""Every `fal dev` argument definition.

`fal` builds all parsers on each run, so this module imports nothing heavy. A
command's handler (`cmd_<name>` in box.py, hot.py, dash.py or view.py)
is only imported once that command runs.
"""

from __future__ import annotations

import argparse
import functools
import importlib
import sys
from typing import Callable

from fal.cli.parser import FalClientParser
from fal.dev import config
from fal.dev.errors import (
    EXIT_AUTH,
    EXIT_INTERRUPTED,
    EXIT_NO_BOX,
    EXIT_USAGE,
    DevError,
)

EXPERIMENTAL = "(experimental) "

# `up` may build the image, deploy and wait for a machine; a proxy start only
# wakes a deployed box.
UP_TIMEOUT = 1800
PROXY_START_TIMEOUT = 900

NAME_HELP = "Box name (default: this folder's box)"
TEAM_HELP = "Team the box runs as (remembered per box)"
SSH_CONFIG_TEAM_HELP = (
    "Write --team into the entry's ProxyCommand. Nothing is remembered until "
    "ssh-proxy runs."
)
# Each command's --yes skips one thing; the help names it.
YES_DEPLOY_HELP = (
    "Skip the confirmation before deploying into a team that has production "
    "apps (needed without a terminal)."
)
YES_DOWN_HELP = (
    "Skip the confirmation when the GPU is busy or a job is running (needed "
    "without a terminal)."
)
YES_DESTROY_HELP = (
    "Skip typing the box name to confirm the delete (needed without a "
    "terminal). Requires --name."
)
VERBOSE_HELP = "Show more detail, such as the account the box runs as."
JSON_HELP = "Print the result as one JSON object on stdout (errors too)."

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

The stock box image has torch, torchvision, transformers, diffusers, accelerate,
safetensors, huggingface_hub and uv preinstalled (`fal dev exec -- pip list`
shows the versions). A custom image (`fal dev deploy --dockerfile`) has what its
Dockerfile installs.

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

HOT_DESCRIPTION = """\
Keep a model loaded on the box and reload your inference code on save.

A project folder has load.py (def load() -> state: loads the weights, runs once)
and infer.py (def infer(state, **params): runs per request, reloaded on save).
Saving load.py or requirements.txt reloads the weights; any other .py file
reloads code only. Layout and requirements: fal dev hot start --help

Example: fal dev hot start ./my-model, then
fal dev hot sync --watch --run --prompt 'a red fox'
"""

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


ENV_NOTES = """\
Secrets and environment: --secret NAME gives the box a team secret. Create it
with `fal secrets set NAME=value`; the box sees only the names you list, as
environment variables in shells, `fal dev exec`, ssh, `hot` and Claude
sessions, and `fal dev status` shows the names, never the values. --set-env
KEY=VALUE sets a plain variable whose value is stored in the box's settings, so
use --secret for anything sensitive."""

STARSHIP_NOTE = """\
--starship copies only your starship.toml ($STARSHIP_CONFIG or
~/.config/starship.toml) into the box's settings (keep secrets out of it) and
installs starship there."""

TEAM_DATA_NOTE = """\
--team-data keeps the box on the team-shared /data (shared model weights, other
apps' files) instead of its own volume. By default /data is the box's volume,
devbox-<name>, and the team /data is not mounted, so a box whose files are on
the team /data needs this flag to keep them. It is not remembered: pass it on
every deploy."""

NAME_RULES = (
    "Which box and team\n"
    "  Box:   --name, else the box this folder last used (remembered once the\n"
    f"         box exists), else ${config.ENV_DEFAULT_NAME}, else your OS username\n"
    "         (lowercased, other characters become dashes, 40 characters at\n"
    f"         most). Names use lowercase letters, digits and dashes; the app is\n"
    f"         {config.APP_PREFIX}<name>.\n"
    "  Team:  --team, else the team remembered for the box, else your personal\n"
    "         account. `fal account set` never applies to a box. Under an API key\n"
    "         the key's own account is used and --team is ignored.\n"
    f"  State is kept in ~/.fal/dev on this machine."
)

EXIT_CODES = (
    "Exit codes\n"
    "  0 ok, 1 error, 2 usage, 3 box not found or stopped, 4 auth or team,\n"
    "  5 timeout, 130 interrupted with Ctrl-C. `exec` returns the remote\n"
    "  command's own code. With --json, a failure also prints\n"
    '  {"error": {"title", "message", "fixes"}} on stdout.'
)

QUICK_START = (
    "Quick start\n"
    "  fal dev up           start your box (creates it the first time)\n"
    "  fal dev shell        open a shell on it\n"
    "  fal dev hot start    keep a model loaded and reload code on save\n"
    "  fal dev down         save /work and stop it (stops billing)"
)


def examples(*rows: tuple[str, str], notes: str = "") -> str:
    """An `examples:` epilog: (command, what it does) rows, then optional notes."""
    width = max(len(command) for command, _ in rows)
    lines = [f"  {command.ljust(width)}   {what}" for command, what in rows]
    text = "examples:\n" + "\n".join(lines)
    return f"{text}\n\n{notes}" if notes else text


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
    later = "fal dev deploy"
    p.add_argument(
        "--secret",
        action="append",
        default=[],
        metavar="NAME",
        type=env_name,
        help=(
            "Give the box a team secret, by name (repeatable), e.g. HF_TOKEN. "
            + (
                "Remembered for later deploys."
                if removable
                else f"Applies when the box is created; later use `{later} --secret`."
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
            "Set an environment variable on the box (repeatable; not for secrets). "
            + (
                "Remembered for later deploys."
                if removable
                else f"Applies when the box is created; later use `{later} --set-env`."
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


class DevParser(FalClientParser):
    """A `fal dev` parser: its own usage and a `try:` fix on a parse error."""

    def error(self, message):
        _parse_error(self, message)


def _parse_fixes(parser: argparse.ArgumentParser, message: str) -> list:
    import difflib
    import re

    fixes = []
    wrong = re.search(r"invalid choice: '([^']+)'", message)
    if wrong:
        names = [
            name
            for action in parser._actions
            if isinstance(action.choices, dict)
            for name in action.choices
        ]
    else:
        wrong = re.search(r"unrecognized arguments: (--[^\s=]+)", message)
        names = [o for a in parser._actions for o in a.option_strings]
    if wrong:
        # Strict enough that an unrelated word ("bogus") suggests nothing.
        close = difflib.get_close_matches(wrong.group(1), names, n=1, cutoff=0.75)
        if close:
            fixes.append((f"{parser.prog} {close[0]}", "did you mean this?"))
    fixes.append((f"{parser.prog} --help", "see the options and examples"))
    return fixes


def _parse_error(parser: argparse.ArgumentParser, message: str) -> None:
    """Show a parse error with the command's own usage, then exit 2."""
    from fal.dev.cli import error_panel

    exc = DevError(
        message,
        title=parser.prog,
        fixes=_parse_fixes(parser, message),
        code=EXIT_USAGE,
        usage=parser.format_usage(),
    )
    if "--json" in sys.argv[1:]:
        print(error_panel.as_json(exc, parser.prog))
    error_panel.show(exc, _stderr_console())
    parser.exit(EXIT_USAGE)


def _command(
    module: str, handler: str, command: str | None = None
) -> Callable[[argparse.Namespace], object]:
    """A command that imports its handler `cmd_<handler>` when it runs.

    `command` names it in error panels. Failures become an error panel on
    stderr and the error's exit code (and a JSON error object on stdout under
    --json); Ctrl-C exits 130.
    """

    def run(args: argparse.Namespace) -> object:
        err = _stderr_console()
        if args is not None:
            args.err = err
        implementation = importlib.import_module(f"fal.dev.cli.{module}")
        try:
            return getattr(implementation, f"cmd_{handler}")(args)
        except DevError as exc:
            return _fail(args, exc, command, err)
        except KeyboardInterrupt:
            err.print("interrupted", highlight=False)
            return EXIT_INTERRUPTED
        except Exception as exc:
            wrapped = known_failure(exc)
            if wrapped is None:
                raise
            wrapped.__cause__ = exc
            return _fail(args, wrapped, command, err)

    run.__name__ = f"{module}.{handler}"
    return run


def _stderr_console():
    from rich.console import Console

    return Console(stderr=True, soft_wrap=True)


def known_failure(exc: Exception) -> DevError | None:
    """Logged out, box not deployed or host unreachable, as a DevError; else None."""
    return _not_logged_in(exc) or _not_deployed(exc) or _unreachable(exc)


def _not_logged_in(exc: Exception) -> DevError | None:
    from fal.exceptions.auth import UnauthenticatedException

    if not isinstance(exc, UnauthenticatedException):
        return None
    return DevError(
        "you are not logged in, and no API key is set",
        title="not logged in",
        fixes=[
            ("fal auth login", "log in with your browser"),
            ("fal profile key", "or set an API key"),
        ],
        code=EXIT_AUTH,
    )


def _not_deployed(exc: Exception) -> DevError | None:
    """The platform's NOT_FOUND for the box's app, said in box terms."""
    import re

    import grpc

    if not (
        isinstance(exc, grpc.RpcError)
        and callable(getattr(exc, "code", None))
        and exc.code() == grpc.StatusCode.NOT_FOUND
        and "Application" in (exc.details() or "")
    ):
        return None
    found = re.search(r"'name': '([^']+)'", exc.details())
    app = found.group(1) if found else ""
    name = app.removeprefix(config.APP_PREFIX) if app else None
    flag = f" --name {name}" if name else ""
    return DevError(
        f"box {name} is not deployed" if name else "this box is not deployed",
        title="no such box",
        fixes=[
            (f"fal dev up{flag}", "create and start it"),
            ("fal dev list", "see your boxes"),
        ],
        code=EXIT_NO_BOX,
    )


def _fail(
    args: argparse.Namespace | None, exc: DevError, command: str | None, err
) -> int:
    """Show a failure on stderr, plus a JSON error object on stdout under --json."""
    from fal.dev.cli import error_panel

    if args is not None and getattr(args, "json", False) is True:
        args.console.print_json(error_panel.as_json(exc, command))
    error_panel.show(exc, err, command)
    return exc.code


def _hold_hours(text: str) -> float:
    """Hours from `2`, `0.5`, `30m` or `2h`."""
    value = text.strip().lower()
    scale = 1.0
    if value.endswith("m"):
        value, scale = value[:-1], 1 / 60
    elif value.endswith("h"):
        value = value[:-1]
    try:
        return float(value) * scale
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"{text!r} is not a duration; use hours (2, 0.5) or 30m, 2h"
        ) from None


def _add_take_over(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--take-over",
        action="store_true",
        help=(
            "Deploy over a box that another owner created. Without it a deploy "
            "refuses; owners are recorded on each deploy."
        ),
    )


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
    elif (
        isinstance(exc, grpc.RpcError)
        and callable(getattr(exc, "code", None))
        and exc.code() == grpc.StatusCode.UNAVAILABLE
    ):
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


def _add_yes(parser: argparse.ArgumentParser, help_text: str) -> None:
    parser.add_argument("--yes", action="store_true", help=help_text)


def _add_json(parser: argparse.ArgumentParser, help_text: str = JSON_HELP) -> None:
    parser.add_argument("--json", action="store_true", help=help_text)


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
    parser.set_defaults(
        func=_command(module, handler or cmd.replace("-", "_"), parser.prog)
    )
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
        parser_class=DevParser,
    )


def _name_parser(
    help_text: str, type_=None, *, verbose: bool = True
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--name", type=type_, help=help_text)
    if verbose:
        parser.add_argument("-v", "--verbose", action="store_true", help=VERBOSE_HELP)
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
    box = _name_parser(NAME_HELP, box_name)
    # `up` has its own --verbose (the whole deploy log).
    box_up = _name_parser(NAME_HELP, box_name, verbose=False)
    add_box = functools.partial(add, subparsers, "box", [*parents, box])
    no_save_help = (
        f"Skip saving {config.LOCAL_WORK} first; changes since the last save are lost."
    )
    # Same list as the dash help: what keeps a box from idle shutdown.
    activity = (
        "Idle shutdown counts terminal input, a connected SSH client or editor, "
        "GPU use of 10% or more, Claude sessions, /hot requests and holds as "
        "activity."
    )

    p = add_box(
        "deploy",
        "Build the image and (re)deploy your box as a private app. Ends open shells.",
        description=(
            "Build the image and (re)deploy your box as a private app. Ends open "
            f"shells.\n\n{ENV_NOTES}\n\n{STARSHIP_NOTE}\n\n{TEAM_DATA_NOTE}"
        ),
        epilog=examples(
            ("fal dev deploy", "redeploy with the current settings"),
            (
                "fal dev deploy --machine-type GPU-H100",
                "switch the machine type (restarts the box)",
            ),
            ("fal dev deploy --secret HF_TOKEN", "give the box a secret"),
        ),
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
        help="Use your local starship prompt on the box (not remembered).",
    )
    p.add_argument(
        "--dockerfile",
        help="Your own image, based on the default (see fal/dev/remote/Dockerfile).",
    )
    add_user_env_arguments(p, removable=True)
    _add_yes(p, YES_DEPLOY_HELP)
    _add_take_over(p)
    p.add_argument(
        "--team-data",
        action="store_true",
        help="Keep the box on the team-shared /data (not remembered).",
    )

    p = add(
        subparsers,
        "box",
        [*parents, box_up],
        "up",
        "Start the box (deploying it the first time) and wait until ready.",
        description=(
            "Start the box (deploying it the first time) and wait until ready.\n\n"
            f"{ENV_NOTES}\n\n{STARSHIP_NOTE}"
        ),
        epilog=examples(
            ("fal dev up", "start this folder's box"),
            ("fal dev up --name demo", "start (or create) a box named demo"),
            ("fal dev up --machine-type GPU-H100", "start it on another machine type"),
            ("fal dev up --json", "print the result as JSON (scripts)"),
            notes=(
                "Starting takes a few minutes; the last start's time is remembered "
                "on this machine and shown as the typical wait. Secrets, --set-env "
                "and --starship apply when the box is created; to change them on an "
                "existing box use `fal dev deploy`."
            ),
        ),
    )
    p.add_argument(
        "--machine-type",
        type=config.machine_type_name,
        help=(
            "Switch the machine type of a stopped box; it stays until changed "
            f"again (default: the box's current type, else {config.DEFAULT_GPU} "
            f"for a new box). Limited by ${config.ENV_ALLOWED_MACHINES}."
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
        help="Use your local starship prompt on the box (applies when created).",
    )
    add_user_env_arguments(p, removable=False)
    _add_yes(p, YES_DEPLOY_HELP)
    _add_take_over(p)
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help=(
            "Show the whole deploy log after the run, not its last lines on "
            "failure, and the account the box runs as."
        ),
    )
    _add_json(p)
    p.add_argument(
        "--timeout",
        type=positive_int,
        default=UP_TIMEOUT,
        help=(
            "Seconds to wait for the box to be ready (default: %(default)s; the "
            "first deploy builds the image)."
        ),
    )

    p = add_box(
        "down",
        f"Save {config.LOCAL_WORK}, then stop the box. /data is kept.",
        epilog=examples(
            ("fal dev down", f"save {config.LOCAL_WORK}, then stop the box"),
            ("fal dev down --no-save", "stop without saving"),
            ("fal dev down --yes", "stop even if the GPU is busy"),
            ("fal dev down --json", "print the result as JSON"),
        ),
    )
    p.add_argument("--no-save", action="store_true", help=no_save_help)
    _add_yes(p, YES_DOWN_HELP)
    _add_json(p)
    p = add_box(
        "status",
        "Show whether the box is running.",
        epilog=examples(
            ("fal dev status", "one line: state, machine, uptime, hold"),
            ("fal dev status --json", "the full snapshot for scripts"),
            ("fal dev status --name demo", "another box"),
        ),
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Print the same snapshot as `fal dev dash --json` and exit.",
    )
    p = add(
        subparsers,
        "box",
        parents,
        "list",
        "List the dev boxes of your account or team.",
        team_help=(
            "List this team's boxes (default: your personal account). Not "
            "remembered: pass it each time."
        ),
        epilog=examples(
            ("fal dev list", "your boxes, running first"),
            ("fal dev list --team acme", "a team's boxes"),
            ("fal dev list --all --json", "every box, for scripts"),
            notes=(
                "Running boxes come first, then the rest by how recently this "
                "machine used them. The box of this folder is highlighted."
            ),
        ),
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="Show every stopped box, not just the 10 most recent.",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Print a JSON list, one object per box (all of them).",
    )
    p = add_box(
        "destroy",
        "Delete the box's app. Its /data volume is kept: there is no delete API yet.",
        epilog=examples(
            ("fal dev destroy", "delete this folder's box (asks you to type its name)"),
            ("fal dev destroy --name demo --yes", "delete demo without asking"),
            notes=(
                "--yes needs --name, so a remembered box is never deleted by accident."
            ),
        ),
    )
    _add_yes(p, YES_DESTROY_HELP)
    add_box(
        "shell",
        "Open an interactive shell on the box.",
        epilog=examples(
            ("fal dev shell", "a shell on this folder's box"),
            ("fal dev shell --name demo", "a shell on another box"),
        ),
        description="Open an interactive shell on the box. Your own shell setup "
        f"goes in ~/.devboxrc on the box (kept at {config.WORKSPACE_ROOT}/<name>/"
        ".devboxrc, so it survives restarts). ssh sessions get HISTFILE, DEVBOX_* "
        "and the uv warning too. Needs a running box: it does not start a stopped "
        "one (run `fal dev up` first). Leave with exit or ctrl-d; the box keeps "
        "running.",
    )

    p = add_box(
        "exec",
        "Run one command on the box: fal dev exec -- nvidia-smi",
        description=EXEC_DESCRIPTION,
        epilog=examples(
            ("fal dev exec -- nvidia-smi", "run one command and print its output"),
            ("fal dev exec -- bash -c 'df -h && ls /work'", "use a shell for pipes"),
            ("echo hi | fal dev exec -i -- cat", "forward piped input"),
            notes=(
                "Needs a running box: it does not start a stopped one (run `fal dev "
                "up` first).\n"
                "Stdin is not forwarded unless you pass -i. There is no -t: a remote "
                "terminal merges stderr into stdout, which exec's output-loss check "
                "needs apart; use `fal dev shell` or ssh for an interactive program.\n"
                "exit codes: the remote command's own code (127 means command not "
                "found), 75 the output may be incomplete (the connection ended before "
                "the command's end marker arrived: run it again), 1 an error before "
                "the command ran"
            ),
        ),
    )
    p.add_argument(
        "-i",
        "--stdin",
        action="store_true",
        help="Forward piped input to the command (needs input piped in).",
    )
    p.add_argument(
        "command", nargs=argparse.REMAINDER, help="The command to run, after --."
    )

    p = add_box(
        "logs",
        "Follow the box's logs.",
        epilog=(
            "examples:\n"
            "  fal dev logs                 last 20 lines, then follow\n"
            "  fal dev logs -n 100 --no-follow\n"
            "  fal dev logs --all           the latest runner's whole log\n\n"
            "The platform keeps only the last 23 hours of start times, so --all "
            "cannot reach a runner older than that."
        ),
    )
    p.add_argument(
        "--all",
        action="store_true",
        help=(
            "Show the full log of the latest runner (a live one if there is one), "
            "even if it stopped. Only runners started in the last 23 hours."
        ),
    )
    p.add_argument(
        "-n",
        "--tail",
        type=int,
        metavar="N",
        help=(
            "Lines of history to show first (default: 20; with --all, the whole log)."
        ),
    )
    p.add_argument(
        "--no-follow",
        action="store_true",
        help="Print the history and exit instead of following new lines.",
    )
    p.add_argument(
        "--full",
        action="store_true",
        help="Show log lines over 500 characters in full instead of cut short.",
    )

    p = add_box(
        "hold",
        f"Suspend idle shutdown for N hours (at most {config.MAX_HOLD_HOURS}); "
        "--off clears the hold.",
        epilog=(
            "examples:\n"
            "  fal dev hold 4       keep the box up for 4 hours\n"
            "  fal dev hold --off   let idle shutdown apply again\n"
            f"\n{activity} A hold lasts at most {config.MAX_HOLD_HOURS} hours and "
            "lives on the running box only: stopping the box (fal dev down, or an "
            "idle stop) clears it, and a new start has none. Holding a shorter "
            "time than one already set keeps the longer one."
        ),
    )
    p.add_argument(
        "hours",
        type=_hold_hours,
        nargs="?",
        help=(
            "How long to keep the box up regardless of activity: hours (2, 0.5) "
            "or a duration (30m, 2h)."
        ),
    )
    p.add_argument(
        "--off",
        action="store_true",
        help="Clear the hold so idle shutdown applies again. The box keeps running.",
    )
    _add_json(p)

    p = add_box(
        "ssh-config",
        "Print an ~/.ssh/config entry for Cursor / VS Code Remote-SSH.",
        team_help=SSH_CONFIG_TEAM_HELP,
        epilog=(
            "examples:\n"
            "  fal dev ssh-config               print the entry\n"
            "  fal dev ssh-config --install     add it to ~/.ssh/config, then: "
            "ssh devbox-<name>\n"
            "--host (the fal API host, as for every fal command) is written into "
            "the entry's ProxyCommand, so ssh reaches the same environment. "
            "--install adds the entry between marker comments in ~/.ssh/config "
            "(running it again replaces the entry) and keeps the previous file as "
            "~/.ssh/config.fal-dev.bak."
        ),
    )
    p.add_argument(
        "--install",
        action="store_true",
        help="Add or replace this box's entry in ~/.ssh/config instead of printing.",
    )
    p = add_box(
        "cp",
        "Copy files between this laptop and the box; ':' marks the box side. "
        "Starts a stopped box.",
        epilog=(
            "examples:\n"
            "  fal dev cp :/work/out.mp4 .      download a file\n"
            "  fal dev cp -r ./data :/data/x    upload a folder\n"
            "A folder copied onto a folder that already exists lands inside it: "
            "-r ./data :/data/x makes /data/x/data when /data/x exists, and "
            "`:/work/out .` into an existing local folder makes ./out. The final "
            "line shows where a download landed. Relative box paths start in the "
            "box's home folder. Each call opens its own connection (about 6 s "
            "each); to move many files, copy a folder or use ssh. Uses scp over the "
            "ssh-proxy with your default keys in ~/.ssh (your ~/.ssh/config is "
            "not used); ~/.ssh/known_hosts is not touched."
        ),
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
        "Used by the ssh-config entry, not for direct use; starts a stopped box.",
        description=(
            "The ProxyCommand that the `fal dev ssh-config` entry runs: it carries "
            "the ssh connection over the box and starts a stopped box first. It is "
            "not for direct use (stdout is the ssh connection); run `fal dev "
            "ssh-config --install` and use ssh."
        ),
    )
    p.add_argument(
        "--timeout",
        type=positive_int,
        default=PROXY_START_TIMEOUT,
        help=(
            "Seconds to wait for a stopped box to start (default: %(default)s; "
            "no image build, unlike `up`)."
        ),
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
    hot_preview = _preview_parser(compare=True)
    add_hot(
        "start",
        "Start the hot server and wait for the model to load.",
        extra_parents=(project,),
        description=HOT_START_DESCRIPTION,
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
        help=(
            "Send a request after each push (with --watch), or once after "
            "the push (without)."
        ),
    )
    add_hot(
        "run",
        "Send one request to the hot server.",
        extra_parents=(request, hot_preview),
    )
    p = add_hot("status", "Show the hot server's state and last reload and request.")
    p.add_argument("--log", action="store_true", help="Also show the log tail.")
    p.add_argument("--json", action="store_true", help="Print the raw status as JSON.")
    add_hot("logs", "Show the hot server's log.")
    add_hot(
        "restart",
        "Restart the hot server; the weights reload.",
        description=(
            "Restart the hot server and reload the weights, without changing any "
            "file. Use it after changing something outside the project folder, "
            "or when the model is stuck in a bad state (such as out of GPU memory)."
        ),
    )
    add_hot(
        "stop",
        "Stop the hot server and unload the weights (the box keeps running).",
    )


def _add_dash_command(subparsers, parents) -> None:
    box = _name_parser(NAME_HELP, box_name)
    p = add(
        subparsers,
        "dash",
        parents,
        "dash",
        "Live dashboard for a running dev box.",
        description=DASH_DESCRIPTION,
        extra_parents=(box,),
        epilog=examples(
            ("fal dev dash", "live view of this folder's box (q quits)"),
            ("fal dev dash --once", "print one frame and exit"),
            ("fal dev dash --json", "the raw status for scripts"),
        ),
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


# `fal dev --help` lists commands in the order of a session; plumbing comes last.
COMMAND_ORDER = (
    "up",
    "shell",
    "exec",
    "cp",
    "status",
    "list",
    "logs",
    "dash",
    "hold",
    "view",
    "hot",
    "down",
    "destroy",
    "deploy",
    "ssh-config",
    "ssh-proxy",
)


def _order_commands(subparsers) -> None:
    """Put the commands in COMMAND_ORDER, in `--help` and everywhere else."""
    rank = {name: i for i, name in enumerate(COMMAND_ORDER)}
    last = len(rank)
    choices = subparsers._name_parser_map
    ordered = sorted(choices.items(), key=lambda kv: rank.get(kv[0], last))
    # In place: `choices` is the same dict, and argparse lists it in errors.
    choices.clear()
    choices.update(ordered)
    subparsers._choices_actions.sort(key=lambda a: rank.get(a.dest, last))


def _epilog() -> str:
    import textwrap

    network = textwrap.fill(NETWORK_NOTE, 78)
    animation = textwrap.fill(
        "Waits animate on a terminal; set FAL_DEV_NO_ANIMATION=1 for plain "
        "status lines.",
        78,
    )
    return "\n\n".join(
        [QUICK_START, NAME_RULES, EXIT_CODES, f"{animation}\n\n{network}"]
    )


def add_parser(main_subparsers, parents) -> None:
    dev_help = (
        f"{EXPERIMENTAL}Personal dev boxes (CPU or GPU): a persistent box you shell "
        "or SSH into, with hot reload of inference code."
    )
    parser = main_subparsers.add_parser(
        "dev",
        description=dev_help,
        epilog=_epilog(),
        help=dev_help,
        parents=parents,
    )
    # This parser is built by `fal`, so its errors are routed here.
    parser.error = functools.partial(_parse_error, parser)
    subparsers = parser.add_subparsers(
        title="Commands",
        metavar="command",
        required=True,
        parser_class=DevParser,
    )
    _add_box_commands(subparsers, parents)
    _add_dash_command(subparsers, parents)
    _add_view_command(subparsers, parents)
    _order_commands(subparsers)
