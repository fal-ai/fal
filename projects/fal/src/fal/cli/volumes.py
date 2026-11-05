import posixpath

from fal.api.client import SyncServerlessClient

from .parser import FalClientParser


def _create(args):
    client = SyncServerlessClient(host=args.host, team=args.team)
    name = client.volumes.create(args.name)
    args.console.print(name)


def _add_create_parser(subparsers, parents):
    create_help = "Create a volume and print its canonical name."
    parser = subparsers.add_parser(
        "create",
        description=create_help,
        help=create_help,
        parents=parents,
    )
    parser.add_argument(
        "name",
        help="Volume name (letters, digits, '.', '_', '-'; stored lowercase).",
    )
    parser.set_defaults(func=_create)


def _weights_pull(args):
    from fal.weights import (  # noqa: PLC0415
        DEFAULT_HF_CACHE_DIR,
        DEFAULT_HF_HOME,
        pull_weights,
        validate_dest,
    )

    try:
        if args.workers < 1:
            raise ValueError("--workers must be at least 1")
        validate_dest(args.dest)
    except ValueError as exc:
        args.parser.error(str(exc))

    host = SyncServerlessClient(host=args.host, team=args.team)._create_host()
    args.console.print(
        f"Pulling [bold]{args.repo}[/] on a {args.machine_type} CPU machine..."
    )
    try:
        result = pull_weights(
            args.repo,
            revision=args.revision,
            dest=args.dest,
            workers=args.workers,
            machine_type=args.machine_type,
            host=host,
        )
    except Exception:
        args.console.print(
            "Pull did not finish. If it was interrupted (network or timeout), "
            "run the same command again to resume; files already on /data are kept."
        )
        raise
    gib = result["bytes"] / 2**30
    args.console.print(
        f"Done in {result['seconds']:.0f}s: {gib:.2f} GiB snapshot at {result['path']}"
    )
    if args.dest is None:
        commit = posixpath.basename(result["path"])
        args.console.print(
            f"Snapshot commit {commit}. Apps find it with "
            f"HF_HUB_CACHE={DEFAULT_HF_CACHE_DIR} (or HF_HOME={DEFAULT_HF_HOME})."
        )
        if result.get("marker_advanced"):
            args.console.print(
                "New runners of LLMApps without model_revision will serve this commit."
            )
        elif "marker_error" in result:
            args.console.print(
                "Could not update the LLMApp snapshot marker "
                f"({result['marker_error']}); unpinned apps keep their snapshot."
            )


def _add_weights_parser(subparsers, parents):
    from fal.weights import (  # noqa: PLC0415
        DEFAULT_HF_CACHE_DIR,
        DEFAULT_MACHINE_TYPE,
        DEFAULT_WORKERS,
    )

    weights_help = "Pre-stage model weights on /data."
    parser = subparsers.add_parser(
        "weights",
        parents=parents,
        description=weights_help,
        help=weights_help,
    )
    weights_subparsers = parser.add_subparsers(
        title="Commands",
        metavar="command",
        required=True,
        parser_class=FalClientParser,
    )
    pull_help = (
        "Download public Hugging Face weights to /data using a CPU machine, "
        "so the first GPU boot does not download them."
    )
    pull = weights_subparsers.add_parser(
        "pull",
        description=pull_help,
        help=pull_help,
        parents=parents,
    )
    pull.add_argument("repo", help="Hugging Face repo id, e.g. Qwen/Qwen2.5-7B.")
    pull.add_argument("--revision", default=None, help="Branch, tag or commit.")
    pull.add_argument(
        "--dest",
        default=None,
        help=(
            "Plain directory under /data to write to. Default: the Hugging Face "
            f"hub cache ({DEFAULT_HF_CACHE_DIR})."
        ),
    )
    pull.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Parallel file downloads (default {DEFAULT_WORKERS}).",
    )
    pull.add_argument(
        "--machine-type",
        default=DEFAULT_MACHINE_TYPE,
        help=f"CPU machine type (default {DEFAULT_MACHINE_TYPE}).",
    )
    pull.set_defaults(func=_weights_pull, parser=pull)


def add_parser(main_subparsers, parents):
    volumes_help = "Manage fal volumes."
    parser = main_subparsers.add_parser(
        "volumes",
        aliases=["volume"],
        parents=parents,
        description=volumes_help,
        help=volumes_help,
    )

    subparsers = parser.add_subparsers(
        title="Commands",
        metavar="command",
        required=True,
        parser_class=FalClientParser,
    )

    _add_create_parser(subparsers, parents)
    _add_weights_parser(subparsers, parents)
