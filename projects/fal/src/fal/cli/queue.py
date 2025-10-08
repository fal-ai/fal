from __future__ import annotations

import json
from http import HTTPStatus

import httpx

from fal.rest_client import REST_CLIENT

from .parser import FalClientParser, get_output_parser


def _queue_size(args):
    from fal.api.deploy import _get_user

    user = _get_user()

    url = f"{REST_CLIENT.base_url}/applications/{user.username}/{args.app_name}/queue"
    headers = REST_CLIENT.get_headers()

    with httpx.Client(base_url=REST_CLIENT.base_url, headers=headers, timeout=300) as c:
        resp = c.get(url)

    if resp.status_code != HTTPStatus.OK:
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:
            detail = resp.text
        raise RuntimeError(f"Failed to get queue size: {detail}")

    data = resp.json()
    size = data.get("size", 0)

    if args.output == "json":
        args.console.print(json.dumps({"size": size}))
    else:
        args.console.print(f"Queue size: {size}")


def _queue_flush(args):
    from fal.api.deploy import _get_user

    user = _get_user()

    url = f"{REST_CLIENT.base_url}/applications/{user.username}/{args.app_name}/queue"
    headers = REST_CLIENT.get_headers()

    with httpx.Client(base_url=REST_CLIENT.base_url, headers=headers, timeout=300) as c:
        resp = c.delete(url)

    if resp.status_code != HTTPStatus.OK:
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:
            detail = resp.text
        raise RuntimeError(f"Failed to flush queue: {detail}")

    args.console.print("Queue flushed successfully.")


def add_parser(main_subparsers, parents):
    queue_help = "Manage application queues."
    parser = main_subparsers.add_parser(
        "queue",
        description=queue_help,
        help=queue_help,
        parents=parents,
    )

    subparsers = parser.add_subparsers(
        title="Commands",
        metavar="command",
        required=True,
        parser_class=FalClientParser,
    )

    size_help = "Get queue size for an application."
    size_parser = subparsers.add_parser(
        "size",
        description=size_help,
        help=size_help,
        parents=[*parents, get_output_parser()],
    )
    size_parser.add_argument(
        "app_name",
        help="Application name (do not prefix with owner).",
    )
    size_parser.set_defaults(func=_queue_size)

    flush_help = "Flush all pending requests in an application queue."
    flush_parser = subparsers.add_parser(
        "flush",
        description=flush_help,
        help=flush_help,
        parents=parents,
    )
    flush_parser.add_argument(
        "app_name",
        help="Application name.",
    )
    flush_parser.set_defaults(func=_queue_flush)
