import argparse
import sys
from datetime import datetime, timedelta
from typing import Optional

import rich_argparse


class FalParserExit(Exception):
    def __init__(self, status=0):
        self.status = status


class RefAction(argparse.Action):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("default", (None, None))
        super().__init__(*args, **kwargs)

    @classmethod
    def split_ref(cls, value):
        if isinstance(value, tuple):
            return value

        if value.find("::") > 1:
            return value.split("::", 1)

        return value, None

    def __call__(self, parser, args, values, option_string=None):  # noqa: ARG002
        file_path, obj_path = self.split_ref(values)
        setattr(args, self.dest, (file_path, obj_path))


class DictAction(argparse.Action):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("metavar", "<name>=<value>")
        super().__init__(*args, **kwargs)

    def __call__(self, parser, args, values, option_string=None):  # noqa: ARG002
        d = getattr(args, self.dest) or {}

        if isinstance(values, list):
            kvs = values
        else:
            kvs = [values]

        for kv in kvs:
            parts = kv.split("=", 1)
            if len(parts) != 2:
                raise argparse.ArgumentError(
                    self,
                    f'Could not parse argument "{values}" as k1=v1 k2=v2 ... format',
                )
            key, value = parts
            d[key] = value

        setattr(args, self.dest, d)


class SinceAction(argparse.Action):
    LIMIT_LEEWAY = timedelta(minutes=1)

    def _parse_since(self, value: str) -> Optional[datetime]:
        import dateparser

        return dateparser.parse(
            value,
            settings={
                "PREFER_DATES_FROM": "past",
            },
        )

    def __init__(self, *args, **kwargs):
        self._limit = kwargs.pop("limit", None)
        if self._limit:
            if not isinstance(self._limit, str):
                raise ValueError(
                    f"Invalid 'limit' value for SinceAction: {self._limit!r}"
                )

            self._limit_dt = self._parse_since(self._limit)
            if not self._limit_dt:
                raise ValueError(
                    f"Invalid 'limit' value for SinceAction: {self._limit!r}"
                )

        super().__init__(*args, **kwargs)

        # If a default is provided as a string like "1h ago", parse it into a datetime
        # so callers can rely on receiving a datetime even when the flag isn't passed.
        default_value = getattr(self, "default", None)
        if default_value is not None and default_value is not argparse.SUPPRESS:
            if isinstance(default_value, str):
                dt = self._parse_since(default_value)
                if not dt:
                    raise ValueError(
                        f"Invalid 'default' value for SinceAction: {default_value!r}"
                    )
                if (
                    self._limit
                    and self._limit_dt is not None
                    and dt < self._limit_dt - self.LIMIT_LEEWAY
                ):
                    raise ValueError(
                        "Default since value is older than the allowed limit "
                        f"{self._limit}."
                    )
                self.default = dt
            elif isinstance(default_value, datetime):
                if (
                    self._limit
                    and self._limit_dt is not None
                    and default_value < self._limit_dt - self.LIMIT_LEEWAY
                ):
                    raise ValueError(
                        "Default since value is older than the allowed limit "
                        f"{self._limit}."
                    )

    def __call__(self, parser, args, values, option_string=None):  # noqa: ARG002
        if values is None:
            setattr(args, self.dest, None)
            return

        dt = self._parse_since(values)
        if not dt:
            raise argparse.ArgumentError(
                self,
                (
                    f"Invalid since value: {values}. "
                    "Use 'now', relative like '15m' or '24h ago', "
                    "or an ISO timestamp."
                ),
            )

        if self._limit and self._limit_dt is not None:
            if dt < self._limit_dt - self.LIMIT_LEEWAY:
                raise argparse.ArgumentError(
                    self,
                    f"Since value is older than the allowed limit {self._limit}.",
                )

        setattr(args, self.dest, dt)


def _find_parser(parser, func):
    defaults = parser._defaults
    if not func or func == defaults.get("func"):
        return parser

    actions = parser._actions
    for action in actions:
        if not isinstance(action.choices, dict):
            continue
        for subparser in action.choices.values():
            par = _find_parser(subparser, func)
            if par:
                return par
    return None


class FalParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("formatter_class", rich_argparse.RawTextRichHelpFormatter)
        super().__init__(*args, **kwargs)

    def exit(self, status=0, message=None):
        if message:
            self._print_message(message, sys.stderr)
        raise FalParserExit(status)

    def parse_args(self, args=None, namespace=None):
        args, argv = self.parse_known_args(args, namespace)
        if argv:
            parser = _find_parser(self, getattr(args, "func", None)) or self
            parser.error("unrecognized arguments: %s" % " ".join(argv))
        return args


class FalClientParser(FalParser):
    def __init__(self, *args, **kwargs):
        from fal.flags import GRPC_HOST

        super().__init__(*args, **kwargs)
        self.add_argument(
            "--host",
            default=GRPC_HOST,
            help=argparse.SUPPRESS,
        )
        self.add_argument(
            "--team",
            help="The team to use.",
        )


def get_output_parser():
    parser = FalParser(add_help=False)
    group = parser.add_argument_group(title="Output")
    group.add_argument(
        "--output",
        type=str,
        default="pretty",
        choices=["pretty", "json"],
        help="Modify the command output",
    )
    group.add_argument(
        "--json",
        action="store_const",
        const="json",
        dest="output",
        help="Output in JSON format (same as --output json)",
    )
    return parser
