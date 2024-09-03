import argparse
import sys

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
