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

    def __call__(self, parser, args, values, option_string=None):  # noqa: ARG002
        if isinstance(values, tuple):
            file_path, obj_path = values
        elif values.find("::") > 1:
            file_path, obj_path = values.split("::", 1)
        else:
            file_path, obj_path = values, None

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


class FalParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("formatter_class", rich_argparse.RawTextRichHelpFormatter)
        super().__init__(*args, **kwargs)

    def exit(self, status=0, message=None):
        if message:
            self._print_message(message, sys.stderr)
        raise FalParserExit(status)


class FalClientParser(FalParser):
    def __init__(self, *args, **kwargs):
        from fal.flags import GRPC_HOST

        super().__init__(*args, **kwargs)
        self.add_argument(
            "--host",
            default=GRPC_HOST,
            help=argparse.SUPPRESS,
        )
