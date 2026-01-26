import argparse
from contextlib import ExitStack, contextmanager

from .parser import FalParser


@contextmanager
def _pdb():
    try:
        yield
    except Exception:
        try:
            import ipdb as pdb  # noqa: T100
        except ImportError:
            import pdb  # noqa: T100
        pdb.post_mortem()


@contextmanager
def _cprofile():
    import cProfile

    prof = cProfile.Profile()
    prof.enable()

    try:
        yield
    finally:
        prof.disable()
        prof.print_stats(sort="cumtime")


@contextmanager
def debugtools(args):
    with ExitStack() as stack:
        if args.pdb:
            stack.enter_context(_pdb())
        if args.cprofile:
            stack.enter_context(_cprofile())
        try:
            yield
        except Exception:
            if args.debug:
                args.console.print_exception()
            raise


class DebugAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        import fal.flags as flags  # noqa: PLC0415

        flags.DEBUG = True
        setattr(namespace, self.dest, True)


def get_debug_parser():
    from fal.flags import DEBUG

    parser = FalParser(add_help=False)
    group = parser.add_argument_group(title="Debug")
    group.add_argument(
        "--debug",
        action=DebugAction,
        nargs=0,
        help="Show verbose errors." if DEBUG else argparse.SUPPRESS,
    )
    group.add_argument(
        "--pdb",
        action="store_true",
        help="Start pdb on error." if DEBUG else argparse.SUPPRESS,
    )
    group.add_argument(
        "--cprofile",
        action="store_true",
        help="Show cProfile report." if DEBUG else argparse.SUPPRESS,
    )
    return parser
