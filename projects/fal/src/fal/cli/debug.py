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


def get_debug_parser():
    parser = FalParser(add_help=False)
    group = parser.add_argument_group(title="Debug")
    group.add_argument("--debug", action="store_true", help="Show verbose errors.")
    group.add_argument(
        "--pdb",
        action="store_true",
        help="Start pdb on error.",
    )
    group.add_argument(
        "--cprofile",
        action="store_true",
        help="Show cProfile report.",
    )
    return parser
