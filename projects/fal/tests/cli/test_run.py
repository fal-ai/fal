from fal.cli.main import parse_args
from fal.cli.run import _run


def test_run():
    args = parse_args(["run", "/my/path.py::myfunc"])
    assert args.func == _run
    assert args.func_ref == ("/my/path.py", "myfunc")
