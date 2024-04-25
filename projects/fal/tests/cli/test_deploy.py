from fal.cli.deploy import _deploy
from fal.cli.main import parse_args


def test_deploy():
    args = parse_args(["deploy", "myfile.py::MyApp"])
    assert args.func == _deploy
    assert args.app_ref == ("myfile.py", "MyApp")
