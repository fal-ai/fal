import pytest

from fal.dev import files


def test_walk_leaves_out_skipped_folders_and_secrets(tmp_path):
    for name in ("a.py", "pkg/b.py", ".hidden/c.py", "node_modules/d.js", ".ssh/id_x"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x")
    (tmp_path / ".dotfile").write_text("x")
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    (outside / "private.py").write_text("x")
    (tmp_path / "linked.py").symlink_to(outside / "private.py")
    (tmp_path / "linked_dir").symlink_to(outside, target_is_directory=True)

    found = {p.as_posix() for p in files.walk(tmp_path)}
    assert found == {"a.py", "pkg/b.py", ".hidden/c.py", ".dotfile"}
    visible = {p.as_posix() for p in files.walk(tmp_path, hidden=False)}
    assert visible == {"a.py", "pkg/b.py"}


@pytest.mark.parametrize(
    "parts, secret",
    [
        ((".env",), True),
        (("app", ".env.local"), True),
        (("deploy", "key.PEM"), True),
        (("home", ".aws", "config"), True),
        (("secrets.json",), True),
        (("secrets.py",), False),
        (("secrets", "x.py"), False),
        (("credentials", "a.yaml"), True),
        (("credentials.py",), False),
        (("secrets",), True),
        (("src", "keyboard.py"), False),
    ],
)
def test_secret_like(parts, secret):
    assert files.secret_like(parts) is secret


@pytest.mark.parametrize(
    "name",
    [
        "terraform.tfstate.backup",
        "prod.tfvars.json",
        "my-adminsdk-x1.json",
        "deploy-key.json",
        "deploy_key.json",
        "credentials.yaml",
        "credentials",
        "client_secret_123.apps.json",
        "serviceAccount.json",
        "ServiceAccountKey.JSON",
    ],
)
def test_more_secret_file_names(name):
    assert files.secret_like(("cfg", name))
