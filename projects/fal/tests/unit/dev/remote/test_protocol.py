from pathlib import Path

import pytest

from fal.dev.errors import DevError
from fal.dev.remote import protocol

pytestmark = pytest.mark.usefixtures("dev_account")


@pytest.mark.parametrize(
    "key,value",
    [("repeat", 1.5), ("repeat", True), ("warmups", -1), ("timeout", "soon")],
)
def test_request_numbers_are_checked(key, value):
    with pytest.raises(ValueError, match=key):
        protocol.number({key: value}, key, 1 if key != "timeout" else 1.0, 0, 100)


def test_session_env_never_overrides_what_the_sdk_reads():
    """Every FAL_ variable the SDK reads is refused, so new ones can't slip by."""
    import re

    import fal
    from fal.dev.remote.protocol import check_session_env

    names = set()
    for path in Path(fal.__file__).parent.rglob("*.py"):
        for line in path.read_text(errors="ignore").splitlines():
            if any(read in line for read in ("environ", "getenv", "bool_envvar(")):
                names.update(re.findall(r"[\"'](FAL_[A-Z0-9_]+)[\"']", line))
    assert names
    for name in sorted(names):
        with pytest.raises(DevError, match="cannot set"):
            check_session_env([name])


def test_redact_hides_credentials_and_keeps_refs():
    from fal.dev.remote.protocol import redact

    for url in (
        "https://user:ab/cd@host.example/simple",
        "https://user:p@ss@host.example/simple",
        "https://tok@host.example/simple",
    ):
        assert redact(f"failed for {url} twice") == (
            "failed for https://***@host.example/simple twice"
        )
    assert redact("http://u:tok@pypi:8080/simple") == "http://***@pypi:8080/simple"
    assert redact("see http://tok@pypi/simple.") == "see http://***@pypi/simple."
    assert redact("git+https://github.com/org/repo.git@v1.2.3") == (
        "git+https://github.com/org/repo.git@v1.2.3"
    )
    assert redact("http://pypi:8080/o.git@v1") == "http://pypi:8080/o.git@v1"
    # Only the userinfo goes; a git ref stays readable.
    assert redact("git+https://tok@github.com/o/r.git@v1") == (
        "git+https://***@github.com/o/r.git@v1"
    )
    assert redact("git+https://github.com/o/r.git@v1") == (
        "git+https://github.com/o/r.git@v1"
    )
    # Only the conventional ssh `git` user is kept.
    assert redact("ssh://ghp_xxx@host/r") == "ssh://***@host/r"
    assert redact("xssh://sekret@h") == "xssh://***@h"
    assert redact("https://git@host/r") == "https://***@host/r"
    assert redact("git+ssh://git@github.com/o/r.git@v1") == (
        "git+ssh://git@github.com/o/r.git@v1"
    )
    assert redact("ssh://u:pw@host/r.git") == "ssh://***@host/r.git"
    assert redact("https://h.example/x?a=1&token=abc&Password=p%20q") == (
        "https://h.example/x?a=1&token=***&Password=***"
    )
