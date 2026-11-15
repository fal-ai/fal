import re

import pytest

from fal.cli.main import main


def _help(argv, capfd):
    try:
        main(["dev", *argv, "--help"])
    except SystemExit:
        pass
    return " ".join(capfd.readouterr().out.split())


def test_hot_help_is_a_short_summary_with_an_example(capfd):
    try:
        main(["dev", "hot", "--help"])
    except SystemExit:
        pass
    raw = capfd.readouterr().out
    text = " ".join(raw.split())

    assert "def load() -> state" in text and "def infer(state, **params)" in text
    assert "fal dev hot start --help" in text
    assert "fal dev hot sync --watch --run" in text
    assert "torch 2.8.0" not in text
    assert len(re.split("options:", raw, flags=re.I)[0].splitlines()) < 16


@pytest.mark.parametrize("argv", [["hot", "start"]])
def test_hot_start_help_documents_the_contract_and_environment(argv, capfd):
    text = _help(argv, capfd)

    assert "def load() -> state" in text
    assert "def infer(state, **params)" in text
    assert "PIL image" in text and "bytes" in text and "JSON-serializable" in text
    assert "torch 2.8.0" not in text and "diffusers 0.35.1" not in text
    assert "requirements.txt is installed with uv into <project>/.venv" in text
    assert "pinned to the base image's versions" in text
    assert "rebuilt whenever the file changes" in text


def test_hot_restart_help_says_when_to_use_it(capfd):
    assert "Use it after" in _help(["hot", "restart"], capfd)
