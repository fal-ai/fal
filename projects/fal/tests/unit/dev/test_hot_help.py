import pytest

from fal.cli.main import main


@pytest.mark.parametrize("argv", [["hot"], ["hot", "start"]])
def test_hot_help_documents_the_contract_and_environment(argv, capfd):
    try:
        main(["dev", *argv, "--help"])
    except SystemExit:
        pass
    text = " ".join(capfd.readouterr().out.split())

    assert "def load() -> state" in text
    assert "def infer(state, **params)" in text
    assert "PIL image" in text and "bytes" in text and "JSON-serializable" in text
    assert "torch 2.8.0" in text and "torchvision 0.23.0" in text
    assert "requirements.txt is installed with uv into <project>/.venv" in text
    assert "pinned to the base image's versions" in text
    assert "rebuilt whenever the file changes" in text
