import runpy

from fal.dev.box import BOX_APP, REMOTE


def test_a_resync_of_unchanged_files_never_restarts_the_hot_server(tmp_path):
    """`hot sync --watch` pushes every file first; load.py must not restart it."""
    server = runpy.run_path(str(REMOTE / "hot_server.py"))
    box_app = runpy.run_path(str(BOX_APP))
    box_app["_hot_dir"].__globals__["LOCAL_WORK"] = tmp_path
    project = tmp_path / "demo"
    project.mkdir()
    sources = {"load.py": "def load(): ...", "infer.py": "def infer(state): ..."}
    for name, text in sources.items():
        (project / name).write_text(text)
    running = server["Project"](project)

    result = box_app["_hot_write_files"]({"dir": str(project), "files": sources})

    assert result == {"written": []}
    assert running.check() is None
    sources["load.py"] = "def load(): return 1"
    box_app["_hot_write_files"]({"dir": str(project), "files": sources})
    assert running.check() == "load.py changed"


def test_load_failure_names_the_innermost_exception():
    server = runpy.run_path(str(REMOTE / "hot_server.py"))
    try:
        try:
            raise RuntimeError("operator torchvision::nms does not exist")
        except RuntimeError as inner:
            raise ModuleNotFoundError(
                "Could not import module 'PreTrainedModel'"
            ) from inner
    except ModuleNotFoundError as exc:
        text = server["innermost"](exc)

    assert text == "RuntimeError: operator torchvision::nms does not exist"
