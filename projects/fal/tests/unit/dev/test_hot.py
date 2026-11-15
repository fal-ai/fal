import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from dev.fakes import make_box, make_runner
from fal.dev import config, hot, state
from fal.dev.errors import DevError


def test_parse_params():
    params = hot.parse_params(["steps=4", "name=fox", "size=[1, 2]"], "a fox")

    assert params == {"steps": 4, "name": "fox", "size": [1, 2], "prompt": "a fox"}
    with pytest.raises(DevError):
        hot.parse_params(["steps"], None)


def test_remote_dir(tmp_path):
    assert hot.remote_dir(None, "/work/x") == "/work/x"
    assert hot.remote_dir(str(tmp_path / "demo"), None) == "/work/demo"
    with pytest.raises(DevError):
        hot.remote_dir(None, None)


def test_scan_skips_hidden_venv_and_large_files(tmp_path):
    (tmp_path / "infer.py").write_text("x")
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "util.py").write_text("x")
    for skipped in (".venv", "hot-out", ".git"):
        (tmp_path / skipped).mkdir()
        (tmp_path / skipped / "f.py").write_text("x")
    (tmp_path / ".env").write_text("x")
    (tmp_path / "big.bin").write_bytes(b"x" * (hot.MAX_FILE_BYTES + 1))

    assert sorted(hot.scan(tmp_path)) == ["infer.py", "pkg/util.py"]


def test_scan_never_picks_up_secrets(tmp_path):
    (tmp_path / "infer.py").write_text("x")
    (tmp_path / "secrets.py").write_text("x")  # code, not a secret file
    (tmp_path / "keys").mkdir()
    for secret in (
        "server.pem",
        "credentials.json",
        "id_rsa",
        "secrets.yaml",
        "keys/token.json",
        ".env",
        ".env.production",
        "prod.env",
        "secrets/api.txt",
        "vars.tfvars",
        "state.tfstate",
        "service_account_prod.json",
    ):
        path = tmp_path / secret
        path.parent.mkdir(exist_ok=True)
        path.write_text("secret")

    withheld: list[Path] = []
    assert sorted(hot.scan(tmp_path, withheld)) == ["infer.py", "secrets.py"]
    assert "server.pem" in {p.as_posix() for p in withheld}


def test_read_token_uses_cache_for_same_runner(monkeypatch, tmp_path):
    monkeypatch.setattr(state, "STATE_DIR", tmp_path)
    box = make_box([make_runner("r1")])
    token = "a" * 43
    box.run = MagicMock(return_value=(0, f"{token}\n"))

    assert hot.read_token(box, "r1") == token
    assert hot.read_token(box, "r1") == token
    box.run.assert_called_once()

    box.run.return_value = (0, "cat: no such file\n")
    with pytest.raises(DevError, match="redeploy"):
        hot.read_token(box, "r2")


def test_project_paths_keeps_remembered_dir_for_same_project(tmp_path):
    a, b = tmp_path / "model-a", tmp_path / "model-b"
    remembered = {"hot_local": str(a.resolve()), "hot_dir": "/work/custom"}

    assert hot.project_paths(None, None, remembered) == (
        str(a.resolve()),
        "/work/custom",
    )
    assert hot.project_paths(str(a), None, remembered)[1] == "/work/custom"
    # A different project gets its own folder, not the remembered one.
    assert hot.project_paths(str(b), None, remembered) == (
        str(b.resolve()),
        "/work/model-b",
    )
    assert hot.project_paths(str(b), "/work/x", remembered)[1] == "/work/x"


def _hot_box_with_auth(auth):
    box = hot.HotBox.__new__(hot.HotBox)
    box.base, box.token = "https://fal.run/me/devbox-demo/hot", "tok"
    box.box = MagicMock(auth_headers=auth)
    return box


def test_hot_box_posts_with_its_token_and_reports_failures_as_no_response(monkeypatch):
    import httpx

    calls = []

    def fake_post_json(url, payload, headers, timeout):
        calls.append((url, headers))
        if len(calls) == 2:
            raise httpx.ConnectError("down")
        return httpx.Response(409, headers={"X-Hot-Code-Version": "3"}, content=b"{}")

    monkeypatch.setattr(hot, "post_json", fake_post_json)
    box = _hot_box_with_auth(lambda: {"Authorization": "Bearer t"})

    status, headers, body = box.post("status")
    assert (status, headers["x-hot-code-version"], body) == (409, "3", b"{}")
    assert calls[0] == (
        "https://fal.run/me/devbox-demo/hot/status",
        {"Authorization": "Bearer t", config.HOT_TOKEN_HEADER: "tok"},
    )
    status, _, body = box.post("status")
    assert (status, json.loads(body)) == (0, {"error": "down"})


def test_a_credential_failure_is_retried_by_watch_and_wait_ready(monkeypatch, tmp_path):
    import httpx

    monkeypatch.setattr(hot.time, "sleep", lambda s: None)
    attempts = []

    def auth():
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("login expired")
        return {}

    monkeypatch.setattr(
        hot,
        "post_json",
        lambda url, payload, headers, timeout: httpx.Response(
            200, json={"ready": True, "load_seconds": 1, "written": ["infer.py"]}
        ),
    )
    box = _hot_box_with_auth(auth)

    assert box.wait_ready(timeout=60)["load_seconds"] == 1
    assert len(attempts) == 2

    # A push that failed is sent again with the next save.
    (tmp_path / "infer.py").write_text("x")
    attempts.clear()
    with pytest.raises(DevError, match="sync failed"):
        box.push(tmp_path, "/work/x", ["infer.py"])
    assert box.push(tmp_path, "/work/x", ["infer.py"]) == ["infer.py"]


def _hot_box(responses):
    box = hot.HotBox.__new__(hot.HotBox)
    box.post = MagicMock(side_effect=responses)
    return box


def test_request_reports_a_stopped_server_instead_of_waiting():
    body = json.dumps(
        {"error": "the hot server is not running (fal dev hot start), or is restarting"}
    ).encode()
    box = _hot_box([(409, {}, body)])
    box.wait_ready = MagicMock()

    assert box.request({}) is False
    box.wait_ready.assert_not_called()


def test_request_waits_through_a_full_restart(tmp_path):
    restarting = b'{"restarting": "load.py changed"}'
    box = _hot_box(
        [(409, {}, restarting), (200, {"content-type": "application/json"}, b"{}")]
    )
    box.wait_ready = MagicMock(return_value={"load_seconds": 1.0})

    assert box.request({}) is True
    box.wait_ready.assert_called_once()


def test_wait_ready_rides_out_a_network_blip(monkeypatch):
    monkeypatch.setattr(hot.time, "sleep", lambda s: None)
    ready = json.dumps({"ready": True, "load_seconds": 2}).encode()
    box = _hot_box([(0, {}, b'{"error": "timed out"}'), (200, {}, ready)])

    assert box.wait_ready(timeout=60)["load_seconds"] == 2


def test_wait_ready_stops_when_the_server_waits_for_a_fix(monkeypatch):
    monkeypatch.setattr(hot.time, "sleep", lambda s: None)
    log_rows = [
        "00:00:01 [hot] loading the model (load.load()) ...",
        "00:00:02 [hot] load.load() failed; fix load.py and save to retry:",
        "Traceback (most recent call last):",
        *[f'  File "load.py", line {i}' for i in range(10)],
        "ModuleNotFoundError: No module named 'missing'",
    ]
    stuck = json.dumps({"error": "not running", "log": log_rows}).encode()
    box = _hot_box([(409, {}, stuck)])

    with pytest.raises(DevError, match="stopped on an error"):
        box.wait_ready(timeout=60)


def _said(monkeypatch) -> list[str]:
    said: list[str] = []
    wait = MagicMock()
    wait.say.side_effect = said.append
    loading = MagicMock()
    loading.return_value.__enter__.return_value = wait
    monkeypatch.setattr(hot, "Loading", loading)
    return said


def test_wait_ready_hides_blank_and_progress_lines_until_the_load_fails(monkeypatch):
    monkeypatch.setattr(hot.time, "sleep", lambda s: None)
    loading = "00:00:01 [hot] loading the model (load.load()) ..."
    bar = "Loading weights:  50%|#####     | 199/398"
    loaded = "00:00:09 [hot] model loaded in 8.2s"
    file_tail = json.dumps({"log": [loading, "", bar, ""]}).encode()
    # The restarted server answers with its own log, which has no blank lines.
    ready = json.dumps(
        {"ready": True, "load_seconds": 8.2, "log": [loading, loaded]}
    ).encode()
    said = _said(monkeypatch)
    _hot_box([(409, {}, file_tail), (200, {}, ready)]).wait_ready(timeout=60)
    assert said == [f"  {loading}", f"  {loaded}"]

    failed = "00:00:02 [hot] load.load() failed; fix load.py and save to retry"
    stuck = json.dumps({"log": [loading, bar, "", failed]}).encode()
    said = _said(monkeypatch)
    with pytest.raises(DevError, match="stopped on an error"):
        _hot_box([(409, {}, stuck)]).wait_ready(timeout=60)
    assert said == [f"  {line}" for line in (loading, bar, "", failed)]


def test_waiting_for_fix_ignores_a_failure_followed_by_a_retry():
    assert not hot.waiting_for_fix(
        ["[hot] load.load() failed", "[hot] loading the model"]
    )
    assert hot.waiting_for_fix(["[hot] restarting", "[hot] server exited with code 1"])
    assert not hot.waiting_for_fix(["[hot] loading the model (load.load()) ..."])


def test_new_log_lines_prints_each_line_once():
    assert hot.new_log_lines([], ["a", "b"]) == ["a", "b"]
    assert hot.new_log_lines(["a", "b"], ["a", "b"]) == []
    assert hot.new_log_lines(["a", "b"], ["a", "b", "c"]) == ["c"]
    # A sliding tail: the oldest line dropped off the front.
    assert hot.new_log_lines(["a", "b", "c"], ["b", "c", "d"]) == ["d"]
    # A different log (the server restarted): print all of it.
    assert hot.new_log_lines(["a", "b"], ["x", "y"]) == ["x", "y"]


def test_new_log_lines_reprints_only_a_grown_last_line():
    loading = "17:11:59 [hot] loading the model (load.load()) ..."
    warning = "  deprecate('torch_dtype', '1.0.0', _TORCH_DTYPE_DEPRECATION_MESSAGE)"
    bar = "Loading pipeline components...:  80%|########  | 4/5"
    printed = [loading, warning, bar]
    # The bar line was read mid-write; a warning then landed on the same line.
    grown = bar + "The config attributes {'force_upcast': True} were passed"
    lines = [loading, warning, grown, "model loaded in 8.2s"]
    assert hot.new_log_lines(printed, lines) == [grown, "model loaded in 8.2s"]
    # The same while the tail slides.
    assert hot.new_log_lines(printed, lines[1:]) == [grown, "model loaded in 8.2s"]
    # The restarted server answers with its own log, without the file's extra lines.
    loaded = "17:16:26 [hot] model loaded in 8.5s"
    file_tail = [loading, warning, "", bar, ""]
    assert hot.new_log_lines(file_tail, [loading, loaded]) == [loaded]
    # An exact match still wins over skipping lines already printed.
    assert hot.new_log_lines(["a", "b"], ["a", "b", "b"]) == ["b"]
    assert hot.new_log_lines(["a", "b"], ["a", "c"]) == ["c"]


def test_restarting_reads_the_log_order():
    serving = "00:00:03 [hot] serving /work/demo on 127.0.0.1:7860"
    restart = "00:00:09 [hot] FULL RESTART: load.py changed; the weights reload"
    loading = "00:00:01 [hot] loading the model (load.load()) ..."

    assert hot.restarting([loading, serving, restart])
    # Stopped after serving: nothing to wait for.
    assert not hot.restarting([loading, serving])
    assert not hot.restarting([restart, "[hot] load.load() failed; fix load.py"])


def test_request_waits_when_the_server_already_exited_to_restart():
    log_rows = ["[hot] serving /work/demo", "[hot] FULL RESTART: load.py changed"]
    closed = json.dumps({"error": "not running", "log": log_rows}).encode()
    box = _hot_box([(409, {}, closed), (200, {"content-type": "text/plain"}, b"ok")])
    box.wait_ready = MagicMock(return_value={"load_seconds": 1.0})

    assert box.request({}) is True
    box.wait_ready.assert_called_once()


def test_push_reports_a_restarted_box(tmp_path):
    (tmp_path / "infer.py").write_text("x")
    box = _hot_box([(403, {}, b'{"error": "bad or missing hot token"}')])

    with pytest.raises(hot.HotTokenRejected, match="restarted"):
        box.push(tmp_path, "/work/x", ["infer.py"])


def test_push_leaves_out_a_file_that_became_a_symlink(tmp_path):
    (tmp_path / "infer.py").write_text("x")
    (tmp_path / "other.py").write_text("y")
    outside = tmp_path.parent / f"{tmp_path.name}-secret.py"
    outside.write_text("secret")
    (tmp_path / "other.py").unlink()
    (tmp_path / "other.py").symlink_to(outside)
    box = _hot_box([(200, {}, b'{"written": ["infer.py"]}')])

    box.push(tmp_path, "/work/x", ["infer.py", "other.py"])

    assert list(box.post.call_args.args[1]["files"]) == ["infer.py"]


def test_wait_shows_only_the_current_attempt_without_progress_bars():
    log = [
        "10:14:36 [hot] infer() raised:",
        "NameError: name 'sepia' is not defined",
        "10:15:45 [hot] FULL RESTART: restart requested; the weights reload",
        "Loading weights: 100%|##########| 398/398 [00:00<00:00, 8386.33it/s]",
        "10:15:54 [hot] model loaded in 8.2s",
    ]
    assert hot.attempt_lines(log) == log[2:]
    shown = [line for line in hot.attempt_lines(log) if not hot.PROGRESS.search(line)]
    assert shown == [log[2], log[4]]
    assert hot.attempt_lines(["a", "b"]) == ["a", "b"]


def _waiting_box(monkeypatch, answers):
    box = hot.HotBox.__new__(hot.HotBox)
    calls = iter(answers)
    monkeypatch.setattr(box, "json", lambda action, payload=None: next(calls))
    monkeypatch.setattr(hot.time, "sleep", lambda s: None)
    return box


class _Said:
    def __init__(self):
        self.lines = []

    def say(self, line):
        self.lines.append(line.strip())


OLD = ["10:14 [hot] infer 728 ms", "10:15 [hot] FULL RESTART: restart requested"]


def test_wait_ready_shows_this_attempt_without_progress_bars(monkeypatch):
    bar = "Loading weights: 100%|##| 398/398 [00:00<00:00, 8386.33it/s]"
    box = _waiting_box(
        monkeypatch,
        [
            (409, {"log": OLD + [bar]}),
            (200, {"ready": True, "log": OLD + [bar, "10:16 [hot] model loaded"]}),
        ],
    )
    said = _Said()
    assert box._wait_ready(60, said)["ready"]
    assert said.lines == [OLD[1], "10:16 [hot] model loaded"]


def test_wait_ready_prints_the_whole_failed_attempt_then_raises(monkeypatch):
    bar = "  50%|#   | 2/4 [00:00<00:00, 11.8it/s]"
    failed = OLD + [bar, "Traceback ...", "[hot] load.load() failed"]
    box = _waiting_box(monkeypatch, [(409, {"log": failed})])
    said = _Said()
    with pytest.raises(DevError, match="stopped on an error"):
        box._wait_ready(60, said)
    assert said.lines == [line.strip() for line in failed[1:]]


def test_wait_ready_keeps_lines_of_an_attempt_that_ended_between_polls(monkeypatch):
    first = OLD + ["w1"]
    second = first + ["w2", "[hot] load.load() failed", "[hot] restarting", "n1"]
    box = _waiting_box(
        monkeypatch,
        [(409, {"log": first}), (409, {"log": second}), (200, {"ready": True})],
    )
    said = _Said()
    box._wait_ready(60, said)
    assert said.lines == [
        OLD[1],
        "w1",
        "w2",
        "[hot] load.load() failed",
        "[hot] restarting",
        "n1",
    ]


def test_reload_note_classifies_like_the_server():
    assert hot.reload_note(["infer.py"]) == (
        "infer.py changed: code reload, weights stay loaded"
    )
    assert hot.reload_note(["load.py"]) == "load.py changed: weights reload"
    assert "requirements reinstall, weights reload" in hot.reload_note(
        ["infer.py", "requirements.txt"]
    )
    # Only the project root counts, as in the server.
    assert "code reload" in hot.reload_note(["pkg/load.py"])
    assert "nothing reloads" in hot.reload_note(["config.json"])
    assert hot.weights_files(["a.py", "load.py", "sub/load.py"]) == ["load.py"]


def test_clean_log_collapses_progress_redraws_and_carriage_returns():
    bar = "Loading weights: {}%|##| {}/4 [00:00<00:00, 8.1it/s]"
    lines = [
        "[hot] loading",
        "",
        bar.format(25, 1),
        bar.format(50, 2),
        "old\rLoading: 100%|##| 4/4 [00:01<00:00, 3.0it/s]",
        "[hot] model loaded",
    ]
    assert hot.clean_log(lines) == [
        "[hot] loading",
        "Loading: 100%|##| 4/4 [00:01<00:00, 3.0it/s]",
        "[hot] model loaded",
    ]


def test_quiet_load_lines_drops_warnings_and_their_source_line():
    lines = [
        "[hot] loading the model",
        "/x/diffusers.py:3: FutureWarning: torch_dtype is deprecated",
        "  deprecate('torch_dtype')",
        "Note: HF_TOKEN is not set",
        "[hot] model loaded",
    ]
    assert hot.quiet_load_lines(lines) == [
        "[hot] loading the model",
        "[hot] model loaded",
    ]


def test_server_state_reads_status_and_log():
    assert hot.server_state(200, {"ready": True}) == ("ready", None)
    assert hot.server_state(200, {"restarting": "load.py changed"}) == (
        "restarting",
        "load.py changed",
    )
    serving = "[hot] serving /work/x"
    restart = "[hot] FULL RESTART: load.py changed; the weights reload"
    assert hot.server_state(409, {"log": [serving, restart]}) == (
        "restarting",
        "load.py changed",
    )
    assert hot.server_state(409, {"log": [serving]})[0] == "stopped"
    assert (
        hot.server_state(409, {"log": ["[hot] load.load() failed: x"]})[0] == "failed"
    )
    assert hot.server_state(0, {"error": "down"}) == ("unreachable", "down")


def test_code_version_explains_reloads():
    assert hot.code_version("0") == "code v0 (no reloads since weights loaded)"
    assert hot.code_version("1") == "code v1 (1 reload since weights loaded)"
    assert hot.code_version("3") == "code v3 (3 reloads since weights loaded)"


def test_run_result_error_is_plain_text_without_server_frames():
    tb = (
        "Traceback (most recent call last):\n"
        '  File "/usr/lib/hot_server.py", line 9, in run\n'
        "    result = self.infer(self.state, **params)\n"
        '  File "/work/klein/infer.py", line 14, in infer\n'
        "    image = sepia(image)\n"
        "NameError: name 'sepia' is not defined\n"
    )
    body = json.dumps({"error": "infer() raised", "traceback": tb}).encode()
    error = hot.RunResult(422, {}, body, 0.1, root="/work/klein").error
    assert error["message"] == (
        "infer() raised (server still running; fix and save to retry)"
    )
    assert "hot_server" not in error["traceback"]
    assert 'File "infer.py", line 14' in error["traceback"]
    stopped = json.dumps({"error": "the hot server is not running (x)"}).encode()
    assert (
        hot.RunResult(409, {}, stopped, 0.1)
        .error["message"]
        .startswith("the hot server is not running: start it")
    )
    assert (
        hot.RunResult(502, {}, b'{"detail": "bad gateway"}', 0).error["message"]
        == "bad gateway"
    )


def test_clean_traceback_in_the_server_module():
    from fal.dev.remote.hot_server import clean_traceback

    tb = (
        "Traceback (most recent call last):\n"
        '  File "/srv/hot_server.py", line 1, in run\n'
        "    x()\n"
        "    ^^^\n"
        '  File "/work/p/pkg/util.py", line 2, in f\n'
        "    1/0\n"
        "ZeroDivisionError: division by zero\n"
    )
    out = clean_traceback(tb, "/work/p")
    assert "hot_server" not in out and "x()" not in out
    assert 'File "pkg/util.py", line 2, in f' in out
    assert out.endswith("ZeroDivisionError: division by zero\n")
    assert clean_traceback(None) is None


def test_request_echoes_params_labels_timings_and_ends_with_the_path(capsys, tmp_path):
    box = _hot_box(
        [
            (
                200,
                {
                    "content-type": "image/png",
                    "x-hot-code-version": "1",
                    "x-hot-infer-ms": "571",
                },
                b"PNG",
            )
        ]
    )
    box.remote = None
    box.on_image = None
    out = tmp_path / "o.png"
    assert box.request({"prompt": "a red fox"}, str(out))
    captured = capsys.readouterr()
    assert '"prompt": "a red fox"' in captured.err
    lines = captured.out.splitlines()
    assert "code v1 (1 reload since weights loaded)" in lines[0]
    assert "(model only)" in lines[0] and "round trip" in lines[0]
    assert lines[-1] == f"saved {out}"


def test_a_failed_request_says_the_server_is_still_running(capsys):
    body = json.dumps({"error": "infer() raised", "traceback": "NameError: x\n"})
    box = _hot_box([(422, {}, body.encode())])
    box.remote = None

    assert box.request({}) is False
    err = capsys.readouterr().err
    assert "infer() raised (server still running; fix and save to retry)" in err
    assert "error (422)" not in err


def test_wait_ready_after_a_pid_ignores_the_old_server(monkeypatch):
    box = _waiting_box(
        monkeypatch,
        [
            (200, {"ready": True, "pid": 7}),
            (409, {"log": []}),
            (200, {"ready": True, "pid": 9}),
        ],
    )
    said = _Said()
    assert box._wait_ready(60, said, after_pid=7)["pid"] == 9


def test_a_push_of_load_py_remembers_the_serving_process(tmp_path):
    (tmp_path / "load.py").write_text("x")
    box = _hot_box(
        [
            (200, {}, b'{"ready": true, "pid": 5}'),
            (200, {}, b'{"written": ["load.py"]}'),
        ]
    )
    box.json = hot.HotBox.json.__get__(box)
    assert box.push(tmp_path, "/work/x", ["load.py"]) == ["load.py"]
    assert box.pid_before == 5
    box.post = MagicMock(return_value=(200, {}, b'{"written": ["infer.py"]}'))
    (tmp_path / "infer.py").write_text("x")
    box.push(tmp_path, "/work/x", ["infer.py"])
    assert box.pid_before is None


def _cli_args(argv):
    import io

    from rich.console import Console

    from fal.cli.main import parse_args

    args = parse_args(argv)
    args.console = Console(file=io.StringIO())
    err = io.StringIO()
    args.err = Console(file=err, width=200)
    return args, err


def test_hot_status_waits_briefly_then_shows_the_state_it_finds(monkeypatch):
    from fal.dev.cli import hot as cli_hot

    fake = MagicMock()
    fake.json.side_effect = [
        (409, {"restarting": "load.py changed"}),
        (409, {"restarting": "load.py changed"}),
    ]
    fake.wait_ready.side_effect = DevError("timed out waiting for the hot server")
    args, _ = _cli_args(["dev", "hot", "status"])

    assert cli_hot._wait_if_restarting(args, fake) == (
        409,
        {"restarting": "load.py changed"},
    )
    assert fake.wait_ready.call_args.kwargs["timeout"] == 30


def test_watch_keeps_watching_after_a_broken_first_reload(monkeypatch):
    from fal.dev.cli import hot as cli_hot

    fake = MagicMock()
    fake.wait_reload.side_effect = DevError("the hot server stopped on an error")
    fake.json.return_value = (200, {})
    monkeypatch.setattr(cli_hot.hot, "HotBox", lambda box: fake)
    monkeypatch.setattr(cli_hot, "get_box", lambda args: object())
    monkeypatch.setattr(cli_hot.preview, "HotPreview", lambda args: None)
    monkeypatch.setattr(cli_hot, "_project", lambda args: ("/l", "/r"))
    monkeypatch.setattr(cli_hot, "_push", lambda *a: [])
    monkeypatch.setattr(cli_hot, "LoopView", lambda *a, **k: "view")
    args, err = _cli_args(["dev", "hot", "sync", "--watch"])

    assert cli_hot.cmd_sync(args) == 0
    fake.watch.assert_called_once()
    assert "stopped on an error; fix it and save again" in err.getvalue()
