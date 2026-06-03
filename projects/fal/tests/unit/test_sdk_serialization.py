from __future__ import annotations

import importlib
import pickle
import sys
import traceback

import isolate_proto
import pytest
from isolate.connections.common import ExceptionDeserializationError
from isolate.server.interface import from_grpc

from fal.sdk import _remote_error_summary

importlib.import_module("fal.sdk")  # register the from_grpc handlers


def _serialize_exception_from_temporary_module(tmp_path) -> tuple[str, bytes, str]:
    module_name = "remote_only_pkg_for_fal_test"
    module_path = tmp_path / f"{module_name}.py"
    module_path.write_text("class RemoteOnlyError(Exception):\n    pass\n")

    sys.path.insert(0, str(tmp_path))
    importlib.invalidate_caches()
    try:
        remote_module = importlib.import_module(module_name)
        try:
            raise remote_module.RemoteOnlyError("remote boom")
        except remote_module.RemoteOnlyError as exc:
            return module_name, pickle.dumps(exc), traceback.format_exc()
    finally:
        sys.modules.pop(module_name, None)
        sys.path.remove(str(tmp_path))
        importlib.invalidate_caches()


def test_remote_error_summary_extracts_root_cause_not_wrapper() -> None:
    # A chained traceback: the runner wraps the user's error. We want the root
    # cause (first block), not the outermost wrapper (last block).
    stringized = (
        "Traceback (most recent call last):\n"
        '  File "/app/handler.py", line 13, in boom\n'
        '    raise LocalEntryNotFoundError("test error")\n'
        "huggingface_hub.errors.LocalEntryNotFoundError: test error\n"
        "\n"
        "The above exception was the direct cause of the following exception:\n"
        "\n"
        "Traceback (most recent call last):\n"
        '  File "/app/fal_app.py", line 99, in run\n'
        "    result = user_fn()\n"
        "fal.api.api.UserFunctionException: Uncaught user function exception\n"
    )
    assert (
        _remote_error_summary(stringized)
        == "huggingface_hub.errors.LocalEntryNotFoundError: test error"
    )


def test_remote_error_summary_returns_none_without_exception_summary() -> None:
    assert _remote_error_summary(None) is None
    assert _remote_error_summary("") is None
    assert _remote_error_summary("no traceback header here") is None
    assert (
        _remote_error_summary(
            "Traceback (most recent call last):\n"
            '  File "/app/handler.py", line 13, in boom\n'
            "    result = user_fn()\n"
        )
        is None
    )


def test_hosted_run_result_recovers_remote_exception_identity(tmp_path) -> None:
    module_name, serialized, stringized = _serialize_exception_from_temporary_module(
        tmp_path
    )

    result = isolate_proto.HostedRunResult(
        run_id="run-1",
        return_value=isolate_proto.SerializedObject(
            method="pickle",
            definition=serialized,
            was_it_raised=True,
            stringized_traceback=stringized,
        ),
    )

    with pytest.raises(ExceptionDeserializationError) as exc_info:
        from_grpc(result)

    assert exc_info.value.original_traceback is not None
    assert exc_info.value.remote_error == f"{module_name}.RemoteOnlyError: remote boom"
