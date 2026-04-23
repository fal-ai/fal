from __future__ import annotations

import isolate_proto


def test_proto():
    test_mod = isolate_proto.MachineRequirements(machine_type="XL")
    assert test_mod.machine_type == "XL"


def test_callable_entrypoint_fields():
    serialized_function = isolate_proto.SerializedObject(
        method="pickle",
        definition=b"callable",
    )

    hosted_map = isolate_proto.HostedMap(
        entrypoint="pkg.mod:func",
        run_on_main_thread=True,
    )
    assert hosted_map.WhichOneof("callable") == "entrypoint"
    assert hosted_map.entrypoint == "pkg.mod:func"
    assert hosted_map.run_on_main_thread is True

    hosted_map_with_function = isolate_proto.HostedMap(function=serialized_function)
    assert hosted_map_with_function.WhichOneof("callable") == "function"
    assert hosted_map_with_function.function == serialized_function

    hosted_map_last_write_wins = isolate_proto.HostedMap(
        function=serialized_function,
        entrypoint="pkg.mod:func",
    )
    assert hosted_map_last_write_wins.WhichOneof("callable") == "entrypoint"
    assert hosted_map_last_write_wins.entrypoint == "pkg.mod:func"

    hosted_run = isolate_proto.HostedRun(
        entrypoint="pkg.mod:func",
        run_on_main_thread=True,
    )
    assert hosted_run.WhichOneof("callable") == "entrypoint"
    assert hosted_run.entrypoint == "pkg.mod:func"
    assert hosted_run.run_on_main_thread is True

    hosted_run_with_function = isolate_proto.HostedRun(function=serialized_function)
    assert hosted_run_with_function.WhichOneof("callable") == "function"
    assert hosted_run_with_function.function == serialized_function

    hosted_run_last_write_wins = isolate_proto.HostedRun(
        function=serialized_function,
        entrypoint="pkg.mod:func",
    )
    assert hosted_run_last_write_wins.WhichOneof("callable") == "entrypoint"
    assert hosted_run_last_write_wins.entrypoint == "pkg.mod:func"

    register_request = isolate_proto.RegisterApplicationRequest(
        entrypoint="pkg.mod:App.run",
        run_on_main_thread=True,
    )
    assert register_request.WhichOneof("callable") == "entrypoint"
    assert register_request.entrypoint == "pkg.mod:App.run"
    assert register_request.run_on_main_thread is True

    register_request_with_function = isolate_proto.RegisterApplicationRequest(
        function=serialized_function
    )
    assert register_request_with_function.WhichOneof("callable") == "function"
    assert register_request_with_function.function == serialized_function

    register_request_last_write_wins = isolate_proto.RegisterApplicationRequest(
        function=serialized_function,
        entrypoint="pkg.mod:App.run",
    )
    assert register_request_last_write_wins.WhichOneof("callable") == "entrypoint"
    assert register_request_last_write_wins.entrypoint == "pkg.mod:App.run"


def test_register_application_private_logs_presence():
    request_without_private_logs = isolate_proto.RegisterApplicationRequest()
    assert request_without_private_logs.HasField("private_logs") is False

    request_with_private_logs_false = isolate_proto.RegisterApplicationRequest(
        private_logs=False
    )
    assert request_with_private_logs_false.HasField("private_logs") is True
    assert request_with_private_logs_false.private_logs is False

    request_with_private_logs_true = isolate_proto.RegisterApplicationRequest(
        private_logs=True
    )
    assert request_with_private_logs_true.HasField("private_logs") is True
    assert request_with_private_logs_true.private_logs is True
