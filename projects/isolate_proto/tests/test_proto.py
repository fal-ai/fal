from __future__ import annotations

import isolate_proto
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory


def test_proto():
    test_mod = isolate_proto.MachineRequirements(machine_type="XL")
    assert test_mod.machine_type == "XL"


def test_machine_requirements_metrics_port_presence():
    requirements = isolate_proto.MachineRequirements()
    assert requirements.HasField("metrics_port") is False

    requirements.metrics_port = 9090
    assert requirements.HasField("metrics_port") is True
    assert requirements.metrics_port == 9090


def test_callable_entrypoint_fields():
    serialized_function = isolate_proto.SerializedObject(
        method="pickle",
        definition=b"callable",
    )

    hosted_map_without_run_on_main_thread = isolate_proto.HostedMap(
        entrypoint="pkg.mod:func"
    )
    assert (
        hosted_map_without_run_on_main_thread.HasField("run_on_main_thread") is False
    )

    hosted_map_with_run_on_main_thread_false = isolate_proto.HostedMap(
        entrypoint="pkg.mod:func",
        run_on_main_thread=False,
    )
    assert (
        hosted_map_with_run_on_main_thread_false.HasField("run_on_main_thread") is True
    )
    assert hosted_map_with_run_on_main_thread_false.run_on_main_thread is False

    hosted_map = isolate_proto.HostedMap(
        entrypoint="pkg.mod:func",
        run_on_main_thread=True,
    )
    assert hosted_map.WhichOneof("callable") == "entrypoint"
    assert hosted_map.entrypoint == "pkg.mod:func"
    assert hosted_map.HasField("run_on_main_thread") is True
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

    hosted_run_without_run_on_main_thread = isolate_proto.HostedRun(
        entrypoint="pkg.mod:func"
    )
    assert (
        hosted_run_without_run_on_main_thread.HasField("run_on_main_thread") is False
    )

    hosted_run_with_run_on_main_thread_false = isolate_proto.HostedRun(
        entrypoint="pkg.mod:func",
        run_on_main_thread=False,
    )
    assert (
        hosted_run_with_run_on_main_thread_false.HasField("run_on_main_thread") is True
    )
    assert hosted_run_with_run_on_main_thread_false.run_on_main_thread is False

    hosted_run = isolate_proto.HostedRun(
        entrypoint="pkg.mod:func",
        run_on_main_thread=True,
    )
    assert hosted_run.WhichOneof("callable") == "entrypoint"
    assert hosted_run.entrypoint == "pkg.mod:func"
    assert hosted_run.HasField("run_on_main_thread") is True
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

    register_request_without_run_on_main_thread = (
        isolate_proto.RegisterApplicationRequest(entrypoint="pkg.mod:App.run")
    )
    assert (
        register_request_without_run_on_main_thread.HasField("run_on_main_thread")
        is False
    )

    register_request_with_run_on_main_thread_false = (
        isolate_proto.RegisterApplicationRequest(
            entrypoint="pkg.mod:App.run",
            run_on_main_thread=False,
        )
    )
    assert (
        register_request_with_run_on_main_thread_false.HasField(
            "run_on_main_thread"
        )
        is True
    )
    assert (
        register_request_with_run_on_main_thread_false.run_on_main_thread is False
    )

    register_request = isolate_proto.RegisterApplicationRequest(
        entrypoint="pkg.mod:App.run",
        run_on_main_thread=True,
    )
    assert register_request.WhichOneof("callable") == "entrypoint"
    assert register_request.entrypoint == "pkg.mod:App.run"
    assert register_request.HasField("run_on_main_thread") is True
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


def test_volume_fields_are_protobuf_compatible_with_old_readers():
    file_descriptor = descriptor_pb2.FileDescriptorProto(
        name="legacy_volume_compat.proto",
        package="legacy_volume_compat",
        syntax="proto3",
    )
    for message_name, known_field_name, known_field_number in (
        ("HostedRun", "environment_name", 6),
        ("RegisterApplicationRequest", "application_name", 5),
    ):
        message = file_descriptor.message_type.add(name=message_name)
        message.field.add(
            name=known_field_name,
            number=known_field_number,
            label=descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL,
            type=descriptor_pb2.FieldDescriptorProto.TYPE_STRING,
        )

    pool = descriptor_pool.DescriptorPool()
    pool.Add(file_descriptor)
    legacy_run_type = message_factory.GetMessageClass(
        pool.FindMessageTypeByName("legacy_volume_compat.HostedRun")
    )
    legacy_registration_type = message_factory.GetMessageClass(
        pool.FindMessageTypeByName(
            "legacy_volume_compat.RegisterApplicationRequest"
        )
    )

    run = isolate_proto.HostedRun(
        environment_name="main",
        volumes=[isolate_proto.VolumeMount(volume_name="data", mount_path="/data")],
    )
    registration = isolate_proto.RegisterApplicationRequest(
        application_name="legacy",
        volumes=[isolate_proto.VolumeMount(volume_name="data", mount_path="/data")],
    )

    legacy_run = legacy_run_type.FromString(run.SerializeToString())
    legacy_registration = legacy_registration_type.FromString(
        registration.SerializeToString()
    )

    assert legacy_run.environment_name == "main"
    assert legacy_registration.application_name == "legacy"

    current_run = isolate_proto.HostedRun.FromString(
        legacy_run_type(environment_name="main").SerializeToString()
    )
    current_registration = isolate_proto.RegisterApplicationRequest.FromString(
        legacy_registration_type(application_name="legacy").SerializeToString()
    )

    assert list(current_run.volumes) == []
    assert list(current_registration.volumes) == []


def test_build_environment_field_presence():
    hosted_run = isolate_proto.HostedRun(entrypoint="pkg.mod:func")
    assert hosted_run.HasField("build_environment") is False

    hosted_run_skip = isolate_proto.HostedRun(
        entrypoint="pkg.mod:func", build_environment=False
    )
    assert hosted_run_skip.HasField("build_environment") is True
    assert hosted_run_skip.build_environment is False

    register = isolate_proto.RegisterApplicationRequest()
    assert register.HasField("build_environment") is False

    register_skip = isolate_proto.RegisterApplicationRequest(build_environment=False)
    assert register_skip.HasField("build_environment") is True
    assert register_skip.build_environment is False


def test_hosted_run_health_check_config_presence():
    hosted_run = isolate_proto.HostedRun(entrypoint="pkg.mod:func")
    assert hosted_run.HasField("health_check_config") is False

    hosted_run_with_health_check = isolate_proto.HostedRun(
        entrypoint="pkg.mod:func",
        health_check_config=isolate_proto.ApplicationHealthCheckConfig(
            path="/ready",
            method=isolate_proto.ApplicationHealthCheckConfig.GET,
        ),
    )
    assert hosted_run_with_health_check.HasField("health_check_config") is True
    assert hosted_run_with_health_check.health_check_config.path == "/ready"
    assert (
        hosted_run_with_health_check.health_check_config.method
        == isolate_proto.ApplicationHealthCheckConfig.GET
    )


def test_env_id_field_presence_on_run_and_register():
    hosted_run = isolate_proto.HostedRun(entrypoint="pkg.mod:func")
    assert hosted_run.HasField("env_id") is False

    hosted_run_with_env = isolate_proto.HostedRun(
        entrypoint="pkg.mod:func", env_id="abc123"
    )
    assert hosted_run_with_env.HasField("env_id") is True
    assert hosted_run_with_env.env_id == "abc123"

    register = isolate_proto.RegisterApplicationRequest()
    assert register.HasField("env_id") is False

    register_with_env = isolate_proto.RegisterApplicationRequest(env_id="abc123")
    assert register_with_env.HasField("env_id") is True
    assert register_with_env.env_id == "abc123"

    run_result = isolate_proto.HostedRunResult(run_id="r1")
    assert run_result.HasField("env_id") is False
    run_result_with_env = isolate_proto.HostedRunResult(run_id="r1", env_id="abc123")
    assert run_result_with_env.env_id == "abc123"

    register_result = isolate_proto.RegisterApplicationResult()
    assert register_result.HasField("env_id") is False
    register_result_with_env = isolate_proto.RegisterApplicationResult(env_id="abc123")
    assert register_result_with_env.env_id == "abc123"


def test_build_environment_request_construction():
    request = isolate_proto.BuildEnvironmentRequest(
        environment_name="main",
        application_name="my-app",
    )
    assert request.environment_name == "main"
    assert request.application_name == "my-app"


def test_build_environment_result_construction():
    result = isolate_proto.BuildEnvironmentResult()
    assert result.HasField("status") is False
    assert result.env_id == ""
    assert list(result.logs) == []

    result_with_status = isolate_proto.BuildEnvironmentResult(
        status=isolate_proto.HostedRunStatus(
            state=isolate_proto.RunState.SUCCESS,
        ),
        env_id="abc123",
    )
    assert result_with_status.HasField("status") is True
    assert result_with_status.status.state == isolate_proto.RunState.SUCCESS
    assert result_with_status.env_id == "abc123"


def test_shell_runner_tty_presence_survives_serialization():
    legacy_request = isolate_proto.ShellRunnerInput(runner_id="runner-id")
    assert legacy_request.HasField("tty") is False

    request = isolate_proto.ShellRunnerInput(runner_id="runner-id", tty=False)
    restored = isolate_proto.ShellRunnerInput.FromString(request.SerializeToString())

    assert restored.HasField("tty") is True
    assert restored.tty is False


def test_shell_runner_output_stream_presence_survives_serialization():
    legacy_output = isolate_proto.ShellRunnerOutput(data=b"output")
    assert legacy_output.HasField("stream") is False

    output = isolate_proto.ShellRunnerOutput(data=b"error", stream=2)
    restored = isolate_proto.ShellRunnerOutput.FromString(output.SerializeToString())

    assert restored.HasField("stream") is True
    assert restored.stream == 2


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


def test_create_user_key_v2_policy_fields():
    request = isolate_proto.CreateUserKeyRequest(alias="ci")
    assert request.HasField("scope") is False
    assert request.HasField("policy_preset") is False
    assert request.HasField("policy") is False

    admin_request = isolate_proto.CreateUserKeyRequest(
        scope=isolate_proto.CreateUserKeyRequest.ADMIN
    )
    round_tripped = isolate_proto.CreateUserKeyRequest.FromString(
        admin_request.SerializeToString()
    )
    assert round_tripped.HasField("scope") is True
    assert round_tripped.scope == isolate_proto.CreateUserKeyRequest.ADMIN

    preset_request = isolate_proto.CreateUserKeyRequest(policy_preset="FULL")
    assert preset_request.HasField("policy_preset") is True
    assert preset_request.policy_preset == "FULL"

    policy_request = isolate_proto.CreateUserKeyRequest(
        policy=isolate_proto.KeyPolicy(permissions=["serverless:apps:run"])
    )
    assert policy_request.HasField("policy") is True
    assert list(policy_request.policy.permissions) == ["serverless:apps:run"]
