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


def test_container_config_construction_and_presence():
    assert isolate_proto.HostedRun().HasField("container_config") is False
    assert (
        isolate_proto.RegisterApplicationRequest().HasField("container_config")
        is False
    )

    container_config = isolate_proto.ContainerConfig(
        dockerfile_str="FROM python:3.12-slim\n",
        build_args=[
            isolate_proto.ContainerBuildArg(name="FOO", value="bar"),
        ],
        registries=[
            isolate_proto.ContainerRegistry(
                registry="registry.example.com",
                username="user",
                password="pass",
            )
        ],
        builder="depot",
        compression="gzip",
        force_compression=False,
        secrets=[
            isolate_proto.ContainerSecret(name="TOKEN", value="secret"),
        ],
        docker_context_dir="/workspace/app",
        docker_files_list=["src/", "requirements.txt"],
        docker_ignore=["\\.git", "__pycache__"],
        entrypoint=isolate_proto.ContainerCommand(
            argv=isolate_proto.ContainerCommandArgs(
                args=["python", "-m", "server"],
            )
        ),
        cmd=isolate_proto.ContainerCommand(shell="--host 0.0.0.0 --port 8080"),
        use_isolate=False,
    )

    assert container_config.dockerfile_str == "FROM python:3.12-slim\n"
    assert list(container_config.build_args) == [
        isolate_proto.ContainerBuildArg(name="FOO", value="bar"),
    ]
    assert list(container_config.registries) == [
        isolate_proto.ContainerRegistry(
            registry="registry.example.com",
            username="user",
            password="pass",
        )
    ]
    assert container_config.HasField("builder") is True
    assert container_config.builder == "depot"
    assert container_config.HasField("compression") is True
    assert container_config.compression == "gzip"
    assert container_config.HasField("force_compression") is True
    assert container_config.force_compression is False
    assert list(container_config.secrets) == [
        isolate_proto.ContainerSecret(name="TOKEN", value="secret"),
    ]
    assert container_config.HasField("docker_context_dir") is True
    assert container_config.docker_context_dir == "/workspace/app"
    assert list(container_config.docker_files_list) == ["src/", "requirements.txt"]
    assert list(container_config.docker_ignore) == ["\\.git", "__pycache__"]
    assert container_config.entrypoint.WhichOneof("value") == "argv"
    assert list(container_config.entrypoint.argv.args) == ["python", "-m", "server"]
    assert container_config.cmd.WhichOneof("value") == "shell"
    assert container_config.cmd.shell == "--host 0.0.0.0 --port 8080"
    assert container_config.HasField("use_isolate") is True
    assert container_config.use_isolate is False

    hosted_run = isolate_proto.HostedRun(
        entrypoint="pkg.mod:func",
        container_config=container_config,
    )
    assert hosted_run.HasField("container_config") is True
    assert hosted_run.container_config == container_config

    register = isolate_proto.RegisterApplicationRequest(
        entrypoint="pkg.mod:App.run",
        container_config=container_config,
    )
    assert register.HasField("container_config") is True
    assert register.container_config == container_config
