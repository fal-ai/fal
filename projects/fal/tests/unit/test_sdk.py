from unittest.mock import MagicMock, patch

import pytest

from fal.sdk import (
    ApplicationHealthCheckConfig,
    FalServerlessConnection,
    FalServerlessKeyCredentials,
    RetryConfig,
    get_credentials,
    isolate_proto,
    validate_retry_config_dict,
)


class RecordingStub:
    def __init__(self):
        self.register_request = None
        self.run_request = None

    def RegisterApplication(self, request):
        self.register_request = request
        return iter([])

    def Run(self, request):
        self.run_request = request
        return iter([])


def test_get_credentials_returns_key_credentials_when_available():
    with patch("fal.sdk.key_credentials", return_value=("key-id", "key-secret")):
        with patch("fal.config.Config") as mock_config_class:
            mock_config_class.return_value.get_internal.return_value = None
            credentials = get_credentials()

    assert isinstance(credentials, FalServerlessKeyCredentials)
    assert credentials.key_id == "key-id"
    assert credentials.key_secret == "key-secret"


def test_get_credentials_prints_when_team_differs_from_key_owner_nickname():
    message = (
        "Ignoring explicit team config-team because key is used. "
        "The key belongs to Test User: my-nickname - usr_123."
    )
    with patch("fal.sdk.key_credentials", return_value=("key-id", "key-secret")):
        with patch(
            "fal.sdk.current_user_info",
            return_value={
                "full_name": "Test User",
                "nickname": "my-nickname",
                "user_id": "usr_123",
            },
        ):
            with patch("fal.sdk.console.print") as mock_console_print:
                credentials = get_credentials(team="config-team")

    mock_console_print.assert_called_once_with(message)
    assert isinstance(credentials, FalServerlessKeyCredentials)


def test_get_credentials_does_not_print_when_team_matches_nickname():
    with patch("fal.sdk.key_credentials", return_value=("key-id", "key-secret")):
        with patch(
            "fal.sdk.current_user_info",
            return_value={
                "full_name": "Test User",
                "nickname": "my-team",
                "user_id": "usr_123",
            },
        ):
            with patch("fal.sdk.console.print") as mock_console_print:
                credentials = get_credentials(team="my-team")

    mock_console_print.assert_not_called()
    assert isinstance(credentials, FalServerlessKeyCredentials)


def test_get_credentials_raises_when_key_owner_lookup_fails():
    with patch("fal.sdk.key_credentials", return_value=("key-id", "key-secret")):
        with patch("fal.sdk.current_user_info", side_effect=RuntimeError("boom")):
            with pytest.raises(RuntimeError, match="boom"):
                get_credentials(team="my-team")


def test_get_credentials_passes_profile_to_key_credentials():
    with patch("fal.sdk.key_credentials", return_value=None) as mock_key_credentials:
        with patch("fal.config.Config") as mock_config_class:
            mock_config_class.return_value.get_internal.return_value = None
            get_credentials(profile="my-profile")

    mock_key_credentials.assert_called_once_with(profile="my-profile")


def test_connection_register_allows_no_isolate_container_without_callable():
    connection = FalServerlessConnection("api.alpha.fal.ai", MagicMock())
    stub = RecordingStub()
    connection._stub = stub  # type: ignore[assignment]
    environment = connection.define_environment(
        "container",
        image={"dockerfile_str": "FROM debian:bookworm-slim", "use_isolate": False},
    )

    list(
        connection.register(
            None,
            [environment],
            application_name="container-app",
            deployment_strategy="rolling",
        )
    )

    assert stub.register_request is not None
    assert stub.register_request.WhichOneof("callable") is None


def test_connection_register_forwards_container_image_reference():
    connection = FalServerlessConnection("api.alpha.fal.ai", MagicMock())
    stub = RecordingStub()
    connection._stub = stub  # type: ignore[assignment]
    environment = connection.define_environment(
        "container",
        image={
            "image": "ghcr.io/fal-ai/container-app:latest",
            "use_isolate": False,
        },
    )

    list(
        connection.register(
            None,
            [environment],
            application_name="container-app",
            deployment_strategy="rolling",
        )
    )

    assert stub.register_request is not None
    image = (
        stub.register_request.environments[0]
        .configuration.fields["image"]
        .struct_value.fields
    )
    assert image["image"].string_value == "ghcr.io/fal-ai/container-app:latest"
    assert image["use_isolate"].bool_value is False


def test_connection_register_forwards_health_check_method():
    connection = FalServerlessConnection("api.alpha.fal.ai", MagicMock())
    stub = RecordingStub()
    connection._stub = stub  # type: ignore[assignment]
    environment = connection.define_environment(
        "container",
        image={"dockerfile_str": "FROM debian:bookworm-slim", "use_isolate": False},
    )

    list(
        connection.register(
            None,
            [environment],
            application_name="container-app",
            deployment_strategy="rolling",
            health_check_config=ApplicationHealthCheckConfig(
                path="/health/ready",
                start_period_seconds=30,
                timeout_seconds=5,
                failure_threshold=3,
                call_regularly=True,
                method="get",
            ),
        )
    )

    assert stub.register_request is not None
    health_check_config = stub.register_request.health_check_config
    assert health_check_config.path == "/health/ready"
    assert health_check_config.method == isolate_proto.ApplicationHealthCheckConfig.GET


def test_connection_register_preserves_explicit_pure_docker_health_check_method():
    connection = FalServerlessConnection("api.alpha.fal.ai", MagicMock())
    stub = RecordingStub()
    connection._stub = stub  # type: ignore[assignment]
    environment = connection.define_environment(
        "container",
        image={"dockerfile_str": "FROM debian:bookworm-slim", "use_isolate": False},
    )

    list(
        connection.register(
            None,
            [environment],
            application_name="container-app",
            deployment_strategy="rolling",
            health_check_config=ApplicationHealthCheckConfig(
                path="/health/ready",
                start_period_seconds=30,
                timeout_seconds=5,
                failure_threshold=3,
                call_regularly=True,
                method="POST",
            ),
        )
    )

    assert stub.register_request is not None
    health_check_config = stub.register_request.health_check_config
    assert health_check_config.method == isolate_proto.ApplicationHealthCheckConfig.POST


def test_connection_register_leaves_isolate_health_check_method_unset():
    connection = FalServerlessConnection("api.alpha.fal.ai", MagicMock())
    stub = RecordingStub()
    connection._stub = stub  # type: ignore[assignment]
    environment = connection.define_environment("virtualenv")

    list(
        connection.register(
            lambda: None,
            [environment],
            application_name="function-app",
            deployment_strategy="rolling",
            health_check_config=ApplicationHealthCheckConfig(
                path="/health/ready",
                start_period_seconds=30,
                timeout_seconds=5,
                failure_threshold=3,
                call_regularly=True,
            ),
        )
    )

    assert stub.register_request is not None
    health_check_config = stub.register_request.health_check_config
    assert not health_check_config.HasField("method")


def test_connection_register_rejects_invalid_health_check_method():
    connection = FalServerlessConnection("api.alpha.fal.ai", MagicMock())
    stub = RecordingStub()
    connection._stub = stub  # type: ignore[assignment]
    environment = connection.define_environment(
        "container",
        image={"dockerfile_str": "FROM debian:bookworm-slim", "use_isolate": False},
    )

    with pytest.raises(ValueError, match='unknown enum label "TRACE"'):
        list(
            connection.register(
                None,
                [environment],
                application_name="container-app",
                deployment_strategy="rolling",
                health_check_config=ApplicationHealthCheckConfig(
                    path="/health/ready",
                    start_period_seconds=30,
                    timeout_seconds=5,
                    failure_threshold=3,
                    call_regularly=True,
                    method="TRACE",
                ),
            )
        )


def test_connection_register_rejects_isolate_container_without_callable():
    connection = FalServerlessConnection("api.alpha.fal.ai", MagicMock())
    environment = connection.define_environment(
        "container",
        image={"dockerfile_str": "FROM debian:bookworm-slim"},
    )

    with pytest.raises(ValueError, match="either function or entrypoint"):
        list(
            connection.register(
                None,
                [environment],
                application_name="container-app",
                deployment_strategy="rolling",
            )
        )


def test_connection_run_allows_no_isolate_container_without_callable():
    connection = FalServerlessConnection("api.alpha.fal.ai", MagicMock())
    stub = RecordingStub()
    connection._stub = stub  # type: ignore[assignment]
    environment = connection.define_environment(
        "container",
        image={"dockerfile_str": "FROM debian:bookworm-slim", "use_isolate": False},
    )

    list(connection.run(None, [environment]))

    assert stub.run_request is not None
    assert stub.run_request.WhichOneof("callable") is None


def test_connection_run_rejects_isolate_container_without_callable():
    connection = FalServerlessConnection("api.alpha.fal.ai", MagicMock())
    environment = connection.define_environment(
        "container",
        image={"dockerfile_str": "FROM debian:bookworm-slim"},
    )

    with pytest.raises(ValueError, match="either function or entrypoint"):
        list(connection.run(None, [environment]))


def test_retry_config_to_dict_behaviors():
    # Test emits nested form and omits None
    cfg = RetryConfig(server_error=3, timeout=1)
    assert cfg.to_dict() == {
        "server_error": {"retries": 3},
        "timeout": {"retries": 1},
    }
    assert "connection_error" not in cfg.to_dict()

    # Test clamps negatives
    assert RetryConfig(server_error=-5).to_dict() == {"server_error": {"retries": 0}}

    # Test empty config errors: users must define at least one condition
    with pytest.raises(ValueError, match="at least one condition"):
        RetryConfig().to_dict()


def test_validate_retry_config_dict_behaviors():
    cases = [
        (
            {"bad_condition": {"retries": 1}},
            "Invalid retry_config condition",
        ),
        (
            {"server_error": 3},
            "must be an object",
        ),
        (
            {"server_error": {"retries": "3"}},
            "must be an integer",
        ),
        (
            {"server_error": {"retries": True}},
            "must be an integer",
        ),
        (
            {"server_error": {"retries": 1, "extra": 2}},
            "must be an object",
        ),
        (
            [("server_error", 1)],  # Not a dict, but a list of tuples
            "must be a mapping",
        ),
    ]
    for invalid_input, expected_msg in cases:
        with pytest.raises(ValueError, match=expected_msg):
            validate_retry_config_dict(invalid_input)

    assert validate_retry_config_dict({"server_error": {"retries": 3}}) == {
        "server_error": {"retries": 3}
    }
    assert validate_retry_config_dict({"timeout": {"retries": -2}}) == {
        "timeout": {"retries": 0}
    }


def test_register_sets_retry_config_from_dataclass():
    connection = FalServerlessConnection("api.alpha.fal.ai", MagicMock())
    stub = RecordingStub()
    connection._stub = stub  # type: ignore[assignment]
    environment = connection.define_environment(
        "container",
        image={"dockerfile_str": "FROM debian:bookworm-slim", "use_isolate": False},
    )

    list(
        connection.register(
            None,
            [environment],
            application_name="container-app",
            deployment_strategy="rolling",
            retry_config=RetryConfig(server_error=3),
        )
    )

    assert stub.register_request.retry_config == '{"server_error": {"retries": 3}}'


def test_register_sets_retry_config_from_dict():
    connection = FalServerlessConnection("api.alpha.fal.ai", MagicMock())
    stub = RecordingStub()
    connection._stub = stub  # type: ignore[assignment]
    environment = connection.define_environment(
        "container",
        image={"dockerfile_str": "FROM debian:bookworm-slim", "use_isolate": False},
    )

    list(
        connection.register(
            None,
            [environment],
            application_name="container-app",
            deployment_strategy="rolling",
            retry_config={"timeout": {"retries": 2}},
        )
    )

    assert stub.register_request.retry_config == '{"timeout": {"retries": 2}}'


def test_register_rejects_invalid_retry_config_dict():
    connection = FalServerlessConnection("api.alpha.fal.ai", MagicMock())
    stub = RecordingStub()
    connection._stub = stub  # type: ignore[assignment]
    environment = connection.define_environment(
        "container",
        image={"dockerfile_str": "FROM debian:bookworm-slim", "use_isolate": False},
    )

    with pytest.raises(ValueError, match="Invalid retry_config condition"):
        list(
            connection.register(
                None,
                [environment],
                application_name="container-app",
                deployment_strategy="rolling",
                retry_config={"bogus": {"retries": 1}},
            )
        )


def test_register_omits_retry_config_when_none():
    connection = FalServerlessConnection("api.alpha.fal.ai", MagicMock())
    stub = RecordingStub()
    connection._stub = stub  # type: ignore[assignment]
    environment = connection.define_environment(
        "container",
        image={"dockerfile_str": "FROM debian:bookworm-slim", "use_isolate": False},
    )

    list(
        connection.register(
            None,
            [environment],
            application_name="container-app",
            deployment_strategy="rolling",
        )
    )

    assert not stub.register_request.HasField("retry_config")


def test_register_rejects_empty_retry_config():
    connection = FalServerlessConnection("api.alpha.fal.ai", MagicMock())
    stub = RecordingStub()
    connection._stub = stub  # type: ignore[assignment]
    environment = connection.define_environment(
        "container",
        image={"dockerfile_str": "FROM debian:bookworm-slim", "use_isolate": False},
    )

    with pytest.raises(ValueError, match="at least one condition"):
        list(
            connection.register(
                None,
                [environment],
                application_name="container-app",
                deployment_strategy="rolling",
                retry_config=RetryConfig(),
            )
        )
