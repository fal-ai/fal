from __future__ import annotations

import os
from contextlib import ExitStack
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Generic, Iterator, TypeVar

import grpc
import isolate_proto
from fal_serverless import flags
from fal_serverless.auth import USER
from fal_serverless.logging.trace import TraceContextInterceptor
from isolate.connections.common import is_agent
from isolate.logs import Log
from isolate.server.interface import from_grpc, to_serialized_object
from isolate_proto.configuration import GRPC_OPTIONS

ResultT = TypeVar("ResultT")
InputT = TypeVar("InputT")
UNSET = object()

_DEFAULT_SERIALIZATION_METHOD = "dill"
FAL_SERVERLESS_DEFAULT_KEEP_ALIVE = 10


class Credentials:
    def to_grpc(self) -> grpc.ChannelCredentials:
        raise NotImplementedError

    @property
    def extra_options(self) -> list[tuple[str, str]]:
        return GRPC_OPTIONS


class LocalCredentials(Credentials):
    def to_grpc(self) -> grpc.ChannelCredentials:
        return grpc.local_channel_credentials()


@dataclass
class _GRPCMetadata(grpc.AuthMetadataPlugin):
    """Key value metadata bundle for gRPC credentials"""

    _key: str
    _value: str

    def __call__(
        self,
        context: grpc.AuthMetadataContext,
        callback: grpc.AuthMetadataPluginCallback,
    ) -> None:
        callback(((self._key, self._value),), None)


@dataclass
class FalServerlessKeyCredentials(Credentials):
    key_id: str
    key_secret: str

    def to_grpc(self) -> grpc.ChannelCredentials:
        return grpc.composite_channel_credentials(
            grpc.ssl_channel_credentials(),
            grpc.metadata_call_credentials(_GRPCMetadata("auth-key", self.key_secret)),
            grpc.metadata_call_credentials(_GRPCMetadata("auth-key-id", self.key_id)),
        )


@dataclass
class AuthenticatedCredentials(Credentials):
    user = USER

    def to_grpc(self) -> grpc.ChannelCredentials:
        return grpc.composite_channel_credentials(
            grpc.ssl_channel_credentials(),
            grpc.access_token_call_credentials(USER.access_token),
        )


def _key_credentials() -> FalServerlessKeyCredentials | None:
    # Ignore key credentials when the user forces auth by user.
    if os.environ.get("FAL_FORCE_AUTH_BY_USER") == "1":
        return None

    if "FAL_KEY_ID" in os.environ and "FAL_KEY_SECRET" in os.environ:
        return FalServerlessKeyCredentials(
            os.environ["FAL_KEY_ID"],
            os.environ["FAL_KEY_SECRET"],
        )
    else:
        return None


def _get_agent_credentials(original_credentials: Credentials) -> Credentials:
    """If running inside a fal Serverless box, use the preconfigured credentials
    instead of the user provided ones."""

    key_creds = _key_credentials()
    if is_agent() and key_creds:
        return key_creds
    else:
        return original_credentials


def get_default_credentials() -> Credentials:
    if flags.TEST_MODE:
        return LocalCredentials()
    else:
        key_creds = _key_credentials()
        if key_creds:
            return key_creds
        else:
            return AuthenticatedCredentials()


@dataclass
class FalServerlessClient:
    hostname: str
    credentials: Credentials = field(default_factory=get_default_credentials)

    def connect(self) -> FalServerlessConnection:
        return FalServerlessConnection(self.hostname, self.credentials)


class ScheduledRunState(Enum):
    SCHEDULED = 0
    INTERNAL_FAILURE = 1
    USER_FAILURE = 2


class HostedRunState(Enum):
    IN_PROGRESS = 0
    SUCCESS = 1
    INTERNAL_FAILURE = 2


@dataclass
class HostedRunStatus:
    state: HostedRunState


@dataclass
class ScheduledRun:
    run_id: str
    state: ScheduledRunState
    cron: str


@dataclass
class ScheduledRunActivation:
    run_id: str
    activation_id: str


@dataclass
class HostedRunResult(Generic[ResultT]):
    run_id: str
    status: HostedRunStatus
    logs: list[Log] = field(default_factory=list)
    result: ResultT | None = None


@dataclass
class RegisterApplicationResult:
    result: RegisterApplicationResultType | None
    logs: list[Log] = field(default_factory=list)


@dataclass
class RegisterApplicationResultType:
    application_id: str


@dataclass
class UserKeyInfo:
    key_id: str
    created_at: datetime


@dataclass
class WorkerStatus:
    worker_id: str
    start_time: datetime
    end_time: datetime
    duration: timedelta
    user_id: str
    machine_type: str


@from_grpc.register(isolate_proto.RegisterApplicationResult)
def _from_grpc_register_application_result(
    message: isolate_proto.RegisterApplicationResult,
) -> RegisterApplicationResult:
    return RegisterApplicationResult(
        logs=[from_grpc(log) for log in message.logs],
        result=None
        if not message.HasField("result")
        else RegisterApplicationResultType(message.result.application_id),
    )


@from_grpc.register(isolate_proto.HostedRunStatus)
def _from_grpc_hosted_run_status(
    message: isolate_proto.HostedRunStatus,
) -> HostedRunStatus:
    return HostedRunStatus(HostedRunState(message.state))


@from_grpc.register(isolate_proto.HostedRunResult)
def _from_grpc_hosted_run_result(
    message: isolate_proto.HostedRunResult,
) -> HostedRunResult[Any]:
    if message.return_value.definition:
        return_value = from_grpc(message.return_value)
    else:
        return_value = UNSET

    return HostedRunResult(
        message.run_id,
        from_grpc(message.status),
        logs=[from_grpc(log) for log in message.logs],
        result=return_value,
    )


def _get_run_id(run: ScheduledRun | str) -> str:
    if isinstance(run, ScheduledRun):
        return run.run_id
    else:
        return run


@dataclass
class MachineRequirements:
    machine_type: str
    keep_alive: int = FAL_SERVERLESS_DEFAULT_KEEP_ALIVE


@dataclass
class FalServerlessConnection:
    hostname: str
    credentials: Credentials

    _stack: ExitStack = field(default_factory=ExitStack)
    _stub: isolate_proto.IsolateControllerStub | None = None

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self._stack.close()

    def close(self):
        self._stack.close()

    @property
    def stub(self) -> isolate_proto.IsolateControllerStub:
        if self._stub:
            return self._stub

        options = self.credentials.extra_options
        channel_creds = self.credentials.to_grpc()
        channel = self._stack.enter_context(
            grpc.secure_channel(self.hostname, channel_creds, options)
        )
        channel = grpc.intercept_channel(channel, TraceContextInterceptor())
        self._stub = isolate_proto.IsolateControllerStub(channel)
        return self._stub

    def create_user_key(self) -> tuple[str, str]:
        request = isolate_proto.CreateUserKeyRequest()
        response = self.stub.CreateUserKey(request)
        return response.key_secret, response.key_id

    def list_user_keys(self) -> list[UserKeyInfo]:
        request = isolate_proto.ListUserKeysRequest()
        response: isolate_proto.ListUserKeysResponse = self.stub.ListUserKeys(request)
        return [
            UserKeyInfo(
                key.key_id,
                isolate_proto.datetime_from_timestamp(key.created_at),
            )
            for key in response.user_keys
        ]

    def revoke_user_key(self, key_id) -> None:
        request = isolate_proto.RevokeUserKeyRequest(key_id=key_id)
        self.stub.RevokeUserKey(request)

    # TODO: get rid of this in favor of define_environment
    def create_environment(
        self,
        kind: str,
        configuration_options: dict[str, Any],
    ) -> isolate_proto.EnvironmentDefinition:
        assert isinstance(
            configuration_options, dict
        ), "configuration_options must be a dict"
        struct = isolate_proto.Struct()
        struct.update(configuration_options)

        return isolate_proto.EnvironmentDefinition(
            kind=kind,
            configuration=struct,
        )

    def define_environment(
        self, kind: str, **options: Any
    ) -> isolate_proto.EnvironmentDefinition:
        return self.create_environment(
            kind=kind,
            configuration_options=options,
        )

    def register(
        self,
        function: Callable[..., ResultT],
        environments: list[isolate_proto.EnvironmentDefinition],
        *,
        serialization_method: str = _DEFAULT_SERIALIZATION_METHOD,
        machine_requirements: MachineRequirements | None = None,
    ) -> Iterator[isolate_proto.RegisterApplicationResult]:

        wrapped_function = to_serialized_object(function, serialization_method)
        if machine_requirements:
            wrapped_requirements = isolate_proto.MachineRequirements(
                machine_type=machine_requirements.machine_type,
                keep_alive=machine_requirements.keep_alive,
            )
        else:
            wrapped_requirements = None

        request = isolate_proto.RegisterApplicationRequest(
            function=wrapped_function,
            environments=environments,
            machine_requirements=wrapped_requirements,
        )
        for partial_result in self.stub.RegisterApplication(request):
            yield from_grpc(partial_result)

    def run(
        self,
        function: Callable[..., ResultT],
        environments: list[isolate_proto.EnvironmentDefinition],
        *,
        serialization_method: str = _DEFAULT_SERIALIZATION_METHOD,
        machine_requirements: MachineRequirements | None = None,
        setup_function: Callable[[], InputT] | None = None,
    ) -> Iterator[HostedRunResult[ResultT]]:
        wrapped_function = to_serialized_object(function, serialization_method)
        if machine_requirements:
            wrapped_requirements = isolate_proto.MachineRequirements(
                machine_type=machine_requirements.machine_type,
                keep_alive=machine_requirements.keep_alive,
            )
        else:
            wrapped_requirements = None

        request = isolate_proto.HostedRun(
            function=wrapped_function,
            environments=environments,
            machine_requirements=wrapped_requirements,
        )
        if setup_function:
            request.setup_func.MergeFrom(
                to_serialized_object(setup_function, serialization_method)
            )
        for partial_result in self.stub.Run(request):
            yield from_grpc(partial_result)

    def schedule_run(
        self,
        function: Callable[[], ResultT],
        environments: list[isolate_proto.EnvironmentDefinition],
        cron: str,
        *,
        serialization_method: str = _DEFAULT_SERIALIZATION_METHOD,
        machine_requirements: MachineRequirements | None = None,
    ) -> ScheduledRun:
        wrapped_function = to_serialized_object(function, serialization_method)
        if machine_requirements:
            wrapped_requirements = isolate_proto.MachineRequirements(
                machine_type=machine_requirements.machine_type
            )
        else:
            wrapped_requirements = None

        request = isolate_proto.HostedRunCron(
            function=wrapped_function,
            environments=environments,
            cron=cron,
            machine_requirements=wrapped_requirements,
        )
        response = self.stub.Schedule(request)
        return ScheduledRun(
            response.run_id,
            state=ScheduledRunState(response.state),
            cron=cron,
        )

    def list_scheduled_runs(self) -> list[ScheduledRun]:
        request = isolate_proto.ListScheduledRunsRequest()
        response = self.stub.ListScheduledRuns(request)
        return [
            ScheduledRun(
                run.run_id,
                state=ScheduledRunState(run.state),
                cron=run.cron,
            )
            for run in response.scheduled_runs
        ]

    def cancel_scheduled_run(self, run: ScheduledRun | str) -> None:
        request = isolate_proto.CancelScheduledRunRequest(run_id=_get_run_id(run))
        self.stub.CancelScheduledRun(request)

    def list_run_activations(
        self, run: ScheduledRun | str
    ) -> list[ScheduledRunActivation]:
        request = isolate_proto.ListScheduledRunActivationsRequest(
            run_id=_get_run_id(run)
        )
        response = self.stub.ListScheduledRunActivations(request)
        return [
            ScheduledRunActivation(
                run_id=_get_run_id(run),
                activation_id=activation_id,
            )
            for activation_id in response.activation_ids
        ]

    def get_activation_logs(self, activation: ScheduledRunActivation) -> bytes:
        request = isolate_proto.GetScheduledActivationLogsRequest(
            run_id=activation.run_id,
            activation_id=activation.activation_id,
        )
        response = self.stub.GetScheduledActivationLogs(request)
        return response.raw_logs

    def list_worker_status(self, user_id: str | None = None) -> list[WorkerStatus]:
        request = isolate_proto.WorkerStatusListRequest(user_id=user_id)
        response = self.stub.WorkerStatusList(request)
        return [
            WorkerStatus(
                ws.worker_id,
                isolate_proto.datetime_from_timestamp(ws.start_time),
                isolate_proto.datetime_from_timestamp(ws.end_time),
                isolate_proto.timedelta_from_duration(ws.duration),
                ws.user_id,
                ws.machine_type,
            )
            for ws in response.worker_status
        ]
