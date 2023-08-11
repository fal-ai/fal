from __future__ import annotations

import enum
import os
from contextlib import ExitStack
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Generic, Iterator, Literal, TypeVar

import grpc
import isolate_proto
from fal_serverless import flags
from fal_serverless._serialization import patch_dill
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

patch_dill()


class ServerCredentials:
    def to_grpc(self) -> grpc.ChannelCredentials:
        raise NotImplementedError

    @property
    def extra_options(self) -> list[tuple[str, str]]:
        return GRPC_OPTIONS


class LocalCredentials(ServerCredentials):
    def to_grpc(self) -> grpc.ChannelCredentials:
        return grpc.local_channel_credentials()


class RemoteCredentials(ServerCredentials):
    def to_grpc(self) -> grpc.ChannelCredentials:
        return grpc.ssl_channel_credentials()


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


def get_default_server_credentials() -> ServerCredentials:
    if flags.TEST_MODE:
        return LocalCredentials()
    else:
        return RemoteCredentials()


class Credentials:
    # Cannot use `field` because child classes don't have default for all properties.
    server_credentials: ServerCredentials = get_default_server_credentials()

    def to_grpc(self) -> grpc.ChannelCredentials:
        return self.server_credentials.to_grpc()

    def to_headers(self) -> dict[str, str]:
        return {}


@dataclass
class FalServerlessKeyCredentials(Credentials):
    key_id: str
    key_secret: str

    def to_grpc(self) -> grpc.ChannelCredentials:
        return grpc.composite_channel_credentials(
            self.server_credentials.to_grpc(),
            grpc.metadata_call_credentials(_GRPCMetadata("auth-key", self.key_secret)),
            grpc.metadata_call_credentials(_GRPCMetadata("auth-key-id", self.key_id)),
        )

    def to_headers(self) -> dict[str, str]:
        return {"Authorization": f"Key {self.key_id}:{self.key_secret}"}


@dataclass
class AuthenticatedCredentials(Credentials):
    user = USER

    def to_grpc(self) -> grpc.ChannelCredentials:
        return grpc.composite_channel_credentials(
            self.server_credentials.to_grpc(),
            grpc.access_token_call_credentials(USER.access_token),
        )

    def to_headers(self) -> dict[str, str]:
        token = USER.bearer_token
        return {"Authorization": token}


@dataclass
class ServerlessSecret:
    name: str
    created_at: datetime


def key_credentials() -> FalServerlessKeyCredentials | None:
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

    key_creds = key_credentials()
    if is_agent() and key_creds:
        return key_creds
    else:
        return original_credentials


def get_default_credentials() -> Credentials:
    if flags.AUTH_DISABLED:
        return Credentials()

    key_creds = key_credentials()
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


class HostedRunState(Enum):
    IN_PROGRESS = 0
    SUCCESS = 1
    INTERNAL_FAILURE = 2


@dataclass
class HostedRunStatus:
    state: HostedRunState


@dataclass
class AliasInfo:
    alias: str
    revision: str
    auth_mode: str
    max_concurrency: int


@dataclass(frozen=True)
class Cron:
    cron_id: str
    cron_string: str
    next_run: datetime
    active: bool


@dataclass(frozen=True)
class ScheduledRunActivation:
    cron_id: str
    activation_id: str
    started_at: datetime
    finished_at: datetime


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


@dataclass(frozen=True)
class RegisterCronResultType:
    cron_id: str


@dataclass(frozen=True)
class RegisterCronResult:
    result: RegisterCronResultType


@dataclass
class RegisterApplicationResultType:
    application_id: str


@dataclass
class UserKeyInfo:
    key_id: str
    created_at: datetime
    scope: KeyScope
    alias: str


@dataclass
class WorkerStatus:
    worker_id: str
    start_time: datetime
    end_time: datetime
    duration: timedelta
    user_id: str
    machine_type: str


class KeyScope(enum.Enum):
    ADMIN = "ADMIN"
    API = "API"

    @staticmethod
    def from_proto(
        proto: isolate_proto.CreateUserKeyRequest.Scope.ValueType | None,
    ) -> KeyScope:
        if proto is None:
            return KeyScope.API

        if proto is isolate_proto.CreateUserKeyRequest.Scope.ADMIN:
            return KeyScope.ADMIN
        elif proto is isolate_proto.CreateUserKeyRequest.Scope.API:
            return KeyScope.API
        else:
            raise ValueError(f"Unknown KeyScope: {proto}")


@from_grpc.register(isolate_proto.RegisterCronResult)
def _from_grpc_register_cron_result(
    message: isolate_proto.RegisterCronResult,
) -> RegisterCronResult:
    return RegisterCronResult(
        result=RegisterCronResultType(message.result.cron_id),
    )


@from_grpc.register(isolate_proto.CronResultType)
def _from_grpc_cron_result_type(
    message: isolate_proto.CronResultType,
) -> Cron:
    return Cron(
        cron_id=message.cron_id,
        cron_string=message.cron_string,
        next_run=message.next_run.ToDatetime(),
        active=message.is_active,
    )


@from_grpc.register(isolate_proto.AliasInfo)
def _from_grpc_alias_info(message: isolate_proto.AliasInfo) -> AliasInfo:
    if message.auth_mode is isolate_proto.ApplicationAuthMode.PUBLIC:
        auth_mode = "public"
    elif message.auth_mode is isolate_proto.ApplicationAuthMode.PRIVATE:
        auth_mode = "private"
    elif message.auth_mode is isolate_proto.ApplicationAuthMode.SHARED:
        auth_mode = "shared"
    else:
        raise ValueError(f"Unknown auth mode: {message.auth_mode}")

    return AliasInfo(
        alias=message.alias,
        revision=message.revision,
        auth_mode=auth_mode,
        max_concurrency=message.max_concurrency,
    )


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


def _get_cron_id(run: Cron | str) -> str:
    if isinstance(run, Cron):
        return run.cron_id
    else:
        return run


@dataclass
class MachineRequirements:
    machine_type: str
    keep_alive: int = FAL_SERVERLESS_DEFAULT_KEEP_ALIVE
    base_image: str | None = None
    exposed_port: int | None = None


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

        options = self.credentials.server_credentials.extra_options
        channel_creds = self.credentials.to_grpc()
        channel = self._stack.enter_context(
            grpc.secure_channel(self.hostname, channel_creds, options)
        )
        channel = grpc.intercept_channel(channel, TraceContextInterceptor())
        self._stub = isolate_proto.IsolateControllerStub(channel)
        return self._stub

    def create_user_key(self, scope: KeyScope, alias: str | None) -> tuple[str, str]:
        scope_proto = (
            isolate_proto.CreateUserKeyRequest.Scope.ADMIN
            if scope is KeyScope.ADMIN
            else isolate_proto.CreateUserKeyRequest.Scope.API
        )

        request = isolate_proto.CreateUserKeyRequest(scope=scope_proto, alias=alias)
        response = self.stub.CreateUserKey(request)
        return response.key_secret, response.key_id

    def list_user_keys(self) -> list[UserKeyInfo]:
        request = isolate_proto.ListUserKeysRequest()
        response: isolate_proto.ListUserKeysResponse = self.stub.ListUserKeys(request)
        return [
            UserKeyInfo(
                key.key_id,
                isolate_proto.datetime_from_timestamp(key.created_at),
                KeyScope.from_proto(key.scope),
                key.alias,
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
        application_name: str | None = None,
        application_auth_mode: Literal["public", "private", "shared"] | None = None,
        *,
        max_concurrency: int | None = None,
        serialization_method: str = _DEFAULT_SERIALIZATION_METHOD,
        machine_requirements: MachineRequirements | None = None,
    ) -> Iterator[isolate_proto.RegisterApplicationResult]:
        wrapped_function = to_serialized_object(function, serialization_method)
        if machine_requirements:
            wrapped_requirements = isolate_proto.MachineRequirements(
                machine_type=machine_requirements.machine_type,
                keep_alive=machine_requirements.keep_alive,
                base_image=machine_requirements.base_image,
                exposed_port=machine_requirements.exposed_port,
            )
        else:
            wrapped_requirements = None

        if application_auth_mode == "public":
            auth_mode = isolate_proto.ApplicationAuthMode.PUBLIC
        elif application_auth_mode == "shared":
            auth_mode = isolate_proto.ApplicationAuthMode.SHARED
        else:
            auth_mode = isolate_proto.ApplicationAuthMode.PRIVATE

        request = isolate_proto.RegisterApplicationRequest(
            function=wrapped_function,
            environments=environments,
            machine_requirements=wrapped_requirements,
            max_concurrency=max_concurrency,
            application_name=application_name,
            auth_mode=auth_mode,
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
                base_image=machine_requirements.base_image,
                exposed_port=machine_requirements.exposed_port,
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

    def schedule_cronjob(
        self,
        application_id: str,
        cron: str,
    ) -> str:
        request = isolate_proto.RegisterCronRequest(
            application_id=application_id, cron=cron
        )
        response: isolate_proto.RegisterCronResult = self.stub.RegisterCron(request)
        return response.result.cron_id

    def list_aliases(self) -> list[AliasInfo]:
        request = isolate_proto.ListAliasesRequest()
        response: isolate_proto.ListAliasesResult = self.stub.ListAliases(request)
        return [from_grpc(alias) for alias in response.aliases]

    def list_scheduled_runs(self) -> list[Cron]:
        request = isolate_proto.ListCronsRequest()
        response: isolate_proto.ListCronsResult = self.stub.ListCrons(request)
        return [from_grpc(cron) for cron in response.crons]

    def list_run_activations(self, run: str | Cron) -> list[ScheduledRunActivation]:
        request = isolate_proto.ListActivationsRequest(cron_id=_get_cron_id(run))
        response: isolate_proto.ListActivationsResult = self.stub.ListActivations(
            request
        )
        return [
            ScheduledRunActivation(
                cron_id=_get_cron_id(run),
                activation_id=activation.activation_id,
                started_at=activation.started_at.ToDatetime(),
                finished_at=activation.finished_at.ToDatetime(),
            )
            for activation in response.activations
        ]

    def cancel_scheduled_run(self, cron_id: str) -> None:
        request = isolate_proto.CancelCronRequest(cron_id=cron_id)
        response: isolate_proto.CancelCronResult = self.stub.CancelCron(request)
        return

    def get_activation_logs(self, cron_id: str, activation_id: str) -> list[Log]:
        request = isolate_proto.GetActivationLogsRequest(
            cron_id=cron_id, activation_id=activation_id
        )
        response = self.stub.GetActivationLogs(request)
        return [from_grpc(log) for log in response.logs]

    def get_logs(
        self, lines: int | None = None, url: str | None = None
    ) -> Iterator[Log]:
        filter = isolate_proto.LogsFilter(lines=lines, url=url)
        request = isolate_proto.GetLogsRequest(filter=filter)
        for partial_result in self.stub.GetLogs(request):
            yield from_grpc(partial_result.log_entry)

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

    def set_secret(self, name: str, value: str) -> None:
        request = isolate_proto.SetSecretRequest(name=name, value=value)
        self.stub.SetSecret(request)

    def delete_secret(self, name: str) -> None:
        request = isolate_proto.SetSecretRequest(name=name, value=None)
        self.stub.SetSecret(request)

    def list_secrets(self) -> list[ServerlessSecret]:
        request = isolate_proto.ListSecretsRequest()
        response = self.stub.ListSecrets(request)
        return [
            ServerlessSecret(
                name=secret.name,
                created_at=isolate_proto.datetime_from_timestamp(secret.created_time),
            )
            for secret in response.secrets
        ]
