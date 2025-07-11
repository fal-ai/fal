from __future__ import annotations

import enum
from contextlib import ExitStack
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    Iterator,
    Literal,
    Optional,
    TypeVar,
)

import grpc
import isolate_proto
from isolate.server.interface import from_grpc, to_serialized_object, to_struct

from fal import flags
from fal._serialization import patch_pickle
from fal.auth import UserAccess, key_credentials
from fal.logging import get_logger
from fal.logging.trace import TraceContextInterceptor

if TYPE_CHECKING:
    from isolate.logs import Log

ResultT = TypeVar("ResultT")
InputT = TypeVar("InputT")
UNSET = object()

_DEFAULT_SERIALIZATION_METHOD = "cloudpickle"
FAL_SERVERLESS_DEFAULT_KEEP_ALIVE = 10
FAL_SERVERLESS_DEFAULT_MAX_MULTIPLEXING = 1
FAL_SERVERLESS_DEFAULT_MIN_CONCURRENCY = 0
FAL_SERVERLESS_DEFAULT_CONCURRENCY_BUFFER = 0
ALIAS_AUTH_MODES = ["public", "private", "shared"]

logger = get_logger(__name__)

patch_pickle()


AuthMode = Optional[Literal["public", "private", "shared"]]


class ServerCredentials:
    def to_grpc(self) -> grpc.ChannelCredentials:
        raise NotImplementedError

    @property
    def base_options(self) -> dict[str, str | int]:
        import json

        from isolate_proto.configuration import GRPC_OPTIONS

        grpc_ops: dict[str, str | int] = dict(GRPC_OPTIONS)
        grpc_ops["grpc.enable_retries"] = 1
        grpc_ops["grpc.service_config"] = json.dumps(
            {
                "methodConfig": [
                    {
                        "name": [{}],
                        "retryPolicy": {
                            "maxAttempts": 5,
                            "initialBackoff": "0.1s",
                            "maxBackoff": "5s",
                            "backoffMultiplier": 2,
                            "retryableStatusCodes": ["UNAVAILABLE"],
                        },
                    }
                ]
            }
        )

        return grpc_ops


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
    user: UserAccess = field(default_factory=UserAccess)
    team: str | None = None

    def to_grpc(self) -> grpc.ChannelCredentials:
        creds = [
            self.server_credentials.to_grpc(),
            grpc.access_token_call_credentials(self.user.access_token),
        ]

        if self.team:
            team_id = self.user.get_account(self.team)["user_id"]
            creds.append(
                grpc.metadata_call_credentials(_GRPCMetadata("fal-user-id", team_id))
            )

        return grpc.composite_channel_credentials(*creds)

    def to_headers(self) -> dict[str, str]:
        token = self.user.bearer_token
        headers = {
            "Authorization": token,
        }

        if self.team:
            team_id = self.user.get_account(self.team)["user_id"]
            headers["X-Fal-User-Id"] = team_id

        return headers


@dataclass
class ServerlessSecret:
    name: str
    created_at: datetime


def get_agent_credentials(original_credentials: Credentials) -> Credentials:
    """If running inside a fal Serverless box, use the preconfigured credentials
    instead of the user provided ones."""

    from isolate.connections.common import is_agent

    key_creds = key_credentials()
    if is_agent() and key_creds:
        return FalServerlessKeyCredentials(key_creds[0], key_creds[1])
    else:
        return original_credentials


def get_default_credentials(team: str | None = None) -> Credentials:
    from fal.config import Config

    if flags.AUTH_DISABLED:
        return Credentials()

    key_creds = key_credentials()
    if key_creds:
        logger.debug("Using key credentials")
        if team:
            raise ValueError("Using explicit team with key credentials is not allowed")
        return FalServerlessKeyCredentials(key_creds[0], key_creds[1])
    else:
        config = Config()
        team = team or config.get_internal("team")
        return AuthenticatedCredentials(team=team)


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
class ApplicationInfo:
    application_id: str
    keep_alive: int
    max_concurrency: int
    max_multiplexing: int
    active_runners: int
    min_concurrency: int
    concurrency_buffer: int
    machine_types: list[str]
    request_timeout: int
    startup_timeout: int
    valid_regions: list[str]
    created_at: datetime


@dataclass
class AliasInfo:
    alias: str
    revision: str
    auth_mode: str
    keep_alive: int
    max_concurrency: int
    max_multiplexing: int
    active_runners: int
    min_concurrency: int
    concurrency_buffer: int
    machine_types: list[str]
    request_timeout: int
    startup_timeout: int
    valid_regions: list[str]


@dataclass
class RunnerInfo:
    runner_id: str
    in_flight_requests: int
    expiration_countdown: Optional[int]
    uptime: timedelta
    external_metadata: dict[str, Any]
    revision: str
    alias: str


@dataclass
class HostedRunResult(Generic[ResultT]):
    run_id: str
    status: HostedRunStatus
    logs: list[Log] = field(default_factory=list)
    result: ResultT | None = None
    stream: Any = None


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


class DeploymentStrategy(enum.Enum):
    RECREATE = "recreate"
    ROLLING = "rolling"

    @staticmethod
    def from_proto(
        proto: isolate_proto.DeploymentStrategy.ValueType | None,
    ) -> DeploymentStrategy:
        if proto is None:
            return DeploymentStrategy.RECREATE

        if proto is isolate_proto.DeploymentStrategy.RECREATE:
            return DeploymentStrategy.RECREATE
        elif proto is isolate_proto.DeploymentStrategy.ROLLING:
            return DeploymentStrategy.ROLLING
        else:
            raise ValueError(f"Unknown DeploymentStrategy: {proto}")

    def to_proto(self) -> isolate_proto.DeploymentStrategy.ValueType:
        if self is DeploymentStrategy.RECREATE:
            return isolate_proto.DeploymentStrategy.RECREATE
        elif self is DeploymentStrategy.ROLLING:
            return isolate_proto.DeploymentStrategy.ROLLING
        else:
            raise ValueError(f"Unknown DeploymentStrategy: {self}")


@from_grpc.register(isolate_proto.ApplicationInfo)
def _from_grpc_application_info(
    message: isolate_proto.ApplicationInfo,
) -> ApplicationInfo:
    return ApplicationInfo(
        application_id=message.application_id,
        keep_alive=message.keep_alive,
        max_concurrency=message.max_concurrency,
        max_multiplexing=message.max_multiplexing,
        active_runners=message.active_runners,
        min_concurrency=message.min_concurrency,
        concurrency_buffer=message.concurrency_buffer,
        machine_types=list(message.machine_types),
        request_timeout=message.request_timeout,
        startup_timeout=message.startup_timeout,
        valid_regions=list(message.valid_regions),
        created_at=isolate_proto.datetime_from_timestamp(message.created_at),
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
        keep_alive=message.keep_alive,
        max_concurrency=message.max_concurrency,
        max_multiplexing=message.max_multiplexing,
        active_runners=message.active_runners,
        min_concurrency=message.min_concurrency,
        concurrency_buffer=message.concurrency_buffer,
        machine_types=list(message.machine_types),
        request_timeout=message.request_timeout,
        startup_timeout=message.startup_timeout,
        valid_regions=list(message.valid_regions),
    )


@from_grpc.register(isolate_proto.RunnerInfo)
def _from_grpc_runner_info(message: isolate_proto.RunnerInfo) -> RunnerInfo:
    from isolate.server import definitions as worker_definitions

    external_metadata = worker_definitions.struct_to_dict(message.external_metadata)
    return RunnerInfo(
        runner_id=message.runner_id,
        in_flight_requests=message.in_flight_requests,
        expiration_countdown=(
            message.expiration_countdown
            if message.HasField("expiration_countdown")
            else None
        ),
        uptime=timedelta(seconds=message.uptime),
        external_metadata=external_metadata,
        revision=message.revision,
        alias=message.alias,
    )


@from_grpc.register(isolate_proto.RegisterApplicationResult)
def _from_grpc_register_application_result(
    message: isolate_proto.RegisterApplicationResult,
) -> RegisterApplicationResult:
    return RegisterApplicationResult(
        logs=[from_grpc(log) for log in message.logs],
        result=(
            None
            if not message.HasField("result")
            else RegisterApplicationResultType(message.result.application_id)
        ),
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


@dataclass
class MachineRequirements:
    machine_types: list[str]
    num_gpus: int | None = field(default=None)
    keep_alive: int = FAL_SERVERLESS_DEFAULT_KEEP_ALIVE
    base_image: str | None = None
    exposed_port: int | None = None
    scheduler: str | None = None
    scheduler_options: dict[str, Any] | None = None
    max_concurrency: int | None = None
    max_multiplexing: int | None = None
    min_concurrency: int | None = None
    concurrency_buffer: int | None = None
    request_timeout: int | None = None
    startup_timeout: int | None = None

    def __post_init__(self):
        if isinstance(self.machine_types, str):
            self.machine_types = [self.machine_types]

        if not isinstance(self.machine_types, list):
            raise ValueError("machine_types must be a list of strings.")

        if not self.machine_types:
            raise ValueError("No machine type provided.")


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

        options = self.credentials.server_credentials.base_options
        channel_creds = self.credentials.to_grpc()
        channel = self._stack.enter_context(
            grpc.secure_channel(
                target=self.hostname,
                credentials=channel_creds,
                options=list(options.items()),
            )
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

    def define_environment(
        self, kind: str, **options: Any
    ) -> isolate_proto.EnvironmentDefinition:
        struct = isolate_proto.Struct()
        struct.update(options)

        return isolate_proto.EnvironmentDefinition(
            kind=kind,
            configuration=struct,
        )

    def register(
        self,
        function: Callable[..., ResultT],
        environments: list[isolate_proto.EnvironmentDefinition],
        application_name: str | None = None,
        auth_mode: AuthMode = None,
        *,
        serialization_method: str = _DEFAULT_SERIALIZATION_METHOD,
        machine_requirements: MachineRequirements | None = None,
        metadata: dict[str, Any] | None = None,
        deployment_strategy: Literal["recreate", "rolling"] = "recreate",
        scale: bool = True,
        private_logs: bool = False,
    ) -> Iterator[isolate_proto.RegisterApplicationResult]:
        wrapped_function = to_serialized_object(function, serialization_method)
        if machine_requirements:
            wrapped_requirements = isolate_proto.MachineRequirements(
                # NOTE: backwards compatibility with old API
                machine_type=machine_requirements.machine_types[0],
                machine_types=machine_requirements.machine_types,
                num_gpus=machine_requirements.num_gpus,
                keep_alive=machine_requirements.keep_alive,
                base_image=machine_requirements.base_image,
                exposed_port=machine_requirements.exposed_port,
                scheduler=machine_requirements.scheduler,
                scheduler_options=to_struct(
                    machine_requirements.scheduler_options or {}
                ),
                max_concurrency=machine_requirements.max_concurrency,
                min_concurrency=machine_requirements.min_concurrency,
                concurrency_buffer=machine_requirements.concurrency_buffer,
                max_multiplexing=machine_requirements.max_multiplexing,
                request_timeout=machine_requirements.request_timeout,
                startup_timeout=machine_requirements.startup_timeout,
            )
        else:
            wrapped_requirements = None

        if auth_mode == "public":
            auth = isolate_proto.ApplicationAuthMode.PUBLIC
        elif auth_mode == "shared":
            auth = isolate_proto.ApplicationAuthMode.SHARED
        elif auth_mode == "private":
            auth = isolate_proto.ApplicationAuthMode.PRIVATE
        else:
            auth = None

        struct_metadata = None
        if metadata:
            struct_metadata = isolate_proto.Struct()
            struct_metadata.update(metadata)

        deployment_strategy_proto = DeploymentStrategy[
            deployment_strategy.upper()
        ].to_proto()

        request = isolate_proto.RegisterApplicationRequest(
            function=wrapped_function,
            environments=environments,
            machine_requirements=wrapped_requirements,
            application_name=application_name,
            auth_mode=auth,
            metadata=struct_metadata,
            deployment_strategy=deployment_strategy_proto,
            scale=scale,
            private_logs=private_logs,
        )
        for partial_result in self.stub.RegisterApplication(request):
            yield from_grpc(partial_result)

    def scale(self, application_name: str, max_concurrency: int | None = None) -> None:
        raise NotImplementedError

    def update_application(
        self,
        application_name: str,
        keep_alive: int | None = None,
        max_multiplexing: int | None = None,
        max_concurrency: int | None = None,
        min_concurrency: int | None = None,
        concurrency_buffer: int | None = None,
        request_timeout: int | None = None,
        startup_timeout: int | None = None,
        valid_regions: list[str] | None = None,
        machine_types: list[str] | None = None,
    ) -> AliasInfo:
        request = isolate_proto.UpdateApplicationRequest(
            application_name=application_name,
            keep_alive=keep_alive,
            max_multiplexing=max_multiplexing,
            max_concurrency=max_concurrency,
            min_concurrency=min_concurrency,
            concurrency_buffer=concurrency_buffer,
            request_timeout=request_timeout,
            startup_timeout=startup_timeout,
            valid_regions=valid_regions,
            machine_types=machine_types,
        )
        res: isolate_proto.UpdateApplicationResult = self.stub.UpdateApplication(
            request
        )
        return from_grpc(res.alias_info)

    def list_applications(
        self, application_name: str | None = None
    ) -> list[ApplicationInfo]:
        request = isolate_proto.ListApplicationsRequest(
            application_name=application_name
        )
        res: isolate_proto.ListApplicationsResult = self.stub.ListApplications(request)
        return [from_grpc(app) for app in res.applications]

    def delete_application(
        self,
        application_id: str,
    ) -> None:
        request = isolate_proto.DeleteApplicationRequest(application_id=application_id)
        self.stub.DeleteApplication(request)

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
                # NOTE: backwards compatibility with old API
                machine_type=machine_requirements.machine_types[0],
                machine_types=machine_requirements.machine_types,
                num_gpus=machine_requirements.num_gpus,
                keep_alive=machine_requirements.keep_alive,
                base_image=machine_requirements.base_image,
                exposed_port=machine_requirements.exposed_port,
                scheduler=machine_requirements.scheduler,
                scheduler_options=to_struct(
                    machine_requirements.scheduler_options or {}
                ),
                max_concurrency=machine_requirements.max_concurrency,
                max_multiplexing=machine_requirements.max_multiplexing,
                min_concurrency=machine_requirements.min_concurrency,
                concurrency_buffer=machine_requirements.concurrency_buffer,
                request_timeout=machine_requirements.request_timeout,
                startup_timeout=machine_requirements.startup_timeout,
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
        stream = self.stub.Run(request)
        for partial_result in stream:
            res = from_grpc(partial_result)
            res.stream = stream
            yield res

    def create_alias(
        self,
        alias: str,
        revision: str,
        auth_mode: AuthMode,
    ) -> AliasInfo:
        if auth_mode == "public":
            auth = isolate_proto.ApplicationAuthMode.PUBLIC
        elif auth_mode == "shared":
            auth = isolate_proto.ApplicationAuthMode.SHARED
        elif auth_mode == "private":
            auth = isolate_proto.ApplicationAuthMode.PRIVATE
        else:
            auth = None

        request = isolate_proto.SetAliasRequest(
            alias=alias,
            revision=revision,
            auth_mode=auth,
        )
        res = self.stub.SetAlias(request)
        return from_grpc(res.alias_info)

    def delete_alias(self, alias: str) -> str | None:
        request = isolate_proto.DeleteAliasRequest(alias=alias)
        try:
            res: isolate_proto.DeleteAliasResult = self.stub.DeleteAlias(request)
            return res.revision
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.NOT_FOUND:
                return None
            raise

    def list_aliases(self) -> list[AliasInfo]:
        request = isolate_proto.ListAliasesRequest()
        response: isolate_proto.ListAliasesResult = self.stub.ListAliases(request)
        return [from_grpc(alias) for alias in response.aliases]

    def list_alias_runners(self, alias: str) -> list[RunnerInfo]:
        request = isolate_proto.ListAliasRunnersRequest(alias=alias)
        response = self.stub.ListAliasRunners(request)
        return [from_grpc(runner) for runner in response.runners]

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

    def kill_runner(self, runner_id: str) -> None:
        request = isolate_proto.KillRunnerRequest(runner_id=runner_id)
        self.stub.KillRunner(request)

    def list_runners(self) -> list[RunnerInfo]:
        request = isolate_proto.ListRunnersRequest()
        response = self.stub.ListRunners(request)
        return [from_grpc(runner) for runner in response.runners]
