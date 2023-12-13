from __future__ import annotations

import json
from datetime import datetime
from http import HTTPStatus
from sys import argv
from typing import Literal
from uuid import uuid4

import click
import fal.auth as auth
import grpc
from fal import api, sdk
from fal.console import console
from fal.exceptions import ApplicationExceptionHandler
from fal.logging import get_logger, set_debug_logging
from fal.logging.isolate import IsolateLogPrinter
from fal.logging.trace import get_tracer
from fal.rest_client import REST_CLIENT
from fal.sdk import AliasInfo, KeyScope
from isolate.logs import Log, LogLevel, LogSource
from rich.table import Table

import openapi_fal_rest.api.billing.get_user_details as get_user_details
import openapi_fal_rest.api.logs.list_since as list_logs

DEFAULT_HOST = "api.alpha.fal.ai"
HOST_ENVVAR = "FAL_HOST"

DEFAULT_PORT = "443"
PORT_ENVVAR = "FAL_PORT"

DEBUG_ENABLED = False


log = get_logger(__name__)


class ExecutionInfo:
    debug: bool
    invocation_id: str

    def __init__(self, debug=False):
        self.debug = debug
        self.invocation_id = str(uuid4())


class MainGroup(click.Group):
    """A custom implementation of the top-level group
    (i.e. called on all commands and subcommands).

    This implementation allows for centralized behavior, including
    exception handling.
    """

    _exception_handler = ApplicationExceptionHandler()

    _tracer = get_tracer(__name__)

    def invoke(self, ctx):
        execution_info = ExecutionInfo(debug=ctx.params["debug"])
        qualified_name = " ".join([ctx.info_name] + argv[1:])
        invocation_id = execution_info.invocation_id
        set_debug_logging(execution_info.debug)

        with self._tracer.start_as_current_span(
            qualified_name, attributes={"invocation_id": invocation_id}
        ):
            try:
                log.debug(
                    f"Executing command: {qualified_name}",
                    command=qualified_name,
                )
                return super().invoke(ctx)
            except Exception as exception:
                log.error(exception)
                if execution_info.debug:
                    # Here we supress detailed errors on click lines because
                    # they're mostly decorator calls, irrelevant to the dev's error tracing
                    console.print_exception(suppress=[click])
                    console.print()
                    console.print(
                        f"The [markdown.code]invocation_id[/] for this operation is: [white]{invocation_id}[/]"
                    )
                else:
                    self._exception_handler.handle(exception)

    def add_command(
        self,
        cmd: click.Command,
        name: str | None = None,
        aliases: list[str] | None = None,
    ) -> None:
        name = name or cmd.name
        assert name, "Command must have a name"

        if not aliases:
            aliases = []

        if aliases:
            # Add aliases to the help text
            aliases_str = "Alias: " + ", ".join([name, *aliases])
            cmd.help = (cmd.help or "") + "\n\nAlias: " + ", ".join([name, *aliases])
            cmd.short_help = (
                (cmd.short_help or "") + "(Alias: " + ", ".join(aliases) + ")"
            )

        super().add_command(cmd, name)
        alias_cmd = AliasCommand(cmd)

        for alias in aliases:
            self.add_command(alias_cmd, alias)


class AliasCommand(click.Command):
    def __init__(self, wrapped):
        self._wrapped = wrapped

    def __getattribute__(self, __name: str):
        if __name == "_wrapped":
            # To be able to call `self._wrapped` below
            return super().__getattribute__(__name)

        if __name == "hidden":
            return True

        return self._wrapped.__getattribute__(__name)


@click.group(cls=MainGroup)
@click.option(
    "--debug", is_flag=True, help="Enable detailed errors and verbose logging."
)
@click.version_option()
def cli(debug):
    pass


###### Auth group ######
@click.group
def auth_cli():
    pass


@auth_cli.command(name="login")
def auth_login():
    auth.login()
    try:
        client = sdk.FalServerlessClient(f"{DEFAULT_HOST}:{DEFAULT_PORT}")
        with client.connect() as connection:
            connection.list_aliases()
    except grpc.RpcError as e:
        if "Insufficient permissions" in e.details():
            console.print(e.details())
        else:
            raise e


@auth_cli.command(name="logout")
def auth_logout():
    auth.logout()


@auth_cli.command(name="hello", hidden=True)
def auth_test():
    """
    To test auth.
    """
    print(f"Hello, {auth.USER.info['name']} - '{auth.USER.info['sub']}'")


###### Key group ######
@click.group
@click.option("--host", default=DEFAULT_HOST, envvar=HOST_ENVVAR)
@click.option("--port", default=DEFAULT_PORT, envvar=PORT_ENVVAR, hidden=True)
@click.pass_context
def key_cli(ctx, host: str, port: str):
    ctx.obj = sdk.FalServerlessClient(f"{host}:{port}")


@key_cli.command(name="generate", no_args_is_help=True)
@click.option(
    "--scope",
    default=None,
    required=True,
    type=click.Choice([KeyScope.ADMIN.value, KeyScope.API.value]),
    help="The privilage scope of the key.",
)
@click.option(
    "--alias",
    default=None,
    help="An alias for the key.",
)
@click.pass_obj
def key_generate(client: sdk.FalServerlessClient, scope: str, alias: str | None):
    with client.connect() as connection:
        parsed_scope = KeyScope(scope)
        result = connection.create_user_key(parsed_scope, alias)
        print(
            f"Generated key id and key secret, with the scope `{scope}`.\n"
            "This is the only time the secret will be visible.\n"
            "You will need to generate a new key pair if you lose access to this secret."
        )
        print(f"FAL_KEY_ID='{result[1]}'\nFAL_KEY_SECRET='{result[0]}'")


@key_cli.command(name="list")
@click.pass_obj
def key_list(client: sdk.FalServerlessClient):
    table = Table(title="Keys")
    table.add_column("Key ID")
    table.add_column("Created At")
    table.add_column("Scope")
    table.add_column("Alias")

    with client.connect() as connection:
        keys = connection.list_user_keys()
        for key in keys:
            table.add_row(
                key.key_id, str(key.created_at), str(key.scope.value), key.alias
            )

    console.print(table)


@key_cli.command(name="revoke")
@click.argument("key_id", required=True)
@click.pass_obj
def key_revoke(client: sdk.FalServerlessClient, key_id: str):
    with client.connect() as connection:
        connection.revoke_user_key(key_id)


##### Function group #####
ALIAS_AUTH_OPTIONS = ["public", "private", "shared"]
ALIAS_AUTH_TYPE = Literal["public", "private", "shared"]


@click.group
@click.option("--host", default=DEFAULT_HOST, envvar=HOST_ENVVAR)
@click.option("--port", default=DEFAULT_PORT, envvar=PORT_ENVVAR, hidden=True)
@click.pass_context
def function_cli(ctx, host: str, port: str):
    ctx.obj = api.FalServerlessHost(f"{host}:{port}")


@function_cli.command("serve")
@click.option("--alias", default=None)
@click.option(
    "--auth",
    "auth_mode",
    type=click.Choice(ALIAS_AUTH_OPTIONS),
    default="private",
)
@click.argument("file_path", required=True)
@click.argument("function_name", required=True)
@click.pass_obj
def register_application(
    host: api.FalServerlessHost,
    file_path: str,
    function_name: str,
    alias: str | None,
    auth_mode: ALIAS_AUTH_TYPE,
):
    import runpy

    user_id = _get_user_id()

    module = runpy.run_path(file_path)
    if function_name not in module:
        raise api.FalServerlessError(f"Function '{function_name}' not found in module")

    isolated_function: api.IsolatedFunction = module[function_name]
    gateway_options = isolated_function.options.gateway
    if "serve" not in gateway_options and "exposed_port" not in gateway_options:
        raise api.FalServerlessError(
            "One of `serve` or `exposed_port` options needs to be specified in the isolated annotation to register a function"
        )
    elif (
        "exposed_port" in gateway_options
        and str(gateway_options.get("exposed_port")) != "8080"
    ):
        raise api.FalServerlessError(
            "Must expose port 8080 for now. This will be configurable in the future."
        )

    id = host.register(
        func=isolated_function.func,
        options=isolated_function.options,
        application_name=alias,
        application_auth_mode=auth_mode,
        metadata={},
    )

    if id:
        # TODO: should we centralize this URL format?
        gateway_host = host.url.replace("api.", "gateway.")
        gateway_host = remove_http_and_port_from_url(gateway_host)

        if alias:
            console.print(
                f"Registered a new revision for function '{alias}' (revision='{id}')."
            )
            console.print(f"URL: https://{user_id}-{alias}.{gateway_host}")
        else:
            console.print(f"Registered anonymous function '{id}'.")
            console.print(f"URL: https://{user_id}-{id}.{gateway_host}")


@function_cli.command("logs")
@click.option("--lines", default=100)
@click.option("--url", default=None)
@click.pass_obj
def get_logs(
    host: api.FalServerlessHost, lines: int | None = 100, url: str | None = None
):
    log_printer = IsolateLogPrinter(debug=True)
    logs_response = list_logs.sync_detailed(
        client=REST_CLIENT, limit=lines, url_query=url
    )
    if not logs_response.status_code == 200 or type(logs_response.parsed) != list:
        raise api.FalServerlessError(str(logs_response.parsed))
    if len(logs_response.parsed) == 0:
        console.print("No logs found")
    for log in logs_response.parsed:
        app = log.app or "fal"

        log_printer.print(
            Log(
                message=f"{app}: {log.message}",
                source=LogSource.USER,
                level=LogLevel[log.level],
                timestamp=datetime.fromisoformat(log.timestamp),
            )
        )


##### Alias group #####
@click.group
@click.option("--host", default=DEFAULT_HOST, envvar=HOST_ENVVAR)
@click.option("--port", default=DEFAULT_PORT, envvar=PORT_ENVVAR, hidden=True)
@click.pass_context
def alias_cli(ctx, host: str, port: str):
    ctx.obj = api.FalServerlessClient(f"{host}:{port}")


def _alias_table(aliases: list[AliasInfo]):
    table = Table(title="Function Aliases")
    table.add_column("Alias")
    table.add_column("Revision")
    table.add_column("Auth")
    table.add_column("Max Concurrency")
    table.add_column("Max Multiplexing")
    table.add_column("Keep Alive")

    for app_alias in aliases:
        table.add_row(
            app_alias.alias,
            app_alias.revision,
            app_alias.auth_mode,
            str(app_alias.max_concurrency),
            str(app_alias.max_multiplexing),
            str(app_alias.keep_alive),
        )

    return table


@alias_cli.command("set")
@click.argument("alias", required=True)
@click.argument("revision", required=True)
@click.option(
    "--auth",
    "auth_mode",
    type=click.Choice(ALIAS_AUTH_OPTIONS),
    default="private",
)
@click.pass_obj
def alias_set(
    client: api.FalServerlessClient,
    alias: str,
    revision: str,
    auth_mode: ALIAS_AUTH_TYPE,
):
    with client.connect() as connection:
        connection.create_alias(alias, revision, auth_mode)


@alias_cli.command("delete")
@click.argument("alias", required=True)
@click.pass_obj
def alias_delete(client: api.FalServerlessClient, alias: str):
    with client.connect() as connection:
        application_id = connection.delete_alias(alias)

        console.print(f"Deleted alias '{alias}' for application '{application_id}'.")


@alias_cli.command("list")
@click.pass_obj
def alias_list(client: api.FalServerlessClient):
    with client.connect() as connection:
        aliases = connection.list_aliases()
        table = _alias_table(aliases)

    console.print(table)


@alias_cli.command("update")
@click.argument("alias", required=True)
@click.option("--keep-alive", "-k", type=int)
@click.option("--max-multiplexing", "-m", type=int)
@click.option("--max-concurrency", "-c", type=int)
# TODO: add auth_mode
# @click.option(
#     "--auth",
#     "auth_mode",
#     type=click.Choice(ALIAS_AUTH_OPTIONS),
# )
@click.pass_obj
def alias_update(
    client: api.FalServerlessClient,
    alias: str,
    keep_alive: int | None,
    max_multiplexing: int | None,
    max_concurrency: int | None,
):
    with client.connect() as connection:
        if keep_alive is None and max_multiplexing is None and max_concurrency is None:
            console.log("No parameters for update were provided, ignoring.")
            return

        alias_info = connection.update_application(
            application_name=alias,
            keep_alive=keep_alive,
            max_multiplexing=max_multiplexing,
            max_concurrency=max_concurrency,
        )
        table = _alias_table([alias_info])

    console.print(table)


@alias_cli.command("scale")
@click.argument("alias", required=True)
@click.argument("max_concurrency", required=True, type=int)
def alias_scale(alias: str, max_concurrency: int):
    alias_update.callback(
        alias=alias,
        keep_alive=None,
        max_multiplexing=None,
        max_concurrency=max_concurrency,
    )  # type: ignore


##### Secrets group #####
@click.group
@click.option("--host", default=DEFAULT_HOST, envvar=HOST_ENVVAR)
@click.option("--port", default=DEFAULT_PORT, envvar=PORT_ENVVAR, hidden=True)
@click.pass_context
def secrets_cli(ctx, host: str, port: str):
    ctx.obj = sdk.FalServerlessClient(f"{host}:{port}")


@secrets_cli.command("list")
@click.pass_obj
def list_secrets(client: api.FalServerlessClient):
    table = Table(title="Secrets")
    table.add_column("Secret Name")
    table.add_column("Created At")

    with client.connect() as connection:
        for secret in connection.list_secrets():
            table.add_row(secret.name, str(secret.created_at))

    console.print(table)


@secrets_cli.command("set")
@click.argument("secret_name", required=True)
@click.argument("secret_value", required=True)
@click.pass_obj
def set_secret(client: api.FalServerlessClient, secret_name: str, secret_value: str):
    with client.connect() as connection:
        connection.set_secret(secret_name, secret_value)
        console.print(f"Secret '{secret_name}' has set")


@secrets_cli.command("delete")
@click.argument("secret_name", required=True)
@click.pass_obj
def delete_secret(client: api.FalServerlessClient, secret_name: str):
    with client.connect() as connection:
        connection.delete_secret(secret_name)
        console.print(f"Secret '{secret_name}' has deleted")


# Setup of groups
cli.add_command(auth_cli, name="auth")
cli.add_command(key_cli, name="key", aliases=["keys"])
cli.add_command(function_cli, name="function", aliases=["fn"])
cli.add_command(alias_cli, name="alias", aliases=["aliases"])
cli.add_command(secrets_cli, name="secret", aliases=["secrets"])


def remove_http_and_port_from_url(url):
    # Remove http://
    if "http://" in url:
        url = url.replace("http://", "")

    # Remove https://
    if "https://" in url:
        url = url.replace("https://", "")

    # Remove port information
    url_parts = url.split(":")
    if len(url_parts) > 1:
        url = url_parts[0]

    return url


# dbt-fal commands to be errored out
DBT_FAL_COMMAND_NOTICE = """
The dbt tool `fal` and `dbt-fal` adapter have been merged into a single tool.
Please use the new `dbt-fal` command line tool instead.
Running `pip install dbt-fal` will install the new tool and the adapter alongside.
Then run your command like

    dbt-fal <command>

"""


@cli.command("run", context_settings={"ignore_unknown_options": True})
@click.argument("any", nargs=-1, type=click.UNPROCESSED)
def dbt_run(any):
    raise click.BadArgumentUsage(DBT_FAL_COMMAND_NOTICE)


@cli.command("flow", context_settings={"ignore_unknown_options": True})
@click.argument("any", nargs=-1, type=click.UNPROCESSED)
def dbt_flow(any):
    raise click.BadArgumentUsage(DBT_FAL_COMMAND_NOTICE)


def _get_user_id() -> str:
    try:
        user_details_response = get_user_details.sync_detailed(
            client=REST_CLIENT,
        )
    except Exception as e:
        raise api.FalServerlessError(f"Error fetching user details: {str(e)}")

    if user_details_response.status_code != HTTPStatus.OK:
        try:
            content = json.loads(user_details_response.content.decode("utf8"))
        except:
            raise api.FalServerlessError(
                f"Error fetching user details: {user_details_response}"
            )
        else:
            raise api.FalServerlessError(content["detail"])
    try:
        full_user_id = user_details_response.parsed.user_id
        user_id = full_user_id.split("|")[1]
        return user_id
    except Exception as e:
        raise api.FalServerlessError(f"Could not parse the user data: {e}")


if __name__ == "__main__":
    cli()
