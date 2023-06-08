from __future__ import annotations

import datetime
import operator
from sys import argv
from typing import Literal
from uuid import uuid4

import click
import fal_serverless.auth as auth
import grpc
from fal_serverless import api, sdk
from fal_serverless.console import console
from fal_serverless.exceptions import ApplicationExceptionHandler
from fal_serverless.logging import get_logger, set_debug_logging
from fal_serverless.logging.isolate import IsolateLogPrinter
from fal_serverless.logging.trace import get_tracer
from fal_serverless.sync import list_children, parse_logs
from rich.table import Table

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
            connection.list_worker_status()
    except grpc.RpcError as e:
        if "Insufficient permissions" in e.details():
            console.print(e.details())
        else:
            raise e


@auth_cli.command(name="logout")
def auth_logout():
    auth.logout()


@auth_cli.command(name="refresh")
def auth_refresh():
    auth.refresh()


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


@key_cli.command(name="generate")
@click.pass_obj
def key_generate(client: sdk.FalServerlessClient):
    with client.connect() as connection:
        result = connection.create_user_key()
        print(
            "Generated key id and key secret.\n"
            "This is the only time the secret will be visible.\n"
            "You will need to generate a new key pair if you lose access to this secret."
        )
        print(f"KEY_ID='{result[1]}'\nKEY_SECRET='{result[0]}'")


@key_cli.command(name="list")
@click.pass_obj
def key_list(client: sdk.FalServerlessClient):
    table = Table(title="Keys")
    table.add_column("Key ID")
    table.add_column("Created At")
    with client.connect() as connection:
        keys = connection.list_user_keys()
        for key in keys:
            table.add_row(key.key_id, str(key.created_at))

    console.print(table)


@key_cli.command(name="revoke")
@click.argument("key_id", required=True)
@click.pass_obj
def key_revoke(client: sdk.FalServerlessClient, key_id: str):
    with client.connect() as connection:
        connection.revoke_user_key(key_id)


###### Usage group ######
@click.group
@click.option("--host", default=DEFAULT_HOST, envvar=HOST_ENVVAR)
@click.option("--port", default=DEFAULT_PORT, envvar=PORT_ENVVAR, hidden=True)
@click.pass_context
def usage_cli(ctx, host: str, port: str):
    ctx.obj = sdk.FalServerlessClient(f"{host}:{port}")


@usage_cli.command(name="workers")
@click.option("--user", hidden=True, default=None)
@click.pass_obj
def usage_worker_status(client: sdk.FalServerlessClient, user: str | None):
    table = Table(title="Worker status")
    table.add_column("Worker ID")
    table.add_column("User ID")
    table.add_column("Machine type")
    table.add_column("Start time")
    table.add_column("End time")
    table.add_column("Duration")

    with client.connect() as connection:
        for ws in connection.list_worker_status(user_id=user):
            table.add_row(
                ws.worker_id,
                ws.user_id,
                ws.machine_type,
                str(ws.start_time),
                str(ws.end_time),
                str(ws.duration),
            )

    console.print(table)


##### Function group #####
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
    # TODO: handle shared auth mode
    type=click.Choice(["public", "private"]),
    default="private",
)
@click.argument("file_path", required=True)
@click.argument("function_name", required=True)
@click.pass_obj
def register_application(
    host: api.FalServerlessHost,
    file_path: str,
    function_name: str,
    auth_mode: Literal["public", "private"],
    alias: str | None = None,
):
    import runpy

    module = runpy.run_path(file_path)
    isolated_function = module[function_name]
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
        application_is_public=auth_mode == "public",
    )
    if id:
        # TODO: should we centralize this URL format?
        gateway_host = host.url.replace("api.", "gateway.")
        gateway_host = remove_http_and_port_from_url(gateway_host)

        # Encode since user_id can contain special characters
        user_id = auth.USER.info["sub"].split("|")[1]
        if alias:
            console.print(
                f"Registered a new revision for function '{alias}' (revision='{id}')."
            )
            console.print(f"URL: https://{user_id}-{alias}.{gateway_host}")
        else:
            console.print(f"Registered anonymous function '{id}'.")
            console.print(f"URL: https://{user_id}-{id}.{gateway_host}")


@function_cli.command("schedule")
@click.argument("cron_string", required=True)
@click.argument("file_path", required=True)
@click.argument("function_name", required=True)
@click.pass_obj
def register_schedulded(
    host: api.FalServerlessHost, cron_string: str, file_path: str, function_name: str
):
    import runpy

    module = runpy.run_path(file_path)
    isolated_function = module[function_name]

    cron_id = host.schedule(
        func=isolated_function.func, cron=cron_string, options=isolated_function.options
    )
    if cron_id:
        console.print(cron_id)


@function_cli.command("logs")
@click.argument("url", required=True)
@click.argument("call_id", required=False)
@click.pass_obj
def get_logs(host: api.FalServerlessHost, url: str, call_id: str | None):
    url = remove_http_and_port_from_url(url)
    if call_id is None:
        calls = list_children(f"/data/logs/gateway/{url}")
        if not calls:
            console.print(
                f"Web endpoint {url} has not been called yet. Logs are not available."
            )
            return
        else:
            sorted_calls = sorted(calls, key=lambda x: x["updated_time"], reverse=True)
            call_id = sorted_calls[0].get("name")
    logs = parse_logs(f"/data/logs/gateway/{url}/{call_id}")
    log_printer = IsolateLogPrinter(debug=True)
    for log in logs:
        log_printer.print_dict(log)


@function_cli.command("calls")
@click.argument("url", required=True)
@click.pass_obj
def get_function_call_ids(host: api.FalServerlessHost, url: str):
    # This will only return a list calls that we have logs for.
    url = remove_http_and_port_from_url(url)
    calls = list_children(f"/data/logs/gateway/{url}")
    calls.sort(key=operator.itemgetter("updated_time"))
    for call in calls:
        name = call["name"].split(".")[0]
        timestamp = datetime.datetime.fromtimestamp(call["updated_time"])
        timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        console.print(f"{timestamp_str}: {name}")


##### Alias group #####
@click.group
@click.option("--host", default=DEFAULT_HOST, envvar=HOST_ENVVAR)
@click.option("--port", default=DEFAULT_PORT, envvar=PORT_ENVVAR, hidden=True)
@click.pass_context
def alias_cli(ctx, host: str, port: str):
    ctx.obj = api.FalServerlessClient(f"{host}:{port}")


@alias_cli.command("list")
@click.pass_obj
def alias_list(client: api.FalServerlessClient):
    with client.connect() as connection:
        table = Table(title="Function Aliases")
        table.add_column("Alias")
        table.add_column("Revision")
        table.add_column("Auth")

        for app_alias in connection.list_aliases():
            table.add_row(
                app_alias.alias,
                app_alias.revision,
                "public" if app_alias.public else "private",
            )

    console.print(table)


##### Crons group #####
@click.group
@click.option("--host", default=DEFAULT_HOST, envvar=HOST_ENVVAR)
@click.option("--port", default=DEFAULT_PORT, envvar=PORT_ENVVAR, hidden=True)
@click.pass_context
def crons_cli(ctx, host: str, port: str):
    ctx.obj = api.FalServerlessHost(f"{host}:{port}")


@crons_cli.command(name="list")
@click.pass_obj
def list_scheduled(host: api.FalServerlessHost):
    table = Table(title="Cronjobs")
    table.add_column("Cron ID")
    table.add_column("Cron")
    table.add_column("ETA next run")
    table.add_column("State")
    for cron in host._connection.list_scheduled_runs():
        state_string = ["Not Active", "Active"][cron.active]
        table.add_row(cron.cron_id, cron.cron_string, str(cron.next_run), state_string)

    console.print(table)


@crons_cli.command(name="activations")
@click.argument("cron_id", required=True)
@click.argument("limit", default=15)
@click.pass_obj
def list_activations(host: api.FalServerlessHost, cron_id: str, limit: int = 15):
    table = Table(title="Cron activations")
    table.add_column("Activation ID")
    table.add_column("Start Date")
    table.add_column("Finish Date")

    for activation in host._connection.list_run_activations(cron_id)[:limit]:
        table.add_row(
            str(activation.activation_id),
            str(activation.started_at),
            str(activation.finished_at),
        )

    console.print(table)


@crons_cli.command(name="logs")
@click.argument("cron_id", required=True)
@click.argument("activation_id", required=True)
@click.pass_obj
def print_logs(host: api.FalServerlessHost, cron_id: str, activation_id: str):
    logs = host._connection.get_activation_logs(cron_id, activation_id)
    if not logs:
        console.print(f"No logs found for activation {activation_id}")
        return
    log_printer = IsolateLogPrinter(debug=True)
    for log in logs:
        log_printer.print(log)


@crons_cli.command("cancel")
@click.argument("cron_id", required=True)
@click.pass_obj
def cancel_scheduled(host: api.FalServerlessHost, cron_id: str):
    host._connection.cancel_scheduled_run(cron_id)
    console.print("Cancelled", repr(cron_id))


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


cli.add_command(auth_cli, name="auth")
cli.add_command(key_cli, name="key", aliases=["keys"])
cli.add_command(function_cli, name="function", aliases=["fn"])
cli.add_command(alias_cli, name="alias")
cli.add_command(crons_cli, name="cron", aliases=["crons"])
cli.add_command(usage_cli, name="usage")
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


if __name__ == "__main__":
    cli()
