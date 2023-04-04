from __future__ import annotations

from sys import argv
from urllib.parse import quote as urlquote
from uuid import uuid4

import click
import fal_serverless.auth as auth
from fal_serverless import api, sdk
from fal_serverless.console import console
from fal_serverless.exceptions import ApplicationExceptionHandler
from rich.table import Table

from .logging import get_logger, set_debug_logging
from .logging.trace import get_tracer

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
                    f"Executing command",
                    command=qualified_name,
                    invocation_id=invocation_id,
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


@click.group(cls=MainGroup)
@click.option(
    "--debug", is_flag=True, help="Enable detailed errors and verbose logging."
)
@click.version_option()
def cli(debug):
    pass


###### Auth group ######
@cli.group("auth")
def auth_cli():
    pass


@auth_cli.command(name="login")
def auth_login():
    auth.login()


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
@cli.group("key")
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
@click.argument("key-id", required=True)
@click.pass_obj
def key_revoke(client: sdk.FalServerlessClient, key_id: str):
    with client.connect() as connection:
        connection.revoke_user_key(key_id)


###### Scheduled group ######
@cli.group("scheduled", hidden=True)
@click.option("--host", default=DEFAULT_HOST, envvar=HOST_ENVVAR)
@click.option("--port", default=DEFAULT_PORT, envvar=PORT_ENVVAR, hidden=True)
@click.pass_context
def scheduled_cli(ctx, host: str, port: str):
    ctx.obj = api.FalServerlessHost(f"{host}:{port}")


@scheduled_cli.command("register")
@click.argument("cron_string", required=True)
@click.argument("file_path", required=True)
@click.argument("function_name", required=True)
@click.pass_obj
def register_schedulded(
    client: api.FalServerlessHost, cron_string: str, file_path: str, function_name: str
):
    import runpy

    module = runpy.run_path(file_path)
    isolated_function = module[function_name]

    cron_id = client.schedule(
        func=isolated_function.func, cron=cron_string, options=isolated_function.options
    )
    if cron_id:
        console.print(cron_id)


@scheduled_cli.command(name="list")
@click.pass_obj
def list_scheduled(client: api.FalServerlessHost):
    table = Table(title="Cronjobs")
    table.add_column("Cron ID")
    table.add_column("Cron")
    table.add_column("ETA next run")
    table.add_column("State")

    for cron in client._connection.list_scheduled_runs():
        state_string = ["Not Active", "Active"][cron.active]
        table.add_row(cron.cron_id, cron.cron_string, str(cron.next_run), state_string)

    console.print(table)


@scheduled_cli.command(name="activations")
@click.argument("cron-id", required=True)
@click.argument("limit", default=15)
@click.pass_obj
def list_activations(client: api.FalServerlessHost, cron_id: str, limit: int = 15):
    table = Table(title="Cron activations")
    table.add_column("Cron ID")
    table.add_column("Activation ID")
    table.add_column("Activation Start Date")
    table.add_column("Activation Finish Date")

    for activation in client._connection.list_run_activations(cron_id)[-limit:]:
        table.add_row(
            cron_id,
            str(activation.activation_id),
            str(activation.started_at),
            str(activation.finished_at),
        )

    console.print(table)


@scheduled_cli.command(name="logs")
@click.argument("cron_id", required=True)
@click.argument("activation-id", required=True)
@click.pass_obj
def print_logs(client: api.FalServerlessHost, cron_id: str, activation_id: str):
    raw_logs = client._connection.get_activation_logs(activation_id)
    console.print(raw_logs.decode(errors="ignore"), highlight=False)


@scheduled_cli.command("cancel")
@click.argument("cron_id", required=True)
@click.pass_obj
def cancel_scheduled(client: api.FalServerlessHost, cron_id: str):
    client._connection.cancel_scheduled_run(cron_id)
    console.print("Cancelled", repr(cron_id))


###### Usage group ######
@cli.group("usage")
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


@cli.command("register")
@click.option("--host", default=DEFAULT_HOST, envvar=HOST_ENVVAR)
@click.option("--port", default=DEFAULT_PORT, envvar=PORT_ENVVAR, hidden=True)
@click.argument("file_path", required=True)
@click.argument("function_name", required=True)
def register_application(host: str, port: str, file_path: str, function_name: str):
    import runpy

    module = runpy.run_path(file_path)
    isolated_function = module[function_name]
    gateway_options = isolated_function.options.gateway
    if "serve" not in gateway_options and "exposed_port" not in gateway_options:
        raise api.FalServerlessError(
            "One of `serve` or `exposed-port` options needs to be specified in the isolated annotation to register a function"
        )
    id = api.FalServerlessHost(f"{host}:{port}").register(
        func=isolated_function.func, options=isolated_function.options
    )
    if id:
        console.print(f"Registered function {id}")

        # TODO: should we centralize this URL format?
        gateway_host = host.replace("api.", "gateway.")
        user_id = auth.USER.info["sub"]
        app_url = f"http://{gateway_host}/trigger/{user_id}/{id}/"
        # Encode since user_id can contain special characters
        console.print(f"Function URL: {urlquote(app_url)}")


cli.add_command(auth_cli, name="auth")
cli.add_command(key_cli, name="key")
cli.add_command(scheduled_cli, name="scheduled")
cli.add_command(usage_cli, name="usage")
cli.add_command(register_application, name="register")


if __name__ == "__main__":
    cli()
