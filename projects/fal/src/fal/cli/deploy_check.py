from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

from fal.flags import bool_envvar
from fal.logging import get_logger
from fal.sdk import construct_alias

logger = get_logger(__name__)

if TYPE_CHECKING:
    from fal.api.client import SyncServerlessClient
    from fal.api.deploy import DeploymentResult, PreparedDeployment
    from fal.sdk import AliasInfo

DeployCheckSource = Literal["flag", "env", "admin"]

# Mirrors the server-side scale inheritance contract.
INHERITED_SCALE_FIELDS = (
    "keep_alive",
    "max_concurrency",
    "min_concurrency",
    "concurrency_buffer",
    "concurrency_buffer_perc",
    "scaling_delay",
    "request_timeout",
    "valid_regions",
)
CODE_CONTROLLED_SCALE_FIELDS = (
    "machine_types",
    "max_multiplexing",
    "startup_timeout",
)
ALL_SCALE_FIELDS = (
    "machine_types",
    "keep_alive",
    "max_concurrency",
    "min_concurrency",
    "concurrency_buffer",
    "concurrency_buffer_perc",
    "scaling_delay",
    "max_multiplexing",
    "request_timeout",
    "startup_timeout",
    "valid_regions",
)
SERVER_DEFAULT_SCALE_VALUES = {
    "keep_alive": 10,
    "max_concurrency": 2,
    "min_concurrency": 0,
    "concurrency_buffer": 0,
    "concurrency_buffer_perc": 0,
    "scaling_delay": 0,
    "max_multiplexing": 1,
    "request_timeout": 3600,
    "startup_timeout": 600,
    "valid_regions": [],
    "machine_types": ["XS"],
}
SCALE_FIELD_LABELS = {
    "machine_types": "Machine Types",
    "keep_alive": "Keep Alive",
    "max_concurrency": "Max Concurrency",
    "min_concurrency": "Min Concurrency",
    "concurrency_buffer": "Concurrency Buffer",
    "concurrency_buffer_perc": "Concurrency Buffer %",
    "scaling_delay": "Scaling Delay",
    "max_multiplexing": "Max Multiplexing",
    "request_timeout": "Request Timeout",
    "startup_timeout": "Startup Timeout",
    "valid_regions": "Regions",
}
DEPLOY_CHECK_FLAG_KEYS = (
    "deploy_check",
    "serverless_deploy_check",
    "require_deploy_check",
    "require_deployment_confirmation",
    "deployment_confirmation",
)


@dataclass(frozen=True)
class DeploymentDiffRow:
    label: str
    production: str
    after_deploy: str
    note: str | None = None


@dataclass(frozen=True)
class DeploymentCheckSummary:
    source: DeployCheckSource
    app_name: str
    display_name: str
    environment_name: str | None
    current_revision: str | None
    current_auth_mode: str | None
    next_auth_mode: str
    strategy: str
    force_env_build: bool
    effective_changes: list[DeploymentDiffRow]
    effective_scale_values: list[DeploymentDiffRow]


def deploy_with_check(
    args,
    client: SyncServerlessClient,
    app_ref: tuple[str | None, str | None],
    *,
    force_env_build: bool,
    source: DeployCheckSource,
) -> DeploymentResult:
    from fal.api import deploy as deploy_api

    prepared = deploy_api.prepare_deployment(
        client,
        app_ref,
        app_name=args.app_name,
        auth=args.auth,
        strategy=args.strategy,
        reset_scale=args.app_scale_settings,
        force_env_build=force_env_build,
        environment_name=args.env,
    )
    production_alias = _get_production_alias(
        client,
        prepared.loaded.app_name,
        environment_name=args.env,
    )
    summary = _build_deployment_check_summary(
        prepared,
        production_alias,
        source=source,
        force_env_build=force_env_build,
    )
    _render_deployment_check_summary(args.console, summary)
    _confirm_deployment(
        summary.app_name,
        source,
        console=args.console,
        assume_yes=args.yes,
    )
    return deploy_api.execute_prepared_deployment(prepared)


def _resolve_deploy_check_source(
    args,
    client: SyncServerlessClient,
) -> DeployCheckSource | None:
    from fal.config import Config

    if getattr(args, "check", False):
        return "flag"

    if getattr(args, "yes", False):
        return None

    if bool_envvar("FAL_DEPLOY_CHECK"):
        return "env"

    if not (client.team or Config().get_internal("team")):
        return None

    if _admin_requires_deploy_check(client):
        return "admin"

    return None


def _admin_requires_deploy_check(client: SyncServerlessClient) -> bool:
    from http import HTTPStatus

    import httpx
    import openapi_fal_rest.api.users.get_current_user as get_current_user
    from openapi_fal_rest.errors import UnexpectedStatus

    try:
        response = get_current_user.sync_detailed(client=client._create_rest_client())
    except (UnexpectedStatus, httpx.HTTPError, ValueError):
        logger.warning("Failed to fetch deploy check admin policy", exc_info=True)
        return False

    if response.status_code != HTTPStatus.OK:
        logger.warning(
            "Failed to fetch deploy check admin policy",
            status_code=response.status_code,
        )
        return False

    if response.parsed is None:
        logger.warning(
            "Failed to fetch deploy check admin policy",
            reason="empty parsed payload",
        )
        return False

    return _payload_requires_deploy_check(response.parsed)


def _payload_requires_deploy_check(payload: Any) -> bool:
    for key in DEPLOY_CHECK_FLAG_KEYS:
        if _is_truthy(getattr(payload, key, None)):
            return True

    mapping = _as_mapping(payload)
    for key in DEPLOY_CHECK_FLAG_KEYS:
        if _is_truthy(mapping.get(key)):
            return True

    for container_key in ("org_config", "admin_config", "team_config", "settings"):
        nested = getattr(payload, container_key, None)
        if _nested_payload_requires_deploy_check(nested):
            return True

        if _nested_payload_requires_deploy_check(mapping.get(container_key)):
            return True

    return False


def _nested_payload_requires_deploy_check(payload: Any) -> bool:
    for key in DEPLOY_CHECK_FLAG_KEYS:
        if _is_truthy(getattr(payload, key, None)):
            return True

    mapping = _as_mapping(payload)
    for key in DEPLOY_CHECK_FLAG_KEYS:
        if _is_truthy(mapping.get(key)):
            return True
    return False


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value

    additional_properties = getattr(value, "additional_properties", None)
    if isinstance(additional_properties, dict):
        return additional_properties

    return {}


def _is_truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    if isinstance(value, int):
        return value != 0

    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "off"}

    return False


def _get_production_alias(
    client: SyncServerlessClient,
    app_name: str | None,
    *,
    environment_name: str | None = None,
) -> AliasInfo | None:
    if not app_name:
        return None

    full_alias = construct_alias(app_name, environment_name)
    for alias in client.apps.list(environment_name=environment_name):
        if alias.alias == full_alias:
            return alias

    return None


def _build_deployment_check_summary(
    prepared: PreparedDeployment,
    production_alias: AliasInfo | None,
    *,
    source: DeployCheckSource,
    force_env_build: bool,
) -> DeploymentCheckSummary:
    desired_config = _desired_scale_config(prepared)
    effective_config = _effective_scale_config(prepared, production_alias)
    effective_changes: list[DeploymentDiffRow] = []
    effective_scale_values: list[DeploymentDiffRow] = []

    if production_alias is None:
        effective_scale_values = [
            DeploymentDiffRow(
                label=SCALE_FIELD_LABELS[field],
                production="n/a",
                after_deploy=_format_setting_value(effective_config[field]),
            )
            for field in ALL_SCALE_FIELDS
        ]
    else:
        production_config = _production_scale_config(production_alias)
        for field in ALL_SCALE_FIELDS:
            production_value = production_config[field]
            effective_value = effective_config[field]
            desired_value = desired_config[field]

            if production_value != effective_value:
                effective_changes.append(
                    DeploymentDiffRow(
                        label=SCALE_FIELD_LABELS[field],
                        production=_format_setting_value(production_value),
                        after_deploy=_format_setting_value(effective_value),
                    )
                )
            elif (
                not prepared.app_data.reset_scale
                and field in INHERITED_SCALE_FIELDS
                and desired_value != production_value
            ):
                effective_changes.append(
                    DeploymentDiffRow(
                        label=SCALE_FIELD_LABELS[field],
                        production=_format_setting_value(production_value),
                        after_deploy=_format_setting_value(desired_value),
                        note="Code value will not apply without --reset-scale",
                    )
                )

    return DeploymentCheckSummary(
        source=source,
        app_name=prepared.loaded.app_name or prepared.display_name,
        display_name=prepared.display_name,
        environment_name=prepared.environment_name,
        current_revision=production_alias.revision if production_alias else None,
        current_auth_mode=production_alias.auth_mode if production_alias else None,
        next_auth_mode=prepared.loaded.app_auth or "private",
        strategy=prepared.app_data.deployment_strategy or "rolling",
        force_env_build=force_env_build,
        effective_changes=effective_changes,
        effective_scale_values=effective_scale_values,
    )


def _desired_scale_config(prepared: PreparedDeployment) -> dict[str, Any]:
    host_options = prepared.loaded.function.options.host
    return {
        "machine_types": _normalize_machine_types(
            host_options.get(
                "machine_type", SERVER_DEFAULT_SCALE_VALUES["machine_types"]
            )
        ),
        "keep_alive": host_options.get(
            "keep_alive", SERVER_DEFAULT_SCALE_VALUES["keep_alive"]
        ),
        "max_concurrency": host_options.get(
            "max_concurrency", SERVER_DEFAULT_SCALE_VALUES["max_concurrency"]
        ),
        "min_concurrency": host_options.get(
            "min_concurrency", SERVER_DEFAULT_SCALE_VALUES["min_concurrency"]
        ),
        "concurrency_buffer": host_options.get(
            "concurrency_buffer", SERVER_DEFAULT_SCALE_VALUES["concurrency_buffer"]
        ),
        "concurrency_buffer_perc": host_options.get(
            "concurrency_buffer_perc",
            SERVER_DEFAULT_SCALE_VALUES["concurrency_buffer_perc"],
        ),
        "scaling_delay": host_options.get(
            "scaling_delay", SERVER_DEFAULT_SCALE_VALUES["scaling_delay"]
        ),
        "max_multiplexing": host_options.get(
            "max_multiplexing", SERVER_DEFAULT_SCALE_VALUES["max_multiplexing"]
        ),
        "request_timeout": host_options.get(
            "request_timeout", SERVER_DEFAULT_SCALE_VALUES["request_timeout"]
        ),
        "startup_timeout": host_options.get(
            "startup_timeout", SERVER_DEFAULT_SCALE_VALUES["startup_timeout"]
        ),
        "valid_regions": _normalize_regions(
            host_options.get("regions", SERVER_DEFAULT_SCALE_VALUES["valid_regions"])
        ),
    }


def _production_scale_config(production_alias: AliasInfo) -> dict[str, Any]:
    return {
        "machine_types": list(production_alias.machine_types),
        "keep_alive": production_alias.keep_alive,
        "max_concurrency": production_alias.max_concurrency,
        "min_concurrency": production_alias.min_concurrency,
        "concurrency_buffer": production_alias.concurrency_buffer,
        "concurrency_buffer_perc": production_alias.concurrency_buffer_perc,
        "scaling_delay": production_alias.scaling_delay,
        "max_multiplexing": production_alias.max_multiplexing,
        "request_timeout": production_alias.request_timeout,
        "startup_timeout": production_alias.startup_timeout,
        "valid_regions": list(production_alias.valid_regions),
    }


def _effective_scale_config(
    prepared: PreparedDeployment,
    production_alias: AliasInfo | None,
) -> dict[str, Any]:
    desired_config = _desired_scale_config(prepared)
    if production_alias is None or prepared.app_data.reset_scale:
        return desired_config

    production_config = _production_scale_config(production_alias)
    effective_config = dict(desired_config)
    for field in INHERITED_SCALE_FIELDS:
        effective_config[field] = production_config[field]

    return effective_config


def _normalize_machine_types(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]

    if isinstance(value, list):
        return [str(item) for item in value]

    default_machine_types = cast(
        list[Any], SERVER_DEFAULT_SCALE_VALUES["machine_types"]
    )
    return [str(item) for item in default_machine_types]


def _normalize_regions(value: Any) -> list[str]:
    if value is None:
        return []

    if isinstance(value, list):
        return [str(item) for item in value]

    return []


def _format_setting_value(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(value) if value else "any"

    if value is None:
        return "none"

    return str(value)


def _render_auth_line(current_auth_mode: str | None, next_auth_mode: str):
    from rich.text import Text

    rendered_current_auth = current_auth_mode or "none"
    auth_value = f"{rendered_current_auth} -> {next_auth_mode}"

    line = Text()
    line.append("Auth: ", style="bold")
    line.append(
        auth_value,
        style="red" if current_auth_mode != next_auth_mode else None,
    )
    return line


def _render_deployment_strategy_line(strategy: str):
    from rich.text import Text

    line = Text()
    line.append("Deployment strategy: ", style="bold")
    line.append(strategy, style="green" if strategy == "rolling" else "red")
    line.append(" (this deployment only)")
    return line


def _render_environment_build_cache_line(force_env_build: bool):
    from rich.text import Text

    line = Text()
    line.append("Environment build cache: ", style="bold")
    line.append(
        "disabled (--no-cache)" if force_env_build else "enabled",
        style="#ff8800" if force_env_build else "green",
    )
    return line


def _render_deployment_check_summary(console, summary: DeploymentCheckSummary) -> None:
    from rich.rule import Rule

    console.print("")
    console.print(Rule(f"Deployment Check: {summary.app_name}", style="yellow"))
    console.print(
        f"[bold]Target:[/bold] {summary.app_name} "
        f"(env: {summary.environment_name or 'main'})"
    )
    console.print(
        f"[bold]Current revision:[/bold] "
        f"{summary.current_revision or 'none (first deployment)'}"
    )
    console.print(f"[bold]Source object:[/bold] {summary.display_name}")
    console.print(_render_auth_line(summary.current_auth_mode, summary.next_auth_mode))
    console.print(_render_deployment_strategy_line(summary.strategy))
    console.print(_render_environment_build_cache_line(summary.force_env_build))

    if summary.effective_changes:
        console.print("")
        console.print(
            _diff_table("Effective Production Diff", summary.effective_changes)
        )
    elif summary.current_revision is not None:
        console.print("")
        console.print("[dim]No production-facing config changes detected.[/dim]")

    if summary.effective_scale_values:
        console.print("")
        console.print(
            _diff_table(
                "Effective Deployment Values",
                summary.effective_scale_values,
            )
        )

    console.print(Rule("", style="yellow"))


def _diff_table(
    title: str,
    rows: list[DeploymentDiffRow],
):
    from rich.table import Table
    from rich.text import Text

    table = Table(title=title)
    table.add_column("Setting", style="bold")
    table.add_column("Production")
    table.add_column("After Deploy")

    include_note = any(row.note for row in rows)
    if include_note:
        table.add_column("Note")

    for row in rows:
        after_deploy = (
            Text(row.after_deploy, style="yellow")
            if row.note
            else Text(row.after_deploy)
        )
        values = [row.label, row.production, after_deploy]
        if include_note:
            values.append(row.note or "")
        table.add_row(*values)

    return table


def _confirm_deployment(
    app_name: str,
    source: DeployCheckSource,
    *,
    console,
    assume_yes: bool = False,
) -> None:
    if assume_yes:
        return

    if not sys.stdin.isatty():
        console.print(
            f"Deploy confirmation for '{app_name}' requires interactive input. "
            "Re-run with --yes to bypass the prompt."
        )
        raise SystemExit(1)

    confirmation = input("Type 'confirm' to confirm deployment: ")
    if confirmation.strip().lower() != "confirm":
        console.print("Deployment aborted.")
        raise SystemExit(1)
