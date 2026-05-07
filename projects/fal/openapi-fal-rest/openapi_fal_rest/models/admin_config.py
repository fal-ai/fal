from typing import TYPE_CHECKING, Any, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.admin_config_burst_billing_config import AdminConfigBurstBillingConfig
    from ..models.admin_config_concurrency_surcharge import AdminConfigConcurrencySurcharge
    from ..models.admin_config_concurrent_requests_limit_per_endpoint import (
        AdminConfigConcurrentRequestsLimitPerEndpoint,
    )
    from ..models.admin_config_max_registered_max_gpus_by_type import AdminConfigMaxRegisteredMaxGpusByType
    from ..models.admin_config_serverless_surge_thresholds import AdminConfigServerlessSurgeThresholds


T = TypeVar("T", bound="AdminConfig")


@_attrs_define
class AdminConfig:
    """
    Attributes:
        queue_priority (Union[Unset, str]):
        deprioritize_queue (Union[Unset, bool]):
        rate_limit_exempt (Union[Unset, bool]):
        deploy_shared_app (Union[Unset, bool]):
        enable_per_user_auth (Union[Unset, bool]):
        purchase_limit_exempt (Union[Unset, bool]):
        max_registered_max_concurrency (Union[Unset, int]):
        max_registered_min_concurrency (Union[Unset, int]):
        max_registered_max_gpus (Union[Unset, int]):
        max_registered_max_gpus_by_type (Union[Unset, AdminConfigMaxRegisteredMaxGpusByType]):
        max_registered_keepalive (Union[Unset, int]):
        max_registered_request_timeout (Union[Unset, int]):
        max_registered_startup_timeout (Union[Unset, int]):
        cdn_token_generation (Union[Unset, bool]):
        beta_access (Union[Unset, bool]):
        pricing_access (Union[Unset, bool]):
        compute_access (Union[Unset, bool]):
        lock_limit (Union[Unset, int]):
        concurrent_requests_limit (Union[Unset, int]):
        concurrent_requests_limit_per_endpoint (Union[Unset, AdminConfigConcurrentRequestsLimitPerEndpoint]):
        is_org (Union[Unset, bool]):  Default: False.
        unlimited_teams (Union[Unset, bool]):
        spending_alerts (Union[Unset, bool]):
        spending_alerts_v2 (Union[Unset, bool]):
        spending_locks (Union[Unset, bool]):
        focus_reports (Union[Unset, bool]):
        allow_net_admin (Union[Unset, bool]):
        allow_snapshots (Union[Unset, bool]):
        allow_sys_resource (Union[Unset, bool]):
        allow_sys_ptrace (Union[Unset, bool]):
        enriched_invoices (Union[Unset, bool]):
        endpoint_access_controls (Union[Unset, bool]):
        disable_fallbacks (Union[Unset, bool]):
        owned_sso_connections (Union[Unset, list[str]]):
        model_status_access (Union[Unset, bool]):
        concurrency_surcharge (Union[Unset, AdminConfigConcurrencySurcharge]):
        max_credit_amount_usd (Union[Unset, float]):
        enable_weekly_billing_reports (Union[Unset, bool]):
        serverless_surge_thresholds (Union[Unset, AdminConfigServerlessSurgeThresholds]):
        burst_billing_config (Union[Unset, AdminConfigBurstBillingConfig]):
        enable_payload_embedding (Union[Unset, bool]):
        enable_ui_payload_embedding (Union[Unset, bool]):
        can_disable_filters (Union[Unset, bool]):
        default_private_logs (Union[Unset, bool]):
        skip_runner_connection_check (Union[Unset, bool]):
        is_early_access (Union[Unset, bool]):
    """

    queue_priority: Union[Unset, str] = UNSET
    deprioritize_queue: Union[Unset, bool] = UNSET
    rate_limit_exempt: Union[Unset, bool] = UNSET
    deploy_shared_app: Union[Unset, bool] = UNSET
    enable_per_user_auth: Union[Unset, bool] = UNSET
    purchase_limit_exempt: Union[Unset, bool] = UNSET
    max_registered_max_concurrency: Union[Unset, int] = UNSET
    max_registered_min_concurrency: Union[Unset, int] = UNSET
    max_registered_max_gpus: Union[Unset, int] = UNSET
    max_registered_max_gpus_by_type: Union[Unset, "AdminConfigMaxRegisteredMaxGpusByType"] = UNSET
    max_registered_keepalive: Union[Unset, int] = UNSET
    max_registered_request_timeout: Union[Unset, int] = UNSET
    max_registered_startup_timeout: Union[Unset, int] = UNSET
    cdn_token_generation: Union[Unset, bool] = UNSET
    beta_access: Union[Unset, bool] = UNSET
    pricing_access: Union[Unset, bool] = UNSET
    compute_access: Union[Unset, bool] = UNSET
    lock_limit: Union[Unset, int] = UNSET
    concurrent_requests_limit: Union[Unset, int] = UNSET
    concurrent_requests_limit_per_endpoint: Union[Unset, "AdminConfigConcurrentRequestsLimitPerEndpoint"] = UNSET
    is_org: Union[Unset, bool] = False
    unlimited_teams: Union[Unset, bool] = UNSET
    spending_alerts: Union[Unset, bool] = UNSET
    spending_alerts_v2: Union[Unset, bool] = UNSET
    spending_locks: Union[Unset, bool] = UNSET
    focus_reports: Union[Unset, bool] = UNSET
    allow_net_admin: Union[Unset, bool] = UNSET
    allow_snapshots: Union[Unset, bool] = UNSET
    allow_sys_resource: Union[Unset, bool] = UNSET
    allow_sys_ptrace: Union[Unset, bool] = UNSET
    enriched_invoices: Union[Unset, bool] = UNSET
    endpoint_access_controls: Union[Unset, bool] = UNSET
    disable_fallbacks: Union[Unset, bool] = UNSET
    owned_sso_connections: Union[Unset, list[str]] = UNSET
    model_status_access: Union[Unset, bool] = UNSET
    concurrency_surcharge: Union[Unset, "AdminConfigConcurrencySurcharge"] = UNSET
    max_credit_amount_usd: Union[Unset, float] = UNSET
    enable_weekly_billing_reports: Union[Unset, bool] = UNSET
    serverless_surge_thresholds: Union[Unset, "AdminConfigServerlessSurgeThresholds"] = UNSET
    burst_billing_config: Union[Unset, "AdminConfigBurstBillingConfig"] = UNSET
    enable_payload_embedding: Union[Unset, bool] = UNSET
    enable_ui_payload_embedding: Union[Unset, bool] = UNSET
    can_disable_filters: Union[Unset, bool] = UNSET
    default_private_logs: Union[Unset, bool] = UNSET
    skip_runner_connection_check: Union[Unset, bool] = UNSET
    is_early_access: Union[Unset, bool] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        queue_priority = self.queue_priority

        deprioritize_queue = self.deprioritize_queue

        rate_limit_exempt = self.rate_limit_exempt

        deploy_shared_app = self.deploy_shared_app

        enable_per_user_auth = self.enable_per_user_auth

        purchase_limit_exempt = self.purchase_limit_exempt

        max_registered_max_concurrency = self.max_registered_max_concurrency

        max_registered_min_concurrency = self.max_registered_min_concurrency

        max_registered_max_gpus = self.max_registered_max_gpus

        max_registered_max_gpus_by_type: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.max_registered_max_gpus_by_type, Unset):
            max_registered_max_gpus_by_type = self.max_registered_max_gpus_by_type.to_dict()

        max_registered_keepalive = self.max_registered_keepalive

        max_registered_request_timeout = self.max_registered_request_timeout

        max_registered_startup_timeout = self.max_registered_startup_timeout

        cdn_token_generation = self.cdn_token_generation

        beta_access = self.beta_access

        pricing_access = self.pricing_access

        compute_access = self.compute_access

        lock_limit = self.lock_limit

        concurrent_requests_limit = self.concurrent_requests_limit

        concurrent_requests_limit_per_endpoint: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.concurrent_requests_limit_per_endpoint, Unset):
            concurrent_requests_limit_per_endpoint = self.concurrent_requests_limit_per_endpoint.to_dict()

        is_org = self.is_org

        unlimited_teams = self.unlimited_teams

        spending_alerts = self.spending_alerts

        spending_alerts_v2 = self.spending_alerts_v2

        spending_locks = self.spending_locks

        focus_reports = self.focus_reports

        allow_net_admin = self.allow_net_admin

        allow_snapshots = self.allow_snapshots

        allow_sys_resource = self.allow_sys_resource

        allow_sys_ptrace = self.allow_sys_ptrace

        enriched_invoices = self.enriched_invoices

        endpoint_access_controls = self.endpoint_access_controls

        disable_fallbacks = self.disable_fallbacks

        owned_sso_connections: Union[Unset, list[str]] = UNSET
        if not isinstance(self.owned_sso_connections, Unset):
            owned_sso_connections = self.owned_sso_connections

        model_status_access = self.model_status_access

        concurrency_surcharge: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.concurrency_surcharge, Unset):
            concurrency_surcharge = self.concurrency_surcharge.to_dict()

        max_credit_amount_usd = self.max_credit_amount_usd

        enable_weekly_billing_reports = self.enable_weekly_billing_reports

        serverless_surge_thresholds: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.serverless_surge_thresholds, Unset):
            serverless_surge_thresholds = self.serverless_surge_thresholds.to_dict()

        burst_billing_config: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.burst_billing_config, Unset):
            burst_billing_config = self.burst_billing_config.to_dict()

        enable_payload_embedding = self.enable_payload_embedding

        enable_ui_payload_embedding = self.enable_ui_payload_embedding

        can_disable_filters = self.can_disable_filters

        default_private_logs = self.default_private_logs

        skip_runner_connection_check = self.skip_runner_connection_check

        is_early_access = self.is_early_access

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if queue_priority is not UNSET:
            field_dict["queue_priority"] = queue_priority
        if deprioritize_queue is not UNSET:
            field_dict["deprioritize_queue"] = deprioritize_queue
        if rate_limit_exempt is not UNSET:
            field_dict["rate_limit_exempt"] = rate_limit_exempt
        if deploy_shared_app is not UNSET:
            field_dict["deploy_shared_app"] = deploy_shared_app
        if enable_per_user_auth is not UNSET:
            field_dict["enable_per_user_auth"] = enable_per_user_auth
        if purchase_limit_exempt is not UNSET:
            field_dict["purchase_limit_exempt"] = purchase_limit_exempt
        if max_registered_max_concurrency is not UNSET:
            field_dict["max_registered_max_concurrency"] = max_registered_max_concurrency
        if max_registered_min_concurrency is not UNSET:
            field_dict["max_registered_min_concurrency"] = max_registered_min_concurrency
        if max_registered_max_gpus is not UNSET:
            field_dict["max_registered_max_gpus"] = max_registered_max_gpus
        if max_registered_max_gpus_by_type is not UNSET:
            field_dict["max_registered_max_gpus_by_type"] = max_registered_max_gpus_by_type
        if max_registered_keepalive is not UNSET:
            field_dict["max_registered_keepalive"] = max_registered_keepalive
        if max_registered_request_timeout is not UNSET:
            field_dict["max_registered_request_timeout"] = max_registered_request_timeout
        if max_registered_startup_timeout is not UNSET:
            field_dict["max_registered_startup_timeout"] = max_registered_startup_timeout
        if cdn_token_generation is not UNSET:
            field_dict["cdn_token_generation"] = cdn_token_generation
        if beta_access is not UNSET:
            field_dict["beta_access"] = beta_access
        if pricing_access is not UNSET:
            field_dict["pricing_access"] = pricing_access
        if compute_access is not UNSET:
            field_dict["compute_access"] = compute_access
        if lock_limit is not UNSET:
            field_dict["lock_limit"] = lock_limit
        if concurrent_requests_limit is not UNSET:
            field_dict["concurrent_requests_limit"] = concurrent_requests_limit
        if concurrent_requests_limit_per_endpoint is not UNSET:
            field_dict["concurrent_requests_limit_per_endpoint"] = concurrent_requests_limit_per_endpoint
        if is_org is not UNSET:
            field_dict["is_org"] = is_org
        if unlimited_teams is not UNSET:
            field_dict["unlimited_teams"] = unlimited_teams
        if spending_alerts is not UNSET:
            field_dict["spending_alerts"] = spending_alerts
        if spending_alerts_v2 is not UNSET:
            field_dict["spending_alerts_v2"] = spending_alerts_v2
        if spending_locks is not UNSET:
            field_dict["spending_locks"] = spending_locks
        if focus_reports is not UNSET:
            field_dict["focus_reports"] = focus_reports
        if allow_net_admin is not UNSET:
            field_dict["allow_net_admin"] = allow_net_admin
        if allow_snapshots is not UNSET:
            field_dict["allow_snapshots"] = allow_snapshots
        if allow_sys_resource is not UNSET:
            field_dict["allow_sys_resource"] = allow_sys_resource
        if allow_sys_ptrace is not UNSET:
            field_dict["allow_sys_ptrace"] = allow_sys_ptrace
        if enriched_invoices is not UNSET:
            field_dict["enriched_invoices"] = enriched_invoices
        if endpoint_access_controls is not UNSET:
            field_dict["endpoint_access_controls"] = endpoint_access_controls
        if disable_fallbacks is not UNSET:
            field_dict["disable_fallbacks"] = disable_fallbacks
        if owned_sso_connections is not UNSET:
            field_dict["owned_sso_connections"] = owned_sso_connections
        if model_status_access is not UNSET:
            field_dict["model_status_access"] = model_status_access
        if concurrency_surcharge is not UNSET:
            field_dict["concurrency_surcharge"] = concurrency_surcharge
        if max_credit_amount_usd is not UNSET:
            field_dict["max_credit_amount_usd"] = max_credit_amount_usd
        if enable_weekly_billing_reports is not UNSET:
            field_dict["enable_weekly_billing_reports"] = enable_weekly_billing_reports
        if serverless_surge_thresholds is not UNSET:
            field_dict["serverless_surge_thresholds"] = serverless_surge_thresholds
        if burst_billing_config is not UNSET:
            field_dict["burst_billing_config"] = burst_billing_config
        if enable_payload_embedding is not UNSET:
            field_dict["enable_payload_embedding"] = enable_payload_embedding
        if enable_ui_payload_embedding is not UNSET:
            field_dict["enable_ui_payload_embedding"] = enable_ui_payload_embedding
        if can_disable_filters is not UNSET:
            field_dict["can_disable_filters"] = can_disable_filters
        if default_private_logs is not UNSET:
            field_dict["default_private_logs"] = default_private_logs
        if skip_runner_connection_check is not UNSET:
            field_dict["skip_runner_connection_check"] = skip_runner_connection_check
        if is_early_access is not UNSET:
            field_dict["is_early_access"] = is_early_access

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.admin_config_burst_billing_config import AdminConfigBurstBillingConfig
        from ..models.admin_config_concurrency_surcharge import AdminConfigConcurrencySurcharge
        from ..models.admin_config_concurrent_requests_limit_per_endpoint import (
            AdminConfigConcurrentRequestsLimitPerEndpoint,
        )
        from ..models.admin_config_max_registered_max_gpus_by_type import AdminConfigMaxRegisteredMaxGpusByType
        from ..models.admin_config_serverless_surge_thresholds import AdminConfigServerlessSurgeThresholds

        d = src_dict.copy()
        queue_priority = d.pop("queue_priority", UNSET)

        deprioritize_queue = d.pop("deprioritize_queue", UNSET)

        rate_limit_exempt = d.pop("rate_limit_exempt", UNSET)

        deploy_shared_app = d.pop("deploy_shared_app", UNSET)

        enable_per_user_auth = d.pop("enable_per_user_auth", UNSET)

        purchase_limit_exempt = d.pop("purchase_limit_exempt", UNSET)

        max_registered_max_concurrency = d.pop("max_registered_max_concurrency", UNSET)

        max_registered_min_concurrency = d.pop("max_registered_min_concurrency", UNSET)

        max_registered_max_gpus = d.pop("max_registered_max_gpus", UNSET)

        _max_registered_max_gpus_by_type = d.pop("max_registered_max_gpus_by_type", UNSET)
        max_registered_max_gpus_by_type: Union[Unset, AdminConfigMaxRegisteredMaxGpusByType]
        if isinstance(_max_registered_max_gpus_by_type, Unset):
            max_registered_max_gpus_by_type = UNSET
        else:
            max_registered_max_gpus_by_type = AdminConfigMaxRegisteredMaxGpusByType.from_dict(
                _max_registered_max_gpus_by_type
            )

        max_registered_keepalive = d.pop("max_registered_keepalive", UNSET)

        max_registered_request_timeout = d.pop("max_registered_request_timeout", UNSET)

        max_registered_startup_timeout = d.pop("max_registered_startup_timeout", UNSET)

        cdn_token_generation = d.pop("cdn_token_generation", UNSET)

        beta_access = d.pop("beta_access", UNSET)

        pricing_access = d.pop("pricing_access", UNSET)

        compute_access = d.pop("compute_access", UNSET)

        lock_limit = d.pop("lock_limit", UNSET)

        concurrent_requests_limit = d.pop("concurrent_requests_limit", UNSET)

        _concurrent_requests_limit_per_endpoint = d.pop("concurrent_requests_limit_per_endpoint", UNSET)
        concurrent_requests_limit_per_endpoint: Union[Unset, AdminConfigConcurrentRequestsLimitPerEndpoint]
        if isinstance(_concurrent_requests_limit_per_endpoint, Unset):
            concurrent_requests_limit_per_endpoint = UNSET
        else:
            concurrent_requests_limit_per_endpoint = AdminConfigConcurrentRequestsLimitPerEndpoint.from_dict(
                _concurrent_requests_limit_per_endpoint
            )

        is_org = d.pop("is_org", UNSET)

        unlimited_teams = d.pop("unlimited_teams", UNSET)

        spending_alerts = d.pop("spending_alerts", UNSET)

        spending_alerts_v2 = d.pop("spending_alerts_v2", UNSET)

        spending_locks = d.pop("spending_locks", UNSET)

        focus_reports = d.pop("focus_reports", UNSET)

        allow_net_admin = d.pop("allow_net_admin", UNSET)

        allow_snapshots = d.pop("allow_snapshots", UNSET)

        allow_sys_resource = d.pop("allow_sys_resource", UNSET)

        allow_sys_ptrace = d.pop("allow_sys_ptrace", UNSET)

        enriched_invoices = d.pop("enriched_invoices", UNSET)

        endpoint_access_controls = d.pop("endpoint_access_controls", UNSET)

        disable_fallbacks = d.pop("disable_fallbacks", UNSET)

        owned_sso_connections = cast(list[str], d.pop("owned_sso_connections", UNSET))

        model_status_access = d.pop("model_status_access", UNSET)

        _concurrency_surcharge = d.pop("concurrency_surcharge", UNSET)
        concurrency_surcharge: Union[Unset, AdminConfigConcurrencySurcharge]
        if isinstance(_concurrency_surcharge, Unset):
            concurrency_surcharge = UNSET
        else:
            concurrency_surcharge = AdminConfigConcurrencySurcharge.from_dict(_concurrency_surcharge)

        max_credit_amount_usd = d.pop("max_credit_amount_usd", UNSET)

        enable_weekly_billing_reports = d.pop("enable_weekly_billing_reports", UNSET)

        _serverless_surge_thresholds = d.pop("serverless_surge_thresholds", UNSET)
        serverless_surge_thresholds: Union[Unset, AdminConfigServerlessSurgeThresholds]
        if isinstance(_serverless_surge_thresholds, Unset):
            serverless_surge_thresholds = UNSET
        else:
            serverless_surge_thresholds = AdminConfigServerlessSurgeThresholds.from_dict(_serverless_surge_thresholds)

        _burst_billing_config = d.pop("burst_billing_config", UNSET)
        burst_billing_config: Union[Unset, AdminConfigBurstBillingConfig]
        if isinstance(_burst_billing_config, Unset):
            burst_billing_config = UNSET
        else:
            burst_billing_config = AdminConfigBurstBillingConfig.from_dict(_burst_billing_config)

        enable_payload_embedding = d.pop("enable_payload_embedding", UNSET)

        enable_ui_payload_embedding = d.pop("enable_ui_payload_embedding", UNSET)

        can_disable_filters = d.pop("can_disable_filters", UNSET)

        default_private_logs = d.pop("default_private_logs", UNSET)

        skip_runner_connection_check = d.pop("skip_runner_connection_check", UNSET)

        is_early_access = d.pop("is_early_access", UNSET)

        admin_config = cls(
            queue_priority=queue_priority,
            deprioritize_queue=deprioritize_queue,
            rate_limit_exempt=rate_limit_exempt,
            deploy_shared_app=deploy_shared_app,
            enable_per_user_auth=enable_per_user_auth,
            purchase_limit_exempt=purchase_limit_exempt,
            max_registered_max_concurrency=max_registered_max_concurrency,
            max_registered_min_concurrency=max_registered_min_concurrency,
            max_registered_max_gpus=max_registered_max_gpus,
            max_registered_max_gpus_by_type=max_registered_max_gpus_by_type,
            max_registered_keepalive=max_registered_keepalive,
            max_registered_request_timeout=max_registered_request_timeout,
            max_registered_startup_timeout=max_registered_startup_timeout,
            cdn_token_generation=cdn_token_generation,
            beta_access=beta_access,
            pricing_access=pricing_access,
            compute_access=compute_access,
            lock_limit=lock_limit,
            concurrent_requests_limit=concurrent_requests_limit,
            concurrent_requests_limit_per_endpoint=concurrent_requests_limit_per_endpoint,
            is_org=is_org,
            unlimited_teams=unlimited_teams,
            spending_alerts=spending_alerts,
            spending_alerts_v2=spending_alerts_v2,
            spending_locks=spending_locks,
            focus_reports=focus_reports,
            allow_net_admin=allow_net_admin,
            allow_snapshots=allow_snapshots,
            allow_sys_resource=allow_sys_resource,
            allow_sys_ptrace=allow_sys_ptrace,
            enriched_invoices=enriched_invoices,
            endpoint_access_controls=endpoint_access_controls,
            disable_fallbacks=disable_fallbacks,
            owned_sso_connections=owned_sso_connections,
            model_status_access=model_status_access,
            concurrency_surcharge=concurrency_surcharge,
            max_credit_amount_usd=max_credit_amount_usd,
            enable_weekly_billing_reports=enable_weekly_billing_reports,
            serverless_surge_thresholds=serverless_surge_thresholds,
            burst_billing_config=burst_billing_config,
            enable_payload_embedding=enable_payload_embedding,
            enable_ui_payload_embedding=enable_ui_payload_embedding,
            can_disable_filters=can_disable_filters,
            default_private_logs=default_private_logs,
            skip_runner_connection_check=skip_runner_connection_check,
            is_early_access=is_early_access,
        )

        admin_config.additional_properties = d
        return admin_config

    @property
    def additional_keys(self) -> list[str]:
        return list(self.additional_properties.keys())

    def __getitem__(self, key: str) -> Any:
        return self.additional_properties[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.additional_properties[key] = value

    def __delitem__(self, key: str) -> None:
        del self.additional_properties[key]

    def __contains__(self, key: str) -> bool:
        return key in self.additional_properties
