from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.billing_account_locked_notification import BillingAccountLockedNotification
from ...models.billing_budget_reached_notification import BillingBudgetReachedNotification
from ...models.billing_budget_warning_notification import BillingBudgetWarningNotification
from ...models.billing_order_confirmation_notification import BillingOrderConfirmationNotification
from ...models.billing_payment_failed_notification import BillingPaymentFailedNotification
from ...models.billing_payment_successful_notification import BillingPaymentSuccessfulNotification
from ...models.billing_spending_alert_triggered_notification import BillingSpendingAlertTriggeredNotification
from ...models.billing_spending_alert_v2_triggered_notification import BillingSpendingAlertV2TriggeredNotification
from ...models.collaboration_team_invite_notification import CollaborationTeamInviteNotification
from ...models.collaboration_team_member_added_notification import CollaborationTeamMemberAddedNotification
from ...models.collaboration_team_member_removed_notification import CollaborationTeamMemberRemovedNotification
from ...models.create_billing_account_locked_notification import CreateBillingAccountLockedNotification
from ...models.create_billing_budget_reached_notification import CreateBillingBudgetReachedNotification
from ...models.create_billing_budget_warning_notification import CreateBillingBudgetWarningNotification
from ...models.create_billing_order_confirmation_notification import CreateBillingOrderConfirmationNotification
from ...models.create_billing_payment_failed_notification import CreateBillingPaymentFailedNotification
from ...models.create_billing_payment_successful_notification import CreateBillingPaymentSuccessfulNotification
from ...models.create_billing_spending_alert_triggered_notification import (
    CreateBillingSpendingAlertTriggeredNotification,
)
from ...models.create_collaboration_team_invite_notification import CreateCollaborationTeamInviteNotification
from ...models.create_collaboration_team_member_added_notification import CreateCollaborationTeamMemberAddedNotification
from ...models.create_collaboration_team_member_removed_notification import (
    CreateCollaborationTeamMemberRemovedNotification,
)
from ...models.create_marketing_newsletter_notification import CreateMarketingNewsletterNotification
from ...models.create_marketing_product_announcement_notification import CreateMarketingProductAnnouncementNotification
from ...models.create_platform_announcement_notification import CreatePlatformAnnouncementNotification
from ...models.create_platform_app_created_notification import CreatePlatformAppCreatedNotification
from ...models.create_platform_app_deleted_notification import CreatePlatformAppDeletedNotification
from ...models.create_platform_app_http_5_xx_errors_notification import CreatePlatformAppHttp5XxErrorsNotification
from ...models.create_platform_app_modified_notification import CreatePlatformAppModifiedNotification
from ...models.create_platform_app_oom_notification import CreatePlatformAppOomNotification
from ...models.create_platform_app_queue_size_notification import CreatePlatformAppQueueSizeNotification
from ...models.create_platform_app_startup_failure_notification import CreatePlatformAppStartupFailureNotification
from ...models.create_platform_app_startup_timeout_notification import CreatePlatformAppStartupTimeoutNotification
from ...models.create_platform_compute_instance_created_notification import (
    CreatePlatformComputeInstanceCreatedNotification,
)
from ...models.create_platform_compute_instance_provisioning_failed_notification import (
    CreatePlatformComputeInstanceProvisioningFailedNotification,
)
from ...models.create_platform_compute_insufficient_capacity_notification import (
    CreatePlatformComputeInsufficientCapacityNotification,
)
from ...models.create_platform_concurrent_requests_limit_reached_notification import (
    CreatePlatformConcurrentRequestsLimitReachedNotification,
)
from ...models.create_platform_endpoint_enterprise_ready_notification import (
    CreatePlatformEndpointEnterpriseReadyNotification,
)
from ...models.create_platform_maintenance_completed_notification import CreatePlatformMaintenanceCompletedNotification
from ...models.create_platform_maintenance_scheduled_notification import CreatePlatformMaintenanceScheduledNotification
from ...models.create_platform_maintenance_started_notification import CreatePlatformMaintenanceStartedNotification
from ...models.create_platform_welcome_with_credits_notification import CreatePlatformWelcomeWithCreditsNotification
from ...models.create_platform_welcome_without_credits_notification import (
    CreatePlatformWelcomeWithoutCreditsNotification,
)
from ...models.create_security_api_key_created_notification import CreateSecurityApiKeyCreatedNotification
from ...models.create_security_api_key_revoked_notification import CreateSecurityApiKeyRevokedNotification
from ...models.http_validation_error import HTTPValidationError
from ...models.marketing_newsletter_notification import MarketingNewsletterNotification
from ...models.marketing_product_announcement_notification import MarketingProductAnnouncementNotification
from ...models.platform_announcement_notification import PlatformAnnouncementNotification
from ...models.platform_app_created_notification import PlatformAppCreatedNotification
from ...models.platform_app_deleted_notification import PlatformAppDeletedNotification
from ...models.platform_app_http_5_xx_errors_notification import PlatformAppHttp5XxErrorsNotification
from ...models.platform_app_modified_notification import PlatformAppModifiedNotification
from ...models.platform_app_oom_notification import PlatformAppOomNotification
from ...models.platform_app_queue_size_notification import PlatformAppQueueSizeNotification
from ...models.platform_app_startup_failure_notification import PlatformAppStartupFailureNotification
from ...models.platform_app_startup_timeout_notification import PlatformAppStartupTimeoutNotification
from ...models.platform_compute_instance_created_notification import PlatformComputeInstanceCreatedNotification
from ...models.platform_compute_instance_provisioning_failed_notification import (
    PlatformComputeInstanceProvisioningFailedNotification,
)
from ...models.platform_compute_insufficient_capacity_notification import (
    PlatformComputeInsufficientCapacityNotification,
)
from ...models.platform_concurrent_requests_limit_reached_notification import (
    PlatformConcurrentRequestsLimitReachedNotification,
)
from ...models.platform_endpoint_enterprise_ready_notification import PlatformEndpointEnterpriseReadyNotification
from ...models.platform_maintenance_completed_notification import PlatformMaintenanceCompletedNotification
from ...models.platform_maintenance_scheduled_notification import PlatformMaintenanceScheduledNotification
from ...models.platform_maintenance_started_notification import PlatformMaintenanceStartedNotification
from ...models.platform_welcome_with_credits_notification import PlatformWelcomeWithCreditsNotification
from ...models.platform_welcome_without_credits_notification import PlatformWelcomeWithoutCreditsNotification
from ...models.security_api_key_created_notification import SecurityApiKeyCreatedNotification
from ...models.security_api_key_revoked_notification import SecurityApiKeyRevokedNotification
from ...types import Response


def _get_kwargs(
    *,
    body: Union[
        "CreateBillingAccountLockedNotification",
        "CreateBillingBudgetReachedNotification",
        "CreateBillingBudgetWarningNotification",
        "CreateBillingOrderConfirmationNotification",
        "CreateBillingPaymentFailedNotification",
        "CreateBillingPaymentSuccessfulNotification",
        "CreateBillingSpendingAlertTriggeredNotification",
        "CreateCollaborationTeamInviteNotification",
        "CreateCollaborationTeamMemberAddedNotification",
        "CreateCollaborationTeamMemberRemovedNotification",
        "CreateMarketingNewsletterNotification",
        "CreateMarketingProductAnnouncementNotification",
        "CreatePlatformAnnouncementNotification",
        "CreatePlatformAppCreatedNotification",
        "CreatePlatformAppDeletedNotification",
        "CreatePlatformAppHttp5XxErrorsNotification",
        "CreatePlatformAppModifiedNotification",
        "CreatePlatformAppOomNotification",
        "CreatePlatformAppQueueSizeNotification",
        "CreatePlatformAppStartupFailureNotification",
        "CreatePlatformAppStartupTimeoutNotification",
        "CreatePlatformComputeInstanceCreatedNotification",
        "CreatePlatformComputeInstanceProvisioningFailedNotification",
        "CreatePlatformComputeInsufficientCapacityNotification",
        "CreatePlatformConcurrentRequestsLimitReachedNotification",
        "CreatePlatformEndpointEnterpriseReadyNotification",
        "CreatePlatformMaintenanceCompletedNotification",
        "CreatePlatformMaintenanceScheduledNotification",
        "CreatePlatformMaintenanceStartedNotification",
        "CreatePlatformWelcomeWithCreditsNotification",
        "CreatePlatformWelcomeWithoutCreditsNotification",
        "CreateSecurityApiKeyCreatedNotification",
        "CreateSecurityApiKeyRevokedNotification",
    ],
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/admin/notifications",
    }

    _body: dict[str, Any]
    if isinstance(body, CreateBillingPaymentSuccessfulNotification):
        _body = body.to_dict()
    elif isinstance(body, CreateBillingPaymentFailedNotification):
        _body = body.to_dict()
    elif isinstance(body, CreateBillingBudgetWarningNotification):
        _body = body.to_dict()
    elif isinstance(body, CreateBillingBudgetReachedNotification):
        _body = body.to_dict()
    elif isinstance(body, CreateBillingOrderConfirmationNotification):
        _body = body.to_dict()
    elif isinstance(body, CreateBillingAccountLockedNotification):
        _body = body.to_dict()
    elif isinstance(body, CreateBillingSpendingAlertTriggeredNotification):
        _body = body.to_dict()
    elif isinstance(body, CreateSecurityApiKeyCreatedNotification):
        _body = body.to_dict()
    elif isinstance(body, CreateSecurityApiKeyRevokedNotification):
        _body = body.to_dict()
    elif isinstance(body, CreateCollaborationTeamInviteNotification):
        _body = body.to_dict()
    elif isinstance(body, CreateCollaborationTeamMemberAddedNotification):
        _body = body.to_dict()
    elif isinstance(body, CreateCollaborationTeamMemberRemovedNotification):
        _body = body.to_dict()
    elif isinstance(body, CreateMarketingNewsletterNotification):
        _body = body.to_dict()
    elif isinstance(body, CreateMarketingProductAnnouncementNotification):
        _body = body.to_dict()
    elif isinstance(body, CreatePlatformMaintenanceScheduledNotification):
        _body = body.to_dict()
    elif isinstance(body, CreatePlatformMaintenanceStartedNotification):
        _body = body.to_dict()
    elif isinstance(body, CreatePlatformMaintenanceCompletedNotification):
        _body = body.to_dict()
    elif isinstance(body, CreatePlatformWelcomeWithCreditsNotification):
        _body = body.to_dict()
    elif isinstance(body, CreatePlatformWelcomeWithoutCreditsNotification):
        _body = body.to_dict()
    elif isinstance(body, CreatePlatformAppCreatedNotification):
        _body = body.to_dict()
    elif isinstance(body, CreatePlatformAppModifiedNotification):
        _body = body.to_dict()
    elif isinstance(body, CreatePlatformAppDeletedNotification):
        _body = body.to_dict()
    elif isinstance(body, CreatePlatformAppQueueSizeNotification):
        _body = body.to_dict()
    elif isinstance(body, CreatePlatformAppHttp5XxErrorsNotification):
        _body = body.to_dict()
    elif isinstance(body, CreatePlatformAppOomNotification):
        _body = body.to_dict()
    elif isinstance(body, CreatePlatformAppStartupTimeoutNotification):
        _body = body.to_dict()
    elif isinstance(body, CreatePlatformAppStartupFailureNotification):
        _body = body.to_dict()
    elif isinstance(body, CreatePlatformConcurrentRequestsLimitReachedNotification):
        _body = body.to_dict()
    elif isinstance(body, CreatePlatformComputeInstanceCreatedNotification):
        _body = body.to_dict()
    elif isinstance(body, CreatePlatformComputeInstanceProvisioningFailedNotification):
        _body = body.to_dict()
    elif isinstance(body, CreatePlatformComputeInsufficientCapacityNotification):
        _body = body.to_dict()
    elif isinstance(body, CreatePlatformEndpointEnterpriseReadyNotification):
        _body = body.to_dict()
    else:
        _body = body.to_dict()

    _kwargs["json"] = _body
    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[
    Union[
        HTTPValidationError,
        Union[
            "BillingAccountLockedNotification",
            "BillingBudgetReachedNotification",
            "BillingBudgetWarningNotification",
            "BillingOrderConfirmationNotification",
            "BillingPaymentFailedNotification",
            "BillingPaymentSuccessfulNotification",
            "BillingSpendingAlertTriggeredNotification",
            "BillingSpendingAlertV2TriggeredNotification",
            "CollaborationTeamInviteNotification",
            "CollaborationTeamMemberAddedNotification",
            "CollaborationTeamMemberRemovedNotification",
            "MarketingNewsletterNotification",
            "MarketingProductAnnouncementNotification",
            "PlatformAnnouncementNotification",
            "PlatformAppCreatedNotification",
            "PlatformAppDeletedNotification",
            "PlatformAppHttp5XxErrorsNotification",
            "PlatformAppModifiedNotification",
            "PlatformAppOomNotification",
            "PlatformAppQueueSizeNotification",
            "PlatformAppStartupFailureNotification",
            "PlatformAppStartupTimeoutNotification",
            "PlatformComputeInstanceCreatedNotification",
            "PlatformComputeInstanceProvisioningFailedNotification",
            "PlatformComputeInsufficientCapacityNotification",
            "PlatformConcurrentRequestsLimitReachedNotification",
            "PlatformEndpointEnterpriseReadyNotification",
            "PlatformMaintenanceCompletedNotification",
            "PlatformMaintenanceScheduledNotification",
            "PlatformMaintenanceStartedNotification",
            "PlatformWelcomeWithCreditsNotification",
            "PlatformWelcomeWithoutCreditsNotification",
            "SecurityApiKeyCreatedNotification",
            "SecurityApiKeyRevokedNotification",
        ],
    ]
]:
    if response.status_code == 201:

        def _parse_response_201(
            data: object,
        ) -> Union[
            "BillingAccountLockedNotification",
            "BillingBudgetReachedNotification",
            "BillingBudgetWarningNotification",
            "BillingOrderConfirmationNotification",
            "BillingPaymentFailedNotification",
            "BillingPaymentSuccessfulNotification",
            "BillingSpendingAlertTriggeredNotification",
            "BillingSpendingAlertV2TriggeredNotification",
            "CollaborationTeamInviteNotification",
            "CollaborationTeamMemberAddedNotification",
            "CollaborationTeamMemberRemovedNotification",
            "MarketingNewsletterNotification",
            "MarketingProductAnnouncementNotification",
            "PlatformAnnouncementNotification",
            "PlatformAppCreatedNotification",
            "PlatformAppDeletedNotification",
            "PlatformAppHttp5XxErrorsNotification",
            "PlatformAppModifiedNotification",
            "PlatformAppOomNotification",
            "PlatformAppQueueSizeNotification",
            "PlatformAppStartupFailureNotification",
            "PlatformAppStartupTimeoutNotification",
            "PlatformComputeInstanceCreatedNotification",
            "PlatformComputeInstanceProvisioningFailedNotification",
            "PlatformComputeInsufficientCapacityNotification",
            "PlatformConcurrentRequestsLimitReachedNotification",
            "PlatformEndpointEnterpriseReadyNotification",
            "PlatformMaintenanceCompletedNotification",
            "PlatformMaintenanceScheduledNotification",
            "PlatformMaintenanceStartedNotification",
            "PlatformWelcomeWithCreditsNotification",
            "PlatformWelcomeWithoutCreditsNotification",
            "SecurityApiKeyCreatedNotification",
            "SecurityApiKeyRevokedNotification",
        ]:
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                response_201_type_0 = BillingPaymentSuccessfulNotification.from_dict(data)

                return response_201_type_0
            except:  # noqa: E722
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                response_201_type_1 = BillingPaymentFailedNotification.from_dict(data)

                return response_201_type_1
            except:  # noqa: E722
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                response_201_type_2 = BillingBudgetWarningNotification.from_dict(data)

                return response_201_type_2
            except:  # noqa: E722
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                response_201_type_3 = BillingBudgetReachedNotification.from_dict(data)

                return response_201_type_3
            except:  # noqa: E722
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                response_201_type_4 = BillingOrderConfirmationNotification.from_dict(data)

                return response_201_type_4
            except:  # noqa: E722
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                response_201_type_5 = BillingAccountLockedNotification.from_dict(data)

                return response_201_type_5
            except:  # noqa: E722
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                response_201_type_6 = BillingSpendingAlertTriggeredNotification.from_dict(data)

                return response_201_type_6
            except:  # noqa: E722
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                response_201_type_7 = BillingSpendingAlertV2TriggeredNotification.from_dict(data)

                return response_201_type_7
            except:  # noqa: E722
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                response_201_type_8 = SecurityApiKeyCreatedNotification.from_dict(data)

                return response_201_type_8
            except:  # noqa: E722
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                response_201_type_9 = SecurityApiKeyRevokedNotification.from_dict(data)

                return response_201_type_9
            except:  # noqa: E722
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                response_201_type_10 = CollaborationTeamInviteNotification.from_dict(data)

                return response_201_type_10
            except:  # noqa: E722
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                response_201_type_11 = CollaborationTeamMemberAddedNotification.from_dict(data)

                return response_201_type_11
            except:  # noqa: E722
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                response_201_type_12 = CollaborationTeamMemberRemovedNotification.from_dict(data)

                return response_201_type_12
            except:  # noqa: E722
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                response_201_type_13 = MarketingNewsletterNotification.from_dict(data)

                return response_201_type_13
            except:  # noqa: E722
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                response_201_type_14 = MarketingProductAnnouncementNotification.from_dict(data)

                return response_201_type_14
            except:  # noqa: E722
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                response_201_type_15 = PlatformMaintenanceScheduledNotification.from_dict(data)

                return response_201_type_15
            except:  # noqa: E722
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                response_201_type_16 = PlatformMaintenanceStartedNotification.from_dict(data)

                return response_201_type_16
            except:  # noqa: E722
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                response_201_type_17 = PlatformMaintenanceCompletedNotification.from_dict(data)

                return response_201_type_17
            except:  # noqa: E722
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                response_201_type_18 = PlatformWelcomeWithCreditsNotification.from_dict(data)

                return response_201_type_18
            except:  # noqa: E722
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                response_201_type_19 = PlatformWelcomeWithoutCreditsNotification.from_dict(data)

                return response_201_type_19
            except:  # noqa: E722
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                response_201_type_20 = PlatformAppCreatedNotification.from_dict(data)

                return response_201_type_20
            except:  # noqa: E722
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                response_201_type_21 = PlatformAppModifiedNotification.from_dict(data)

                return response_201_type_21
            except:  # noqa: E722
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                response_201_type_22 = PlatformAppDeletedNotification.from_dict(data)

                return response_201_type_22
            except:  # noqa: E722
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                response_201_type_23 = PlatformAppQueueSizeNotification.from_dict(data)

                return response_201_type_23
            except:  # noqa: E722
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                response_201_type_24 = PlatformAppHttp5XxErrorsNotification.from_dict(data)

                return response_201_type_24
            except:  # noqa: E722
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                response_201_type_25 = PlatformAppOomNotification.from_dict(data)

                return response_201_type_25
            except:  # noqa: E722
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                response_201_type_26 = PlatformAppStartupTimeoutNotification.from_dict(data)

                return response_201_type_26
            except:  # noqa: E722
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                response_201_type_27 = PlatformAppStartupFailureNotification.from_dict(data)

                return response_201_type_27
            except:  # noqa: E722
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                response_201_type_28 = PlatformConcurrentRequestsLimitReachedNotification.from_dict(data)

                return response_201_type_28
            except:  # noqa: E722
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                response_201_type_29 = PlatformComputeInstanceCreatedNotification.from_dict(data)

                return response_201_type_29
            except:  # noqa: E722
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                response_201_type_30 = PlatformComputeInstanceProvisioningFailedNotification.from_dict(data)

                return response_201_type_30
            except:  # noqa: E722
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                response_201_type_31 = PlatformComputeInsufficientCapacityNotification.from_dict(data)

                return response_201_type_31
            except:  # noqa: E722
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                response_201_type_32 = PlatformEndpointEnterpriseReadyNotification.from_dict(data)

                return response_201_type_32
            except:  # noqa: E722
                pass
            if not isinstance(data, dict):
                raise TypeError()
            response_201_type_33 = PlatformAnnouncementNotification.from_dict(data)

            return response_201_type_33

        response_201 = _parse_response_201(response.json())

        return response_201
    if response.status_code == 422:
        response_422 = HTTPValidationError.from_dict(response.json())

        return response_422
    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Response[
    Union[
        HTTPValidationError,
        Union[
            "BillingAccountLockedNotification",
            "BillingBudgetReachedNotification",
            "BillingBudgetWarningNotification",
            "BillingOrderConfirmationNotification",
            "BillingPaymentFailedNotification",
            "BillingPaymentSuccessfulNotification",
            "BillingSpendingAlertTriggeredNotification",
            "BillingSpendingAlertV2TriggeredNotification",
            "CollaborationTeamInviteNotification",
            "CollaborationTeamMemberAddedNotification",
            "CollaborationTeamMemberRemovedNotification",
            "MarketingNewsletterNotification",
            "MarketingProductAnnouncementNotification",
            "PlatformAnnouncementNotification",
            "PlatformAppCreatedNotification",
            "PlatformAppDeletedNotification",
            "PlatformAppHttp5XxErrorsNotification",
            "PlatformAppModifiedNotification",
            "PlatformAppOomNotification",
            "PlatformAppQueueSizeNotification",
            "PlatformAppStartupFailureNotification",
            "PlatformAppStartupTimeoutNotification",
            "PlatformComputeInstanceCreatedNotification",
            "PlatformComputeInstanceProvisioningFailedNotification",
            "PlatformComputeInsufficientCapacityNotification",
            "PlatformConcurrentRequestsLimitReachedNotification",
            "PlatformEndpointEnterpriseReadyNotification",
            "PlatformMaintenanceCompletedNotification",
            "PlatformMaintenanceScheduledNotification",
            "PlatformMaintenanceStartedNotification",
            "PlatformWelcomeWithCreditsNotification",
            "PlatformWelcomeWithoutCreditsNotification",
            "SecurityApiKeyCreatedNotification",
            "SecurityApiKeyRevokedNotification",
        ],
    ]
]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    body: Union[
        "CreateBillingAccountLockedNotification",
        "CreateBillingBudgetReachedNotification",
        "CreateBillingBudgetWarningNotification",
        "CreateBillingOrderConfirmationNotification",
        "CreateBillingPaymentFailedNotification",
        "CreateBillingPaymentSuccessfulNotification",
        "CreateBillingSpendingAlertTriggeredNotification",
        "CreateCollaborationTeamInviteNotification",
        "CreateCollaborationTeamMemberAddedNotification",
        "CreateCollaborationTeamMemberRemovedNotification",
        "CreateMarketingNewsletterNotification",
        "CreateMarketingProductAnnouncementNotification",
        "CreatePlatformAnnouncementNotification",
        "CreatePlatformAppCreatedNotification",
        "CreatePlatformAppDeletedNotification",
        "CreatePlatformAppHttp5XxErrorsNotification",
        "CreatePlatformAppModifiedNotification",
        "CreatePlatformAppOomNotification",
        "CreatePlatformAppQueueSizeNotification",
        "CreatePlatformAppStartupFailureNotification",
        "CreatePlatformAppStartupTimeoutNotification",
        "CreatePlatformComputeInstanceCreatedNotification",
        "CreatePlatformComputeInstanceProvisioningFailedNotification",
        "CreatePlatformComputeInsufficientCapacityNotification",
        "CreatePlatformConcurrentRequestsLimitReachedNotification",
        "CreatePlatformEndpointEnterpriseReadyNotification",
        "CreatePlatformMaintenanceCompletedNotification",
        "CreatePlatformMaintenanceScheduledNotification",
        "CreatePlatformMaintenanceStartedNotification",
        "CreatePlatformWelcomeWithCreditsNotification",
        "CreatePlatformWelcomeWithoutCreditsNotification",
        "CreateSecurityApiKeyCreatedNotification",
        "CreateSecurityApiKeyRevokedNotification",
    ],
) -> Response[
    Union[
        HTTPValidationError,
        Union[
            "BillingAccountLockedNotification",
            "BillingBudgetReachedNotification",
            "BillingBudgetWarningNotification",
            "BillingOrderConfirmationNotification",
            "BillingPaymentFailedNotification",
            "BillingPaymentSuccessfulNotification",
            "BillingSpendingAlertTriggeredNotification",
            "BillingSpendingAlertV2TriggeredNotification",
            "CollaborationTeamInviteNotification",
            "CollaborationTeamMemberAddedNotification",
            "CollaborationTeamMemberRemovedNotification",
            "MarketingNewsletterNotification",
            "MarketingProductAnnouncementNotification",
            "PlatformAnnouncementNotification",
            "PlatformAppCreatedNotification",
            "PlatformAppDeletedNotification",
            "PlatformAppHttp5XxErrorsNotification",
            "PlatformAppModifiedNotification",
            "PlatformAppOomNotification",
            "PlatformAppQueueSizeNotification",
            "PlatformAppStartupFailureNotification",
            "PlatformAppStartupTimeoutNotification",
            "PlatformComputeInstanceCreatedNotification",
            "PlatformComputeInstanceProvisioningFailedNotification",
            "PlatformComputeInsufficientCapacityNotification",
            "PlatformConcurrentRequestsLimitReachedNotification",
            "PlatformEndpointEnterpriseReadyNotification",
            "PlatformMaintenanceCompletedNotification",
            "PlatformMaintenanceScheduledNotification",
            "PlatformMaintenanceStartedNotification",
            "PlatformWelcomeWithCreditsNotification",
            "PlatformWelcomeWithoutCreditsNotification",
            "SecurityApiKeyCreatedNotification",
            "SecurityApiKeyRevokedNotification",
        ],
    ]
]:
    """Admin Create Notification

    Args:
        body (Union['CreateBillingAccountLockedNotification',
            'CreateBillingBudgetReachedNotification', 'CreateBillingBudgetWarningNotification',
            'CreateBillingOrderConfirmationNotification', 'CreateBillingPaymentFailedNotification',
            'CreateBillingPaymentSuccessfulNotification',
            'CreateBillingSpendingAlertTriggeredNotification',
            'CreateCollaborationTeamInviteNotification',
            'CreateCollaborationTeamMemberAddedNotification',
            'CreateCollaborationTeamMemberRemovedNotification',
            'CreateMarketingNewsletterNotification', 'CreateMarketingProductAnnouncementNotification',
            'CreatePlatformAnnouncementNotification', 'CreatePlatformAppCreatedNotification',
            'CreatePlatformAppDeletedNotification', 'CreatePlatformAppHttp5XxErrorsNotification',
            'CreatePlatformAppModifiedNotification', 'CreatePlatformAppOomNotification',
            'CreatePlatformAppQueueSizeNotification', 'CreatePlatformAppStartupFailureNotification',
            'CreatePlatformAppStartupTimeoutNotification',
            'CreatePlatformComputeInstanceCreatedNotification',
            'CreatePlatformComputeInstanceProvisioningFailedNotification',
            'CreatePlatformComputeInsufficientCapacityNotification',
            'CreatePlatformConcurrentRequestsLimitReachedNotification',
            'CreatePlatformEndpointEnterpriseReadyNotification',
            'CreatePlatformMaintenanceCompletedNotification',
            'CreatePlatformMaintenanceScheduledNotification',
            'CreatePlatformMaintenanceStartedNotification',
            'CreatePlatformWelcomeWithCreditsNotification',
            'CreatePlatformWelcomeWithoutCreditsNotification',
            'CreateSecurityApiKeyCreatedNotification', 'CreateSecurityApiKeyRevokedNotification']):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, Union['BillingAccountLockedNotification', 'BillingBudgetReachedNotification', 'BillingBudgetWarningNotification', 'BillingOrderConfirmationNotification', 'BillingPaymentFailedNotification', 'BillingPaymentSuccessfulNotification', 'BillingSpendingAlertTriggeredNotification', 'BillingSpendingAlertV2TriggeredNotification', 'CollaborationTeamInviteNotification', 'CollaborationTeamMemberAddedNotification', 'CollaborationTeamMemberRemovedNotification', 'MarketingNewsletterNotification', 'MarketingProductAnnouncementNotification', 'PlatformAnnouncementNotification', 'PlatformAppCreatedNotification', 'PlatformAppDeletedNotification', 'PlatformAppHttp5XxErrorsNotification', 'PlatformAppModifiedNotification', 'PlatformAppOomNotification', 'PlatformAppQueueSizeNotification', 'PlatformAppStartupFailureNotification', 'PlatformAppStartupTimeoutNotification', 'PlatformComputeInstanceCreatedNotification', 'PlatformComputeInstanceProvisioningFailedNotification', 'PlatformComputeInsufficientCapacityNotification', 'PlatformConcurrentRequestsLimitReachedNotification', 'PlatformEndpointEnterpriseReadyNotification', 'PlatformMaintenanceCompletedNotification', 'PlatformMaintenanceScheduledNotification', 'PlatformMaintenanceStartedNotification', 'PlatformWelcomeWithCreditsNotification', 'PlatformWelcomeWithoutCreditsNotification', 'SecurityApiKeyCreatedNotification', 'SecurityApiKeyRevokedNotification']]]
    """

    kwargs = _get_kwargs(
        body=body,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: Union[AuthenticatedClient, Client],
    body: Union[
        "CreateBillingAccountLockedNotification",
        "CreateBillingBudgetReachedNotification",
        "CreateBillingBudgetWarningNotification",
        "CreateBillingOrderConfirmationNotification",
        "CreateBillingPaymentFailedNotification",
        "CreateBillingPaymentSuccessfulNotification",
        "CreateBillingSpendingAlertTriggeredNotification",
        "CreateCollaborationTeamInviteNotification",
        "CreateCollaborationTeamMemberAddedNotification",
        "CreateCollaborationTeamMemberRemovedNotification",
        "CreateMarketingNewsletterNotification",
        "CreateMarketingProductAnnouncementNotification",
        "CreatePlatformAnnouncementNotification",
        "CreatePlatformAppCreatedNotification",
        "CreatePlatformAppDeletedNotification",
        "CreatePlatformAppHttp5XxErrorsNotification",
        "CreatePlatformAppModifiedNotification",
        "CreatePlatformAppOomNotification",
        "CreatePlatformAppQueueSizeNotification",
        "CreatePlatformAppStartupFailureNotification",
        "CreatePlatformAppStartupTimeoutNotification",
        "CreatePlatformComputeInstanceCreatedNotification",
        "CreatePlatformComputeInstanceProvisioningFailedNotification",
        "CreatePlatformComputeInsufficientCapacityNotification",
        "CreatePlatformConcurrentRequestsLimitReachedNotification",
        "CreatePlatformEndpointEnterpriseReadyNotification",
        "CreatePlatformMaintenanceCompletedNotification",
        "CreatePlatformMaintenanceScheduledNotification",
        "CreatePlatformMaintenanceStartedNotification",
        "CreatePlatformWelcomeWithCreditsNotification",
        "CreatePlatformWelcomeWithoutCreditsNotification",
        "CreateSecurityApiKeyCreatedNotification",
        "CreateSecurityApiKeyRevokedNotification",
    ],
) -> Optional[
    Union[
        HTTPValidationError,
        Union[
            "BillingAccountLockedNotification",
            "BillingBudgetReachedNotification",
            "BillingBudgetWarningNotification",
            "BillingOrderConfirmationNotification",
            "BillingPaymentFailedNotification",
            "BillingPaymentSuccessfulNotification",
            "BillingSpendingAlertTriggeredNotification",
            "BillingSpendingAlertV2TriggeredNotification",
            "CollaborationTeamInviteNotification",
            "CollaborationTeamMemberAddedNotification",
            "CollaborationTeamMemberRemovedNotification",
            "MarketingNewsletterNotification",
            "MarketingProductAnnouncementNotification",
            "PlatformAnnouncementNotification",
            "PlatformAppCreatedNotification",
            "PlatformAppDeletedNotification",
            "PlatformAppHttp5XxErrorsNotification",
            "PlatformAppModifiedNotification",
            "PlatformAppOomNotification",
            "PlatformAppQueueSizeNotification",
            "PlatformAppStartupFailureNotification",
            "PlatformAppStartupTimeoutNotification",
            "PlatformComputeInstanceCreatedNotification",
            "PlatformComputeInstanceProvisioningFailedNotification",
            "PlatformComputeInsufficientCapacityNotification",
            "PlatformConcurrentRequestsLimitReachedNotification",
            "PlatformEndpointEnterpriseReadyNotification",
            "PlatformMaintenanceCompletedNotification",
            "PlatformMaintenanceScheduledNotification",
            "PlatformMaintenanceStartedNotification",
            "PlatformWelcomeWithCreditsNotification",
            "PlatformWelcomeWithoutCreditsNotification",
            "SecurityApiKeyCreatedNotification",
            "SecurityApiKeyRevokedNotification",
        ],
    ]
]:
    """Admin Create Notification

    Args:
        body (Union['CreateBillingAccountLockedNotification',
            'CreateBillingBudgetReachedNotification', 'CreateBillingBudgetWarningNotification',
            'CreateBillingOrderConfirmationNotification', 'CreateBillingPaymentFailedNotification',
            'CreateBillingPaymentSuccessfulNotification',
            'CreateBillingSpendingAlertTriggeredNotification',
            'CreateCollaborationTeamInviteNotification',
            'CreateCollaborationTeamMemberAddedNotification',
            'CreateCollaborationTeamMemberRemovedNotification',
            'CreateMarketingNewsletterNotification', 'CreateMarketingProductAnnouncementNotification',
            'CreatePlatformAnnouncementNotification', 'CreatePlatformAppCreatedNotification',
            'CreatePlatformAppDeletedNotification', 'CreatePlatformAppHttp5XxErrorsNotification',
            'CreatePlatformAppModifiedNotification', 'CreatePlatformAppOomNotification',
            'CreatePlatformAppQueueSizeNotification', 'CreatePlatformAppStartupFailureNotification',
            'CreatePlatformAppStartupTimeoutNotification',
            'CreatePlatformComputeInstanceCreatedNotification',
            'CreatePlatformComputeInstanceProvisioningFailedNotification',
            'CreatePlatformComputeInsufficientCapacityNotification',
            'CreatePlatformConcurrentRequestsLimitReachedNotification',
            'CreatePlatformEndpointEnterpriseReadyNotification',
            'CreatePlatformMaintenanceCompletedNotification',
            'CreatePlatformMaintenanceScheduledNotification',
            'CreatePlatformMaintenanceStartedNotification',
            'CreatePlatformWelcomeWithCreditsNotification',
            'CreatePlatformWelcomeWithoutCreditsNotification',
            'CreateSecurityApiKeyCreatedNotification', 'CreateSecurityApiKeyRevokedNotification']):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, Union['BillingAccountLockedNotification', 'BillingBudgetReachedNotification', 'BillingBudgetWarningNotification', 'BillingOrderConfirmationNotification', 'BillingPaymentFailedNotification', 'BillingPaymentSuccessfulNotification', 'BillingSpendingAlertTriggeredNotification', 'BillingSpendingAlertV2TriggeredNotification', 'CollaborationTeamInviteNotification', 'CollaborationTeamMemberAddedNotification', 'CollaborationTeamMemberRemovedNotification', 'MarketingNewsletterNotification', 'MarketingProductAnnouncementNotification', 'PlatformAnnouncementNotification', 'PlatformAppCreatedNotification', 'PlatformAppDeletedNotification', 'PlatformAppHttp5XxErrorsNotification', 'PlatformAppModifiedNotification', 'PlatformAppOomNotification', 'PlatformAppQueueSizeNotification', 'PlatformAppStartupFailureNotification', 'PlatformAppStartupTimeoutNotification', 'PlatformComputeInstanceCreatedNotification', 'PlatformComputeInstanceProvisioningFailedNotification', 'PlatformComputeInsufficientCapacityNotification', 'PlatformConcurrentRequestsLimitReachedNotification', 'PlatformEndpointEnterpriseReadyNotification', 'PlatformMaintenanceCompletedNotification', 'PlatformMaintenanceScheduledNotification', 'PlatformMaintenanceStartedNotification', 'PlatformWelcomeWithCreditsNotification', 'PlatformWelcomeWithoutCreditsNotification', 'SecurityApiKeyCreatedNotification', 'SecurityApiKeyRevokedNotification']]
    """

    return sync_detailed(
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    body: Union[
        "CreateBillingAccountLockedNotification",
        "CreateBillingBudgetReachedNotification",
        "CreateBillingBudgetWarningNotification",
        "CreateBillingOrderConfirmationNotification",
        "CreateBillingPaymentFailedNotification",
        "CreateBillingPaymentSuccessfulNotification",
        "CreateBillingSpendingAlertTriggeredNotification",
        "CreateCollaborationTeamInviteNotification",
        "CreateCollaborationTeamMemberAddedNotification",
        "CreateCollaborationTeamMemberRemovedNotification",
        "CreateMarketingNewsletterNotification",
        "CreateMarketingProductAnnouncementNotification",
        "CreatePlatformAnnouncementNotification",
        "CreatePlatformAppCreatedNotification",
        "CreatePlatformAppDeletedNotification",
        "CreatePlatformAppHttp5XxErrorsNotification",
        "CreatePlatformAppModifiedNotification",
        "CreatePlatformAppOomNotification",
        "CreatePlatformAppQueueSizeNotification",
        "CreatePlatformAppStartupFailureNotification",
        "CreatePlatformAppStartupTimeoutNotification",
        "CreatePlatformComputeInstanceCreatedNotification",
        "CreatePlatformComputeInstanceProvisioningFailedNotification",
        "CreatePlatformComputeInsufficientCapacityNotification",
        "CreatePlatformConcurrentRequestsLimitReachedNotification",
        "CreatePlatformEndpointEnterpriseReadyNotification",
        "CreatePlatformMaintenanceCompletedNotification",
        "CreatePlatformMaintenanceScheduledNotification",
        "CreatePlatformMaintenanceStartedNotification",
        "CreatePlatformWelcomeWithCreditsNotification",
        "CreatePlatformWelcomeWithoutCreditsNotification",
        "CreateSecurityApiKeyCreatedNotification",
        "CreateSecurityApiKeyRevokedNotification",
    ],
) -> Response[
    Union[
        HTTPValidationError,
        Union[
            "BillingAccountLockedNotification",
            "BillingBudgetReachedNotification",
            "BillingBudgetWarningNotification",
            "BillingOrderConfirmationNotification",
            "BillingPaymentFailedNotification",
            "BillingPaymentSuccessfulNotification",
            "BillingSpendingAlertTriggeredNotification",
            "BillingSpendingAlertV2TriggeredNotification",
            "CollaborationTeamInviteNotification",
            "CollaborationTeamMemberAddedNotification",
            "CollaborationTeamMemberRemovedNotification",
            "MarketingNewsletterNotification",
            "MarketingProductAnnouncementNotification",
            "PlatformAnnouncementNotification",
            "PlatformAppCreatedNotification",
            "PlatformAppDeletedNotification",
            "PlatformAppHttp5XxErrorsNotification",
            "PlatformAppModifiedNotification",
            "PlatformAppOomNotification",
            "PlatformAppQueueSizeNotification",
            "PlatformAppStartupFailureNotification",
            "PlatformAppStartupTimeoutNotification",
            "PlatformComputeInstanceCreatedNotification",
            "PlatformComputeInstanceProvisioningFailedNotification",
            "PlatformComputeInsufficientCapacityNotification",
            "PlatformConcurrentRequestsLimitReachedNotification",
            "PlatformEndpointEnterpriseReadyNotification",
            "PlatformMaintenanceCompletedNotification",
            "PlatformMaintenanceScheduledNotification",
            "PlatformMaintenanceStartedNotification",
            "PlatformWelcomeWithCreditsNotification",
            "PlatformWelcomeWithoutCreditsNotification",
            "SecurityApiKeyCreatedNotification",
            "SecurityApiKeyRevokedNotification",
        ],
    ]
]:
    """Admin Create Notification

    Args:
        body (Union['CreateBillingAccountLockedNotification',
            'CreateBillingBudgetReachedNotification', 'CreateBillingBudgetWarningNotification',
            'CreateBillingOrderConfirmationNotification', 'CreateBillingPaymentFailedNotification',
            'CreateBillingPaymentSuccessfulNotification',
            'CreateBillingSpendingAlertTriggeredNotification',
            'CreateCollaborationTeamInviteNotification',
            'CreateCollaborationTeamMemberAddedNotification',
            'CreateCollaborationTeamMemberRemovedNotification',
            'CreateMarketingNewsletterNotification', 'CreateMarketingProductAnnouncementNotification',
            'CreatePlatformAnnouncementNotification', 'CreatePlatformAppCreatedNotification',
            'CreatePlatformAppDeletedNotification', 'CreatePlatformAppHttp5XxErrorsNotification',
            'CreatePlatformAppModifiedNotification', 'CreatePlatformAppOomNotification',
            'CreatePlatformAppQueueSizeNotification', 'CreatePlatformAppStartupFailureNotification',
            'CreatePlatformAppStartupTimeoutNotification',
            'CreatePlatformComputeInstanceCreatedNotification',
            'CreatePlatformComputeInstanceProvisioningFailedNotification',
            'CreatePlatformComputeInsufficientCapacityNotification',
            'CreatePlatformConcurrentRequestsLimitReachedNotification',
            'CreatePlatformEndpointEnterpriseReadyNotification',
            'CreatePlatformMaintenanceCompletedNotification',
            'CreatePlatformMaintenanceScheduledNotification',
            'CreatePlatformMaintenanceStartedNotification',
            'CreatePlatformWelcomeWithCreditsNotification',
            'CreatePlatformWelcomeWithoutCreditsNotification',
            'CreateSecurityApiKeyCreatedNotification', 'CreateSecurityApiKeyRevokedNotification']):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, Union['BillingAccountLockedNotification', 'BillingBudgetReachedNotification', 'BillingBudgetWarningNotification', 'BillingOrderConfirmationNotification', 'BillingPaymentFailedNotification', 'BillingPaymentSuccessfulNotification', 'BillingSpendingAlertTriggeredNotification', 'BillingSpendingAlertV2TriggeredNotification', 'CollaborationTeamInviteNotification', 'CollaborationTeamMemberAddedNotification', 'CollaborationTeamMemberRemovedNotification', 'MarketingNewsletterNotification', 'MarketingProductAnnouncementNotification', 'PlatformAnnouncementNotification', 'PlatformAppCreatedNotification', 'PlatformAppDeletedNotification', 'PlatformAppHttp5XxErrorsNotification', 'PlatformAppModifiedNotification', 'PlatformAppOomNotification', 'PlatformAppQueueSizeNotification', 'PlatformAppStartupFailureNotification', 'PlatformAppStartupTimeoutNotification', 'PlatformComputeInstanceCreatedNotification', 'PlatformComputeInstanceProvisioningFailedNotification', 'PlatformComputeInsufficientCapacityNotification', 'PlatformConcurrentRequestsLimitReachedNotification', 'PlatformEndpointEnterpriseReadyNotification', 'PlatformMaintenanceCompletedNotification', 'PlatformMaintenanceScheduledNotification', 'PlatformMaintenanceStartedNotification', 'PlatformWelcomeWithCreditsNotification', 'PlatformWelcomeWithoutCreditsNotification', 'SecurityApiKeyCreatedNotification', 'SecurityApiKeyRevokedNotification']]]
    """

    kwargs = _get_kwargs(
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: Union[AuthenticatedClient, Client],
    body: Union[
        "CreateBillingAccountLockedNotification",
        "CreateBillingBudgetReachedNotification",
        "CreateBillingBudgetWarningNotification",
        "CreateBillingOrderConfirmationNotification",
        "CreateBillingPaymentFailedNotification",
        "CreateBillingPaymentSuccessfulNotification",
        "CreateBillingSpendingAlertTriggeredNotification",
        "CreateCollaborationTeamInviteNotification",
        "CreateCollaborationTeamMemberAddedNotification",
        "CreateCollaborationTeamMemberRemovedNotification",
        "CreateMarketingNewsletterNotification",
        "CreateMarketingProductAnnouncementNotification",
        "CreatePlatformAnnouncementNotification",
        "CreatePlatformAppCreatedNotification",
        "CreatePlatformAppDeletedNotification",
        "CreatePlatformAppHttp5XxErrorsNotification",
        "CreatePlatformAppModifiedNotification",
        "CreatePlatformAppOomNotification",
        "CreatePlatformAppQueueSizeNotification",
        "CreatePlatformAppStartupFailureNotification",
        "CreatePlatformAppStartupTimeoutNotification",
        "CreatePlatformComputeInstanceCreatedNotification",
        "CreatePlatformComputeInstanceProvisioningFailedNotification",
        "CreatePlatformComputeInsufficientCapacityNotification",
        "CreatePlatformConcurrentRequestsLimitReachedNotification",
        "CreatePlatformEndpointEnterpriseReadyNotification",
        "CreatePlatformMaintenanceCompletedNotification",
        "CreatePlatformMaintenanceScheduledNotification",
        "CreatePlatformMaintenanceStartedNotification",
        "CreatePlatformWelcomeWithCreditsNotification",
        "CreatePlatformWelcomeWithoutCreditsNotification",
        "CreateSecurityApiKeyCreatedNotification",
        "CreateSecurityApiKeyRevokedNotification",
    ],
) -> Optional[
    Union[
        HTTPValidationError,
        Union[
            "BillingAccountLockedNotification",
            "BillingBudgetReachedNotification",
            "BillingBudgetWarningNotification",
            "BillingOrderConfirmationNotification",
            "BillingPaymentFailedNotification",
            "BillingPaymentSuccessfulNotification",
            "BillingSpendingAlertTriggeredNotification",
            "BillingSpendingAlertV2TriggeredNotification",
            "CollaborationTeamInviteNotification",
            "CollaborationTeamMemberAddedNotification",
            "CollaborationTeamMemberRemovedNotification",
            "MarketingNewsletterNotification",
            "MarketingProductAnnouncementNotification",
            "PlatformAnnouncementNotification",
            "PlatformAppCreatedNotification",
            "PlatformAppDeletedNotification",
            "PlatformAppHttp5XxErrorsNotification",
            "PlatformAppModifiedNotification",
            "PlatformAppOomNotification",
            "PlatformAppQueueSizeNotification",
            "PlatformAppStartupFailureNotification",
            "PlatformAppStartupTimeoutNotification",
            "PlatformComputeInstanceCreatedNotification",
            "PlatformComputeInstanceProvisioningFailedNotification",
            "PlatformComputeInsufficientCapacityNotification",
            "PlatformConcurrentRequestsLimitReachedNotification",
            "PlatformEndpointEnterpriseReadyNotification",
            "PlatformMaintenanceCompletedNotification",
            "PlatformMaintenanceScheduledNotification",
            "PlatformMaintenanceStartedNotification",
            "PlatformWelcomeWithCreditsNotification",
            "PlatformWelcomeWithoutCreditsNotification",
            "SecurityApiKeyCreatedNotification",
            "SecurityApiKeyRevokedNotification",
        ],
    ]
]:
    """Admin Create Notification

    Args:
        body (Union['CreateBillingAccountLockedNotification',
            'CreateBillingBudgetReachedNotification', 'CreateBillingBudgetWarningNotification',
            'CreateBillingOrderConfirmationNotification', 'CreateBillingPaymentFailedNotification',
            'CreateBillingPaymentSuccessfulNotification',
            'CreateBillingSpendingAlertTriggeredNotification',
            'CreateCollaborationTeamInviteNotification',
            'CreateCollaborationTeamMemberAddedNotification',
            'CreateCollaborationTeamMemberRemovedNotification',
            'CreateMarketingNewsletterNotification', 'CreateMarketingProductAnnouncementNotification',
            'CreatePlatformAnnouncementNotification', 'CreatePlatformAppCreatedNotification',
            'CreatePlatformAppDeletedNotification', 'CreatePlatformAppHttp5XxErrorsNotification',
            'CreatePlatformAppModifiedNotification', 'CreatePlatformAppOomNotification',
            'CreatePlatformAppQueueSizeNotification', 'CreatePlatformAppStartupFailureNotification',
            'CreatePlatformAppStartupTimeoutNotification',
            'CreatePlatformComputeInstanceCreatedNotification',
            'CreatePlatformComputeInstanceProvisioningFailedNotification',
            'CreatePlatformComputeInsufficientCapacityNotification',
            'CreatePlatformConcurrentRequestsLimitReachedNotification',
            'CreatePlatformEndpointEnterpriseReadyNotification',
            'CreatePlatformMaintenanceCompletedNotification',
            'CreatePlatformMaintenanceScheduledNotification',
            'CreatePlatformMaintenanceStartedNotification',
            'CreatePlatformWelcomeWithCreditsNotification',
            'CreatePlatformWelcomeWithoutCreditsNotification',
            'CreateSecurityApiKeyCreatedNotification', 'CreateSecurityApiKeyRevokedNotification']):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, Union['BillingAccountLockedNotification', 'BillingBudgetReachedNotification', 'BillingBudgetWarningNotification', 'BillingOrderConfirmationNotification', 'BillingPaymentFailedNotification', 'BillingPaymentSuccessfulNotification', 'BillingSpendingAlertTriggeredNotification', 'BillingSpendingAlertV2TriggeredNotification', 'CollaborationTeamInviteNotification', 'CollaborationTeamMemberAddedNotification', 'CollaborationTeamMemberRemovedNotification', 'MarketingNewsletterNotification', 'MarketingProductAnnouncementNotification', 'PlatformAnnouncementNotification', 'PlatformAppCreatedNotification', 'PlatformAppDeletedNotification', 'PlatformAppHttp5XxErrorsNotification', 'PlatformAppModifiedNotification', 'PlatformAppOomNotification', 'PlatformAppQueueSizeNotification', 'PlatformAppStartupFailureNotification', 'PlatformAppStartupTimeoutNotification', 'PlatformComputeInstanceCreatedNotification', 'PlatformComputeInstanceProvisioningFailedNotification', 'PlatformComputeInsufficientCapacityNotification', 'PlatformConcurrentRequestsLimitReachedNotification', 'PlatformEndpointEnterpriseReadyNotification', 'PlatformMaintenanceCompletedNotification', 'PlatformMaintenanceScheduledNotification', 'PlatformMaintenanceStartedNotification', 'PlatformWelcomeWithCreditsNotification', 'PlatformWelcomeWithoutCreditsNotification', 'SecurityApiKeyCreatedNotification', 'SecurityApiKeyRevokedNotification']]
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
        )
    ).parsed
