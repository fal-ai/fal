from http import HTTPStatus
from typing import Any, Optional, Union
from uuid import UUID

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
    notification_id: UUID,
) -> dict[str, Any]:
    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": f"/admin/notifications/{notification_id}",
    }

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
    if response.status_code == 200:

        def _parse_response_200(
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
                response_200_type_0 = BillingPaymentSuccessfulNotification.from_dict(data)

                return response_200_type_0
            except:  # noqa: E722
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                response_200_type_1 = BillingPaymentFailedNotification.from_dict(data)

                return response_200_type_1
            except:  # noqa: E722
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                response_200_type_2 = BillingBudgetWarningNotification.from_dict(data)

                return response_200_type_2
            except:  # noqa: E722
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                response_200_type_3 = BillingBudgetReachedNotification.from_dict(data)

                return response_200_type_3
            except:  # noqa: E722
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                response_200_type_4 = BillingOrderConfirmationNotification.from_dict(data)

                return response_200_type_4
            except:  # noqa: E722
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                response_200_type_5 = BillingAccountLockedNotification.from_dict(data)

                return response_200_type_5
            except:  # noqa: E722
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                response_200_type_6 = BillingSpendingAlertTriggeredNotification.from_dict(data)

                return response_200_type_6
            except:  # noqa: E722
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                response_200_type_7 = BillingSpendingAlertV2TriggeredNotification.from_dict(data)

                return response_200_type_7
            except:  # noqa: E722
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                response_200_type_8 = SecurityApiKeyCreatedNotification.from_dict(data)

                return response_200_type_8
            except:  # noqa: E722
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                response_200_type_9 = SecurityApiKeyRevokedNotification.from_dict(data)

                return response_200_type_9
            except:  # noqa: E722
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                response_200_type_10 = CollaborationTeamInviteNotification.from_dict(data)

                return response_200_type_10
            except:  # noqa: E722
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                response_200_type_11 = CollaborationTeamMemberAddedNotification.from_dict(data)

                return response_200_type_11
            except:  # noqa: E722
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                response_200_type_12 = CollaborationTeamMemberRemovedNotification.from_dict(data)

                return response_200_type_12
            except:  # noqa: E722
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                response_200_type_13 = MarketingNewsletterNotification.from_dict(data)

                return response_200_type_13
            except:  # noqa: E722
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                response_200_type_14 = MarketingProductAnnouncementNotification.from_dict(data)

                return response_200_type_14
            except:  # noqa: E722
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                response_200_type_15 = PlatformMaintenanceScheduledNotification.from_dict(data)

                return response_200_type_15
            except:  # noqa: E722
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                response_200_type_16 = PlatformMaintenanceStartedNotification.from_dict(data)

                return response_200_type_16
            except:  # noqa: E722
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                response_200_type_17 = PlatformMaintenanceCompletedNotification.from_dict(data)

                return response_200_type_17
            except:  # noqa: E722
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                response_200_type_18 = PlatformWelcomeWithCreditsNotification.from_dict(data)

                return response_200_type_18
            except:  # noqa: E722
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                response_200_type_19 = PlatformWelcomeWithoutCreditsNotification.from_dict(data)

                return response_200_type_19
            except:  # noqa: E722
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                response_200_type_20 = PlatformAppCreatedNotification.from_dict(data)

                return response_200_type_20
            except:  # noqa: E722
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                response_200_type_21 = PlatformAppModifiedNotification.from_dict(data)

                return response_200_type_21
            except:  # noqa: E722
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                response_200_type_22 = PlatformAppDeletedNotification.from_dict(data)

                return response_200_type_22
            except:  # noqa: E722
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                response_200_type_23 = PlatformAppQueueSizeNotification.from_dict(data)

                return response_200_type_23
            except:  # noqa: E722
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                response_200_type_24 = PlatformAppHttp5XxErrorsNotification.from_dict(data)

                return response_200_type_24
            except:  # noqa: E722
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                response_200_type_25 = PlatformAppOomNotification.from_dict(data)

                return response_200_type_25
            except:  # noqa: E722
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                response_200_type_26 = PlatformAppStartupTimeoutNotification.from_dict(data)

                return response_200_type_26
            except:  # noqa: E722
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                response_200_type_27 = PlatformAppStartupFailureNotification.from_dict(data)

                return response_200_type_27
            except:  # noqa: E722
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                response_200_type_28 = PlatformConcurrentRequestsLimitReachedNotification.from_dict(data)

                return response_200_type_28
            except:  # noqa: E722
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                response_200_type_29 = PlatformComputeInstanceCreatedNotification.from_dict(data)

                return response_200_type_29
            except:  # noqa: E722
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                response_200_type_30 = PlatformComputeInstanceProvisioningFailedNotification.from_dict(data)

                return response_200_type_30
            except:  # noqa: E722
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                response_200_type_31 = PlatformComputeInsufficientCapacityNotification.from_dict(data)

                return response_200_type_31
            except:  # noqa: E722
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                response_200_type_32 = PlatformEndpointEnterpriseReadyNotification.from_dict(data)

                return response_200_type_32
            except:  # noqa: E722
                pass
            if not isinstance(data, dict):
                raise TypeError()
            response_200_type_33 = PlatformAnnouncementNotification.from_dict(data)

            return response_200_type_33

        response_200 = _parse_response_200(response.json())

        return response_200
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
    notification_id: UUID,
    *,
    client: Union[AuthenticatedClient, Client],
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
    """Admin Get Notification

    Args:
        notification_id (UUID):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, Union['BillingAccountLockedNotification', 'BillingBudgetReachedNotification', 'BillingBudgetWarningNotification', 'BillingOrderConfirmationNotification', 'BillingPaymentFailedNotification', 'BillingPaymentSuccessfulNotification', 'BillingSpendingAlertTriggeredNotification', 'BillingSpendingAlertV2TriggeredNotification', 'CollaborationTeamInviteNotification', 'CollaborationTeamMemberAddedNotification', 'CollaborationTeamMemberRemovedNotification', 'MarketingNewsletterNotification', 'MarketingProductAnnouncementNotification', 'PlatformAnnouncementNotification', 'PlatformAppCreatedNotification', 'PlatformAppDeletedNotification', 'PlatformAppHttp5XxErrorsNotification', 'PlatformAppModifiedNotification', 'PlatformAppOomNotification', 'PlatformAppQueueSizeNotification', 'PlatformAppStartupFailureNotification', 'PlatformAppStartupTimeoutNotification', 'PlatformComputeInstanceCreatedNotification', 'PlatformComputeInstanceProvisioningFailedNotification', 'PlatformComputeInsufficientCapacityNotification', 'PlatformConcurrentRequestsLimitReachedNotification', 'PlatformEndpointEnterpriseReadyNotification', 'PlatformMaintenanceCompletedNotification', 'PlatformMaintenanceScheduledNotification', 'PlatformMaintenanceStartedNotification', 'PlatformWelcomeWithCreditsNotification', 'PlatformWelcomeWithoutCreditsNotification', 'SecurityApiKeyCreatedNotification', 'SecurityApiKeyRevokedNotification']]]
    """

    kwargs = _get_kwargs(
        notification_id=notification_id,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    notification_id: UUID,
    *,
    client: Union[AuthenticatedClient, Client],
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
    """Admin Get Notification

    Args:
        notification_id (UUID):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, Union['BillingAccountLockedNotification', 'BillingBudgetReachedNotification', 'BillingBudgetWarningNotification', 'BillingOrderConfirmationNotification', 'BillingPaymentFailedNotification', 'BillingPaymentSuccessfulNotification', 'BillingSpendingAlertTriggeredNotification', 'BillingSpendingAlertV2TriggeredNotification', 'CollaborationTeamInviteNotification', 'CollaborationTeamMemberAddedNotification', 'CollaborationTeamMemberRemovedNotification', 'MarketingNewsletterNotification', 'MarketingProductAnnouncementNotification', 'PlatformAnnouncementNotification', 'PlatformAppCreatedNotification', 'PlatformAppDeletedNotification', 'PlatformAppHttp5XxErrorsNotification', 'PlatformAppModifiedNotification', 'PlatformAppOomNotification', 'PlatformAppQueueSizeNotification', 'PlatformAppStartupFailureNotification', 'PlatformAppStartupTimeoutNotification', 'PlatformComputeInstanceCreatedNotification', 'PlatformComputeInstanceProvisioningFailedNotification', 'PlatformComputeInsufficientCapacityNotification', 'PlatformConcurrentRequestsLimitReachedNotification', 'PlatformEndpointEnterpriseReadyNotification', 'PlatformMaintenanceCompletedNotification', 'PlatformMaintenanceScheduledNotification', 'PlatformMaintenanceStartedNotification', 'PlatformWelcomeWithCreditsNotification', 'PlatformWelcomeWithoutCreditsNotification', 'SecurityApiKeyCreatedNotification', 'SecurityApiKeyRevokedNotification']]
    """

    return sync_detailed(
        notification_id=notification_id,
        client=client,
    ).parsed


async def asyncio_detailed(
    notification_id: UUID,
    *,
    client: Union[AuthenticatedClient, Client],
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
    """Admin Get Notification

    Args:
        notification_id (UUID):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, Union['BillingAccountLockedNotification', 'BillingBudgetReachedNotification', 'BillingBudgetWarningNotification', 'BillingOrderConfirmationNotification', 'BillingPaymentFailedNotification', 'BillingPaymentSuccessfulNotification', 'BillingSpendingAlertTriggeredNotification', 'BillingSpendingAlertV2TriggeredNotification', 'CollaborationTeamInviteNotification', 'CollaborationTeamMemberAddedNotification', 'CollaborationTeamMemberRemovedNotification', 'MarketingNewsletterNotification', 'MarketingProductAnnouncementNotification', 'PlatformAnnouncementNotification', 'PlatformAppCreatedNotification', 'PlatformAppDeletedNotification', 'PlatformAppHttp5XxErrorsNotification', 'PlatformAppModifiedNotification', 'PlatformAppOomNotification', 'PlatformAppQueueSizeNotification', 'PlatformAppStartupFailureNotification', 'PlatformAppStartupTimeoutNotification', 'PlatformComputeInstanceCreatedNotification', 'PlatformComputeInstanceProvisioningFailedNotification', 'PlatformComputeInsufficientCapacityNotification', 'PlatformConcurrentRequestsLimitReachedNotification', 'PlatformEndpointEnterpriseReadyNotification', 'PlatformMaintenanceCompletedNotification', 'PlatformMaintenanceScheduledNotification', 'PlatformMaintenanceStartedNotification', 'PlatformWelcomeWithCreditsNotification', 'PlatformWelcomeWithoutCreditsNotification', 'SecurityApiKeyCreatedNotification', 'SecurityApiKeyRevokedNotification']]]
    """

    kwargs = _get_kwargs(
        notification_id=notification_id,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    notification_id: UUID,
    *,
    client: Union[AuthenticatedClient, Client],
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
    """Admin Get Notification

    Args:
        notification_id (UUID):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, Union['BillingAccountLockedNotification', 'BillingBudgetReachedNotification', 'BillingBudgetWarningNotification', 'BillingOrderConfirmationNotification', 'BillingPaymentFailedNotification', 'BillingPaymentSuccessfulNotification', 'BillingSpendingAlertTriggeredNotification', 'BillingSpendingAlertV2TriggeredNotification', 'CollaborationTeamInviteNotification', 'CollaborationTeamMemberAddedNotification', 'CollaborationTeamMemberRemovedNotification', 'MarketingNewsletterNotification', 'MarketingProductAnnouncementNotification', 'PlatformAnnouncementNotification', 'PlatformAppCreatedNotification', 'PlatformAppDeletedNotification', 'PlatformAppHttp5XxErrorsNotification', 'PlatformAppModifiedNotification', 'PlatformAppOomNotification', 'PlatformAppQueueSizeNotification', 'PlatformAppStartupFailureNotification', 'PlatformAppStartupTimeoutNotification', 'PlatformComputeInstanceCreatedNotification', 'PlatformComputeInstanceProvisioningFailedNotification', 'PlatformComputeInsufficientCapacityNotification', 'PlatformConcurrentRequestsLimitReachedNotification', 'PlatformEndpointEnterpriseReadyNotification', 'PlatformMaintenanceCompletedNotification', 'PlatformMaintenanceScheduledNotification', 'PlatformMaintenanceStartedNotification', 'PlatformWelcomeWithCreditsNotification', 'PlatformWelcomeWithoutCreditsNotification', 'SecurityApiKeyCreatedNotification', 'SecurityApiKeyRevokedNotification']]
    """

    return (
        await asyncio_detailed(
            notification_id=notification_id,
            client=client,
        )
    ).parsed
