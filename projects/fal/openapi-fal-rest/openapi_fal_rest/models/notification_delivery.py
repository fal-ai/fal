import datetime
from typing import TYPE_CHECKING, Any, TypeVar, Union
from uuid import UUID

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.billing_account_locked_notification import BillingAccountLockedNotification
    from ..models.billing_budget_reached_notification import BillingBudgetReachedNotification
    from ..models.billing_budget_warning_notification import BillingBudgetWarningNotification
    from ..models.billing_order_confirmation_notification import BillingOrderConfirmationNotification
    from ..models.billing_payment_failed_notification import BillingPaymentFailedNotification
    from ..models.billing_payment_successful_notification import BillingPaymentSuccessfulNotification
    from ..models.billing_spending_alert_triggered_notification import BillingSpendingAlertTriggeredNotification
    from ..models.billing_spending_alert_v2_triggered_notification import BillingSpendingAlertV2TriggeredNotification
    from ..models.collaboration_team_invite_notification import CollaborationTeamInviteNotification
    from ..models.collaboration_team_member_added_notification import CollaborationTeamMemberAddedNotification
    from ..models.collaboration_team_member_removed_notification import CollaborationTeamMemberRemovedNotification
    from ..models.marketing_newsletter_notification import MarketingNewsletterNotification
    from ..models.marketing_product_announcement_notification import MarketingProductAnnouncementNotification
    from ..models.platform_announcement_notification import PlatformAnnouncementNotification
    from ..models.platform_app_created_notification import PlatformAppCreatedNotification
    from ..models.platform_app_deleted_notification import PlatformAppDeletedNotification
    from ..models.platform_app_http_5_xx_errors_notification import PlatformAppHttp5XxErrorsNotification
    from ..models.platform_app_modified_notification import PlatformAppModifiedNotification
    from ..models.platform_app_oom_notification import PlatformAppOomNotification
    from ..models.platform_app_queue_size_notification import PlatformAppQueueSizeNotification
    from ..models.platform_app_startup_failure_notification import PlatformAppStartupFailureNotification
    from ..models.platform_app_startup_timeout_notification import PlatformAppStartupTimeoutNotification
    from ..models.platform_compute_instance_created_notification import PlatformComputeInstanceCreatedNotification
    from ..models.platform_compute_instance_provisioning_failed_notification import (
        PlatformComputeInstanceProvisioningFailedNotification,
    )
    from ..models.platform_compute_insufficient_capacity_notification import (
        PlatformComputeInsufficientCapacityNotification,
    )
    from ..models.platform_concurrent_requests_limit_reached_notification import (
        PlatformConcurrentRequestsLimitReachedNotification,
    )
    from ..models.platform_endpoint_enterprise_ready_notification import PlatformEndpointEnterpriseReadyNotification
    from ..models.platform_maintenance_completed_notification import PlatformMaintenanceCompletedNotification
    from ..models.platform_maintenance_scheduled_notification import PlatformMaintenanceScheduledNotification
    from ..models.platform_maintenance_started_notification import PlatformMaintenanceStartedNotification
    from ..models.platform_welcome_with_credits_notification import PlatformWelcomeWithCreditsNotification
    from ..models.platform_welcome_without_credits_notification import PlatformWelcomeWithoutCreditsNotification
    from ..models.security_api_key_created_notification import SecurityApiKeyCreatedNotification
    from ..models.security_api_key_revoked_notification import SecurityApiKeyRevokedNotification


T = TypeVar("T", bound="NotificationDelivery")


@_attrs_define
class NotificationDelivery:
    """
    Attributes:
        delivery_id (UUID):
        notification_id (UUID):
        user_id (str):
        channel (str):
        status (str):
        notification (Union['BillingAccountLockedNotification', 'BillingBudgetReachedNotification',
            'BillingBudgetWarningNotification', 'BillingOrderConfirmationNotification', 'BillingPaymentFailedNotification',
            'BillingPaymentSuccessfulNotification', 'BillingSpendingAlertTriggeredNotification',
            'BillingSpendingAlertV2TriggeredNotification', 'CollaborationTeamInviteNotification',
            'CollaborationTeamMemberAddedNotification', 'CollaborationTeamMemberRemovedNotification',
            'MarketingNewsletterNotification', 'MarketingProductAnnouncementNotification',
            'PlatformAnnouncementNotification', 'PlatformAppCreatedNotification', 'PlatformAppDeletedNotification',
            'PlatformAppHttp5XxErrorsNotification', 'PlatformAppModifiedNotification', 'PlatformAppOomNotification',
            'PlatformAppQueueSizeNotification', 'PlatformAppStartupFailureNotification',
            'PlatformAppStartupTimeoutNotification', 'PlatformComputeInstanceCreatedNotification',
            'PlatformComputeInstanceProvisioningFailedNotification', 'PlatformComputeInsufficientCapacityNotification',
            'PlatformConcurrentRequestsLimitReachedNotification', 'PlatformEndpointEnterpriseReadyNotification',
            'PlatformMaintenanceCompletedNotification', 'PlatformMaintenanceScheduledNotification',
            'PlatformMaintenanceStartedNotification', 'PlatformWelcomeWithCreditsNotification',
            'PlatformWelcomeWithoutCreditsNotification', 'SecurityApiKeyCreatedNotification',
            'SecurityApiKeyRevokedNotification']):
        sent_at (Union[Unset, datetime.datetime]):
        delivered_at (Union[Unset, datetime.datetime]):
        failed_at (Union[Unset, datetime.datetime]):
        read_at (Union[Unset, datetime.datetime]):
    """

    delivery_id: UUID
    notification_id: UUID
    user_id: str
    channel: str
    status: str
    notification: Union[
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
    ]
    sent_at: Union[Unset, datetime.datetime] = UNSET
    delivered_at: Union[Unset, datetime.datetime] = UNSET
    failed_at: Union[Unset, datetime.datetime] = UNSET
    read_at: Union[Unset, datetime.datetime] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from ..models.billing_account_locked_notification import BillingAccountLockedNotification
        from ..models.billing_budget_reached_notification import BillingBudgetReachedNotification
        from ..models.billing_budget_warning_notification import BillingBudgetWarningNotification
        from ..models.billing_order_confirmation_notification import BillingOrderConfirmationNotification
        from ..models.billing_payment_failed_notification import BillingPaymentFailedNotification
        from ..models.billing_payment_successful_notification import BillingPaymentSuccessfulNotification
        from ..models.billing_spending_alert_triggered_notification import BillingSpendingAlertTriggeredNotification
        from ..models.billing_spending_alert_v2_triggered_notification import (
            BillingSpendingAlertV2TriggeredNotification,
        )
        from ..models.collaboration_team_invite_notification import CollaborationTeamInviteNotification
        from ..models.collaboration_team_member_added_notification import CollaborationTeamMemberAddedNotification
        from ..models.collaboration_team_member_removed_notification import CollaborationTeamMemberRemovedNotification
        from ..models.marketing_newsletter_notification import MarketingNewsletterNotification
        from ..models.marketing_product_announcement_notification import MarketingProductAnnouncementNotification
        from ..models.platform_app_created_notification import PlatformAppCreatedNotification
        from ..models.platform_app_deleted_notification import PlatformAppDeletedNotification
        from ..models.platform_app_http_5_xx_errors_notification import PlatformAppHttp5XxErrorsNotification
        from ..models.platform_app_modified_notification import PlatformAppModifiedNotification
        from ..models.platform_app_oom_notification import PlatformAppOomNotification
        from ..models.platform_app_queue_size_notification import PlatformAppQueueSizeNotification
        from ..models.platform_app_startup_failure_notification import PlatformAppStartupFailureNotification
        from ..models.platform_app_startup_timeout_notification import PlatformAppStartupTimeoutNotification
        from ..models.platform_compute_instance_created_notification import PlatformComputeInstanceCreatedNotification
        from ..models.platform_compute_instance_provisioning_failed_notification import (
            PlatformComputeInstanceProvisioningFailedNotification,
        )
        from ..models.platform_compute_insufficient_capacity_notification import (
            PlatformComputeInsufficientCapacityNotification,
        )
        from ..models.platform_concurrent_requests_limit_reached_notification import (
            PlatformConcurrentRequestsLimitReachedNotification,
        )
        from ..models.platform_endpoint_enterprise_ready_notification import PlatformEndpointEnterpriseReadyNotification
        from ..models.platform_maintenance_completed_notification import PlatformMaintenanceCompletedNotification
        from ..models.platform_maintenance_scheduled_notification import PlatformMaintenanceScheduledNotification
        from ..models.platform_maintenance_started_notification import PlatformMaintenanceStartedNotification
        from ..models.platform_welcome_with_credits_notification import PlatformWelcomeWithCreditsNotification
        from ..models.platform_welcome_without_credits_notification import PlatformWelcomeWithoutCreditsNotification
        from ..models.security_api_key_created_notification import SecurityApiKeyCreatedNotification
        from ..models.security_api_key_revoked_notification import SecurityApiKeyRevokedNotification

        delivery_id = str(self.delivery_id)

        notification_id = str(self.notification_id)

        user_id = self.user_id

        channel = self.channel

        status = self.status

        notification: dict[str, Any]
        if isinstance(self.notification, BillingPaymentSuccessfulNotification):
            notification = self.notification.to_dict()
        elif isinstance(self.notification, BillingPaymentFailedNotification):
            notification = self.notification.to_dict()
        elif isinstance(self.notification, BillingBudgetWarningNotification):
            notification = self.notification.to_dict()
        elif isinstance(self.notification, BillingBudgetReachedNotification):
            notification = self.notification.to_dict()
        elif isinstance(self.notification, BillingOrderConfirmationNotification):
            notification = self.notification.to_dict()
        elif isinstance(self.notification, BillingAccountLockedNotification):
            notification = self.notification.to_dict()
        elif isinstance(self.notification, BillingSpendingAlertTriggeredNotification):
            notification = self.notification.to_dict()
        elif isinstance(self.notification, BillingSpendingAlertV2TriggeredNotification):
            notification = self.notification.to_dict()
        elif isinstance(self.notification, SecurityApiKeyCreatedNotification):
            notification = self.notification.to_dict()
        elif isinstance(self.notification, SecurityApiKeyRevokedNotification):
            notification = self.notification.to_dict()
        elif isinstance(self.notification, CollaborationTeamInviteNotification):
            notification = self.notification.to_dict()
        elif isinstance(self.notification, CollaborationTeamMemberAddedNotification):
            notification = self.notification.to_dict()
        elif isinstance(self.notification, CollaborationTeamMemberRemovedNotification):
            notification = self.notification.to_dict()
        elif isinstance(self.notification, MarketingNewsletterNotification):
            notification = self.notification.to_dict()
        elif isinstance(self.notification, MarketingProductAnnouncementNotification):
            notification = self.notification.to_dict()
        elif isinstance(self.notification, PlatformMaintenanceScheduledNotification):
            notification = self.notification.to_dict()
        elif isinstance(self.notification, PlatformMaintenanceStartedNotification):
            notification = self.notification.to_dict()
        elif isinstance(self.notification, PlatformMaintenanceCompletedNotification):
            notification = self.notification.to_dict()
        elif isinstance(self.notification, PlatformWelcomeWithCreditsNotification):
            notification = self.notification.to_dict()
        elif isinstance(self.notification, PlatformWelcomeWithoutCreditsNotification):
            notification = self.notification.to_dict()
        elif isinstance(self.notification, PlatformAppCreatedNotification):
            notification = self.notification.to_dict()
        elif isinstance(self.notification, PlatformAppModifiedNotification):
            notification = self.notification.to_dict()
        elif isinstance(self.notification, PlatformAppDeletedNotification):
            notification = self.notification.to_dict()
        elif isinstance(self.notification, PlatformAppQueueSizeNotification):
            notification = self.notification.to_dict()
        elif isinstance(self.notification, PlatformAppHttp5XxErrorsNotification):
            notification = self.notification.to_dict()
        elif isinstance(self.notification, PlatformAppOomNotification):
            notification = self.notification.to_dict()
        elif isinstance(self.notification, PlatformAppStartupTimeoutNotification):
            notification = self.notification.to_dict()
        elif isinstance(self.notification, PlatformAppStartupFailureNotification):
            notification = self.notification.to_dict()
        elif isinstance(self.notification, PlatformConcurrentRequestsLimitReachedNotification):
            notification = self.notification.to_dict()
        elif isinstance(self.notification, PlatformComputeInstanceCreatedNotification):
            notification = self.notification.to_dict()
        elif isinstance(self.notification, PlatformComputeInstanceProvisioningFailedNotification):
            notification = self.notification.to_dict()
        elif isinstance(self.notification, PlatformComputeInsufficientCapacityNotification):
            notification = self.notification.to_dict()
        elif isinstance(self.notification, PlatformEndpointEnterpriseReadyNotification):
            notification = self.notification.to_dict()
        else:
            notification = self.notification.to_dict()

        sent_at: Union[Unset, str] = UNSET
        if not isinstance(self.sent_at, Unset):
            sent_at = self.sent_at.isoformat()

        delivered_at: Union[Unset, str] = UNSET
        if not isinstance(self.delivered_at, Unset):
            delivered_at = self.delivered_at.isoformat()

        failed_at: Union[Unset, str] = UNSET
        if not isinstance(self.failed_at, Unset):
            failed_at = self.failed_at.isoformat()

        read_at: Union[Unset, str] = UNSET
        if not isinstance(self.read_at, Unset):
            read_at = self.read_at.isoformat()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "delivery_id": delivery_id,
                "notification_id": notification_id,
                "user_id": user_id,
                "channel": channel,
                "status": status,
                "notification": notification,
            }
        )
        if sent_at is not UNSET:
            field_dict["sent_at"] = sent_at
        if delivered_at is not UNSET:
            field_dict["delivered_at"] = delivered_at
        if failed_at is not UNSET:
            field_dict["failed_at"] = failed_at
        if read_at is not UNSET:
            field_dict["read_at"] = read_at

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.billing_account_locked_notification import BillingAccountLockedNotification
        from ..models.billing_budget_reached_notification import BillingBudgetReachedNotification
        from ..models.billing_budget_warning_notification import BillingBudgetWarningNotification
        from ..models.billing_order_confirmation_notification import BillingOrderConfirmationNotification
        from ..models.billing_payment_failed_notification import BillingPaymentFailedNotification
        from ..models.billing_payment_successful_notification import BillingPaymentSuccessfulNotification
        from ..models.billing_spending_alert_triggered_notification import BillingSpendingAlertTriggeredNotification
        from ..models.billing_spending_alert_v2_triggered_notification import (
            BillingSpendingAlertV2TriggeredNotification,
        )
        from ..models.collaboration_team_invite_notification import CollaborationTeamInviteNotification
        from ..models.collaboration_team_member_added_notification import CollaborationTeamMemberAddedNotification
        from ..models.collaboration_team_member_removed_notification import CollaborationTeamMemberRemovedNotification
        from ..models.marketing_newsletter_notification import MarketingNewsletterNotification
        from ..models.marketing_product_announcement_notification import MarketingProductAnnouncementNotification
        from ..models.platform_announcement_notification import PlatformAnnouncementNotification
        from ..models.platform_app_created_notification import PlatformAppCreatedNotification
        from ..models.platform_app_deleted_notification import PlatformAppDeletedNotification
        from ..models.platform_app_http_5_xx_errors_notification import PlatformAppHttp5XxErrorsNotification
        from ..models.platform_app_modified_notification import PlatformAppModifiedNotification
        from ..models.platform_app_oom_notification import PlatformAppOomNotification
        from ..models.platform_app_queue_size_notification import PlatformAppQueueSizeNotification
        from ..models.platform_app_startup_failure_notification import PlatformAppStartupFailureNotification
        from ..models.platform_app_startup_timeout_notification import PlatformAppStartupTimeoutNotification
        from ..models.platform_compute_instance_created_notification import PlatformComputeInstanceCreatedNotification
        from ..models.platform_compute_instance_provisioning_failed_notification import (
            PlatformComputeInstanceProvisioningFailedNotification,
        )
        from ..models.platform_compute_insufficient_capacity_notification import (
            PlatformComputeInsufficientCapacityNotification,
        )
        from ..models.platform_concurrent_requests_limit_reached_notification import (
            PlatformConcurrentRequestsLimitReachedNotification,
        )
        from ..models.platform_endpoint_enterprise_ready_notification import PlatformEndpointEnterpriseReadyNotification
        from ..models.platform_maintenance_completed_notification import PlatformMaintenanceCompletedNotification
        from ..models.platform_maintenance_scheduled_notification import PlatformMaintenanceScheduledNotification
        from ..models.platform_maintenance_started_notification import PlatformMaintenanceStartedNotification
        from ..models.platform_welcome_with_credits_notification import PlatformWelcomeWithCreditsNotification
        from ..models.platform_welcome_without_credits_notification import PlatformWelcomeWithoutCreditsNotification
        from ..models.security_api_key_created_notification import SecurityApiKeyCreatedNotification
        from ..models.security_api_key_revoked_notification import SecurityApiKeyRevokedNotification

        d = src_dict.copy()
        delivery_id = UUID(d.pop("delivery_id"))

        notification_id = UUID(d.pop("notification_id"))

        user_id = d.pop("user_id")

        channel = d.pop("channel")

        status = d.pop("status")

        def _parse_notification(
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
                notification_type_0 = BillingPaymentSuccessfulNotification.from_dict(data)

                return notification_type_0
            except:  # noqa: E722
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                notification_type_1 = BillingPaymentFailedNotification.from_dict(data)

                return notification_type_1
            except:  # noqa: E722
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                notification_type_2 = BillingBudgetWarningNotification.from_dict(data)

                return notification_type_2
            except:  # noqa: E722
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                notification_type_3 = BillingBudgetReachedNotification.from_dict(data)

                return notification_type_3
            except:  # noqa: E722
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                notification_type_4 = BillingOrderConfirmationNotification.from_dict(data)

                return notification_type_4
            except:  # noqa: E722
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                notification_type_5 = BillingAccountLockedNotification.from_dict(data)

                return notification_type_5
            except:  # noqa: E722
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                notification_type_6 = BillingSpendingAlertTriggeredNotification.from_dict(data)

                return notification_type_6
            except:  # noqa: E722
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                notification_type_7 = BillingSpendingAlertV2TriggeredNotification.from_dict(data)

                return notification_type_7
            except:  # noqa: E722
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                notification_type_8 = SecurityApiKeyCreatedNotification.from_dict(data)

                return notification_type_8
            except:  # noqa: E722
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                notification_type_9 = SecurityApiKeyRevokedNotification.from_dict(data)

                return notification_type_9
            except:  # noqa: E722
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                notification_type_10 = CollaborationTeamInviteNotification.from_dict(data)

                return notification_type_10
            except:  # noqa: E722
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                notification_type_11 = CollaborationTeamMemberAddedNotification.from_dict(data)

                return notification_type_11
            except:  # noqa: E722
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                notification_type_12 = CollaborationTeamMemberRemovedNotification.from_dict(data)

                return notification_type_12
            except:  # noqa: E722
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                notification_type_13 = MarketingNewsletterNotification.from_dict(data)

                return notification_type_13
            except:  # noqa: E722
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                notification_type_14 = MarketingProductAnnouncementNotification.from_dict(data)

                return notification_type_14
            except:  # noqa: E722
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                notification_type_15 = PlatformMaintenanceScheduledNotification.from_dict(data)

                return notification_type_15
            except:  # noqa: E722
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                notification_type_16 = PlatformMaintenanceStartedNotification.from_dict(data)

                return notification_type_16
            except:  # noqa: E722
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                notification_type_17 = PlatformMaintenanceCompletedNotification.from_dict(data)

                return notification_type_17
            except:  # noqa: E722
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                notification_type_18 = PlatformWelcomeWithCreditsNotification.from_dict(data)

                return notification_type_18
            except:  # noqa: E722
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                notification_type_19 = PlatformWelcomeWithoutCreditsNotification.from_dict(data)

                return notification_type_19
            except:  # noqa: E722
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                notification_type_20 = PlatformAppCreatedNotification.from_dict(data)

                return notification_type_20
            except:  # noqa: E722
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                notification_type_21 = PlatformAppModifiedNotification.from_dict(data)

                return notification_type_21
            except:  # noqa: E722
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                notification_type_22 = PlatformAppDeletedNotification.from_dict(data)

                return notification_type_22
            except:  # noqa: E722
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                notification_type_23 = PlatformAppQueueSizeNotification.from_dict(data)

                return notification_type_23
            except:  # noqa: E722
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                notification_type_24 = PlatformAppHttp5XxErrorsNotification.from_dict(data)

                return notification_type_24
            except:  # noqa: E722
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                notification_type_25 = PlatformAppOomNotification.from_dict(data)

                return notification_type_25
            except:  # noqa: E722
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                notification_type_26 = PlatformAppStartupTimeoutNotification.from_dict(data)

                return notification_type_26
            except:  # noqa: E722
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                notification_type_27 = PlatformAppStartupFailureNotification.from_dict(data)

                return notification_type_27
            except:  # noqa: E722
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                notification_type_28 = PlatformConcurrentRequestsLimitReachedNotification.from_dict(data)

                return notification_type_28
            except:  # noqa: E722
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                notification_type_29 = PlatformComputeInstanceCreatedNotification.from_dict(data)

                return notification_type_29
            except:  # noqa: E722
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                notification_type_30 = PlatformComputeInstanceProvisioningFailedNotification.from_dict(data)

                return notification_type_30
            except:  # noqa: E722
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                notification_type_31 = PlatformComputeInsufficientCapacityNotification.from_dict(data)

                return notification_type_31
            except:  # noqa: E722
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                notification_type_32 = PlatformEndpointEnterpriseReadyNotification.from_dict(data)

                return notification_type_32
            except:  # noqa: E722
                pass
            if not isinstance(data, dict):
                raise TypeError()
            notification_type_33 = PlatformAnnouncementNotification.from_dict(data)

            return notification_type_33

        notification = _parse_notification(d.pop("notification"))

        _sent_at = d.pop("sent_at", UNSET)
        sent_at: Union[Unset, datetime.datetime]
        if isinstance(_sent_at, Unset):
            sent_at = UNSET
        else:
            sent_at = isoparse(_sent_at)

        _delivered_at = d.pop("delivered_at", UNSET)
        delivered_at: Union[Unset, datetime.datetime]
        if isinstance(_delivered_at, Unset):
            delivered_at = UNSET
        else:
            delivered_at = isoparse(_delivered_at)

        _failed_at = d.pop("failed_at", UNSET)
        failed_at: Union[Unset, datetime.datetime]
        if isinstance(_failed_at, Unset):
            failed_at = UNSET
        else:
            failed_at = isoparse(_failed_at)

        _read_at = d.pop("read_at", UNSET)
        read_at: Union[Unset, datetime.datetime]
        if isinstance(_read_at, Unset):
            read_at = UNSET
        else:
            read_at = isoparse(_read_at)

        notification_delivery = cls(
            delivery_id=delivery_id,
            notification_id=notification_id,
            user_id=user_id,
            channel=channel,
            status=status,
            notification=notification,
            sent_at=sent_at,
            delivered_at=delivered_at,
            failed_at=failed_at,
            read_at=read_at,
        )

        notification_delivery.additional_properties = d
        return notification_delivery

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
