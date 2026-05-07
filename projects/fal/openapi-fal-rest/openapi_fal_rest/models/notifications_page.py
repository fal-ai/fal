from typing import TYPE_CHECKING, Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

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


T = TypeVar("T", bound="NotificationsPage")


@_attrs_define
class NotificationsPage:
    """
    Attributes:
        items (list[Union['BillingAccountLockedNotification', 'BillingBudgetReachedNotification',
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
            'SecurityApiKeyRevokedNotification']]):
        total (int):
        page (int):
        size (int):
        pages (int):
    """

    items: list[
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
        ]
    ]
    total: int
    page: int
    size: int
    pages: int
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

        items = []
        for items_item_data in self.items:
            items_item: dict[str, Any]
            if isinstance(items_item_data, BillingPaymentSuccessfulNotification):
                items_item = items_item_data.to_dict()
            elif isinstance(items_item_data, BillingPaymentFailedNotification):
                items_item = items_item_data.to_dict()
            elif isinstance(items_item_data, BillingBudgetWarningNotification):
                items_item = items_item_data.to_dict()
            elif isinstance(items_item_data, BillingBudgetReachedNotification):
                items_item = items_item_data.to_dict()
            elif isinstance(items_item_data, BillingOrderConfirmationNotification):
                items_item = items_item_data.to_dict()
            elif isinstance(items_item_data, BillingAccountLockedNotification):
                items_item = items_item_data.to_dict()
            elif isinstance(items_item_data, BillingSpendingAlertTriggeredNotification):
                items_item = items_item_data.to_dict()
            elif isinstance(items_item_data, BillingSpendingAlertV2TriggeredNotification):
                items_item = items_item_data.to_dict()
            elif isinstance(items_item_data, SecurityApiKeyCreatedNotification):
                items_item = items_item_data.to_dict()
            elif isinstance(items_item_data, SecurityApiKeyRevokedNotification):
                items_item = items_item_data.to_dict()
            elif isinstance(items_item_data, CollaborationTeamInviteNotification):
                items_item = items_item_data.to_dict()
            elif isinstance(items_item_data, CollaborationTeamMemberAddedNotification):
                items_item = items_item_data.to_dict()
            elif isinstance(items_item_data, CollaborationTeamMemberRemovedNotification):
                items_item = items_item_data.to_dict()
            elif isinstance(items_item_data, MarketingNewsletterNotification):
                items_item = items_item_data.to_dict()
            elif isinstance(items_item_data, MarketingProductAnnouncementNotification):
                items_item = items_item_data.to_dict()
            elif isinstance(items_item_data, PlatformMaintenanceScheduledNotification):
                items_item = items_item_data.to_dict()
            elif isinstance(items_item_data, PlatformMaintenanceStartedNotification):
                items_item = items_item_data.to_dict()
            elif isinstance(items_item_data, PlatformMaintenanceCompletedNotification):
                items_item = items_item_data.to_dict()
            elif isinstance(items_item_data, PlatformWelcomeWithCreditsNotification):
                items_item = items_item_data.to_dict()
            elif isinstance(items_item_data, PlatformWelcomeWithoutCreditsNotification):
                items_item = items_item_data.to_dict()
            elif isinstance(items_item_data, PlatformAppCreatedNotification):
                items_item = items_item_data.to_dict()
            elif isinstance(items_item_data, PlatformAppModifiedNotification):
                items_item = items_item_data.to_dict()
            elif isinstance(items_item_data, PlatformAppDeletedNotification):
                items_item = items_item_data.to_dict()
            elif isinstance(items_item_data, PlatformAppQueueSizeNotification):
                items_item = items_item_data.to_dict()
            elif isinstance(items_item_data, PlatformAppHttp5XxErrorsNotification):
                items_item = items_item_data.to_dict()
            elif isinstance(items_item_data, PlatformAppOomNotification):
                items_item = items_item_data.to_dict()
            elif isinstance(items_item_data, PlatformAppStartupTimeoutNotification):
                items_item = items_item_data.to_dict()
            elif isinstance(items_item_data, PlatformAppStartupFailureNotification):
                items_item = items_item_data.to_dict()
            elif isinstance(items_item_data, PlatformConcurrentRequestsLimitReachedNotification):
                items_item = items_item_data.to_dict()
            elif isinstance(items_item_data, PlatformComputeInstanceCreatedNotification):
                items_item = items_item_data.to_dict()
            elif isinstance(items_item_data, PlatformComputeInstanceProvisioningFailedNotification):
                items_item = items_item_data.to_dict()
            elif isinstance(items_item_data, PlatformComputeInsufficientCapacityNotification):
                items_item = items_item_data.to_dict()
            elif isinstance(items_item_data, PlatformEndpointEnterpriseReadyNotification):
                items_item = items_item_data.to_dict()
            else:
                items_item = items_item_data.to_dict()

            items.append(items_item)

        total = self.total

        page = self.page

        size = self.size

        pages = self.pages

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "items": items,
                "total": total,
                "page": page,
                "size": size,
                "pages": pages,
            }
        )

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
        items = []
        _items = d.pop("items")
        for items_item_data in _items:

            def _parse_items_item(
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
                    items_item_type_0 = BillingPaymentSuccessfulNotification.from_dict(data)

                    return items_item_type_0
                except:  # noqa: E722
                    pass
                try:
                    if not isinstance(data, dict):
                        raise TypeError()
                    items_item_type_1 = BillingPaymentFailedNotification.from_dict(data)

                    return items_item_type_1
                except:  # noqa: E722
                    pass
                try:
                    if not isinstance(data, dict):
                        raise TypeError()
                    items_item_type_2 = BillingBudgetWarningNotification.from_dict(data)

                    return items_item_type_2
                except:  # noqa: E722
                    pass
                try:
                    if not isinstance(data, dict):
                        raise TypeError()
                    items_item_type_3 = BillingBudgetReachedNotification.from_dict(data)

                    return items_item_type_3
                except:  # noqa: E722
                    pass
                try:
                    if not isinstance(data, dict):
                        raise TypeError()
                    items_item_type_4 = BillingOrderConfirmationNotification.from_dict(data)

                    return items_item_type_4
                except:  # noqa: E722
                    pass
                try:
                    if not isinstance(data, dict):
                        raise TypeError()
                    items_item_type_5 = BillingAccountLockedNotification.from_dict(data)

                    return items_item_type_5
                except:  # noqa: E722
                    pass
                try:
                    if not isinstance(data, dict):
                        raise TypeError()
                    items_item_type_6 = BillingSpendingAlertTriggeredNotification.from_dict(data)

                    return items_item_type_6
                except:  # noqa: E722
                    pass
                try:
                    if not isinstance(data, dict):
                        raise TypeError()
                    items_item_type_7 = BillingSpendingAlertV2TriggeredNotification.from_dict(data)

                    return items_item_type_7
                except:  # noqa: E722
                    pass
                try:
                    if not isinstance(data, dict):
                        raise TypeError()
                    items_item_type_8 = SecurityApiKeyCreatedNotification.from_dict(data)

                    return items_item_type_8
                except:  # noqa: E722
                    pass
                try:
                    if not isinstance(data, dict):
                        raise TypeError()
                    items_item_type_9 = SecurityApiKeyRevokedNotification.from_dict(data)

                    return items_item_type_9
                except:  # noqa: E722
                    pass
                try:
                    if not isinstance(data, dict):
                        raise TypeError()
                    items_item_type_10 = CollaborationTeamInviteNotification.from_dict(data)

                    return items_item_type_10
                except:  # noqa: E722
                    pass
                try:
                    if not isinstance(data, dict):
                        raise TypeError()
                    items_item_type_11 = CollaborationTeamMemberAddedNotification.from_dict(data)

                    return items_item_type_11
                except:  # noqa: E722
                    pass
                try:
                    if not isinstance(data, dict):
                        raise TypeError()
                    items_item_type_12 = CollaborationTeamMemberRemovedNotification.from_dict(data)

                    return items_item_type_12
                except:  # noqa: E722
                    pass
                try:
                    if not isinstance(data, dict):
                        raise TypeError()
                    items_item_type_13 = MarketingNewsletterNotification.from_dict(data)

                    return items_item_type_13
                except:  # noqa: E722
                    pass
                try:
                    if not isinstance(data, dict):
                        raise TypeError()
                    items_item_type_14 = MarketingProductAnnouncementNotification.from_dict(data)

                    return items_item_type_14
                except:  # noqa: E722
                    pass
                try:
                    if not isinstance(data, dict):
                        raise TypeError()
                    items_item_type_15 = PlatformMaintenanceScheduledNotification.from_dict(data)

                    return items_item_type_15
                except:  # noqa: E722
                    pass
                try:
                    if not isinstance(data, dict):
                        raise TypeError()
                    items_item_type_16 = PlatformMaintenanceStartedNotification.from_dict(data)

                    return items_item_type_16
                except:  # noqa: E722
                    pass
                try:
                    if not isinstance(data, dict):
                        raise TypeError()
                    items_item_type_17 = PlatformMaintenanceCompletedNotification.from_dict(data)

                    return items_item_type_17
                except:  # noqa: E722
                    pass
                try:
                    if not isinstance(data, dict):
                        raise TypeError()
                    items_item_type_18 = PlatformWelcomeWithCreditsNotification.from_dict(data)

                    return items_item_type_18
                except:  # noqa: E722
                    pass
                try:
                    if not isinstance(data, dict):
                        raise TypeError()
                    items_item_type_19 = PlatformWelcomeWithoutCreditsNotification.from_dict(data)

                    return items_item_type_19
                except:  # noqa: E722
                    pass
                try:
                    if not isinstance(data, dict):
                        raise TypeError()
                    items_item_type_20 = PlatformAppCreatedNotification.from_dict(data)

                    return items_item_type_20
                except:  # noqa: E722
                    pass
                try:
                    if not isinstance(data, dict):
                        raise TypeError()
                    items_item_type_21 = PlatformAppModifiedNotification.from_dict(data)

                    return items_item_type_21
                except:  # noqa: E722
                    pass
                try:
                    if not isinstance(data, dict):
                        raise TypeError()
                    items_item_type_22 = PlatformAppDeletedNotification.from_dict(data)

                    return items_item_type_22
                except:  # noqa: E722
                    pass
                try:
                    if not isinstance(data, dict):
                        raise TypeError()
                    items_item_type_23 = PlatformAppQueueSizeNotification.from_dict(data)

                    return items_item_type_23
                except:  # noqa: E722
                    pass
                try:
                    if not isinstance(data, dict):
                        raise TypeError()
                    items_item_type_24 = PlatformAppHttp5XxErrorsNotification.from_dict(data)

                    return items_item_type_24
                except:  # noqa: E722
                    pass
                try:
                    if not isinstance(data, dict):
                        raise TypeError()
                    items_item_type_25 = PlatformAppOomNotification.from_dict(data)

                    return items_item_type_25
                except:  # noqa: E722
                    pass
                try:
                    if not isinstance(data, dict):
                        raise TypeError()
                    items_item_type_26 = PlatformAppStartupTimeoutNotification.from_dict(data)

                    return items_item_type_26
                except:  # noqa: E722
                    pass
                try:
                    if not isinstance(data, dict):
                        raise TypeError()
                    items_item_type_27 = PlatformAppStartupFailureNotification.from_dict(data)

                    return items_item_type_27
                except:  # noqa: E722
                    pass
                try:
                    if not isinstance(data, dict):
                        raise TypeError()
                    items_item_type_28 = PlatformConcurrentRequestsLimitReachedNotification.from_dict(data)

                    return items_item_type_28
                except:  # noqa: E722
                    pass
                try:
                    if not isinstance(data, dict):
                        raise TypeError()
                    items_item_type_29 = PlatformComputeInstanceCreatedNotification.from_dict(data)

                    return items_item_type_29
                except:  # noqa: E722
                    pass
                try:
                    if not isinstance(data, dict):
                        raise TypeError()
                    items_item_type_30 = PlatformComputeInstanceProvisioningFailedNotification.from_dict(data)

                    return items_item_type_30
                except:  # noqa: E722
                    pass
                try:
                    if not isinstance(data, dict):
                        raise TypeError()
                    items_item_type_31 = PlatformComputeInsufficientCapacityNotification.from_dict(data)

                    return items_item_type_31
                except:  # noqa: E722
                    pass
                try:
                    if not isinstance(data, dict):
                        raise TypeError()
                    items_item_type_32 = PlatformEndpointEnterpriseReadyNotification.from_dict(data)

                    return items_item_type_32
                except:  # noqa: E722
                    pass
                if not isinstance(data, dict):
                    raise TypeError()
                items_item_type_33 = PlatformAnnouncementNotification.from_dict(data)

                return items_item_type_33

            items_item = _parse_items_item(items_item_data)

            items.append(items_item)

        total = d.pop("total")

        page = d.pop("page")

        size = d.pop("size")

        pages = d.pop("pages")

        notifications_page = cls(
            items=items,
            total=total,
            page=page,
            size=size,
            pages=pages,
        )

        notifications_page.additional_properties = d
        return notifications_page

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
