from typing import Any, Dict, List, Type, TypeVar

import attr

T = TypeVar("T", bound="HandleStripeWebhookResponseHandleStripeWebhook")


@attr.s(auto_attribs=True)
class HandleStripeWebhookResponseHandleStripeWebhook:
    """ """

    additional_properties: Dict[str, bool] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        handle_stripe_webhook_response_handle_stripe_webhook = cls()

        handle_stripe_webhook_response_handle_stripe_webhook.additional_properties = d
        return handle_stripe_webhook_response_handle_stripe_webhook

    @property
    def additional_keys(self) -> List[str]:
        return list(self.additional_properties.keys())

    def __getitem__(self, key: str) -> bool:
        return self.additional_properties[key]

    def __setitem__(self, key: str, value: bool) -> None:
        self.additional_properties[key] = value

    def __delitem__(self, key: str) -> None:
        del self.additional_properties[key]

    def __contains__(self, key: str) -> bool:
        return key in self.additional_properties
