import datetime
from typing import TYPE_CHECKING, Any, TypeVar, Union, cast
from uuid import UUID

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

from ..models.billing_status import BillingStatus
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.gateway_request_item_json_input_type_0 import GatewayRequestItemJsonInputType0
    from ..models.gateway_request_item_json_output_type_0 import GatewayRequestItemJsonOutputType0
    from ..models.gateway_request_item_logs_item import GatewayRequestItemLogsItem
    from ..models.webhook_request_item import WebhookRequestItem


T = TypeVar("T", bound="GatewayRequestItem")


@_attrs_define
class GatewayRequestItem:
    """
    Attributes:
        request_id (UUID):
        started_at (datetime.datetime):
        sent_at (datetime.datetime):
        caller_user_id (Union[Unset, str]):
        queued_at (Union[Unset, datetime.datetime]):
        request_started_at (Union[Unset, datetime.datetime]):
        submitted_at (Union[Unset, datetime.datetime]):
        started_processing_at (Union[Unset, datetime.datetime]):
        first_byte_at (Union[Unset, datetime.datetime]):
        ended_at (Union[Unset, datetime.datetime]):
        endpoint (Union[Unset, str]):
        called_endpoint (Union[Unset, str]):
        json_input (Union['GatewayRequestItemJsonInputType0', Unset, bool, float, int, list[Any], str]):
        json_output (Union['GatewayRequestItemJsonOutputType0', Unset, bool, float, int, list[Any], str]):
        job_id (Union[Unset, str]):
        status_code (Union[Unset, int]):
        duration (Union[Unset, float]):
        logs (Union[Unset, list['GatewayRequestItemLogsItem']]):
        cost (Union[Unset, float]):
        cost_estimate_nano_usd (Union[Unset, float]):
        webhook (Union[Unset, WebhookRequestItem]):
        retried_by_request_id (Union[Unset, UUID]):
        billable_units (Union[Unset, float]):
        billing_status (Union[Unset, BillingStatus]):
        auth_method (Union[Unset, str]):
        billing_unit (Union[Unset, str]):
        error_type (Union[Unset, str]):
    """

    request_id: UUID
    started_at: datetime.datetime
    sent_at: datetime.datetime
    caller_user_id: Union[Unset, str] = UNSET
    queued_at: Union[Unset, datetime.datetime] = UNSET
    request_started_at: Union[Unset, datetime.datetime] = UNSET
    submitted_at: Union[Unset, datetime.datetime] = UNSET
    started_processing_at: Union[Unset, datetime.datetime] = UNSET
    first_byte_at: Union[Unset, datetime.datetime] = UNSET
    ended_at: Union[Unset, datetime.datetime] = UNSET
    endpoint: Union[Unset, str] = UNSET
    called_endpoint: Union[Unset, str] = UNSET
    json_input: Union["GatewayRequestItemJsonInputType0", Unset, bool, float, int, list[Any], str] = UNSET
    json_output: Union["GatewayRequestItemJsonOutputType0", Unset, bool, float, int, list[Any], str] = UNSET
    job_id: Union[Unset, str] = UNSET
    status_code: Union[Unset, int] = UNSET
    duration: Union[Unset, float] = UNSET
    logs: Union[Unset, list["GatewayRequestItemLogsItem"]] = UNSET
    cost: Union[Unset, float] = UNSET
    cost_estimate_nano_usd: Union[Unset, float] = UNSET
    webhook: Union[Unset, "WebhookRequestItem"] = UNSET
    retried_by_request_id: Union[Unset, UUID] = UNSET
    billable_units: Union[Unset, float] = UNSET
    billing_status: Union[Unset, BillingStatus] = UNSET
    auth_method: Union[Unset, str] = UNSET
    billing_unit: Union[Unset, str] = UNSET
    error_type: Union[Unset, str] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from ..models.gateway_request_item_json_input_type_0 import GatewayRequestItemJsonInputType0
        from ..models.gateway_request_item_json_output_type_0 import GatewayRequestItemJsonOutputType0

        request_id = str(self.request_id)

        started_at = self.started_at.isoformat()

        sent_at = self.sent_at.isoformat()

        caller_user_id = self.caller_user_id

        queued_at: Union[Unset, str] = UNSET
        if not isinstance(self.queued_at, Unset):
            queued_at = self.queued_at.isoformat()

        request_started_at: Union[Unset, str] = UNSET
        if not isinstance(self.request_started_at, Unset):
            request_started_at = self.request_started_at.isoformat()

        submitted_at: Union[Unset, str] = UNSET
        if not isinstance(self.submitted_at, Unset):
            submitted_at = self.submitted_at.isoformat()

        started_processing_at: Union[Unset, str] = UNSET
        if not isinstance(self.started_processing_at, Unset):
            started_processing_at = self.started_processing_at.isoformat()

        first_byte_at: Union[Unset, str] = UNSET
        if not isinstance(self.first_byte_at, Unset):
            first_byte_at = self.first_byte_at.isoformat()

        ended_at: Union[Unset, str] = UNSET
        if not isinstance(self.ended_at, Unset):
            ended_at = self.ended_at.isoformat()

        endpoint = self.endpoint

        called_endpoint = self.called_endpoint

        json_input: Union[Unset, bool, dict[str, Any], float, int, list[Any], str]
        if isinstance(self.json_input, Unset):
            json_input = UNSET
        elif isinstance(self.json_input, GatewayRequestItemJsonInputType0):
            json_input = self.json_input.to_dict()
        elif isinstance(self.json_input, list):
            json_input = self.json_input

        else:
            json_input = self.json_input

        json_output: Union[Unset, bool, dict[str, Any], float, int, list[Any], str]
        if isinstance(self.json_output, Unset):
            json_output = UNSET
        elif isinstance(self.json_output, GatewayRequestItemJsonOutputType0):
            json_output = self.json_output.to_dict()
        elif isinstance(self.json_output, list):
            json_output = self.json_output

        else:
            json_output = self.json_output

        job_id = self.job_id

        status_code = self.status_code

        duration = self.duration

        logs: Union[Unset, list[dict[str, Any]]] = UNSET
        if not isinstance(self.logs, Unset):
            logs = []
            for logs_item_data in self.logs:
                logs_item = logs_item_data.to_dict()
                logs.append(logs_item)

        cost = self.cost

        cost_estimate_nano_usd = self.cost_estimate_nano_usd

        webhook: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.webhook, Unset):
            webhook = self.webhook.to_dict()

        retried_by_request_id: Union[Unset, str] = UNSET
        if not isinstance(self.retried_by_request_id, Unset):
            retried_by_request_id = str(self.retried_by_request_id)

        billable_units = self.billable_units

        billing_status: Union[Unset, str] = UNSET
        if not isinstance(self.billing_status, Unset):
            billing_status = self.billing_status.value

        auth_method = self.auth_method

        billing_unit = self.billing_unit

        error_type = self.error_type

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "request_id": request_id,
                "started_at": started_at,
                "sent_at": sent_at,
            }
        )
        if caller_user_id is not UNSET:
            field_dict["caller_user_id"] = caller_user_id
        if queued_at is not UNSET:
            field_dict["queued_at"] = queued_at
        if request_started_at is not UNSET:
            field_dict["request_started_at"] = request_started_at
        if submitted_at is not UNSET:
            field_dict["submitted_at"] = submitted_at
        if started_processing_at is not UNSET:
            field_dict["started_processing_at"] = started_processing_at
        if first_byte_at is not UNSET:
            field_dict["first_byte_at"] = first_byte_at
        if ended_at is not UNSET:
            field_dict["ended_at"] = ended_at
        if endpoint is not UNSET:
            field_dict["endpoint"] = endpoint
        if called_endpoint is not UNSET:
            field_dict["called_endpoint"] = called_endpoint
        if json_input is not UNSET:
            field_dict["json_input"] = json_input
        if json_output is not UNSET:
            field_dict["json_output"] = json_output
        if job_id is not UNSET:
            field_dict["job_id"] = job_id
        if status_code is not UNSET:
            field_dict["status_code"] = status_code
        if duration is not UNSET:
            field_dict["duration"] = duration
        if logs is not UNSET:
            field_dict["logs"] = logs
        if cost is not UNSET:
            field_dict["cost"] = cost
        if cost_estimate_nano_usd is not UNSET:
            field_dict["cost_estimate_nano_usd"] = cost_estimate_nano_usd
        if webhook is not UNSET:
            field_dict["webhook"] = webhook
        if retried_by_request_id is not UNSET:
            field_dict["retried_by_request_id"] = retried_by_request_id
        if billable_units is not UNSET:
            field_dict["billable_units"] = billable_units
        if billing_status is not UNSET:
            field_dict["billing_status"] = billing_status
        if auth_method is not UNSET:
            field_dict["auth_method"] = auth_method
        if billing_unit is not UNSET:
            field_dict["billing_unit"] = billing_unit
        if error_type is not UNSET:
            field_dict["error_type"] = error_type

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.gateway_request_item_json_input_type_0 import GatewayRequestItemJsonInputType0
        from ..models.gateway_request_item_json_output_type_0 import GatewayRequestItemJsonOutputType0
        from ..models.gateway_request_item_logs_item import GatewayRequestItemLogsItem
        from ..models.webhook_request_item import WebhookRequestItem

        d = src_dict.copy()
        request_id = UUID(d.pop("request_id"))

        started_at = isoparse(d.pop("started_at"))

        sent_at = isoparse(d.pop("sent_at"))

        caller_user_id = d.pop("caller_user_id", UNSET)

        _queued_at = d.pop("queued_at", UNSET)
        queued_at: Union[Unset, datetime.datetime]
        if isinstance(_queued_at, Unset):
            queued_at = UNSET
        else:
            queued_at = isoparse(_queued_at)

        _request_started_at = d.pop("request_started_at", UNSET)
        request_started_at: Union[Unset, datetime.datetime]
        if isinstance(_request_started_at, Unset):
            request_started_at = UNSET
        else:
            request_started_at = isoparse(_request_started_at)

        _submitted_at = d.pop("submitted_at", UNSET)
        submitted_at: Union[Unset, datetime.datetime]
        if isinstance(_submitted_at, Unset):
            submitted_at = UNSET
        else:
            submitted_at = isoparse(_submitted_at)

        _started_processing_at = d.pop("started_processing_at", UNSET)
        started_processing_at: Union[Unset, datetime.datetime]
        if isinstance(_started_processing_at, Unset):
            started_processing_at = UNSET
        else:
            started_processing_at = isoparse(_started_processing_at)

        _first_byte_at = d.pop("first_byte_at", UNSET)
        first_byte_at: Union[Unset, datetime.datetime]
        if isinstance(_first_byte_at, Unset):
            first_byte_at = UNSET
        else:
            first_byte_at = isoparse(_first_byte_at)

        _ended_at = d.pop("ended_at", UNSET)
        ended_at: Union[Unset, datetime.datetime]
        if isinstance(_ended_at, Unset):
            ended_at = UNSET
        else:
            ended_at = isoparse(_ended_at)

        endpoint = d.pop("endpoint", UNSET)

        called_endpoint = d.pop("called_endpoint", UNSET)

        def _parse_json_input(
            data: object,
        ) -> Union["GatewayRequestItemJsonInputType0", Unset, bool, float, int, list[Any], str]:
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                json_input_type_0 = GatewayRequestItemJsonInputType0.from_dict(data)

                return json_input_type_0
            except:  # noqa: E722
                pass
            try:
                if not isinstance(data, list):
                    raise TypeError()
                json_input_type_1 = cast(list[Any], data)

                return json_input_type_1
            except:  # noqa: E722
                pass
            return cast(Union["GatewayRequestItemJsonInputType0", Unset, bool, float, int, list[Any], str], data)

        json_input = _parse_json_input(d.pop("json_input", UNSET))

        def _parse_json_output(
            data: object,
        ) -> Union["GatewayRequestItemJsonOutputType0", Unset, bool, float, int, list[Any], str]:
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                json_output_type_0 = GatewayRequestItemJsonOutputType0.from_dict(data)

                return json_output_type_0
            except:  # noqa: E722
                pass
            try:
                if not isinstance(data, list):
                    raise TypeError()
                json_output_type_1 = cast(list[Any], data)

                return json_output_type_1
            except:  # noqa: E722
                pass
            return cast(Union["GatewayRequestItemJsonOutputType0", Unset, bool, float, int, list[Any], str], data)

        json_output = _parse_json_output(d.pop("json_output", UNSET))

        job_id = d.pop("job_id", UNSET)

        status_code = d.pop("status_code", UNSET)

        duration = d.pop("duration", UNSET)

        logs = []
        _logs = d.pop("logs", UNSET)
        for logs_item_data in _logs or []:
            logs_item = GatewayRequestItemLogsItem.from_dict(logs_item_data)

            logs.append(logs_item)

        cost = d.pop("cost", UNSET)

        cost_estimate_nano_usd = d.pop("cost_estimate_nano_usd", UNSET)

        _webhook = d.pop("webhook", UNSET)
        webhook: Union[Unset, WebhookRequestItem]
        if isinstance(_webhook, Unset):
            webhook = UNSET
        else:
            webhook = WebhookRequestItem.from_dict(_webhook)

        _retried_by_request_id = d.pop("retried_by_request_id", UNSET)
        retried_by_request_id: Union[Unset, UUID]
        if isinstance(_retried_by_request_id, Unset):
            retried_by_request_id = UNSET
        else:
            retried_by_request_id = UUID(_retried_by_request_id)

        billable_units = d.pop("billable_units", UNSET)

        _billing_status = d.pop("billing_status", UNSET)
        billing_status: Union[Unset, BillingStatus]
        if isinstance(_billing_status, Unset):
            billing_status = UNSET
        else:
            billing_status = BillingStatus(_billing_status)

        auth_method = d.pop("auth_method", UNSET)

        billing_unit = d.pop("billing_unit", UNSET)

        error_type = d.pop("error_type", UNSET)

        gateway_request_item = cls(
            request_id=request_id,
            started_at=started_at,
            sent_at=sent_at,
            caller_user_id=caller_user_id,
            queued_at=queued_at,
            request_started_at=request_started_at,
            submitted_at=submitted_at,
            started_processing_at=started_processing_at,
            first_byte_at=first_byte_at,
            ended_at=ended_at,
            endpoint=endpoint,
            called_endpoint=called_endpoint,
            json_input=json_input,
            json_output=json_output,
            job_id=job_id,
            status_code=status_code,
            duration=duration,
            logs=logs,
            cost=cost,
            cost_estimate_nano_usd=cost_estimate_nano_usd,
            webhook=webhook,
            retried_by_request_id=retried_by_request_id,
            billable_units=billable_units,
            billing_status=billing_status,
            auth_method=auth_method,
            billing_unit=billing_unit,
            error_type=error_type,
        )

        gateway_request_item.additional_properties = d
        return gateway_request_item

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
