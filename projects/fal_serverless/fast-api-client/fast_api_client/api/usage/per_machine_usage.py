import datetime
from http import HTTPStatus
from typing import Any, Dict, List, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.usage_per_machine_type import UsagePerMachineType
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    client: AuthenticatedClient,
    per_day: Union[Unset, None, bool] = False,
    start_date: Union[Unset, None, datetime.date] = UNSET,
    end_date: Union[Unset, None, datetime.date] = UNSET,
) -> Dict[str, Any]:
    url = "{}/usage/machine_type".format(client.base_url)

    headers: Dict[str, str] = client.get_headers()
    cookies: Dict[str, Any] = client.get_cookies()

    params: Dict[str, Any] = {}
    params["per_day"] = per_day

    json_start_date: Union[Unset, None, str] = UNSET
    if not isinstance(start_date, Unset):
        json_start_date = start_date.isoformat() if start_date else None

    params["start_date"] = json_start_date

    json_end_date: Union[Unset, None, str] = UNSET
    if not isinstance(end_date, Unset):
        json_end_date = end_date.isoformat() if end_date else None

    params["end_date"] = json_end_date

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    return {
        "method": "get",
        "url": url,
        "headers": headers,
        "cookies": cookies,
        "timeout": client.get_timeout(),
        "follow_redirects": client.follow_redirects,
        "params": params,
    }


def _parse_response(
    *, client: Client, response: httpx.Response
) -> Optional[Union[HTTPValidationError, List["UsagePerMachineType"]]]:
    if response.status_code == HTTPStatus.OK:
        response_200 = []
        _response_200 = response.json()
        for response_200_item_data in _response_200:
            response_200_item = UsagePerMachineType.from_dict(response_200_item_data)

            response_200.append(response_200_item)

        return response_200
    if response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY:
        response_422 = HTTPValidationError.from_dict(response.json())

        return response_422
    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: Client, response: httpx.Response
) -> Response[Union[HTTPValidationError, List["UsagePerMachineType"]]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient,
    per_day: Union[Unset, None, bool] = False,
    start_date: Union[Unset, None, datetime.date] = UNSET,
    end_date: Union[Unset, None, datetime.date] = UNSET,
) -> Response[Union[HTTPValidationError, List["UsagePerMachineType"]]]:
    """Per Machine Usage

    Args:
        per_day (Union[Unset, None, bool]):
        start_date (Union[Unset, None, datetime.date]):
        end_date (Union[Unset, None, datetime.date]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, List['UsagePerMachineType']]]
    """

    kwargs = _get_kwargs(
        client=client,
        per_day=per_day,
        start_date=start_date,
        end_date=end_date,
    )

    response = httpx.request(
        verify=client.verify_ssl,
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: AuthenticatedClient,
    per_day: Union[Unset, None, bool] = False,
    start_date: Union[Unset, None, datetime.date] = UNSET,
    end_date: Union[Unset, None, datetime.date] = UNSET,
) -> Optional[Union[HTTPValidationError, List["UsagePerMachineType"]]]:
    """Per Machine Usage

    Args:
        per_day (Union[Unset, None, bool]):
        start_date (Union[Unset, None, datetime.date]):
        end_date (Union[Unset, None, datetime.date]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, List['UsagePerMachineType']]
    """

    return sync_detailed(
        client=client,
        per_day=per_day,
        start_date=start_date,
        end_date=end_date,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient,
    per_day: Union[Unset, None, bool] = False,
    start_date: Union[Unset, None, datetime.date] = UNSET,
    end_date: Union[Unset, None, datetime.date] = UNSET,
) -> Response[Union[HTTPValidationError, List["UsagePerMachineType"]]]:
    """Per Machine Usage

    Args:
        per_day (Union[Unset, None, bool]):
        start_date (Union[Unset, None, datetime.date]):
        end_date (Union[Unset, None, datetime.date]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, List['UsagePerMachineType']]]
    """

    kwargs = _get_kwargs(
        client=client,
        per_day=per_day,
        start_date=start_date,
        end_date=end_date,
    )

    async with httpx.AsyncClient(verify=client.verify_ssl) as _client:
        response = await _client.request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient,
    per_day: Union[Unset, None, bool] = False,
    start_date: Union[Unset, None, datetime.date] = UNSET,
    end_date: Union[Unset, None, datetime.date] = UNSET,
) -> Optional[Union[HTTPValidationError, List["UsagePerMachineType"]]]:
    """Per Machine Usage

    Args:
        per_day (Union[Unset, None, bool]):
        start_date (Union[Unset, None, datetime.date]):
        end_date (Union[Unset, None, datetime.date]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, List['UsagePerMachineType']]
    """

    return (
        await asyncio_detailed(
            client=client,
            per_day=per_day,
            start_date=start_date,
            end_date=end_date,
        )
    ).parsed
