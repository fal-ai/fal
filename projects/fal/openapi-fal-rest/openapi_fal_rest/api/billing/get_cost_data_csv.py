import datetime
from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    start: datetime.datetime,
    end: datetime.datetime,
    by_auth: Union[Unset, bool] = UNSET,
) -> dict[str, Any]:
    params: dict[str, Any] = {}

    json_start = start.isoformat()
    params["start"] = json_start

    json_end = end.isoformat()
    params["end"] = json_end

    params["by_auth"] = by_auth

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/billing/cost_data_csv",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[Any, HTTPValidationError]]:
    if response.status_code == 200:
        response_200 = response.json()
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
) -> Response[Union[Any, HTTPValidationError]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    start: datetime.datetime,
    end: datetime.datetime,
    by_auth: Union[Unset, bool] = UNSET,
) -> Response[Union[Any, HTTPValidationError]]:
    """Get Cost Data Csv

     Get price evaluation data for the given time period in CSV format.

    Args:
        start (datetime.datetime):
        end (datetime.datetime):
        by_auth (Union[Unset, bool]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[Any, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        start=start,
        end=end,
        by_auth=by_auth,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: Union[AuthenticatedClient, Client],
    start: datetime.datetime,
    end: datetime.datetime,
    by_auth: Union[Unset, bool] = UNSET,
) -> Optional[Union[Any, HTTPValidationError]]:
    """Get Cost Data Csv

     Get price evaluation data for the given time period in CSV format.

    Args:
        start (datetime.datetime):
        end (datetime.datetime):
        by_auth (Union[Unset, bool]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[Any, HTTPValidationError]
    """

    return sync_detailed(
        client=client,
        start=start,
        end=end,
        by_auth=by_auth,
    ).parsed


async def asyncio_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    start: datetime.datetime,
    end: datetime.datetime,
    by_auth: Union[Unset, bool] = UNSET,
) -> Response[Union[Any, HTTPValidationError]]:
    """Get Cost Data Csv

     Get price evaluation data for the given time period in CSV format.

    Args:
        start (datetime.datetime):
        end (datetime.datetime):
        by_auth (Union[Unset, bool]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[Any, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        start=start,
        end=end,
        by_auth=by_auth,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: Union[AuthenticatedClient, Client],
    start: datetime.datetime,
    end: datetime.datetime,
    by_auth: Union[Unset, bool] = UNSET,
) -> Optional[Union[Any, HTTPValidationError]]:
    """Get Cost Data Csv

     Get price evaluation data for the given time period in CSV format.

    Args:
        start (datetime.datetime):
        end (datetime.datetime):
        by_auth (Union[Unset, bool]):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[Any, HTTPValidationError]
    """

    return (
        await asyncio_detailed(
            client=client,
            start=start,
            end=end,
            by_auth=by_auth,
        )
    ).parsed
