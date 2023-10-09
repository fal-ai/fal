from __future__ import annotations

from http import HTTPStatus

from fal.rest_client import REST_CLIENT


# Billing
def test_get_upcoming_invoice():
    import openapi_fal_rest.api.billing.upcoming_invoice as get_upcoming_invoice

    res = get_upcoming_invoice.sync_detailed(client=REST_CLIENT)
    assert res.status_code == HTTPStatus.OK
    invoice = res.parsed
    assert invoice.status == "draft"


def test_get_user_details():
    import openapi_fal_rest.api.billing.get_user_details as get_user_details

    res = get_user_details.sync_detailed(client=REST_CLIENT)
    assert res.status_code == HTTPStatus.OK


def test_get_user_spending():
    import openapi_fal_rest.api.billing.get_user_spending as get_user_spending

    res = get_user_spending.sync_detailed(client=REST_CLIENT)
    assert res.status_code == HTTPStatus.OK


def test_get_setup_intent_key():
    import openapi_fal_rest.api.billing.get_setup_intent_key as get_key

    res = get_key.sync_detailed(client=REST_CLIENT)
    assert res.status_code == HTTPStatus.OK
    data = res.parsed
    split = data.split("_")
    assert split[0] == "seti"


def test_get_payment_methods():
    import openapi_fal_rest.api.billing.get_user_payment_methods as get_methods

    res = get_methods.sync_detailed(client=REST_CLIENT)
    assert res.status_code == HTTPStatus.OK


def test_get_user_invoices():
    import openapi_fal_rest.api.billing.get_user_invoices as get_user_invoices

    res = get_user_invoices.sync_detailed(client=REST_CLIENT)
    assert res.status_code == HTTPStatus.OK


# Gateway stats
def test_gateway_stats():
    from datetime import datetime, timedelta

    import openapi_fal_rest.api.usage.get_gateway_request_stats as get_stats

    end_time = datetime.now()
    start_time = end_time - timedelta(hours=3)
    res = get_stats.sync_detailed(
        client=REST_CLIENT, start_time=start_time, end_time=end_time
    )
    assert res.status_code == HTTPStatus.OK

    # Not allowed to request time frames longer than 5 weeks
    earlier_start_time = end_time - timedelta(weeks=6)
    res2 = get_stats.sync_detailed(
        client=REST_CLIENT, start_time=earlier_start_time, end_time=end_time
    )
    assert res2.status_code == HTTPStatus.BAD_REQUEST
