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


def test_update_budget():
    import openapi_fal_rest.api.billing.update_customer_budget as update_budget

    # Not allowed without a payment method
    res = update_budget.sync_detailed(
        client=REST_CLIENT, hard_monthly_budget=100, soft_monthly_budget=90
    )

    assert res.status_code == HTTPStatus.BAD_REQUEST


# Files
def test_relative_path_vs_absolute():
    import openapi_fal_rest.api.files.delete as delete_file
    import openapi_fal_rest.api.files.list_directory as list_dir
    import openapi_fal_rest.api.files.upload_from_url as upload_file
    import openapi_fal_rest.models.url_file_upload as url_file_upload

    # Delete file if it exists (ignore HTTP error if it does not exist)
    res = delete_file.sync_detailed(client=REST_CLIENT, file="test/google.png")

    res = upload_file.sync_detailed(
        client=REST_CLIENT,
        file="test/google.png",
        json_body=url_file_upload.UrlFileUpload(
            url="https://www.google.com/images/branding/googlelogo/1x/googlelogo_color_272x92dp.png"
        ),
    )
    assert res.status_code == HTTPStatus.OK

    for dir_test in ["test", "/data/test"]:
        res = list_dir.sync_detailed(client=REST_CLIENT, dir_=dir_test)
        assert res.status_code == HTTPStatus.OK
        assert isinstance(res.parsed, list), "Expected a list of files"
        # Maybe more files are present
        found = next(filter(lambda x: x.name == "google.png", res.parsed), None)
        assert (
            found
        ), f"Could not find file 'google.png' in directory '{dir_test}' (found {res.parsed}))"

        assert (
            found.is_file
        ), f"Expected 'google.png' to be a file, but it is a directory"
        assert (
            found.path == f"/data/test/google.png"
        ), f"Expected path to be '/data/test/google.png', but got '{found.path}'"


# Gateway stats
def test_gateway_stats():
    from datetime import datetime, timedelta

    import openapi_fal_rest.api.usage.get_gateway_request_stats as get_stats
    from openapi_fal_rest.models.stats_timeframe import StatsTimeframe

    end_time = datetime.now()
    start_time = end_time - timedelta(hours=3)
    res = get_stats.sync_detailed(
        client=REST_CLIENT,
        start_time=start_time,
        end_time=end_time,
        timeframe=StatsTimeframe.DAY,
    )
    assert res.status_code == HTTPStatus.OK

    end_time = datetime.now()
    start_time = end_time - timedelta(hours=3)
    res = get_stats.sync_detailed(
        client=REST_CLIENT,
        start_time=start_time,
        end_time=end_time,
        timeframe=StatsTimeframe.DAY,
        app_alias="test",
    )
    assert res.status_code == HTTPStatus.OK

    # Not allowed to request time frames longer than 24 weeks
    earlier_start_time = end_time - timedelta(weeks=30)
    res2 = get_stats.sync_detailed(
        client=REST_CLIENT,
        start_time=earlier_start_time,
        end_time=end_time,
        timeframe=StatsTimeframe.MONTH,
    )
    assert res2.status_code == HTTPStatus.BAD_REQUEST
