from io import BytesIO
from urllib.error import HTTPError

import pytest

from fal.toolkit import kv
from fal.toolkit.exceptions import KVStoreException


@pytest.fixture
def not_found(monkeypatch):
    def raise_not_found(request):
        raise HTTPError(request.full_url, 404, "Not Found", {}, BytesIO(b""))

    monkeypatch.setattr(kv, "_maybe_retry_request", raise_not_found)
    monkeypatch.setattr(
        kv.KVStore, "auth_headers", property(lambda self: {"Authorization": "x"})
    )


def test_get_returns_none_on_404(not_found):
    assert kv.KVStore("db").get("key") is None


def test_set_raises_on_404(not_found):
    with pytest.raises(KVStoreException, match="Status 404"):
        kv.KVStore("db").set("key", "value")


def test_delete_raises_on_404(not_found):
    with pytest.raises(KVStoreException, match="Status 404"):
        kv.KVStore("db").delete("key")
