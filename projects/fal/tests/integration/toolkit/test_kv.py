import uuid

from fal.toolkit.kv import KVStore


def test_kv_store():
    kv = KVStore("test")
    kv.set("test", "test")
    assert kv.get("test") == "test"
    assert kv.get("test2") is None


def test_kv_store_delete():
    kv = KVStore("test")
    key = f"delete-{uuid.uuid4()}"

    kv.set(key, "value")
    assert kv.get(key) == "value"

    kv.delete(key)
    assert kv.get(key) is None

    kv.delete(key)


def test_kv_store_set_with_ttl():
    kv = KVStore("test")
    key = f"ttl-{uuid.uuid4()}"

    kv.set(key, "value", ttl=60)
    assert kv.get(key) == "value"

    kv.delete(key)
