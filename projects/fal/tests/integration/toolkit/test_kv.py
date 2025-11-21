from fal.toolkit.kv import KVStore


def test_kv_store():
    kv = KVStore("test")
    kv.set("test", "test")
    assert kv.get("test") == "test"
    assert kv.get("test2") is None
