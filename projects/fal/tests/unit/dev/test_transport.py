def test_post_json_keeps_payloads_out_of_request_storage(monkeypatch):
    import httpx

    from fal.dev.transport import post_json

    seen = {}

    def fake_post(url, **kwargs):
        seen.update(url=url, **kwargs)
        return httpx.Response(200)

    monkeypatch.setattr(httpx, "post", fake_post)

    post_json("https://fal.run/a/b/hot/files", {"x": 1}, {"Authorization": "Bearer t"})

    assert seen["json"] == {"x": 1}
    assert seen["headers"] == {"Authorization": "Bearer t", "X-Fal-Store-IO": "0"}
