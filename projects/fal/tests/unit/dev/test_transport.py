def test_post_json_keeps_payloads_out_of_request_storage(monkeypatch):
    import httpx

    from fal.dev import transport

    seen = {}

    def handler(request):
        seen.update(url=str(request.url), headers=request.headers, body=request.content)
        return httpx.Response(200)

    monkeypatch.setattr(
        transport, "_pool", [httpx.Client(transport=httpx.MockTransport(handler))]
    )

    transport.post_json(
        "https://fal.run/a/b/hot/files", {"x": 1}, {"Authorization": "Bearer t"}
    )

    assert seen["body"] == b'{"x":1}'
    assert seen["headers"]["authorization"] == "Bearer t"
    assert seen["headers"]["x-fal-store-io"] == "0"
