"""The raw-socket streaming client against hand-written HTTP responses."""

from __future__ import annotations

import asyncio

import pytest

from fal.toolkit.llm._upstream import UpstreamError, open_stream


async def _serve(handler):
    server = await asyncio.start_server(handler, "127.0.0.1", 0)
    return server, server.sockets[0].getsockname()[1]


def _chunked(*payloads: bytes) -> bytes:
    return b"".join(b"%x\r\n%s\r\n" % (len(p), p) for p in payloads)


HEAD = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n"


async def _roundtrip(raw_pieces, *, delay=0.0, close=True):
    """Serve ``raw_pieces`` (one write each) and return what the client decodes."""
    seen = {}

    async def handler(reader, writer):
        seen["request"] = await reader.readuntil(b"\r\n\r\n")
        for piece in raw_pieces:
            writer.write(piece)
            await writer.drain()
            if delay:
                await asyncio.sleep(delay)
        if close:
            writer.close()

    server, port = await _serve(handler)
    try:
        up = await open_stream("127.0.0.1", port, "/v1/x", b'{"a":1}', 5.0)
        try:
            pieces = [p async for p in up.aiter_raw()]
        finally:
            await up.aclose()
        return up, pieces, seen
    finally:
        server.close()


def test_chunked_events_arrive_whole_and_request_is_well_formed():
    body = [b"data: 1\n\n", b"data: 2\n\n", b"data: [DONE]\n\n"]
    up, pieces, seen = asyncio.run(_roundtrip([HEAD + _chunked(*body) + b"0\r\n\r\n"]))
    assert up.status_code == 200
    assert b"".join(pieces) == b"".join(body)
    assert seen["request"].startswith(b"POST /v1/x HTTP/1.1\r\n")
    assert b"Content-Length: 7" in seen["request"]
    assert b"Connection: close" in seen["request"]


def test_wire_splits_anywhere_decode_the_same():
    body = [b"data: " + b"x" * n + b"\n\n" for n in (1, 40, 300)]
    wire = HEAD + _chunked(*body) + b"0\r\n\r\n"
    # One byte per write: every split point of every size line, CRLF and body.
    _, pieces, _ = asyncio.run(
        _roundtrip([wire[i : i + 1] for i in range(len(wire))], delay=0)
    )
    assert b"".join(pieces) == b"".join(body)


def test_chunk_extensions_and_trailers_are_skipped():
    wire = HEAD + b"5;ext=1\r\nhello\r\n" + b"0\r\nX-Trailer: 1\r\n\r\n"
    _, pieces, _ = asyncio.run(_roundtrip([wire]))
    assert b"".join(pieces) == b"hello"


def test_content_length_body_and_error_status():
    wire = b"HTTP/1.1 400 Bad Request\r\nContent-Length: 11\r\n\r\n" + b'{"e":"boom"}'
    wire = wire[: wire.index(b"\r\n\r\n") + 4] + b'{"e":"boom"}'[:11]

    async def go():
        async def handler(reader, writer):
            await reader.readuntil(b"\r\n\r\n")
            writer.write(wire)
            await writer.drain()
            writer.close()

        server, port = await _serve(handler)
        try:
            up = await open_stream("127.0.0.1", port, "/", b"{}", 5.0)
            body = await up.aread()
            await up.aclose()
            return up.status_code, body
        finally:
            server.close()

    assert asyncio.run(go()) == (400, b'{"e":"boom"')


def test_close_delimited_body():
    wire = b"HTTP/1.1 200 OK\r\nConnection: close\r\n\r\ndata: x\n\n"
    _, pieces, _ = asyncio.run(_roundtrip([wire]))
    assert b"".join(pieces) == b"data: x\n\n"


def test_engine_closing_mid_stream_raises():
    wire = HEAD + _chunked(b"data: 1\n\n")  # no terminating chunk
    with pytest.raises(UpstreamError):
        asyncio.run(_roundtrip([wire]))


def test_garbage_chunk_size_raises():
    with pytest.raises(UpstreamError):
        asyncio.run(_roundtrip([HEAD + b"zz\r\nhello\r\n"]))


def test_closed_before_head_raises_and_frees_the_slot():
    async def go():
        async def handler(reader, writer):
            await reader.readuntil(b"\r\n\r\n")
            writer.close()

        server, port = await _serve(handler)
        slots = asyncio.Semaphore(1)
        try:
            with pytest.raises(UpstreamError):
                await open_stream("127.0.0.1", port, "/", b"{}", 5.0, slots)
            await asyncio.wait_for(slots.acquire(), 1)  # slot was released
        finally:
            server.close()

    asyncio.run(go())


def test_connect_failure_frees_the_slot():
    async def go():
        slots = asyncio.Semaphore(1)
        with pytest.raises(OSError):
            await open_stream("127.0.0.1", 1, "/", b"{}", 5.0, slots)
        await asyncio.wait_for(slots.acquire(), 1)

    asyncio.run(go())


def test_cancel_while_waiting_for_head_closes_the_connection():
    async def go():
        closed = asyncio.Event()

        async def handler(reader, writer):
            await reader.readuntil(b"\r\n\r\n")
            # Never answers; the client going away is what we wait for.
            await reader.readexactly(2)  # the request body
            assert await reader.read() == b""
            closed.set()

        server, port = await _serve(handler)
        slots = asyncio.Semaphore(1)
        try:
            task = asyncio.ensure_future(
                open_stream("127.0.0.1", port, "/", b"{}", 5.0, slots)
            )
            await asyncio.sleep(0.2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            await asyncio.wait_for(closed.wait(), 2)
            await asyncio.wait_for(slots.acquire(), 1)
        finally:
            server.close()

    asyncio.run(go())


def test_slots_bound_concurrent_streams():
    async def go():
        async def handler(reader, writer):
            await reader.readuntil(b"\r\n\r\n")
            writer.write(HEAD + _chunked(b"x") + b"0\r\n\r\n")
            await writer.drain()
            writer.close()

        server, port = await _serve(handler)
        slots = asyncio.Semaphore(1)
        try:
            first = await open_stream("127.0.0.1", port, "/", b"{}", 5.0, slots)
            second = asyncio.ensure_future(
                open_stream("127.0.0.1", port, "/", b"{}", 5.0, slots)
            )
            await asyncio.sleep(0.2)
            assert not second.done()
            await first.aclose()
            await first.aclose()  # idempotent: releases once
            up = await asyncio.wait_for(second, 2)
            assert await up.aread() == b"x"
            await up.aclose()
        finally:
            server.close()

    asyncio.run(go())


def test_terminator_split_after_the_zero_size_line():
    wire = HEAD + _chunked(b"abc")
    _, pieces, _ = asyncio.run(_roundtrip([wire, b"0\r\n", b"\r\n"], delay=0.05))
    assert b"".join(pieces) == b"abc"


def test_trailers_split_across_reads():
    wire = HEAD + _chunked(b"abc")
    pieces_in = [wire, b"0\r\nX-T: 1\r", b"\n\r", b"\n"]
    _, pieces, _ = asyncio.run(_roundtrip(pieces_in, delay=0.05))
    assert b"".join(pieces) == b"abc"


def test_slot_wait_times_out(monkeypatch):
    monkeypatch.setattr("fal.toolkit.llm._upstream._SLOT_TIMEOUT_S", 0.1)

    async def go():
        slots = asyncio.Semaphore(0)
        with pytest.raises(UpstreamError):
            await open_stream("127.0.0.1", 1, "/", b"{}", 5.0, slots)

    asyncio.run(go())


@pytest.mark.parametrize("size", [b"-5", b"1_0", b"0x10", b" 3", b""])
def test_lenient_chunk_sizes_are_rejected(size):
    with pytest.raises(UpstreamError):
        asyncio.run(_roundtrip([HEAD + size + b"\r\nabc\r\n"]))


def test_error_status_body_read_then_close_frees_the_slot():
    async def go():
        async def handler(reader, writer):
            await reader.readuntil(b"\r\n\r\n")
            writer.write(b"HTTP/1.1 500 X\r\nContent-Length: 2\r\n\r\n{}")
            await writer.drain()
            writer.close()

        server, port = await _serve(handler)
        slots = asyncio.Semaphore(1)
        try:
            up = await open_stream("127.0.0.1", port, "/", b"{}", 5.0, slots)
            assert up.status_code == 500
            assert await up.aread() == b"{}"
            await up.aclose()
            await asyncio.wait_for(slots.acquire(), 1)
        finally:
            server.close()

    asyncio.run(go())


def test_chunk_without_trailing_crlf_is_rejected():
    with pytest.raises(UpstreamError):
        asyncio.run(_roundtrip([HEAD + b"3\r\nabcXX0\r\n\r\n"]))


def test_cancel_in_the_tick_the_connect_finishes_does_not_leak(monkeypatch):
    opened, aborted = [], []

    class _Transport:
        def abort(self):
            aborted.append(1)

    class _Writer:
        transport = _Transport()

        def write(self, _):
            pass

    gate = asyncio.Event()
    entered = asyncio.Event()

    async def fake_open(*a, **k):
        entered.set()
        await gate.wait()
        opened.append(1)
        return object(), _Writer()

    monkeypatch.setattr("fal.toolkit.llm._upstream.asyncio.open_connection", fake_open)

    async def go():
        slots = asyncio.Semaphore(1)
        task = asyncio.ensure_future(open_stream("h", 1, "/", b"{}", 5.0, slots))
        await asyncio.wait_for(entered.wait(), 2)
        gate.set()  # the connect finishes in the same tick as the cancel
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0.05)
        await asyncio.wait_for(slots.acquire(), 1)

    asyncio.run(go())
    assert len(aborted) == len(opened)  # no connection left open


def test_bad_content_length_is_an_upstream_error_and_frees_the_slot():
    async def go():
        async def handler(reader, writer):
            await reader.readuntil(b"\r\n\r\n")
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: abc\r\n\r\n")
            await writer.drain()

        server, port = await _serve(handler)
        slots = asyncio.Semaphore(1)
        try:
            with pytest.raises(UpstreamError):
                await open_stream("127.0.0.1", port, "/", b"{}", 5.0, slots)
            await asyncio.wait_for(slots.acquire(), 1)
        finally:
            server.close()

    asyncio.run(go())
