"""A lean HTTP/1.1 client for the engine's streaming responses.

httpx plus httpcore, h11 and anyio cost more than half of the proxy's CPU per
relayed chunk (a cancel scope, a timer and one parsed event per SSE event).
The engine is on localhost, so a streaming request needs none of that: one
connection per request (``Connection: close``), the body decoded in place, and
everything a socket read delivered returned in one piece. Closing the
connection is what makes the engine abort the request, exactly as before.
"""

import asyncio
import re
from typing import AsyncIterator, Dict, Optional, Tuple

_READ_BYTES = 256 * 1024
_MAX_HEAD_BYTES = 64 * 1024
_HEX = re.compile(rb"[0-9a-fA-F]+")
# How long a request waits for one of the open-stream slots (httpx's pool timeout).
_SLOT_TIMEOUT_S = 60.0


class UpstreamError(ConnectionError):
    """The engine closed or garbled a response."""


class UpstreamStream:
    """The response to one request; the subset of httpx.Response the proxy uses."""

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        status_code: int,
        headers: Dict[str, str],
        buffered: bytes,
        release: Optional[asyncio.Semaphore] = None,
    ) -> None:
        self._reader = reader
        self._writer = writer
        self.status_code = status_code
        self.headers = headers
        self._buf = bytearray(buffered)
        self._release = release
        self._closed = False
        self._chunked = "chunked" in headers.get("transfer-encoding", "").lower()
        length = headers.get("content-length")
        self._remaining = None
        if not self._chunked and length is not None:
            digits = length.strip()
            if not (digits.isascii() and digits.isdigit() and len(digits) <= 18):
                raise UpstreamError(f"bad Content-Length: {length!r}")
            self._remaining = int(digits)
        # Chunked decoding: bytes left in the current chunk (-1 = need a size line).
        self._chunk_left = -1
        self._done = False

    async def aiter_raw(self) -> AsyncIterator[bytes]:
        """Decoded body pieces, one per socket read that produced any."""
        while not self._done:
            out = self._decode()
            if out:
                yield out
                continue
            if self._done:
                break
            data = await self._reader.read(_READ_BYTES)
            if not data:
                if self._chunked or self._remaining:
                    raise UpstreamError("engine closed the response early")
                break  # close-delimited body (or empty content-length)
            self._buf += data

    async def aread(self) -> bytes:
        return b"".join([piece async for piece in self.aiter_raw()])

    def _decode(self) -> bytes:
        buf = self._buf
        if not self._chunked:
            if self._remaining is None:
                out = bytes(buf)
                buf.clear()
                return out
            take = min(len(buf), self._remaining)
            out = bytes(buf[:take])
            del buf[:take]
            self._remaining -= take
            self._done = self._remaining == 0
            return out
        parts = []
        while True:
            if self._chunk_left == -2:
                if buf.startswith(b"\r\n"):  # no trailers, just the final CRLF
                    del buf[:2]
                    self._done = True
                    break
                end = buf.find(b"\r\n\r\n")
                if end < 0:
                    if len(buf) > _MAX_HEAD_BYTES:
                        raise UpstreamError("trailers too long")
                    break
                del buf[: end + 4]
                self._done = True
                break
            elif self._chunk_left < 0:
                end = buf.find(b"\r\n")
                if end < 0:
                    if len(buf) > _MAX_HEAD_BYTES:
                        raise UpstreamError("chunk size line too long")
                    break
                digits = bytes(buf[:end]).split(b";", 1)[0]
                if not _HEX.fullmatch(digits):
                    raise UpstreamError("bad chunk size")
                size = int(digits, 16)
                del buf[: end + 2]
                if size == 0:
                    # Optional trailers, then the final blank line.
                    if buf.startswith(b"\r\n"):
                        del buf[:2]
                        self._done = True
                    else:
                        end = buf.find(b"\r\n\r\n")
                        if end < 0:
                            if len(buf) > _MAX_HEAD_BYTES:
                                raise UpstreamError("trailers too long")
                            self._chunk_left = -2  # waiting for the trailers
                            break
                        del buf[: end + 4]
                        self._done = True
                    break
                self._chunk_left = size
            else:
                need = self._chunk_left + 2  # the data and its CRLF
                if len(buf) < need:
                    # Hand over the part of the chunk already here.
                    take = min(len(buf), self._chunk_left)
                    if take:
                        parts.append(bytes(buf[:take]))
                        del buf[:take]
                        self._chunk_left -= take
                    break
                if buf[self._chunk_left : need] != b"\r\n":
                    raise UpstreamError("chunk not terminated by CRLF")
                parts.append(bytes(buf[: self._chunk_left]))
                del buf[:need]
                self._chunk_left = -1
        return b"".join(parts)

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            # Abort, not close: a close first flushes unsent request bytes,
            # which never completes against an engine that stopped reading.
            self._writer.transport.abort()
        finally:
            if self._release is not None:
                self._release.release()


async def _acquire(slots: asyncio.Semaphore) -> None:
    """Take a slot within ``_SLOT_TIMEOUT_S``, never leaking it on cancel.

    Not wait_for(): before Python 3.12 it can drop a semaphore acquired in the
    same tick as the timeout or cancellation.
    """
    task = asyncio.ensure_future(slots.acquire())
    try:
        await asyncio.wait({task}, timeout=_SLOT_TIMEOUT_S)
    except BaseException:
        task.cancel()
        if task.done() and not task.cancelled() and task.exception() is None:
            slots.release()
        raise
    if not task.done():
        task.cancel()
        try:
            await asyncio.gather(task, return_exceptions=True)
        except BaseException:
            if task.done() and not task.cancelled() and task.exception() is None:
                slots.release()
            raise
    if task.cancelled():
        raise UpstreamError("no free engine connection slot")


async def _connect(
    host: str, port: int, timeout: float
) -> Tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    """open_connection with a timeout that never orphans a connected socket.

    Not wait_for(): before Python 3.12 a cancel in the tick the connect
    finishes can be swallowed, leaving a live connection nobody closes.
    """
    task = asyncio.ensure_future(asyncio.open_connection(host, port, limit=_READ_BYTES))
    try:
        await asyncio.wait({task}, timeout=timeout)
    except BaseException:
        task.cancel()
        _abort_if_connected(task)
        raise
    if not task.done():
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        _abort_if_connected(task)
        raise UpstreamError("timed out connecting to the engine")
    return task.result()


def _abort_if_connected(task: "asyncio.Future") -> None:
    if task.done() and not task.cancelled() and task.exception() is None:
        task.result()[1].transport.abort()


async def open_stream(
    host: str,
    port: int,
    path: str,
    body: bytes,
    connect_timeout: float,
    slots: Optional[asyncio.Semaphore] = None,
) -> UpstreamStream:
    """POST ``body`` and return once the response head has been read.

    Cancelling closes the connection, which makes the engine abort the request.
    ``slots`` bounds how many streams are open at once.
    """
    if slots is not None:
        await _acquire(slots)
    writer: Optional[asyncio.StreamWriter] = None
    try:
        reader, writer = await _connect(host, port, connect_timeout)
        head = (
            f"POST {path} HTTP/1.1\r\nHost: {host}:{port}\r\n"
            "Content-Type: application/json\r\n"
            f"Content-Length: {len(body)}\r\n"
            "Accept: text/event-stream\r\nConnection: close\r\n\r\n"
        ).encode("latin-1")
        writer.write(head + body)
        buf = bytearray()
        while True:
            end = buf.find(b"\r\n\r\n")
            if end >= 0:
                break
            if len(buf) > _MAX_HEAD_BYTES:
                raise UpstreamError("response head too large")
            data = await reader.read(_READ_BYTES)
            if not data:
                raise UpstreamError("engine closed the connection before responding")
            buf += data
        status, headers = _parse_head(bytes(buf[:end]))
        leftover = bytes(buf[end + 4 :])
        return UpstreamStream(reader, writer, status, headers, leftover, slots)
    except BaseException:
        if writer is not None:
            writer.transport.abort()
        if slots is not None:
            slots.release()
        raise


def _parse_head(head: bytes) -> Tuple[int, Dict[str, str]]:
    lines = head.decode("latin-1").split("\r\n")
    try:
        status = int(lines[0].split(" ", 2)[1])
    except (IndexError, ValueError):
        raise UpstreamError(f"bad status line: {lines[0]!r}") from None
    headers: Dict[str, str] = {}
    for line in lines[1:]:
        name, _, value = line.partition(":")
        headers[name.strip().lower()] = value.strip()
    return status, headers
