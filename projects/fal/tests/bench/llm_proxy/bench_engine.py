"""Cheap CPU-only stand-in for a vLLM server, for proxy benchmarks.

Raw asyncio HTTP/1.1 with chunked transfer encoding, one SSE event per HTTP
chunk like vLLM, at a configurable token rate. Several processes can share a
port (SO_REUSEPORT) so the engine is never the bottleneck. Each event carries
its send time (``"ts"``) so a client can measure path latency per chunk.

Request body fields: n_tokens, tok_per_s (per stream), tokens_per_chunk.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import multiprocessing
import socket
import time

import uvloop

PAD = "x" * 8  # ~190 byte events, like a real chat.completion.chunk


class Proto(asyncio.Protocol):
    def connection_made(self, transport):
        self.t = transport
        self.buf = b""
        self.task = None
        self.closed = False

    def connection_lost(self, exc):
        self.closed = True
        if self.task:
            self.task.cancel()

    def data_received(self, data):
        self.buf += data
        head, sep, rest = self.buf.partition(b"\r\n\r\n")
        if not sep:
            return
        lines = head.split(b"\r\n")
        method, path, _ = lines[0].split(b" ", 2)
        length = 0
        for ln in lines[1:]:
            if ln.lower().startswith(b"content-length:"):
                length = int(ln.split(b":")[1])
        if len(rest) < length:
            return
        body, self.buf = rest[:length], rest[length:]
        if path == b"/health":
            self._plain(b"")
        elif path == b"/metrics":
            self._plain(b'vllm:num_requests_running{engine="0"} 2.0\n')
        elif path == b"/v1/models":
            self._plain(
                b'{"object":"list","data":[{"id":"fake"}]}', b"application/json"
            )
        else:
            req = json.loads(body or b"{}")
            if req.get("stream"):
                self.task = asyncio.ensure_future(self._stream(req))
            else:
                self._plain(
                    b'{"choices":[{"message":{"content":"hi"}}],"usage":'
                    b'{"prompt_tokens":7,"completion_tokens":5,"total_tokens":12}}',
                    b"application/json",
                )

    def _plain(self, body: bytes, ctype: bytes = b"text/plain"):
        self.t.write(
            b"HTTP/1.1 200 OK\r\nContent-Type: "
            + ctype
            + b"\r\nContent-Length: "
            + str(len(body)).encode()
            + b"\r\n\r\n"
            + body
        )

    def _event(self, data: str) -> bytes:
        b = f"data: {data}\n\n".encode()
        return b"%x\r\n" % len(b) + b + b"\r\n"

    async def _stream(self, req):
        n = int(req.get("n_tokens", 100))
        rate = float(req.get("tok_per_s", 50))
        tpc = int(req.get("tokens_per_chunk", 1))
        interval = tpc / rate
        self.t.write(
            b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
            b"Transfer-Encoding: chunked\r\n\r\n"
        )
        loop = asyncio.get_running_loop()
        nxt = loop.time()
        sent = 0
        try:
            while sent < n:
                k = min(tpc, n - sent)
                ev = (
                    '{"id":"chatcmpl-bench","object":"chat.completion.chunk",'
                    f'"created":1730000000,"ts":{time.time():.6f},"model":"fake",'
                    f'"choices":[{{"index":0,"delta":{{"content":"{PAD[:k]}"}},'
                    '"logprobs":null,"finish_reason":null}]}'
                )
                if self.closed:
                    return
                self.t.write(self._event(ev))
                sent += k
                nxt += interval
                delay = nxt - loop.time()
                if delay > 0:
                    await asyncio.sleep(delay)
            fin = (
                '{"id":"chatcmpl-bench","object":"chat.completion.chunk","created":1730000000,'
                f'"ts":{time.time():.6f},"model":"fake","choices":[{{"index":0,"delta":{{}},'
                '"finish_reason":"length"}]}'
            )
            self.t.write(self._event(fin))
            usage = {"prompt_tokens": 7, "completion_tokens": n, "total_tokens": n + 7}
            self.t.write(
                self._event(
                    json.dumps({"ts": time.time(), "choices": [], "usage": usage})
                )
            )
            self.t.write(self._event("[DONE]"))
            self.t.write(b"0\r\n\r\n")
        except asyncio.CancelledError:
            pass


def serve(port: int) -> None:
    async def main():
        s = socket.socket()
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", port))
        s.listen(4096)
        loop = asyncio.get_running_loop()
        await loop.create_server(Proto, sock=s, backlog=4096)
        await asyncio.Event().wait()

    uvloop.run(main())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--procs", type=int, default=4)
    a, _ = p.parse_known_args()
    ps = [multiprocessing.Process(target=serve, args=(a.port,)) for _ in range(a.procs)]
    for x in ps:
        x.start()
    for x in ps:
        x.join()
