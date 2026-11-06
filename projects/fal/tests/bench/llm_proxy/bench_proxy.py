"""Run an LLMApp subclass (no GPU) in front of an already-running bench engine.

The engine "process" LLMApp launches is a sleeper; the real bench engine
(bench_engine.py) must already listen on --engine-port.
"""

from __future__ import annotations

import argparse
import cProfile
import os
import signal
import tempfile

import uvicorn

from fal.toolkit.llm import LLMApp


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--engine-port", type=int, required=True)
    p.add_argument("--loop", default="auto")
    p.add_argument("--http", default="auto")
    p.add_argument("--cprofile", default="")
    p.add_argument("--no-usage", action="store_true")
    a = p.parse_args()
    os.environ["IS_ISOLATE_AGENT"] = "1"
    model_dir = tempfile.mkdtemp()
    open(os.path.join(model_dir, "m.safetensors"), "wb").write(b"0" * 4096)

    class BenchLLM(LLMApp):
        model = model_dir
        engine_port = a.engine_port
        persistent_caches = False
        preread_workers = 0
        engine_startup_timeout_s = 60.0
        if a.no_usage:
            report_usage = False

        def engine_command(self, model_path):
            return ["sleep", "100000"]

    app = BenchLLM()
    prof = None
    if a.cprofile:
        prof = cProfile.Profile()
        prof.enable()

        def dump(*_):
            prof.disable()
            prof.dump_stats(a.cprofile)

        signal.signal(signal.SIGUSR1, dump)
    uvicorn.run(
        app._build_app(),
        host="127.0.0.1",
        port=a.port,
        log_level="warning",
        loop=a.loop,
        http=a.http,
        backlog=4096,
    )
    if prof:
        prof.disable()
        prof.dump_stats(a.cprofile)


if __name__ == "__main__":
    main()
