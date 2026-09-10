import fal
import time
from fal.api.client import SyncServerlessClient
from fal.api.api import function
from fal.api.run import run
from queue import  Queue

def _devbox_process() -> None:
    print("Starting")
    time.sleep(24 * 60 * 60)


client = SyncServerlessClient(host="api.alpha.fal.ai", team=None)
host = client._create_host(local_file_path=str("/Users/vedat/fal-ai/fal"))
devbox_function = function(host=host)(_devbox_process)
devbox_function.options.host.update(
    machine_type="S",
    keep_alive=120,
)


errors: Queue = Queue(maxsize=1)

def run_devbox() -> None:
    try:
        run(devbox_function)
    except BaseException as exc:
        errors.put(exc)

run_devbox()
