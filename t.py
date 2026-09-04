import time
import uuid

from fal.api import SyncServerlessClient

client = SyncServerlessClient()

for attempt in range(10):
    name = f"replica-lag-repro-{uuid.uuid4().hex[:8]}"
    print(f"testing {name}")
    env = client.environments.create(name)

    assert name in [env.name for env in client.environments.list()]

    client.environments.delete(name)

    assert name not in [env.name for env in client.environments.list()]

    time.sleep(0.1)
    print(f"Attempt {attempt + 1}/ 10 passed")
