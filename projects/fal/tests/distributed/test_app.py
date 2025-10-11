import pytest
import fal
from fal.app import AppClient
from fal.distributed.worker import DistributedRunner, DistributedWorker
from pydantic import BaseModel

class EchoRequest(BaseModel):
    message: str

class EchoResponse(BaseModel):
    echo: str

class EchoWorker(DistributedWorker):
    def __call__(self, message: str = "", **kwargs):
        # Accept additional kwargs like 'streaming' that the runner may pass
        return {"echo": message}

class EchoApp(fal.App):
    # Specify requirements and machine type for remote deployment
    machine_type = "H100"
    num_gpus = 1  # Request a GPU machine for distributed processing
    keep_alive = 300
    requirements = [
        "pyzmq",  # Required for distributed communication
        "torch",  # Required for distributed processing
    ]
    
    async def setup(self):
        self.runner = DistributedRunner(EchoWorker, world_size=1)
        await self.runner.start()

    @fal.endpoint("/")
    async def echo(self, request: EchoRequest) -> EchoResponse:
        result = await self.runner.invoke({"message": request.message})
        # Handle potential error in result
        if "error" in result:
            raise RuntimeError(f"Worker error: {result['error']}")
        return EchoResponse(echo=result.get("echo", ""))

@pytest.fixture(scope="module")
def echo_app_client():
    with AppClient.connect(EchoApp) as client:
        yield client

@pytest.mark.skip(reason="Requires GPU machine")
def test_echo_endpoint(echo_app_client):
    """Test that the distributed echo app works correctly on remote GPU.
    
    This test verifies:
    1. The DistributedRunner can start on a remote GPU
    2. Workers can process requests
    3. Results are returned correctly
    """
    response = echo_app_client.echo(EchoRequest(message="hello"))
    assert response.echo == "hello"
    
    # Test with different message
    response2 = echo_app_client.echo(EchoRequest(message="world"))
    assert response2.echo == "world"
