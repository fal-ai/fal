# fal.ai Python client

This is a Python client library for interacting with ML models deployed on [fal.ai](https://fal.ai).

## Getting started

To install the client, run:

```bash
pip install fal-client
```

To use the client, you need to have an API key. You can get one by signing up at [fal.ai](https://fal.ai). Once you have it, set
it as an environment variable:

```bash
export FAL_KEY=your-api-key
```

Now you can use the client to interact with your models. Here's an example of how to use it:

```python
import fal_client

response = fal_client.run("fal-ai/fast-sdxl", arguments={"prompt": "a cute cat, realistic, orange"})
print(response["images"][0]["url"])
```

## Asynchronous requests

The client also supports asynchronous requests out of the box. Here's an example:

```python
import asyncio
import fal_client

async def main():
    response = await fal_client.run_async("fal-ai/fast-sdxl", arguments={"prompt": "a cute cat, realistic, orange"})
    print(response["images"][0]["url"])


asyncio.run(main())
```

## Uploading files

If the model requires files as input, you can upload them directly to fal.media (our CDN) and pass the URLs to the client. Here's an example:

```python
import fal_client

audio_url = fal_client.upload_file("path/to/audio.wav")
response = fal_client.run("fal-ai/whisper", arguments={"audio_url": audio_url})
print(response["text"])
```

## Encoding files as in-memory data URLs

If you don't want to upload your file to our CDN service (for latency reasons, for example), you can encode it as a data URL and pass it directly to the client. Here's an example:

```python
import fal_client

audio_data_url = fal_client.encode_file("path/to/audio.wav")
response = fal_client.run("fal-ai/whisper", arguments={"audio_url": audio_data_url})
print(response["text"])
```

## Queuing requests

When you want to send a request and keep receiving updates on its status, you can use the `submit` method. Here's an example:

```python
import asyncio
import fal_client

async def main():
    response = await fal_client.submit_async("fal-ai/fast-sdxl", arguments={"prompt": "a cute cat, realistic, orange"})

    logs_index = 0
    async for event in response.iter_events(with_logs=True):
        if isinstance(event, fal_client.Queued):
            print("Queued. Position:", event.position)
        elif isinstance(event, (fal_client.InProgress, fal_client.Completed)):
            new_logs = event.logs[logs_index:]
            for log in new_logs:
                print(log["message"])
            logs_index = len(event.logs)

    result = await response.get()
    print(result["images"][0]["url"])


asyncio.run(main())
```

