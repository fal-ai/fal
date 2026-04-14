[![PyPI](https://img.shields.io/pypi/v/fal.svg?logo=PyPI)](https://pypi.org/project/fal)
[![Tests](https://img.shields.io/github/actions/workflow/status/fal-ai/fal/fal-unit-tests.yml?label=Tests)](https://github.com/fal-ai/fal/actions)

# fal

fal is a serverless Python runtime that lets you run and scale code in the cloud with no infra management.

With fal, you can build pipelines, serve ML models and scale them up to many users. You scale down to 0 when you don't use any resources.

This repository contains the main Python packages for building on [fal](https://fal.ai):

- `fal` (in `projects/fal`): define, test, and deploy serverless apps on fal
- `fal-client` (in `projects/fal_client`): call fal model APIs or your deployed endpoints from Python

For full product and platform documentation, see [fal.ai/docs](https://fal.ai/docs/documentation).

## Which package should I use?

### Use `fal` when you want to deploy Python code to fal

The `fal` package includes the Python SDK and CLI for building serverless apps, testing them with temporary URLs, and deploying them to production.

```bash
pip install fal
fal auth login
```

Create a minimal app:

```python
import fal


class MyApp(fal.App):
    @fal.endpoint("/")
    def run(self) -> dict:
        return {"message": "Hello, World!"}
```

Run it on fal for testing:

```bash
fal run hello_world.py::MyApp
```

Deploy it to a persistent endpoint:

```bash
fal deploy hello_world.py::MyApp
```

Docs:

- [Quick start](https://fal.ai/docs/documentation/development/getting-started/quick-start)
- [Deploy to production](https://fal.ai/docs/documentation/deployment/deploy-to-production)
- [Serverless documentation](https://fal.ai/docs/documentation/serverless)

### Use `fal-client` when you want to call models or deployed endpoints

The `fal-client` package is the simplest way to call model APIs on fal from Python.

```bash
pip install fal-client
export FAL_KEY="your-api-key"
```

Call a model:

```python
import fal_client

result = fal_client.subscribe(
    "fal-ai/flux/schnell",
    arguments={
        "prompt": "a futuristic cityscape at sunset",
        "image_size": "landscape_16_9",
    },
)

print(result["images"][0]["url"])
```

You can also use `fal-client` to call your own deployed `fal` apps by passing your endpoint ID instead of a model ID.

Docs:

- [Client setup](https://fal.ai/docs/documentation/model-apis/inference/client-setup)
- [Inference methods](https://fal.ai/docs/documentation/model-apis/inference)
- [Model APIs documentation](https://fal.ai/docs/documentation/model-apis)

## Install from source

From the repository root:

```bash
pip install -e 'projects/fal[dev]'
pip install -e 'projects/fal_client[dev]'
pip install -e 'projects/isolate_proto[dev]'
```

## More resources

- [Documentation index](https://fal.ai/docs/documentation)
- [API reference](https://fal.ai/docs/api-reference)
- [Model gallery](https://fal.ai/models)
- [Dashboard](https://fal.ai/dashboard)
