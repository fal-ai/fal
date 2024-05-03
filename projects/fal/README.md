[![PyPI](https://img.shields.io/pypi/v/fal.svg?logo=PyPI)](https://pypi.org/project/fal)
[![Tests](https://img.shields.io/github/actions/workflow/status/fal-ai/fal/integration_tests.yaml?label=Tests)](https://github.com/fal-ai/fal/actions)

# fal

fal is a serverless Python runtime that lets you run and scale code in the cloud with no infra management.

With fal, you can build pipelines, serve ML models and scale them up to many users. You scale down to 0 when you don't use any resources.

## Quickstart

First, you need to install the `fal` package. You can do so using pip:
```shell
pip install fal
```

Then you need to authenticate:
```shell
fal auth login
```

You can also use fal keys that you can get from [our dashboard](https://fal.ai/dashboard/keys).

Now can use the fal package in your Python scripts as follows:

```py
import fal

@fal.function(
    "virtualenv",
    requirements=["pyjokes"],
)
def tell_joke() -> str:
    import pyjokes

    joke = pyjokes.get_joke()
    return joke

print("Joke from the clouds: ", tell_joke())
```

A new virtual environment will be created by fal in the cloud and the set of requirements that we passed will be installed as soon as this function is called. From that point on, our code will be executed as if it were running locally, and the joke prepared by the pyjokes library will be returned.

## Next steps

If you would like to find out more about the capabilities of fal, check out to the [docs](https://fal.ai/docs). You can learn more about persistent storage, function caches and deploying your functions as API endpoints.

## Contributing

### Installing in editable mode with dev dependencies

```py
pip install -e 'projects/fal[dev]'
pip install -e 'projects/fal_client[dev]'
pip install -e 'projects/isolate_proto[dev]'
```

### Running tests

```py
pytest
```

### Pre-commit

```
cd projects/fal
pre-commit install
```

### Commit format

Please follow [conventional commits specification](https://www.conventionalcommits.org/) for descriptions/messages.
