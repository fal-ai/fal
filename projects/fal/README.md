[![PyPI](https://img.shields.io/pypi/v/fal.svg?logo=PyPI)](https://pypi.org/project/fal)
[![Tests](https://img.shields.io/github/actions/workflow/status/fal-ai/fal/fal-unit-tests.yml?label=Tests)](https://github.com/fal-ai/fal/actions)

# fal

fal is a serverless Python runtime that lets you run and scale code in the cloud with no infra management.

With fal, you can build pipelines, serve ML models and scale them up to many users. You scale down to 0 when you don't use any resources.

For full product and platform documentation, see [fal.ai/docs](https://fal.ai/docs/documentation).

## Quickstart

Install the package and authenticate:

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

## Next steps

If you want to go deeper, start with:

- [Quick start](https://fal.ai/docs/documentation/development/getting-started/quick-start)
- [Deploy to production](https://fal.ai/docs/documentation/deployment/deploy-to-production)
- [Serverless documentation](https://fal.ai/docs/documentation/serverless)

## Install from source

From the repository root:

```bash
pip install -e 'projects/fal[dev]'
```

## Contributing

### Running tests

Use the smallest relevant scope first:

```bash
pytest -n auto -v projects/fal/tests/unit
```

### Pre-commit

Run the repository hooks before opening or finishing work:

```bash
pre-commit run --all-files
```

### Commit format

Please follow the [Conventional Commits](https://www.conventionalcommits.org/) specification for commit messages.
