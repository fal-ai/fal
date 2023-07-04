# fal-serverless

Library to run, serve or schedule your Python functions in the cloud with any machine type you may need.

Check out to the [docs](https://docs.fal.ai/fal-serverless/quickstart) for more details.

## Generate OpenAPI client for the REST API

The recommended way is to use bop. First set the `FAL_REST` variable to the right value and then run:

```sh
bop openapi
```

Initial client was generated using

```sh
cd projects/fal_serverless
# Notice that you can point to any environment
openapi-python-client generate --url https://rest.shark.fal.ai/openapi.json --meta none --config openapi_rest.config.yaml
```

And can be manually updated with

```sh
cd projects/fal_serverless
# Notice that you can point to any environment
openapi-python-client update --url https://rest.shark.fal.ai/openapi.json --meta none --config openapi_rest.config.yaml
```
