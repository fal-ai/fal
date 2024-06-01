# gRPC definitions for Isolate Controller (this should really be called isolate_controller_proto)

For regenerating definitions:

```
$ cd projects/isolate_proto
$ pip install -e '.[dev]'
$ python ../../tools/regen_grpc.py --isolate-version <isolate version>
$ pre-commit run --all-files
```

The `<isolate version>` argument needs to be a [tag from the isolate Github project](https://github.com/fal-ai/isolate/tags) minus the leading `v`.
