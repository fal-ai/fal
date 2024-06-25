from __future__ import annotations

import json
import webbrowser
from argparse import ArgumentParser
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterator, Union, cast

import graphlib
import rich
from openapi_fal_rest.api.workflows import (
    create_workflow as publish_workflow,
)
from openapi_fal_rest.models.http_validation_error import HTTPValidationError
from pydantic import BaseModel
from rich.syntax import Syntax

import fal
from fal import flags
from fal.exceptions import FalServerlessException
from fal.rest_client import REST_CLIENT

JSONType = Union[dict[str, Any], list[Any], str, int, float, bool, None, "Leaf"]
SchemaType = dict[str, Any]

VARIABLE_PREFIX = "$"
INPUT_VARIABLE_NAME = "input"

# Will be 1.0 once the server is finalized and anything <1.0
# is going to be rejected.
WORKFLOW_EXPORT_VERSION = "0.1"


class WorkflowSyntaxError(FalServerlessException):
    pass


class MisconfiguredGraphError(WorkflowSyntaxError):
    pass


def parse_leaf(raw_leaf: str) -> Leaf:
    """Parses a leaf (which is in the form of $variable.field.field_2[index] etc.)
    into a tree of Leaf objects."""
    raw_parts = raw_leaf.split(".")
    reference, *raw_parts = raw_parts
    if not reference.startswith(VARIABLE_PREFIX):
        raise WorkflowSyntaxError(
            f"Invalid leaf: {raw_leaf} (must start with a reference)"
        )

    leaf: Leaf = ReferenceLeaf(reference.removeprefix(VARIABLE_PREFIX))
    for raw_part in raw_parts:
        if raw_part.isdigit():
            leaf = IndexLeaf(leaf, int(raw_part))
        elif raw_part.isidentifier():
            leaf = AttributeLeaf(leaf, raw_part)
        else:
            raise WorkflowSyntaxError(
                f"Invalid leaf: {raw_leaf} (unexpected {raw_part})"
            )

    return leaf


def export_workflow_json(data: JSONType) -> JSONType:
    if isinstance(data, dict):
        return {k: export_workflow_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [export_workflow_json(v) for v in data]
    elif isinstance(data, Leaf):
        return repr(data)
    else:
        return data


def import_workflow_json(data: JSONType) -> JSONType:
    if isinstance(data, dict):
        return {k: import_workflow_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [import_workflow_json(v) for v in data]
    elif isinstance(data, str) and data.startswith(VARIABLE_PREFIX):
        return parse_leaf(data)
    else:
        return data


def iter_leaves(data: JSONType) -> Iterator[JSONType]:
    if isinstance(data, dict):
        for value in data.values():
            yield from iter_leaves(value)
    elif isinstance(data, list):
        for item in data:
            yield from iter_leaves(item)
    else:
        yield data


def depends(data: JSONType) -> set[str]:
    return {
        leaf.referee.id  # type: ignore
        for leaf in iter_leaves(data)
        if isinstance(leaf, Leaf)
    }


@dataclass
class Context:
    vars: dict[str, JSONType]

    def hydrate(self, input: JSONType) -> JSONType:
        if isinstance(input, dict):
            return {k: self.hydrate(v) for k, v in input.items()}
        elif isinstance(input, list):
            return [self.hydrate(v) for v in input]
        elif isinstance(input, Leaf):
            return input.execute(self)
        else:
            return input


@dataclass
class Leaf:
    def execute(self, context: Context) -> JSONType:
        raise NotImplementedError

    def __getattr__(self, name: str) -> AttributeLeaf:
        return AttributeLeaf(self, name)

    def __getitem__(self, index: int) -> IndexLeaf:
        return IndexLeaf(self, index)

    @property
    def referee(self) -> ReferenceLeaf:
        raise NotImplementedError


@dataclass
class AttributeLeaf(Leaf):
    leaf: Leaf
    attribute: str

    def execute(self, context: Context) -> JSONType:
        output = self.leaf.execute(context)
        assert isinstance(output, dict), f"{self.leaf!r} is not a dict"
        return output[self.attribute]

    def __repr__(self) -> str:
        return f"{self.leaf!r}.{self.attribute}"

    @property
    def referee(self) -> ReferenceLeaf:
        return self.leaf.referee


@dataclass
class IndexLeaf(Leaf):
    leaf: Leaf
    index: int

    def execute(self, context: Context) -> JSONType:
        output = self.leaf.execute(context)
        assert isinstance(output, list), f"{self.leaf!r} is not an array"
        return output[self.index]

    def __repr__(self) -> str:
        return f"{self.leaf!r}.{self.index}"

    @property
    def referee(self) -> ReferenceLeaf:
        return self.leaf.referee


@dataclass
class ReferenceLeaf(Leaf):
    id: str

    def execute(self, context: Context) -> JSONType:
        try:
            return context.vars[self.id]
        except KeyError:
            raise MisconfiguredGraphError(f"Variable {self.id!r} is not defined")

    def __repr__(self) -> str:
        return VARIABLE_PREFIX + self.id

    @property
    def referee(self) -> ReferenceLeaf:
        return self


@dataclass
class Node:
    id: str
    depends: set[str]

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Node:
        type = data.pop("type")
        if type == "display":
            return Display.from_json(data)
        elif type == "run":
            return Run.from_json(data)
        else:
            raise WorkflowSyntaxError(f"Invalid node type: {type}")

    def to_json(self) -> dict[str, Any]:
        raise NotImplementedError

    def execute(self, context: Context) -> JSONType:
        raise NotImplementedError


@dataclass
class Display(Node):
    fields: list[Leaf]

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Display:
        return cls(
            id=data["id"],
            depends=set(data["depends"]),
            fields=import_workflow_json(data["fields"]),  # type: ignore
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "type": "display",
            "id": self.id,
            "depends": list(self.depends),
            "fields": export_workflow_json(self.fields),
        }

    def execute(self, context: Context) -> JSONType:
        for url in context.hydrate(self.fields):  # type: ignore
            if flags.DONT_OPEN_LINKS:
                print("Link:", url)
            else:
                webbrowser.open(url)


@dataclass
class Run(Node):
    app: str
    input: JSONType

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Run:
        return cls(
            id=data["id"],
            depends=set(data["depends"]),
            app=data["app"],
            input=import_workflow_json(data["input"]),
        )

    def execute(self, context: Context) -> JSONType:
        input = context.hydrate(self.input)
        assert isinstance(input, dict)
        return cast(JSONType, fal.apps.run(self.app, input))

    def to_json(self) -> dict[str, Any]:
        return {
            "type": "run",
            "id": self.id,
            "app": self.app,
            "depends": list(self.depends),
            "input": export_workflow_json(self.input),  # type: ignore
        }


@dataclass
class Workflow:
    name: str
    input_schema: SchemaType
    output_schema: SchemaType
    nodes: dict[str, Node] = field(default_factory=dict)
    output: dict[str, Any] | None = None
    _app_counter: Counter = field(default_factory=Counter)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Workflow:
        data = import_workflow_json(data)  # type: ignore
        return cls(
            name=data["name"],
            input_schema=data["schema"]["input"],
            output_schema=data["schema"]["output"],
            nodes={
                node_id: Node.from_json(node_data)
                for node_id, node_data in data["nodes"].items()
            },
            output=data["output"],
        )

    def __post_init__(self) -> None:
        for node in self.nodes.values():
            if isinstance(node, Run):
                self._app_counter[node.app] += 1

    def _generate_node_id(self, app: str) -> str:
        self._app_counter[app] += 1
        return f"{app.replace('/', '_').replace('-', '_')}_{self._app_counter[app]}"

    def run(self, app: str, input: JSONType) -> ReferenceLeaf:
        node_id = self._generate_node_id(app)
        node = self.nodes[node_id] = Run(
            id=node_id,
            depends=depends(input),
            app=app,
            input=input,
        )
        return ReferenceLeaf(node.id)

    def display(self, *fields: Leaf) -> None:
        node_id = self._generate_node_id("display")
        self.nodes[node_id] = Display(
            node_id,
            depends=depends(list(fields)),
            fields=list(fields),
        )

    def set_output(self, output: JSONType) -> None:
        self.output = output  # type: ignore

    def execute(self, input: JSONType) -> JSONType:
        if not self.output:
            raise WorkflowSyntaxError(
                "Can't execute the workflow before the output is set."
            )

        context = Context({INPUT_VARIABLE_NAME: input})

        sorter = graphlib.TopologicalSorter(
            graph={
                node.id: node.depends - {INPUT_VARIABLE_NAME}
                for node in self.nodes.values()
            }
        )
        for node_id in sorter.static_order():
            node = self.nodes[node_id]
            context.vars[node_id] = node.execute(context)

        return context.hydrate(self.output)

    @property
    def input(self) -> ReferenceLeaf:
        return ReferenceLeaf(INPUT_VARIABLE_NAME)

    def to_json(self) -> dict[str, JSONType]:
        if not self.output:
            raise WorkflowSyntaxError(
                "Can't serialize the workflow before the output is set."
            )

        return {
            "name": self.name,
            "schema": {
                "input": self.input_schema,
                "output": self.output_schema,
            },
            "nodes": {node.id: node.to_json() for node in self.nodes.values()},
            "output": export_workflow_json(self.output),
            "version": WORKFLOW_EXPORT_VERSION,
        }

    to_dict = to_json

    def publish(self, title: str, *, is_public: bool = True):
        workflow_contents = publish_workflow.TypedWorkflow(
            name=self.name,
            title=title,
            contents=self,  # type: ignore
            is_public=is_public,
        )
        published_workflow = publish_workflow.sync(
            client=REST_CLIENT,
            json_body=workflow_contents,
        )
        if isinstance(published_workflow, Exception):
            raise published_workflow
        if isinstance(published_workflow, HTTPValidationError):
            raise RuntimeError(published_workflow.detail)
        if not published_workflow:
            raise RuntimeError("Failed to publish the workflow")

        # NOTE: dropping the provider prefix from the user_id
        user_id_part = published_workflow.user_id.split("|")[-1]
        return f"{user_id_part}/{published_workflow.name}"


def create_workflow(
    name: str,
    input: type[BaseModel],
    output: type[BaseModel],
) -> Workflow:
    return Workflow(
        name=name,
        input_schema=input.schema(),
        output_schema=output.schema(),
    )


def main() -> None:
    import cli_nested_json

    parser = ArgumentParser()
    parser.add_argument("workflow_file", type=str)
    args, input_params = parser.parse_known_args()

    with open(args.workflow_file) as stream:
        workflow = Workflow.from_json(json.load(stream))

    payload = cli_nested_json.interpret_nested_json(
        [part.split("=") for part in input_params]
    )
    console = rich.get_console()
    console.print(
        f"🤧 Loaded {workflow.name!r} with {len(workflow.nodes)} nodes!",
        style="bold magenta",
    )

    context = Context({INPUT_VARIABLE_NAME: payload})

    sorter = graphlib.TopologicalSorter(
        graph={
            node.id: node.depends - {INPUT_VARIABLE_NAME}
            for node in workflow.nodes.values()
        }
    )
    with console.status("Starting the execution", spinner="bouncingBall") as status:
        for n, node_id in enumerate(sorter.static_order()):
            node = workflow.nodes[node_id]
            status.update(
                status=f"Executing {node_id!r} ({n}/{len(workflow.nodes)})",
                spinner="runner",
            )
            if isinstance(node, Run):
                input = context.hydrate(node.input)
                assert isinstance(input, dict)

                owner, _, app = node.app.partition("/")
                app, sep, path = app.partition("/")

                handle = fal.apps.submit(
                    f"{owner}/{app}",
                    path=f"{sep}{path}",
                    arguments=input,
                )
                log_count = 0
                for event in handle.iter_events(logs=True):
                    if isinstance(event, fal.apps.Queued):
                        status.update(
                            status=(
                                "Queued for "
                                f"{node_id!r} "
                                f"(position={event.position}) "
                                f"({n}/{len(workflow.nodes)})",
                            ),
                            spinner="dots",
                        )
                    elif isinstance(event, fal.apps.InProgress):
                        status.update(
                            status=f"Executing {node_id!r} ({n}/{len(workflow.nodes)})",
                            spinner="runner",
                        )
                        for log in event.logs[log_count:]:  # type: ignore
                            console.log(log["message"], style="dim")
                            log_count += 1

                handle_status = handle.status(logs=True)
                assert isinstance(handle_status, fal.apps.Completed)
                for log in handle_status.logs[log_count:]:  # type: ignore
                    console.log(log["message"], style="dim")

                context.vars[node_id] = handle.get()
            else:
                context.vars[node_id] = node.execute(context)

        console.print(
            "🎉 Execution complete!",
            style="bold green",
        )
        output = context.hydrate(workflow.output)
        console.print(Syntax(json.dumps(output, indent=2), "json"))


if __name__ == "__main__":
    main()
