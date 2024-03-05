from __future__ import annotations

import ast
import graphlib
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterator, Union, cast

import fal

JSONType = Union[dict[str, Any], list[Any], str, int, float, bool, None]
VARIABLE_PREFIX = "$"
INPUT_VARIABLE_NAME = "input"


def parse_leaf(raw_leaf: str) -> Leaf:
    raw_leaf = raw_leaf.removeprefix(VARIABLE_PREFIX)
    leaf_tree = ast.parse(raw_leaf, mode="eval").body

    def parse_node(node: ast.AST) -> Leaf:
        if isinstance(node, ast.Name):
            return ReferenceLeaf(node.id)
        elif isinstance(node, ast.Attribute):
            return AttributeLeaf(parse_node(node.value), node.attr)
        elif isinstance(node, ast.Subscript):
            return IndexLeaf(parse_node(node.value), ast.literal_eval(node.slice))
        else:
            raise ValueError(f"Invalid leaf: {raw_leaf}")

    return parse_node(leaf_tree)


def export_workflow_json(data: dict[str, Any]) -> dict[str, Any]:
    if isinstance(data, dict):
        return {k: export_workflow_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [export_workflow_json(v) for v in data]
    elif isinstance(data, Leaf):
        return repr(data)
    else:
        return data


def import_workflow_json(data: dict[str, Any]) -> dict[str, Any]:
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
    elif isinstance(data, dict):
        for item in data:
            yield from iter_leaves(item)
    else:
        yield data


@dataclass
class Context:
    vars: dict[str, JSONType]

    def hydrate(self, input: JSONType) -> JSONType:
        if isinstance(input, dict):
            return {k: self.hydrate(v) for k, v in input.items()}
        elif isinstance(input, dict):
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
        return f"{self.leaf!r}[{self.index}]"

    @property
    def referee(self) -> ReferenceLeaf:
        return self.leaf.referee


@dataclass
class ReferenceLeaf(Leaf):
    name: str

    def execute(self, context: Context) -> JSONType:
        return context.vars[self.name]

    def __repr__(self) -> str:
        return VARIABLE_PREFIX + self.name

    @property
    def referee(self) -> ReferenceLeaf:
        return self


@dataclass
class Node:
    name: str
    app: str
    input: JSONType

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Node:
        return cls(
            name=data["name"],
            app=data["app"],
            input=import_workflow_json(data["input"]),
        )

    def execute(self, context: Context) -> JSONType:
        input = context.hydrate(self.input)
        assert isinstance(input, dict)
        return cast(JSONType, fal.apps.run(self.app, input))

    @property
    def requires(self) -> set[str]:
        return {
            leaf.referee.name  # type: ignore
            for leaf in iter_leaves(self.input)
            if isinstance(leaf, Leaf)
        }

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "app": self.app,
            "input": export_workflow_json(self.input),  # type: ignore
        }


@dataclass
class Workflow:
    nodes: dict[str, Node] = field(default_factory=dict)
    output: dict[str, Any] | None = None
    _app_counter: Counter = field(default_factory=Counter)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Workflow:
        data = import_workflow_json(data)
        return cls(
            nodes={
                node_name: Node.from_json(node_data)
                for node_name, node_data in data["nodes"].items()
            },
            output=data["output"],
        )

    def __post_init__(self) -> None:
        for node in self.nodes.values():
            self._app_counter[node.app] += 1

    def _generate_node_name(self, app: str) -> str:
        self._app_counter[app] += 1
        return f"{app.replace('/', '_').replace('-', '_')}_{self._app_counter[app]}"

    def call(self, app: str, input: JSONType) -> ReferenceLeaf:
        node_name = self._generate_node_name(app)
        node = self.nodes[node_name] = Node(node_name, app, input)
        return ReferenceLeaf(node.name)

    def set_output(self, output: JSONType) -> None:
        self.output = output  # type: ignore

    def execute(self, input: JSONType) -> JSONType:
        if not self.output:
            raise ValueError("Can't execute the workflow before the output is set.")

        context = Context({INPUT_VARIABLE_NAME: input})

        graph = graphlib.TopologicalSorter(
            graph={
                node.name: node.requires - {INPUT_VARIABLE_NAME}
                for node in self.nodes.values()
            }
        )
        for node_name in graph.static_order():
            print("Running: ", node_name)
            node = self.nodes[node_name]
            context.vars[node_name] = node.execute(context)

        return context.hydrate(self.output)

    @property
    def input(self) -> ReferenceLeaf:
        return ReferenceLeaf(INPUT_VARIABLE_NAME)

    def to_json(self) -> dict[str, JSONType]:
        if not self.output:
            raise ValueError("Can't serialize the workflow before the output is set.")

        return {
            "nodes": {node.name: node.to_json() for node in self.nodes.values()},
            "output": export_workflow_json(self.output),
        }
