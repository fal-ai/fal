from __future__ import annotations

import ast
import shutil
import subprocess
import sys
import tempfile
from argparse import ArgumentParser
from pathlib import Path

from refactor import Rule, Session, actions

PROJECT_ROOT = (Path(__file__).parent.parent).resolve()


def _to_qualified_name(file: Path, project_root: Path) -> str:
    """Convert the given path into an importable module name.

    src/isolate/server/definitions with src/ would become isolate.server.definitions.
    """

    relative_path = file.relative_to(project_root)
    return ".".join(relative_path.parent.parts + (relative_path.stem,))


def _as_relative_path(file: Path) -> Path:
    for parent_project in PROJECT_ROOT.iterdir():
        try:
            return file.resolve().relative_to(parent_project / "src")
        except ValueError:
            continue  # Not a subpath
    else:
        raise ValueError(f"Could not find relative path for {file}")


class FixGRPCImports(Rule):
    """Change all unqualified imports to qualified imports."""

    def match(self, node: ast.AST) -> actions.Replace:
        # import *_pb2
        assert isinstance(node, ast.Import)
        assert len(node.names) == 1
        assert not node.names[0].name.startswith("google")
        assert node.names[0].name.endswith("_pb2")

        # If we know where the import is coming from, use that.
        qualified_name = self.context.config.known_imports.get(
            node.names[0].name[: -len("_pb2")]
        )
        if not qualified_name:
            # Otherwise discover it from the current file path.
            parent_dir = _as_relative_path(self.context.file).parent
            qualified_name = ".".join(parent_dir.parts) or "."

        # Change import *_pb2 to from <qualified_name> import *_pb2
        return actions.Replace(
            node,
            ast.ImportFrom(module=qualified_name, names=node.names, level=0),
        )


def regen_grpc(
    file: Path, link_paths: list[Path], known_imports: dict[str, str]
) -> None:
    assert file.exists()

    parent_dir = file.parent

    link_commands = [f"-I={link_path.resolve()}" for link_path in link_paths]

    subprocess.check_output(
        [
            sys.executable,
            "-m",
            "grpc_tools.protoc",
            "--proto_path=.",
            "--python_out=.",
            "--grpc_python_out=.",
            "--mypy_out=.",
            *link_commands,
            file.name,
        ],
        cwd=parent_dir,
    )

    # Python gRPC compiler is bad at using the proper import
    # notation so it doesn't work with our package structure.
    #
    # See: https://github.com/protocolbuffers/protobuf/issues/1491

    # For fixing this we are going to manually correct the generated
    # source.
    for grpc_output_file in parent_dir.glob("*_pb2*.py*"):
        session = Session(rules=[FixGRPCImports])
        session.config.known_imports = known_imports
        changes = session.run_file(grpc_output_file)
        if changes:
            changes.apply_diff()


def sync_isolate(isolate_version: str, cwd: Path) -> set[str]:
    # Clone isolate into a temporary directory with
    # the specified git revision.

    target_repo_path = cwd / "_isolate_git"
    src_dir = target_repo_path / "src"
    subprocess.check_call(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--branch",
            f"v{isolate_version}",
            "https://github.com/fal-ai/isolate",
            target_repo_path,
        ],
        cwd=cwd,
    )

    known_imports = {}
    for proto_file in target_repo_path.rglob("*.proto"):
        shutil.copy(proto_file, cwd)
        known_imports[proto_file.stem] = _to_qualified_name(proto_file.parent, src_dir)
    return known_imports


# Use Python3.9 and pip install mypy-protobuf, grpcio-tools and refactor
# first.
def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("definition_file", type=Path)
    parser.add_argument("isolate_version", type=str, nargs="?")

    options = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        if options.isolate_version:
            isolate_imports = sync_isolate(options.isolate_version, cwd=tmp_path)
        else:
            isolate_imports = {}
            print("No isolate version specified, no imports passed")

        regen_grpc(
            options.definition_file,
            link_paths=[tmp_path],
            known_imports=isolate_imports,
        )


if __name__ == "__main__":
    main()
