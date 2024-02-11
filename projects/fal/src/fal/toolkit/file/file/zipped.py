from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Optional
from zipfile import ZipFile

from pydantic import PrivateAttr

from fal.toolkit.mainify import mainify

from .normal import File

__all__ = ["CompressedFile"]


@mainify
class CompressedFile(File):
    _extract_dir: Optional[TemporaryDirectory] = PrivateAttr(default=None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._extract_dir = None

    def __iter__(self):
        if not self._extract_dir:
            self._extract_files()

        files = Path(self._extract_dir.name).iterdir()  # type: ignore
        return iter(files)

    def _extract_files(self):
        self._extract_dir = TemporaryDirectory()

        with NamedTemporaryFile() as temp_file:
            file_path = temp_file.name
            self.save(file_path, overwrite=True)

            with ZipFile(file_path) as zip_file:
                zip_file.extractall(self._extract_dir.name)

    def glob(self, pattern: str):
        if not self._extract_dir:
            self._extract_files()

        return Path(self._extract_dir.name).glob(pattern)  # type: ignore

    def __del__(self):
        if self._extract_dir:
            self._extract_dir.cleanup()
