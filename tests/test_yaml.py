"""Test the reader that loads YAML and JSON files into validated models."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from pydantic import TypeAdapter

from kgeval._yaml import load_json, load_yaml


@pytest.mark.parametrize("load", [load_yaml, load_json])
@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig", "utf-16"])
def test_files_load_in_the_encodings_that_editors_and_shells_write(
    tmp_path: Path, load: Callable[[Path, TypeAdapter[Any]], Any], encoding: str
) -> None:
    path = tmp_path / "file.json"
    path.write_text('{"name": "Lee Kuan Yew"}', encoding=encoding)

    assert load(path, TypeAdapter(dict[str, str])) == {"name": "Lee Kuan Yew"}
