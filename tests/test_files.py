"""Test the reader that loads YAML and JSON files into validated models."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from pydantic import TypeAdapter

from kgeval._files import load_json, load_yaml
from kgeval.errors import InputError


@pytest.mark.parametrize("load", [load_yaml, load_json])
@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig", "utf-16"])
def test_files_load_in_the_encodings_that_editors_and_shells_write(
    tmp_path: Path, load: Callable[[Path, TypeAdapter[Any]], Any], encoding: str
) -> None:
    path = tmp_path / "file.json"
    path.write_text('{"name": "Lee Kuan Yew"}', encoding=encoding)

    assert load(path, TypeAdapter(dict[str, str])) == {"name": "Lee Kuan Yew"}


@pytest.mark.parametrize("load", [load_yaml, load_json])
def test_files_reject_repeated_keys(
    tmp_path: Path, load: Callable[[Path, TypeAdapter[Any]], Any]
) -> None:
    path = tmp_path / "file.json"
    path.write_text('{"name": "Lee Kuan Yew", "name": "Lee"}', encoding="utf-8")

    with pytest.raises(InputError, match="repeated keys: name"):
        load(path, TypeAdapter(dict[str, str]))


def test_yaml_files_reject_keys_that_cannot_be_dictionary_keys(tmp_path: Path) -> None:
    path = tmp_path / "file.yaml"
    path.write_text("? [Lee, Kuan Yew]\n: 1\n", encoding="utf-8")

    with pytest.raises(InputError, match="found unhashable key"):
        load_yaml(path, TypeAdapter(dict[str, int]))
