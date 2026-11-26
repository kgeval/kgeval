"""Test the reader that loads YAML and JSON files into validated models."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from pydantic import TypeAdapter

from kgeval._files import load_json, load_yaml
from kgeval.errors import InputError


@pytest.mark.parametrize(("load", "name"), [(load_yaml, "file.yaml"), (load_json, "file.json")])
@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig", "utf-16"])
def test_files_load_in_the_encodings_that_editors_and_shells_write(
    tmp_path: Path, load: Callable[[Path, TypeAdapter[Any]], Any], name: str, encoding: str
) -> None:
    path = tmp_path / name
    path.write_text('{"name": "Lee Kuan Yew"}', encoding=encoding)

    assert load(path, TypeAdapter(dict[str, str])) == {"name": "Lee Kuan Yew"}


@pytest.mark.parametrize(("load", "name"), [(load_yaml, "file.yaml"), (load_json, "file.json")])
def test_files_reject_repeated_keys(
    tmp_path: Path, load: Callable[[Path, TypeAdapter[Any]], Any], name: str
) -> None:
    path = tmp_path / name
    path.write_text('{"name": "Lee Kuan Yew", "name": "Lee"}', encoding="utf-8")

    with pytest.raises(InputError, match="repeated keys: name"):
        load(path, TypeAdapter(dict[str, str]))


def test_yaml_files_reject_keys_that_cannot_be_dictionary_keys(tmp_path: Path) -> None:
    path = tmp_path / "file.yaml"
    path.write_text("? [Lee, Kuan Yew]\n: 1\n", encoding="utf-8")

    with pytest.raises(InputError, match="found unhashable key"):
        load_yaml(path, TypeAdapter(dict[str, int]))


def test_files_whose_names_end_in_json_are_read_as_json(tmp_path: Path) -> None:
    path = tmp_path / "file.json"
    # Write as json.dump does, escaping the emoji and writing 1e+16, which YAML 1.1 misreads.
    path.write_text(json.dumps({"name": "Lee 😀", "born": 1e16}), encoding="utf-8")

    assert load_yaml(path, TypeAdapter(dict[str, Any])) == {"name": "Lee 😀", "born": 1e16}
