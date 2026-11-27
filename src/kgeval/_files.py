"""Implement the reader that loads YAML and JSON files into validated models."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Hashable
from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import TypeAdapter, ValidationError

from kgeval.errors import InputError

__all__ = ["load_json", "load_yaml"]

_T = TypeVar("_T")


class _Loader(yaml.SafeLoader):
    """Load YAML as ``yaml.safe_load`` does, but reject repeated keys, which it drops."""

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
        keys = [self.construct_object(key, deep=deep) for key, _ in node.value]
        if repeated := _repeated(keys):
            raise yaml.constructor.ConstructorError(
                problem=f"repeated keys: {repeated}", problem_mark=node.start_mark
            )
        return super().construct_mapping(node, deep=deep)


def load_yaml(path: str | Path, adapter: TypeAdapter[_T]) -> _T:
    """Read and validate a YAML file, or JSON when its name ends in .json, raising InputError."""
    # Read JSON as JSON, since YAML 1.1 parses some of it differently, such as escaped emoji.
    if Path(path).suffix.lower() == ".json":
        return load_json(path, adapter)
    try:
        content = yaml.load(Path(path).read_bytes(), Loader=_Loader)
        return adapter.validate_python(content)
    except (OSError, yaml.YAMLError, ValidationError) as error:
        raise InputError(f"invalid file {path}: {error}") from error


def load_json(path: str | Path, adapter: TypeAdapter[_T]) -> _T:
    """Read and validate a JSON file in any encoding JSON allows, raising InputError."""
    try:
        # Pass bytes, so that json detects their encoding, such as the UTF-16 of PowerShell.
        content = json.loads(Path(path).read_bytes(), object_pairs_hook=_unique)
        return adapter.validate_python(content)
    # Catch decoding, parsing, and validation errors, which are all ValueErrors.
    except (OSError, ValueError) as error:
        raise InputError(f"invalid file {path}: {error}") from error


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Return a JSON object's pairs as a dict, rejecting repeated keys, which json drops."""
    if repeated := _repeated([key for key, _ in pairs]):
        raise ValueError(f"repeated keys: {repeated}")
    return dict(pairs)


def _repeated(keys: list[Any]) -> str:
    """Return the keys that occur more than once, sorted and joined by commas."""
    # Count only hashable keys, since the YAML loader rejects the others itself.
    counts = Counter(key for key in keys if isinstance(key, Hashable))
    return ", ".join(sorted(str(key) for key, count in counts.items() if count > 1))
