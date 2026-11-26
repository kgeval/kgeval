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
    """Read a YAML file, which may also be JSON, and validate its content.

    Args:
        path: The file to read, in UTF-8, or in UTF-16 with a byte order mark.
        adapter: Validates the parsed content.

    Returns:
        The validated content.

    Raises:
        InputError: If the file cannot be read, parsed, or validated.
    """
    try:
        content = yaml.load(Path(path).read_bytes(), Loader=_Loader)
        return adapter.validate_python(content)
    except (OSError, yaml.YAMLError, ValidationError) as error:
        raise InputError(f"invalid file {path}: {error}") from error


def load_json(path: str | Path, adapter: TypeAdapter[_T]) -> _T:
    """Read a JSON file in any encoding that JSON allows, and validate its content.

    Args:
        path: The file to read, in UTF-8, UTF-16, or UTF-32.
        adapter: Validates the parsed content.

    Returns:
        The validated content.

    Raises:
        InputError: If the file cannot be read, decoded, parsed, or validated, or
            repeats a key.
    """
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
