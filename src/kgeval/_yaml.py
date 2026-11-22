"""Load YAML and JSON files into validated models."""

from __future__ import annotations

from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import TypeAdapter, ValidationError

from kgeval.errors import InputError

__all__ = ["load_yaml"]

_T = TypeVar("_T")


def load_yaml(path: str | Path, adapter: TypeAdapter[_T]) -> _T:
    """Read a YAML file, which may also be JSON, and validate its content.

    Args:
        path: The file to read.
        adapter: Validates the parsed content.

    Returns:
        The validated content.

    Raises:
        InputError: If the file cannot be read, parsed or validated.
    """
    try:
        return adapter.validate_python(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
    except (OSError, yaml.YAMLError, ValidationError) as error:
        raise InputError(f"invalid file {path}: {error}") from error
