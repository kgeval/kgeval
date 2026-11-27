"""Implement the ground truth that lists the facts a graph is expected to contain."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Annotated, TypeAlias

from pydantic import Field, TypeAdapter, model_validator

from kgeval._files import load_yaml
from kgeval.reference import NamedFact, Reference

__all__ = ["DocumentSelector", "GroundTruth", "load_ground_truth"]

DocumentSelector: TypeAlias = Annotated[dict[str, str], Field(min_length=1)]
"""Select the lexical nodes that have every listed property with the listed value."""


class GroundTruth(Reference):
    """Hold the facts a graph should contain, and the variants under which they match.

    Facts match as ``Reference`` describes. Validation also accepts a bare list of
    ground truth facts, the format of ground truths that need none of the other fields.

    Attributes:
        documents: Selectors of the lexical nodes of the documents the ground truth
            covers. When set, the ground truth lists every fact these documents state,
            and the ground truth metrics compare only the facts and entities grounded in
            the documents' chunks; when empty, it covers every chunk.
        facts: The ground truth facts.
    """

    documents: tuple[DocumentSelector, ...] = ()
    facts: tuple[NamedFact, ...] = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def _from_facts(cls, data: object) -> object:
        # Accept a bare list of ground truth facts, which needs none of the other fields.
        return {"facts": data} if isinstance(data, list | tuple) else data

    def named_facts(self) -> Iterable[NamedFact]:
        """Return the ground truth facts.

        Returns:
            The facts.
        """
        return self.facts

    def named_entities(self) -> Iterable[str]:
        """Return the source and target names of every ground truth fact.

        Returns:
            The entity names.
        """
        return [name for source, _, target in self.facts for name in (source, target)]


def load_ground_truth(path: str | Path) -> GroundTruth:
    """Load a ground truth from a YAML or JSON file.

    The file holds either a list of ground truth facts or a mapping with the fields of
    ``GroundTruth``. Each ground truth fact is a ``[source entity name, relationship
    type, target entity name]`` list, such as ``[Lee Kuan Yew, BORN_IN, Singapore]``.

    Args:
        path: The ground truth file.

    Returns:
        The ground truth.

    Raises:
        InputError: If the file cannot be read or holds no valid ground truth, such as
            one without facts, with a part of a ground truth fact that has no letter or
            digit, or with an alias that conflicts with a ground truth name.
    """
    return load_yaml(path, TypeAdapter(GroundTruth))
