"""Define the base class of ground truths and test questions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Annotated, TypeAlias

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from kgeval.graph.model import Relationship, content_digest
from kgeval.graph.names import normalize_name, normalize_type
from kgeval.graph.view import GraphView

__all__ = ["Name", "NamedFact", "Reference"]


def _require_alphanumeric(text: str) -> str:
    if not any(character.isalnum() for character in text):
        raise ValueError("needs a letter or digit")
    return text


Name: TypeAlias = Annotated[str, AfterValidator(_require_alphanumeric)]
"""Hold an entity name or relationship type of a reference, which needs a letter or digit."""

NamedFact: TypeAlias = tuple[Name, Name, Name]
"""Name a fact by its source entity name, relationship type, and target entity name.

It has the shape of a schema pattern, with entity names in place of node types.
"""


class Reference(BaseModel, ABC):
    """Hold the variants under which a reference's entities and facts match a graph's.

    Entity names match under ``normalize_name`` and relationship types under
    ``normalize_type``. Aliases add the other names and relationship types the graph
    may use for an entity or relationship type of the reference; an alias may not be a
    name of the reference itself, nor stand for two.

    Attributes:
        aliases: Maps an entity name of the reference to other names the graph may give
            the entity.
        relationship_type_aliases: Maps a relationship type of the reference to other
            relationship types the graph may use for it.
        symmetric_relationship_types: Relationship types whose facts match in either
            direction, such as ``MARRIED``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    aliases: dict[Name, tuple[Name, ...]] = Field(default_factory=dict)
    relationship_type_aliases: dict[Name, tuple[Name, ...]] = Field(default_factory=dict)
    symmetric_relationship_types: tuple[Name, ...] = ()

    _name_keys: dict[str, str] = PrivateAttr(default_factory=dict)
    _type_keys: dict[str, str] = PrivateAttr(default_factory=dict)
    _symmetric_keys: frozenset[str] = PrivateAttr(default=frozenset())

    @abstractmethod
    def named_facts(self) -> Iterable[NamedFact]:
        """Return every fact the reference names.

        Returns:
            The facts.
        """

    @abstractmethod
    def named_entities(self) -> Iterable[str]:
        """Return every entity name the reference uses, its facts' included.

        Returns:
            The names.
        """

    @model_validator(mode="after")
    def _index_variants(self) -> Reference:
        names = {normalize_name(name) for name in self.named_entities()}
        types = {normalize_type(fact[1]) for fact in self.named_facts()}
        self._name_keys = _alias_keys(self.aliases, names, normalize_name, "entity name")
        self._type_keys = _alias_keys(
            self.relationship_type_aliases, types, normalize_type, "relationship type"
        )
        for name in self.symmetric_relationship_types:
            if normalize_type(name) not in types:
                raise ValueError(f"no fact has the symmetric relationship type {name!r}")
        self._symmetric_keys = frozenset(map(normalize_type, self.symmetric_relationship_types))
        return self

    def digest(self) -> str:
        """Return a short digest that changes whenever the reference changes.

        Returns:
            The first 12 hex digits of the reference's content digest.
        """
        return content_digest(self.model_dump(mode="json"))[:12]

    def entity_key(self, name: str) -> str:
        """Return the key under which an entity name matches the reference's names.

        Args:
            name: An entity name, from the graph or the reference.

        Returns:
            The key ``normalize_name`` returns, or the key of the name it is an
            alias of.
        """
        key = normalize_name(name)
        return self._name_keys.get(key, key)

    def fact_key(self, source: str, relationship_type: str, target: str) -> tuple[str, str, str]:
        """Return the key under which a fact matches the reference's facts.

        Args:
            source: The name of its source entity.
            relationship_type: Its relationship type.
            target: The name of its target entity.

        Returns:
            The keys of its source name, relationship type, and target name, the names
            in sorted order when the relationship type is symmetric.
        """
        type_key = normalize_type(relationship_type)
        type_key = self._type_keys.get(type_key, type_key)
        source_key, target_key = self.entity_key(source), self.entity_key(target)
        if type_key in self._symmetric_keys:
            source_key, target_key = sorted((source_key, target_key))
        return source_key, type_key, target_key

    def graph_fact_key(self, graph: GraphView, fact: Relationship) -> tuple[str, str, str]:
        """Return the key of a fact of the graph, read through the graph's layout.

        Args:
            graph: The graph the fact belongs to.
            fact: The fact.

        Returns:
            The fact's key, as ``fact_key`` returns it; a part is empty when the fact's
            entity has no name, so the fact cannot match.
        """
        source = graph.name(graph.node(fact.source_id)) or ""
        target = graph.name(graph.node(fact.target_id)) or ""
        return self.fact_key(source, graph.relationship_type(fact), target)


def _alias_keys(
    aliases: Mapping[str, Sequence[str]],
    keys: set[str],
    normalize: Callable[[str], str],
    noun: str,
) -> dict[str, str]:
    """Map the key of every alias to the key of the name it stands for."""
    alias_keys: dict[str, str] = {}
    for name, others in aliases.items():
        target = normalize(name)
        if target not in keys:
            raise ValueError(f"aliases are given for {name!r}, not among the reference's {noun}s")
        for other in others:
            key = normalize(other)
            if key in keys and key != target:
                raise ValueError(f"alias {other!r} of {name!r} is itself a {noun} of the reference")
            if alias_keys.setdefault(key, target) != target:
                raise ValueError(f"alias {other!r} stands for two {noun}s")
    return alias_keys
