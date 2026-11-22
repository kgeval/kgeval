"""Load and check test questions: the questions a graph is expected to answer."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import NamedTuple

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from kgeval._yaml import load_yaml
from kgeval.graph.model import content_digest
from kgeval.graph.view import GraphView
from kgeval.reference import Name, NamedFact, Reference

__all__ = ["Question", "QuestionSet", "Support", "load_questions"]


class Question(BaseModel):
    """Pair a question with the facts and entities that a correct answer rests on.

    Attributes:
        text: The question in natural language.
        supporting_facts: The facts the answer needs, each as ``[source entity name,
            relationship type, target entity name]``.
        supporting_entities: The entities the answer needs beyond those of its facts,
            such as the entity whose property a question asks for.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: Name
    supporting_facts: tuple[NamedFact, ...] = ()
    supporting_entities: tuple[Name, ...] = ()

    @model_validator(mode="after")
    def _require_support(self) -> Question:
        if not (self.supporting_facts or self.supporting_entities):
            raise ValueError(f"{self.text!r} needs a supporting fact or entity")
        return self


class Support(NamedTuple):
    """Name one supporting fact or entity of a question, with the graph elements that match it.

    Attributes:
        label: The supporting fact or entity as the question set spells it.
        is_fact: Whether it is a fact rather than an entity.
        node_ids: The matching entities, or the source and target entities of the
            matching facts.
        relationship_ids: The matching facts; empty for an entity.
        chunk_ids: The chunks that ground the matching entities or facts, as
            ``GraphView.chunk_ids`` and ``GraphView.fact_chunk_ids`` return them.
    """

    label: str
    is_fact: bool
    node_ids: frozenset[str]
    relationship_ids: frozenset[str]
    chunk_ids: frozenset[str]

    @property
    def element_ids(self) -> frozenset[str]:
        """Return the elements that hold it: the matching facts, or the matching entities."""
        return self.relationship_ids if self.is_fact else self.node_ids


class QuestionSet(Reference):
    """Hold test questions and the variants under which their supporting facts match.

    Supporting facts and entities match the graph's as ``Reference`` describes.
    Validation also accepts a bare list of questions, the format of question sets that
    need none of the other fields.

    Attributes:
        questions: The questions, each with a distinct text.
    """

    questions: tuple[Question, ...] = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def _from_questions(cls, data: object) -> object:
        # Accept a bare list of questions, which needs none of the other fields.
        return {"questions": data} if isinstance(data, list | tuple) else data

    @model_validator(mode="after")
    def _require_distinct_texts(self) -> QuestionSet:
        # Keep texts distinct: retrievals are keyed by question text.
        counts = Counter(question.text for question in self.questions)
        if repeated := sorted(text for text, count in counts.items() if count > 1):
            raise ValueError(f"questions must differ, repeated: {', '.join(map(repr, repeated))}")
        return self

    def named_facts(self) -> Iterable[NamedFact]:
        """Return the supporting facts of every question.

        Returns:
            The facts.
        """
        return [fact for question in self.questions for fact in question.supporting_facts]

    def named_entities(self) -> Iterable[str]:
        """Return the supporting entities of every question, its supporting facts' included.

        Returns:
            The entity names.
        """
        names = [name for source, _, target in self.named_facts() for name in (source, target)]
        return names + [
            name for question in self.questions for name in question.supporting_entities
        ]

    def support(self, graph: GraphView) -> list[list[Support]]:
        """Match every question's supporting facts and entities with the graph's.

        Args:
            graph: The graph to match against.

        Returns:
            One list per question, in order: its supporting facts, then its supporting
            entities, each with the graph elements that match it, if any.
        """
        entities: defaultdict[str, set[str]] = defaultdict(set)
        entity_chunks: defaultdict[str, set[str]] = defaultdict(set)
        for entity in graph.entities:
            if name_key := self.entity_key(graph.name(entity) or ""):
                entities[name_key].add(entity.id)
                entity_chunks[name_key].update(graph.chunk_ids(entity.id))
        facts: defaultdict[tuple[str, str, str], set[str]] = defaultdict(set)
        endpoints: defaultdict[tuple[str, str, str], set[str]] = defaultdict(set)
        fact_chunks: defaultdict[tuple[str, str, str], set[str]] = defaultdict(set)
        for fact in graph.facts:
            fact_key = self.graph_fact_key(graph, fact)
            if all(fact_key):
                facts[fact_key].add(fact.id)
                endpoints[fact_key].update(fact.node_ids)
                fact_chunks[fact_key].update(graph.fact_chunk_ids(fact))
        supports: list[list[Support]] = []
        for question in self.questions:
            support: list[Support] = []
            for source, relationship_type, target in question.supporting_facts:
                fact_key = self.fact_key(source, relationship_type, target)
                support.append(
                    Support(
                        f'"{source}" -[:{relationship_type}]-> "{target}"',
                        True,
                        frozenset(endpoints.get(fact_key, ())),
                        frozenset(facts.get(fact_key, ())),
                        frozenset(fact_chunks.get(fact_key, ())),
                    )
                )
            for name in question.supporting_entities:
                name_key = self.entity_key(name)
                support.append(
                    Support(
                        f'"{name}"',
                        False,
                        frozenset(entities.get(name_key, ())),
                        frozenset(),
                        frozenset(entity_chunks.get(name_key, ())),
                    )
                )
            supports.append(support)
        return supports

    def digest(self) -> str:
        """Return a short digest that changes whenever the question set changes.

        Returns:
            The first 12 hex digits of the question set's content digest.
        """
        return content_digest(self.model_dump(mode="json"))[:12]


def load_questions(path: str | Path) -> QuestionSet:
    """Load test questions from a YAML or JSON file.

    The file holds either a list of questions or a mapping with the fields of
    ``QuestionSet``. Each question has a ``text`` and at least one supporting fact or
    entity.

    Args:
        path: The questions file.

    Returns:
        The question set.

    Raises:
        InputError: If the file cannot be read or holds no valid question set.
    """
    return load_yaml(path, TypeAdapter(QuestionSet))
