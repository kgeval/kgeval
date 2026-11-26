"""Test that graphs from well-known builders need no configuration."""

from dataclasses import dataclass
from datetime import UTC, datetime

import pytest

from kgeval.evaluation import evaluate
from kgeval.graph.layout import Layout
from kgeval.graph.model import Graph
from kgeval.graph.view import GraphView
from kgeval.metrics.base import Inputs
from kgeval.metrics.model.facts import FactAccuracy
from tests.doubles import ScriptedModel
from tests.factories import graph, node, relationship

TEXT = "Ada Lovelace worked with Charles Babbage on the Analytical Engine."
VALID_AT = datetime(2025, 1, 1, tzinfo=UTC)
NAMES_FROM_ID = ("name", "title", "id")
UNGROUNDED = frozenset({"entity_provenance", "fact_provenance", "extraction_coverage"})


@dataclass(frozen=True)
class Shape:
    """Hold a builder's graph with the layout and model prompt it should get."""

    graph: Graph
    layout: Layout
    names: frozenset[str]
    statement: str
    skipped: frozenset[str] = frozenset()


SHAPES = {
    "llm-graph-builder": Shape(
        graph(
            node("d", "Document", fileName="engine.pdf"),
            node("c", "Chunk", embeddings={"embedding": 384}, text=TEXT, fileName="engine.pdf"),
            relationship("part_of", "c", "PART_OF", "d"),
            relationship("first_chunk", "d", "FIRST_CHUNK", "c"),
            node("ada", "__Entity__", "Person", id="Ada Lovelace"),
            node("babbage", "__Entity__", "Person", id="Charles Babbage"),
            relationship("has_ada", "c", "HAS_ENTITY", "ada"),
            relationship("has_babbage", "c", "HAS_ENTITY", "babbage"),
            relationship("fact", "ada", "WORKED_WITH", "babbage"),
            node("k", "__Community__", id="0-0", level=0, title="Analytical Engine"),
            relationship("ada_in_k", "ada", "IN_COMMUNITY", "k"),
            relationship("babbage_in_k", "babbage", "IN_COMMUNITY", "k"),
        ),
        Layout(
            lexical_labels=frozenset({"Chunk", "Document"}),
            community_labels=frozenset({"__Community__"}),
            base_labels=frozenset({"__Entity__"}),
            name_keys=NAMES_FROM_ID,
        ),
        frozenset({"Ada Lovelace", "Charles Babbage"}),
        'Fact: "Ada Lovelace" (Person) -[:WORKED_WITH]-> "Charles Babbage" (Person)\n\n',
    ),
    "neo4j-graphrag": Shape(
        graph(
            node("d", "Document", "__KGBuilder__", path="engine.pdf"),
            node("c", "Chunk", "__KGBuilder__", embeddings={"embedding": 1536}, text=TEXT, index=0),
            relationship("from_document", "c", "FROM_DOCUMENT", "d"),
            node("ada", "__KGBuilder__", "__Entity__", "Person", name="Ada Lovelace"),
            node("babbage", "__KGBuilder__", "__Entity__", "Person", name="Charles Babbage"),
            relationship("ada_from_c", "ada", "FROM_CHUNK", "c"),
            relationship("babbage_from_c", "babbage", "FROM_CHUNK", "c"),
            relationship("fact", "ada", "WORKED_WITH", "babbage"),
        ),
        Layout(
            lexical_labels=frozenset({"Chunk", "Document"}),
            base_labels=frozenset({"__Entity__", "__KGBuilder__"}),
            name_keys=NAMES_FROM_ID,
        ),
        frozenset({"Ada Lovelace", "Charles Babbage"}),
        'Fact: "Ada Lovelace" (Person) -[:WORKED_WITH]-> "Charles Babbage" (Person)\n\n',
    ),
    "llamaindex": Shape(
        graph(
            node("c", "__Node__", "Chunk", embeddings={"embedding": 1536}, id="c1", text=TEXT),
            *(
                node(
                    person.split()[0].lower(),
                    "__Node__",
                    "__Entity__",
                    "PERSON",
                    embeddings={"embedding": 1536},
                    id=person,
                    name=person,
                    triplet_source_id="c1",
                )
                for person in ("Ada Lovelace", "Charles Babbage")
            ),
            relationship("mentions_ada", "c", "MENTIONS", "ada"),
            relationship("mentions_charles", "c", "MENTIONS", "charles"),
            relationship("fact", "ada", "WORKED_WITH", "charles"),
        ),
        Layout(
            lexical_labels=frozenset({"Chunk"}),
            base_labels=frozenset({"__Entity__", "__Node__"}),
            name_keys=NAMES_FROM_ID,
        ),
        frozenset({"Ada Lovelace", "Charles Babbage"}),
        'Fact: "Ada Lovelace" (PERSON) -[:WORKED_WITH]-> "Charles Babbage" (PERSON)\n\n',
    ),
    "langchain": Shape(
        graph(
            node("ada", "Person", id="Ada Lovelace"),
            node("babbage", "Person", id="Charles Babbage"),
            relationship("fact", "ada", "WORKED_WITH", "babbage"),
        ),
        Layout(name_keys=NAMES_FROM_ID),
        frozenset({"Ada Lovelace", "Charles Babbage"}),
        "",
        skipped=UNGROUNDED | {"embedding_consistency"},
    ),
    "langchain-with-sources": Shape(
        graph(
            node("d", "Document", id="9f2c", text=TEXT, source="engine.pdf"),
            node("ada", "__Entity__", "Person", id="Ada Lovelace"),
            node("babbage", "__Entity__", "Person", id="Charles Babbage"),
            relationship("mentions_ada", "d", "MENTIONS", "ada"),
            relationship("mentions_babbage", "d", "MENTIONS", "babbage"),
            relationship("fact", "ada", "WORKED_WITH", "babbage"),
        ),
        Layout(
            lexical_labels=frozenset({"Document"}),
            base_labels=frozenset({"__Entity__"}),
            name_keys=NAMES_FROM_ID,
        ),
        frozenset({"Ada Lovelace", "Charles Babbage"}),
        'Fact: "Ada Lovelace" (Person) -[:WORKED_WITH]-> "Charles Babbage" (Person)\n\n',
        skipped=frozenset({"embedding_consistency"}),
    ),
    "microsoft-graphrag": Shape(
        graph(
            node("d", "__Document__", id="doc-1", title="engine.txt"),
            node("c", "__Chunk__", id="unit-1", text=TEXT, n_tokens=14),
            relationship("part_of", "c", "PART_OF", "d"),
            *(
                node(
                    person.split()[0].lower(),
                    "__Entity__",
                    "Person",
                    embeddings={"description_embedding": 1536},
                    id=f"uuid-{index}",
                    name=person.upper(),
                    description=f"{person} was a mathematician.",
                    human_readable_id=index,
                )
                for index, person in enumerate(("Ada Lovelace", "Charles Babbage"))
            ),
            relationship("has_ada", "c", "HAS_ENTITY", "ada"),
            relationship("has_charles", "c", "HAS_ENTITY", "charles"),
            relationship(
                "fact",
                "ada",
                "RELATED",
                "charles",
                id="rel-1",
                description="Ada Lovelace worked with Charles Babbage on the Analytical Engine.",
                rank=2,
                weight=1.0,
                human_readable_id=0,
                text_unit_ids=["unit-1"],
            ),
            node("k", "__Community__", community="0", level=0, title="Community 0"),
            relationship("ada_in_k", "ada", "IN_COMMUNITY", "k"),
            relationship("charles_in_k", "charles", "IN_COMMUNITY", "k"),
        ),
        Layout(
            lexical_labels=frozenset({"__Chunk__", "__Document__"}),
            community_labels=frozenset({"__Community__"}),
            base_labels=frozenset({"__Entity__"}),
            name_keys=NAMES_FROM_ID,
        ),
        frozenset({"ADA LOVELACE", "CHARLES BABBAGE"}),
        'Fact: "ADA LOVELACE" (Person) -[:RELATED]-> "CHARLES BABBAGE" (Person)\n'
        "Description: Ada Lovelace worked with Charles Babbage on the Analytical Engine.\n\n",
    ),
    "graphiti": Shape(
        graph(
            node(
                "e",
                "Episodic",
                uuid="episode-1",
                name="engine notes",
                content=TEXT,
                source="text",
                group_id="g",
                valid_at=VALID_AT,
                entity_edges=["edge-1"],
            ),
            *(
                node(
                    person.split()[0].lower(),
                    "Entity",
                    "Person",
                    embeddings={"name_embedding": 1024},
                    uuid=f"entity-{index}",
                    name=person,
                    summary="A mathematician.",
                    group_id="g",
                )
                for index, person in enumerate(("Ada Lovelace", "Charles Babbage"))
            ),
            relationship("mentions_ada", "e", "MENTIONS", "ada", uuid="m1", group_id="g"),
            relationship("mentions_charles", "e", "MENTIONS", "charles", uuid="m2", group_id="g"),
            relationship(
                "fact",
                "ada",
                "RELATES_TO",
                "charles",
                embeddings={"fact_embedding": 1024},
                uuid="edge-1",
                name="WORKED_WITH",
                fact="Ada Lovelace worked with Charles Babbage.",
                episodes=["episode-1"],
                valid_at=VALID_AT,
                group_id="g",
            ),
            node("k", "Community", embeddings={"name_embedding": 1024}, name="Analytical Engine"),
            relationship("k_has_ada", "k", "HAS_MEMBER", "ada", uuid="h1", group_id="g"),
            relationship("k_has_charles", "k", "HAS_MEMBER", "charles", uuid="h2", group_id="g"),
            node("s", "Saga", uuid="saga-1", name="notes", group_id="g"),
            relationship("has_episode", "s", "HAS_EPISODE", "e", uuid="h3", group_id="g"),
        ),
        Layout(
            lexical_labels=frozenset({"Episodic"}),
            community_labels=frozenset({"Community"}),
            base_labels=frozenset({"Entity"}),
            type_keys=("name",),
            expired_keys=("expired_at",),
        ),
        frozenset({"Ada Lovelace", "Charles Babbage"}),
        'Fact: "Ada Lovelace" (Person) -[:WORKED_WITH]-> "Charles Babbage" (Person)\n'
        "Description: Ada Lovelace worked with Charles Babbage.\n\n",
    ),
}


@pytest.fixture(params=list(SHAPES), ids=list(SHAPES))
def shape(request: pytest.FixtureRequest) -> Shape:
    return SHAPES[request.param]


def test_detect_reads_each_builders_shape(shape: Shape) -> None:
    view = GraphView(shape.graph)

    assert view.layout == shape.layout
    assert {view.name(entity) for entity in view.entities} == shape.names
    assert len(view.facts) == 1


def test_every_intrinsic_metric_passes_or_skips_for_the_builders_shape(shape: Shape) -> None:
    report = evaluate(shape.graph)

    assert {result.metric for result in report.results if result.score is None} == shape.skipped
    assert all(result.score == 1.0 for result in report.results if result.score is not None)


def test_model_sees_the_relationship_type_and_description_of_the_builders_facts(
    shape: Shape,
) -> None:
    model = ScriptedModel()

    result = FactAccuracy().evaluate(GraphView(shape.graph), Inputs(model=model))

    if shape.statement:
        assert (result.evaluated, result.failed) == (1, 0)
        assert shape.statement in model.prompts[0]
    else:
        assert result.skip_reason == (
            "no fact between entities with a name is grounded in a chunk with text"
        )
