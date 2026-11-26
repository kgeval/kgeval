"""Test the property type consistency and embedding consistency metrics."""

from kgeval.metrics.base import Inputs
from kgeval.metrics.intrinsic.properties import EmbeddingConsistency, PropertyTypeConsistency
from tests.factories import example_view, node, relationship, view


def test_values_of_a_minority_type_fail_and_only_entities_and_facts_count() -> None:
    result = PropertyTypeConsistency().evaluate(example_view(), Inputs())

    assert (result.evaluated, result.failed) == (13, 1)
    [finding] = result.findings
    assert finding.node_ids == ("kwa",)
    assert finding.message == (
        '"Kwa Geok Choo" (Person): property born is STRING, while other Person entities hold DATE'
    )


def test_ties_go_to_the_alphabetically_first_type() -> None:
    result = PropertyTypeConsistency().evaluate(
        view(node("a", "Movie", released=1999), node("b", "Movie", released="1999")), Inputs()
    )

    assert [finding.node_ids for finding in result.findings] == [("b",)]


def test_each_node_type_and_key_is_its_own_group() -> None:
    result = PropertyTypeConsistency().evaluate(
        view(node("a", "Person", code=1), node("b", "Company", code="X1")), Inputs()
    )

    assert (result.evaluated, result.failed) == (2, 0)


def test_fact_properties_are_grouped_by_relationship_type() -> None:
    result = PropertyTypeConsistency().evaluate(
        view(
            node("a", name="Ada"),
            node("b", name="Bob"),
            relationship("r1", "a", "WORKS_AT", "b", since=1990),
            relationship("r2", "a", "WORKS_AT", "b", since="1990"),
            relationship("r3", "a", "WORKS_AT", "b", since=2001),
        ),
        Inputs(),
    )

    [finding] = result.findings
    assert (finding.node_ids, finding.relationship_ids) == (("a", "b"), ("r2",))
    assert finding.message == (
        '"Ada" -[:WORKS_AT]-> "Bob": property since is STRING, '
        "while other WORKS_AT facts hold INTEGER"
    )


def test_every_value_of_another_type_is_its_own_finding() -> None:
    result = PropertyTypeConsistency().evaluate(
        view(
            *(node(f"n{number}", "Movie", year=number) for number in range(3)),
            node("s1", "Movie", year="1"),
            node("s2", "Movie", year=1.5),
        ),
        Inputs(),
    )

    assert [finding.node_ids for finding in result.findings] == [("s2",), ("s1",)]


def test_embedding_consistency_reports_chunks_missing_their_groups_embedding() -> None:
    result = EmbeddingConsistency().evaluate(example_view(), Inputs())

    assert (result.unit, result.evaluated, result.failed) == ("expected embeddings", 5, 1)
    [finding] = result.findings
    assert finding.node_ids == ("chunk3",)
    assert finding.message == (
        '"Table of contents" (Chunk): no embedding under property embedding, '
        "while other Chunk nodes have one"
    )


def test_embedding_consistency_expects_the_dimension_most_of_the_group_has() -> None:
    graph = view(
        node("c1", "Chunk", embeddings={"embedding": 768}, text="Small model"),
        node("c2", "Chunk", embeddings={"embedding": 1536}, text="Large model"),
        node("c3", "Chunk", embeddings={"embedding": 1536}, text="Large model again"),
        node("a", "Person", embeddings={"embedding": 768}, name="Ada"),
        relationship("r", "a", "KNOWS", "a", embeddings={"fact_embedding": 4}),
        relationship("h", "c1", "HAS_ENTITY", "a"),
    )

    result = EmbeddingConsistency().evaluate(graph, Inputs())

    # Group facts, entities, and chunks apart, so the person's own dimension is consistent.
    assert (result.evaluated, result.failed) == (5, 1)
    assert [finding.message for finding in result.findings] == [
        '"Small model" (Chunk): the embedding under property embedding has 768 dimensions, '
        "while other Chunk nodes have 1536"
    ]


def test_embedding_consistency_expects_only_embeddings_most_of_a_group_holds() -> None:
    people = [node(f"p{number}", "Person", name=f"P{number}") for number in range(9)]
    graph = view(*people, node("odd", "Person", embeddings={"embedding": 8}, name="Odd"))
    half = view(people[0], node("odd", "Person", embeddings={"embedding": 8}, name="Odd"))

    stray, expected = (
        EmbeddingConsistency().evaluate(graph, Inputs()),
        EmbeddingConsistency().evaluate(half, Inputs()),
    )

    # Fail no one for one stray embedding, but expect one that half of the group holds.
    assert stray.skip_reason is not None
    assert (expected.evaluated, expected.findings[0].node_ids) == (2, ("p0",))


def test_embedding_consistency_skips_graphs_without_embeddings() -> None:
    result = EmbeddingConsistency().evaluate(
        view(node("a", name="Ada", scores=[1.0, 2.0])), Inputs()
    )

    assert result.skip_reason == (
        "no group holds an embedding on at least half its members; kgeval reads "
        "non-empty lists of floats as embeddings"
    )
