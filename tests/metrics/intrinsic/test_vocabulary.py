"""Test the vocabulary uniqueness metric."""

from kgeval.graph.layout import Layout
from kgeval.metrics.base import Inputs
from kgeval.metrics.intrinsic.vocabulary import VocabularyUniqueness
from tests.factories import example_view, node, relationship, view


def test_vocabulary_uniqueness_reports_spelling_variants_of_each_kind_of_term() -> None:
    result = VocabularyUniqueness().evaluate(example_view(), Inputs())

    # Count 5 node types, 7 relationship types, and the property keys id and born.
    assert (result.unit, result.evaluated, result.failed) == ("vocabulary terms", 14, 2)
    assert [
        (finding.message, finding.node_ids, finding.relationship_ids) for finding in result.findings
    ] == [
        ("PERSON spells the node type Person differently", ("he",), ()),
        ("Founded spells the relationship type FOUNDED differently", (), ("he_founded",)),
    ]


def test_vocabulary_uniqueness_counts_every_variant_but_the_intended_one() -> None:
    graph = view(
        node("a", "Person", name="Ada"),
        node("b", "PERSON", name="Bob"),
        node("c", "Persons", name="Cy"),
        node("d", "Company", name="Acme"),
    )

    result = VocabularyUniqueness().evaluate(graph, Inputs())

    # Break the tie by name: PERSON sorts before Person and Persons.
    assert (result.evaluated, result.failed) == (5, 2)
    [finding] = result.findings
    assert finding.message == "Person and Persons spell the node type PERSON differently"
    assert finding.node_ids == ("a", "c")


def test_vocabulary_uniqueness_compares_property_keys_of_entities_and_facts() -> None:
    graph = view(
        node("a", "Person", name="Ada", birth_date="1815"),
        node("b", "Person", name="Bob", birthDate="1791"),
        relationship("r", "a", "KNOWS", "b", birth_date="1830"),
    )

    result = VocabularyUniqueness().evaluate(graph, Inputs())

    [finding] = result.findings
    assert finding.message == "birthDate spells the property key birth_date differently"
    assert finding.node_ids == ("b",)


def test_vocabulary_uniqueness_reads_relationship_types_from_the_layouts_type_keys() -> None:
    graph = view(
        node("a", name="Ada"),
        node("b", name="Bob"),
        relationship("r1", "a", "RELATES_TO", "b", name="WORKS_WITH"),
        relationship("r2", "a", "RELATES_TO", "b", name="worksWith"),
        layout=Layout(type_keys=("name",)),
    )

    result = VocabularyUniqueness().evaluate(graph, Inputs())

    assert result.findings[0].message == (
        "worksWith spells the relationship type WORKS_WITH differently"
    )
