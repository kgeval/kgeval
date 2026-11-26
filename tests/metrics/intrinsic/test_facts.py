"""Test the fact uniqueness metric."""

from kgeval.graph.layout import Layout
from kgeval.metrics.base import Inputs
from kgeval.metrics.intrinsic.facts import FactUniqueness
from tests.factories import example_view, node, relationship, view


def test_fact_uniqueness_counts_redundant_copies() -> None:
    result = FactUniqueness().evaluate(example_view(), Inputs())

    assert (result.evaluated, result.failed) == (11, 1)
    [finding] = result.findings
    assert finding.relationship_ids == ("born_in", "born_in_copy")
    assert finding.node_ids == ("lee", "singapore")
    assert finding.message == (
        '2 facts repeat "Lee Kuan Yew" (Person) -[:BORN_IN]-> "Singapore" (Location)'
    )
    assert finding.subject == '"Lee Kuan Yew" (Person) -[:BORN_IN]-> "Singapore" (Location)'


def test_fact_uniqueness_finds_repeats_whatever_their_properties() -> None:
    result = FactUniqueness().evaluate(
        view(
            node("a"),
            node("b"),
            relationship("r1", "a", "WORKS_AT", "b", chunk="c1", uuid="u1"),
            relationship("r2", "a", "WORKS_AT", "b", chunk="c2", uuid="u2"),
            relationship("r3", "b", "WORKS_AT", "a"),
        ),
        Inputs(),
    )

    assert (result.evaluated, result.failed) == (3, 1)
    assert result.findings[0].relationship_ids == ("r1", "r2")


def test_fact_uniqueness_reads_relationship_types_through_the_layout() -> None:
    graphiti = view(
        node("a", name="Ada"),
        node("b", name="Babbage"),
        relationship("r1", "a", "RELATES_TO", "b", name="WORKED_WITH"),
        relationship("r2", "a", "RELATES_TO", "b", name="CORRESPONDED_WITH"),
        layout=Layout(type_keys=("name",)),
    )

    assert FactUniqueness().evaluate(graphiti, Inputs()).failed == 0


def test_fact_uniqueness_skips_graphs_without_facts() -> None:
    graph = view(
        node("a"), node("c", "Chunk", text="Ada."), relationship("r", "c", "HAS_ENTITY", "a")
    )

    assert FactUniqueness().evaluate(graph, Inputs()).skip_reason == "no facts to evaluate"
