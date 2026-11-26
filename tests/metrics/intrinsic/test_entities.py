"""Test the entity uniqueness, connectivity, and naming metrics."""

import pytest

from kgeval.metrics.base import Inputs
from kgeval.metrics.intrinsic.entities import EntityConnectivity, EntityNaming, EntityUniqueness
from tests.factories import example_view, node, relationship, view


def test_entity_uniqueness_reports_entities_that_share_a_normalized_name() -> None:
    result = EntityUniqueness().evaluate(example_view(), Inputs())

    assert (result.unit, result.evaluated, result.failed) == ("entities with a name", 10, 1)
    [finding] = result.findings
    assert finding.node_ids == ("duplicate", "lee")
    assert finding.message == (
        "2 entities share the normalized name 'lee kuan yew': "
        '"Lee Kuan Yew" (Person), "lee kuan yew" (Person)'
    )


def test_entity_uniqueness_lists_duplicates_in_the_same_order_whatever_their_ids() -> None:
    def message(first_id: str, second_id: str) -> str:
        graph = view(node(first_id, name="Ada"), node(second_id, name="ADA"))
        return EntityUniqueness().evaluate(graph, Inputs()).findings[0].message

    assert message("a", "b") == message("b", "a")


def test_entity_connectivity_reports_entities_in_no_fact_with_another_entity() -> None:
    result = EntityConnectivity().evaluate(example_view(), Inputs())

    assert (result.evaluated, result.failed) == (11, 1)
    [finding] = result.findings
    assert finding.message == ('"Marina Bay Sands" (Landmark) is in no fact with another entity')


def test_entity_connectivity_ignores_self_loops() -> None:
    result = EntityConnectivity().evaluate(
        view(node("a"), relationship("r", "a", "IS", "a")), Inputs()
    )

    assert result.failed == 1


def test_entity_naming_reports_unnamed_entities_and_entities_named_by_a_pronoun() -> None:
    result = EntityNaming().evaluate(example_view(), Inputs())

    assert (result.unit, result.evaluated, result.failed) == ("entities", 11, 2)
    assert [(finding.message, finding.node_ids) for finding in result.findings] == [
        ('"He" (PERSON) is named by a pronoun', ("he",)),
        ("unnamed (Organization) has no non-blank string in name, title, id", ("unnamed",)),
    ]


@pytest.mark.parametrize(
    ("name", "message"),
    [
        ("they", "is named by a pronoun"),
        ("Himself.", "is named by a pronoun"),
        ("no one", "is named by a pronoun"),
        ("I", "is named by a pronoun"),
        ("1923", "is named by a value without a letter"),
        ("50%", "is named by a value without a letter"),
        ("$3,000", "is named by a value without a letter"),
        ("US", None),
        ("IT", None),
        ("Hera", None),
        ("The Who", None),
        ("Q3 2024", None),
    ],
)
def test_entity_naming_fails_pronouns_and_values_but_not_acronyms(
    name: str, message: str | None
) -> None:
    result = EntityNaming().evaluate(view(node("a", name=name)), Inputs())

    assert [finding.message for finding in result.findings] == (
        [f'"{name}" {message}'] if message else []
    )
