"""Test the entity and fact criteria, which define model metrics in one sentence."""

from datetime import date

import pytest
from pydantic import ValidationError

from kgeval.metrics.base import Inputs
from kgeval.metrics.model.criteria import CriterionVerdict, EntityCriterion, FactCriterion
from tests.doubles import ScriptedModel
from tests.factories import example_view, node, relationship, view

FOUNDED_OR_BORN_IN = frozenset({"FOUNDED", "BORN_IN"})


def test_model_sees_the_criterion_and_each_entity_with_its_properties_and_chunks() -> None:
    model = ScriptedModel()
    graph = view(
        node("c", "Chunk", text="Ada Lovelace was born in 1815."),
        node(
            "ada",
            "__Entity__",
            "Person",
            name="Ada Lovelace",
            born=date(1815, 12, 10),
            notes="x" * 2000,
        ),
        node("london", "__Entity__", "City", name="London"),
        relationship("h", "c", "HAS_ENTITY", "ada"),
    )
    metric = EntityCriterion(
        name="dated_people",
        criterion="The entity gives a birth date.",
        node_types=frozenset({"Person"}),
    )

    result = metric.evaluate(graph, Inputs(model=model))

    assert (result.evaluated, result.unit) == (1, "entities the criterion applies to")
    [prompt] = model.prompts
    assert (
        "Criterion: The entity gives a birth date.\n\n"
        'Entity: "Ada Lovelace" (Person)\n'
        'Properties:\n- born: "1815-12-10"\n- name: "Ada Lovelace"\n'
        f'- notes: "{"x" * 998}…\n\n'
        "Chunks of text the entity was extracted from:\n"
        "<chunk>\nAda Lovelace was born in 1815.\n</chunk>\n\n"
    ) in prompt


def test_facts_that_fail_become_findings_with_the_models_reason() -> None:
    model = ScriptedModel("-[:FOUNDED]-> unnamed")
    metric = FactCriterion(
        name="named_organizations",
        criterion="The fact names both entities.",
        relationship_types=FOUNDED_OR_BORN_IN,
    )

    result = metric.evaluate(example_view(), Inputs(model=model))

    # Leave out he_founded, whose relationship type Founded is not FOUNDED.
    assert (result.evaluated, result.failed) == (6, 1)
    [finding] = result.findings
    assert finding.message == (
        '"Lee Kuan Yew" (Person) -[:FOUNDED]-> unnamed (Organization) does not meet the '
        "criterion: Scripted."
    )
    assert (finding.node_ids, finding.relationship_ids) == (
        ("lee", "unnamed"),
        ("founded_unnamed",),
    )
    [ungrounded] = [prompt for prompt in model.prompts if '"lee kuan yew" (Person) -[' in prompt]
    assert (
        'Fact: "lee kuan yew" (Person) -[:FOUNDED]-> "People\'s Action Party" (Organization)\n'
        "Properties: none\n\nChunks of text the fact was extracted from:\n(none)\n\n"
    ) in ungrounded


def test_units_the_criterion_does_not_apply_to_are_left_out() -> None:
    def checked(rejected: str) -> tuple[int, str | None]:
        model = ScriptedModel(rejected, rejection={"meets": "not applicable"})
        metric = EntityCriterion(name="people", criterion="It is a person.")
        result = metric.evaluate(example_view(), Inputs(model=model))
        return result.evaluated, result.skip_reason

    assert checked("(Location)") == (10, None)
    assert checked("Entity:") == (0, "the criterion applies to no sampled entity")


def test_criteria_skip_graphs_without_units_of_their_types() -> None:
    model = ScriptedModel()
    drugs = EntityCriterion(
        name="drugs", criterion="It names a class.", node_types=frozenset({"Drug", "Chemical"})
    )
    facts = FactCriterion(name="facts", criterion="It is specific.")

    inputs = Inputs(model=model)
    assert drugs.evaluate(example_view(), inputs).skip_reason == (
        "no entity has the node type Chemical or Drug"
    )
    assert facts.evaluate(view(node("a", name="Ada")), inputs).skip_reason == (
        "the graph has no facts"
    )
    assert model.prompts == []


def test_signature_changes_with_the_criterion_what_it_applies_to_and_its_types() -> None:
    inputs = Inputs(model=ScriptedModel())

    def signature(metric: EntityCriterion | FactCriterion) -> str:
        return metric.signature(inputs)

    dated = signature(FactCriterion(name="dated", criterion="It gives a year."))

    assert dated.startswith("dated|model=scripted|sample_size=100|seed=0|criterion=")
    assert dated == signature(FactCriterion(name="dated", criterion=" It gives a year.\n"))
    assert dated != signature(FactCriterion(name="dated", criterion="It gives a date."))
    assert dated != signature(EntityCriterion(name="dated", criterion="It gives a year."))
    assert dated != signature(
        FactCriterion(
            name="dated", criterion="It gives a year.", relationship_types=FOUNDED_OR_BORN_IN
        )
    )


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"name": "Dated", "criterion": "It gives a year."}, "should match pattern"),
        ({"name": "dated", "criterion": " "}, "at least 1 character"),
        ({"name": "dated", "criterion": "x", "node_types": ["Person"]}, "Extra inputs"),
    ],
)
def test_criteria_reject_invalid_fields(fields: dict[str, object], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        FactCriterion.model_validate(fields)


def test_verdict_lists_reason_before_verdict() -> None:
    assert list(CriterionVerdict.model_fields) == ["reason", "meets"]
