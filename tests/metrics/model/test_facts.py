"""Test the fact accuracy metric."""

import pytest

from kgeval.errors import ModelError
from kgeval.metrics.base import Inputs
from kgeval.metrics.model.facts import FactAccuracy, SupportVerdict
from tests.doubles import FailingModel, ScriptedModel
from tests.factories import example_view, node, relationship, view


def test_model_sees_each_grounded_fact_with_its_shared_chunks() -> None:
    model = ScriptedModel()

    result = FactAccuracy().evaluate(example_view(), Inputs(model=model))

    # Leave out duplicate_founded, whose entities share no chunk, and the unnamed founding.
    assert (result.evaluated, result.failed, result.confidence_interval) == (9, 0, None)
    assert not [prompt for prompt in model.prompts if "unnamed" in prompt]
    [married] = [prompt for prompt in model.prompts if "-[:MARRIED]->" in prompt]
    assert 'Fact: "Kwa Geok Choo" (Person) -[:MARRIED]-> "Lee Kuan Yew" (Person)' in married
    assert "<chunk>\nKwa Geok Choo married Lee Kuan Yew.\n</chunk>" in married
    assert "Singapore" not in married


def test_unsupported_facts_become_findings_with_the_models_reason() -> None:
    result = FactAccuracy().evaluate(example_view(), Inputs(model=ScriptedModel("-[:KNOWS]->")))

    [finding] = result.findings
    assert finding.message == (
        'the chunks do not support "Kwa Geok Choo" (Person) -[:KNOWS]-> '
        '"Kwa Geok Choo" (Person): Scripted.'
    )
    assert (finding.node_ids, finding.relationship_ids) == (("kwa",), ("knows",))


def test_model_sees_descriptions_and_no_element_ids() -> None:
    model = ScriptedModel()
    graph = view(
        node("c", "Chunk", text="Ada founded a company."),
        node("ada", "__Entity__", "Person", name="Ada"),
        node("4:a1b2:7", "__Entity__", "Company", name="Analytical Engines"),
        relationship("h1", "c", "HAS_ENTITY", "ada"),
        relationship("h2", "c", "HAS_ENTITY", "4:a1b2:7"),
        relationship("r", "ada", "FOUNDED", "4:a1b2:7", description="Ada founded it in 1843."),
    )

    FactAccuracy().evaluate(graph, Inputs(model=model))

    assert (
        'Fact: "Ada" (Person) -[:FOUNDED]-> "Analytical Engines" (Company)\n'
        "Description: Ada founded it in 1843.\n\n<chunk>"
    ) in model.prompts[0]
    assert "4:a1b2:7" not in model.prompts[0]


def test_sample_and_prompts_depend_on_content_not_element_ids() -> None:
    def sampled(prefix: str, chunk_ids: tuple[str, str]) -> set[str]:
        people = [f"Person {number}" for number in range(10)]
        model = ScriptedModel()
        chunks = [
            node(f"{prefix}{chunk_id}", "Chunk", text=f"Hub {verb} {', '.join(people)}.")
            for chunk_id, verb in zip(chunk_ids, ("knows", "met"), strict=True)
        ]
        names = ["hub", *people]
        graph = view(
            *chunks,
            node(f"{prefix}hub", "__Entity__", name="Hub"),
            *(node(f"{prefix}{name}", "__Entity__", name=name) for name in people),
            *(
                relationship(f"{chunk.id}>{name}", chunk.id, "HAS_ENTITY", f"{prefix}{name}")
                for chunk in chunks
                for name in names
            ),
            *(
                relationship(f"{prefix}r{name}", f"{prefix}hub", "KNOWS", f"{prefix}{name}")
                for name in people
            ),
        )
        FactAccuracy(sample_size=3).evaluate(graph, Inputs(model=model))
        return set(model.prompts)

    # Reverse the chunks' element ID order, which must change neither the sample nor its prompts.
    assert len(sampled("4:x:", ("a", "b"))) == 3
    assert sampled("4:x:", ("a", "b")) == sampled("5:y:", ("b", "a"))


def test_metric_skips_graphs_without_grounded_facts() -> None:
    graph = view(node("a", name="Ada"), node("b", name="Bob"), relationship("r", "a", "KNOWS", "b"))

    assert FactAccuracy().evaluate(graph, Inputs(model=ScriptedModel())).skip_reason == (
        "no fact between entities with a name is grounded in a chunk with text"
    )


def test_model_failures_propagate() -> None:
    with pytest.raises(ModelError, match="model unavailable"):
        FactAccuracy().evaluate(example_view(), Inputs(model=FailingModel()))


def test_verdict_lists_reason_before_verdict() -> None:
    assert list(SupportVerdict.model_fields) == ["reason", "supported"]
