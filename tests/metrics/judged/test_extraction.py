"""Test the extraction completeness metric."""

from kgeval.graph.layout import Layout
from kgeval.graph.model import Node, Relationship
from kgeval.metrics.base import NO_LEXICAL_NODES
from kgeval.metrics.judged.extraction import ExtractionCompleteness
from tests.doubles import ScriptedJudge
from tests.factories import example_view, node, relationship, view


def test_judge_sees_each_chunk_with_what_was_extracted_from_it() -> None:
    judge = ScriptedJudge()

    result = ExtractionCompleteness(judge=judge).evaluate(example_view())

    assert len(judge.prompts) == 5
    # Skip the metric when the judge finds no chunk stating a fact.
    assert result.skip_reason == "no facts stated in chunks to evaluate"
    [chunk1] = [prompt for prompt in judge.prompts if "born in Singapore" in prompt]
    assert '- "Lee Kuan Yew" (Person)\n' in chunk1
    assert '- "Lee Kuan Yew" (Person) -[:BORN_IN]-> "Singapore" (Location)\n' in chunk1
    assert "Kwa Geok Choo" not in chunk1
    [contents] = [prompt for prompt in judge.prompts if "Table of contents" in prompt]
    assert "Entities extracted from the chunk:\n(none)" in contents


def test_chunks_that_state_missing_facts_fail_with_the_facts_listed() -> None:
    judge = ScriptedJudge(
        "born in Singapore",
        rejection={
            "facts": [
                {"fact": "Lee Kuan Yew was born in Singapore", "extracted": True},
                {"fact": "Lee Kuan Yew co-founded Lee & Lee", "extracted": False},
                {"fact": "Lee & Lee is a law firm", "extracted": False},
            ]
        },
    )

    result = ExtractionCompleteness(judge=judge).evaluate(example_view())

    [finding] = result.findings
    assert finding.message.endswith(
        "(Chunk) states facts the graph lacks: "
        "Lee Kuan Yew co-founded Lee & Lee. Lee & Lee is a law firm."
    )
    assert finding.node_ids == ("chunk1",)
    # Count each stated fact, so the chunk's one extracted fact of three scores a third.
    assert (result.unit, result.evaluated, result.failed) == ("facts stated in chunks", 3, 2)
    assert result.confidence_interval is None


def test_metric_skips_graphs_without_chunk_text() -> None:
    def skip_reason(*elements: Node | Relationship) -> str | None:
        return ExtractionCompleteness(judge=ScriptedJudge()).evaluate(view(*elements)).skip_reason

    assert skip_reason(node("a", "Person")) == NO_LEXICAL_NODES
    chunk_without_text = view(
        node("c", "Chunk"),
        node("a", "Person"),
        relationship("h", "c", "HAS_ENTITY", "a"),
        layout=Layout(lexical_labels=frozenset({"Chunk"})),
    )
    result = ExtractionCompleteness(judge=ScriptedJudge()).evaluate(chunk_without_text)
    assert result.skip_reason == "no chunk has text"
