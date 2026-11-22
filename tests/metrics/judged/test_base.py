"""Test the base classes of the judged metrics."""

from typing import ClassVar

import pytest
from pydantic import BaseModel

from kgeval.errors import JudgeError, VerdictError
from kgeval.graph.view import GraphView
from kgeval.metrics.base import Finding, MetricResult
from kgeval.metrics.judged._prompts import chunk_texts, entity_name, quote_chunks, spell
from kgeval.metrics.judged.base import SampledMetric
from tests.doubles import FailingJudge, ScriptedJudge, UnusableJudge
from tests.factories import example_view, node, view


class Verdict(BaseModel):
    reason: str
    approved: bool


class EntityApproval(SampledMetric):
    """Ask the judge to approve each sampled entity, by ID."""

    name: ClassVar[str] = "entity_approval"
    unit: ClassVar[str] = "entities"

    def evaluate(self, graph: GraphView) -> MetricResult:
        sample = self.sample(graph.entities, lambda entity: entity.id)
        verdicts = self.ask_all([entity.id for entity in sample], Verdict)
        findings = [
            Finding(message=entity.id, node_ids=(entity.id,))
            for entity, verdict in zip(sample, verdicts, strict=True)
            if verdict is not None and not verdict.approved
        ]
        return self.estimate(len(graph.entities), len(sample), findings, verdicts.count(None))


def sampled(size: int, seed: int = 0) -> set[str]:
    judge = ScriptedJudge()
    EntityApproval(judge=judge, sample_size=size, seed=seed).evaluate(example_view())
    return set(judge.prompts)


def test_sample_is_reproducible_and_only_grows_with_its_size() -> None:
    assert sampled(3) == sampled(3)
    assert sampled(3) < sampled(5)
    assert sampled(3) != sampled(3, seed=1)


def test_estimates_from_a_sample_carry_a_wilson_interval() -> None:
    metric = EntityApproval(judge=ScriptedJudge())
    findings = [Finding(message="rejected"), Finding(message="rejected")]

    result = metric.estimate(population=20, sampled=10, findings=findings)

    assert (result.evaluated, result.failed) == (10, 2)
    assert result.confidence_interval == pytest.approx((0.4902, 0.9433), abs=1e-4)


def test_clusters_of_one_unit_get_the_plain_wilson_interval() -> None:
    metric = EntityApproval(judge=ScriptedJudge())
    clusters = [(1, 1)] * 8 + [(1, 0)] * 2

    result = metric.estimate_clusters(20, clusters, [Finding(message="rejected")] * 2)

    assert (result.evaluated, result.failed) == (10, 2)
    assert result.confidence_interval == pytest.approx((0.4902, 0.9433), abs=1e-4)


def test_clusters_that_pass_or_fail_together_widen_the_interval() -> None:
    metric = EntityApproval(judge=ScriptedJudge())
    together = metric.estimate_clusters(100, [(10, 10)] * 8 + [(10, 0)] * 2, [])
    apart = metric.estimate_clusters(100, [(10, 8)] * 10, [])
    perfect = metric.estimate_clusters(100, [(10, 10)] * 10, [])

    # Judge 100 facts in each: failures bunched in two chunks tell less than failures spread out.
    assert together.score == apart.score == 0.8
    assert together.confidence_interval == pytest.approx((0.4902, 0.9433), abs=1e-4)
    assert apart.confidence_interval is not None
    assert together.confidence_interval is not None
    assert apart.confidence_interval[0] > together.confidence_interval[0]
    # Keep a perfect sample's interval open, as for ten independent units.
    assert perfect.confidence_interval == pytest.approx((0.7225, 1.0), abs=1e-4)


def test_judging_the_whole_population_needs_no_interval() -> None:
    result = EntityApproval(judge=ScriptedJudge("lee"), sample_size=50).evaluate(example_view())

    assert (result.evaluated, result.failed, result.confidence_interval) == (11, 1, None)
    assert result.findings[0].node_ids == ("lee",)


def test_a_prompt_without_a_usable_verdict_is_retried_once_then_left_unjudged() -> None:
    retried = EntityApproval(judge=UnusableJudge("lee", failures=1)).evaluate(example_view())
    unjudged = EntityApproval(judge=UnusableJudge("lee")).evaluate(example_view())

    assert (retried.evaluated, retried.unjudged, retried.confidence_interval) == (11, 0, None)
    assert (unjudged.evaluated, unjudged.unjudged) == (10, 1)
    # Give the score an interval, since the judged units no longer cover the population.
    assert unjudged.confidence_interval is not None
    assert unjudged.findings == ()


def test_no_usable_verdict_for_any_prompt_fails_the_metric() -> None:
    with pytest.raises(VerdictError, match="no usable verdict for any prompt"):
        EntityApproval(judge=UnusableJudge("")).evaluate(example_view())


def test_a_judge_failure_stops_the_metric_and_cancels_queued_prompts() -> None:
    # Slow each call so the one worker cannot drain the queue before the cancellation.
    judge = FailingJudge(delay=0.05)

    with pytest.raises(JudgeError, match="model unavailable"):
        EntityApproval(judge=judge, concurrency=1).evaluate(example_view())
    assert judge.calls < 8


def test_chunk_texts_put_texts_naming_everyone_first_and_respect_the_limit() -> None:
    graph = view(*(node(f"c{number}", "Chunk", text=f"Chunk {number}") for number in range(5)))

    texts = chunk_texts(graph, ["c4", "c3", "c2", "c1", "c0"], ["chunk 3"])

    # Rank c3, which names everyone, first, then keep element ID order up to the limit.
    assert texts == ["Chunk 3", "Chunk 0", "Chunk 1"]


def test_chunk_texts_leave_out_nodes_without_text() -> None:
    texts = chunk_texts(example_view(), ["document", "chunk2"], ["Kwa Geok Choo", "LEE KUAN YEW"])

    assert texts == ["Kwa Geok Choo married Lee Kuan Yew."]


def test_judges_read_chunks_between_tags_and_elements_spelled_in_full() -> None:
    long_name = "International Business Machines Corporation of Armonk, New York, United States"
    graph = view(node("a", "Company", name=long_name), node("b"))

    assert quote_chunks(["One.", "Two."]) == "<chunk>\nOne.\n</chunk>\n\n<chunk>\nTwo.\n</chunk>"
    assert spell(graph, graph.node("a")) == f'"{long_name}" (Company)'
    assert spell(graph, graph.node("b")) == "unnamed"


def test_entities_without_a_letter_or_digit_have_no_name_to_look_up() -> None:
    graph = view(node("a", name="Ada"), node("dash", name="—"), node("none"))

    assert [entity_name(graph, entity) for entity in graph.entities] == ["Ada", None, None]


def test_signature_names_the_judge_and_sampling() -> None:
    metric = EntityApproval(judge=ScriptedJudge(), sample_size=50, seed=7)

    assert metric.signature() == "entity_approval|judge=scripted|sample_size=50|seed=7"
    assert metric.category.value == "judged"
