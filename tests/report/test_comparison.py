"""Test the comparison of two reports, such as a baseline and a new run, metric by metric."""

import pytest

from kgeval.errors import InputError
from kgeval.evaluation import evaluate
from kgeval.graph.model import Graph
from kgeval.ground_truth import GroundTruth, load_ground_truth
from kgeval.metrics import (
    EntityConnectivity,
    EntityNaming,
    EntityUniqueness,
    FactAccuracy,
    FactProvenance,
    FactRecall,
    FactUniqueness,
    Finding,
    Inputs,
    MetricResult,
)
from kgeval.questions import load_questions
from kgeval.report.comparison import MetricChange, Trend, compare
from kgeval.report.model import Report
from kgeval.schema import load_schema
from tests.doubles import ScriptedModel
from tests.factories import EXAMPLE_DIR, example_graph, graph, node


def deterministic_report(original: Graph) -> Report:
    """Evaluate a graph with the example's schema, ground truth, and test questions."""
    return evaluate(
        original,
        schema=load_schema(EXAMPLE_DIR / "schema.yaml"),
        ground_truth=load_ground_truth(EXAMPLE_DIR / "ground_truth.yaml"),
        questions=load_questions(EXAMPLE_DIR / "questions.yaml"),
    )


def rebuilt(original: Graph) -> Graph:
    """Return the graph with new element IDs, as a fresh import would assign."""
    return Graph(
        nodes=tuple(
            element.model_copy(update={"id": f"4:x:{element.id}"}) for element in original.nodes
        ),
        relationships=tuple(
            element.model_copy(
                update={
                    "id": f"5:x:{element.id}",
                    "source_id": f"4:x:{element.source_id}",
                    "target_id": f"4:x:{element.target_id}",
                }
            )
            for element in original.relationships
        ),
    )


def without(original: Graph, *relationship_ids: str) -> Graph:
    return Graph(
        nodes=original.nodes,
        relationships=tuple(
            element for element in original.relationships if element.id not in relationship_ids
        ),
    )


def sampled(passed: int, decided: int) -> MetricResult:
    """Return a fact_accuracy result that checked a sample of a larger population."""
    findings = [
        Finding(message=f"fact {number} is unsupported") for number in range(decided - passed)
    ]
    return FactAccuracy().estimate(Inputs(model=ScriptedModel()), 1_000, decided, findings)


def exact(passed: int, evaluated: int) -> MetricResult:
    """Return an entity_naming result over a whole population."""
    findings = [Finding(message=f"entity {number}") for number in range(evaluated - passed)]
    return EntityNaming().result(Inputs(), evaluated, evaluated - passed, findings)


def sampled_change(base: MetricResult, head: MetricResult) -> MetricChange:
    return MetricChange(metric=base.metric, category=base.category, base=base, head=head)


def report_of(*results: MetricResult) -> Report:
    return evaluate(graph(), []).model_copy(update={"results": results})


def test_a_rebuilt_graph_with_new_element_ids_matches_every_finding() -> None:
    base = deterministic_report(example_graph())
    head = deterministic_report(rebuilt(example_graph()))

    comparison = compare(base, head)

    assert base.graph_fingerprint != head.graph_fingerprint
    assert not comparison.regressed
    assert all(
        not change.new_findings and not change.resolved_findings for change in comparison.changes
    )
    assert all(change.score_change == 0 for change in comparison.changes)


def test_fixed_defects_are_resolved_and_new_ones_regress() -> None:
    before = evaluate(example_graph(), [FactUniqueness(), EntityConnectivity()])
    after = evaluate(
        without(example_graph(), "born_in_copy", "married", "knows"),
        [FactUniqueness(), EntityConnectivity()],
    )

    fixed = compare(before, after)
    broken = compare(after, before)

    uniqueness, connectivity = fixed.changes
    assert [finding.message for finding in uniqueness.resolved_findings] == [
        '2 facts repeat "Lee Kuan Yew" (Person) -[:BORN_IN]-> "Singapore" (Location)'
    ]
    assert [finding.message for finding in connectivity.new_findings] == [
        '"Kwa Geok Choo" (Person) is in no fact with another entity'
    ]
    assert fixed.regressed
    assert [change.regressed for change in broken.changes] == [True, False]


def test_repeated_messages_pair_up_one_by_one() -> None:
    def report(isolated: int) -> Report:
        nodes = [node(f"o{number}", "Organization") for number in range(isolated)]
        return evaluate(graph(node("a", name="Ada"), *nodes), [EntityConnectivity()])

    [change] = compare(report(1), report(3)).changes

    assert [finding.message for finding in change.new_findings] == [
        "unnamed (Organization) is in no fact with another entity"
    ] * 2


def test_skipping_a_metric_the_base_scored_regresses() -> None:
    base = evaluate(example_graph(), [EntityNaming()])
    head = evaluate(graph(), [EntityNaming()])

    [change] = compare(base, head).changes

    # Resolve nothing, since a skipped metric cannot show that a defect was fixed.
    assert (change.regressed, change.score_change, change.resolved_findings) == (True, None, ())
    assert change.trend is None


def test_a_metric_the_head_drops_regresses_and_one_it_adds_lists_its_findings_as_new() -> None:
    base = evaluate(example_graph(), [EntityNaming()])
    head = evaluate(example_graph(), [EntityConnectivity()])

    added, dropped = compare(base, head).changes

    assert (added.metric, len(added.new_findings), added.regressed) == (
        "entity_connectivity",
        1,
        False,
    )
    # Regress when the head stops scoring a metric, so dropping an input cannot pass the gate.
    assert (dropped.metric, dropped.resolved_findings, dropped.regressed) == (
        "entity_naming",
        (),
        True,
    )


def test_new_findings_without_a_lower_score_do_not_regress() -> None:
    def report(*names: str) -> Report:
        return evaluate(graph(*(node(name, name=name) for name in names)), [EntityNaming()])

    [change] = compare(report("Ada", "he", "Bob"), report("Ada", "Bob", "she")).changes

    assert ([finding.subject for finding in change.new_findings], change.regressed) == (
        ['"she"'],
        False,
    )


def test_findings_pair_up_by_subject_when_their_details_change() -> None:
    def report(copies: int) -> Report:
        names = [node(f"ada{number}", name="Ada") for number in range(copies)]
        return evaluate(graph(node("bob", name="Bob"), *names), [EntityUniqueness()])

    [change] = compare(report(2), report(3)).changes

    # Keep one finding about 'ada', though its message now counts three entities.
    assert (change.new_findings, change.resolved_findings) == ((), ())
    assert change.regressed


def test_a_lower_score_from_a_smaller_population_alone_does_not_regress() -> None:
    before = evaluate(example_graph(), [FactProvenance()])
    # Remove a duplicate that fact_provenance passed, which lowers its share alone.
    after = evaluate(without(example_graph(), "born_in_copy"), [FactProvenance()])

    [change] = compare(before, after).changes

    assert (after["fact_provenance"].failed, change.trend, change.regressed) == (
        before["fact_provenance"].failed,
        Trend.WORSENED,
        False,
    )


def test_a_significant_drop_of_a_sampled_score_regresses() -> None:
    dropped = sampled_change(sampled(95, 100), sampled(40, 100))
    noisy = sampled_change(sampled(95, 100), sampled(92, 100))

    assert (dropped.trend, dropped.regressed) == (Trend.WORSENED, True)
    assert (noisy.trend, noisy.regressed) == (Trend.INCONCLUSIVE, False)


def test_drops_within_max_drop_do_not_regress() -> None:
    base, head = report_of(exact(90, 100)), report_of(exact(89, 100))

    assert compare(base, head).regressed
    assert not compare(base, head, max_drop=0.01).regressed
    assert compare(base, head, max_drop=0.009).regressed
    with pytest.raises(InputError, match="max_drop must be from 0 to 1"):
        compare(base, head, max_drop=1.5)


def test_findings_of_model_metrics_are_not_compared() -> None:
    base = evaluate(example_graph(), [FactAccuracy()], model=ScriptedModel("-[:KNOWS]->"))
    head = evaluate(example_graph(), [FactAccuracy()], model=ScriptedModel("-[:MARRIED]->"))

    comparison = compare(base, head)

    [change] = comparison.changes
    assert (change.new_findings, change.resolved_findings, comparison.regressed) == ((), (), False)
    assert "findings of model metrics are not compared" in comparison.to_markdown()


def test_deterministic_score_changes_are_exact() -> None:
    metrics = [FactUniqueness(), EntityConnectivity(), EntityNaming()]
    before = evaluate(example_graph(), metrics)
    after = evaluate(without(example_graph(), "born_in_copy", "married", "knows"), metrics)

    changes = compare(before, after).changes

    assert [change.trend for change in changes] == [
        Trend.IMPROVED,
        Trend.WORSENED,
        Trend.UNCHANGED,
    ]
    assert all(change.change_interval is None for change in changes)


def test_sampled_score_changes_carry_newcombes_interval() -> None:
    change = sampled_change(sampled(48, 80), sampled(56, 70))

    # Match Newcombe (1998), Table II, method 10, which gives 0.0524 to 0.3339 for 56/70 - 48/80.
    assert change.change_interval == pytest.approx((0.0524, 0.3339), abs=5e-5)
    assert change.trend is Trend.IMPROVED


def test_a_change_that_sampling_can_explain_is_inconclusive() -> None:
    change = sampled_change(sampled(0, 29), sampled(5, 56))

    # Match Newcombe (1998), Table II, method 10, which gives -0.0381 to 0.1926 for 5/56 - 0/29.
    assert change.change_interval == pytest.approx((-0.0381, 0.1926), abs=5e-5)
    assert change.trend is Trend.INCONCLUSIVE


def test_trend_counts_leave_out_metrics_without_two_scores() -> None:
    base = evaluate(example_graph(), [EntityNaming(), EntityConnectivity()])
    head = evaluate(graph(node("a", name="Ada")), [EntityNaming(), EntityConnectivity()])
    named = evaluate(example_graph(), [EntityNaming()])
    skipped = evaluate(graph(), [EntityNaming()])

    assert compare(base, head).trend_counts == {
        Trend.IMPROVED: 1,
        Trend.WORSENED: 1,
        Trend.UNCHANGED: 0,
        Trend.INCONCLUSIVE: 0,
    }
    assert "0 inconclusive, 1 without two comparable scores\n" in (
        compare(named, skipped).to_markdown()
    )


def test_markdown_summarizes_trends_and_the_metadata_that_differs() -> None:
    metrics = [FactUniqueness(), EntityConnectivity(), EntityNaming()]
    before = evaluate(example_graph(), metrics, metadata={"model": "gpt-5-mini", "schema": "v2"})
    after = evaluate(
        without(example_graph(), "born_in_copy", "married", "knows"),
        metrics,
        metadata={"model": "gpt-5", "schema": "v2", "seed": "7"},
    )

    markdown = compare(before, after).to_markdown()

    assert "## Summary\n\nScores: 1 improved, 1 worsened, 1 unchanged, 0 inconclusive\n" in markdown
    assert "| model | gpt-5-mini | gpt-5 |\n| seed | — | 7 |\n\n## Scores" in markdown
    assert "| schema |" not in markdown
    assert "| Metadata |" not in compare(before, before).to_markdown()


def test_markdown_shows_the_interval_and_trend_of_sampled_score_changes() -> None:
    comparison = compare(report_of(sampled(48, 80)), report_of(sampled(56, 70)))

    assert (
        "| fact_accuracy | 60.0% of 80 | 80.0% of 70 | +20.0 pp (95% CI +5.2 pp to +33.4 pp) "
        "| — | — | improved; findings of model metrics are not compared |"
    ) in comparison.to_markdown()


def test_markdown_shows_score_changes_and_changed_findings() -> None:
    metrics = [FactUniqueness(), EntityConnectivity()]
    before = evaluate(example_graph(), metrics)
    after = evaluate(without(example_graph(), "born_in_copy", "married", "knows"), metrics)

    markdown = compare(before, after).to_markdown()

    assert markdown.startswith("# kgeval comparison\n\n**REGRESSED**\n")
    assert "| fact_uniqueness | 90.9% of 11 | 100.0% of 8 | +9.1 pp | 0 | 1 |  |" in markdown
    assert (
        "| entity_connectivity | 90.9% of 11 | 81.8% of 11 | -9.1 pp | 1 | 0 | **REGRESSED** |"
        in markdown
    )
    assert "## New findings\n\n### entity_connectivity: 1 new finding\n\n" in markdown
    assert "## Resolved findings\n\n### fact_uniqueness: 1 resolved finding\n\n" in markdown
    assert (
        compare(before, before)
        .to_markdown()
        .startswith("# kgeval comparison\n\n**NO REGRESSION**\n")
    )


def test_markdown_notes_metrics_whose_parameters_changed() -> None:
    def report(*ground_truth_facts: tuple[str, str, str]) -> Report:
        return evaluate(
            example_graph(), [FactRecall()], ground_truth=GroundTruth(facts=ground_truth_facts)
        )

    comparison = compare(report(("Ada", "KNOWS", "Bob")), report(("Ada", "KNOWS", "Cy")))

    [change] = comparison.changes
    assert (change.comparable, change.trend, change.regressed) == (False, None, False)
    assert "| — | — | — | parameters changed, so not compared |" in comparison.to_markdown()


def test_markdown_warns_when_reports_come_from_different_kgeval_versions() -> None:
    base = evaluate(example_graph(), [EntityNaming()])
    head = base.model_copy(update={"kgeval_version": "9.9.9"})

    assert (
        f"The reports come from kgeval {base.kgeval_version} and 9.9.9, whose metrics may differ."
        in compare(base, head).to_markdown()
    )


def test_comparisons_render_as_markdown_in_notebooks() -> None:
    report = evaluate(example_graph(), [EntityConnectivity()])
    comparison = compare(report, report)

    assert comparison._repr_markdown_() == comparison.to_markdown()
