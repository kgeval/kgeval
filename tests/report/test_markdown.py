"""Test the Markdown rendering of reports and comparisons."""

from kgeval.evaluation import evaluate
from kgeval.graph.layout import Layout
from kgeval.metrics import EntityConnectivity, EntityUniqueness, FactAccuracy, FactProvenance
from tests.doubles import UnusableModel
from tests.factories import example_graph, graph, node, relationship


def test_markdown_shows_summary_scores_and_findings_with_element_ids() -> None:
    report = evaluate(example_graph(), [EntityUniqueness(threshold=0.95), FactProvenance()])

    markdown = report.to_markdown()

    assert "**FAILED**" in markdown
    assert (
        "- Graph: 19 nodes (11 entities, 7 lexical nodes, 1 community node, 0 other nodes) and "
        "40 relationships (11 facts, 14 provenance relationships)"
    ) in markdown
    assert (
        "| entity_uniqueness | 90.0% | 1 / 10 entities with a name | ≥ 95.0% | **FAIL** |"
        in markdown
    )
    assert "| fact_provenance | 90.9% | 1 / 11 facts | — |  |" in markdown
    assert "### entity_uniqueness: 1 of 10 entities with a name failed" in markdown
    assert "  - nodes: `duplicate`, `lee`" in markdown
    assert "  - relationships: `duplicate_founded`" in markdown


def test_markdown_marks_skipped_metrics() -> None:
    report = evaluate(graph(node("a", "Person")), [FactProvenance()])

    assert "| fact_provenance | — | — | — | skipped: the graph has no lexical nodes;" in (
        report.to_markdown()
    )


def test_markdown_escapes_text_taken_from_the_graph() -> None:
    name = "*Ada* [click](https://example.com) <img> __init__ snake_case"

    markdown = evaluate(graph(node("a", name=name)), [EntityConnectivity()]).to_markdown()

    assert r"\*Ada\* \[click](https://example.com) \<img> \_\_init\_\_ snake_case" in markdown
    assert " [click](" not in markdown


def test_markdown_limits_findings_and_ids_shown() -> None:
    nodes = [node(f"n{number:02}", name=f"Same {number % 2}") for number in range(30)]
    report = evaluate(graph(*nodes), [EntityUniqueness(), EntityConnectivity()])

    markdown = report.to_markdown(max_findings=1)

    assert "- and 29 more findings" in markdown
    assert "`n00`, `n02`, `n04`" in markdown
    assert "`n18` and 5 more" in markdown


def test_markdown_names_the_layouts_type_keys_only_when_it_has_some() -> None:
    plain = evaluate(graph(node("a", name="Ada")), [EntityConnectivity()]).to_markdown()
    typed = evaluate(
        graph(node("a", name="Ada")), [EntityConnectivity()], layout=Layout(type_keys=("name",))
    ).to_markdown()

    assert "relationship types from" not in plain
    assert "names from name, title; relationship types from name\n" in typed


def test_markdown_lists_metadata_only_when_given() -> None:
    plain = evaluate(graph(node("a", name="Ada")), [EntityConnectivity()]).to_markdown()
    described = evaluate(
        graph(node("a", name="Ada")),
        [EntityConnectivity()],
        metadata={"model": "gpt-5-mini", "prompt": "v|2"},
    ).to_markdown()

    assert "- Metadata:" not in plain
    assert "# kgeval report\n\n- Metadata: model=gpt-5-mini, prompt=v\\|2\n- Graph:" in described


def test_reports_render_as_markdown_in_notebooks() -> None:
    report = evaluate(example_graph(), [EntityUniqueness()])

    assert report._repr_markdown_() == report.to_markdown()


def test_markdown_counts_expired_facts_and_undecided_units_only_when_there_are_some() -> None:
    history = graph(
        node("e", "Episodic", content="Ada left Acme."),
        node("ada", "Entity", "Person", name="Ada"),
        node("acme", "Entity", "Company", name="Acme"),
        relationship("left", "ada", "RELATES_TO", "acme", name="WORKS_AT", expired_at="2024"),
    )
    model = UnusableModel("KNOWS")

    expired = evaluate(history, [EntityConnectivity()]).to_markdown()
    undecided = evaluate(example_graph(), [FactAccuracy()], model=model).to_markdown()

    assert "1 relationship (0 facts, 1 expired fact, 0 provenance relationships)" in expired
    assert "; expired facts by expired_at" in expired
    assert "1 expired fact" not in evaluate(example_graph(), [EntityConnectivity()]).to_markdown()
    assert (
        "0 / 8 facts between entities with a name, grounded in a chunk with text, 1 undecided |"
        in (undecided)
    )
