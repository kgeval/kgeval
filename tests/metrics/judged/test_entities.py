"""Test the entity resolution, entity disambiguation, and node type accuracy metrics."""

from kgeval.graph.view import GraphView
from kgeval.metrics.judged.entities import (
    DuplicateVerdict,
    EntityDisambiguation,
    EntityResolution,
    IdentityVerdict,
    NodeTypeAccuracy,
    TypeVerdict,
)
from tests.doubles import ScriptedJudge
from tests.factories import example_view, node, relationship, view

SWIFT_CHUNK = "x" * 300 + " Taylor Swift sang. " + "y" * 300


def namesakes() -> GraphView:
    return view(
        node("c", "Chunk", text=SWIFT_CHUNK),
        node("ts", "__Entity__", "Person", name="Taylor Swift"),
        node("tas", "__Entity__", "Person", name="Taylor Alison Swift"),
        node("tl", "__Entity__", "Person", name="Taylor Lautner"),
        node("ibm", "__Entity__", "Company", name="IBM"),
        node("intl", "__Entity__", "Company", name="International Business Machines"),
        node("usa", "__Entity__", "Country", name="U.S.A."),
        node("us", "__Entity__", "Country", name="United States of America"),
        relationship("h", "c", "HAS_ENTITY", "ts"),
    )


def entity_prompts(judge: ScriptedJudge) -> dict[str, str]:
    """Map the name of each entity the judge was asked about to its prompt."""
    return {prompt.split("The entity:\n")[1].split('"')[1]: prompt for prompt in judge.prompts}


def test_resolution_asks_only_about_entities_with_candidates_named_alike_or_by_acronym() -> None:
    judge = ScriptedJudge()

    result = EntityResolution(judge=judge).evaluate(namesakes())

    assert (result.unit, result.evaluated, result.failed) == ("entities with a name", 7, 0)
    prompts = entity_prompts(judge)
    # Ask about the later of each pair, in name order; Taylor Lautner has no candidate.
    assert set(prompts) == {
        "International Business Machines",
        "Taylor Swift",
        "United States of America",
    }
    assert '1. "IBM" (Company)' in prompts["International Business Machines"]
    assert '1. "U.S.A." (Country)' in prompts["United States of America"]
    assert '1. "Taylor Alison Swift" (Person)' in prompts["Taylor Swift"]
    assert "Taylor Lautner" not in "".join(judge.prompts)


def test_resolution_shows_excerpts_around_the_name() -> None:
    judge = ScriptedJudge()

    EntityResolution(judge=judge).evaluate(namesakes())

    prompt = entity_prompts(judge)["Taylor Swift"]
    assert '"Taylor Swift" (Person)\n<excerpt>…' in prompt
    assert "Taylor Swift sang." in prompt
    assert "x" * 250 not in prompt


def test_confirmed_duplicates_fail_the_later_entity_and_ignore_unknown_numbers() -> None:
    judge = ScriptedJudge('The entity:\n"Taylor Swift"', rejection={"duplicates": [0, 1, 1, 7]})

    result = EntityResolution(judge=judge).evaluate(namesakes())

    [finding] = result.findings
    assert finding.message == (
        '"Taylor Swift" (Person) duplicates "Taylor Alison Swift" (Person): Scripted.'
    )
    assert finding.node_ids == ("ts", "tas")
    assert (result.evaluated, result.failed) == (7, 1)


def test_resolution_proposes_exact_and_surname_duplicates_of_the_example() -> None:
    judge = ScriptedJudge()

    EntityResolution(judge=judge).evaluate(example_view())

    prompts = entity_prompts(judge)
    assert set(prompts) == {"Lee", "lee kuan yew"}
    assert '1. "Lee Kuan Yew" (Person)' in prompts["Lee"]
    assert '1. "Lee Kuan Yew" (Person)' in prompts["lee kuan yew"]
    assert '2. "Lee" (Person)' in prompts["lee kuan yew"]


def test_resolution_proposes_candidates_through_words_that_many_names_share() -> None:
    johns = [node(f"j{number}", "Person", name=f"John Doe{number}") for number in range(60)]
    graph = view(
        node("smith", "Person", name="John Smith"),
        node("smyth", "Person", name="John Smyth"),
        *johns,
    )
    judge = ScriptedJudge()

    EntityResolution(judge=judge).evaluate(graph)

    # Pair through "john", which more than 50 names share, however large the graph grows.
    assert entity_prompts(judge)["John Smyth"].count('. "John') == 1
    assert '1. "John Smith" (Person)' in entity_prompts(judge)["John Smyth"]


def test_disambiguation_shows_every_chunk_of_entities_in_two_or_more() -> None:
    judge = ScriptedJudge('Entity: "Lee Kuan Yew"')

    result = EntityDisambiguation(judge=judge).evaluate(example_view())

    # Consider Lee Kuan Yew, Raffles, and Singapore, each grounded in two chunks or more.
    assert (result.unit, result.evaluated, result.failed) == (
        "entities with a name in two or more chunks with text",
        3,
        1,
    )
    [lee] = [prompt for prompt in judge.prompts if 'Entity: "Lee Kuan Yew"' in prompt]
    assert "Kwa Geok Choo married Lee Kuan Yew." in lee
    assert "born in Singapore" in lee
    assert "attended Raffles Institution" in lee
    assert result.findings[0].message == (
        '"Lee Kuan Yew" (Person) stands for more than one thing: Scripted.'
    )


def test_disambiguation_skips_graphs_without_entities_in_two_chunks() -> None:
    graph = view(
        node("c", "Chunk", text="Ada."),
        node("a", "Person", name="Ada"),
        relationship("h", "c", "HAS_ENTITY", "a"),
    )

    assert EntityDisambiguation(judge=ScriptedJudge()).evaluate(graph).skip_reason == (
        "no entity with a name is grounded in two chunks with text"
    )


def test_node_type_judge_sees_the_name_node_types_and_chunks_of_each_entity() -> None:
    judge = ScriptedJudge()

    result = NodeTypeAccuracy(judge=judge).evaluate(example_view())

    # Leave out the unnamed entity and Marina Bay Sands, which no chunk grounds.
    assert (result.unit, result.evaluated, result.failed) == (
        "entities with a name, a node type, and a chunk with text",
        9,
        0,
    )
    assert not any("Marina Bay Sands" in prompt for prompt in judge.prompts)
    [lee] = [prompt for prompt in judge.prompts if 'Entity: "Lee Kuan Yew"' in prompt]
    assert 'Entity: "Lee Kuan Yew"\nNode types: Person' in lee
    assert "born in Singapore" in lee


def test_rejected_types_become_findings_with_the_judges_reason() -> None:
    result = NodeTypeAccuracy(judge=ScriptedJudge('Entity: "Singapore"')).evaluate(example_view())

    [finding] = result.findings
    assert finding.message == (
        'the chunks do not support the node types of "Singapore" (Location): Scripted.'
    )
    assert finding.node_ids == ("singapore",)


def test_node_type_accuracy_skips_graphs_without_grounded_entities_with_a_node_type() -> None:
    graph = view(node("a", "Person", name="Ada"))

    assert NodeTypeAccuracy(judge=ScriptedJudge()).evaluate(graph).skip_reason == (
        "no entity with a node type is grounded in a chunk with text"
    )


def test_verdicts_list_reasons_before_verdicts() -> None:
    assert list(DuplicateVerdict.model_fields) == ["reason", "duplicates"]
    assert list(IdentityVerdict.model_fields) == ["reason", "one_thing"]
    assert list(TypeVerdict.model_fields) == ["reason", "correct"]
