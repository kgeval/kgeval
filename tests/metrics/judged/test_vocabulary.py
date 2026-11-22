"""Test the vocabulary resolution metric."""

from kgeval.metrics.judged.vocabulary import SynonymGroup, SynonymVerdict, VocabularyResolution
from tests.doubles import ScriptedJudge, UnusableJudge
from tests.factories import example_view, node, relationship, view


def test_judge_sees_each_terms_usage_with_grounded_examples() -> None:
    judge = ScriptedJudge()

    result = VocabularyResolution(judge=judge).evaluate(example_view())

    # Count Person and PERSON, and FOUNDED and Founded, once: vocabulary_uniqueness owns them.
    assert (result.unit, result.evaluated, result.failed) == ("distinct vocabulary terms", 12, 0)
    assert len(judge.prompts) == 3
    [node_types] = [prompt for prompt in judge.prompts if "node types of" in prompt]
    [relationship_types] = [prompt for prompt in judge.prompts if "relationship types of" in prompt]
    [property_keys] = [prompt for prompt in judge.prompts if "property keys of" in prompt]
    assert "- Person, also spelled PERSON (used 6 times)\n" in node_types
    assert "  in facts: FOUNDED (3), BORN_IN (2), MARRIED (2)\n" in node_types
    # Show the sentence that says what ATTENDED means here, which its name alone does not.
    assert (
        "- ATTENDED (used once)\n  connects: Person → Organization (1)\n"
        '  e.g. "Lee Kuan Yew" (Person) -[:ATTENDED]-> "Raffles" (Organization) <excerpt>'
    ) in relationship_types
    assert "attended Raffles Institution.</excerpt>" in relationship_types
    assert "  values: DATE (2), STRING (1)\n" in property_keys
    assert result.signature == "vocabulary_resolution|judge=scripted"


def test_a_group_may_name_a_term_by_any_of_its_spellings() -> None:
    graph = view(
        node("a", "Person", name="Ada"),
        node("b", "Company", name="Acme"),
        relationship("r1", "a", "WORKS_AT", "b"),
        relationship("r2", "a", "WorksAt", "b"),
        relationship("r3", "a", "EMPLOYED_BY", "b"),
    )
    judge = ScriptedJudge(
        "relationship types of",
        rejection={
            "groups": [
                {"reason": "Both link people to employers.", "terms": ["WorksAt", "EMPLOYED_BY"]}
            ]
        },
    )

    result = VocabularyResolution(judge=judge).evaluate(graph)

    # Fail EMPLOYED_BY once against the distinct term, whose two spellings two facts use.
    [finding] = result.findings
    assert finding.message == (
        "EMPLOYED_BY means the same as the relationship type WORKS_AT: "
        "Both link people to employers."
    )
    assert (result.evaluated, result.failed) == (5, 1)


def test_synonyms_fail_every_term_but_the_most_used_one() -> None:
    graph = view(
        node("a", "Person", name="Ada"),
        node("b", "Company", name="Acme"),
        node("c", "Company", name="Initech"),
        relationship("r1", "a", "WORKS_AT", "b"),
        relationship("r2", "a", "WORKS_AT", "c"),
        relationship("r3", "a", "EMPLOYED_BY", "c"),
    )
    judge = ScriptedJudge(
        "relationship types of",
        rejection={
            "groups": [{"reason": "All employ.", "terms": ["EMPLOYED_BY", "WORKS_AT", "HIRED_BY"]}]
        },
    )

    result = VocabularyResolution(judge=judge).evaluate(graph)

    # Ignore HIRED_BY, which the graph does not use.
    [finding] = result.findings
    assert (
        finding.message
        == "EMPLOYED_BY means the same as the relationship type WORKS_AT: All employ."
    )
    assert finding.relationship_ids == ("r3",)
    assert result.failed == 1


def test_groups_that_share_a_term_merge_and_count_each_term_once() -> None:
    graph = view(node("a", "Person", name="Ada"), node("b", "Human", name="Bob"), node("c", "Man"))
    judge = ScriptedJudge(
        "node types of",
        rejection={
            "groups": [
                {"reason": "Humans are people.", "terms": ["Human", "Person"]},
                {"reason": "Men are people.", "terms": ["Man", "Person"]},
                {"reason": "Men are people.", "terms": ["Man"]},
            ]
        },
    )

    result = VocabularyResolution(judge=judge).evaluate(graph)

    assert [finding.message for finding in result.findings] == [
        # Keep each merged group's reason once, in order.
        "Man and Person mean the same as the node type Human: Humans are people. Men are people."
    ]
    assert result.failed == 2


def test_large_vocabularies_are_judged_in_bounded_calls_led_by_the_most_used_terms() -> None:
    types = [f"TYPE_{number:03}" for number in range(250)]
    graph = view(
        node("a", name="Ada"),
        node("b", name="Bob"),
        *(relationship(f"r{number}", "a", type_, "b") for number, type_ in enumerate(types)),
        *(relationship(f"x{number}", "b", "TYPE_249", "a") for number in range(3)),
    )
    judge = ScriptedJudge(
        "TYPE_100",
        rejection={"groups": [{"reason": "Same use.", "terms": ["TYPE_249", "TYPE_100"]}]},
    )

    result = VocabularyResolution(judge=judge).evaluate(graph)

    relationship_prompts = [prompt for prompt in judge.prompts if "relationship types of" in prompt]
    assert len(relationship_prompts) == 2
    # List the most used term in every call, so its rare synonyms are found in any of them.
    assert all("- TYPE_249 (used 4 times)\n" in prompt for prompt in relationship_prompts)
    assert [finding.message for finding in result.findings] == [
        "TYPE_100 means the same as the relationship type TYPE_249: Same use."
    ]


def test_terms_only_in_calls_without_a_usable_verdict_are_unjudged() -> None:
    types = [f"TYPE_{number:03}" for number in range(250)]
    graph = view(
        node("a", name="Ada"),
        node("b", name="Bob"),
        *(relationship(f"r{number}", "a", type_, "b") for number, type_ in enumerate(types)),
    )

    result = VocabularyResolution(judge=UnusableJudge("TYPE_249")).evaluate(graph)

    # Leave out the 50 terms of the second call, but keep the 100 most used it shares.
    assert result.unjudged == 50


def test_verdicts_list_reasons_before_groups_and_terms() -> None:
    assert list(SynonymVerdict.model_fields) == ["reason", "groups"]
    assert list(SynonymGroup.model_fields) == ["reason", "terms"]


def test_rare_terms_used_alike_share_a_call_however_their_names_sort() -> None:
    def endpoints(number: int) -> tuple[str, str]:
        return ("b", "c") if number in (120, 240) else ("a", "b")

    graph = view(
        node("a", "Person", name="Ada"),
        node("b", "Company", name="Acme"),
        node("c", "City", name="Oslo"),
        *(
            relationship(
                f"r{number}", endpoints(number)[0], f"TYPE_{number:03}", endpoints(number)[1]
            )
            for number in range(250)
        ),
    )
    judge = ScriptedJudge()

    VocabularyResolution(judge=judge).evaluate(graph)

    # Sort the terms outside the anchors by usage, so the two Company → City terms meet.
    calls = [prompt for prompt in judge.prompts if "relationship types of" in prompt]
    assert any("- TYPE_120 " in prompt and "- TYPE_240 " in prompt for prompt in calls)
