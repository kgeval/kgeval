"""Test the test questions that a graph is expected to answer."""

from pathlib import Path

import pytest

from kgeval.errors import InputError
from kgeval.questions import QuestionSet, load_questions
from tests.factories import EXAMPLE_DIR, example_view, node, relationship, view


def write(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "questions.yaml"
    path.write_text(content, encoding="utf-8")
    return path


def test_load_questions_reads_the_example_questions() -> None:
    questions = load_questions(EXAMPLE_DIR / "questions.yaml")

    assert len(questions.questions) == 5
    assert questions.questions[1].supporting_facts == (("Lee Kuan Yew", "BORN_IN", "Singapore"),)
    assert questions.questions[3].supporting_entities == ("Lee Kuan Yew",)


@pytest.mark.parametrize(
    "content",
    [
        "[]\n",
        "- text: Who?\n",
        "- text: Who?\n  supporting_facts: [[Ada, KNOWS]]\n",
        "- text: Who?\n  cypher: MATCH (n) RETURN n\n  supporting_entities: [Ada]\n",
        "- {text: Who?, supporting_entities: [Ada]}\n- {text: Who?, supporting_entities: [Bob]}\n",
        "questions: [{text: Who?, supporting_entities: [Ada]}]\naliases: {Bob: [B]}\n",
    ],
)
def test_load_questions_rejects_invalid_question_sets(tmp_path: Path, content: str) -> None:
    with pytest.raises(InputError, match=r"questions\.yaml"):
        load_questions(write(tmp_path, content))


def test_support_matches_facts_and_entities_with_their_elements() -> None:
    questions = load_questions(EXAMPLE_DIR / "questions.yaml")

    born, founded, birth_date = (questions.support(example_view())[index] for index in (1, 2, 3))

    # Match the duplicate "lee kuan yew" too, since names match under normalization.
    [birthplace] = born
    assert birthplace.label == '"Lee Kuan Yew" -[:BORN_IN]-> "Singapore"'
    assert birthplace.relationship_ids == {"born_in", "born_in_copy"}
    assert birthplace.element_ids == birthplace.relationship_ids
    assert birthplace.node_ids == {"lee", "singapore"}
    assert [item.element_ids for item in founded][1] == frozenset()
    assert birth_date[0].element_ids == {"lee", "duplicate"}


def test_aliases_and_symmetric_types_match_variants_of_supporting_facts() -> None:
    graph = view(
        node("lky", "Person", name="Harry Lee"),
        node("kwa", "Person", name="Kwa Geok Choo"),
        relationship("r", "lky", "WED", "kwa"),
    )
    questions = QuestionSet.model_validate(
        {
            "questions": [
                {
                    "text": "Whom did Kwa Geok Choo marry?",
                    "supporting_facts": [["Kwa Geok Choo", "MARRIED", "Lee Kuan Yew"]],
                }
            ],
            "aliases": {"Lee Kuan Yew": ["Harry Lee"]},
            "relationship_type_aliases": {"MARRIED": ["WED"]},
            "symmetric_relationship_types": ["MARRIED"],
        }
    )

    [[married]] = questions.support(graph)

    assert married.relationship_ids == {"r"}
