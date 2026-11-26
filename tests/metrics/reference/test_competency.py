"""Test the competency metric."""

from kgeval.metrics.base import Inputs
from kgeval.metrics.reference.competency import Competency
from kgeval.questions import Question, QuestionSet, load_questions
from tests.factories import EXAMPLE_DIR, example_view

QUESTIONS = load_questions(EXAMPLE_DIR / "questions.yaml")


def test_questions_pass_when_the_graph_holds_every_supporting_fact() -> None:
    result = Competency().evaluate(example_view(), Inputs(questions=QUESTIONS))

    assert (result.unit, result.evaluated, result.failed) == ("test questions", 5, 1)
    assert [finding.message for finding in result.findings] == [
        "'What did Lee Kuan Yew co-found?': the graph lacks "
        '"Lee Kuan Yew" -[:FOUNDED]-> "Lee & Lee"'
    ]


def test_supporting_entities_need_a_matching_entity() -> None:
    questions = QuestionSet(
        questions=(
            Question(text="Who is Lee Hsien Loong?", supporting_entities=("Lee Hsien Loong",)),
        )
    )

    result = Competency().evaluate(example_view(), Inputs(questions=questions))

    assert result.findings[0].message == (
        "'Who is Lee Hsien Loong?': the graph lacks \"Lee Hsien Loong\""
    )


def test_signature_changes_with_the_questions() -> None:
    fewer = QuestionSet(questions=QUESTIONS.questions[1:])

    metric = Competency()

    assert metric.signature(Inputs(questions=QUESTIONS)) != metric.signature(
        Inputs(questions=fewer)
    )
