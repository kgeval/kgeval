"""Test the retrieval recall and retrieval precision metrics."""

import pytest

from kgeval.errors import InputError
from kgeval.metrics.base import Inputs
from kgeval.metrics.retrieval.context import RetrievalPrecision, RetrievalRecall
from kgeval.questions import Question, QuestionSet, load_questions
from tests.factories import EXAMPLE_DIR, EXAMPLE_RETRIEVALS, example_view

QUESTIONS = load_questions(EXAMPLE_DIR / "questions.yaml")


def test_recall_scores_only_the_supporting_facts_and_entities_the_graph_holds() -> None:
    result = RetrievalRecall().evaluate(
        example_view(), Inputs(questions=QUESTIONS, retrievals=EXAMPLE_RETRIEVALS)
    )

    # Leave out Lee & Lee, which the graph lacks; competency reports it.
    assert (result.unit, result.evaluated, result.failed) == (
        "supporting facts and entities in the graph",
        6,
        2,
    )
    # Average over questions, so the two-hop question's half counts as much as any other.
    assert result.score == pytest.approx((1 + 0 + 1 + 1 + 0.5) / 5)
    finding = result.findings[0]
    assert finding.message == (
        "'Where was Lee Kuan Yew born?': the retrieval lacks "
        '"Lee Kuan Yew" -[:BORN_IN]-> "Singapore"'
    )
    assert finding.subject == "Where was Lee Kuan Yew born?"
    assert finding.relationship_ids == ("born_in", "born_in_copy")
    assert finding.node_ids == ("lee", "singapore")


def test_recall_leaves_out_questions_whose_support_the_graph_lacks() -> None:
    question = Question.model_validate(
        {
            "text": "Who co-founded Lee & Lee?",
            "supporting_facts": [["Lee Kuan Yew", "FOUNDED", "Lee & Lee"]],
        }
    )
    retrievals = EXAMPLE_RETRIEVALS | {question.text: ("lee",)}

    every = QuestionSet(questions=(*QUESTIONS.questions, question))

    result = RetrievalRecall().evaluate(
        example_view(), Inputs(questions=every, retrievals=retrievals)
    )
    alone = RetrievalRecall().evaluate(
        example_view(), Inputs(questions=QuestionSet(questions=(question,)), retrievals=retrievals)
    )

    # Average over the other five questions only: the retriever cannot return what is missing.
    assert result.score == pytest.approx((1 + 0 + 1 + 1 + 0.5) / 5)
    assert alone.skip_reason is not None


def test_recall_needs_the_fact_not_only_its_entities() -> None:
    retrievals = EXAMPLE_RETRIEVALS | {"Where was Lee Kuan Yew born?": ("lee", "singapore")}

    result = RetrievalRecall().evaluate(
        example_view(), Inputs(questions=QUESTIONS, retrievals=retrievals)
    )

    # Miss the birthplace for the two-hop question too, as the example retrievals do.
    assert result.failed == 2


def test_recall_counts_a_retrieved_chunk_that_grounds_the_fact() -> None:
    retrievals = EXAMPLE_RETRIEVALS | {"Where was Lee Kuan Yew born?": ("chunk1",)}

    result = RetrievalRecall().evaluate(
        example_view(), Inputs(questions=QUESTIONS, retrievals=retrievals)
    )

    assert result.failed == 1


def test_recall_evaluates_only_the_top_k_of_each_retrieval() -> None:
    retrievals = EXAMPLE_RETRIEVALS | {"When was Lee Kuan Yew born?": ("kwa", "lee")}

    top = RetrievalRecall(k=1)
    inputs = Inputs(questions=QUESTIONS, retrievals=retrievals)

    # Miss Lee Kuan Yew, ranked second, besides the two misses of the full retrievals.
    assert top.evaluate(example_view(), inputs).failed == 3
    assert top.signature(inputs).endswith("|k=1")


def test_precision_fails_retrieved_elements_the_answer_does_not_need() -> None:
    result = RetrievalPrecision().evaluate(
        example_view(), Inputs(questions=QUESTIONS, retrievals=EXAMPLE_RETRIEVALS)
    )

    # Count chunk1, which grounds the fact the first question needs.
    assert (result.unit, result.evaluated, result.failed) == (
        "retrieved entities, facts, and chunks",
        11,
        1,
    )
    assert result.score == pytest.approx((1 + 1 + 1 + 0.5 + 1) / 5)
    [finding] = result.findings
    assert finding.message == (
        "'When was Lee Kuan Yew born?': 1 of 2 retrieved entities, facts, and chunks "
        "support nothing the answer needs"
    )
    assert finding.node_ids == ("kwa",)


def test_metrics_need_a_retrieval_for_every_question() -> None:
    retrievals = dict(list(EXAMPLE_RETRIEVALS.items())[1:])

    with pytest.raises(InputError, match="no retrieval for 1 of 5 questions"):
        RetrievalRecall().evaluate(
            example_view(), Inputs(questions=QUESTIONS, retrievals=retrievals)
        )


def test_unknown_element_ids_are_ignored_with_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    retrievals = dict.fromkeys(EXAMPLE_RETRIEVALS, ("4:other:1", "document"))

    result = RetrievalPrecision().evaluate(
        example_view(), Inputs(questions=QUESTIONS, retrievals=retrievals)
    )

    assert result.skip_reason == "no retrieval holds an entity, fact, or chunk of the graph"
    assert "Ignored 1 retrieved element IDs that are not in the graph" in caplog.text


def test_retrievals_without_any_element_id_of_the_graph_are_rejected() -> None:
    # Reject chunk texts, or IDs from another database, rather than score them as misses.
    retrievals = dict.fromkeys(EXAMPLE_RETRIEVALS, ("Lee Kuan Yew was born in Singapore.",))

    with pytest.raises(InputError, match="no retrieved element ID is in the graph"):
        RetrievalRecall().evaluate(
            example_view(), Inputs(questions=QUESTIONS, retrievals=retrievals)
        )


def test_signature_names_the_questions_but_not_the_retrievals_it_evaluates() -> None:
    retrievals = EXAMPLE_RETRIEVALS | {"Where was Lee Kuan Yew born?": ("born_in",)}

    assert (
        RetrievalRecall().signature(Inputs(questions=QUESTIONS, retrievals=EXAMPLE_RETRIEVALS))
        == RetrievalRecall().signature(Inputs(questions=QUESTIONS, retrievals=retrievals))
        == f"retrieval_recall|questions={QUESTIONS.digest()}"
    )
