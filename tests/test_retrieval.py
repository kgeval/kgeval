"""Test collecting what a retriever returns for each test question."""

from pathlib import Path

import pytest
from neo4j import Record
from neo4j.graph import Graph, Node, Relationship
from neo4j.graph import Path as GraphPath

from kgeval.errors import InputError
from kgeval.questions import load_questions
from kgeval.retrieval import load_retrievals, retrieve
from tests.factories import EXAMPLE_DIR

QUESTIONS = load_questions(EXAMPLE_DIR / "questions.yaml")


def test_retrieve_collects_the_element_ids_of_driver_nodes_relationships_and_paths() -> None:
    graph = Graph()
    lee = Node(graph, "4:x:1", 1, ["Person"], {"name": "Lee Kuan Yew"})
    singapore = Node(graph, "4:x:2", 2, ["Location"], {"name": "Singapore"})
    born_in = Relationship(graph, "5:x:3", 3, {})
    # Connect the relationship as the driver does when it reads one.
    born_in._start_node, born_in._end_node = lee, singapore

    def retriever(question: str) -> list[object]:
        if "born" in question:
            return [lee, "4:x:9", GraphPath(lee, born_in)]
        # Return records and lists, as a query that collects elements does.
        return [Record({"facts": [born_in], "people": [lee]})]  # type: ignore[no-untyped-call]

    retrievals = retrieve(QUESTIONS, retriever)

    # Keep rank order and first ranks, with a path's elements in traversal order.
    assert retrievals["Where was Lee Kuan Yew born?"] == ("4:x:1", "4:x:9", "5:x:3", "4:x:2")
    assert retrievals["Who founded the People's Action Party?"] == ("5:x:3", "4:x:1")


def test_retrieve_rejects_anything_but_element_ids_and_graph_elements() -> None:
    with pytest.raises(InputError, match="returned a dict for"):
        retrieve(QUESTIONS, lambda question: [{"name": "Lee Kuan Yew"}])


def test_load_retrievals_keeps_rank_order_and_lists_each_element_once(tmp_path: Path) -> None:
    path = tmp_path / "retrievals.json"
    path.write_text('{"Who?": ["5:x:3", "4:x:1", "5:x:3"]}', encoding="utf-8")

    assert load_retrievals(path) == {"Who?": ("5:x:3", "4:x:1")}


def test_load_retrievals_rejects_other_shapes(tmp_path: Path) -> None:
    path = tmp_path / "retrievals.yaml"
    path.write_text("- 4:x:1\n", encoding="utf-8")

    with pytest.raises(InputError, match=r"retrievals\.yaml"):
        load_retrievals(path)
